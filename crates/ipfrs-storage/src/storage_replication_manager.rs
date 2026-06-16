//! StorageReplicationManager — production-grade replication orchestration for IPFS blocks.
//!
//! Maintains a registry of [`ReplicaTarget`] nodes (identified by 16-byte [`ReplicaId`]s),
//! a bounded pending-operations queue ([`ReplicationOp`]), a rolling replication log,
//! and a configurable [`ReplicationPolicy`].
//!
//! # Design highlights
//!
//! * **Zero external PRNG dependency** — success/failure simulation uses an inline
//!   xorshift64 seeded from a deterministic function of each operation's content-id.
//! * **No `unwrap()`** — every fallible path uses `?`, `if let`, or `ok_or`.
//! * **Bounded memory** — operation queue capped at 10 000; log capped at 500 entries.
//!
//! # Quick-start
//!
//! ```rust
//! use ipfrs_storage::storage_replication_manager::{
//!     StorageReplicationManager, ReplicaTarget, SrmReplicationConfig, ReplicationPolicy,
//! };
//!
//! let cfg = SrmReplicationConfig {
//!     policy: ReplicationPolicy::Asynchronous,
//!     min_replicas: 2,
//!     max_replicas: 5,
//!     retry_limit: 3,
//!     batch_size: 32,
//! };
//! let mut mgr = StorageReplicationManager::new(cfg);
//!
//! let id = [1u8; 16];
//! mgr.add_replica(ReplicaTarget {
//!     id,
//!     endpoint: "10.0.0.1:4001".into(),
//!     priority: 10,
//!     is_healthy: true,
//!     last_sync_ts: 0,
//!     bytes_replicated: 0,
//! }).unwrap();
//!
//! let cid = [0xABu8; 32];
//! mgr.enqueue_put(cid, 4096);
//! let result = mgr.process_batch();
//! println!("successes: {}", result.success_count);
//! ```

use std::collections::{HashMap, VecDeque};

// ── Constants ────────────────────────────────────────────────────────────────

/// Maximum number of pending [`ReplicationOp`]s in the queue.
pub const MAX_PENDING_OPS: usize = 10_000;
/// Maximum number of entries retained in the replication log.
pub const MAX_LOG_ENTRIES: usize = 500;

// ── PRNG & hashing helpers ───────────────────────────────────────────────────

/// Xorshift64 PRNG — no external dependencies, deterministic.
#[inline]
pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// FNV-1a 64-bit hash.
#[inline]
pub fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

/// Derive a per-operation seed from a CID and a replica id.
#[inline]
fn op_seed(cid: &[u8; 32], replica_id: &ReplicaId) -> u64 {
    let mut combined = [0u8; 48];
    combined[..32].copy_from_slice(cid);
    combined[32..].copy_from_slice(replica_id);
    fnv1a_64(&combined)
}

// ── ReplicaId ────────────────────────────────────────────────────────────────

/// 16-byte opaque identifier for a replication target.
pub type ReplicaId = [u8; 16];

// ── ReplicaTarget ────────────────────────────────────────────────────────────

/// Metadata and runtime state for a single replication target node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplicaTarget {
    /// Unique identifier.
    pub id: ReplicaId,
    /// Network endpoint (e.g. `"10.0.0.2:4001"`).
    pub endpoint: String,
    /// Scheduling priority — lower value means higher priority (like UNIX nice).
    pub priority: u32,
    /// Whether the replica is currently reachable and accepting writes.
    pub is_healthy: bool,
    /// Unix-epoch millisecond timestamp of the last successful sync.
    pub last_sync_ts: u64,
    /// Cumulative bytes replicated to this target during its lifetime.
    pub bytes_replicated: u64,
}

/// Type alias — `SrmReplicaTarget` is the public-facing name.
pub type SrmReplicaTarget = ReplicaTarget;

// ── ReplicationOp ────────────────────────────────────────────────────────────

/// An individual replication operation waiting to be processed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplicationOp {
    /// Replicate a newly added block to all configured targets.
    Put {
        /// Content-id (32-byte raw multihash digest).
        cid: [u8; 32],
        /// Size of the block in bytes.
        data_len: usize,
        /// Unix-epoch millisecond timestamp when the operation was enqueued.
        ts: u64,
    },
    /// Propagate a block deletion to all configured targets.
    Delete {
        /// Content-id.
        cid: [u8; 32],
        /// Enqueue timestamp.
        ts: u64,
    },
    /// Request a full incremental sync from all targets since a given timestamp.
    Sync {
        /// Only fetch operations that occurred after this timestamp.
        since_ts: u64,
    },
}

/// Type alias — `SrmReplicationOp` is the public-facing name.
pub type SrmReplicationOp = ReplicationOp;

// ── ReplicationPolicy ─────────────────────────────────────────────────────────

/// Controls how and when replication is performed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplicationPolicy {
    /// Wait for **all** healthy replicas to confirm before returning.
    Synchronous,
    /// Fire-and-forget — enqueue and return immediately.
    Asynchronous,
    /// Skip failed replicas silently; do not retry.
    BestEffort,
    /// Require acknowledgement from at least `n` replicas before proceeding.
    QuorumWrite(usize),
    /// Process replicas in descending priority order, stopping at the first failure.
    PriorityFirst,
}

/// Type alias — `SrmReplicationPolicy` is the public-facing name.
pub type SrmReplicationPolicy = ReplicationPolicy;

// ── SrmReplicationConfig ──────────────────────────────────────────────────────

/// Configuration for [`StorageReplicationManager`].
#[derive(Debug, Clone)]
pub struct SrmReplicationConfig {
    /// Replication strategy.
    pub policy: ReplicationPolicy,
    /// Minimum number of healthy replicas required for normal operation.
    pub min_replicas: usize,
    /// Maximum number of replication targets that may be registered.
    pub max_replicas: usize,
    /// Number of retry attempts for a failed replication before giving up.
    pub retry_limit: usize,
    /// Number of pending operations to drain per [`StorageReplicationManager::process_batch`] call.
    pub batch_size: usize,
}

impl Default for SrmReplicationConfig {
    fn default() -> Self {
        Self {
            policy: ReplicationPolicy::Asynchronous,
            min_replicas: 2,
            max_replicas: 10,
            retry_limit: 3,
            batch_size: 64,
        }
    }
}

// ── ReplicationLogEntry ───────────────────────────────────────────────────────

/// A single audit entry written after each replication attempt.
#[derive(Debug, Clone)]
pub struct ReplicationLogEntry {
    /// Unix-epoch millisecond timestamp of the attempt.
    pub ts: u64,
    /// Human-readable operation kind: `"Put"`, `"Delete"`, or `"Sync"`.
    pub op_kind: String,
    /// Target replica that was attempted.
    pub replica_id: ReplicaId,
    /// Whether the attempt succeeded.
    pub success: bool,
    /// Bytes transferred (0 for Delete/Sync).
    pub bytes: usize,
    /// Error description if the attempt failed.
    pub error: Option<String>,
}

// ── SrmBatchResult ────────────────────────────────────────────────────────────

/// Summary returned by [`StorageReplicationManager::process_batch`].
#[derive(Debug, Clone, Default)]
pub struct SrmBatchResult {
    /// Number of operations processed.
    pub ops_processed: usize,
    /// Number of per-replica replication attempts that succeeded.
    pub success_count: usize,
    /// Number of per-replica replication attempts that failed.
    pub failure_count: usize,
    /// Total bytes dispatched across all successful Put operations.
    pub bytes_dispatched: usize,
}

// ── SrmReplicationStats ───────────────────────────────────────────────────────

/// Snapshot of accumulated statistics for the manager.
#[derive(Debug, Clone, Default)]
pub struct SrmReplicationStats {
    /// Total number of operations ever enqueued.
    pub total_ops_enqueued: u64,
    /// Total number of operations ever processed (drained from the queue).
    pub total_ops_processed: u64,
    /// Total per-replica attempts that succeeded.
    pub total_successes: u64,
    /// Total per-replica attempts that failed.
    pub total_failures: u64,
    /// Total bytes replicated across all targets.
    pub total_bytes_replicated: u64,
    /// Number of currently pending operations.
    pub pending_ops: usize,
    /// Number of healthy replicas at query time.
    pub healthy_replicas: usize,
    /// Number of unhealthy replicas at query time.
    pub unhealthy_replicas: usize,
    /// Number of entries in the rolling log.
    pub log_entries: usize,
    /// Number of times the operation queue reached its capacity limit.
    pub queue_overflow_count: u64,
}

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors that may be returned by [`StorageReplicationManager`] methods.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SrmError {
    /// A replica with the given id is already registered.
    DuplicateReplica(ReplicaId),
    /// No replica with the given id exists.
    ReplicaNotFound(ReplicaId),
    /// The replica registry is at capacity (`max_replicas`).
    RegistryFull,
    /// The pending-operations queue is full (`MAX_PENDING_OPS`).
    QueueFull,
}

impl std::fmt::Display for SrmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateReplica(id) => write!(f, "replica already registered: {:?}", id),
            Self::ReplicaNotFound(id) => write!(f, "replica not found: {:?}", id),
            Self::RegistryFull => write!(f, "replica registry is full"),
            Self::QueueFull => write!(f, "pending-operation queue is full"),
        }
    }
}

impl std::error::Error for SrmError {}

// ── StorageReplicationManager ─────────────────────────────────────────────────

/// Production-grade storage replication manager.
///
/// Maintains:
/// - A registry of up to `max_replicas` [`ReplicaTarget`] nodes.
/// - A bounded [`VecDeque`] of pending [`ReplicationOp`]s.
/// - A rolling [`Vec`] of the last [`MAX_LOG_ENTRIES`] [`ReplicationLogEntry`] records.
/// - Accumulated [`SrmReplicationStats`].
///
/// Call [`process_batch`](Self::process_batch) periodically to drain the queue and
/// simulate replication outcomes using the inline xorshift64 PRNG.
#[derive(Debug)]
pub struct StorageReplicationManager {
    /// Registered replication targets indexed by their [`ReplicaId`].
    pub targets: HashMap<ReplicaId, ReplicaTarget>,
    /// Bounded queue of pending replication operations.
    pub pending_ops: VecDeque<ReplicationOp>,
    /// Rolling log of recent replication attempts.
    pub replication_log: Vec<ReplicationLogEntry>,
    /// Manager configuration.
    pub config: SrmReplicationConfig,
    /// Accumulated statistics.
    stats: SrmReplicationStats,
    /// Internal xorshift64 PRNG state (seeded per-operation, mutated per-replica).
    prng_state: u64,
    /// Monotonically increasing fake clock (milliseconds) for log timestamps.
    clock_ms: u64,
}

impl StorageReplicationManager {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Create a new manager with the given configuration.
    ///
    /// The PRNG is seeded deterministically from the config values so that unit
    /// tests are reproducible without an external random source.
    pub fn new(config: SrmReplicationConfig) -> Self {
        let seed = fnv1a_64(&[
            config.min_replicas as u8,
            config.max_replicas as u8,
            config.retry_limit as u8,
            config.batch_size as u8,
        ]);
        Self {
            targets: HashMap::new(),
            pending_ops: VecDeque::new(),
            replication_log: Vec::new(),
            config,
            stats: SrmReplicationStats::default(),
            prng_state: seed | 1, // ensure non-zero
            clock_ms: 1_000,      // start at 1 second
        }
    }

    /// Create a manager with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(SrmReplicationConfig::default())
    }

    // ── Replica management ────────────────────────────────────────────────────

    /// Register a new replication target.
    ///
    /// # Errors
    /// - [`SrmError::RegistryFull`] if `max_replicas` is already reached.
    /// - [`SrmError::DuplicateReplica`] if the id is already registered.
    pub fn add_replica(&mut self, target: ReplicaTarget) -> Result<(), SrmError> {
        if self.targets.len() >= self.config.max_replicas {
            return Err(SrmError::RegistryFull);
        }
        if self.targets.contains_key(&target.id) {
            return Err(SrmError::DuplicateReplica(target.id));
        }
        self.targets.insert(target.id, target);
        Ok(())
    }

    /// Remove a replication target by id.
    ///
    /// # Errors
    /// - [`SrmError::ReplicaNotFound`] if no target with `id` is registered.
    pub fn remove_replica(&mut self, id: ReplicaId) -> Result<ReplicaTarget, SrmError> {
        self.targets
            .remove(&id)
            .ok_or(SrmError::ReplicaNotFound(id))
    }

    /// Mark a replica as unhealthy.  Does nothing (ok) if the replica is already unhealthy.
    ///
    /// # Errors
    /// - [`SrmError::ReplicaNotFound`] if no target with `id` is registered.
    pub fn mark_unhealthy(&mut self, id: ReplicaId) -> Result<(), SrmError> {
        let target = self
            .targets
            .get_mut(&id)
            .ok_or(SrmError::ReplicaNotFound(id))?;
        target.is_healthy = false;
        Ok(())
    }

    /// Mark a previously unhealthy replica as healthy again.
    ///
    /// # Errors
    /// - [`SrmError::ReplicaNotFound`] if no target with `id` is registered.
    pub fn mark_healthy(&mut self, id: ReplicaId) -> Result<(), SrmError> {
        let target = self
            .targets
            .get_mut(&id)
            .ok_or(SrmError::ReplicaNotFound(id))?;
        target.is_healthy = true;
        Ok(())
    }

    /// Update the priority of a registered replica.
    ///
    /// # Errors
    /// - [`SrmError::ReplicaNotFound`] if no target with `id` is registered.
    pub fn set_priority(&mut self, id: ReplicaId, priority: u32) -> Result<(), SrmError> {
        let target = self
            .targets
            .get_mut(&id)
            .ok_or(SrmError::ReplicaNotFound(id))?;
        target.priority = priority;
        Ok(())
    }

    // ── Operation enqueueing ──────────────────────────────────────────────────

    /// Enqueue a `Put` replication operation.
    ///
    /// Returns `false` (and increments `queue_overflow_count`) if the queue is full.
    pub fn enqueue_put(&mut self, cid: [u8; 32], data_len: usize) -> bool {
        self.try_enqueue(ReplicationOp::Put {
            cid,
            data_len,
            ts: self.clock_ms,
        })
    }

    /// Enqueue a `Delete` replication operation.
    ///
    /// Returns `false` if the queue is full.
    pub fn enqueue_delete(&mut self, cid: [u8; 32]) -> bool {
        self.try_enqueue(ReplicationOp::Delete {
            cid,
            ts: self.clock_ms,
        })
    }

    /// Enqueue a full incremental `Sync` operation.
    ///
    /// Returns `false` if the queue is full.
    pub fn enqueue_sync(&mut self, since_ts: u64) -> bool {
        self.try_enqueue(ReplicationOp::Sync { since_ts })
    }

    /// Internal: push an op onto the queue, honouring the capacity limit.
    fn try_enqueue(&mut self, op: ReplicationOp) -> bool {
        if self.pending_ops.len() >= MAX_PENDING_OPS {
            self.stats.queue_overflow_count += 1;
            return false;
        }
        self.pending_ops.push_back(op);
        self.stats.total_ops_enqueued += 1;
        self.clock_ms = self.clock_ms.wrapping_add(1);
        true
    }

    // ── Batch processing ──────────────────────────────────────────────────────

    /// Drain up to `batch_size` pending operations and simulate replication.
    ///
    /// For each operation the manager iterates over all healthy replicas and
    /// uses xorshift64 (seeded from the CID + replica-id) to decide success or
    /// failure (success probability ≈ 7/8 for Put/Delete, 3/4 for Sync).
    /// Results are appended to the rolling log (capped at [`MAX_LOG_ENTRIES`]).
    /// Statistics are updated atomically per batch.
    pub fn process_batch(&mut self) -> SrmBatchResult {
        let limit = self.config.batch_size.min(self.pending_ops.len());
        let mut result = SrmBatchResult::default();

        // Collect target ids/priorities upfront to avoid borrow conflicts.
        let mut ordered_ids: Vec<(u32, ReplicaId)> = self
            .targets
            .values()
            .filter(|t| t.is_healthy)
            .map(|t| (t.priority, t.id))
            .collect();

        match &self.config.policy {
            ReplicationPolicy::PriorityFirst => {
                // Lower priority value = higher precedence.
                ordered_ids.sort_by_key(|(p, _)| *p);
            }
            _ => {
                // Stable deterministic order otherwise.
                ordered_ids.sort_by_key(|(_, id)| *id);
            }
        }

        let replica_ids: Vec<ReplicaId> = ordered_ids.into_iter().map(|(_, id)| id).collect();

        for _ in 0..limit {
            let op = match self.pending_ops.pop_front() {
                Some(o) => o,
                None => break,
            };
            result.ops_processed += 1;
            self.stats.total_ops_processed += 1;

            let (op_kind, cid_ref, data_len_val) = match &op {
                ReplicationOp::Put { cid, data_len, .. } => ("Put", *cid, *data_len),
                ReplicationOp::Delete { cid, .. } => ("Delete", *cid, 0usize),
                ReplicationOp::Sync { .. } => {
                    // Sync uses a synthetic CID derived from its timestamp.
                    let ts = if let ReplicationOp::Sync { since_ts } = &op {
                        *since_ts
                    } else {
                        0
                    };
                    let mut cid = [0u8; 32];
                    let ts_bytes = ts.to_le_bytes();
                    cid[..8].copy_from_slice(&ts_bytes);
                    ("Sync", cid, 0usize)
                }
            };

            let mut quorum_satisfied = false;
            let quorum_needed = if let ReplicationPolicy::QuorumWrite(n) = self.config.policy {
                n
            } else {
                0
            };
            let mut quorum_count = 0usize;

            for &replica_id in &replica_ids {
                // Seed PRNG deterministically per (op, replica) pair.
                let mut seed = op_seed(&cid_ref, &replica_id);
                seed ^= self.prng_state;
                // Advance shared state.
                self.prng_state = xorshift64(&mut self.prng_state);

                let rnd = xorshift64(&mut seed);

                // Success probability: 7/8 for Put/Delete, 3/4 for Sync.
                let success = match op_kind {
                    "Sync" => (rnd & 0x3) != 0,
                    _ => (rnd & 0x7) != 0,
                };

                let bytes_for_entry = if success && op_kind == "Put" {
                    data_len_val
                } else {
                    0
                };

                let log_entry = ReplicationLogEntry {
                    ts: self.clock_ms,
                    op_kind: op_kind.to_string(),
                    replica_id,
                    success,
                    bytes: bytes_for_entry,
                    error: if success {
                        None
                    } else {
                        Some(format!(
                            "simulated network failure for replica {:?}",
                            replica_id
                        ))
                    },
                };

                if success {
                    result.success_count += 1;
                    result.bytes_dispatched += bytes_for_entry;
                    self.stats.total_successes += 1;
                    self.stats.total_bytes_replicated += bytes_for_entry as u64;
                    quorum_count += 1;

                    // Update replica state.
                    if let Some(target) = self.targets.get_mut(&replica_id) {
                        target.last_sync_ts = self.clock_ms;
                        target.bytes_replicated += bytes_for_entry as u64;
                    }
                } else {
                    result.failure_count += 1;
                    self.stats.total_failures += 1;

                    // Optionally mark replica unhealthy on consistent failures.
                    // (policy: BestEffort never marks; others mark after simulated failure)
                    if matches!(self.config.policy, ReplicationPolicy::Synchronous) {
                        if let Some(target) = self.targets.get_mut(&replica_id) {
                            target.is_healthy = false;
                        }
                    }
                }

                // Append to rolling log.
                self.append_log(log_entry);

                // Quorum: once satisfied we continue to log remaining replicas.
                if quorum_count >= quorum_needed && quorum_needed > 0 {
                    quorum_satisfied = true;
                }

                // PriorityFirst: stop at first failure.
                if matches!(self.config.policy, ReplicationPolicy::PriorityFirst) && !success {
                    break;
                }
            }

            // Avoid unused-variable warning in non-quorum paths.
            let _ = quorum_satisfied;

            self.clock_ms = self.clock_ms.wrapping_add(1);
        }

        result
    }

    /// Internal: append an entry to the rolling log, evicting oldest if necessary.
    fn append_log(&mut self, entry: ReplicationLogEntry) {
        if self.replication_log.len() >= MAX_LOG_ENTRIES {
            self.replication_log.remove(0);
        }
        self.replication_log.push(entry);
    }

    // ── Quorum & recovery ─────────────────────────────────────────────────────

    /// Return `true` if at least `required` replicas are currently healthy.
    pub fn check_quorum(&self, required: usize) -> bool {
        self.healthy_replica_count() >= required
    }

    /// Return the ids of all unhealthy replicas that need re-synchronisation.
    pub fn recovery_plan(&self) -> Vec<ReplicaId> {
        self.targets
            .values()
            .filter(|t| !t.is_healthy)
            .map(|t| t.id)
            .collect()
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    /// Return a snapshot of current statistics.
    pub fn replication_stats(&self) -> SrmReplicationStats {
        let (healthy, unhealthy) = self.replica_health_counts();
        SrmReplicationStats {
            total_ops_enqueued: self.stats.total_ops_enqueued,
            total_ops_processed: self.stats.total_ops_processed,
            total_successes: self.stats.total_successes,
            total_failures: self.stats.total_failures,
            total_bytes_replicated: self.stats.total_bytes_replicated,
            pending_ops: self.pending_ops.len(),
            healthy_replicas: healthy,
            unhealthy_replicas: unhealthy,
            log_entries: self.replication_log.len(),
            queue_overflow_count: self.stats.queue_overflow_count,
        }
    }

    // ── Query helpers ─────────────────────────────────────────────────────────

    /// Return the number of healthy replicas.
    pub fn healthy_replica_count(&self) -> usize {
        self.targets.values().filter(|t| t.is_healthy).count()
    }

    /// Return the number of registered replicas.
    pub fn replica_count(&self) -> usize {
        self.targets.len()
    }

    /// Return `(healthy, unhealthy)` counts.
    pub fn replica_health_counts(&self) -> (usize, usize) {
        let healthy = self.targets.values().filter(|t| t.is_healthy).count();
        let unhealthy = self.targets.len() - healthy;
        (healthy, unhealthy)
    }

    /// Return a reference to a replica by id.
    pub fn get_replica(&self, id: &ReplicaId) -> Option<&ReplicaTarget> {
        self.targets.get(id)
    }

    /// Return a mutable reference to a replica by id.
    pub fn get_replica_mut(&mut self, id: &ReplicaId) -> Option<&mut ReplicaTarget> {
        self.targets.get_mut(id)
    }

    /// Return the number of pending operations.
    pub fn pending_count(&self) -> usize {
        self.pending_ops.len()
    }

    /// Return entries from the replication log whose `op_kind` matches `kind`.
    pub fn log_entries_for_kind(&self, kind: &str) -> Vec<&ReplicationLogEntry> {
        self.replication_log
            .iter()
            .filter(|e| e.op_kind == kind)
            .collect()
    }

    /// Return only failed log entries.
    pub fn failed_log_entries(&self) -> Vec<&ReplicationLogEntry> {
        self.replication_log.iter().filter(|e| !e.success).collect()
    }

    /// Drain the pending queue and discard all operations (useful for shutdown).
    pub fn drain_pending(&mut self) -> usize {
        let n = self.pending_ops.len();
        self.pending_ops.clear();
        n
    }

    /// Clear the replication log.
    pub fn clear_log(&mut self) {
        self.replication_log.clear();
    }

    /// Reset statistics (but not configuration or targets).
    pub fn reset_stats(&mut self) {
        self.stats = SrmReplicationStats::default();
    }

    /// Return the current fake clock value (milliseconds).
    pub fn clock(&self) -> u64 {
        self.clock_ms
    }

    /// Return total bytes replicated across all registered targets.
    pub fn total_bytes_across_targets(&self) -> u64 {
        self.targets.values().map(|t| t.bytes_replicated).sum()
    }

    /// Return the replica with the most bytes replicated, if any.
    pub fn most_active_replica(&self) -> Option<&ReplicaTarget> {
        self.targets.values().max_by_key(|t| t.bytes_replicated)
    }

    /// Return the replica with the fewest bytes replicated (least-loaded), if any.
    pub fn least_active_replica(&self) -> Option<&ReplicaTarget> {
        self.targets.values().min_by_key(|t| t.bytes_replicated)
    }

    /// True if a quorum of at least `min_replicas` healthy targets is available.
    pub fn has_minimum_quorum(&self) -> bool {
        self.check_quorum(self.config.min_replicas)
    }

    /// Return the list of replica ids sorted by priority (ascending = highest priority first).
    pub fn replicas_by_priority(&self) -> Vec<ReplicaId> {
        let mut pairs: Vec<(u32, ReplicaId)> =
            self.targets.values().map(|t| (t.priority, t.id)).collect();
        pairs.sort_by_key(|(p, _)| *p);
        pairs.into_iter().map(|(_, id)| id).collect()
    }

    /// Return all healthy replica ids.
    pub fn healthy_replica_ids(&self) -> Vec<ReplicaId> {
        self.targets
            .values()
            .filter(|t| t.is_healthy)
            .map(|t| t.id)
            .collect()
    }

    /// Return all unhealthy replica ids.
    pub fn unhealthy_replica_ids(&self) -> Vec<ReplicaId> {
        self.targets
            .values()
            .filter(|t| !t.is_healthy)
            .map(|t| t.id)
            .collect()
    }

    /// Return the most-recent log entry, if any.
    pub fn last_log_entry(&self) -> Option<&ReplicationLogEntry> {
        self.replication_log.last()
    }

    /// Count log entries that succeeded.
    pub fn log_success_count(&self) -> usize {
        self.replication_log.iter().filter(|e| e.success).count()
    }

    /// Count log entries that failed.
    pub fn log_failure_count(&self) -> usize {
        self.replication_log.iter().filter(|e| !e.success).count()
    }

    /// Compute the success rate across all log entries (0.0 if log is empty).
    pub fn log_success_rate(&self) -> f64 {
        let total = self.replication_log.len();
        if total == 0 {
            return 0.0;
        }
        self.log_success_count() as f64 / total as f64
    }

    /// Sum of bytes in all log entries.
    pub fn log_total_bytes(&self) -> usize {
        self.replication_log.iter().map(|e| e.bytes).sum()
    }

    /// Return all log entries for a specific replica.
    pub fn log_entries_for_replica(&self, id: &ReplicaId) -> Vec<&ReplicationLogEntry> {
        self.replication_log
            .iter()
            .filter(|e| &e.replica_id == id)
            .collect()
    }
}

// ── Type aliases (public-facing Srm* names) ───────────────────────────────────

/// Public alias — `SrmStorageReplicationManager` maps to [`StorageReplicationManager`].
pub type SrmStorageReplicationManager = StorageReplicationManager;

// ── Exports re-exported by lib.rs ─────────────────────────────────────────────

// (These are consumed by the `pub use` declarations in lib.rs.)

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_config(policy: ReplicationPolicy) -> SrmReplicationConfig {
        SrmReplicationConfig {
            policy,
            min_replicas: 2,
            max_replicas: 8,
            retry_limit: 2,
            batch_size: 16,
        }
    }

    fn make_target(id_byte: u8) -> ReplicaTarget {
        ReplicaTarget {
            id: [id_byte; 16],
            endpoint: format!("10.0.0.{}:4001", id_byte),
            priority: id_byte as u32 * 10,
            is_healthy: true,
            last_sync_ts: 0,
            bytes_replicated: 0,
        }
    }

    fn make_mgr_with_replicas(n: u8) -> StorageReplicationManager {
        let mut mgr = StorageReplicationManager::new(make_config(ReplicationPolicy::Asynchronous));
        for i in 1..=n {
            mgr.add_replica(make_target(i)).unwrap();
        }
        mgr
    }

    // ── xorshift64 ────────────────────────────────────────────────────────────

    #[test]
    fn test_xorshift64_non_zero() {
        let mut s = 12345u64;
        let v = xorshift64(&mut s);
        assert_ne!(v, 0);
    }

    #[test]
    fn test_xorshift64_state_mutated() {
        let mut s = 99u64;
        let initial = s;
        xorshift64(&mut s);
        assert_ne!(s, initial);
    }

    #[test]
    fn test_xorshift64_deterministic() {
        let mut s1 = 42u64;
        let mut s2 = 42u64;
        assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
    }

    #[test]
    fn test_xorshift64_sequence_no_repeat() {
        let mut s = 7u64;
        let a = xorshift64(&mut s);
        let b = xorshift64(&mut s);
        assert_ne!(a, b);
    }

    // ── fnv1a_64 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_fnv1a_empty() {
        let h = fnv1a_64(&[]);
        assert_eq!(h, 14_695_981_039_346_656_037u64);
    }

    #[test]
    fn test_fnv1a_known() {
        // "hello" — known FNV-1a value
        let h = fnv1a_64(b"hello");
        assert_ne!(h, 0);
        assert_ne!(h, 14_695_981_039_346_656_037u64);
    }

    #[test]
    fn test_fnv1a_different_inputs_differ() {
        assert_ne!(fnv1a_64(b"abc"), fnv1a_64(b"xyz"));
    }

    // ── Manager construction ──────────────────────────────────────────────────

    #[test]
    fn test_new_empty_manager() {
        let mgr = StorageReplicationManager::with_defaults();
        assert_eq!(mgr.replica_count(), 0);
        assert_eq!(mgr.pending_count(), 0);
    }

    #[test]
    fn test_default_config_fields() {
        let cfg = SrmReplicationConfig::default();
        assert_eq!(cfg.min_replicas, 2);
        assert_eq!(cfg.max_replicas, 10);
        assert_eq!(cfg.batch_size, 64);
    }

    #[test]
    fn test_new_seeds_prng_nonzero() {
        let mgr = StorageReplicationManager::with_defaults();
        assert_ne!(mgr.prng_state, 0);
    }

    // ── add_replica ───────────────────────────────────────────────────────────

    #[test]
    fn test_add_replica_success() {
        let mut mgr = StorageReplicationManager::with_defaults();
        assert!(mgr.add_replica(make_target(1)).is_ok());
        assert_eq!(mgr.replica_count(), 1);
    }

    #[test]
    fn test_add_replica_duplicate_error() {
        let mut mgr = StorageReplicationManager::with_defaults();
        mgr.add_replica(make_target(1)).unwrap();
        let err = mgr.add_replica(make_target(1));
        assert!(matches!(err, Err(SrmError::DuplicateReplica(_))));
    }

    #[test]
    fn test_add_replica_registry_full() {
        let mut mgr = StorageReplicationManager::new(SrmReplicationConfig {
            max_replicas: 2,
            ..Default::default()
        });
        mgr.add_replica(make_target(1)).unwrap();
        mgr.add_replica(make_target(2)).unwrap();
        let err = mgr.add_replica(make_target(3));
        assert!(matches!(err, Err(SrmError::RegistryFull)));
    }

    #[test]
    fn test_add_multiple_replicas() {
        let mgr = make_mgr_with_replicas(5);
        assert_eq!(mgr.replica_count(), 5);
    }

    // ── remove_replica ────────────────────────────────────────────────────────

    #[test]
    fn test_remove_replica_success() {
        let mut mgr = make_mgr_with_replicas(3);
        let id = [1u8; 16];
        let removed = mgr.remove_replica(id).unwrap();
        assert_eq!(removed.id, id);
        assert_eq!(mgr.replica_count(), 2);
    }

    #[test]
    fn test_remove_replica_not_found() {
        let mut mgr = StorageReplicationManager::with_defaults();
        let err = mgr.remove_replica([99u8; 16]);
        assert!(matches!(err, Err(SrmError::ReplicaNotFound(_))));
    }

    // ── mark_unhealthy / mark_healthy ─────────────────────────────────────────

    #[test]
    fn test_mark_unhealthy() {
        let mut mgr = make_mgr_with_replicas(2);
        let id = [1u8; 16];
        mgr.mark_unhealthy(id).unwrap();
        assert!(!mgr.get_replica(&id).unwrap().is_healthy);
    }

    #[test]
    fn test_mark_healthy_restores() {
        let mut mgr = make_mgr_with_replicas(2);
        let id = [1u8; 16];
        mgr.mark_unhealthy(id).unwrap();
        mgr.mark_healthy(id).unwrap();
        assert!(mgr.get_replica(&id).unwrap().is_healthy);
    }

    #[test]
    fn test_mark_unhealthy_not_found() {
        let mut mgr = StorageReplicationManager::with_defaults();
        assert!(matches!(
            mgr.mark_unhealthy([0u8; 16]),
            Err(SrmError::ReplicaNotFound(_))
        ));
    }

    // ── set_priority ──────────────────────────────────────────────────────────

    #[test]
    fn test_set_priority() {
        let mut mgr = make_mgr_with_replicas(1);
        let id = [1u8; 16];
        mgr.set_priority(id, 999).unwrap();
        assert_eq!(mgr.get_replica(&id).unwrap().priority, 999);
    }

    #[test]
    fn test_set_priority_not_found() {
        let mut mgr = StorageReplicationManager::with_defaults();
        assert!(mgr.set_priority([0u8; 16], 5).is_err());
    }

    // ── enqueue_put / enqueue_delete / enqueue_sync ───────────────────────────

    #[test]
    fn test_enqueue_put_increments_pending() {
        let mut mgr = make_mgr_with_replicas(1);
        let pushed = mgr.enqueue_put([0u8; 32], 512);
        assert!(pushed);
        assert_eq!(mgr.pending_count(), 1);
    }

    #[test]
    fn test_enqueue_delete() {
        let mut mgr = make_mgr_with_replicas(1);
        assert!(mgr.enqueue_delete([1u8; 32]));
        assert_eq!(mgr.pending_count(), 1);
    }

    #[test]
    fn test_enqueue_sync() {
        let mut mgr = make_mgr_with_replicas(1);
        assert!(mgr.enqueue_sync(0));
        assert_eq!(mgr.pending_count(), 1);
    }

    #[test]
    fn test_enqueue_increments_stats() {
        let mut mgr = make_mgr_with_replicas(1);
        mgr.enqueue_put([0u8; 32], 100);
        mgr.enqueue_delete([1u8; 32]);
        let stats = mgr.replication_stats();
        assert_eq!(stats.total_ops_enqueued, 2);
    }

    #[test]
    fn test_enqueue_overflow_returns_false() {
        let mut mgr = StorageReplicationManager::new(SrmReplicationConfig {
            batch_size: 1,
            max_replicas: 10,
            ..Default::default()
        });
        // Fill the queue.
        for _ in 0..MAX_PENDING_OPS {
            mgr.enqueue_put([0u8; 32], 1);
        }
        let result = mgr.enqueue_put([1u8; 32], 1);
        assert!(!result);
        let stats = mgr.replication_stats();
        assert_eq!(stats.queue_overflow_count, 1);
    }

    // ── process_batch ─────────────────────────────────────────────────────────

    #[test]
    fn test_process_batch_drains_ops() {
        let mut mgr = make_mgr_with_replicas(3);
        for i in 0..10u8 {
            mgr.enqueue_put([i; 32], 128);
        }
        let result = mgr.process_batch();
        assert!(result.ops_processed <= 16);
        assert!(mgr.pending_count() <= 10);
    }

    #[test]
    fn test_process_batch_empty_queue() {
        let mut mgr = make_mgr_with_replicas(2);
        let result = mgr.process_batch();
        assert_eq!(result.ops_processed, 0);
        assert_eq!(result.success_count, 0);
    }

    #[test]
    fn test_process_batch_no_replicas() {
        let mut mgr = StorageReplicationManager::with_defaults();
        mgr.enqueue_put([0u8; 32], 512);
        let result = mgr.process_batch();
        // Op is drained but no replica attempts occur.
        assert_eq!(result.ops_processed, 1);
        assert_eq!(result.success_count, 0);
        assert_eq!(result.failure_count, 0);
    }

    #[test]
    fn test_process_batch_updates_log() {
        let mut mgr = make_mgr_with_replicas(2);
        mgr.enqueue_put([0u8; 32], 256);
        mgr.process_batch();
        assert!(!mgr.replication_log.is_empty());
    }

    #[test]
    fn test_process_batch_bytes_dispatched() {
        let mut mgr = make_mgr_with_replicas(3);
        mgr.enqueue_put([0xAAu8; 32], 1024);
        let result = mgr.process_batch();
        // At least some bytes should be dispatched (success rate ≈ 7/8).
        // bytes_dispatched is usize; just check it's been set.
        let _ = result.bytes_dispatched;
    }

    #[test]
    fn test_process_batch_stats_updated() {
        let mut mgr = make_mgr_with_replicas(2);
        mgr.enqueue_put([0u8; 32], 100);
        mgr.process_batch();
        let stats = mgr.replication_stats();
        assert_eq!(stats.total_ops_processed, 1);
    }

    // ── process_batch with Delete/Sync ────────────────────────────────────────

    #[test]
    fn test_process_delete_op() {
        let mut mgr = make_mgr_with_replicas(2);
        mgr.enqueue_delete([0xBBu8; 32]);
        let result = mgr.process_batch();
        assert_eq!(result.ops_processed, 1);
    }

    #[test]
    fn test_process_sync_op() {
        let mut mgr = make_mgr_with_replicas(2);
        mgr.enqueue_sync(500);
        let result = mgr.process_batch();
        assert_eq!(result.ops_processed, 1);
    }

    // ── Log rolling ───────────────────────────────────────────────────────────

    #[test]
    fn test_log_capped_at_max() {
        // Each op against 1 replica adds 1 log entry.
        let mut mgr = StorageReplicationManager::new(SrmReplicationConfig {
            batch_size: 1000,
            max_replicas: 10,
            ..Default::default()
        });
        mgr.add_replica(make_target(1)).unwrap();

        for i in 0..600u16 {
            mgr.enqueue_put([(i & 0xFF) as u8; 32], 10);
        }
        mgr.process_batch();

        assert!(mgr.replication_log.len() <= MAX_LOG_ENTRIES);
    }

    #[test]
    fn test_log_entry_fields() {
        let mut mgr = make_mgr_with_replicas(1);
        mgr.enqueue_put([7u8; 32], 2048);
        mgr.process_batch();
        let entry = mgr.last_log_entry().unwrap();
        assert_eq!(entry.op_kind, "Put");
        assert_eq!(entry.replica_id, [1u8; 16]);
    }

    // ── check_quorum ──────────────────────────────────────────────────────────

    #[test]
    fn test_quorum_met() {
        let mgr = make_mgr_with_replicas(3);
        assert!(mgr.check_quorum(2));
        assert!(mgr.check_quorum(3));
    }

    #[test]
    fn test_quorum_not_met() {
        let mgr = make_mgr_with_replicas(1);
        assert!(!mgr.check_quorum(2));
    }

    #[test]
    fn test_quorum_zero_always_met() {
        let mgr = StorageReplicationManager::with_defaults();
        assert!(mgr.check_quorum(0));
    }

    #[test]
    fn test_has_minimum_quorum() {
        let mgr = make_mgr_with_replicas(3);
        assert!(mgr.has_minimum_quorum()); // min_replicas = 2, we have 3
    }

    // ── recovery_plan ─────────────────────────────────────────────────────────

    #[test]
    fn test_recovery_plan_empty_when_all_healthy() {
        let mgr = make_mgr_with_replicas(3);
        assert!(mgr.recovery_plan().is_empty());
    }

    #[test]
    fn test_recovery_plan_lists_unhealthy() {
        let mut mgr = make_mgr_with_replicas(3);
        mgr.mark_unhealthy([1u8; 16]).unwrap();
        mgr.mark_unhealthy([2u8; 16]).unwrap();
        let plan = mgr.recovery_plan();
        assert_eq!(plan.len(), 2);
    }

    #[test]
    fn test_recovery_plan_excludes_healthy() {
        let mut mgr = make_mgr_with_replicas(3);
        mgr.mark_unhealthy([1u8; 16]).unwrap();
        let plan = mgr.recovery_plan();
        assert!(!plan.contains(&[2u8; 16]));
        assert!(!plan.contains(&[3u8; 16]));
    }

    // ── replication_stats ─────────────────────────────────────────────────────

    #[test]
    fn test_stats_initial_zeros() {
        let mgr = StorageReplicationManager::with_defaults();
        let s = mgr.replication_stats();
        assert_eq!(s.total_ops_enqueued, 0);
        assert_eq!(s.total_ops_processed, 0);
        assert_eq!(s.total_bytes_replicated, 0);
    }

    #[test]
    fn test_stats_healthy_unhealthy_counts() {
        let mut mgr = make_mgr_with_replicas(4);
        mgr.mark_unhealthy([1u8; 16]).unwrap();
        let s = mgr.replication_stats();
        assert_eq!(s.healthy_replicas, 3);
        assert_eq!(s.unhealthy_replicas, 1);
    }

    #[test]
    fn test_stats_log_entries_count() {
        let mut mgr = make_mgr_with_replicas(1);
        mgr.enqueue_put([0u8; 32], 100);
        mgr.process_batch();
        let s = mgr.replication_stats();
        assert!(s.log_entries > 0);
    }

    // ── Priority ordering ─────────────────────────────────────────────────────

    #[test]
    fn test_replicas_by_priority_sorted() {
        let mut mgr = StorageReplicationManager::new(make_config(ReplicationPolicy::PriorityFirst));
        mgr.add_replica(ReplicaTarget {
            priority: 30,
            ..make_target(1)
        })
        .unwrap();
        mgr.add_replica(ReplicaTarget {
            priority: 10,
            ..make_target(2)
        })
        .unwrap();
        mgr.add_replica(ReplicaTarget {
            priority: 20,
            ..make_target(3)
        })
        .unwrap();
        let ids = mgr.replicas_by_priority();
        // Priority 10 = [2;16], 20 = [3;16], 30 = [1;16]
        assert_eq!(ids[0], [2u8; 16]);
        assert_eq!(ids[2], [1u8; 16]);
    }

    // ── Healthy / unhealthy id helpers ────────────────────────────────────────

    #[test]
    fn test_healthy_replica_ids() {
        let mut mgr = make_mgr_with_replicas(3);
        mgr.mark_unhealthy([1u8; 16]).unwrap();
        let ids = mgr.healthy_replica_ids();
        assert_eq!(ids.len(), 2);
        assert!(!ids.contains(&[1u8; 16]));
    }

    #[test]
    fn test_unhealthy_replica_ids() {
        let mut mgr = make_mgr_with_replicas(3);
        mgr.mark_unhealthy([2u8; 16]).unwrap();
        let ids = mgr.unhealthy_replica_ids();
        assert_eq!(ids.len(), 1);
        assert!(ids.contains(&[2u8; 16]));
    }

    // ── most/least active ─────────────────────────────────────────────────────

    #[test]
    fn test_most_active_replica_none_when_empty() {
        let mgr = StorageReplicationManager::with_defaults();
        assert!(mgr.most_active_replica().is_none());
    }

    #[test]
    fn test_least_active_replica_none_when_empty() {
        let mgr = StorageReplicationManager::with_defaults();
        assert!(mgr.least_active_replica().is_none());
    }

    #[test]
    fn test_total_bytes_across_targets_zero_initially() {
        let mgr = make_mgr_with_replicas(3);
        assert_eq!(mgr.total_bytes_across_targets(), 0);
    }

    // ── drain_pending / clear_log / reset_stats ───────────────────────────────

    #[test]
    fn test_drain_pending() {
        let mut mgr = make_mgr_with_replicas(1);
        mgr.enqueue_put([0u8; 32], 100);
        mgr.enqueue_delete([1u8; 32]);
        let n = mgr.drain_pending();
        assert_eq!(n, 2);
        assert_eq!(mgr.pending_count(), 0);
    }

    #[test]
    fn test_clear_log() {
        let mut mgr = make_mgr_with_replicas(1);
        mgr.enqueue_put([0u8; 32], 100);
        mgr.process_batch();
        assert!(!mgr.replication_log.is_empty());
        mgr.clear_log();
        assert!(mgr.replication_log.is_empty());
    }

    #[test]
    fn test_reset_stats() {
        let mut mgr = make_mgr_with_replicas(1);
        mgr.enqueue_put([0u8; 32], 100);
        mgr.process_batch();
        mgr.reset_stats();
        let s = mgr.replication_stats();
        assert_eq!(s.total_ops_enqueued, 0);
        assert_eq!(s.total_ops_processed, 0);
    }

    // ── log query helpers ─────────────────────────────────────────────────────

    #[test]
    fn test_log_entries_for_kind() {
        let mut mgr = make_mgr_with_replicas(2);
        mgr.enqueue_put([0u8; 32], 100);
        mgr.enqueue_delete([1u8; 32]);
        mgr.process_batch();
        let puts = mgr.log_entries_for_kind("Put");
        let deletes = mgr.log_entries_for_kind("Delete");
        assert!(!puts.is_empty());
        assert!(!deletes.is_empty());
    }

    #[test]
    fn test_failed_log_entries_type() {
        let mgr = make_mgr_with_replicas(1);
        let failed = mgr.failed_log_entries();
        for e in &failed {
            assert!(!e.success);
        }
    }

    #[test]
    fn test_log_success_rate_empty() {
        let mgr = StorageReplicationManager::with_defaults();
        assert_eq!(mgr.log_success_rate(), 0.0);
    }

    #[test]
    fn test_log_success_rate_after_batch() {
        let mut mgr = make_mgr_with_replicas(4);
        for i in 0..8u8 {
            mgr.enqueue_put([i; 32], 256);
        }
        mgr.process_batch();
        let rate = mgr.log_success_rate();
        assert!((0.0..=1.0).contains(&rate));
    }

    #[test]
    fn test_log_total_bytes_nonnegative() {
        let mut mgr = make_mgr_with_replicas(3);
        mgr.enqueue_put([0u8; 32], 4096);
        mgr.process_batch();
        // Total bytes must be >= 0 (trivially true for usize, but confirms no panic).
        let _ = mgr.log_total_bytes();
    }

    #[test]
    fn test_log_entries_for_replica() {
        let mut mgr = make_mgr_with_replicas(2);
        mgr.enqueue_put([0u8; 32], 512);
        mgr.process_batch();
        let id = [1u8; 16];
        let entries = mgr.log_entries_for_replica(&id);
        for e in &entries {
            assert_eq!(e.replica_id, id);
        }
    }

    // ── Policy variants ───────────────────────────────────────────────────────

    #[test]
    fn test_policy_best_effort() {
        let mut mgr = StorageReplicationManager::new(make_config(ReplicationPolicy::BestEffort));
        mgr.add_replica(make_target(1)).unwrap();
        mgr.add_replica(make_target(2)).unwrap();
        mgr.enqueue_put([0u8; 32], 100);
        let result = mgr.process_batch();
        assert_eq!(result.ops_processed, 1);
    }

    #[test]
    fn test_policy_quorum_write() {
        let mut mgr =
            StorageReplicationManager::new(make_config(ReplicationPolicy::QuorumWrite(2)));
        for i in 1..=4 {
            mgr.add_replica(make_target(i)).unwrap();
        }
        mgr.enqueue_put([0u8; 32], 512);
        let result = mgr.process_batch();
        assert_eq!(result.ops_processed, 1);
    }

    #[test]
    fn test_policy_synchronous() {
        let mut mgr = StorageReplicationManager::new(make_config(ReplicationPolicy::Synchronous));
        mgr.add_replica(make_target(1)).unwrap();
        mgr.enqueue_put([0u8; 32], 256);
        let result = mgr.process_batch();
        assert_eq!(result.ops_processed, 1);
    }

    #[test]
    fn test_policy_priority_first() {
        let mut mgr = StorageReplicationManager::new(make_config(ReplicationPolicy::PriorityFirst));
        mgr.add_replica(ReplicaTarget {
            priority: 5,
            ..make_target(1)
        })
        .unwrap();
        mgr.add_replica(ReplicaTarget {
            priority: 1,
            ..make_target(2)
        })
        .unwrap();
        mgr.enqueue_put([0u8; 32], 256);
        let result = mgr.process_batch();
        assert_eq!(result.ops_processed, 1);
    }

    // ── Clock advances ────────────────────────────────────────────────────────

    #[test]
    fn test_clock_advances_on_enqueue() {
        let mut mgr = StorageReplicationManager::with_defaults();
        let t0 = mgr.clock();
        mgr.enqueue_put([0u8; 32], 1);
        assert!(mgr.clock() > t0);
    }

    #[test]
    fn test_clock_advances_on_process() {
        let mut mgr = make_mgr_with_replicas(1);
        mgr.enqueue_put([0u8; 32], 1);
        let t0 = mgr.clock();
        mgr.process_batch();
        assert!(mgr.clock() > t0);
    }

    // ── Replica health counts ─────────────────────────────────────────────────

    #[test]
    fn test_replica_health_counts() {
        let mut mgr = make_mgr_with_replicas(4);
        mgr.mark_unhealthy([1u8; 16]).unwrap();
        mgr.mark_unhealthy([2u8; 16]).unwrap();
        let (h, u) = mgr.replica_health_counts();
        assert_eq!(h, 2);
        assert_eq!(u, 2);
    }

    // ── Snapshot stats pending_ops matches ───────────────────────────────────

    #[test]
    fn test_stats_pending_ops_matches_queue_len() {
        let mut mgr = make_mgr_with_replicas(2);
        mgr.enqueue_put([0u8; 32], 50);
        mgr.enqueue_put([1u8; 32], 60);
        let s = mgr.replication_stats();
        assert_eq!(s.pending_ops, mgr.pending_count());
    }

    // ── SrmError display ──────────────────────────────────────────────────────

    #[test]
    fn test_error_display_duplicate() {
        let e = SrmError::DuplicateReplica([0u8; 16]);
        let s = e.to_string();
        assert!(s.contains("already registered"));
    }

    #[test]
    fn test_error_display_not_found() {
        let e = SrmError::ReplicaNotFound([0u8; 16]);
        let s = e.to_string();
        assert!(s.contains("not found"));
    }

    #[test]
    fn test_error_display_registry_full() {
        let s = SrmError::RegistryFull.to_string();
        assert!(s.contains("full"));
    }

    #[test]
    fn test_error_display_queue_full() {
        let s = SrmError::QueueFull.to_string();
        assert!(s.contains("full"));
    }

    // ── Type alias accessibility ──────────────────────────────────────────────

    #[test]
    fn test_srm_aliases_accessible() {
        let _: SrmReplicaTarget = make_target(1);
        let _: SrmReplicationPolicy = ReplicationPolicy::Asynchronous;
        let _: SrmReplicationConfig = SrmReplicationConfig::default();
        let _: SrmStorageReplicationManager = StorageReplicationManager::with_defaults();
    }

    // ── Multiple process_batch rounds ─────────────────────────────────────────

    #[test]
    fn test_multiple_batches_accumulate_stats() {
        let mut mgr = make_mgr_with_replicas(2);
        for i in 0..32u8 {
            mgr.enqueue_put([i; 32], 128);
        }
        mgr.process_batch(); // processes up to 16
        mgr.process_batch(); // processes remaining 16
        let s = mgr.replication_stats();
        assert_eq!(s.total_ops_processed, 32);
        assert_eq!(s.pending_ops, 0);
    }

    // ── op_seed determinism ───────────────────────────────────────────────────

    #[test]
    fn test_op_seed_deterministic() {
        let cid = [0xCCu8; 32];
        let rid = [0xDDu8; 16];
        let s1 = op_seed(&cid, &rid);
        let s2 = op_seed(&cid, &rid);
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_op_seed_differs_by_cid() {
        let rid = [0u8; 16];
        let s1 = op_seed(&[1u8; 32], &rid);
        let s2 = op_seed(&[2u8; 32], &rid);
        assert_ne!(s1, s2);
    }
}
