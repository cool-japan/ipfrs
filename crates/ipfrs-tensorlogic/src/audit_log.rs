//! Inference Audit Log
//!
//! Provides an immutable, append-only audit trail for every distributed inference
//! query: which peers were queried, which rules fired, what bindings were produced,
//! and in what order. Designed for compliance and post-hoc debugging.
//!
//! # Overview
//!
//! [`InferenceAuditLog`] accumulates [`AuditEntry`] records in a bounded ring
//! buffer (default cap 100 000 entries). Each entry carries a globally-monotonic
//! `sequence` number, a `trace_id` that groups all events belonging to a single
//! logical session, and an [`AuditEvent`] payload.
//!
//! [`AuditStats`] tracks high-level counters (total appended/trimmed, queries
//! started/completed/failed) using atomic operations so that stat reads never
//! block appenders.
//!
//! # Examples
//!
//! ```
//! use ipfrs_tensorlogic::audit_log::{AuditEvent, InferenceAuditLog};
//!
//! let log = InferenceAuditLog::default();
//!
//! let seq = log.append(
//!     "session-1",
//!     AuditEvent::QueryStarted {
//!         goal: "parent(X, alice)".to_string(),
//!         session_id: "session-1".to_string(),
//!         timestamp_ms: 1_000,
//!     },
//! ).expect("example: should succeed in docs");
//!
//! assert_eq!(seq, 0);
//! assert_eq!(log.entry_count(), 1);
//!
//! let entries = log.entries_for_trace("session-1");
//! assert_eq!(entries.len(), 1);
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use thiserror::Error;

// ─── AuditEvent ──────────────────────────────────────────────────────────────

/// A single observable event in the lifetime of a distributed inference query.
#[derive(Debug, Clone)]
pub enum AuditEvent {
    /// The inference session was initiated for the given goal.
    QueryStarted {
        /// The goal string being resolved (e.g. `"parent(X, alice)"`).
        goal: String,
        /// Session identifier that scopes this query.
        session_id: String,
        /// Wall-clock time in milliseconds since UNIX epoch.
        timestamp_ms: u64,
    },

    /// A remote peer was contacted to help resolve the goal.
    PeerQueried {
        /// Identifier of the remote peer (e.g. multiaddr or libp2p PeerId).
        peer_id: String,
        /// The specific sub-goal forwarded to that peer.
        goal: String,
        /// Wall-clock time in milliseconds since UNIX epoch.
        timestamp_ms: u64,
    },

    /// A local or remote rule matched and produced bindings.
    RuleFired {
        /// Stable rule identifier (e.g. `"rule:parent/2#3"`).
        rule_id: String,
        /// Variable bindings produced by this rule application.
        bindings: HashMap<String, String>,
        /// Wall-clock time in milliseconds since UNIX epoch.
        timestamp_ms: u64,
    },

    /// A batch of results was received from a remote peer.
    ResultReceived {
        /// Peer that returned this batch.
        peer_id: String,
        /// Number of variable bindings in this batch.
        binding_count: usize,
        /// Whether this is the peer's final response for the sub-goal.
        is_final: bool,
        /// Wall-clock time in milliseconds since UNIX epoch.
        timestamp_ms: u64,
    },

    /// The inference session completed successfully.
    QueryCompleted {
        /// Session identifier.
        session_id: String,
        /// Total number of result bindings produced across all peers.
        total_results: usize,
        /// Elapsed time from `QueryStarted` to `QueryCompleted` in milliseconds.
        duration_ms: u64,
        /// Wall-clock time in milliseconds since UNIX epoch.
        timestamp_ms: u64,
    },

    /// The inference session failed.
    QueryFailed {
        /// Session identifier.
        session_id: String,
        /// Human-readable failure reason.
        reason: String,
        /// Wall-clock time in milliseconds since UNIX epoch.
        timestamp_ms: u64,
    },
}

impl AuditEvent {
    /// Returns a stable, lowercase string tag for this event variant.
    ///
    /// Useful for serialization, metrics labeling, and log filtering.
    pub fn event_type(&self) -> &str {
        match self {
            AuditEvent::QueryStarted { .. } => "query_started",
            AuditEvent::PeerQueried { .. } => "peer_queried",
            AuditEvent::RuleFired { .. } => "rule_fired",
            AuditEvent::ResultReceived { .. } => "result_received",
            AuditEvent::QueryCompleted { .. } => "query_completed",
            AuditEvent::QueryFailed { .. } => "query_failed",
        }
    }

    /// Returns the wall-clock timestamp (ms since UNIX epoch) carried by this event.
    pub fn timestamp_ms(&self) -> u64 {
        match self {
            AuditEvent::QueryStarted { timestamp_ms, .. } => *timestamp_ms,
            AuditEvent::PeerQueried { timestamp_ms, .. } => *timestamp_ms,
            AuditEvent::RuleFired { timestamp_ms, .. } => *timestamp_ms,
            AuditEvent::ResultReceived { timestamp_ms, .. } => *timestamp_ms,
            AuditEvent::QueryCompleted { timestamp_ms, .. } => *timestamp_ms,
            AuditEvent::QueryFailed { timestamp_ms, .. } => *timestamp_ms,
        }
    }
}

// ─── AuditEntry ──────────────────────────────────────────────────────────────

/// A single record in the audit log.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    /// Globally monotonic sequence counter — lower value means earlier insertion.
    pub sequence: u64,
    /// The event payload.
    pub event: AuditEvent,
    /// Groups all events belonging to one logical query (usually equals `session_id`).
    pub trace_id: String,
}

// ─── AuditError ──────────────────────────────────────────────────────────────

/// Errors returned by [`InferenceAuditLog`] operations.
#[derive(Debug, Error)]
pub enum AuditError {
    /// The log has reached its configured capacity and will not accept more entries.
    #[error("audit log capacity exceeded: current={current}, max={max}")]
    CapacityExceeded {
        /// Current number of entries in the log.
        current: usize,
        /// Configured maximum number of entries.
        max: usize,
    },
}

// ─── AuditStats ──────────────────────────────────────────────────────────────

/// A point-in-time snapshot of [`AuditStats`] counters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditStatsSnapshot {
    /// Cumulative number of entries ever appended (including trimmed ones).
    pub total_appended: u64,
    /// Cumulative number of entries removed by [`InferenceAuditLog::trim_before`].
    pub total_trimmed: u64,
    /// Number of [`AuditEvent::QueryStarted`] events appended.
    pub total_queries_started: u64,
    /// Number of [`AuditEvent::QueryCompleted`] events appended.
    pub total_queries_completed: u64,
    /// Number of [`AuditEvent::QueryFailed`] events appended.
    pub total_queries_failed: u64,
}

/// Atomic counters tracking aggregate log activity.
///
/// All fields are [`AtomicU64`] so that `snapshot()` can be called without
/// taking any lock.
#[derive(Debug)]
pub struct AuditStats {
    /// Cumulative entries appended.
    pub total_appended: AtomicU64,
    /// Cumulative entries trimmed.
    pub total_trimmed: AtomicU64,
    /// Cumulative `QueryStarted` events.
    pub total_queries_started: AtomicU64,
    /// Cumulative `QueryCompleted` events.
    pub total_queries_completed: AtomicU64,
    /// Cumulative `QueryFailed` events.
    pub total_queries_failed: AtomicU64,
}

impl Default for AuditStats {
    fn default() -> Self {
        Self {
            total_appended: AtomicU64::new(0),
            total_trimmed: AtomicU64::new(0),
            total_queries_started: AtomicU64::new(0),
            total_queries_completed: AtomicU64::new(0),
            total_queries_failed: AtomicU64::new(0),
        }
    }
}

impl AuditStats {
    /// Captures a consistent-enough snapshot of all counters.
    ///
    /// Because each counter is updated atomically but independently, there is
    /// no cross-counter transaction guarantee; the snapshot is suitable for
    /// monitoring and reporting but not for distributed coordination.
    pub fn snapshot(&self) -> AuditStatsSnapshot {
        AuditStatsSnapshot {
            total_appended: self.total_appended.load(Ordering::Acquire),
            total_trimmed: self.total_trimmed.load(Ordering::Acquire),
            total_queries_started: self.total_queries_started.load(Ordering::Acquire),
            total_queries_completed: self.total_queries_completed.load(Ordering::Acquire),
            total_queries_failed: self.total_queries_failed.load(Ordering::Acquire),
        }
    }
}

// ─── InferenceAuditLog ───────────────────────────────────────────────────────

/// Immutable, append-only audit trail for distributed inference queries.
///
/// # Thread Safety
///
/// All public methods are `&self` and safe to call from multiple threads
/// concurrently. Sequence assignment and entry insertion are performed under
/// a single [`Mutex`] to guarantee ordering; stat counters use relaxed/acquire
/// atomics for low-overhead reads.
///
/// # Capacity
///
/// When `entry_count()` equals `max_entries`, [`append`] returns
/// [`AuditError::CapacityExceeded`]. Call [`trim_before`] to free space.
///
/// [`append`]: InferenceAuditLog::append
/// [`trim_before`]: InferenceAuditLog::trim_before
pub struct InferenceAuditLog {
    /// Ordered list of audit entries guarded by a mutex.
    pub entries: Mutex<Vec<AuditEntry>>,
    /// Monotonically increasing sequence counter.
    pub next_sequence: AtomicU64,
    /// Maximum number of entries the log will hold before rejecting appends.
    pub max_entries: usize,
    /// Aggregate statistics.
    pub stats: AuditStats,
}

impl Default for InferenceAuditLog {
    fn default() -> Self {
        Self::new(100_000)
    }
}

impl InferenceAuditLog {
    /// Creates a new `InferenceAuditLog` with the given capacity.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
            next_sequence: AtomicU64::new(0),
            max_entries,
            stats: AuditStats::default(),
        }
    }

    /// Appends a new audit event to the log and returns its sequence number.
    ///
    /// # Errors
    ///
    /// Returns [`AuditError::CapacityExceeded`] if the log is full.
    pub fn append(&self, trace_id: &str, event: AuditEvent) -> Result<u64, AuditError> {
        // Update per-event-type stats before acquiring the lock so we don't
        // hold the lock while doing atomic work.
        match &event {
            AuditEvent::QueryStarted { .. } => {
                self.stats
                    .total_queries_started
                    .fetch_add(1, Ordering::Relaxed);
            }
            AuditEvent::QueryCompleted { .. } => {
                self.stats
                    .total_queries_completed
                    .fetch_add(1, Ordering::Relaxed);
            }
            AuditEvent::QueryFailed { .. } => {
                self.stats
                    .total_queries_failed
                    .fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }

        let mut guard = self.entries.lock().expect("audit log mutex poisoned");

        let current = guard.len();
        if current >= self.max_entries {
            return Err(AuditError::CapacityExceeded {
                current,
                max: self.max_entries,
            });
        }

        let seq = self.next_sequence.fetch_add(1, Ordering::AcqRel);
        guard.push(AuditEntry {
            sequence: seq,
            event,
            trace_id: trace_id.to_string(),
        });

        self.stats.total_appended.fetch_add(1, Ordering::Relaxed);

        Ok(seq)
    }

    /// Returns all audit entries whose `trace_id` matches the given string.
    ///
    /// Entries are returned in insertion (sequence) order.
    pub fn entries_for_trace(&self, trace_id: &str) -> Vec<AuditEntry> {
        let guard = self.entries.lock().expect("audit log mutex poisoned");
        guard
            .iter()
            .filter(|e| e.trace_id == trace_id)
            .cloned()
            .collect()
    }

    /// Returns all entries with `sequence > given`.
    ///
    /// Useful for tailing the log from a known checkpoint.
    pub fn events_since(&self, sequence: u64) -> Vec<AuditEntry> {
        let guard = self.entries.lock().expect("audit log mutex poisoned");
        guard
            .iter()
            .filter(|e| e.sequence > sequence)
            .cloned()
            .collect()
    }

    /// Removes all entries with `sequence < given` and returns how many were removed.
    ///
    /// This operation is O(n) in the number of current entries. It updates the
    /// `total_trimmed` stat by the number of entries actually removed.
    pub fn trim_before(&self, sequence: u64) -> usize {
        let mut guard = self.entries.lock().expect("audit log mutex poisoned");
        let before = guard.len();
        guard.retain(|e| e.sequence >= sequence);
        let removed = before - guard.len();
        if removed > 0 {
            self.stats
                .total_trimmed
                .fetch_add(removed as u64, Ordering::Relaxed);
        }
        removed
    }

    /// Returns the current number of entries in the log.
    pub fn entry_count(&self) -> usize {
        self.entries.lock().expect("audit log mutex poisoned").len()
    }

    /// Returns a sorted, deduplicated list of all trace IDs present in the log.
    pub fn trace_ids(&self) -> Vec<String> {
        let guard = self.entries.lock().expect("audit log mutex poisoned");
        let set: HashSet<&str> = guard.iter().map(|e| e.trace_id.as_str()).collect();
        let mut ids: Vec<String> = set.into_iter().map(str::to_string).collect();
        ids.sort();
        ids
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_started(session_id: &str, ts: u64) -> AuditEvent {
        AuditEvent::QueryStarted {
            goal: format!("parent(X, {})", session_id),
            session_id: session_id.to_string(),
            timestamp_ms: ts,
        }
    }

    fn make_completed(session_id: &str, ts: u64) -> AuditEvent {
        AuditEvent::QueryCompleted {
            session_id: session_id.to_string(),
            total_results: 3,
            duration_ms: ts,
            timestamp_ms: ts,
        }
    }

    fn make_failed(session_id: &str, ts: u64) -> AuditEvent {
        AuditEvent::QueryFailed {
            session_id: session_id.to_string(),
            reason: "timeout".to_string(),
            timestamp_ms: ts,
        }
    }

    fn make_peer_queried(peer_id: &str, ts: u64) -> AuditEvent {
        AuditEvent::PeerQueried {
            peer_id: peer_id.to_string(),
            goal: "grandparent(X, Y)".to_string(),
            timestamp_ms: ts,
        }
    }

    fn make_rule_fired(rule_id: &str, ts: u64) -> AuditEvent {
        AuditEvent::RuleFired {
            rule_id: rule_id.to_string(),
            bindings: {
                let mut m = HashMap::new();
                m.insert("X".to_string(), "alice".to_string());
                m
            },
            timestamp_ms: ts,
        }
    }

    fn make_result_received(peer_id: &str, ts: u64) -> AuditEvent {
        AuditEvent::ResultReceived {
            peer_id: peer_id.to_string(),
            binding_count: 2,
            is_final: true,
            timestamp_ms: ts,
        }
    }

    // ── 1. Append and retrieve by trace_id ──────────────────────────────────

    #[test]
    fn test_append_and_retrieve_by_trace_id() {
        let log = InferenceAuditLog::default();
        log.append("trace-a", make_started("trace-a", 1000))
            .expect("test: should succeed");
        log.append("trace-b", make_started("trace-b", 2000))
            .expect("test: should succeed");
        log.append("trace-a", make_peer_queried("peer-1", 1001))
            .expect("test: should succeed");

        let entries = log.entries_for_trace("trace-a");
        assert_eq!(entries.len(), 2, "trace-a should have 2 entries");
        for e in &entries {
            assert_eq!(e.trace_id, "trace-a");
        }
    }

    // ── 2. events_since sequence filtering ──────────────────────────────────

    #[test]
    fn test_events_since_filters_by_sequence() {
        let log = InferenceAuditLog::default();
        for i in 0..5_u64 {
            log.append("t", make_started("t", i * 100))
                .expect("test: should succeed");
        }
        // sequences 0..4 inclusive; ask for > 2 → [3, 4]
        let since = log.events_since(2);
        assert_eq!(since.len(), 2);
        assert!(since.iter().all(|e| e.sequence > 2));
    }

    // ── 3. trim_before removes correct entries ───────────────────────────────

    #[test]
    fn test_trim_before_removes_correct_entries() {
        let log = InferenceAuditLog::default();
        for i in 0..10_u64 {
            log.append("t", make_started("t", i))
                .expect("test: should succeed");
        }
        // Remove sequences 0..4 (i.e. seq < 5)
        let removed = log.trim_before(5);
        assert_eq!(removed, 5, "should have removed 5 entries");

        let remaining = log.entries_for_trace("t");
        assert!(remaining.iter().all(|e| e.sequence >= 5));
    }

    // ── 4. trace_ids returns unique, sorted values ───────────────────────────

    #[test]
    fn test_trace_ids_unique_sorted() {
        let log = InferenceAuditLog::default();
        log.append("zebra", make_started("zebra", 1))
            .expect("test: should succeed");
        log.append("alpha", make_started("alpha", 2))
            .expect("test: should succeed");
        log.append("zebra", make_peer_queried("p", 3))
            .expect("test: should succeed");
        log.append("mango", make_started("mango", 4))
            .expect("test: should succeed");

        let ids = log.trace_ids();
        assert_eq!(ids, vec!["alpha", "mango", "zebra"]);
    }

    // ── 5. entry_count decreases after trim ──────────────────────────────────

    #[test]
    fn test_entry_count_decreases_after_trim() {
        let log = InferenceAuditLog::default();
        for i in 0..20_u64 {
            log.append("t", make_started("t", i))
                .expect("test: should succeed");
        }
        assert_eq!(log.entry_count(), 20);
        log.trim_before(10);
        assert_eq!(log.entry_count(), 10);
    }

    // ── 6. AuditEvent::event_type for each variant ──────────────────────────

    #[test]
    fn test_event_type_query_started() {
        assert_eq!(make_started("s", 0).event_type(), "query_started");
    }

    #[test]
    fn test_event_type_peer_queried() {
        assert_eq!(make_peer_queried("p", 0).event_type(), "peer_queried");
    }

    #[test]
    fn test_event_type_rule_fired() {
        assert_eq!(make_rule_fired("r", 0).event_type(), "rule_fired");
    }

    #[test]
    fn test_event_type_result_received() {
        assert_eq!(make_result_received("p", 0).event_type(), "result_received");
    }

    #[test]
    fn test_event_type_query_completed() {
        assert_eq!(make_completed("s", 0).event_type(), "query_completed");
    }

    #[test]
    fn test_event_type_query_failed() {
        assert_eq!(make_failed("s", 0).event_type(), "query_failed");
    }

    // ── 7. Stats auto-increment for started/completed/failed ─────────────────

    #[test]
    fn test_stats_queries_started_increments() {
        let log = InferenceAuditLog::default();
        log.append("t", make_started("t", 0))
            .expect("test: should succeed");
        log.append("t", make_started("t", 1))
            .expect("test: should succeed");
        let snap = log.stats.snapshot();
        assert_eq!(snap.total_queries_started, 2);
    }

    #[test]
    fn test_stats_queries_completed_increments() {
        let log = InferenceAuditLog::default();
        log.append("t", make_completed("t", 0))
            .expect("test: should succeed");
        let snap = log.stats.snapshot();
        assert_eq!(snap.total_queries_completed, 1);
        assert_eq!(snap.total_queries_started, 0);
    }

    #[test]
    fn test_stats_queries_failed_increments() {
        let log = InferenceAuditLog::default();
        log.append("t", make_failed("t", 0))
            .expect("test: should succeed");
        log.append("t", make_failed("t", 1))
            .expect("test: should succeed");
        log.append("t", make_failed("t", 2))
            .expect("test: should succeed");
        let snap = log.stats.snapshot();
        assert_eq!(snap.total_queries_failed, 3);
    }

    #[test]
    fn test_stats_total_appended_and_trimmed() {
        let log = InferenceAuditLog::default();
        for i in 0..6_u64 {
            log.append("t", make_started("t", i))
                .expect("test: should succeed");
        }
        let removed = log.trim_before(3);
        let snap = log.stats.snapshot();
        assert_eq!(snap.total_appended, 6);
        assert_eq!(snap.total_trimmed, removed as u64);
    }

    // ── 8. Capacity limit enforcement ────────────────────────────────────────

    #[test]
    fn test_capacity_exceeded_error() {
        let log = InferenceAuditLog::new(3);
        log.append("t", make_started("t", 0))
            .expect("test: should succeed");
        log.append("t", make_started("t", 1))
            .expect("test: should succeed");
        log.append("t", make_started("t", 2))
            .expect("test: should succeed");

        let err = log.append("t", make_started("t", 3)).unwrap_err();
        match err {
            AuditError::CapacityExceeded { current, max } => {
                assert_eq!(current, 3);
                assert_eq!(max, 3);
            }
        }
    }

    // ── 9. Multiple trace IDs coexist ────────────────────────────────────────

    #[test]
    fn test_multiple_trace_ids_coexist() {
        let log = InferenceAuditLog::default();
        for i in 0..5_u64 {
            log.append("alpha", make_started("alpha", i))
                .expect("test: should succeed");
            log.append("beta", make_peer_queried("peer", i))
                .expect("test: should succeed");
        }
        assert_eq!(log.entries_for_trace("alpha").len(), 5);
        assert_eq!(log.entries_for_trace("beta").len(), 5);
        assert_eq!(log.entry_count(), 10);
    }

    // ── 10. Sequence is monotonically increasing ──────────────────────────────

    #[test]
    fn test_sequence_monotonically_increasing() {
        let log = InferenceAuditLog::default();
        let mut seqs = Vec::new();
        for i in 0..10_u64 {
            let seq = log
                .append("t", make_started("t", i))
                .expect("test: should succeed");
            seqs.push(seq);
        }
        // Each sequence must be strictly greater than the previous
        for w in seqs.windows(2) {
            assert!(
                w[1] > w[0],
                "sequences must be strictly increasing: {} <= {}",
                w[1],
                w[0]
            );
        }
    }

    // ── 11. events_since with sequence = 0 returns all after first ────────────

    #[test]
    fn test_events_since_zero_excludes_first() {
        let log = InferenceAuditLog::default();
        log.append("t", make_started("t", 0))
            .expect("test: should succeed");
        log.append("t", make_started("t", 1))
            .expect("test: should succeed");
        log.append("t", make_started("t", 2))
            .expect("test: should succeed");
        // sequence > 0 means sequences 1 and 2
        let since = log.events_since(0);
        assert_eq!(since.len(), 2);
    }

    // ── 12. trim_before with seq = 0 removes nothing ─────────────────────────

    #[test]
    fn test_trim_before_zero_removes_nothing() {
        let log = InferenceAuditLog::default();
        log.append("t", make_started("t", 0))
            .expect("test: should succeed");
        log.append("t", make_started("t", 1))
            .expect("test: should succeed");
        let removed = log.trim_before(0);
        assert_eq!(removed, 0);
        assert_eq!(log.entry_count(), 2);
    }

    // ── 13. timestamp_ms accessor correctness ────────────────────────────────

    #[test]
    fn test_timestamp_ms_accessor() {
        assert_eq!(make_started("s", 42).timestamp_ms(), 42);
        assert_eq!(make_peer_queried("p", 99).timestamp_ms(), 99);
        assert_eq!(make_rule_fired("r", 7).timestamp_ms(), 7);
        assert_eq!(make_result_received("p", 500).timestamp_ms(), 500);
        assert_eq!(make_completed("s", 1234).timestamp_ms(), 1234);
        assert_eq!(make_failed("s", 9999).timestamp_ms(), 9999);
    }

    // ── 14. trim_before returns 0 when nothing qualifies ─────────────────────

    #[test]
    fn test_trim_before_high_seq_removes_all() {
        let log = InferenceAuditLog::default();
        for i in 0..5_u64 {
            log.append("t", make_started("t", i))
                .expect("test: should succeed");
        }
        // trim_before 100 removes everything (all seq < 100)
        let removed = log.trim_before(100);
        assert_eq!(removed, 5);
        assert_eq!(log.entry_count(), 0);
    }

    // ── 15. trace_ids is empty when log is empty ─────────────────────────────

    #[test]
    fn test_trace_ids_empty_log() {
        let log = InferenceAuditLog::default();
        assert!(log.trace_ids().is_empty());
    }

    // ── 16. capacity allows appending after trim ──────────────────────────────

    #[test]
    fn test_append_succeeds_after_trim_frees_capacity() {
        let log = InferenceAuditLog::new(3);
        log.append("t", make_started("t", 0))
            .expect("test: should succeed");
        log.append("t", make_started("t", 1))
            .expect("test: should succeed");
        log.append("t", make_started("t", 2))
            .expect("test: should succeed");
        // Full — trim one entry
        log.trim_before(1);
        // Now there's room for one more
        log.append("t", make_started("t", 3))
            .expect("test: should succeed");
        assert_eq!(log.entry_count(), 3);
    }

    // ── 17. RuleFired bindings are preserved ─────────────────────────────────

    #[test]
    fn test_rule_fired_bindings_preserved() {
        let log = InferenceAuditLog::default();
        let mut bindings = HashMap::new();
        bindings.insert("X".to_string(), "alice".to_string());
        bindings.insert("Y".to_string(), "bob".to_string());
        log.append(
            "t",
            AuditEvent::RuleFired {
                rule_id: "rule:parent/2".to_string(),
                bindings: bindings.clone(),
                timestamp_ms: 55,
            },
        )
        .expect("test: should succeed");

        let entries = log.entries_for_trace("t");
        if let AuditEvent::RuleFired {
            rule_id,
            bindings: b,
            ..
        } = &entries[0].event
        {
            assert_eq!(rule_id, "rule:parent/2");
            assert_eq!(b["X"], "alice");
            assert_eq!(b["Y"], "bob");
        } else {
            panic!("expected RuleFired event");
        }
    }
}
