//! Distributed Inference Session Manager
//!
//! Production-hardening component for managing concurrent distributed
//! inference sessions with fault tolerance, lifecycle tracking, GC,
//! and atomic metrics collection.
//!
//! # Overview
//!
//! [`DistributedSessionManager`] acts as the central registry for all
//! in-flight and recently-completed distributed reasoning sessions.
//! Each session is identified by a [`SessionId`] (a 128-bit random value
//! rendered as 32 lowercase hex digits).
//!
//! ## Limits
//!
//! At most [`MAX_CONCURRENT_SESSIONS`] (256) sessions may be active at once.
//! Attempts to exceed this limit return [`SessionError::CapacityExceeded`].
//!
//! ## Garbage Collection
//!
//! Call [`DistributedSessionManager::gc_expired_sessions`] periodically to
//! remove sessions whose age exceeds the supplied `max_age` duration.
//!
//! # Examples
//!
//! ```
//! use ipfrs_tensorlogic::session_manager::{
//!     DistributedSessionManager, PeerId, SessionStatus,
//! };
//! use std::time::Duration;
//!
//! let mgr = DistributedSessionManager::new();
//! let id = mgr.create_session("parent(X, Y)", vec![PeerId::new("peer-1")]).expect("example: should succeed in docs");
//!
//! // Transition the session to Running
//! mgr.set_running(&id).expect("example: should succeed in docs");
//!
//! // Inspect status
//! assert!(matches!(mgr.session_status(&id), Some(SessionStatus::Running { .. })));
//!
//! // Cancel it
//! mgr.cancel_session(&id).expect("example: should succeed in docs");
//! assert!(matches!(mgr.session_status(&id), Some(SessionStatus::Cancelled)));
//!
//! // GC sessions older than 1 second
//! mgr.gc_expired_sessions(Duration::from_secs(1));
//! ```

use parking_lot::RwLock;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

// ─── Constants ──────────────────────────────────────────────────────────────

/// Maximum number of concurrent sessions the manager will accept.
pub const MAX_CONCURRENT_SESSIONS: usize = 256;

// ─── PeerId ──────────────────────────────────────────────────────────────────

/// Opaque identifier for a remote peer participating in a session.
///
/// Represented as a plain `String` for compatibility with libp2p `PeerId`
/// string encoding, but kept as a newtype for type safety.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PeerId(pub String);

impl PeerId {
    /// Create a [`PeerId`] from any string-like value.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

impl fmt::Display for PeerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for PeerId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for PeerId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

// ─── SessionId ──────────────────────────────────────────────────────────────

/// 128-bit random session identifier.
///
/// Displayed as 32 lowercase hex characters (no dashes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId([u8; 16]);

impl SessionId {
    /// Generate a new random [`SessionId`] using the OS entropy source.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError::IdGeneration`] if the OS fails to supply
    /// random bytes.
    pub fn new() -> Result<Self, SessionError> {
        let mut buf = [0u8; 16];
        getrandom::fill(&mut buf).map_err(|e| SessionError::IdGeneration(e.to_string()))?;
        Ok(Self(buf))
    }

    /// Construct a [`SessionId`] directly from raw bytes.
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// Return a reference to the underlying raw bytes.
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

// ─── SessionStatus ──────────────────────────────────────────────────────────

/// Lifecycle state of a distributed inference session.
#[derive(Debug, Clone)]
pub enum SessionStatus {
    /// Session has been created but not yet started.
    Pending,

    /// Session is actively running.
    Running {
        /// Wall-clock instant at which `Running` state was entered.
        started_at: Instant,
    },

    /// Session finished successfully.
    Completed {
        /// Optional content-identifier for the assembled proof / result.
        result_cid: Option<String>,
    },

    /// Session terminated due to an error.
    Failed {
        /// Human-readable error description.
        reason: String,
    },

    /// Session was explicitly cancelled by the caller.
    Cancelled,
}

// ─── SessionRecord ───────────────────────────────────────────────────────────

/// Internal record stored per session in the registry.
#[derive(Debug)]
struct SessionRecord {
    /// Canonical goal string supplied at creation time.
    goal: String,

    /// Participating peers at the time of creation.
    peers: Vec<PeerId>,

    /// Current lifecycle status.
    status: SessionStatus,

    /// Wall-clock time at which the session was first created.
    created_at: Instant,
}

// ─── SessionError ────────────────────────────────────────────────────────────

/// Error type for all session manager operations.
#[derive(Debug, Error)]
pub enum SessionError {
    /// Returned when the random-byte generation fails.
    #[error("session id generation failed: {0}")]
    IdGeneration(String),

    /// Returned when the requested session does not exist.
    #[error("session not found")]
    NotFound,

    /// Returned when attempting to transition from an incompatible state.
    #[error("invalid state transition: {0}")]
    InvalidTransition(String),

    /// Returned when the concurrent-session limit is reached.
    #[error("capacity exceeded: maximum {MAX_CONCURRENT_SESSIONS} concurrent sessions")]
    CapacityExceeded,
}

// ─── SessionMetrics ──────────────────────────────────────────────────────────

/// Atomic counters tracking aggregate session outcomes.
///
/// Individual counter values are accessed via [`SessionMetrics::snapshot`]
/// which returns a plain [`SessionMetricsSnapshot`] with no atomics.
#[derive(Debug)]
pub struct SessionMetrics {
    /// Total sessions ever created by this manager instance.
    pub total_created: AtomicU64,

    /// Total sessions that transitioned to [`SessionStatus::Completed`].
    pub total_completed: AtomicU64,

    /// Total sessions that transitioned to [`SessionStatus::Failed`].
    pub total_failed: AtomicU64,

    /// Total sessions that were [`SessionStatus::Cancelled`].
    pub total_cancelled: AtomicU64,
}

impl SessionMetrics {
    /// Create zeroed metrics.
    fn new() -> Self {
        Self {
            total_created: AtomicU64::new(0),
            total_completed: AtomicU64::new(0),
            total_failed: AtomicU64::new(0),
            total_cancelled: AtomicU64::new(0),
        }
    }

    /// Return a plain snapshot of current counter values.
    ///
    /// Counters are read with `Relaxed` ordering; individual values are
    /// consistent but the snapshot is not a linearisable point-in-time view.
    pub fn snapshot(&self) -> SessionMetricsSnapshot {
        SessionMetricsSnapshot {
            total_created: self.total_created.load(Ordering::Relaxed),
            total_completed: self.total_completed.load(Ordering::Relaxed),
            total_failed: self.total_failed.load(Ordering::Relaxed),
            total_cancelled: self.total_cancelled.load(Ordering::Relaxed),
        }
    }
}

// ─── SessionMetricsSnapshot ──────────────────────────────────────────────────

/// A point-in-time copy of [`SessionMetrics`] with plain `u64` fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionMetricsSnapshot {
    /// Total sessions ever created.
    pub total_created: u64,

    /// Total sessions completed successfully.
    pub total_completed: u64,

    /// Total sessions that failed.
    pub total_failed: u64,

    /// Total sessions that were cancelled.
    pub total_cancelled: u64,
}

// ─── DistributedSessionManager ───────────────────────────────────────────────

/// Manages concurrent distributed inference sessions.
///
/// The manager is cheap to clone because all state is held behind an
/// `Arc`-guarded lock.  Clone the manager to share it across threads or
/// async tasks.
///
/// # Thread safety
///
/// All public methods acquire a short-lived read or write lock on the
/// internal registry.  Metrics are updated via lock-free atomics.
#[derive(Debug, Clone)]
pub struct DistributedSessionManager {
    inner: Arc<ManagerInner>,
}

#[derive(Debug)]
struct ManagerInner {
    registry: RwLock<HashMap<SessionId, SessionRecord>>,
    metrics: SessionMetrics,
}

impl DistributedSessionManager {
    /// Create a new, empty session manager.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(ManagerInner {
                registry: RwLock::new(HashMap::new()),
                metrics: SessionMetrics::new(),
            }),
        }
    }

    // ── Lifecycle operations ─────────────────────────────────────────────────

    /// Create a new distributed inference session.
    ///
    /// # Errors
    ///
    /// - [`SessionError::CapacityExceeded`] when there are already
    ///   [`MAX_CONCURRENT_SESSIONS`] active sessions.
    /// - [`SessionError::IdGeneration`] when OS entropy is unavailable.
    pub fn create_session(
        &self,
        goal: &str,
        peers: Vec<PeerId>,
    ) -> Result<SessionId, SessionError> {
        // Check capacity under a read lock first (fast path).
        {
            let guard = self.inner.registry.read();
            if guard.len() >= MAX_CONCURRENT_SESSIONS {
                return Err(SessionError::CapacityExceeded);
            }
        }

        let id = SessionId::new()?;

        let record = SessionRecord {
            goal: goal.to_string(),
            peers,
            status: SessionStatus::Pending,
            created_at: Instant::now(),
        };

        // Re-check capacity under the write lock to close the TOCTOU window.
        let mut guard = self.inner.registry.write();
        if guard.len() >= MAX_CONCURRENT_SESSIONS {
            return Err(SessionError::CapacityExceeded);
        }
        guard.insert(id, record);
        self.inner
            .metrics
            .total_created
            .fetch_add(1, Ordering::Relaxed);

        Ok(id)
    }

    /// Transition a session to [`SessionStatus::Running`].
    ///
    /// Only valid from the [`SessionStatus::Pending`] state.
    ///
    /// # Errors
    ///
    /// - [`SessionError::NotFound`] when `id` is not registered.
    /// - [`SessionError::InvalidTransition`] when the current state is not
    ///   `Pending`.
    pub fn set_running(&self, id: &SessionId) -> Result<(), SessionError> {
        let mut guard = self.inner.registry.write();
        let record = guard.get_mut(id).ok_or(SessionError::NotFound)?;
        match &record.status {
            SessionStatus::Pending => {
                record.status = SessionStatus::Running {
                    started_at: Instant::now(),
                };
                Ok(())
            }
            other => Err(SessionError::InvalidTransition(format!(
                "expected Pending, got {other:?}"
            ))),
        }
    }

    /// Transition a session to [`SessionStatus::Completed`].
    ///
    /// Valid from either `Pending` or `Running`.
    ///
    /// # Errors
    ///
    /// - [`SessionError::NotFound`] when `id` is not registered.
    /// - [`SessionError::InvalidTransition`] when the state is not
    ///   `Pending` or `Running`.
    pub fn set_completed(
        &self,
        id: &SessionId,
        result_cid: Option<String>,
    ) -> Result<(), SessionError> {
        let mut guard = self.inner.registry.write();
        let record = guard.get_mut(id).ok_or(SessionError::NotFound)?;
        match &record.status {
            SessionStatus::Pending | SessionStatus::Running { .. } => {
                record.status = SessionStatus::Completed { result_cid };
                self.inner
                    .metrics
                    .total_completed
                    .fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            other => Err(SessionError::InvalidTransition(format!(
                "cannot complete from {other:?}"
            ))),
        }
    }

    /// Transition a session to [`SessionStatus::Failed`].
    ///
    /// Valid from either `Pending` or `Running`.
    ///
    /// # Errors
    ///
    /// - [`SessionError::NotFound`] when `id` is not registered.
    /// - [`SessionError::InvalidTransition`] when the state is terminal.
    pub fn set_failed(
        &self,
        id: &SessionId,
        reason: impl Into<String>,
    ) -> Result<(), SessionError> {
        let mut guard = self.inner.registry.write();
        let record = guard.get_mut(id).ok_or(SessionError::NotFound)?;
        match &record.status {
            SessionStatus::Pending | SessionStatus::Running { .. } => {
                record.status = SessionStatus::Failed {
                    reason: reason.into(),
                };
                self.inner
                    .metrics
                    .total_failed
                    .fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            other => Err(SessionError::InvalidTransition(format!(
                "cannot fail from {other:?}"
            ))),
        }
    }

    /// Cancel a session.
    ///
    /// The session is transitioned to [`SessionStatus::Cancelled`].  Only
    /// non-terminal sessions (`Pending`, `Running`) may be cancelled.
    ///
    /// # Errors
    ///
    /// - [`SessionError::NotFound`] when `id` is not registered.
    /// - [`SessionError::InvalidTransition`] when the session is already in
    ///   a terminal state.
    pub fn cancel_session(&self, id: &SessionId) -> Result<(), SessionError> {
        let mut guard = self.inner.registry.write();
        let record = guard.get_mut(id).ok_or(SessionError::NotFound)?;
        match &record.status {
            SessionStatus::Pending | SessionStatus::Running { .. } => {
                record.status = SessionStatus::Cancelled;
                self.inner
                    .metrics
                    .total_cancelled
                    .fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            other => Err(SessionError::InvalidTransition(format!(
                "cannot cancel a session in state {other:?}"
            ))),
        }
    }

    // ── Query operations ─────────────────────────────────────────────────────

    /// Return the current status of a session, or `None` if it does not exist.
    pub fn session_status(&self, id: &SessionId) -> Option<SessionStatus> {
        let guard = self.inner.registry.read();
        guard.get(id).map(|r| r.status.clone())
    }

    /// Return the goal string associated with a session.
    pub fn session_goal(&self, id: &SessionId) -> Option<String> {
        let guard = self.inner.registry.read();
        guard.get(id).map(|r| r.goal.clone())
    }

    /// Return the peers associated with a session.
    pub fn session_peers(&self, id: &SessionId) -> Option<Vec<PeerId>> {
        let guard = self.inner.registry.read();
        guard.get(id).map(|r| r.peers.clone())
    }

    /// Return the number of sessions currently in the registry.
    ///
    /// This includes both active and terminal (Completed / Failed / Cancelled)
    /// sessions that have not yet been garbage-collected.
    pub fn active_count(&self) -> usize {
        self.inner.registry.read().len()
    }

    // ── Garbage collection ───────────────────────────────────────────────────

    /// Remove sessions whose `created_at` age exceeds `max_age`.
    ///
    /// This includes sessions in *any* state (Pending, Running, Completed,
    /// Failed, Cancelled) that are older than the threshold.  The method
    /// returns the number of sessions that were removed.
    pub fn gc_expired_sessions(&self, max_age: Duration) -> usize {
        let now = Instant::now();
        let mut guard = self.inner.registry.write();
        let before = guard.len();
        guard.retain(|_id, record| now.duration_since(record.created_at) <= max_age);
        before - guard.len()
    }

    // ── Metrics ──────────────────────────────────────────────────────────────

    /// Return a reference to the live metrics object.
    ///
    /// Use [`SessionMetrics::snapshot`] to get a plain-value copy.
    pub fn metrics(&self) -> &SessionMetrics {
        &self.inner.metrics
    }
}

impl Default for DistributedSessionManager {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn peers(names: &[&str]) -> Vec<PeerId> {
        names.iter().map(|s| PeerId::new(*s)).collect()
    }

    // ── 1. SessionId display format ──────────────────────────────────────────

    #[test]
    fn test_session_id_display_is_32_hex_chars() {
        let id = SessionId::new().expect("getrandom must succeed in tests");
        let s = id.to_string();
        assert_eq!(s.len(), 32, "SessionId display must be 32 hex chars");
        assert!(
            s.chars().all(|c| c.is_ascii_hexdigit()),
            "All chars must be hex digits"
        );
        // Must be lowercase
        assert_eq!(s, s.to_lowercase(), "Hex must be lowercase");
    }

    #[test]
    fn test_session_id_from_bytes_roundtrip() {
        let bytes = [
            0xde, 0xad, 0xbe, 0xef, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99,
            0xaa, 0xbb,
        ];
        let id = SessionId::from_bytes(bytes);
        assert_eq!(id.as_bytes(), &bytes);
        assert_eq!(id.to_string(), "deadbeef00112233445566778899aabb");
    }

    // ── 2. Session creation and initial status ───────────────────────────────

    #[test]
    fn test_create_session_returns_pending() {
        let mgr = DistributedSessionManager::new();
        let id = mgr
            .create_session("parent(X, Y)", peers(&["p1"]))
            .expect("create must succeed");

        assert!(
            matches!(mgr.session_status(&id), Some(SessionStatus::Pending)),
            "Newly created session must be Pending"
        );
    }

    #[test]
    fn test_create_session_increments_active_count() {
        let mgr = DistributedSessionManager::new();
        assert_eq!(mgr.active_count(), 0);
        let _ = mgr
            .create_session("goal_a", peers(&[]))
            .expect("create must succeed");
        assert_eq!(mgr.active_count(), 1);
        let _ = mgr
            .create_session("goal_b", peers(&[]))
            .expect("create must succeed");
        assert_eq!(mgr.active_count(), 2);
    }

    // ── 3. State transitions ─────────────────────────────────────────────────

    #[test]
    fn test_set_running_transitions_from_pending() {
        let mgr = DistributedSessionManager::new();
        let id = mgr
            .create_session("foo", peers(&[]))
            .expect("test: should succeed");
        mgr.set_running(&id).expect("set_running must succeed");
        assert!(matches!(
            mgr.session_status(&id),
            Some(SessionStatus::Running { .. })
        ));
    }

    #[test]
    fn test_set_running_fails_when_not_pending() {
        let mgr = DistributedSessionManager::new();
        let id = mgr
            .create_session("foo", peers(&[]))
            .expect("test: should succeed");
        mgr.set_running(&id).expect("test: should succeed");
        // Second call must fail
        let result = mgr.set_running(&id);
        assert!(
            matches!(result, Err(SessionError::InvalidTransition(_))),
            "set_running on Running state must fail"
        );
    }

    #[test]
    fn test_set_completed_from_running() {
        let mgr = DistributedSessionManager::new();
        let id = mgr
            .create_session("foo", peers(&[]))
            .expect("test: should succeed");
        mgr.set_running(&id).expect("test: should succeed");
        mgr.set_completed(&id, Some("bafybei...".to_string()))
            .expect("complete must succeed");
        assert!(matches!(
            mgr.session_status(&id),
            Some(SessionStatus::Completed { .. })
        ));
    }

    #[test]
    fn test_set_failed_from_running() {
        let mgr = DistributedSessionManager::new();
        let id = mgr
            .create_session("foo", peers(&[]))
            .expect("test: should succeed");
        mgr.set_running(&id).expect("test: should succeed");
        mgr.set_failed(&id, "peer timed out")
            .expect("fail must succeed");
        assert!(matches!(
            mgr.session_status(&id),
            Some(SessionStatus::Failed { .. })
        ));
    }

    // ── 4. Cancellation ──────────────────────────────────────────────────────

    #[test]
    fn test_cancel_pending_session() {
        let mgr = DistributedSessionManager::new();
        let id = mgr
            .create_session("foo", peers(&[]))
            .expect("test: should succeed");
        mgr.cancel_session(&id).expect("cancel must succeed");
        assert!(matches!(
            mgr.session_status(&id),
            Some(SessionStatus::Cancelled)
        ));
    }

    #[test]
    fn test_cancel_running_session() {
        let mgr = DistributedSessionManager::new();
        let id = mgr
            .create_session("foo", peers(&[]))
            .expect("test: should succeed");
        mgr.set_running(&id).expect("test: should succeed");
        mgr.cancel_session(&id)
            .expect("cancel of running must succeed");
        assert!(matches!(
            mgr.session_status(&id),
            Some(SessionStatus::Cancelled)
        ));
    }

    #[test]
    fn test_cancel_completed_session_fails() {
        let mgr = DistributedSessionManager::new();
        let id = mgr
            .create_session("foo", peers(&[]))
            .expect("test: should succeed");
        mgr.set_completed(&id, None).expect("test: should succeed");
        let result = mgr.cancel_session(&id);
        assert!(
            matches!(result, Err(SessionError::InvalidTransition(_))),
            "Cancelling a Completed session must fail"
        );
    }

    #[test]
    fn test_cancel_nonexistent_session_returns_not_found() {
        let mgr = DistributedSessionManager::new();
        let id = SessionId::from_bytes([0u8; 16]);
        let result = mgr.cancel_session(&id);
        assert!(matches!(result, Err(SessionError::NotFound)));
    }

    // ── 5. GC of expired sessions ────────────────────────────────────────────

    #[test]
    fn test_gc_removes_old_sessions() {
        let mgr = DistributedSessionManager::new();
        let _id = mgr
            .create_session("old", peers(&[]))
            .expect("test: should succeed");
        assert_eq!(mgr.active_count(), 1);

        // GC with zero max_age should remove everything (age > 0 immediately).
        let removed = mgr.gc_expired_sessions(Duration::ZERO);
        assert_eq!(removed, 1, "GC must remove the old session");
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_gc_preserves_fresh_sessions() {
        let mgr = DistributedSessionManager::new();
        let _ = mgr
            .create_session("fresh", peers(&[]))
            .expect("test: should succeed");
        assert_eq!(mgr.active_count(), 1);

        // A very long max_age should not remove anything.
        let removed = mgr.gc_expired_sessions(Duration::from_secs(3600));
        assert_eq!(
            removed, 0,
            "No sessions should be GCed with a 1-hour window"
        );
        assert_eq!(mgr.active_count(), 1);
    }

    // ── 6. Max session limit enforcement ────────────────────────────────────

    #[test]
    fn test_capacity_exceeded_returns_error() {
        let mgr = DistributedSessionManager::new();

        for i in 0..MAX_CONCURRENT_SESSIONS {
            mgr.create_session(&format!("goal_{i}"), peers(&[]))
                .unwrap_or_else(|e| panic!("create #{i} must succeed: {e}"));
        }

        // The 257th session must fail.
        let result = mgr.create_session("one_too_many", peers(&[]));
        assert!(
            matches!(result, Err(SessionError::CapacityExceeded)),
            "Must return CapacityExceeded when limit reached"
        );
    }

    // ── 7. Metrics accumulation ──────────────────────────────────────────────

    #[test]
    fn test_metrics_count_created() {
        let mgr = DistributedSessionManager::new();
        let snap_before = mgr.metrics().snapshot();

        let _ = mgr
            .create_session("a", peers(&[]))
            .expect("test: should succeed");
        let _ = mgr
            .create_session("b", peers(&[]))
            .expect("test: should succeed");

        let snap_after = mgr.metrics().snapshot();
        assert_eq!(
            snap_after.total_created - snap_before.total_created,
            2,
            "Two sessions must be counted as created"
        );
    }

    #[test]
    fn test_metrics_count_completed_failed_cancelled() {
        let mgr = DistributedSessionManager::new();

        let id1 = mgr
            .create_session("complete_me", peers(&[]))
            .expect("test: should succeed");
        let id2 = mgr
            .create_session("fail_me", peers(&[]))
            .expect("test: should succeed");
        let id3 = mgr
            .create_session("cancel_me", peers(&[]))
            .expect("test: should succeed");

        mgr.set_completed(&id1, None).expect("test: should succeed");
        mgr.set_failed(&id2, "oops").expect("test: should succeed");
        mgr.cancel_session(&id3).expect("test: should succeed");

        let snap = mgr.metrics().snapshot();
        assert_eq!(snap.total_completed, 1);
        assert_eq!(snap.total_failed, 1);
        assert_eq!(snap.total_cancelled, 1);
    }

    #[test]
    fn test_metrics_snapshot_is_plain_values() {
        // Ensure the snapshot type carries no atomics – just copy it.
        let mgr = DistributedSessionManager::new();
        let _ = mgr
            .create_session("x", peers(&[]))
            .expect("test: should succeed");
        let snap: SessionMetricsSnapshot = mgr.metrics().snapshot();
        let _snap2 = snap; // Must be Copy
        assert_eq!(snap.total_created, 1);
    }

    // ── 8. Goal and peer retrieval ────────────────────────────────────────────

    #[test]
    fn test_session_goal_and_peers_accessible() {
        let mgr = DistributedSessionManager::new();
        let goal = "ancestor(alice, bob)";
        let p = peers(&["peer-alpha", "peer-beta"]);
        let id = mgr
            .create_session(goal, p.clone())
            .expect("test: should succeed");

        assert_eq!(mgr.session_goal(&id).as_deref(), Some(goal));
        assert_eq!(mgr.session_peers(&id), Some(p));
    }

    // ── 9. Status of unknown session ─────────────────────────────────────────

    #[test]
    fn test_status_of_unknown_session_returns_none() {
        let mgr = DistributedSessionManager::new();
        let id = SessionId::from_bytes([0xff; 16]);
        assert!(mgr.session_status(&id).is_none());
    }

    // ── 10. Clone shares state ───────────────────────────────────────────────

    #[test]
    fn test_cloned_manager_shares_registry() {
        let mgr1 = DistributedSessionManager::new();
        let mgr2 = mgr1.clone();

        let id = mgr1
            .create_session("shared", peers(&[]))
            .expect("create must succeed");

        // mgr2 should see the session created via mgr1
        assert!(
            mgr2.session_status(&id).is_some(),
            "Cloned manager must share the same session registry"
        );
    }
}
