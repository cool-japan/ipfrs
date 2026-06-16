//! Peer session lifecycle management with per-peer caps, state machine, and idle eviction.
//!
//! This module provides [`PeerSessionManager`], which tracks active sessions per peer,
//! manages session lifecycle state transitions, enforces per-peer session caps, and
//! applies global idle timeouts based on logical tick counts rather than wall-clock time.
//!
//! # Session Lifecycle
//!
//! ```text
//! Initiated → Established → Closing → Closed
//!     ↓                        ↑
//!     └────────────────────────┘  (idle eviction marks Closed directly)
//! ```
//!
//! # Example
//!
//! ```
//! use ipfrs_network::session_manager::{
//!     PeerSessionManager, SessionManagerConfig, SessionDirection,
//! };
//!
//! let config = SessionManagerConfig::default();
//! let mut mgr = PeerSessionManager::new(config);
//!
//! let sid = mgr
//!     .open_session("peer-alpha", SessionDirection::Inbound, 0)
//!     .expect("open session");
//!
//! assert!(mgr.get_session(sid).is_some());
//! ```

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// SessionState
// ─────────────────────────────────────────────────────────────────────────────

/// Lifecycle state of a single peer session.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PeerSessionState {
    /// Session created, awaiting confirmation from the remote peer.
    Initiated,
    /// Session is active and usable for data exchange.
    Established,
    /// Graceful shutdown is in progress; no new data should be sent.
    Closing,
    /// Terminal state — the session has been fully torn down.
    Closed,
}

impl PeerSessionState {
    /// Returns `true` for the terminal [`PeerSessionState::Closed`] state.
    #[inline]
    pub fn is_terminal(self) -> bool {
        self == Self::Closed
    }

    /// Returns `true` for states that count as "active" in metrics.
    ///
    /// Active = `Initiated | Established | Closing`.
    #[inline]
    pub fn is_active(self) -> bool {
        !self.is_terminal()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SessionDirection
// ─────────────────────────────────────────────────────────────────────────────

/// Direction in which a session was initiated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionDirection {
    /// The remote peer opened this session.
    Inbound,
    /// We opened this session to the remote peer.
    Outbound,
}

// ─────────────────────────────────────────────────────────────────────────────
// PeerSessionEntry
// ─────────────────────────────────────────────────────────────────────────────

/// A single tracked peer session.
#[derive(Clone, Debug)]
pub struct PeerSessionEntry {
    /// Unique identifier for this session.
    pub session_id: u64,
    /// Identifier of the remote peer.
    pub peer_id: String,
    /// Current lifecycle state.
    pub state: PeerSessionState,
    /// Whether this session was inbound or outbound.
    pub direction: SessionDirection,
    /// Logical tick when this session was created.
    pub created_at_tick: u64,
    /// Logical tick of the most recent recorded activity.
    pub last_active_tick: u64,
    /// Cumulative bytes sent to the peer.
    pub bytes_sent: u64,
    /// Cumulative bytes received from the peer.
    pub bytes_received: u64,
}

impl PeerSessionEntry {
    /// Returns `true` if this session is in the terminal [`PeerSessionState::Closed`] state.
    #[inline]
    pub fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }

    /// Number of ticks elapsed since session creation.
    #[inline]
    pub fn duration_ticks(&self, current_tick: u64) -> u64 {
        current_tick.saturating_sub(self.created_at_tick)
    }

    /// Number of ticks elapsed since the last recorded activity.
    #[inline]
    pub fn idle_ticks(&self, current_tick: u64) -> u64 {
        current_tick.saturating_sub(self.last_active_tick)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SessionManagerConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`PeerSessionManager`].
#[derive(Clone, Debug)]
pub struct SessionManagerConfig {
    /// Maximum number of concurrent *active* (non-Closed) sessions per peer.
    pub max_sessions_per_peer: usize,
    /// Maximum total number of *active* (non-Closed) sessions across all peers.
    pub global_max_sessions: usize,
    /// Number of idle ticks before a session is forcibly closed by [`PeerSessionManager::evict_idle`].
    pub session_idle_timeout_ticks: u64,
}

impl Default for SessionManagerConfig {
    fn default() -> Self {
        Self {
            max_sessions_per_peer: 8,
            global_max_sessions: 256,
            session_idle_timeout_ticks: 300,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SessionManagerStats
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregated statistics snapshot for [`PeerSessionManager`].
#[derive(Clone, Debug, Default)]
pub struct SessionManagerStats {
    /// Total number of sessions ever created (including closed ones).
    pub total_sessions: usize,
    /// Number of currently active sessions (`Initiated + Established + Closing`).
    pub active_sessions: usize,
    /// Number of sessions in the `Closed` terminal state.
    pub closed_sessions: usize,
    /// Per-state breakdown.
    pub by_state: HashMap<PeerSessionState, usize>,
    /// Cumulative bytes sent across all sessions.
    pub total_bytes_sent: u64,
    /// Cumulative bytes received across all sessions.
    pub total_bytes_received: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// PeerSessionManager
// ─────────────────────────────────────────────────────────────────────────────

/// Manages the lifecycle of peer sessions with per-peer caps and idle eviction.
///
/// ## Index maintenance
///
/// `peer_sessions` is a secondary index mapping each `peer_id` to its list of
/// `session_id`s.  Closed entries are **not** removed from this index; instead,
/// query methods filter by session state on the fly.  This makes the index
/// append-only and avoids the bookkeeping cost of removal.
pub struct PeerSessionManager {
    /// Primary store: session_id → session data.
    sessions: HashMap<u64, PeerSessionEntry>,
    /// Secondary index: peer_id → list of session_ids (may include closed sessions).
    peer_sessions: HashMap<String, Vec<u64>>,
    /// Monotonically increasing counter used to generate unique session IDs.
    next_session_id: u64,
    /// Manager configuration.
    config: SessionManagerConfig,
}

impl PeerSessionManager {
    /// Create a new manager with the supplied configuration.
    pub fn new(config: SessionManagerConfig) -> Self {
        Self {
            sessions: HashMap::new(),
            peer_sessions: HashMap::new(),
            next_session_id: 1,
            config,
        }
    }

    // ── Capacity helpers ─────────────────────────────────────────────────────

    /// Count of all non-Closed sessions across every peer.
    fn global_active_count(&self) -> usize {
        self.sessions
            .values()
            .filter(|s| s.state.is_active())
            .count()
    }

    /// Count of all non-Closed sessions for a specific peer.
    fn peer_active_count(&self, peer_id: &str) -> usize {
        match self.peer_sessions.get(peer_id) {
            None => 0,
            Some(ids) => ids
                .iter()
                .filter_map(|id| self.sessions.get(id))
                .filter(|s| s.state.is_active())
                .count(),
        }
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /// Open a new session for `peer_id`.
    ///
    /// Checks global active capacity first, then per-peer active capacity.
    ///
    /// # Errors
    ///
    /// * `"global session limit reached"` — the total number of active sessions
    ///   would exceed [`SessionManagerConfig::global_max_sessions`].
    /// * `"per-peer session limit reached"` — this peer already has
    ///   [`SessionManagerConfig::max_sessions_per_peer`] active sessions.
    pub fn open_session(
        &mut self,
        peer_id: &str,
        direction: SessionDirection,
        current_tick: u64,
    ) -> Result<u64, String> {
        // Global cap checked first.
        if self.global_active_count() >= self.config.global_max_sessions {
            return Err("global session limit reached".to_string());
        }

        // Per-peer cap.
        if self.peer_active_count(peer_id) >= self.config.max_sessions_per_peer {
            return Err("per-peer session limit reached".to_string());
        }

        let session_id = self.next_session_id;
        self.next_session_id = self.next_session_id.wrapping_add(1);

        let entry = PeerSessionEntry {
            session_id,
            peer_id: peer_id.to_string(),
            state: PeerSessionState::Initiated,
            direction,
            created_at_tick: current_tick,
            last_active_tick: current_tick,
            bytes_sent: 0,
            bytes_received: 0,
        };

        self.sessions.insert(session_id, entry);
        self.peer_sessions
            .entry(peer_id.to_string())
            .or_default()
            .push(session_id);

        Ok(session_id)
    }

    /// Advance a session's lifecycle state.
    ///
    /// # Errors
    ///
    /// * `"session not found"` — no session with `session_id` exists.
    /// * `"invalid transition"` — the session is already in the `Closed` terminal
    ///   state and cannot transition further.
    pub fn transition(
        &mut self,
        session_id: u64,
        new_state: PeerSessionState,
    ) -> Result<(), String> {
        let session = self
            .sessions
            .get_mut(&session_id)
            .ok_or_else(|| "session not found".to_string())?;

        if session.state == PeerSessionState::Closed {
            return Err("invalid transition".to_string());
        }

        session.state = new_state;
        Ok(())
    }

    /// Record I/O activity for a session.
    ///
    /// Accumulates `bytes_sent` and `bytes_recv` onto the session totals and
    /// updates `last_active_tick` to `current_tick`.
    ///
    /// Returns `false` if no session with `session_id` exists (no-op).
    pub fn record_activity(
        &mut self,
        session_id: u64,
        bytes_sent: u64,
        bytes_recv: u64,
        current_tick: u64,
    ) -> bool {
        match self.sessions.get_mut(&session_id) {
            None => false,
            Some(session) => {
                session.bytes_sent = session.bytes_sent.saturating_add(bytes_sent);
                session.bytes_received = session.bytes_received.saturating_add(bytes_recv);
                session.last_active_tick = current_tick;
                true
            }
        }
    }

    /// Mark idle sessions as `Closed` and return their IDs.
    ///
    /// A session is considered idle when
    /// `idle_ticks(current_tick) >= config.session_idle_timeout_ticks`.
    /// Only non-`Closed` sessions are evaluated.
    pub fn evict_idle(&mut self, current_tick: u64) -> Vec<u64> {
        let timeout = self.config.session_idle_timeout_ticks;
        let mut evicted = Vec::new();

        for session in self.sessions.values_mut() {
            if session.state != PeerSessionState::Closed
                && session.idle_ticks(current_tick) >= timeout
            {
                session.state = PeerSessionState::Closed;
                evicted.push(session.session_id);
            }
        }

        evicted
    }

    /// Return references to all sessions (including closed) associated with `peer_id`.
    pub fn sessions_for_peer(&self, peer_id: &str) -> Vec<&PeerSessionEntry> {
        match self.peer_sessions.get(peer_id) {
            None => Vec::new(),
            Some(ids) => ids.iter().filter_map(|id| self.sessions.get(id)).collect(),
        }
    }

    /// Count of non-Closed sessions for `peer_id`.
    pub fn active_count_for_peer(&self, peer_id: &str) -> usize {
        self.peer_active_count(peer_id)
    }

    /// Look up a session by its ID.
    pub fn get_session(&self, session_id: u64) -> Option<&PeerSessionEntry> {
        self.sessions.get(&session_id)
    }

    /// Compute a statistics snapshot reflecting the current state of all sessions.
    pub fn stats(&self) -> SessionManagerStats {
        let mut by_state: HashMap<PeerSessionState, usize> = HashMap::new();
        let mut total_bytes_sent: u64 = 0;
        let mut total_bytes_received: u64 = 0;

        for session in self.sessions.values() {
            *by_state.entry(session.state).or_insert(0) += 1;
            total_bytes_sent = total_bytes_sent.saturating_add(session.bytes_sent);
            total_bytes_received = total_bytes_received.saturating_add(session.bytes_received);
        }

        let total_sessions = self.sessions.len();
        let closed_sessions = by_state
            .get(&PeerSessionState::Closed)
            .copied()
            .unwrap_or(0);
        let active_sessions = total_sessions - closed_sessions;

        SessionManagerStats {
            total_sessions,
            active_sessions,
            closed_sessions,
            by_state,
            total_bytes_sent,
            total_bytes_received,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public type aliases for external API compatibility
// ─────────────────────────────────────────────────────────────────────────────

/// Public alias for [`PeerSessionState`] matching the required API surface.
pub type SessionState = PeerSessionState;

/// Public alias for [`PeerSessionEntry`] matching the required API surface.
pub type PeerSession = PeerSessionEntry;

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_mgr() -> PeerSessionManager {
        PeerSessionManager::new(SessionManagerConfig::default())
    }

    // ── 1. open_session creates a session ────────────────────────────────────

    #[test]
    fn open_session_creates_session() {
        let mut mgr = default_mgr();
        let sid = mgr
            .open_session("peer-a", SessionDirection::Inbound, 10)
            .expect("open_session failed");
        let session = mgr.get_session(sid).expect("session should exist");
        assert_eq!(session.peer_id, "peer-a");
        assert_eq!(session.state, PeerSessionState::Initiated);
        assert_eq!(session.direction, SessionDirection::Inbound);
        assert_eq!(session.created_at_tick, 10);
        assert_eq!(session.last_active_tick, 10);
        assert_eq!(session.bytes_sent, 0);
        assert_eq!(session.bytes_received, 0);
    }

    // ── 2. open_session returns unique session IDs ────────────────────────────

    #[test]
    fn open_session_returns_unique_ids() {
        let mut mgr = default_mgr();
        let s1 = mgr
            .open_session("peer-a", SessionDirection::Inbound, 0)
            .expect("s1");
        let s2 = mgr
            .open_session("peer-a", SessionDirection::Outbound, 0)
            .expect("s2");
        assert_ne!(s1, s2);
    }

    // ── 3. per-peer cap enforcement ───────────────────────────────────────────

    #[test]
    fn per_peer_cap_enforced() {
        let mut mgr = PeerSessionManager::new(SessionManagerConfig {
            max_sessions_per_peer: 2,
            global_max_sessions: 100,
            ..Default::default()
        });
        mgr.open_session("peer-x", SessionDirection::Inbound, 0)
            .expect("s1");
        mgr.open_session("peer-x", SessionDirection::Outbound, 0)
            .expect("s2");
        let err = mgr
            .open_session("peer-x", SessionDirection::Inbound, 0)
            .expect_err("should be capped");
        assert_eq!(err, "per-peer session limit reached");
    }

    // ── 4. per-peer cap: different peer succeeds after same-peer hits cap ─────

    #[test]
    fn per_peer_cap_does_not_affect_other_peers() {
        let mut mgr = PeerSessionManager::new(SessionManagerConfig {
            max_sessions_per_peer: 1,
            global_max_sessions: 100,
            ..Default::default()
        });
        mgr.open_session("peer-a", SessionDirection::Inbound, 0)
            .expect("peer-a session");
        // peer-a is at cap; peer-b should still succeed
        mgr.open_session("peer-b", SessionDirection::Inbound, 0)
            .expect("peer-b session");
    }

    // ── 5. global cap enforcement ─────────────────────────────────────────────

    #[test]
    fn global_cap_enforced() {
        let mut mgr = PeerSessionManager::new(SessionManagerConfig {
            max_sessions_per_peer: 100,
            global_max_sessions: 2,
            ..Default::default()
        });
        mgr.open_session("peer-a", SessionDirection::Inbound, 0)
            .expect("s1");
        mgr.open_session("peer-b", SessionDirection::Inbound, 0)
            .expect("s2");
        let err = mgr
            .open_session("peer-c", SessionDirection::Inbound, 0)
            .expect_err("should hit global cap");
        assert_eq!(err, "global session limit reached");
    }

    // ── 6. global cap: closed session frees a slot ────────────────────────────

    #[test]
    fn global_cap_freed_by_close() {
        let mut mgr = PeerSessionManager::new(SessionManagerConfig {
            max_sessions_per_peer: 100,
            global_max_sessions: 1,
            ..Default::default()
        });
        let sid = mgr
            .open_session("peer-a", SessionDirection::Inbound, 0)
            .expect("s1");
        // At capacity
        assert!(mgr
            .open_session("peer-b", SessionDirection::Inbound, 0)
            .is_err());
        // Close the first session
        mgr.transition(sid, PeerSessionState::Closed)
            .expect("transition to Closed");
        // Now there is capacity again
        mgr.open_session("peer-b", SessionDirection::Inbound, 0)
            .expect("s2 after slot freed");
    }

    // ── 7. transition Initiated → Established → Closing → Closed ─────────────

    #[test]
    fn full_lifecycle_transition() {
        let mut mgr = default_mgr();
        let sid = mgr
            .open_session("peer-a", SessionDirection::Outbound, 0)
            .expect("open");

        // Initiated → Established
        mgr.transition(sid, PeerSessionState::Established)
            .expect("→ Established");
        assert_eq!(
            mgr.get_session(sid).expect("s").state,
            PeerSessionState::Established
        );

        // Established → Closing
        mgr.transition(sid, PeerSessionState::Closing)
            .expect("→ Closing");
        assert_eq!(
            mgr.get_session(sid).expect("s").state,
            PeerSessionState::Closing
        );

        // Closing → Closed
        mgr.transition(sid, PeerSessionState::Closed)
            .expect("→ Closed");
        assert_eq!(
            mgr.get_session(sid).expect("s").state,
            PeerSessionState::Closed
        );
    }

    // ── 8. transition from Closed is invalid ──────────────────────────────────

    #[test]
    fn transition_from_closed_is_invalid() {
        let mut mgr = default_mgr();
        let sid = mgr
            .open_session("peer-a", SessionDirection::Inbound, 0)
            .expect("open");
        mgr.transition(sid, PeerSessionState::Closed)
            .expect("close");
        let err = mgr
            .transition(sid, PeerSessionState::Established)
            .expect_err("should be invalid");
        assert_eq!(err, "invalid transition");
    }

    // ── 9. transition on missing session returns session not found ────────────

    #[test]
    fn transition_missing_session_returns_not_found() {
        let mut mgr = default_mgr();
        let err = mgr
            .transition(9999, PeerSessionState::Established)
            .expect_err("should not find session 9999");
        assert_eq!(err, "session not found");
    }

    // ── 10. record_activity accumulates bytes and updates tick ─────────────────

    #[test]
    fn record_activity_accumulates_and_updates_tick() {
        let mut mgr = default_mgr();
        let sid = mgr
            .open_session("peer-a", SessionDirection::Inbound, 0)
            .expect("open");

        assert!(mgr.record_activity(sid, 100, 200, 5));
        assert!(mgr.record_activity(sid, 50, 75, 10));

        let session = mgr.get_session(sid).expect("session");
        assert_eq!(session.bytes_sent, 150);
        assert_eq!(session.bytes_received, 275);
        assert_eq!(session.last_active_tick, 10);
    }

    // ── 11. record_activity returns false for unknown session ─────────────────

    #[test]
    fn record_activity_returns_false_for_missing_session() {
        let mut mgr = default_mgr();
        assert!(!mgr.record_activity(0xdeadbeef, 1, 1, 0));
    }

    // ── 12. evict_idle marks sessions Closed ──────────────────────────────────

    #[test]
    fn evict_idle_marks_sessions_closed() {
        let mut mgr = PeerSessionManager::new(SessionManagerConfig {
            session_idle_timeout_ticks: 10,
            ..Default::default()
        });
        let s1 = mgr
            .open_session("peer-a", SessionDirection::Inbound, 0)
            .expect("s1");
        let s2 = mgr
            .open_session("peer-b", SessionDirection::Inbound, 5)
            .expect("s2");

        // At tick 15: s1 idle for 15 ticks (≥ 10), s2 idle for 10 ticks (≥ 10)
        let evicted = mgr.evict_idle(15);
        assert_eq!(evicted.len(), 2);
        assert!(mgr.get_session(s1).expect("s1").state == PeerSessionState::Closed);
        assert!(mgr.get_session(s2).expect("s2").state == PeerSessionState::Closed);
    }

    // ── 13. evict_idle does not close recently active sessions ────────────────

    #[test]
    fn evict_idle_spares_recently_active_sessions() {
        let mut mgr = PeerSessionManager::new(SessionManagerConfig {
            session_idle_timeout_ticks: 100,
            ..Default::default()
        });
        let s1 = mgr
            .open_session("peer-a", SessionDirection::Inbound, 0)
            .expect("s1");
        // Record activity at tick 50 → last_active_tick = 50
        mgr.record_activity(s1, 0, 0, 50);

        // At tick 100: idle = 100 - 50 = 50 < 100 → should NOT be evicted
        let evicted = mgr.evict_idle(100);
        assert!(evicted.is_empty());
        assert_eq!(
            mgr.get_session(s1).expect("s1").state,
            PeerSessionState::Initiated
        );
    }

    // ── 14. evict_idle skips already-Closed sessions ──────────────────────────

    #[test]
    fn evict_idle_skips_already_closed() {
        let mut mgr = PeerSessionManager::new(SessionManagerConfig {
            session_idle_timeout_ticks: 1,
            ..Default::default()
        });
        let sid = mgr
            .open_session("peer-a", SessionDirection::Inbound, 0)
            .expect("open");
        mgr.transition(sid, PeerSessionState::Closed)
            .expect("close");

        // Even though idle >= timeout, Closed sessions must not appear in eviction list
        let evicted = mgr.evict_idle(100);
        assert!(evicted.is_empty());
    }

    // ── 15. sessions_for_peer returns all sessions including closed ────────────

    #[test]
    fn sessions_for_peer_returns_all_including_closed() {
        let mut mgr = default_mgr();
        let s1 = mgr
            .open_session("peer-z", SessionDirection::Inbound, 0)
            .expect("s1");
        let s2 = mgr
            .open_session("peer-z", SessionDirection::Outbound, 0)
            .expect("s2");
        mgr.transition(s1, PeerSessionState::Closed)
            .expect("close s1");

        let sessions = mgr.sessions_for_peer("peer-z");
        assert_eq!(sessions.len(), 2);
        let ids: Vec<u64> = sessions.iter().map(|s| s.session_id).collect();
        assert!(ids.contains(&s1));
        assert!(ids.contains(&s2));
    }

    // ── 16. sessions_for_peer returns empty for unknown peer ──────────────────

    #[test]
    fn sessions_for_peer_returns_empty_for_unknown_peer() {
        let mgr = default_mgr();
        let sessions = mgr.sessions_for_peer("nobody");
        assert!(sessions.is_empty());
    }

    // ── 17. active_count_for_peer excludes Closed sessions ───────────────────

    #[test]
    fn active_count_for_peer_excludes_closed() {
        let mut mgr = default_mgr();
        let s1 = mgr
            .open_session("peer-q", SessionDirection::Inbound, 0)
            .expect("s1");
        mgr.open_session("peer-q", SessionDirection::Outbound, 0)
            .expect("s2");
        assert_eq!(mgr.active_count_for_peer("peer-q"), 2);

        mgr.transition(s1, PeerSessionState::Closed).expect("close");
        assert_eq!(mgr.active_count_for_peer("peer-q"), 1);
    }

    // ── 18. active_count_for_peer returns 0 for unknown peer ─────────────────

    #[test]
    fn active_count_for_peer_returns_zero_for_unknown() {
        let mgr = default_mgr();
        assert_eq!(mgr.active_count_for_peer("nobody"), 0);
    }

    // ── 19. stats by_state counts correctly ───────────────────────────────────

    #[test]
    fn stats_by_state_counts_correctly() {
        let mut mgr = default_mgr();
        let s1 = mgr
            .open_session("p1", SessionDirection::Inbound, 0)
            .expect("s1");
        let s2 = mgr
            .open_session("p2", SessionDirection::Inbound, 0)
            .expect("s2");
        let s3 = mgr
            .open_session("p3", SessionDirection::Outbound, 0)
            .expect("s3");

        mgr.transition(s1, PeerSessionState::Established)
            .expect("s1 established");
        mgr.transition(s2, PeerSessionState::Closing)
            .expect("s2 closing");
        mgr.transition(s3, PeerSessionState::Closed)
            .expect("s3 closed");

        let stats = mgr.stats();
        assert_eq!(stats.total_sessions, 3);
        assert_eq!(stats.active_sessions, 2); // s1 + s2
        assert_eq!(stats.closed_sessions, 1); // s3

        assert_eq!(
            *stats
                .by_state
                .get(&PeerSessionState::Established)
                .unwrap_or(&0),
            1
        );
        assert_eq!(
            *stats.by_state.get(&PeerSessionState::Closing).unwrap_or(&0),
            1
        );
        assert_eq!(
            *stats.by_state.get(&PeerSessionState::Closed).unwrap_or(&0),
            1
        );
    }

    // ── 20. stats total_bytes aggregates across sessions ─────────────────────

    #[test]
    fn stats_aggregates_bytes() {
        let mut mgr = default_mgr();
        let s1 = mgr
            .open_session("p1", SessionDirection::Inbound, 0)
            .expect("s1");
        let s2 = mgr
            .open_session("p2", SessionDirection::Outbound, 0)
            .expect("s2");

        mgr.record_activity(s1, 1000, 2000, 1);
        mgr.record_activity(s2, 500, 750, 1);

        let stats = mgr.stats();
        assert_eq!(stats.total_bytes_sent, 1500);
        assert_eq!(stats.total_bytes_received, 2750);
    }

    // ── 21. duration_ticks helper ─────────────────────────────────────────────

    #[test]
    fn duration_ticks_helper() {
        let mut mgr = default_mgr();
        let sid = mgr
            .open_session("peer-a", SessionDirection::Inbound, 100)
            .expect("open");
        let session = mgr.get_session(sid).expect("session");
        assert_eq!(session.duration_ticks(150), 50);
        assert_eq!(session.duration_ticks(100), 0);
        // Saturating: current < created → 0
        assert_eq!(session.duration_ticks(50), 0);
    }

    // ── 22. idle_ticks helper ─────────────────────────────────────────────────

    #[test]
    fn idle_ticks_helper() {
        let mut mgr = default_mgr();
        let sid = mgr
            .open_session("peer-a", SessionDirection::Inbound, 0)
            .expect("open");
        mgr.record_activity(sid, 0, 0, 40);

        let session = mgr.get_session(sid).expect("session");
        assert_eq!(session.idle_ticks(60), 20);
        assert_eq!(session.idle_ticks(40), 0);
    }

    // ── 23. is_terminal helper ────────────────────────────────────────────────

    #[test]
    fn is_terminal_helper() {
        let mut mgr = default_mgr();
        let sid = mgr
            .open_session("peer-a", SessionDirection::Inbound, 0)
            .expect("open");
        assert!(!mgr.get_session(sid).expect("s").is_terminal());
        mgr.transition(sid, PeerSessionState::Closed)
            .expect("close");
        assert!(mgr.get_session(sid).expect("s").is_terminal());
    }

    // ── 24. per-peer cap counts only active sessions ──────────────────────────

    #[test]
    fn per_peer_cap_counts_only_active() {
        let mut mgr = PeerSessionManager::new(SessionManagerConfig {
            max_sessions_per_peer: 1,
            global_max_sessions: 100,
            ..Default::default()
        });
        let s1 = mgr
            .open_session("peer-a", SessionDirection::Inbound, 0)
            .expect("s1");
        // At cap for peer-a
        assert!(mgr
            .open_session("peer-a", SessionDirection::Inbound, 0)
            .is_err());
        // Close the first session → slot freed
        mgr.transition(s1, PeerSessionState::Closed).expect("close");
        // Now can open another for the same peer
        mgr.open_session("peer-a", SessionDirection::Outbound, 0)
            .expect("s2 after slot freed");
    }

    // ── 25. evict_idle returns evicted session IDs ────────────────────────────

    #[test]
    fn evict_idle_returns_evicted_ids() {
        let mut mgr = PeerSessionManager::new(SessionManagerConfig {
            session_idle_timeout_ticks: 5,
            ..Default::default()
        });
        let s1 = mgr
            .open_session("peer-a", SessionDirection::Inbound, 0)
            .expect("s1");
        // s1 has last_active_tick = 0; at tick 5 idle = 5 >= 5 → evicted
        let evicted = mgr.evict_idle(5);
        assert_eq!(evicted, vec![s1]);
    }

    // ── 26. get_session returns None for unknown ID ───────────────────────────

    #[test]
    fn get_session_returns_none_for_unknown() {
        let mgr = default_mgr();
        assert!(mgr.get_session(0xdeadbeef).is_none());
    }

    // ── 27. outbound direction recorded correctly ─────────────────────────────

    #[test]
    fn outbound_direction_recorded() {
        let mut mgr = default_mgr();
        let sid = mgr
            .open_session("peer-a", SessionDirection::Outbound, 0)
            .expect("open");
        assert_eq!(
            mgr.get_session(sid).expect("s").direction,
            SessionDirection::Outbound
        );
    }

    // ── 28. multiple evict_idle calls are idempotent for closed sessions ──────

    #[test]
    fn evict_idle_idempotent_for_closed() {
        let mut mgr = PeerSessionManager::new(SessionManagerConfig {
            session_idle_timeout_ticks: 1,
            ..Default::default()
        });
        mgr.open_session("peer-a", SessionDirection::Inbound, 0)
            .expect("open");

        let first = mgr.evict_idle(10);
        assert_eq!(first.len(), 1);

        // Second call: already Closed → nothing new
        let second = mgr.evict_idle(20);
        assert!(second.is_empty());
    }

    // ── 29. stats on empty manager ────────────────────────────────────────────

    #[test]
    fn stats_empty_manager() {
        let mgr = default_mgr();
        let stats = mgr.stats();
        assert_eq!(stats.total_sessions, 0);
        assert_eq!(stats.active_sessions, 0);
        assert_eq!(stats.closed_sessions, 0);
        assert_eq!(stats.total_bytes_sent, 0);
        assert_eq!(stats.total_bytes_received, 0);
        assert!(stats.by_state.is_empty());
    }

    // ── 30. Initiated state counted as active in stats ────────────────────────

    #[test]
    fn initiated_counted_as_active_in_stats() {
        let mut mgr = default_mgr();
        mgr.open_session("peer-a", SessionDirection::Inbound, 0)
            .expect("open");
        let stats = mgr.stats();
        assert_eq!(stats.active_sessions, 1);
        assert_eq!(stats.closed_sessions, 0);
        assert_eq!(
            *stats
                .by_state
                .get(&PeerSessionState::Initiated)
                .unwrap_or(&0),
            1
        );
    }
}
