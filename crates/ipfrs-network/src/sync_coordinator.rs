//! Peer sync coordinator for bidirectional synchronization sessions.
//!
//! Coordinates bidirectional synchronization sessions between peers, tracking
//! what each peer has, what's needed, and managing sync state machine transitions.

use std::collections::HashMap;

/// Phase of a sync session's state machine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SyncPhase {
    /// Initial capability exchange.
    Handshake,
    /// Exchanging have/want lists.
    Discovery,
    /// Actively transferring blocks.
    Transfer,
    /// Checksumming received blocks.
    Verification,
    /// Sync finished successfully.
    Complete,
    /// Terminal failure state.
    Failed { reason: String },
}

/// Direction of data flow in a sync session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyncDirection {
    /// We send to peer.
    Push,
    /// We receive from peer.
    Pull,
    /// Bidirectional transfer.
    Bidirectional,
}

/// A single synchronization session with a remote peer.
#[derive(Clone, Debug)]
pub struct SyncSession {
    /// Unique session identifier.
    pub session_id: u64,
    /// Remote peer identifier.
    pub peer_id: String,
    /// Direction of data flow.
    pub direction: SyncDirection,
    /// Current phase of the state machine.
    pub phase: SyncPhase,
    /// CIDs we have confirmed locally.
    pub have_cids: Vec<String>,
    /// CIDs we still need from the remote peer.
    pub want_cids: Vec<String>,
    /// Total bytes transferred in this session.
    pub transferred_bytes: u64,
    /// Tick at which this session was started.
    pub started_at_tick: u64,
}

impl SyncSession {
    /// Returns `true` if the session has reached a terminal state (`Complete` or `Failed`).
    pub fn is_terminal(&self) -> bool {
        matches!(self.phase, SyncPhase::Complete | SyncPhase::Failed { .. })
    }

    /// Returns the number of CIDs still pending (in `want_cids`).
    pub fn pending_count(&self) -> usize {
        self.want_cids.len()
    }
}

/// Aggregate statistics across all sessions managed by a [`PeerSyncCoordinator`].
#[derive(Clone, Debug, Default)]
pub struct SyncStats {
    /// Total number of sessions ever created.
    pub total_sessions: usize,
    /// Number of sessions currently in a non-terminal state.
    pub active_sessions: usize,
    /// Number of sessions that reached `Complete`.
    pub completed_sessions: usize,
    /// Number of sessions that reached `Failed`.
    pub failed_sessions: usize,
    /// Sum of `transferred_bytes` across all sessions.
    pub total_transferred_bytes: u64,
}

/// Coordinates bidirectional sync sessions between the local node and its peers.
#[derive(Debug, Default)]
pub struct PeerSyncCoordinator {
    /// All sessions keyed by `session_id`.
    pub sessions: HashMap<u64, SyncSession>,
    /// Counter used to assign the next session identifier.
    pub next_session_id: u64,
}

impl PeerSyncCoordinator {
    /// Creates a new, empty coordinator.
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            next_session_id: 0,
        }
    }

    /// Starts a new sync session with `peer_id` in the `Handshake` phase.
    ///
    /// Returns the newly assigned `session_id`.
    pub fn start_session(&mut self, peer_id: &str, direction: SyncDirection, tick: u64) -> u64 {
        let session_id = self.next_session_id;
        self.next_session_id += 1;

        let session = SyncSession {
            session_id,
            peer_id: peer_id.to_owned(),
            direction,
            phase: SyncPhase::Handshake,
            have_cids: Vec::new(),
            want_cids: Vec::new(),
            transferred_bytes: 0,
            started_at_tick: tick,
        };

        self.sessions.insert(session_id, session);
        session_id
    }

    /// Advances `session_id` to `next_phase`.
    ///
    /// Returns `false` when:
    /// - the session does not exist, or
    /// - the session is already in a terminal state.
    pub fn advance_phase(&mut self, session_id: u64, next_phase: SyncPhase) -> bool {
        match self.sessions.get_mut(&session_id) {
            Some(session) if !session.is_terminal() => {
                session.phase = next_phase;
                true
            }
            _ => false,
        }
    }

    /// Appends `cid` to the `want_cids` list of `session_id`.
    ///
    /// Returns `false` when the session does not exist or is terminal.
    pub fn add_want(&mut self, session_id: u64, cid: &str) -> bool {
        match self.sessions.get_mut(&session_id) {
            Some(session) if !session.is_terminal() => {
                session.want_cids.push(cid.to_owned());
                true
            }
            _ => false,
        }
    }

    /// Marks `cid` as received: removes it from `want_cids`, adds it to `have_cids`,
    /// and increments `transferred_bytes` by `bytes`.
    ///
    /// Returns `false` if the session does not exist (even if already terminal — callers
    /// may still want to record bytes for sessions that just completed).
    pub fn mark_received(&mut self, session_id: u64, cid: &str, bytes: u64) -> bool {
        match self.sessions.get_mut(&session_id) {
            Some(session) => {
                session.want_cids.retain(|c| c != cid);
                if !session.have_cids.contains(&cid.to_owned()) {
                    session.have_cids.push(cid.to_owned());
                }
                session.transferred_bytes += bytes;
                true
            }
            None => false,
        }
    }

    /// Transitions `session_id` to `Failed { reason }`.
    ///
    /// Returns `false` if the session does not exist.
    pub fn fail_session(&mut self, session_id: u64, reason: String) -> bool {
        match self.sessions.get_mut(&session_id) {
            Some(session) => {
                session.phase = SyncPhase::Failed { reason };
                true
            }
            None => false,
        }
    }

    /// Transitions `session_id` to `Complete`.
    ///
    /// Returns `false` if the session does not exist.
    pub fn complete_session(&mut self, session_id: u64) -> bool {
        match self.sessions.get_mut(&session_id) {
            Some(session) => {
                session.phase = SyncPhase::Complete;
                true
            }
            None => false,
        }
    }

    /// Returns references to all non-terminal sessions, sorted ascending by `session_id`.
    pub fn active_sessions(&self) -> Vec<&SyncSession> {
        let mut active: Vec<&SyncSession> = self
            .sessions
            .values()
            .filter(|s| !s.is_terminal())
            .collect();
        active.sort_by_key(|s| s.session_id);
        active
    }

    /// Returns a reference to the session with `session_id`, or `None` if not found.
    pub fn session(&self, session_id: u64) -> Option<&SyncSession> {
        self.sessions.get(&session_id)
    }

    /// Computes aggregate statistics over all sessions.
    pub fn stats(&self) -> SyncStats {
        let total_sessions = self.sessions.len();
        let mut active_sessions: usize = 0;
        let mut completed_sessions: usize = 0;
        let mut failed_sessions: usize = 0;
        let mut total_transferred_bytes: u64 = 0;

        for session in self.sessions.values() {
            total_transferred_bytes += session.transferred_bytes;

            match &session.phase {
                SyncPhase::Complete => completed_sessions += 1,
                SyncPhase::Failed { .. } => failed_sessions += 1,
                _ => active_sessions += 1,
            }
        }

        SyncStats {
            total_sessions,
            active_sessions,
            completed_sessions,
            failed_sessions,
            total_transferred_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn coordinator() -> PeerSyncCoordinator {
        PeerSyncCoordinator::new()
    }

    // ── 1. new() starts empty ────────────────────────────────────────────────

    #[test]
    fn test_new_starts_empty() {
        let c = coordinator();
        assert!(c.sessions.is_empty());
        assert_eq!(c.next_session_id, 0);
    }

    // ── 2. start_session creates in Handshake ───────────────────────────────

    #[test]
    fn test_start_session_creates_in_handshake() {
        let mut c = coordinator();
        let id = c.start_session("peer-a", SyncDirection::Pull, 10);
        let s = c.session(id).expect("session must exist");
        assert_eq!(s.phase, SyncPhase::Handshake);
        assert_eq!(s.peer_id, "peer-a");
        assert_eq!(s.started_at_tick, 10);
    }

    // ── 3. start_session increments session_id ──────────────────────────────

    #[test]
    fn test_start_session_increments_id() {
        let mut c = coordinator();
        let id0 = c.start_session("peer-a", SyncDirection::Push, 0);
        let id1 = c.start_session("peer-b", SyncDirection::Pull, 1);
        assert_eq!(id1, id0 + 1);
    }

    // ── 4. advance_phase succeeds ───────────────────────────────────────────

    #[test]
    fn test_advance_phase_succeeds() {
        let mut c = coordinator();
        let id = c.start_session("p", SyncDirection::Bidirectional, 0);
        assert!(c.advance_phase(id, SyncPhase::Discovery));
        assert_eq!(
            c.session(id)
                .expect("test: session must exist after advance_phase")
                .phase,
            SyncPhase::Discovery
        );
    }

    // ── 5. advance_phase fails on unknown session ───────────────────────────

    #[test]
    fn test_advance_phase_unknown_session() {
        let mut c = coordinator();
        assert!(!c.advance_phase(999, SyncPhase::Transfer));
    }

    // ── 6. advance_phase fails on terminal session ──────────────────────────

    #[test]
    fn test_advance_phase_fails_on_terminal() {
        let mut c = coordinator();
        let id = c.start_session("p", SyncDirection::Pull, 0);
        c.complete_session(id);
        assert!(!c.advance_phase(id, SyncPhase::Transfer));
    }

    // ── 7. add_want appends cid ─────────────────────────────────────────────

    #[test]
    fn test_add_want_appends() {
        let mut c = coordinator();
        let id = c.start_session("p", SyncDirection::Pull, 0);
        assert!(c.add_want(id, "cid-1"));
        assert!(c.add_want(id, "cid-2"));
        let s = c.session(id).expect("test: session must exist");
        assert_eq!(s.want_cids, vec!["cid-1", "cid-2"]);
    }

    // ── 8. add_want fails on unknown ────────────────────────────────────────

    #[test]
    fn test_add_want_unknown() {
        let mut c = coordinator();
        assert!(!c.add_want(999, "cid-x"));
    }

    // ── 9. mark_received moves cid from want to have ────────────────────────

    #[test]
    fn test_mark_received_moves_cid() {
        let mut c = coordinator();
        let id = c.start_session("p", SyncDirection::Pull, 0);
        c.add_want(id, "cid-1");
        c.add_want(id, "cid-2");
        assert!(c.mark_received(id, "cid-1", 512));
        let s = c.session(id).expect("test: session must exist");
        assert!(!s.want_cids.contains(&"cid-1".to_owned()));
        assert!(s.have_cids.contains(&"cid-1".to_owned()));
        assert!(s.want_cids.contains(&"cid-2".to_owned()));
    }

    // ── 10. mark_received accumulates bytes ─────────────────────────────────

    #[test]
    fn test_mark_received_accumulates_bytes() {
        let mut c = coordinator();
        let id = c.start_session("p", SyncDirection::Pull, 0);
        c.add_want(id, "cid-a");
        c.add_want(id, "cid-b");
        c.mark_received(id, "cid-a", 100);
        c.mark_received(id, "cid-b", 200);
        assert_eq!(
            c.session(id)
                .expect("test: session must exist")
                .transferred_bytes,
            300
        );
    }

    // ── 11. mark_received returns false if unknown ───────────────────────────

    #[test]
    fn test_mark_received_unknown() {
        let mut c = coordinator();
        assert!(!c.mark_received(999, "cid-x", 0));
    }

    // ── 12. fail_session sets Failed with reason ─────────────────────────────

    #[test]
    fn test_fail_session_sets_reason() {
        let mut c = coordinator();
        let id = c.start_session("p", SyncDirection::Push, 0);
        assert!(c.fail_session(id, "timeout".to_owned()));
        match &c
            .session(id)
            .expect("test: session must exist after fail_session")
            .phase
        {
            SyncPhase::Failed { reason } => assert_eq!(reason, "timeout"),
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    // ── 13. fail_session returns false if unknown ────────────────────────────

    #[test]
    fn test_fail_session_unknown() {
        let mut c = coordinator();
        assert!(!c.fail_session(999, "oops".to_owned()));
    }

    // ── 14. complete_session sets Complete ──────────────────────────────────

    #[test]
    fn test_complete_session() {
        let mut c = coordinator();
        let id = c.start_session("p", SyncDirection::Bidirectional, 0);
        assert!(c.complete_session(id));
        assert_eq!(
            c.session(id)
                .expect("test: session must exist after complete_session")
                .phase,
            SyncPhase::Complete
        );
    }

    // ── 15. is_terminal true for Complete ───────────────────────────────────

    #[test]
    fn test_is_terminal_complete() {
        let mut c = coordinator();
        let id = c.start_session("p", SyncDirection::Pull, 0);
        c.complete_session(id);
        assert!(c
            .session(id)
            .expect("test: session must exist after complete_session")
            .is_terminal());
    }

    // ── 16. is_terminal true for Failed ─────────────────────────────────────

    #[test]
    fn test_is_terminal_failed() {
        let mut c = coordinator();
        let id = c.start_session("p", SyncDirection::Pull, 0);
        c.fail_session(id, "error".to_owned());
        assert!(c
            .session(id)
            .expect("test: session must exist after fail_session")
            .is_terminal());
    }

    // ── 17. is_terminal false for Transfer ──────────────────────────────────

    #[test]
    fn test_is_terminal_false_for_transfer() {
        let mut c = coordinator();
        let id = c.start_session("p", SyncDirection::Pull, 0);
        c.advance_phase(id, SyncPhase::Transfer);
        assert!(!c
            .session(id)
            .expect("test: session must exist after advance_phase")
            .is_terminal());
    }

    // ── 18. active_sessions excludes terminal ───────────────────────────────

    #[test]
    fn test_active_sessions_excludes_terminal() {
        let mut c = coordinator();
        let id0 = c.start_session("p0", SyncDirection::Pull, 0);
        let id1 = c.start_session("p1", SyncDirection::Push, 1);
        c.complete_session(id0);
        let active = c.active_sessions();
        let ids: Vec<u64> = active.iter().map(|s| s.session_id).collect();
        assert!(!ids.contains(&id0));
        assert!(ids.contains(&id1));
    }

    // ── 19. active_sessions sorted by session_id ───────────────────────────

    #[test]
    fn test_active_sessions_sorted() {
        let mut c = coordinator();
        // Insert in reverse order to ensure sort is not relying on insertion order.
        let id2 = c.start_session("p2", SyncDirection::Pull, 0);
        let id1 = c.start_session("p1", SyncDirection::Pull, 1);
        let id0 = c.start_session("p0", SyncDirection::Pull, 2);
        let active = c.active_sessions();
        let ids: Vec<u64> = active.iter().map(|s| s.session_id).collect();
        // id2 < id1 < id0 because they were assigned in ascending order
        assert_eq!(ids, vec![id2, id1, id0]);
    }

    // ── 20. stats total_sessions ─────────────────────────────────────────────

    #[test]
    fn test_stats_total_sessions() {
        let mut c = coordinator();
        c.start_session("p0", SyncDirection::Pull, 0);
        c.start_session("p1", SyncDirection::Push, 1);
        assert_eq!(c.stats().total_sessions, 2);
    }

    // ── 21. stats completed / failed ────────────────────────────────────────

    #[test]
    fn test_stats_completed_and_failed() {
        let mut c = coordinator();
        let id0 = c.start_session("p0", SyncDirection::Pull, 0);
        let id1 = c.start_session("p1", SyncDirection::Push, 1);
        let _id2 = c.start_session("p2", SyncDirection::Bidirectional, 2);
        c.complete_session(id0);
        c.fail_session(id1, "err".to_owned());
        let stats = c.stats();
        assert_eq!(stats.completed_sessions, 1);
        assert_eq!(stats.failed_sessions, 1);
        assert_eq!(stats.active_sessions, 1);
    }

    // ── 22. stats total_transferred_bytes ───────────────────────────────────

    #[test]
    fn test_stats_total_transferred_bytes() {
        let mut c = coordinator();
        let id0 = c.start_session("p0", SyncDirection::Pull, 0);
        let id1 = c.start_session("p1", SyncDirection::Pull, 1);
        c.add_want(id0, "cid-a");
        c.add_want(id1, "cid-b");
        c.mark_received(id0, "cid-a", 1000);
        c.mark_received(id1, "cid-b", 2500);
        assert_eq!(c.stats().total_transferred_bytes, 3500);
    }

    // ── bonus: pending_count ─────────────────────────────────────────────────

    #[test]
    fn test_pending_count() {
        let mut c = coordinator();
        let id = c.start_session("p", SyncDirection::Pull, 0);
        assert_eq!(
            c.session(id)
                .expect("test: session must exist")
                .pending_count(),
            0
        );
        c.add_want(id, "cid-1");
        c.add_want(id, "cid-2");
        assert_eq!(
            c.session(id)
                .expect("test: session must exist")
                .pending_count(),
            2
        );
        c.mark_received(id, "cid-1", 0);
        assert_eq!(
            c.session(id)
                .expect("test: session must exist after mark_received")
                .pending_count(),
            1
        );
    }

    // ── bonus: add_want on terminal session returns false ───────────────────

    #[test]
    fn test_add_want_terminal_returns_false() {
        let mut c = coordinator();
        let id = c.start_session("p", SyncDirection::Push, 0);
        c.complete_session(id);
        assert!(!c.add_want(id, "cid-x"));
    }

    // ── bonus: advance_phase on Failed terminal returns false ────────────────

    #[test]
    fn test_advance_phase_on_failed_terminal() {
        let mut c = coordinator();
        let id = c.start_session("p", SyncDirection::Pull, 0);
        c.fail_session(id, "err".to_owned());
        assert!(!c.advance_phase(id, SyncPhase::Complete));
        // Phase must remain Failed, not silently overwritten.
        assert!(matches!(
            c.session(id).expect("test: session must exist").phase,
            SyncPhase::Failed { .. }
        ));
    }
}
