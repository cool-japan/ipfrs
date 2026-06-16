//! Authenticated peer session management with capability negotiation,
//! session tokens, and expiry tracking.

use std::collections::HashMap;

/// Capabilities that can be negotiated for a peer session.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SessionCapability {
    /// Block exchange protocol support
    BlockExchange,
    /// Vector-based similarity search
    VectorSearch,
    /// Tensor logic operations
    TensorLogic,
    /// Gossip relay forwarding
    GossipRelay,
    /// Bootstrap coordination
    Bootstrap,
}

/// Compute FNV-1a hash of a byte slice.
fn fnv1a(data: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;
    let mut hash = OFFSET_BASIS;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// Compute the token_id as FNV-1a hash of (peer_id + nonce string).
fn compute_token_id(peer_id: &str, nonce: u64) -> u64 {
    let combined = format!("{}{}", peer_id, nonce);
    fnv1a(combined.as_bytes())
}

/// A session token encoding identity, timing, and nonce.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionToken {
    /// FNV-1a hash of (peer_id + nonce string)
    pub token_id: u64,
    /// Peer identifier
    pub peer_id: String,
    /// Random nonce for this session
    pub nonce: u64,
    /// Unix timestamp when the token was issued
    pub issued_at_secs: u64,
    /// Unix timestamp when the token expires
    pub expires_at_secs: u64,
}

impl SessionToken {
    /// Returns `true` if the token has expired relative to `now_secs`.
    pub fn is_expired(&self, now_secs: u64) -> bool {
        now_secs >= self.expires_at_secs
    }

    /// Returns the number of seconds until expiry (saturating at 0).
    pub fn remaining_secs(&self, now_secs: u64) -> u64 {
        self.expires_at_secs.saturating_sub(now_secs)
    }
}

/// A live peer session with traffic statistics and negotiated capabilities.
#[derive(Clone, Debug)]
pub struct PeerSession {
    /// Authentication token for this session
    pub token: SessionToken,
    /// Capabilities negotiated at session open time
    pub capabilities: Vec<SessionCapability>,
    /// Total bytes sent to the peer in this session
    pub bytes_sent: u64,
    /// Total bytes received from the peer in this session
    pub bytes_received: u64,
    /// Total number of messages exchanged
    pub message_count: u64,
}

impl PeerSession {
    /// Returns `true` if the session has the given capability.
    pub fn has_capability(&self, cap: SessionCapability) -> bool {
        self.capabilities.contains(&cap)
    }
}

/// Configuration for `PeerSessionManager`.
#[derive(Clone, Debug)]
pub struct SessionManagerConfig {
    /// Session time-to-live in seconds (default: 3600)
    pub session_ttl_secs: u64,
    /// Maximum number of concurrent sessions (default: 1000)
    pub max_sessions: usize,
    /// Starting nonce value — incremented for each new session
    pub nonce_seed: u64,
}

impl Default for SessionManagerConfig {
    fn default() -> Self {
        Self {
            session_ttl_secs: 3600,
            max_sessions: 1000,
            nonce_seed: 0,
        }
    }
}

/// Manages authenticated peer sessions with capability negotiation and expiry.
pub struct PeerSessionManager {
    /// Active sessions keyed by token_id
    sessions: HashMap<u64, PeerSession>,
    /// Manager configuration
    config: SessionManagerConfig,
    /// Monotonically increasing nonce counter
    next_nonce: u64,
}

impl PeerSessionManager {
    /// Create a new `PeerSessionManager` with the given configuration.
    pub fn new(config: SessionManagerConfig) -> Self {
        let next_nonce = config.nonce_seed;
        Self {
            sessions: HashMap::new(),
            config,
            next_nonce,
        }
    }

    /// Open a new authenticated session for `peer_id` with `capabilities`.
    ///
    /// Returns `Ok(token_id)` on success or `Err("session limit reached")` when
    /// the maximum number of concurrent sessions has been reached.
    pub fn open_session(
        &mut self,
        peer_id: String,
        capabilities: Vec<SessionCapability>,
        now_secs: u64,
    ) -> Result<u64, String> {
        if self.sessions.len() >= self.config.max_sessions {
            return Err("session limit reached".to_string());
        }

        let nonce = self.next_nonce;
        self.next_nonce = self.next_nonce.wrapping_add(1);

        let token_id = compute_token_id(&peer_id, nonce);
        let expires_at_secs = now_secs.saturating_add(self.config.session_ttl_secs);

        let token = SessionToken {
            token_id,
            peer_id,
            nonce,
            issued_at_secs: now_secs,
            expires_at_secs,
        };

        let session = PeerSession {
            token,
            capabilities,
            bytes_sent: 0,
            bytes_received: 0,
            message_count: 0,
        };

        self.sessions.insert(token_id, session);
        Ok(token_id)
    }

    /// Look up an active session by its token_id.
    pub fn get_session(&self, token_id: u64) -> Option<&PeerSession> {
        self.sessions.get(&token_id)
    }

    /// Record traffic for a session: increments `bytes_sent`, `bytes_received`,
    /// and `message_count` by 1 for each call.
    pub fn record_traffic(&mut self, token_id: u64, sent: u64, received: u64) {
        if let Some(session) = self.sessions.get_mut(&token_id) {
            session.bytes_sent = session.bytes_sent.saturating_add(sent);
            session.bytes_received = session.bytes_received.saturating_add(received);
            session.message_count = session.message_count.saturating_add(1);
        }
    }

    /// Close (remove) a session by token_id.  Returns `true` if a session was removed.
    pub fn close_session(&mut self, token_id: u64) -> bool {
        self.sessions.remove(&token_id).is_some()
    }

    /// Remove all sessions that have expired relative to `now_secs`.
    /// Returns the number of sessions that were removed.
    pub fn evict_expired(&mut self, now_secs: u64) -> usize {
        let before = self.sessions.len();
        self.sessions
            .retain(|_, session| !session.token.is_expired(now_secs));
        before - self.sessions.len()
    }

    /// Return references to all sessions that have not expired.
    pub fn active_sessions(&self, now_secs: u64) -> Vec<&PeerSession> {
        self.sessions
            .values()
            .filter(|s| !s.token.is_expired(now_secs))
            .collect()
    }

    /// Return references to all non-expired sessions that have the given capability.
    pub fn sessions_with_capability(
        &self,
        cap: SessionCapability,
        now_secs: u64,
    ) -> Vec<&PeerSession> {
        self.sessions
            .values()
            .filter(|s| !s.token.is_expired(now_secs) && s.has_capability(cap))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_manager() -> PeerSessionManager {
        PeerSessionManager::new(SessionManagerConfig::default())
    }

    fn now() -> u64 {
        1_700_000_000_u64
    }

    // ── 1. open_session returns a token_id ────────────────────────────────────

    #[test]
    fn open_session_returns_token_id() {
        let mut mgr = default_manager();
        let result = mgr.open_session(
            "peer-alpha".to_string(),
            vec![SessionCapability::BlockExchange],
            now(),
        );
        assert!(result.is_ok());
    }

    // ── 2. token_id matches FNV-1a(peer_id + nonce) ───────────────────────────

    #[test]
    fn token_id_is_correct_fnv1a_hash() {
        let mut mgr = PeerSessionManager::new(SessionManagerConfig {
            nonce_seed: 42,
            ..Default::default()
        });
        let token_id = mgr
            .open_session("peer-x".to_string(), vec![], now())
            .expect("open_session failed");
        let expected = compute_token_id("peer-x", 42);
        assert_eq!(token_id, expected);
    }

    // ── 3. get_session returns Some for existing session ──────────────────────

    #[test]
    fn get_session_returns_some_for_existing() {
        let mut mgr = default_manager();
        let tid = mgr
            .open_session("peer-a".to_string(), vec![], now())
            .expect("open_session failed");
        assert!(mgr.get_session(tid).is_some());
    }

    // ── 4. get_session returns None for unknown token_id ─────────────────────

    #[test]
    fn get_session_returns_none_for_unknown() {
        let mgr = default_manager();
        assert!(mgr.get_session(9999).is_none());
    }

    // ── 5. session limit Err ──────────────────────────────────────────────────

    #[test]
    fn open_session_errors_when_at_limit() {
        let mut mgr = PeerSessionManager::new(SessionManagerConfig {
            max_sessions: 2,
            ..Default::default()
        });
        mgr.open_session("p1".to_string(), vec![], now())
            .expect("first session");
        mgr.open_session("p2".to_string(), vec![], now())
            .expect("second session");
        let err = mgr.open_session("p3".to_string(), vec![], now());
        assert_eq!(err, Err("session limit reached".to_string()));
    }

    // ── 6. has_capability true ────────────────────────────────────────────────

    #[test]
    fn has_capability_returns_true_when_present() {
        let mut mgr = default_manager();
        let tid = mgr
            .open_session(
                "peer-b".to_string(),
                vec![
                    SessionCapability::VectorSearch,
                    SessionCapability::Bootstrap,
                ],
                now(),
            )
            .expect("open_session failed");
        let session = mgr.get_session(tid).expect("session not found");
        assert!(session.has_capability(SessionCapability::VectorSearch));
        assert!(session.has_capability(SessionCapability::Bootstrap));
    }

    // ── 7. has_capability false ───────────────────────────────────────────────

    #[test]
    fn has_capability_returns_false_when_absent() {
        let mut mgr = default_manager();
        let tid = mgr
            .open_session(
                "peer-c".to_string(),
                vec![SessionCapability::BlockExchange],
                now(),
            )
            .expect("open_session failed");
        let session = mgr.get_session(tid).expect("session not found");
        assert!(!session.has_capability(SessionCapability::TensorLogic));
    }

    // ── 8. record_traffic updates bytes and message_count ────────────────────

    #[test]
    fn record_traffic_updates_bytes_and_count() {
        let mut mgr = default_manager();
        let tid = mgr
            .open_session("peer-d".to_string(), vec![], now())
            .expect("open_session failed");
        mgr.record_traffic(tid, 100, 200);
        mgr.record_traffic(tid, 50, 75);
        let session = mgr.get_session(tid).expect("session not found");
        assert_eq!(session.bytes_sent, 150);
        assert_eq!(session.bytes_received, 275);
        assert_eq!(session.message_count, 2);
    }

    // ── 9. record_traffic on unknown id is a no-op ────────────────────────────

    #[test]
    fn record_traffic_unknown_id_is_noop() {
        let mut mgr = default_manager();
        // Should not panic
        mgr.record_traffic(0xdeadbeef, 1, 1);
    }

    // ── 10. close_session removes and returns true ────────────────────────────

    #[test]
    fn close_session_removes_session() {
        let mut mgr = default_manager();
        let tid = mgr
            .open_session("peer-e".to_string(), vec![], now())
            .expect("open_session failed");
        assert!(mgr.close_session(tid));
        assert!(mgr.get_session(tid).is_none());
    }

    // ── 11. close_session returns false for unknown id ────────────────────────

    #[test]
    fn close_session_returns_false_for_unknown() {
        let mut mgr = default_manager();
        assert!(!mgr.close_session(0xdeadbeef));
    }

    // ── 12. evict_expired removes only expired sessions ───────────────────────

    #[test]
    fn evict_expired_removes_only_expired() {
        let mut mgr = PeerSessionManager::new(SessionManagerConfig {
            session_ttl_secs: 100,
            ..Default::default()
        });
        let t0 = 1_000_u64;
        // Open two sessions at t0
        let tid_a = mgr
            .open_session("peer-f".to_string(), vec![], t0)
            .expect("open a");
        let tid_b = mgr
            .open_session("peer-g".to_string(), vec![], t0)
            .expect("open b");

        // Advance time past both sessions' expiry
        let t1 = t0 + 200;
        // Also open a fresh session at t1
        let tid_c = mgr
            .open_session("peer-h".to_string(), vec![], t1)
            .expect("open c");

        let removed = mgr.evict_expired(t1);
        assert_eq!(removed, 2);
        assert!(mgr.get_session(tid_a).is_none());
        assert!(mgr.get_session(tid_b).is_none());
        assert!(mgr.get_session(tid_c).is_some());
    }

    // ── 13. evict_expired returns 0 when nothing expires ─────────────────────

    #[test]
    fn evict_expired_returns_zero_when_nothing_expired() {
        let mut mgr = default_manager();
        mgr.open_session("peer-i".to_string(), vec![], now())
            .expect("open");
        let removed = mgr.evict_expired(now());
        assert_eq!(removed, 0);
    }

    // ── 14. active_sessions counts correctly ──────────────────────────────────

    #[test]
    fn active_sessions_counts_correctly() {
        let mut mgr = PeerSessionManager::new(SessionManagerConfig {
            session_ttl_secs: 50,
            ..Default::default()
        });
        let t0 = 1_000_u64;
        mgr.open_session("p1".to_string(), vec![], t0).expect("p1");
        mgr.open_session("p2".to_string(), vec![], t0).expect("p2");

        // At t0+30 both are still active
        assert_eq!(mgr.active_sessions(t0 + 30).len(), 2);

        // At t0+60 both have expired
        assert_eq!(mgr.active_sessions(t0 + 60).len(), 0);
    }

    // ── 15. sessions_with_capability filters correctly ────────────────────────

    #[test]
    fn sessions_with_capability_filters_correctly() {
        let mut mgr = default_manager();
        let t = now();
        mgr.open_session(
            "p1".to_string(),
            vec![SessionCapability::GossipRelay, SessionCapability::Bootstrap],
            t,
        )
        .expect("p1");
        mgr.open_session("p2".to_string(), vec![SessionCapability::BlockExchange], t)
            .expect("p2");
        mgr.open_session("p3".to_string(), vec![SessionCapability::GossipRelay], t)
            .expect("p3");

        let gossip_peers = mgr.sessions_with_capability(SessionCapability::GossipRelay, t);
        assert_eq!(gossip_peers.len(), 2);

        let block_peers = mgr.sessions_with_capability(SessionCapability::BlockExchange, t);
        assert_eq!(block_peers.len(), 1);

        let tensor_peers = mgr.sessions_with_capability(SessionCapability::TensorLogic, t);
        assert_eq!(tensor_peers.len(), 0);
    }

    // ── 16. is_expired / remaining_secs ──────────────────────────────────────

    #[test]
    fn token_is_expired_and_remaining_secs() {
        let token = SessionToken {
            token_id: 1,
            peer_id: "p".to_string(),
            nonce: 0,
            issued_at_secs: 1000,
            expires_at_secs: 1100,
        };

        assert!(!token.is_expired(1099));
        assert!(token.is_expired(1100));
        assert!(token.is_expired(1200));

        assert_eq!(token.remaining_secs(1000), 100);
        assert_eq!(token.remaining_secs(1099), 1);
        assert_eq!(token.remaining_secs(1100), 0);
        // saturating_sub: past expiry → 0
        assert_eq!(token.remaining_secs(1200), 0);
    }

    // ── 17. sessions_with_capability excludes expired sessions ────────────────

    #[test]
    fn sessions_with_capability_excludes_expired() {
        let mut mgr = PeerSessionManager::new(SessionManagerConfig {
            session_ttl_secs: 10,
            ..Default::default()
        });
        let t0 = 500_u64;
        mgr.open_session(
            "old-peer".to_string(),
            vec![SessionCapability::VectorSearch],
            t0,
        )
        .expect("old peer");

        // Advance past expiry
        let t1 = t0 + 20;
        mgr.open_session(
            "new-peer".to_string(),
            vec![SessionCapability::VectorSearch],
            t1,
        )
        .expect("new peer");

        let results = mgr.sessions_with_capability(SessionCapability::VectorSearch, t1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].token.peer_id, "new-peer");
    }

    // ── 18. nonce increments per session ─────────────────────────────────────

    #[test]
    fn nonce_increments_per_session() {
        let mut mgr = PeerSessionManager::new(SessionManagerConfig {
            nonce_seed: 10,
            ..Default::default()
        });
        let t = now();
        let tid1 = mgr.open_session("pa".to_string(), vec![], t).expect("s1");
        let tid2 = mgr.open_session("pa".to_string(), vec![], t).expect("s2");

        // Same peer_id but different nonces → different token_ids
        assert_ne!(tid1, tid2);

        let s1 = mgr.get_session(tid1).expect("s1 not found");
        let s2 = mgr.get_session(tid2).expect("s2 not found");
        assert_eq!(s1.token.nonce, 10);
        assert_eq!(s2.token.nonce, 11);
    }

    // ── 19. token issued_at and expires_at are set correctly ─────────────────

    #[test]
    fn token_timestamps_are_set_correctly() {
        let mut mgr = PeerSessionManager::new(SessionManagerConfig {
            session_ttl_secs: 7200,
            ..Default::default()
        });
        let t = now();
        let tid = mgr
            .open_session("peer-ts".to_string(), vec![], t)
            .expect("open");
        let session = mgr.get_session(tid).expect("get");
        assert_eq!(session.token.issued_at_secs, t);
        assert_eq!(session.token.expires_at_secs, t + 7200);
    }

    // ── 20. after close_session, slot can be reused ───────────────────────────

    #[test]
    fn close_session_frees_slot_for_reuse() {
        let mut mgr = PeerSessionManager::new(SessionManagerConfig {
            max_sessions: 1,
            ..Default::default()
        });
        let t = now();
        let tid = mgr.open_session("p1".to_string(), vec![], t).expect("p1");
        // At capacity
        assert!(mgr.open_session("p2".to_string(), vec![], t).is_err());
        // Free the slot
        mgr.close_session(tid);
        // Now it should succeed
        assert!(mgr.open_session("p2".to_string(), vec![], t).is_ok());
    }
}
