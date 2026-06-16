//! Session management for distributed backward-chaining inference (v2).
//!
//! This sub-module contains:
//! - Wire-format types (`RemoteResult`, `InferenceRequest`, `InferenceResponse`)
//! - Session tracking (`DistributedInferenceSession`)
//! - High-level reasoner (`DistributedReasonerV2`) with caching and metrics
//! - Streaming result delivery (`InferenceResultStream`, `PartialResult`)

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ─── Wire types ─────────────────────────────────────────────────────────────

/// A single result contributed by a remote peer during a distributed inference
/// session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteResult {
    /// The peer that produced this result.
    pub peer_id: String,
    /// Variable-to-value bindings returned by the remote engine.
    pub bindings: HashMap<String, String>,
    /// Depth of the proof tree the remote engine explored.
    pub proof_depth: u32,
    /// Round-trip latency in milliseconds.
    pub latency_ms: u64,
}

/// Wire format sent to a remote peer asking it to prove a goal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// Correlates the request with its reply.
    pub request_id: String,
    /// The goal to prove (human-readable Datalog string).
    pub goal: String,
    /// Maximum proof depth the remote engine should explore.
    pub max_depth: u32,
    /// PeerId (as a string) of the node that issued this request.
    pub requester_peer_id: String,
}

/// Wire format sent back by a remote peer in response to an [`InferenceRequest`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    /// Matches the [`InferenceRequest::request_id`].
    pub request_id: String,
    /// Each element is one complete set of variable bindings that satisfy the
    /// goal.
    pub bindings: Vec<HashMap<String, String>>,
    /// `true` when at least one proof was found.
    pub proof_found: bool,
    /// Non-`None` when the remote engine encountered an error.
    pub error: Option<String>,
}

// ─── Session ────────────────────────────────────────────────────────────────

/// Tracks a single distributed inference run from start to finish.
#[derive(Debug)]
pub struct DistributedInferenceSession {
    /// Unique identifier for this session (UUID v4).
    pub session_id: String,
    /// The goal being proved (Datalog string).
    pub goal: String,
    /// Bindings resolved locally.
    pub local_results: Vec<String>,
    /// Results contributed by remote peers.
    pub remote_results: Vec<RemoteResult>,
    /// Peers that have been queried but have not yet replied.
    pub pending_peers: HashSet<String>,
    /// Peers that have already replied (or been declared timed-out).
    pub completed_peers: HashSet<String>,
    /// Wall-clock instant at which the session was created.
    pub started_at: std::time::Instant,
    /// How long to wait for all peers before declaring the session timed-out.
    pub timeout: std::time::Duration,
}

impl DistributedInferenceSession {
    /// Construct a new session for `goal` with the given `timeout`.
    pub fn new(goal: &str, timeout: std::time::Duration) -> Self {
        Self {
            session_id: uuid::Uuid::new_v4().to_string(),
            goal: goal.to_string(),
            local_results: Vec::new(),
            remote_results: Vec::new(),
            pending_peers: HashSet::new(),
            completed_peers: HashSet::new(),
            started_at: std::time::Instant::now(),
            timeout,
        }
    }

    /// Returns `true` once all registered peers have replied **or** the
    /// timeout has elapsed.
    ///
    /// A session that has had no peers registered is **not** considered
    /// complete (there is no outstanding work to declare done).  It becomes
    /// complete either when every registered peer has replied or when the
    /// timeout fires.
    pub fn is_complete(&self) -> bool {
        let any_peers_registered =
            !self.pending_peers.is_empty() || !self.completed_peers.is_empty();

        if self.started_at.elapsed() >= self.timeout {
            return true; // timed out regardless of peer state
        }

        // No peers were ever registered → not complete yet.
        if !any_peers_registered {
            return false;
        }

        // All registered peers have replied.
        self.pending_peers.is_empty()
    }

    /// Returns `true` when the session has exceeded its timeout.
    pub fn is_expired(&self) -> bool {
        self.started_at.elapsed() >= self.timeout
    }
}

// ─── Error type ─────────────────────────────────────────────────────────────

/// Errors specific to [`DistributedReasonerV2`] operations.
#[derive(Debug, thiserror::Error)]
pub enum ReasoningError {
    #[error("Session not found: {0}")]
    SessionNotFound(String),

    #[error("Peer already registered: {0}")]
    PeerAlreadyRegistered(String),

    #[error("Peer not registered in session: {0}")]
    PeerNotRegistered(String),

    #[error("Session has already expired")]
    SessionExpired,
}

// ─── Config ──────────────────────────────────────────────────────────────────

/// Tuning parameters for [`DistributedReasonerV2`].
#[derive(Debug, Clone)]
pub struct DistributedReasonerConfig {
    /// Maximum proof depth per query (default `10`).
    pub max_depth: usize,
    /// Per-session timeout (default `30 s`).
    pub timeout: std::time::Duration,
    /// Maximum number of peers to query per session (default `5`).
    pub max_peers: usize,
    /// How long a cached result remains valid (default `5 min`).
    pub cache_ttl: std::time::Duration,
    /// Number of in-flight peer queries to allow concurrently (default `3`).
    pub parallel_queries: usize,
}

impl Default for DistributedReasonerConfig {
    fn default() -> Self {
        Self {
            max_depth: 10,
            timeout: std::time::Duration::from_secs(30),
            max_peers: 5,
            cache_ttl: std::time::Duration::from_secs(300),
            parallel_queries: 3,
        }
    }
}

// ─── Stats ───────────────────────────────────────────────────────────────────

/// Aggregate statistics over all live sessions managed by a
/// [`DistributedReasonerV2`].
#[derive(Debug, Default)]
pub struct SessionStats {
    /// Number of sessions that have not yet completed (or expired).
    pub active_sessions: usize,
    /// Total [`RemoteResult`] objects accumulated across all sessions.
    pub total_results: usize,
    /// Average latency in milliseconds across every known remote result.
    pub avg_latency_ms: f64,
    /// Fraction of `get_session_results` calls that were served from the
    /// result cache (0.0 – 1.0).
    pub cache_hit_rate: f64,
}

// ─── DistributedReasonerV2 ───────────────────────────────────────────────────

/// Enhanced distributed reasoner with real session management, result caching,
/// and cycle-safe peer tracking.
///
/// # Design
///
/// Each call to [`start_session`](DistributedReasonerV2::start_session) creates
/// a [`DistributedInferenceSession`] that tracks which peers have been queried
/// and which have replied.  Results stream in via
/// [`record_remote_result`](DistributedReasonerV2::record_remote_result).  A
/// short-lived LRU-style cache keyed on the goal string avoids redundant
/// network round-trips for recently seen goals.
pub struct DistributedReasonerV2 {
    /// Memoised local backward-chaining engine (reserved for future local
    /// inference integration).
    #[allow(dead_code)]
    local_reasoner: crate::reasoning::MemoizedInferenceEngine,
    /// All live (and recently completed) inference sessions, keyed by
    /// `session_id`.
    pub(super) sessions: HashMap<String, DistributedInferenceSession>,
    /// Goal-string → remote results cache.
    cache: HashMap<String, (std::time::Instant, Vec<RemoteResult>)>,
    /// Tuning parameters.
    config: DistributedReasonerConfig,
    /// Total cache lookups (for hit-rate tracking).
    cache_lookups: u64,
    /// Cache lookups that returned a live entry.
    cache_hits: u64,
}

impl DistributedReasonerV2 {
    /// Construct a new [`DistributedReasonerV2`] with the provided config.
    pub fn new(config: DistributedReasonerConfig) -> Self {
        let cache_mgr = std::sync::Arc::new(crate::cache::CacheManager::new());
        Self {
            local_reasoner: crate::reasoning::MemoizedInferenceEngine::new(cache_mgr),
            sessions: HashMap::new(),
            cache: HashMap::new(),
            config,
            cache_lookups: 0,
            cache_hits: 0,
        }
    }

    // ── Session lifecycle ────────────────────────────────────────────────────

    /// Create a new inference session for `goal` and return its `session_id`.
    ///
    /// The timeout is taken from [`DistributedReasonerConfig::timeout`].
    pub fn start_session(&mut self, goal: &str) -> String {
        let session = DistributedInferenceSession::new(goal, self.config.timeout);
        let id = session.session_id.clone();
        self.sessions.insert(id.clone(), session);
        id
    }

    /// Create a new inference session with a caller-supplied `session_id`.
    ///
    /// Unlike [`start_session`](Self::start_session) the UUID is provided by
    /// the caller so it can be correlated with an outgoing
    /// [`InferenceRequest`].  If a session with the same ID already exists it
    /// is silently replaced.
    pub fn start_session_with_id(&mut self, goal: &str, session_id: &str) {
        let mut session = DistributedInferenceSession::new(goal, self.config.timeout);
        session.session_id = session_id.to_string();
        self.sessions.insert(session_id.to_string(), session);
    }

    /// Register `peer_id` as a query target for the given session.
    ///
    /// Returns [`ReasoningError::SessionNotFound`] when `session_id` is
    /// unknown, [`ReasoningError::PeerAlreadyRegistered`] when the peer was
    /// already added, and [`ReasoningError::SessionExpired`] when the session
    /// has timed out.
    pub fn add_session_peer(
        &mut self,
        session_id: &str,
        peer_id: &str,
    ) -> Result<(), ReasoningError> {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| ReasoningError::SessionNotFound(session_id.to_string()))?;

        if session.pending_peers.contains(peer_id) || session.completed_peers.contains(peer_id) {
            return Err(ReasoningError::PeerAlreadyRegistered(peer_id.to_string()));
        }
        session.pending_peers.insert(peer_id.to_string());
        Ok(())
    }

    /// Record a [`RemoteResult`] for an in-progress session.
    ///
    /// The peer is automatically moved from `pending_peers` to
    /// `completed_peers` if it was registered.  Unknown peers are accepted
    /// anyway (unsolicited partial results are common in gossip networks).
    pub fn record_remote_result(
        &mut self,
        session_id: &str,
        result: RemoteResult,
    ) -> Result<(), ReasoningError> {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| ReasoningError::SessionNotFound(session_id.to_string()))?;

        // Move the peer from pending → completed regardless of whether it was
        // pre-registered (gossip networks may send unsolicited replies).
        let peer = result.peer_id.clone();
        session.pending_peers.remove(&peer);
        session.completed_peers.insert(peer);
        session.remote_results.push(result);
        Ok(())
    }

    /// Mark `peer_id` as having responded (without attaching a result).
    ///
    /// Useful for signalling that a peer replied with "no solution".
    pub fn mark_peer_responded(
        &mut self,
        session_id: &str,
        peer_id: &str,
    ) -> Result<(), ReasoningError> {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| ReasoningError::SessionNotFound(session_id.to_string()))?;

        if !session.pending_peers.contains(peer_id) && !session.completed_peers.contains(peer_id) {
            return Err(ReasoningError::PeerNotRegistered(peer_id.to_string()));
        }
        session.pending_peers.remove(peer_id);
        session.completed_peers.insert(peer_id.to_string());
        Ok(())
    }

    // ── Completion / result retrieval ────────────────────────────────────────

    /// Returns `true` when the session has received responses from all
    /// registered peers or has exceeded its timeout.
    pub fn is_session_complete(&self, session_id: &str) -> bool {
        self.sessions
            .get(session_id)
            .map(|s| s.is_complete())
            .unwrap_or(false)
    }

    /// Return all remote results collected for `session_id`, or `None` when
    /// the session is unknown.
    ///
    /// Also consults the goal cache so that repeated queries benefit from
    /// previously gathered results.
    pub fn get_session_results(&mut self, session_id: &str) -> Option<Vec<RemoteResult>> {
        self.cache_lookups += 1;

        let session = self.sessions.get(session_id)?;
        let goal = session.goal.clone();
        let ttl = self.config.cache_ttl;

        // Check goal-level cache.
        if let Some((cached_at, cached)) = self.cache.get(&goal) {
            if cached_at.elapsed() < ttl {
                self.cache_hits += 1;
                return Some(cached.clone());
            }
        }

        // Build result list from the live session.
        let results = session.remote_results.clone();
        if !results.is_empty() {
            self.cache
                .insert(goal, (std::time::Instant::now(), results.clone()));
        }
        Some(results)
    }

    // ── Maintenance ──────────────────────────────────────────────────────────

    /// Remove all sessions that have exceeded their timeout.
    ///
    /// Returns the number of sessions that were cleaned up.
    pub fn cleanup_expired_sessions(&mut self) -> usize {
        let expired: Vec<String> = self
            .sessions
            .iter()
            .filter(|(_, s)| s.is_expired())
            .map(|(id, _)| id.clone())
            .collect();

        let count = expired.len();
        for id in expired {
            self.sessions.remove(&id);
        }
        count
    }

    /// Evict goal-cache entries whose TTL has elapsed.
    ///
    /// Called automatically by `cleanup_expired_sessions` but also available
    /// for manual invocation.
    pub fn evict_stale_cache(&mut self) -> usize {
        let ttl = self.config.cache_ttl;
        let before = self.cache.len();
        self.cache
            .retain(|_, (cached_at, _)| cached_at.elapsed() < ttl);
        before - self.cache.len()
    }

    // ── Statistics ───────────────────────────────────────────────────────────

    /// Produce a snapshot of aggregate statistics over all live sessions.
    pub fn session_stats(&self) -> SessionStats {
        let active_sessions = self.sessions.values().filter(|s| !s.is_complete()).count();

        let all_results: Vec<&RemoteResult> = self
            .sessions
            .values()
            .flat_map(|s| s.remote_results.iter())
            .collect();

        let total_results = all_results.len();

        let avg_latency_ms = if total_results == 0 {
            0.0
        } else {
            let sum: u64 = all_results.iter().map(|r| r.latency_ms).sum();
            sum as f64 / total_results as f64
        };

        let cache_hit_rate = if self.cache_lookups == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.cache_lookups as f64
        };

        SessionStats {
            active_sessions,
            total_results,
            avg_latency_ms,
            cache_hit_rate,
        }
    }

    /// Number of currently tracked sessions (active + recently completed).
    #[inline]
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    // ── Phase 2: session lifecycle management ────────────────────────────────

    /// Garbage-collect sessions older than `max_age_secs`.
    ///
    /// A session is considered "old" when the wall-clock time since it was
    /// started exceeds `max_age_secs`, regardless of whether it completed
    /// normally.  Returns the number of sessions that were removed.
    pub fn gc_sessions(&mut self, max_age_secs: u64) -> usize {
        let max_age = std::time::Duration::from_secs(max_age_secs);
        let expired: Vec<String> = self
            .sessions
            .iter()
            .filter(|(_, s)| s.started_at.elapsed() >= max_age)
            .map(|(id, _)| id.clone())
            .collect();

        let count = expired.len();
        for id in expired {
            self.sessions.remove(&id);
        }
        count
    }

    /// Return a [`SessionMetrics`] snapshot for monitoring purposes.
    ///
    /// - `active_sessions`  — sessions that have not yet completed.
    /// - `completed_sessions` — sessions where all peers replied or timed-out.
    /// - `expired_sessions`  — sessions that exceeded their per-session timeout.
    /// - `avg_peers_per_session` — mean number of peers registered per session.
    /// - `avg_latency_ms`   — mean `latency_ms` across all `RemoteResult`s.
    pub fn session_metrics(&self) -> SessionMetrics {
        let mut active_sessions: usize = 0;
        let mut completed_sessions: usize = 0;
        let mut expired_sessions: usize = 0;
        let mut total_peers: usize = 0;
        let mut latency_sum: u64 = 0;
        let mut result_count: usize = 0;

        for session in self.sessions.values() {
            let is_expired = session.is_expired();
            let is_complete = session.is_complete();

            if is_expired {
                expired_sessions += 1;
            }
            if is_complete {
                completed_sessions += 1;
            } else {
                active_sessions += 1;
            }

            total_peers += session.pending_peers.len() + session.completed_peers.len();

            for result in &session.remote_results {
                latency_sum += result.latency_ms;
                result_count += 1;
            }
        }

        let session_count = self.sessions.len();
        let avg_peers_per_session = if session_count == 0 {
            0.0
        } else {
            total_peers as f64 / session_count as f64
        };

        let avg_latency_ms = if result_count == 0 {
            0.0
        } else {
            latency_sum as f64 / result_count as f64
        };

        SessionMetrics {
            active_sessions,
            completed_sessions,
            expired_sessions,
            avg_peers_per_session,
            avg_latency_ms,
        }
    }
}

/// Metrics snapshot for [`DistributedReasonerV2`] monitoring.
#[derive(Debug, Clone)]
pub struct SessionMetrics {
    /// Sessions that have not yet completed or expired.
    pub active_sessions: usize,
    /// Sessions where all peers replied or the timeout fired.
    pub completed_sessions: usize,
    /// Sessions that exceeded their per-session timeout.
    pub expired_sessions: usize,
    /// Average number of peers registered across all sessions.
    pub avg_peers_per_session: f64,
    /// Average round-trip latency in milliseconds across all remote results.
    pub avg_latency_ms: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2 Feature 1: Incremental result streaming
// ─────────────────────────────────────────────────────────────────────────────

/// A partial result produced by a single peer during a streaming inference session.
#[derive(Debug, Clone)]
pub struct PartialResult {
    /// Peer that produced this batch of bindings.
    pub peer_id: String,
    /// The new binding maps contributed by this peer in this batch.
    pub new_bindings: Vec<HashMap<String, String>>,
    /// Total number of binding sets accumulated so far across all peers.
    pub total_so_far: usize,
    /// `true` when the deadline has passed or all peers have replied, meaning
    /// no further results will arrive.
    pub is_final: bool,
}

/// A streaming view of an incremental distributed inference session.
///
/// Results arrive asynchronously via an internal `mpsc` channel as each peer
/// responds.  The caller polls `next_partial` to receive one
/// [`PartialResult`] per peer response until the session is complete or the
/// deadline expires.
///
/// # Example
///
/// ```no_run
/// # async fn example() {
/// # use ipfrs_tensorlogic::InferenceResultStream;
/// // (obtained from Node::infer_streaming)
/// let mut stream: InferenceResultStream = todo!();
/// while let Some(partial) = stream.next_partial().await {
///     println!("peer {}: {} new bindings", partial.peer_id, partial.new_bindings.len());
///     if partial.is_final { break; }
/// }
/// # }
/// ```
pub struct InferenceResultStream {
    /// Session identifier for correlating with the originating request.
    session_id: String,
    /// Channel on which peer responses are delivered.
    rx: tokio::sync::mpsc::Receiver<InferenceResponse>,
    /// All binding maps accumulated so far.
    accumulated: Vec<HashMap<String, String>>,
    /// Absolute instant after which the stream is declared finished.
    deadline: tokio::time::Instant,
}

impl InferenceResultStream {
    /// Construct a new stream from a channel receiver and a deadline.
    ///
    /// This is used internally by `Node::infer_streaming`; callers should not
    /// normally construct this directly.
    pub fn new(
        session_id: String,
        rx: tokio::sync::mpsc::Receiver<InferenceResponse>,
        deadline: tokio::time::Instant,
    ) -> Self {
        Self {
            session_id,
            rx,
            accumulated: Vec::new(),
            deadline,
        }
    }

    /// The session identifier.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Number of binding sets accumulated so far.
    pub fn result_count(&self) -> usize {
        self.accumulated.len()
    }

    /// Poll for the next partial result.
    ///
    /// Returns `None` when the deadline has passed or the channel is closed,
    /// indicating that no further results will arrive.
    pub async fn next_partial(&mut self) -> Option<PartialResult> {
        let remaining = self
            .deadline
            .saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return None;
        }

        match tokio::time::timeout(remaining, self.rx.recv()).await {
            Ok(Some(resp)) => {
                let new_bindings: Vec<HashMap<String, String>> = resp.bindings;
                self.accumulated.extend(new_bindings.clone());
                let total_so_far = self.accumulated.len();

                // We can't know if more peers will reply, so `is_final` is
                // determined by whether the deadline is now exhausted.
                let is_final = self
                    .deadline
                    .saturating_duration_since(tokio::time::Instant::now())
                    .is_zero();

                Some(PartialResult {
                    peer_id: resp.request_id,
                    new_bindings,
                    total_so_far,
                    is_final,
                })
            }
            // Channel closed or timeout expired → stream is done.
            Ok(None) | Err(_) => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests for DistributedReasonerV2 and related types
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod distributed_v2_tests {
    use super::*;

    /// Helper that builds a reasoner with predictable, short defaults.
    fn make_reasoner() -> DistributedReasonerV2 {
        DistributedReasonerV2::new(DistributedReasonerConfig {
            max_depth: 5,
            timeout: std::time::Duration::from_secs(10),
            max_peers: 3,
            cache_ttl: std::time::Duration::from_secs(60),
            parallel_queries: 2,
        })
    }

    // ── Session lifecycle ────────────────────────────────────────────────────

    #[test]
    fn test_session_lifecycle() {
        let mut reasoner = make_reasoner();

        let session_id = reasoner.start_session("parent(alice, bob)");
        assert!(!reasoner.is_session_complete(&session_id));

        reasoner
            .add_session_peer(&session_id, "peer1")
            .expect("add peer1");
        reasoner
            .mark_peer_responded(&session_id, "peer1")
            .expect("peer1 responded");

        // All registered peers have replied → session is complete.
        assert!(reasoner.is_session_complete(&session_id));
    }

    #[test]
    fn test_multiple_peers_session() {
        let mut reasoner = make_reasoner();
        let sid = reasoner.start_session("ancestor(alice, Z)");

        reasoner.add_session_peer(&sid, "peer1").expect("add peer1");
        reasoner.add_session_peer(&sid, "peer2").expect("add peer2");

        // Only one has responded yet.
        reasoner
            .mark_peer_responded(&sid, "peer1")
            .expect("peer1 responded");
        assert!(!reasoner.is_session_complete(&sid));

        reasoner
            .mark_peer_responded(&sid, "peer2")
            .expect("peer2 responded");
        assert!(reasoner.is_session_complete(&sid));
    }

    // ── Recording remote results ─────────────────────────────────────────────

    #[test]
    fn test_record_remote_result() {
        let mut reasoner = make_reasoner();
        let sid = reasoner.start_session("parent(alice, X)");

        reasoner
            .add_session_peer(&sid, "peer-alpha")
            .expect("add peer");

        let result = RemoteResult {
            peer_id: "peer-alpha".to_string(),
            bindings: [("X".to_string(), "bob".to_string())].into_iter().collect(),
            proof_depth: 1,
            latency_ms: 42,
        };
        reasoner
            .record_remote_result(&sid, result)
            .expect("record result");

        // The peer should now be in completed_peers, not pending.
        let session = reasoner.sessions.get(&sid).expect("session exists");
        assert!(!session.pending_peers.contains("peer-alpha"));
        assert!(session.completed_peers.contains("peer-alpha"));
        assert_eq!(session.remote_results.len(), 1);

        let results = reasoner.get_session_results(&sid).expect("get results");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].latency_ms, 42);
    }

    #[test]
    fn test_record_result_unknown_session() {
        let mut reasoner = make_reasoner();
        let result = RemoteResult {
            peer_id: "ghost".to_string(),
            bindings: HashMap::new(),
            proof_depth: 0,
            latency_ms: 0,
        };
        let err = reasoner.record_remote_result("no-such-id", result);
        assert!(matches!(err, Err(ReasoningError::SessionNotFound(_))));
    }

    // ── Timeout detection ────────────────────────────────────────────────────

    #[test]
    fn test_session_timeout_detection() {
        let mut reasoner = DistributedReasonerV2::new(DistributedReasonerConfig {
            max_depth: 3,
            // Very short timeout so we can test expiry without sleeping.
            timeout: std::time::Duration::from_nanos(1),
            max_peers: 2,
            cache_ttl: std::time::Duration::from_secs(60),
            parallel_queries: 1,
        });

        let sid = reasoner.start_session("slow_predicate(X)");
        reasoner
            .add_session_peer(&sid, "slow-peer")
            .expect("add peer");

        // Even though "slow-peer" has not replied, the session is considered
        // complete because the timeout (1 ns) has elapsed.
        assert!(reasoner.is_session_complete(&sid));

        let session = reasoner.sessions.get(&sid).expect("session present");
        assert!(session.is_expired());
    }

    // ── Cleanup ──────────────────────────────────────────────────────────────

    #[test]
    fn test_cleanup_expired_sessions() {
        let mut reasoner = DistributedReasonerV2::new(DistributedReasonerConfig {
            max_depth: 3,
            timeout: std::time::Duration::from_nanos(1), // instantly expired
            max_peers: 2,
            cache_ttl: std::time::Duration::from_secs(60),
            parallel_queries: 1,
        });

        let _s1 = reasoner.start_session("foo(X)");
        let _s2 = reasoner.start_session("bar(Y)");
        assert_eq!(reasoner.session_count(), 2);

        let cleaned = reasoner.cleanup_expired_sessions();
        assert_eq!(cleaned, 2);
        assert_eq!(reasoner.session_count(), 0);
    }

    #[test]
    fn test_cleanup_keeps_active_sessions() {
        let mut reasoner = make_reasoner(); // 10 s timeout → will not expire

        let _sid = reasoner.start_session("live_goal(X)");
        assert_eq!(reasoner.session_count(), 1);

        let cleaned = reasoner.cleanup_expired_sessions();
        assert_eq!(cleaned, 0);
        assert_eq!(reasoner.session_count(), 1);
    }

    // ── Statistics ───────────────────────────────────────────────────────────

    #[test]
    fn test_session_stats_empty() {
        let reasoner = make_reasoner();
        let stats = reasoner.session_stats();
        assert_eq!(stats.active_sessions, 0);
        assert_eq!(stats.total_results, 0);
        assert_eq!(stats.avg_latency_ms, 0.0);
        assert_eq!(stats.cache_hit_rate, 0.0);
    }

    #[test]
    fn test_session_stats() {
        let mut reasoner = make_reasoner();
        let sid = reasoner.start_session("grandparent(alice, Z)");

        reasoner.add_session_peer(&sid, "p1").expect("add p1");
        reasoner.add_session_peer(&sid, "p2").expect("add p2");

        reasoner
            .record_remote_result(
                &sid,
                RemoteResult {
                    peer_id: "p1".to_string(),
                    bindings: HashMap::new(),
                    proof_depth: 2,
                    latency_ms: 100,
                },
            )
            .expect("record p1 result");

        reasoner
            .record_remote_result(
                &sid,
                RemoteResult {
                    peer_id: "p2".to_string(),
                    bindings: HashMap::new(),
                    proof_depth: 1,
                    latency_ms: 200,
                },
            )
            .expect("record p2 result");

        let stats = reasoner.session_stats();
        // One session still has 0 pending peers → complete, so active == 0.
        assert_eq!(stats.active_sessions, 0);
        assert_eq!(stats.total_results, 2);
        // Average of 100 and 200.
        assert!((stats.avg_latency_ms - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_session_stats_active_count() {
        let mut reasoner = make_reasoner();
        let sid = reasoner.start_session("pending_goal(X)");
        reasoner
            .add_session_peer(&sid, "waiting-peer")
            .expect("add peer");
        // "waiting-peer" has NOT responded → session is active.
        let stats = reasoner.session_stats();
        assert_eq!(stats.active_sessions, 1);
    }

    // ── Wire-format serialization ────────────────────────────────────────────

    #[test]
    fn test_inference_request_serde() {
        let req = InferenceRequest {
            request_id: "req-001".to_string(),
            goal: "ancestor(alice, X)".to_string(),
            max_depth: 7,
            requester_peer_id: "12D3KooW...".to_string(),
        };

        let json = serde_json::to_string(&req).expect("serialize InferenceRequest");
        let decoded: InferenceRequest =
            serde_json::from_str(&json).expect("deserialize InferenceRequest");

        assert_eq!(req.request_id, decoded.request_id);
        assert_eq!(req.goal, decoded.goal);
        assert_eq!(req.max_depth, decoded.max_depth);
        assert_eq!(req.requester_peer_id, decoded.requester_peer_id);
    }

    #[test]
    fn test_inference_response_serde() {
        let mut bindings = HashMap::new();
        bindings.insert("X".to_string(), "charlie".to_string());

        let resp = InferenceResponse {
            request_id: "req-001".to_string(),
            bindings: vec![bindings],
            proof_found: true,
            error: None,
        };

        let json = serde_json::to_string(&resp).expect("serialize InferenceResponse");
        let decoded: InferenceResponse =
            serde_json::from_str(&json).expect("deserialize InferenceResponse");

        assert_eq!(resp.request_id, decoded.request_id);
        assert!(decoded.proof_found);
        assert!(decoded.error.is_none());
        assert_eq!(decoded.bindings.len(), 1);
        assert_eq!(
            decoded.bindings[0].get("X").map(String::as_str),
            Some("charlie")
        );
    }

    #[test]
    fn test_inference_response_with_error_serde() {
        let resp = InferenceResponse {
            request_id: "req-err".to_string(),
            bindings: vec![],
            proof_found: false,
            error: Some("depth limit exceeded".to_string()),
        };

        let json = serde_json::to_string(&resp).expect("serialize");
        let decoded: InferenceResponse = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded.error.as_deref(), Some("depth limit exceeded"));
    }

    #[test]
    fn test_remote_result_serde() {
        let mut bindings = HashMap::new();
        bindings.insert("Y".to_string(), "dave".to_string());

        let result = RemoteResult {
            peer_id: "QmPeer123".to_string(),
            bindings,
            proof_depth: 3,
            latency_ms: 55,
        };

        let json = serde_json::to_string(&result).expect("serialize RemoteResult");
        let decoded: RemoteResult = serde_json::from_str(&json).expect("deserialize RemoteResult");

        assert_eq!(decoded.peer_id, "QmPeer123");
        assert_eq!(decoded.proof_depth, 3);
        assert_eq!(decoded.latency_ms, 55);
        assert_eq!(decoded.bindings.get("Y").map(String::as_str), Some("dave"));
    }

    // ── Error handling ───────────────────────────────────────────────────────

    #[test]
    fn test_add_peer_unknown_session() {
        let mut reasoner = make_reasoner();
        let err = reasoner.add_session_peer("ghost-session", "peer1");
        assert!(matches!(err, Err(ReasoningError::SessionNotFound(_))));
    }

    #[test]
    fn test_add_duplicate_peer() {
        let mut reasoner = make_reasoner();
        let sid = reasoner.start_session("dup_peer_test(X)");
        reasoner
            .add_session_peer(&sid, "peer1")
            .expect("first add ok");
        let err = reasoner.add_session_peer(&sid, "peer1");
        assert!(matches!(err, Err(ReasoningError::PeerAlreadyRegistered(_))));
    }

    #[test]
    fn test_mark_unregistered_peer_responded() {
        let mut reasoner = make_reasoner();
        let sid = reasoner.start_session("unregistered(X)");
        let err = reasoner.mark_peer_responded(&sid, "ghost-peer");
        assert!(matches!(err, Err(ReasoningError::PeerNotRegistered(_))));
    }

    // ── Cache hit-rate tracking ──────────────────────────────────────────────

    #[test]
    fn test_cache_hit_rate() {
        let mut reasoner = make_reasoner();
        let sid = reasoner.start_session("cached_goal(X)");

        let result = RemoteResult {
            peer_id: "cache-peer".to_string(),
            bindings: HashMap::new(),
            proof_depth: 1,
            latency_ms: 10,
        };
        reasoner.record_remote_result(&sid, result).expect("record");

        // First lookup → populates cache, no hit yet.
        let _ = reasoner.get_session_results(&sid);

        // Start a second session for the SAME goal to trigger the cache.
        let sid2 = reasoner.start_session("cached_goal(X)");
        let _ = reasoner.get_session_results(&sid2);

        let stats = reasoner.session_stats();
        // 2 lookups, 1 hit → rate = 0.5
        assert!((stats.cache_hit_rate - 0.5).abs() < f64::EPSILON);
    }

    // ── Phase 2: gc_sessions and session_metrics ─────────────────────────────

    /// Verify `gc_sessions` removes sessions older than `max_age_secs` and
    /// returns the number pruned.
    #[test]
    fn test_session_gc() {
        // Use a 1-ns timeout so sessions are instantly "old".
        let mut reasoner = DistributedReasonerV2::new(DistributedReasonerConfig {
            max_depth: 3,
            timeout: std::time::Duration::from_nanos(1),
            max_peers: 2,
            cache_ttl: std::time::Duration::from_secs(60),
            parallel_queries: 1,
        });

        // Create 5 sessions; all will be instantly aged.
        for i in 0..5 {
            reasoner.start_session(&format!("goal_{i}(X)"));
        }
        assert_eq!(reasoner.session_count(), 5);

        // GC with max_age = 0 s → all 5 sessions are older than 0 s.
        let removed = reasoner.gc_sessions(0);
        assert_eq!(removed, 5, "all 5 sessions should have been gc'd");
        assert_eq!(reasoner.session_count(), 0);
    }

    /// Verify that `gc_sessions` does not remove recently-created sessions.
    #[test]
    fn test_session_gc_keeps_recent() {
        let mut reasoner = make_reasoner(); // 10 s timeout

        for i in 0..5 {
            reasoner.start_session(&format!("fresh_{i}(X)"));
        }
        assert_eq!(reasoner.session_count(), 5);

        // max_age = 3600 s → none of the sessions are that old yet.
        let removed = reasoner.gc_sessions(3600);
        assert_eq!(removed, 0, "no session should have been gc'd");
        assert_eq!(reasoner.session_count(), 5);
    }

    /// Verify `session_metrics` accurately reflects session states.
    #[test]
    fn test_session_metrics() {
        let mut reasoner = make_reasoner(); // 10 s timeout

        // Session A: active (peer registered but not yet replied).
        let sid_a = reasoner.start_session("active_goal(X)");
        reasoner
            .add_session_peer(&sid_a, "peer-a")
            .expect("add peer");

        // Session B: completed (all peers replied).
        let sid_b = reasoner.start_session("done_goal(Y)");
        reasoner
            .add_session_peer(&sid_b, "peer-b")
            .expect("add peer-b");
        reasoner
            .record_remote_result(
                &sid_b,
                RemoteResult {
                    peer_id: "peer-b".to_string(),
                    bindings: HashMap::new(),
                    proof_depth: 1,
                    latency_ms: 80,
                },
            )
            .expect("record peer-b");

        let metrics = reasoner.session_metrics();

        // Session A is active; session B is completed.
        assert_eq!(metrics.active_sessions, 1, "one session should be active");
        assert_eq!(
            metrics.completed_sessions, 1,
            "one session should be completed"
        );
        // avg_peers: session A has 1 peer (pending), session B has 1 peer
        // (completed) → total 2 peers across 2 sessions → avg 1.0.
        assert!(
            (metrics.avg_peers_per_session - 1.0).abs() < f64::EPSILON,
            "avg_peers_per_session should be 1.0, got {}",
            metrics.avg_peers_per_session
        );
        // Only one remote result recorded (peer-b, 80 ms).
        assert!(
            (metrics.avg_latency_ms - 80.0).abs() < f64::EPSILON,
            "avg_latency_ms should be 80.0, got {}",
            metrics.avg_latency_ms
        );
    }

    // ── Phase 2: InferenceResultStream ───────────────────────────────────────

    /// Push 3 mock responses through the channel; verify that `next_partial`
    /// delivers exactly 3 partial results with the correct binding counts.
    #[tokio::test]
    async fn test_inference_result_stream_collects() {
        let (tx, rx) = tokio::sync::mpsc::channel::<InferenceResponse>(16);

        // Deadline 10 seconds from now — plenty of time.
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut stream = InferenceResultStream::new("session-123".to_string(), rx, deadline);

        // Send 3 mock responses before polling.
        for i in 0u32..3 {
            let mut b = HashMap::new();
            b.insert("X".to_string(), format!("val{i}"));
            tx.send(InferenceResponse {
                request_id: format!("peer-{i}"),
                bindings: vec![b],
                proof_found: true,
                error: None,
            })
            .await
            .expect("send");
        }
        // Drop the sender so the channel closes after the 3 messages.
        drop(tx);

        assert_eq!(stream.session_id(), "session-123");

        let mut received_count = 0usize;
        while let Some(partial) = stream.next_partial().await {
            received_count += 1;
            assert_eq!(
                partial.new_bindings.len(),
                1,
                "each peer sends exactly one binding set"
            );
            assert_eq!(
                partial.total_so_far, received_count,
                "accumulated count must grow monotonically"
            );
        }

        assert_eq!(
            received_count, 3,
            "must have received exactly 3 partial results"
        );
        assert_eq!(stream.result_count(), 3);
    }
}
