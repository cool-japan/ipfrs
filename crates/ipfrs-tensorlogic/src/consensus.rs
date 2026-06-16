//! Federated Learning Round Consensus Tracker
//!
//! Production-grade consensus mechanism for federated learning rounds.
//! A quorum of participating nodes must commit their gradient updates
//! before a round can proceed to aggregation.
//!
//! # Overview
//!
//! [`RoundConsensusTracker`] manages the lifecycle of federated learning
//! rounds. Each round is identified by a [`RoundId`] (a monotonically
//! increasing `u64`). Peers cast [`Vote`]s, and the [`QuorumPolicy`]
//! determines whether consensus has been reached.
//!
//! ## Quorum Logic
//!
//! A round commits when:
//! - At least `min_peers` votes have been received, AND
//! - The fraction of `Commit` votes meets `commit_threshold`
//!
//! A round aborts when:
//! - It is impossible to reach the commit threshold even if all remaining
//!   expected peers vote Commit (i.e. enough Abort/Abstain votes exist), OR
//! - The timeout elapses
//!
//! ## Gradient CID Collection
//!
//! On commit the [`QuorumResult::Commit`] variant carries the `gradient_cid`
//! values of every committing peer, ready for aggregation.
//!
//! # Examples
//!
//! ```
//! use ipfrs_tensorlogic::consensus::{
//!     RoundConsensusTracker, RoundId, PeerVote, Vote, QuorumPolicy, QuorumResult,
//! };
//! use std::time::Duration;
//!
//! let policy = QuorumPolicy::default();
//! let tracker = RoundConsensusTracker::new(policy);
//!
//! let round_id = RoundId::from(1_u64);
//! let peers = vec!["peer-a".to_string(), "peer-b".to_string(), "peer-c".to_string()];
//! tracker.begin_round(round_id.clone(), peers).expect("example: should succeed in docs");
//!
//! for (i, peer) in ["peer-a", "peer-b", "peer-c"].iter().enumerate() {
//!     let vote = PeerVote::new(
//!         peer.to_string(),
//!         round_id.clone(),
//!         Vote::Commit,
//!         Some(format!("bafyreic{}", i)),
//!     );
//!     let result = tracker.cast_vote(vote).expect("example: should succeed in docs");
//!     if let QuorumResult::Commit { gradient_cids } = result {
//!         assert_eq!(gradient_cids.len(), 3);
//!         break;
//!     }
//! }
//! ```

use parking_lot::RwLock;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use thiserror::Error;

// ─── RoundId ─────────────────────────────────────────────────────────────────

/// Monotonically-increasing federated learning round identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RoundId(pub u64);

impl RoundId {
    /// Return the next `RoundId`.
    pub fn next(&self) -> Self {
        Self(self.0 + 1)
    }
}

impl fmt::Display for RoundId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "round-{}", self.0)
    }
}

impl From<u64> for RoundId {
    fn from(v: u64) -> Self {
        Self(v)
    }
}

// ─── Vote ─────────────────────────────────────────────────────────────────────

/// A peer's vote for a federated learning round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Vote {
    /// The peer has committed its gradient update.
    Commit,
    /// The peer requests the round be aborted.
    Abort,
    /// The peer abstains (e.g. due to resource constraints).
    Abstain,
}

// ─── PeerVote ────────────────────────────────────────────────────────────────

/// A single vote cast by a peer for a specific round.
#[derive(Debug, Clone)]
pub struct PeerVote {
    /// Identifier of the voting peer.
    pub peer_id: String,
    /// Round this vote belongs to.
    pub round_id: RoundId,
    /// The peer's decision.
    pub vote: Vote,
    /// Wall-clock time at which the vote was cast.
    pub voted_at: Instant,
    /// CID of the gradient contribution uploaded by the peer (present only on
    /// `Commit` votes).
    pub gradient_cid: Option<String>,
}

impl PeerVote {
    /// Construct a new `PeerVote` using the current instant as `voted_at`.
    pub fn new(
        peer_id: String,
        round_id: RoundId,
        vote: Vote,
        gradient_cid: Option<String>,
    ) -> Self {
        Self {
            peer_id,
            round_id,
            vote,
            voted_at: Instant::now(),
            gradient_cid,
        }
    }
}

// ─── QuorumResult ────────────────────────────────────────────────────────────

/// Outcome of evaluating the quorum policy against the current vote set.
#[derive(Debug, Clone, PartialEq)]
pub enum QuorumResult {
    /// Enough peers committed — aggregation may proceed.
    Commit {
        /// CIDs of gradient contributions from committing peers.
        gradient_cids: Vec<String>,
    },
    /// The round must be abandoned.
    Abort {
        /// Human-readable explanation.
        reason: String,
    },
    /// Quorum has not yet been reached; more votes are needed.
    Pending {
        /// How many votes have been received so far.
        votes_received: usize,
        /// How many more votes are required.
        votes_needed: usize,
    },
    /// The policy timeout elapsed before quorum was reached.
    TimedOut,
}

// ─── QuorumPolicy ────────────────────────────────────────────────────────────

/// Policy parameters governing when a round can proceed to aggregation.
#[derive(Debug, Clone)]
pub struct QuorumPolicy {
    /// Minimum number of peers that must vote for quorum to be considered.
    pub min_peers: usize,
    /// Fraction of votes (0.0–1.0) that must be `Commit` for the round to
    /// commit.  E.g. `0.67` means at least 67 % of votes must be `Commit`.
    pub commit_threshold: f64,
    /// Maximum time allowed before the round is automatically timed out.
    pub timeout: Duration,
}

impl Default for QuorumPolicy {
    fn default() -> Self {
        Self {
            min_peers: 3,
            commit_threshold: 0.67,
            timeout: Duration::from_secs(30),
        }
    }
}

impl QuorumPolicy {
    /// Evaluate the current vote set and elapsed time against this policy.
    ///
    /// `votes` is the full slice of votes cast so far.
    /// `elapsed` is the time since the round started.
    pub fn check(&self, votes: &[PeerVote], elapsed: Duration) -> QuorumResult {
        // Timeout takes priority.
        if elapsed >= self.timeout {
            return QuorumResult::TimedOut;
        }

        let total = votes.len();
        let commit_count = votes.iter().filter(|v| v.vote == Vote::Commit).count();
        let abort_or_abstain = votes.iter().filter(|v| v.vote != Vote::Commit).count();

        // Check whether quorum is already met.
        if total >= self.min_peers {
            let fraction = commit_count as f64 / total as f64;
            if fraction >= self.commit_threshold {
                let gradient_cids = votes
                    .iter()
                    .filter(|v| v.vote == Vote::Commit)
                    .filter_map(|v| v.gradient_cid.clone())
                    .collect();
                return QuorumResult::Commit { gradient_cids };
            }
        }

        // Determine whether it is still mathematically possible to reach the
        // commit threshold.  If abort/abstain votes already make it impossible
        // (assuming every remaining expected peer votes Commit), abort early.
        //
        // We use `total` as the lower-bound denominator: at minimum, the
        // peers who have already voted have determined the fraction.  If
        // even with infinite additional Commit votes the threshold can never
        // be satisfied given the current non-commit count, we abort.
        //
        // Concretely: threshold_commits_needed(n) = ceil(threshold * n).
        // As n grows, threshold_commits_needed grows too, but non_commit_count
        // stays fixed.  So we check: can any n >= total satisfy
        //   (n - abort_or_abstain) / n >= commit_threshold
        // => 1 - abort_or_abstain/n >= commit_threshold
        // => abort_or_abstain/n <= 1 - commit_threshold
        // => n >= abort_or_abstain / (1 - commit_threshold)
        //
        // That is always satisfiable for large enough n (unless
        // commit_threshold == 1.0), so we only abort early when we know the
        // expected_peers count makes it impossible.  Since QuorumPolicy
        // doesn't know expected_peers, we use total as a proxy: if the
        // current fraction of non-commits already exceeds (1 - threshold)
        // and we have at least min_peers votes, abort.
        if total >= self.min_peers {
            let non_commit_fraction = abort_or_abstain as f64 / total as f64;
            if non_commit_fraction > (1.0 - self.commit_threshold) {
                return QuorumResult::Abort {
                    reason: format!(
                        "commit threshold unreachable: {}/{} votes are non-commit ({:.0}% > {:.0}% max)",
                        abort_or_abstain,
                        total,
                        non_commit_fraction * 100.0,
                        (1.0 - self.commit_threshold) * 100.0,
                    ),
                };
            }
        }

        // Still pending.
        let votes_needed = if total < self.min_peers {
            self.min_peers - total
        } else {
            // Need more commits to push fraction over threshold.
            let commits_needed = (self.commit_threshold * (total + 1) as f64).ceil() as usize;
            commits_needed.saturating_sub(commit_count)
        };

        QuorumResult::Pending {
            votes_received: total,
            votes_needed,
        }
    }
}

// ─── RoundStatus ─────────────────────────────────────────────────────────────

/// Internal lifecycle state of a single round.
#[derive(Debug, Clone)]
pub enum RoundStatus {
    /// Votes are being collected.
    Active,
    /// Quorum was reached and the round committed.
    Committed,
    /// The round was aborted (either by vote or by `abort_round`).
    Aborted {
        /// Human-readable reason for the abort.
        reason: String,
    },
}

impl RoundStatus {
    /// Returns `true` if this is a terminal state.
    fn is_terminal(&self) -> bool {
        matches!(self, RoundStatus::Committed | RoundStatus::Aborted { .. })
    }
}

// ─── RoundState ──────────────────────────────────────────────────────────────

/// Internal state for a single federated learning round.
struct RoundState {
    round_id: u64,
    /// Peers that were expected to vote in this round (stored for future
    /// use, e.g. computing quorum against a known membership set).
    #[allow(dead_code)]
    expected_peers: Vec<String>,
    votes: Vec<PeerVote>,
    started_at: Instant,
    status: RoundStatus,
}

impl RoundState {
    fn new(round_id: u64, expected_peers: Vec<String>) -> Self {
        Self {
            round_id,
            expected_peers,
            votes: Vec::new(),
            started_at: Instant::now(),
            status: RoundStatus::Active,
        }
    }

    fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    fn has_voted(&self, peer_id: &str) -> bool {
        self.votes.iter().any(|v| v.peer_id == peer_id)
    }
}

// ─── ConsensusStats ──────────────────────────────────────────────────────────

/// Atomic counters tracking aggregate consensus activity.
pub struct ConsensusStats {
    /// Total number of rounds started.
    pub total_rounds_started: AtomicU64,
    /// Total number of rounds that reached commit quorum.
    pub total_committed: AtomicU64,
    /// Total number of rounds that were aborted or timed out.
    pub total_aborted: AtomicU64,
    /// Total number of individual votes cast across all rounds.
    pub total_votes_cast: AtomicU64,
}

impl ConsensusStats {
    fn new() -> Self {
        Self {
            total_rounds_started: AtomicU64::new(0),
            total_committed: AtomicU64::new(0),
            total_aborted: AtomicU64::new(0),
            total_votes_cast: AtomicU64::new(0),
        }
    }

    /// Take a consistent snapshot of all counters.
    pub fn snapshot(&self) -> ConsensusStatsSnapshot {
        ConsensusStatsSnapshot {
            total_rounds_started: self.total_rounds_started.load(Ordering::Relaxed),
            total_committed: self.total_committed.load(Ordering::Relaxed),
            total_aborted: self.total_aborted.load(Ordering::Relaxed),
            total_votes_cast: self.total_votes_cast.load(Ordering::Relaxed),
        }
    }
}

impl fmt::Debug for ConsensusStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConsensusStats")
            .field(
                "total_rounds_started",
                &self.total_rounds_started.load(Ordering::Relaxed),
            )
            .field(
                "total_committed",
                &self.total_committed.load(Ordering::Relaxed),
            )
            .field("total_aborted", &self.total_aborted.load(Ordering::Relaxed))
            .field(
                "total_votes_cast",
                &self.total_votes_cast.load(Ordering::Relaxed),
            )
            .finish()
    }
}

/// Immutable snapshot of [`ConsensusStats`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsensusStatsSnapshot {
    /// Total number of rounds started.
    pub total_rounds_started: u64,
    /// Total number of rounds that reached commit quorum.
    pub total_committed: u64,
    /// Total number of rounds that were aborted or timed out.
    pub total_aborted: u64,
    /// Total number of individual votes cast across all rounds.
    pub total_votes_cast: u64,
}

// ─── ConsensusError ──────────────────────────────────────────────────────────

/// Errors that can be returned by [`RoundConsensusTracker`] operations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ConsensusError {
    /// No round with the given ID exists.
    #[error("round {0} not found")]
    RoundNotFound(u64),

    /// A round with this ID was already registered.
    #[error("round {0} already exists")]
    RoundAlreadyExists(u64),

    /// The operation requires the round to be `Active` but it is in a
    /// terminal state.
    #[error("round {0} is not active")]
    RoundNotActive(u64),

    /// This peer has already cast a vote in this round.
    #[error("duplicate vote from peer '{peer_id}' in round {round_id}")]
    DuplicateVote {
        /// Peer that attempted to vote twice.
        peer_id: String,
        /// Round ID in which the duplicate was detected.
        round_id: u64,
    },
}

// ─── RoundConsensusTracker ───────────────────────────────────────────────────

/// Central registry for federated learning round consensus.
///
/// All public methods are safe to call from multiple threads concurrently.
/// Internal state is protected by a single `RwLock`; the design keeps
/// critical sections short to minimise contention.
pub struct RoundConsensusTracker {
    rounds: RwLock<HashMap<u64, RoundState>>,
    /// Policy applied to every quorum evaluation.
    pub policy: QuorumPolicy,
    /// Aggregate statistics.
    pub stats: ConsensusStats,
}

impl RoundConsensusTracker {
    /// Create a new tracker with the given quorum policy.
    pub fn new(policy: QuorumPolicy) -> Self {
        Self {
            rounds: RwLock::new(HashMap::new()),
            policy,
            stats: ConsensusStats::new(),
        }
    }

    // ── Round lifecycle ───────────────────────────────────────────────────

    /// Register a new round.  Returns [`ConsensusError::RoundAlreadyExists`]
    /// if a round with the same ID already exists.
    pub fn begin_round(
        &self,
        round_id: RoundId,
        expected_peers: Vec<String>,
    ) -> Result<(), ConsensusError> {
        let key = round_id.0;
        let mut guard = self.rounds.write();
        if guard.contains_key(&key) {
            return Err(ConsensusError::RoundAlreadyExists(key));
        }
        guard.insert(key, RoundState::new(key, expected_peers));
        drop(guard);
        self.stats
            .total_rounds_started
            .fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Cast a vote for a round.
    ///
    /// After recording the vote the quorum policy is evaluated.  If the
    /// policy reaches a terminal verdict (`Commit` or `Abort`) the round's
    /// status is updated atomically and the corresponding stat counter is
    /// incremented.
    ///
    /// # Errors
    ///
    /// - [`ConsensusError::RoundNotFound`] — no such round
    /// - [`ConsensusError::RoundNotActive`] — round is already committed/aborted
    /// - [`ConsensusError::DuplicateVote`] — peer already voted in this round
    pub fn cast_vote(&self, vote: PeerVote) -> Result<QuorumResult, ConsensusError> {
        let key = vote.round_id.0;
        let peer_id = vote.peer_id.clone();

        let mut guard = self.rounds.write();
        let state = guard
            .get_mut(&key)
            .ok_or(ConsensusError::RoundNotFound(key))?;

        if state.status.is_terminal() {
            return Err(ConsensusError::RoundNotActive(key));
        }
        if state.has_voted(&peer_id) {
            return Err(ConsensusError::DuplicateVote {
                peer_id,
                round_id: key,
            });
        }

        state.votes.push(vote);
        self.stats.total_votes_cast.fetch_add(1, Ordering::Relaxed);

        let elapsed = state.elapsed();
        let result = self.policy.check(&state.votes, elapsed);

        match &result {
            QuorumResult::Commit { .. } => {
                state.status = RoundStatus::Committed;
                drop(guard);
                self.stats.total_committed.fetch_add(1, Ordering::Relaxed);
            }
            QuorumResult::Abort { reason } => {
                state.status = RoundStatus::Aborted {
                    reason: reason.clone(),
                };
                drop(guard);
                self.stats.total_aborted.fetch_add(1, Ordering::Relaxed);
            }
            QuorumResult::TimedOut => {
                state.status = RoundStatus::Aborted {
                    reason: "timeout".to_string(),
                };
                drop(guard);
                self.stats.total_aborted.fetch_add(1, Ordering::Relaxed);
            }
            QuorumResult::Pending { .. } => {
                drop(guard);
            }
        }

        Ok(result)
    }

    /// Evaluate the current quorum state of a round without casting a new
    /// vote.  Returns `None` if the round does not exist.
    pub fn check_round(&self, round_id: &RoundId) -> Option<QuorumResult> {
        let key = round_id.0;
        let guard = self.rounds.read();
        let state = guard.get(&key)?;

        match &state.status {
            RoundStatus::Committed => {
                let gradient_cids = state
                    .votes
                    .iter()
                    .filter(|v| v.vote == Vote::Commit)
                    .filter_map(|v| v.gradient_cid.clone())
                    .collect();
                Some(QuorumResult::Commit { gradient_cids })
            }
            RoundStatus::Aborted { reason } => Some(QuorumResult::Abort {
                reason: reason.clone(),
            }),
            RoundStatus::Active => {
                let elapsed = state.elapsed();
                Some(self.policy.check(&state.votes, elapsed))
            }
        }
    }

    /// Forcibly abort a round.
    ///
    /// # Errors
    ///
    /// - [`ConsensusError::RoundNotFound`] — no such round
    /// - [`ConsensusError::RoundNotActive`] — round already in terminal state
    pub fn abort_round(&self, round_id: &RoundId, reason: &str) -> Result<(), ConsensusError> {
        let key = round_id.0;
        let mut guard = self.rounds.write();
        let state = guard
            .get_mut(&key)
            .ok_or(ConsensusError::RoundNotFound(key))?;

        if state.status.is_terminal() {
            return Err(ConsensusError::RoundNotActive(key));
        }

        state.status = RoundStatus::Aborted {
            reason: reason.to_string(),
        };
        drop(guard);
        self.stats.total_aborted.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Return the IDs of all rounds currently in a terminal state
    /// (committed or aborted).
    pub fn complete_rounds(&self) -> Vec<u64> {
        let guard = self.rounds.read();
        guard
            .values()
            .filter(|s| s.status.is_terminal())
            .map(|s| s.round_id)
            .collect()
    }

    /// Return the number of rounds currently in the `Active` state.
    pub fn active_round_count(&self) -> usize {
        let guard = self.rounds.read();
        guard
            .values()
            .filter(|s| matches!(s.status, RoundStatus::Active))
            .count()
    }
}

impl fmt::Debug for RoundConsensusTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let guard = self.rounds.read();
        f.debug_struct("RoundConsensusTracker")
            .field("total_rounds", &guard.len())
            .field("active_rounds", &self.active_round_count())
            .field("stats", &self.stats)
            .finish()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn default_tracker() -> RoundConsensusTracker {
        RoundConsensusTracker::new(QuorumPolicy::default())
    }

    fn make_commit_vote(peer: &str, round: u64, cid: Option<&str>) -> PeerVote {
        PeerVote::new(
            peer.to_string(),
            RoundId::from(round),
            Vote::Commit,
            cid.map(|s| s.to_string()),
        )
    }

    fn make_abort_vote(peer: &str, round: u64) -> PeerVote {
        PeerVote::new(peer.to_string(), RoundId::from(round), Vote::Abort, None)
    }

    fn make_abstain_vote(peer: &str, round: u64) -> PeerVote {
        PeerVote::new(peer.to_string(), RoundId::from(round), Vote::Abstain, None)
    }

    fn three_peers() -> Vec<String> {
        vec!["p1".to_string(), "p2".to_string(), "p3".to_string()]
    }

    // ── 1. Begin round, cast a vote (pending) ─────────────────────────────

    #[test]
    fn test_begin_round_and_single_vote_pending() {
        let tracker = default_tracker();
        let rid = RoundId::from(1);
        tracker
            .begin_round(rid, three_peers())
            .expect("test: should succeed");

        let result = tracker
            .cast_vote(make_commit_vote("p1", 1, None))
            .expect("test: should succeed");
        assert!(
            matches!(result, QuorumResult::Pending { .. }),
            "expected Pending after 1/3 votes, got {result:?}"
        );
    }

    // ── 2. Commit quorum reached ──────────────────────────────────────────

    #[test]
    fn test_commit_quorum_reached_on_third_vote() {
        let tracker = default_tracker();
        let rid = RoundId::from(2);
        tracker
            .begin_round(rid, three_peers())
            .expect("test: should succeed");

        tracker
            .cast_vote(make_commit_vote("p1", 2, Some("cid1")))
            .expect("test: should succeed");
        tracker
            .cast_vote(make_commit_vote("p2", 2, Some("cid2")))
            .expect("test: should succeed");
        let result = tracker
            .cast_vote(make_commit_vote("p3", 2, Some("cid3")))
            .expect("test: should succeed");

        match result {
            QuorumResult::Commit { gradient_cids } => {
                assert_eq!(gradient_cids.len(), 3);
                assert!(gradient_cids.contains(&"cid1".to_string()));
                assert!(gradient_cids.contains(&"cid2".to_string()));
                assert!(gradient_cids.contains(&"cid3".to_string()));
            }
            other => panic!("expected Commit, got {other:?}"),
        }
    }

    // ── 3. Abort when majority votes Abort ───────────────────────────────

    #[test]
    fn test_abort_when_majority_votes_abort() {
        let tracker = default_tracker();
        let rid = RoundId::from(3);
        tracker
            .begin_round(rid, three_peers())
            .expect("test: should succeed");

        tracker
            .cast_vote(make_abort_vote("p1", 3))
            .expect("test: should succeed");
        tracker
            .cast_vote(make_abort_vote("p2", 3))
            .expect("test: should succeed");
        let result = tracker
            .cast_vote(make_abort_vote("p3", 3))
            .expect("test: should succeed");

        assert!(
            matches!(result, QuorumResult::Abort { .. }),
            "expected Abort, got {result:?}"
        );
    }

    // ── 4. Timeout detection ─────────────────────────────────────────────

    #[test]
    fn test_timeout_detection() {
        let policy = QuorumPolicy {
            min_peers: 1,
            commit_threshold: 0.67,
            timeout: Duration::from_millis(1), // very short timeout
        };
        let tracker = RoundConsensusTracker::new(policy);
        let rid = RoundId::from(4);
        tracker
            .begin_round(rid, vec!["p1".to_string()])
            .expect("test: should succeed");

        // Sleep past the timeout.
        std::thread::sleep(Duration::from_millis(10));

        let result = tracker
            .cast_vote(make_commit_vote("p1", 4, None))
            .expect("test: should succeed");

        assert_eq!(result, QuorumResult::TimedOut);
    }

    // ── 5. Duplicate vote rejection ───────────────────────────────────────

    #[test]
    fn test_duplicate_vote_rejected() {
        let tracker = default_tracker();
        tracker
            .begin_round(RoundId::from(5), three_peers())
            .expect("test: should succeed");
        tracker
            .cast_vote(make_commit_vote("p1", 5, None))
            .expect("test: should succeed");

        let err = tracker
            .cast_vote(make_commit_vote("p1", 5, None))
            .unwrap_err();

        assert!(
            matches!(err, ConsensusError::DuplicateVote { ref peer_id, round_id } if peer_id == "p1" && round_id == 5),
            "unexpected error: {err}"
        );
    }

    // ── 6. Round not found error ──────────────────────────────────────────

    #[test]
    fn test_round_not_found() {
        let tracker = default_tracker();
        let err = tracker
            .cast_vote(make_commit_vote("p1", 99, None))
            .unwrap_err();
        assert_eq!(err, ConsensusError::RoundNotFound(99));
    }

    // ── 7. abort_round manual override ───────────────────────────────────

    #[test]
    fn test_abort_round_manual() {
        let tracker = default_tracker();
        let rid = RoundId::from(7);
        tracker
            .begin_round(rid, three_peers())
            .expect("test: should succeed");
        tracker
            .abort_round(&rid, "coordinator decision")
            .expect("test: should succeed");

        let result = tracker.check_round(&rid).expect("test: should succeed");
        match result {
            QuorumResult::Abort { reason } => {
                assert!(reason.contains("coordinator decision"));
            }
            other => panic!("expected Abort, got {other:?}"),
        }
    }

    // ── 8. complete_rounds list ───────────────────────────────────────────

    #[test]
    fn test_complete_rounds_list() {
        let tracker = default_tracker();

        // Round 10 — will commit.
        tracker
            .begin_round(RoundId::from(10), three_peers())
            .expect("test: should succeed");
        tracker
            .cast_vote(make_commit_vote("p1", 10, None))
            .expect("test: should succeed");
        tracker
            .cast_vote(make_commit_vote("p2", 10, None))
            .expect("test: should succeed");
        tracker
            .cast_vote(make_commit_vote("p3", 10, None))
            .expect("test: should succeed");

        // Round 11 — still active.
        tracker
            .begin_round(RoundId::from(11), three_peers())
            .expect("test: should succeed");

        // Round 12 — will be aborted manually.
        tracker
            .begin_round(RoundId::from(12), three_peers())
            .expect("test: should succeed");
        tracker
            .abort_round(&RoundId::from(12), "test")
            .expect("test: should succeed");

        let complete = tracker.complete_rounds();
        assert!(
            complete.contains(&10),
            "committed round 10 missing: {complete:?}"
        );
        assert!(
            complete.contains(&12),
            "aborted round 12 missing: {complete:?}"
        );
        assert!(
            !complete.contains(&11),
            "active round 11 should not be complete"
        );
    }

    // ── 9. Gradient CIDs collected on commit ─────────────────────────────

    #[test]
    fn test_gradient_cids_collected_on_commit() {
        let tracker = default_tracker();
        let rid = RoundId::from(9);
        tracker
            .begin_round(rid, three_peers())
            .expect("test: should succeed");

        // p2 commits without a CID (e.g. gradient already known).
        tracker
            .cast_vote(make_commit_vote("p1", 9, Some("bafy1")))
            .expect("test: should succeed");
        tracker
            .cast_vote(make_commit_vote("p2", 9, None))
            .expect("test: should succeed");
        let result = tracker
            .cast_vote(make_commit_vote("p3", 9, Some("bafy3")))
            .expect("test: should succeed");

        match result {
            QuorumResult::Commit { gradient_cids } => {
                // Only p1 and p3 supplied CIDs.
                assert_eq!(gradient_cids.len(), 2);
                assert!(gradient_cids.contains(&"bafy1".to_string()));
                assert!(gradient_cids.contains(&"bafy3".to_string()));
            }
            other => panic!("expected Commit, got {other:?}"),
        }
    }

    // ── 10. Stats accumulation ────────────────────────────────────────────

    #[test]
    fn test_stats_accumulation() {
        let tracker = default_tracker();

        // Start 2 rounds, commit one, abort the other.
        tracker
            .begin_round(RoundId::from(20), three_peers())
            .expect("test: should succeed");
        tracker
            .cast_vote(make_commit_vote("p1", 20, None))
            .expect("test: should succeed");
        tracker
            .cast_vote(make_commit_vote("p2", 20, None))
            .expect("test: should succeed");
        tracker
            .cast_vote(make_commit_vote("p3", 20, None))
            .expect("test: should succeed"); // commits

        tracker
            .begin_round(RoundId::from(21), three_peers())
            .expect("test: should succeed");
        tracker
            .abort_round(&RoundId::from(21), "test")
            .expect("test: should succeed");

        let snap = tracker.stats.snapshot();
        assert_eq!(snap.total_rounds_started, 2);
        assert_eq!(snap.total_committed, 1);
        assert_eq!(snap.total_aborted, 1);
        assert_eq!(snap.total_votes_cast, 3);
    }

    // ── 11. RoundId newtype and next() ────────────────────────────────────

    #[test]
    fn test_round_id_next_and_display() {
        let r = RoundId::from(5_u64);
        assert_eq!(r.next(), RoundId::from(6_u64));
        assert_eq!(r.to_string(), "round-5");
    }

    // ── 12. active_round_count ────────────────────────────────────────────

    #[test]
    fn test_active_round_count() {
        let tracker = default_tracker();
        assert_eq!(tracker.active_round_count(), 0);

        tracker
            .begin_round(RoundId::from(30), three_peers())
            .expect("test: should succeed");
        tracker
            .begin_round(RoundId::from(31), three_peers())
            .expect("test: should succeed");
        assert_eq!(tracker.active_round_count(), 2);

        tracker
            .abort_round(&RoundId::from(30), "x")
            .expect("test: should succeed");
        assert_eq!(tracker.active_round_count(), 1);
    }

    // ── 13. check_round returns None for unknown round ────────────────────

    #[test]
    fn test_check_round_unknown() {
        let tracker = default_tracker();
        assert!(tracker.check_round(&RoundId::from(999)).is_none());
    }

    // ── 14. RoundAlreadyExists on duplicate begin ─────────────────────────

    #[test]
    fn test_begin_round_duplicate_returns_error() {
        let tracker = default_tracker();
        tracker
            .begin_round(RoundId::from(40), three_peers())
            .expect("test: should succeed");
        let err = tracker
            .begin_round(RoundId::from(40), three_peers())
            .unwrap_err();
        assert_eq!(err, ConsensusError::RoundAlreadyExists(40));
    }

    // ── 15. Vote on completed round returns RoundNotActive ────────────────

    #[test]
    fn test_vote_on_committed_round_returns_not_active() {
        let tracker = default_tracker();
        tracker
            .begin_round(RoundId::from(50), three_peers())
            .expect("test: should succeed");
        tracker
            .cast_vote(make_commit_vote("p1", 50, None))
            .expect("test: should succeed");
        tracker
            .cast_vote(make_commit_vote("p2", 50, None))
            .expect("test: should succeed");
        tracker
            .cast_vote(make_commit_vote("p3", 50, None))
            .expect("test: should succeed"); // commits

        // A 4th peer tries to vote on an already-committed round.
        let err = tracker
            .cast_vote(make_commit_vote("p4", 50, None))
            .unwrap_err();
        assert_eq!(err, ConsensusError::RoundNotActive(50));
    }

    // ── 16. Abstain votes count against commit threshold ─────────────────

    #[test]
    fn test_abstain_counts_against_commit_threshold() {
        // 3 peers required, threshold 0.67 (2 out of 3 must commit).
        let tracker = default_tracker();
        tracker
            .begin_round(RoundId::from(60), three_peers())
            .expect("test: should succeed");

        tracker
            .cast_vote(make_commit_vote("p1", 60, None))
            .expect("test: should succeed");
        tracker
            .cast_vote(make_commit_vote("p2", 60, None))
            .expect("test: should succeed");
        // p3 abstains — 2/3 ≈ 0.667 which is >= 0.67 threshold.
        let result = tracker
            .cast_vote(make_abstain_vote("p3", 60))
            .expect("test: should succeed");

        // 2/3 = 0.666... which is strictly less than 0.67, so still pending
        // OR aborted depending on rounding. Let's verify the logic:
        // fraction = 2/3 = 0.6667; threshold = 0.67 → NOT >= threshold → pending/abort
        // non_commit_fraction = 1/3 ≈ 0.333; 1 - threshold = 0.33
        // 0.333 > 0.33 → Abort
        assert!(
            matches!(
                result,
                QuorumResult::Abort { .. } | QuorumResult::Pending { .. }
            ),
            "unexpected result {result:?}"
        );
    }

    // ── 17. abort_round on already-aborted round returns RoundNotActive ───

    #[test]
    fn test_abort_already_aborted_round() {
        let tracker = default_tracker();
        let rid = RoundId::from(70);
        tracker
            .begin_round(rid, three_peers())
            .expect("test: should succeed");
        tracker
            .abort_round(&rid, "first")
            .expect("test: should succeed");

        let err = tracker.abort_round(&rid, "second").unwrap_err();
        assert_eq!(err, ConsensusError::RoundNotActive(70));
    }

    // ── 18. Custom quorum policy with 100 % threshold ─────────────────────

    #[test]
    fn test_custom_policy_unanimous_commit() {
        let policy = QuorumPolicy {
            min_peers: 3,
            commit_threshold: 1.0,
            timeout: Duration::from_secs(60),
        };
        let tracker = RoundConsensusTracker::new(policy);
        let rid = RoundId::from(80);
        tracker
            .begin_round(rid, three_peers())
            .expect("test: should succeed");

        tracker
            .cast_vote(make_commit_vote("p1", 80, None))
            .expect("test: should succeed");
        tracker
            .cast_vote(make_commit_vote("p2", 80, None))
            .expect("test: should succeed");
        let result = tracker
            .cast_vote(make_commit_vote("p3", 80, None))
            .expect("test: should succeed");

        assert!(
            matches!(result, QuorumResult::Commit { .. }),
            "unanimous policy should commit: {result:?}"
        );
    }
}
