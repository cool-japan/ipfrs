//! Peer Reputation Manager — long-term behavioral scoring for trust-weighted routing.
//!
//! Tracks per-peer reputation scores derived from a stream of [`PrReputationEvent`]s.
//! Scores are clamped to `[0.0, 1.0]`, decay toward the neutral value of `0.5` each tick,
//! and drive tier classification used by connection prioritisation and routing.
//!
//! # Quick start
//!
//! ```rust
//! use ipfrs_network::peer_reputation::{
//!     PeerReputationManager, PrReputationConfig, PrReputationEvent,
//! };
//!
//! let cfg = PrReputationConfig::default();
//! let mut mgr = PeerReputationManager::new(cfg);
//!
//! mgr.record_event("peer-A", PrReputationEvent::BlockDelivered { bytes: 4096 });
//! mgr.record_event("peer-A", PrReputationEvent::GoodProof);
//!
//! let score = mgr.score("peer-A").expect("peer exists");
//! assert!(score.is_trusted());
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ReputationEvent
// ---------------------------------------------------------------------------

/// A discrete behavioural observation for a remote peer.
#[derive(Clone, Debug, PartialEq)]
pub enum PrReputationEvent {
    /// A block was successfully transferred from the peer.
    BlockDelivered {
        /// Number of bytes in the delivered block.
        bytes: u64,
    },
    /// A block request to the peer timed out.
    BlockTimeout,
    /// The peer sent an invalid message or otherwise violated the protocol.
    ProtocolViolation,
    /// The peer supplied a cryptographically valid proof.
    GoodProof,
    /// The peer supplied an invalid proof (major penalty).
    BadProof,
    /// The peer disconnected unexpectedly.
    ConnectionDropped,
}

// ---------------------------------------------------------------------------
// ReputationScore
// ---------------------------------------------------------------------------

/// Long-term reputation score for a single peer.
#[derive(Clone, Debug)]
pub struct PrReputationScore {
    /// Current score, always clamped to `[0.0, 1.0]`.
    pub score: f64,
    /// Total number of events recorded for this peer.
    pub total_events: u64,
    /// Number of violations (ProtocolViolation + BadProof) observed.
    pub violations: u32,
}

impl PrReputationScore {
    fn new() -> Self {
        Self {
            score: 0.5,
            total_events: 0,
            violations: 0,
        }
    }

    /// Returns `true` when the peer's score is at or above the trust threshold (≥ 0.7).
    pub fn is_trusted(&self) -> bool {
        self.score >= 0.7
    }

    /// Returns `true` when the peer's score is below the ban threshold (< 0.1)
    /// or the peer has accumulated 10 or more violations.
    pub fn is_banned(&self) -> bool {
        self.score < 0.1 || self.violations >= 10
    }

    /// Human-readable tier label based on the current score.
    ///
    /// | Range          | Label      |
    /// |---------------|------------|
    /// | `[0.0, 0.1)`  | `"banned"` |
    /// | `[0.1, 0.4)`  | `"poor"`   |
    /// | `[0.4, 0.7)`  | `"neutral"`|
    /// | `[0.7, 1.0]`  | `"trusted"`|
    pub fn tier(&self) -> &'static str {
        if self.score < 0.1 {
            "banned"
        } else if self.score < 0.4 {
            "poor"
        } else if self.score < 0.7 {
            "neutral"
        } else {
            "trusted"
        }
    }
}

// ---------------------------------------------------------------------------
// ReputationConfig
// ---------------------------------------------------------------------------

/// Tuning parameters for score deltas and temporal decay.
#[derive(Clone, Debug)]
pub struct PrReputationConfig {
    /// Score delta for a successful block delivery (positive).
    pub block_delivered_delta: f64,
    /// Score delta when a block request times out (negative).
    pub block_timeout_delta: f64,
    /// Score delta for a protocol violation (negative).
    pub protocol_violation_delta: f64,
    /// Score delta for a valid proof (positive).
    pub good_proof_delta: f64,
    /// Score delta for an invalid proof (large negative).
    pub bad_proof_delta: f64,
    /// Score delta for an unexpected disconnect (mild negative).
    pub connection_dropped_delta: f64,
    /// Per-tick decay magnitude toward the neutral score of `0.5`.
    pub decay_rate: f64,
}

impl Default for PrReputationConfig {
    fn default() -> Self {
        Self {
            block_delivered_delta: 0.02,
            block_timeout_delta: -0.05,
            protocol_violation_delta: -0.15,
            good_proof_delta: 0.05,
            bad_proof_delta: -0.25,
            connection_dropped_delta: -0.03,
            decay_rate: 0.001,
        }
    }
}

impl PrReputationConfig {
    fn delta_for(&self, event: &PrReputationEvent) -> f64 {
        match event {
            PrReputationEvent::BlockDelivered { .. } => self.block_delivered_delta,
            PrReputationEvent::BlockTimeout => self.block_timeout_delta,
            PrReputationEvent::ProtocolViolation => self.protocol_violation_delta,
            PrReputationEvent::GoodProof => self.good_proof_delta,
            PrReputationEvent::BadProof => self.bad_proof_delta,
            PrReputationEvent::ConnectionDropped => self.connection_dropped_delta,
        }
    }

    fn is_violation(event: &PrReputationEvent) -> bool {
        matches!(
            event,
            PrReputationEvent::ProtocolViolation | PrReputationEvent::BadProof
        )
    }
}

// ---------------------------------------------------------------------------
// ReputationStats
// ---------------------------------------------------------------------------

/// Aggregate statistics across all tracked peers.
#[derive(Clone, Debug)]
pub struct PrReputationStats {
    /// Total number of peers currently tracked.
    pub total_peers: usize,
    /// Number of peers classified as trusted.
    pub trusted_count: usize,
    /// Number of peers classified as banned.
    pub banned_count: usize,
    /// Mean score across all tracked peers; `0.0` when no peers are tracked.
    pub avg_score: f64,
}

// ---------------------------------------------------------------------------
// PeerReputationManager
// ---------------------------------------------------------------------------

/// Manages long-term reputation scores for a set of remote peers.
///
/// Scores are keyed by an opaque peer-ID string and are automatically
/// created at the neutral value of `0.5` on the first event.
pub struct PeerReputationManager {
    scores: HashMap<String, PrReputationScore>,
    config: PrReputationConfig,
}

impl PeerReputationManager {
    /// Creates a new manager with the supplied configuration.
    pub fn new(config: PrReputationConfig) -> Self {
        Self {
            scores: HashMap::new(),
            config,
        }
    }

    /// Records a behavioural event for `peer_id`.
    ///
    /// If the peer is not yet tracked it is inserted with the neutral score of
    /// `0.5` before the event is applied.  The resulting score is always clamped
    /// to `[0.0, 1.0]`.
    pub fn record_event(&mut self, peer_id: &str, event: PrReputationEvent) {
        let entry = self
            .scores
            .entry(peer_id.to_owned())
            .or_insert_with(PrReputationScore::new);

        let delta = self.config.delta_for(&event);
        entry.score = (entry.score + delta).clamp(0.0, 1.0);

        if PrReputationConfig::is_violation(&event) {
            entry.violations = entry.violations.saturating_add(1);
        }

        entry.total_events = entry.total_events.saturating_add(1);
    }

    /// Nudges every tracked peer's score one step toward `0.5` by `decay_rate`.
    pub fn decay_all(&mut self) {
        let rate = self.config.decay_rate;
        for entry in self.scores.values_mut() {
            if entry.score > 0.5 {
                entry.score = (entry.score - rate).max(0.5);
            } else if entry.score < 0.5 {
                entry.score = (entry.score + rate).min(0.5);
            }
        }
    }

    /// Returns the current score for `peer_id`, or `None` if unknown.
    pub fn score(&self, peer_id: &str) -> Option<&PrReputationScore> {
        self.scores.get(peer_id)
    }

    /// Returns the IDs of all trusted peers, sorted by score descending.
    pub fn trusted_peers(&self) -> Vec<&str> {
        let mut trusted: Vec<(&str, f64)> = self
            .scores
            .iter()
            .filter(|(_, s)| s.is_trusted())
            .map(|(id, s)| (id.as_str(), s.score))
            .collect();

        trusted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        trusted.into_iter().map(|(id, _)| id).collect()
    }

    /// Returns the IDs of all banned peers (order is unspecified).
    pub fn banned_peers(&self) -> Vec<&str> {
        self.scores
            .iter()
            .filter(|(_, s)| s.is_banned())
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// Removes a peer from the tracker.
    ///
    /// Returns `true` if the peer was present, `false` otherwise.
    pub fn remove_peer(&mut self, peer_id: &str) -> bool {
        self.scores.remove(peer_id).is_some()
    }

    /// Computes aggregate statistics over all currently tracked peers.
    pub fn stats(&self) -> PrReputationStats {
        let total_peers = self.scores.len();
        if total_peers == 0 {
            return PrReputationStats {
                total_peers: 0,
                trusted_count: 0,
                banned_count: 0,
                avg_score: 0.0,
            };
        }

        let mut trusted_count = 0usize;
        let mut banned_count = 0usize;
        let mut score_sum = 0.0f64;

        for s in self.scores.values() {
            if s.is_trusted() {
                trusted_count += 1;
            }
            if s.is_banned() {
                banned_count += 1;
            }
            score_sum += s.score;
        }

        PrReputationStats {
            total_peers,
            trusted_count,
            banned_count,
            avg_score: score_sum / total_peers as f64,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_mgr() -> PeerReputationManager {
        PeerReputationManager::new(PrReputationConfig::default())
    }

    // 1. New manager starts empty
    #[test]
    fn test_new_manager_is_empty() {
        let mgr = default_mgr();
        assert!(mgr.scores.is_empty());
        assert_eq!(mgr.stats().total_peers, 0);
    }

    // 2. Auto-create peer on first event at 0.5
    #[test]
    fn test_auto_create_peer_at_neutral_score() {
        let mut mgr = default_mgr();
        mgr.record_event("peer-X", PrReputationEvent::BlockDelivered { bytes: 100 });
        let s = mgr.score("peer-X").expect("peer must exist");
        // Started at 0.5, BlockDelivered adds 0.02 → 0.52
        assert!((s.score - 0.52).abs() < 1e-9);
    }

    // 3. BlockDelivered increases score
    #[test]
    fn test_block_delivered_increases_score() {
        let mut mgr = default_mgr();
        mgr.record_event("p", PrReputationEvent::BlockDelivered { bytes: 512 });
        let s = mgr
            .score("p")
            .expect("test: peer should exist after recording event");
        assert!(s.score > 0.5);
    }

    // 4. BlockTimeout decreases score
    #[test]
    fn test_block_timeout_decreases_score() {
        let mut mgr = default_mgr();
        mgr.record_event("p", PrReputationEvent::BlockTimeout);
        let s = mgr
            .score("p")
            .expect("test: peer should exist after recording event");
        assert!(s.score < 0.5);
    }

    // 5. ProtocolViolation increments violations
    #[test]
    fn test_protocol_violation_increments_violations() {
        let mut mgr = default_mgr();
        mgr.record_event("p", PrReputationEvent::ProtocolViolation);
        let s = mgr
            .score("p")
            .expect("test: peer 'p' should exist after recording ProtocolViolation event");
        assert_eq!(s.violations, 1);
    }

    // 6. BadProof increments violations
    #[test]
    fn test_bad_proof_increments_violations() {
        let mut mgr = default_mgr();
        mgr.record_event("p", PrReputationEvent::BadProof);
        let s = mgr
            .score("p")
            .expect("test: peer 'p' should exist after recording BadProof event");
        assert_eq!(s.violations, 1);
    }

    // 7. Score is clamped to [0.0, 1.0] — upper bound
    #[test]
    fn test_score_clamped_at_upper_bound() {
        let mut mgr = default_mgr();
        for _ in 0..100 {
            mgr.record_event("p", PrReputationEvent::BlockDelivered { bytes: 1 });
        }
        let s = mgr
            .score("p")
            .expect("test: peer 'p' should exist after recording BlockDelivered events");
        assert!(s.score <= 1.0);
    }

    // 8. Score is clamped to [0.0, 1.0] — lower bound
    #[test]
    fn test_score_clamped_at_lower_bound() {
        let mut mgr = default_mgr();
        for _ in 0..100 {
            mgr.record_event("p", PrReputationEvent::BadProof);
        }
        let s = mgr
            .score("p")
            .expect("test: peer 'p' should exist after recording BadProof events");
        assert!(s.score >= 0.0);
    }

    // 9. is_trusted returns true when score >= 0.7
    #[test]
    fn test_is_trusted_at_threshold() {
        let s = PrReputationScore {
            score: 0.7,
            total_events: 1,
            violations: 0,
        };
        assert!(s.is_trusted());
    }

    // 10. is_trusted returns false below threshold
    #[test]
    fn test_is_trusted_below_threshold() {
        let s = PrReputationScore {
            score: 0.699,
            total_events: 1,
            violations: 0,
        };
        assert!(!s.is_trusted());
    }

    // 11. is_banned when score < 0.1
    #[test]
    fn test_is_banned_low_score() {
        let s = PrReputationScore {
            score: 0.05,
            total_events: 5,
            violations: 0,
        };
        assert!(s.is_banned());
    }

    // 12. is_banned when violations >= 10
    #[test]
    fn test_is_banned_high_violations() {
        let s = PrReputationScore {
            score: 0.5,
            total_events: 10,
            violations: 10,
        };
        assert!(s.is_banned());
    }

    // 13. tier() returns correct labels
    #[test]
    fn test_tier_labels() {
        let cases = [
            (0.05, "banned"),
            (0.1, "poor"),
            (0.39, "poor"),
            (0.4, "neutral"),
            (0.69, "neutral"),
            (0.7, "trusted"),
            (1.0, "trusted"),
        ];
        for (score, expected) in cases {
            let s = PrReputationScore {
                score,
                total_events: 0,
                violations: 0,
            };
            assert_eq!(s.tier(), expected, "score={score}");
        }
    }

    // 14. decay_all moves high score toward 0.5
    #[test]
    fn test_decay_all_moves_high_score_toward_neutral() {
        let mut mgr = default_mgr();
        // Manually set a high score
        mgr.scores.insert(
            "p".to_owned(),
            PrReputationScore {
                score: 0.9,
                total_events: 0,
                violations: 0,
            },
        );
        mgr.decay_all();
        let s = mgr
            .score("p")
            .expect("test: peer 'p' should exist after decay_all");
        assert!(s.score < 0.9, "score should have decreased toward 0.5");
        assert!(s.score >= 0.5, "score should not go below 0.5 in one tick");
    }

    // 15. decay_all moves low score toward 0.5
    #[test]
    fn test_decay_all_moves_low_score_toward_neutral() {
        let mut mgr = default_mgr();
        mgr.scores.insert(
            "p".to_owned(),
            PrReputationScore {
                score: 0.2,
                total_events: 0,
                violations: 0,
            },
        );
        mgr.decay_all();
        let s = mgr
            .score("p")
            .expect("test: peer 'p' should exist after decay_all");
        assert!(s.score > 0.2, "score should have increased toward 0.5");
        assert!(s.score <= 0.5, "score should not exceed 0.5 in one tick");
    }

    // 16. trusted_peers returns peers sorted by score descending
    #[test]
    fn test_trusted_peers_sorted_desc() {
        let mut mgr = default_mgr();
        for (id, score) in [("a", 0.8_f64), ("b", 0.95), ("c", 0.72)] {
            mgr.scores.insert(
                id.to_owned(),
                PrReputationScore {
                    score,
                    total_events: 1,
                    violations: 0,
                },
            );
        }
        let trusted = mgr.trusted_peers();
        assert_eq!(trusted.len(), 3);
        // b (0.95) > a (0.80) > c (0.72)
        assert_eq!(trusted[0], "b");
        assert_eq!(trusted[1], "a");
        assert_eq!(trusted[2], "c");
    }

    // 17. banned_peers returns the correct set
    #[test]
    fn test_banned_peers_correct() {
        let mut mgr = default_mgr();
        mgr.scores.insert(
            "good".to_owned(),
            PrReputationScore {
                score: 0.8,
                total_events: 1,
                violations: 0,
            },
        );
        mgr.scores.insert(
            "bad".to_owned(),
            PrReputationScore {
                score: 0.05,
                total_events: 5,
                violations: 0,
            },
        );
        let banned = mgr.banned_peers();
        assert_eq!(banned.len(), 1);
        assert_eq!(banned[0], "bad");
    }

    // 18. remove_peer returns true when present, false when absent
    #[test]
    fn test_remove_peer_returns_correct_bool() {
        let mut mgr = default_mgr();
        mgr.record_event("p", PrReputationEvent::GoodProof);
        assert!(mgr.remove_peer("p"));
        assert!(!mgr.remove_peer("p")); // already gone
        assert!(!mgr.remove_peer("never-existed"));
    }

    // 19. stats counts are correct
    #[test]
    fn test_stats_counts_correct() {
        let mut mgr = default_mgr();
        mgr.scores.insert(
            "trusted".to_owned(),
            PrReputationScore {
                score: 0.8,
                total_events: 1,
                violations: 0,
            },
        );
        mgr.scores.insert(
            "neutral".to_owned(),
            PrReputationScore {
                score: 0.5,
                total_events: 1,
                violations: 0,
            },
        );
        mgr.scores.insert(
            "banned".to_owned(),
            PrReputationScore {
                score: 0.05,
                total_events: 1,
                violations: 0,
            },
        );
        let stats = mgr.stats();
        assert_eq!(stats.total_peers, 3);
        assert_eq!(stats.trusted_count, 1);
        assert_eq!(stats.banned_count, 1);
    }

    // 20. avg_score is computed correctly
    #[test]
    fn test_avg_score_computed_correctly() {
        let mut mgr = default_mgr();
        for (id, score) in [("a", 0.6_f64), ("b", 0.8)] {
            mgr.scores.insert(
                id.to_owned(),
                PrReputationScore {
                    score,
                    total_events: 0,
                    violations: 0,
                },
            );
        }
        let stats = mgr.stats();
        let expected = (0.6 + 0.8) / 2.0;
        assert!((stats.avg_score - expected).abs() < 1e-9);
    }

    // 21. avg_score is 0.0 when no peers
    #[test]
    fn test_avg_score_zero_when_empty() {
        let mgr = default_mgr();
        assert_eq!(mgr.stats().avg_score, 0.0);
    }

    // 22. total_events increments per event
    #[test]
    fn test_total_events_increments() {
        let mut mgr = default_mgr();
        mgr.record_event("p", PrReputationEvent::GoodProof);
        mgr.record_event("p", PrReputationEvent::GoodProof);
        mgr.record_event("p", PrReputationEvent::BlockTimeout);
        let s = mgr
            .score("p")
            .expect("test: peer 'p' should exist after recording 3 events");
        assert_eq!(s.total_events, 3);
    }

    // 23. GoodProof does not increment violations
    #[test]
    fn test_good_proof_no_violation() {
        let mut mgr = default_mgr();
        mgr.record_event("p", PrReputationEvent::GoodProof);
        let s = mgr
            .score("p")
            .expect("test: peer 'p' should exist after GoodProof event");
        assert_eq!(s.violations, 0);
    }

    // 24. ConnectionDropped does not increment violations
    #[test]
    fn test_connection_dropped_no_violation() {
        let mut mgr = default_mgr();
        mgr.record_event("p", PrReputationEvent::ConnectionDropped);
        let s = mgr
            .score("p")
            .expect("test: peer 'p' should exist after ConnectionDropped event");
        assert_eq!(s.violations, 0);
    }

    // 25. Decay does not push neutral scores away from 0.5
    #[test]
    fn test_decay_leaves_neutral_unchanged() {
        let mut mgr = default_mgr();
        mgr.scores.insert(
            "p".to_owned(),
            PrReputationScore {
                score: 0.5,
                total_events: 0,
                violations: 0,
            },
        );
        mgr.decay_all();
        let s = mgr
            .score("p")
            .expect("test: peer 'p' should exist after decay_all on neutral score");
        assert_eq!(s.score, 0.5);
    }
}
