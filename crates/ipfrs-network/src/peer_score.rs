//! Peer Score Tracker for GossipSub mesh management
//!
//! Tracks composite peer quality scores used by GossipSub mesh management.
//! Combines message delivery rate, topic mesh contribution, and
//! application-level quality signals.
//!
//! # Examples
//!
//! ```rust
//! use ipfrs_network::peer_score::{PeerScoreTracker, ScoreParameter, BehaviourPenalty};
//!
//! let mut tracker = PeerScoreTracker::new(ScoreParameter::default());
//! tracker.get_or_create("peer-1");
//! let score = tracker.composite_score("peer-1");
//! println!("Peer score: {score}");
//! ```

use std::collections::HashMap;
use std::time::Instant;

/// Parameters controlling how peer scores are weighted and decayed.
#[derive(Debug, Clone)]
pub struct ScoreParameter {
    /// Weight for topic-specific score component.
    pub topic_weight: f64,
    /// Weight for delivery rate score component.
    pub delivery_weight: f64,
    /// Weight for application behaviour component.
    pub behaviour_weight: f64,
    /// How often (in seconds) scores are decayed.
    pub decay_interval_secs: u64,
    /// Score below which a peer is banned.
    pub score_threshold_ban: f64,
    /// Score below which a peer is greylisted (but above ban).
    pub score_threshold_graylist: f64,
}

impl Default for ScoreParameter {
    fn default() -> Self {
        Self {
            topic_weight: 0.5,
            delivery_weight: 0.3,
            behaviour_weight: 0.2,
            decay_interval_secs: 10,
            score_threshold_ban: -100.0,
            score_threshold_graylist: -10.0,
        }
    }
}

/// Per-peer, per-topic scoring data.
#[derive(Debug, Clone, Default)]
pub struct TopicScore {
    /// Increases when the peer is first to deliver a message on this topic.
    pub first_message_deliveries: f64,
    /// Deliveries counted while the peer is in the mesh for this topic.
    pub mesh_message_deliveries: f64,
    /// Penalty accumulated when a peer was in the mesh but had poor delivery.
    pub mesh_failure_penalty: f64,
    /// Counts of invalid messages; penalises the composite score heavily.
    pub invalid_message_deliveries: f64,
    /// Whether this peer is currently in the mesh for this topic.
    pub in_mesh: bool,
    /// Seconds the peer has been in the mesh for this topic.
    pub mesh_time_secs: f64,
}

impl TopicScore {
    /// Compute the weighted topic score.
    ///
    /// Formula:
    /// - P1: first-message-delivery bonus (capped implicitly by values)
    /// - P2: mesh-message-delivery bonus
    /// - P3: mesh-failure penalty (negative)
    /// - P4: invalid-message penalty (negative, weight ×10)
    ///
    /// The result is scaled by `p.topic_weight`.
    pub fn score(&self, p: &ScoreParameter) -> f64 {
        let p1 = self.first_message_deliveries;
        let p2 = self.mesh_message_deliveries;
        let p3 = -self.mesh_failure_penalty;
        let p4 = -self.invalid_message_deliveries * 10.0;
        (p1 + p2 + p3 + p4) * p.topic_weight
    }
}

/// Application-level behaviour penalties that can be applied to a peer.
#[derive(Debug, Clone, PartialEq)]
pub enum BehaviourPenalty {
    /// Peer sent an invalid message. Score delta: -10.0
    InvalidMessage,
    /// Peer was grafted but immediately triggered a backoff. Score delta: -5.0
    GraftBackoff,
    /// Peer exhibited promiscuous PX behaviour. Score delta: -3.0
    PromiscuousPX,
    /// Custom application-specific score delta.
    AppSpecific(f64),
}

impl BehaviourPenalty {
    /// Returns the score delta associated with this penalty variant.
    pub fn delta(&self) -> f64 {
        match self {
            BehaviourPenalty::InvalidMessage => -10.0,
            BehaviourPenalty::GraftBackoff => -5.0,
            BehaviourPenalty::PromiscuousPX => -3.0,
            BehaviourPenalty::AppSpecific(v) => *v,
        }
    }
}

/// Scoring state for a single peer.
#[derive(Debug, Clone)]
pub struct PeerScore {
    /// Identifier of the peer.
    pub peer_id: String,
    /// Per-topic scoring data.
    pub topic_scores: HashMap<String, TopicScore>,
    /// Accumulated behaviour penalty sum.
    pub behaviour_penalties: f64,
    /// Application-layer score modifier.
    pub app_specific_score: f64,
    /// Time of the last decay pass.
    pub last_decay: Instant,
}

impl PeerScore {
    /// Create a new `PeerScore` for the given peer.
    pub fn new(peer_id: impl Into<String>) -> Self {
        Self {
            peer_id: peer_id.into(),
            topic_scores: HashMap::new(),
            behaviour_penalties: 0.0,
            app_specific_score: 0.0,
            last_decay: Instant::now(),
        }
    }

    /// Compute the composite score combining topic, delivery, and behaviour
    /// components, weighted by `params`.
    ///
    /// ```text
    /// composite = Σ(topic_score_i) * topic_weight
    ///           + behaviour_penalties * behaviour_weight
    ///           + app_specific_score * delivery_weight
    /// ```
    pub fn composite_score(&self, params: &ScoreParameter) -> f64 {
        let topic_sum: f64 = self.topic_scores.values().map(|ts| ts.score(params)).sum();
        let behaviour_part = self.behaviour_penalties * params.behaviour_weight;
        let app_part = self.app_specific_score * params.delivery_weight;
        topic_sum + behaviour_part + app_part
    }

    /// Apply a behaviour penalty to this peer.
    pub fn apply_penalty(&mut self, p: BehaviourPenalty) {
        self.behaviour_penalties += p.delta();
    }

    /// Record that this peer was the first to deliver a message on `topic`.
    pub fn record_first_delivery(&mut self, topic: &str) {
        let ts = self.topic_scores.entry(topic.to_string()).or_default();
        ts.first_message_deliveries += 1.0;
    }

    /// Record a mesh message delivery for `topic`.
    pub fn record_mesh_delivery(&mut self, topic: &str) {
        let ts = self.topic_scores.entry(topic.to_string()).or_default();
        ts.mesh_message_deliveries += 1.0;
    }

    /// Record an invalid message on `topic` and increase the
    /// `invalid_message_deliveries` counter.
    pub fn record_invalid_message(&mut self, topic: &str) {
        let ts = self.topic_scores.entry(topic.to_string()).or_default();
        ts.invalid_message_deliveries += 1.0;
    }

    /// Update the in-mesh flag for `topic`.
    pub fn set_in_mesh(&mut self, topic: &str, in_mesh: bool) {
        let ts = self.topic_scores.entry(topic.to_string()).or_default();
        ts.in_mesh = in_mesh;
    }

    /// Decay all numeric score fields based on time elapsed since the last decay.
    ///
    /// Decay factor: `0.9 ^ (elapsed_secs / decay_interval_secs)`
    pub fn decay(&mut self, params: &ScoreParameter) {
        let elapsed = self.last_decay.elapsed();
        let factor = 0.9_f64.powf(elapsed.as_secs_f64() / params.decay_interval_secs as f64);

        for ts in self.topic_scores.values_mut() {
            ts.first_message_deliveries *= factor;
            ts.mesh_message_deliveries *= factor;
            ts.mesh_failure_penalty *= factor;
            ts.invalid_message_deliveries *= factor;
            ts.mesh_time_secs *= factor;
        }

        self.behaviour_penalties *= factor;
        self.app_specific_score *= factor;
        self.last_decay = Instant::now();
    }
}

/// Central tracker that maintains `PeerScore` entries for every known peer
/// and exposes mesh-management queries.
#[derive(Debug)]
pub struct PeerScoreTracker {
    /// Map from peer ID string to per-peer score state.
    pub scores: HashMap<String, PeerScore>,
    /// Global scoring parameters.
    pub params: ScoreParameter,
}

impl PeerScoreTracker {
    /// Create a new tracker with the given scoring parameters.
    pub fn new(params: ScoreParameter) -> Self {
        Self {
            scores: HashMap::new(),
            params,
        }
    }

    /// Return a mutable reference to the `PeerScore` for `peer_id`,
    /// inserting a default entry if it does not yet exist.
    pub fn get_or_create(&mut self, peer_id: &str) -> &mut PeerScore {
        self.scores
            .entry(peer_id.to_string())
            .or_insert_with(|| PeerScore::new(peer_id))
    }

    /// Return the composite score for `peer_id`, or `0.0` if unknown.
    pub fn composite_score(&self, peer_id: &str) -> f64 {
        self.scores
            .get(peer_id)
            .map(|ps| ps.composite_score(&self.params))
            .unwrap_or(0.0)
    }

    /// Return all peer IDs whose composite score is below the ban threshold.
    pub fn banned_peers(&self) -> Vec<String> {
        self.scores
            .iter()
            .filter(|(_, ps)| ps.composite_score(&self.params) < self.params.score_threshold_ban)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Return all peer IDs whose composite score is below the graylist threshold
    /// but at or above the ban threshold.
    pub fn greylisted_peers(&self) -> Vec<String> {
        self.scores
            .iter()
            .filter(|(_, ps)| {
                let s = ps.composite_score(&self.params);
                s < self.params.score_threshold_graylist && s >= self.params.score_threshold_ban
            })
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Return the top-`n` peer IDs sorted by composite score (descending).
    pub fn best_peers(&self, n: usize) -> Vec<String> {
        let mut scored: Vec<(String, f64)> = self
            .scores
            .iter()
            .map(|(id, ps)| (id.clone(), ps.composite_score(&self.params)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(n).map(|(id, _)| id).collect()
    }

    /// Trigger a decay pass on every tracked peer.
    pub fn apply_decay_all(&mut self) {
        let params = self.params.clone();
        for ps in self.scores.values_mut() {
            ps.decay(&params);
        }
    }

    /// Remove the listed peer IDs from the tracker.
    pub fn prune_peers(&mut self, ids: &[&str]) {
        for id in ids {
            self.scores.remove(*id);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // 1. Default params construction
    #[test]
    fn test_default_params() {
        let p = ScoreParameter::default();
        assert!((p.topic_weight - 0.5).abs() < f64::EPSILON);
        assert!((p.delivery_weight - 0.3).abs() < f64::EPSILON);
        assert!((p.behaviour_weight - 0.2).abs() < f64::EPSILON);
        assert_eq!(p.decay_interval_secs, 10);
        assert!((p.score_threshold_ban - (-100.0)).abs() < f64::EPSILON);
        assert!((p.score_threshold_graylist - (-10.0)).abs() < f64::EPSILON);
    }

    // 2. TopicScore::score formula
    #[test]
    fn test_topic_score_formula() {
        let p = ScoreParameter::default();
        let ts = TopicScore {
            first_message_deliveries: 4.0,
            mesh_message_deliveries: 6.0,
            mesh_failure_penalty: 1.0,
            invalid_message_deliveries: 0.5,
            in_mesh: true,
            mesh_time_secs: 30.0,
        };
        // (4 + 6 - 1 - 5) * 0.5 = 4 * 0.5 = 2.0
        let expected = (4.0 + 6.0 - 1.0 - 5.0) * 0.5;
        assert!((ts.score(&p) - expected).abs() < 1e-9);
    }

    // 3. BehaviourPenalty delta values
    #[test]
    fn test_behaviour_penalty_deltas() {
        assert!((BehaviourPenalty::InvalidMessage.delta() - (-10.0)).abs() < f64::EPSILON);
        assert!((BehaviourPenalty::GraftBackoff.delta() - (-5.0)).abs() < f64::EPSILON);
        assert!((BehaviourPenalty::PromiscuousPX.delta() - (-3.0)).abs() < f64::EPSILON);
        assert!((BehaviourPenalty::AppSpecific(7.5).delta() - 7.5).abs() < f64::EPSILON);
        assert!((BehaviourPenalty::AppSpecific(-2.0).delta() - (-2.0)).abs() < f64::EPSILON);
    }

    // 4. composite_score weighted sum
    #[test]
    fn test_composite_score_weighted_sum() {
        let p = ScoreParameter::default();
        let mut ps = PeerScore::new("peer-a");
        ps.app_specific_score = 10.0;
        ps.behaviour_penalties = -5.0;
        // No topic scores: composite = 0 + (-5 * 0.2) + (10 * 0.3) = -1.0 + 3.0 = 2.0
        let c = ps.composite_score(&p);
        assert!((c - 2.0).abs() < 1e-9, "got {c}");
    }

    // 5. record_first_delivery increments correct field
    #[test]
    fn test_record_first_delivery() {
        let mut ps = PeerScore::new("peer-b");
        ps.record_first_delivery("topic-1");
        ps.record_first_delivery("topic-1");
        let ts = ps.topic_scores.get("topic-1").expect("topic must exist");
        assert!((ts.first_message_deliveries - 2.0).abs() < f64::EPSILON);
        assert!((ts.mesh_message_deliveries).abs() < f64::EPSILON);
    }

    // 6. record_mesh_delivery increments mesh field
    #[test]
    fn test_record_mesh_delivery() {
        let mut ps = PeerScore::new("peer-c");
        ps.record_mesh_delivery("topic-2");
        ps.record_mesh_delivery("topic-2");
        ps.record_mesh_delivery("topic-2");
        let ts = ps.topic_scores.get("topic-2").expect("topic must exist");
        assert!((ts.mesh_message_deliveries - 3.0).abs() < f64::EPSILON);
        assert!((ts.first_message_deliveries).abs() < f64::EPSILON);
    }

    // 7. record_invalid_message increments invalid + penalises composite
    #[test]
    fn test_record_invalid_message_penalises() {
        let p = ScoreParameter::default();
        let mut ps = PeerScore::new("peer-d");
        let before = ps.composite_score(&p);
        ps.record_invalid_message("topic-3");
        let after = ps.composite_score(&p);
        assert!(after < before, "invalid message must lower composite score");
        let ts = ps.topic_scores.get("topic-3").expect("topic must exist");
        assert!((ts.invalid_message_deliveries - 1.0).abs() < f64::EPSILON);
    }

    // 8. set_in_mesh updates in_mesh flag
    #[test]
    fn test_set_in_mesh() {
        let mut ps = PeerScore::new("peer-e");
        ps.set_in_mesh("t", true);
        assert!(ps.topic_scores["t"].in_mesh);
        ps.set_in_mesh("t", false);
        assert!(!ps.topic_scores["t"].in_mesh);
    }

    // 9. apply_penalty BehaviourPenalty variants
    #[test]
    fn test_apply_penalty_variants() {
        let mut ps = PeerScore::new("peer-f");
        assert!((ps.behaviour_penalties).abs() < f64::EPSILON);
        ps.apply_penalty(BehaviourPenalty::InvalidMessage);
        assert!((ps.behaviour_penalties - (-10.0)).abs() < f64::EPSILON);
        ps.apply_penalty(BehaviourPenalty::GraftBackoff);
        assert!((ps.behaviour_penalties - (-15.0)).abs() < f64::EPSILON);
        ps.apply_penalty(BehaviourPenalty::PromiscuousPX);
        assert!((ps.behaviour_penalties - (-18.0)).abs() < f64::EPSILON);
        ps.apply_penalty(BehaviourPenalty::AppSpecific(3.0));
        assert!((ps.behaviour_penalties - (-15.0)).abs() < f64::EPSILON);
    }

    // 10. decay reduces all numeric fields
    #[test]
    fn test_decay_reduces_fields() {
        let p = ScoreParameter {
            decay_interval_secs: 1,
            ..ScoreParameter::default()
        };
        let mut ps = PeerScore::new("peer-g");
        ps.record_first_delivery("t");
        ps.record_mesh_delivery("t");
        ps.record_invalid_message("t");
        ps.behaviour_penalties = -20.0;
        ps.app_specific_score = 5.0;

        // Force last_decay to be far enough in the past for a visible decay
        ps.last_decay = Instant::now() - Duration::from_secs(10);
        ps.decay(&p);

        let ts = &ps.topic_scores["t"];
        assert!(ts.first_message_deliveries < 1.0);
        assert!(ts.mesh_message_deliveries < 1.0);
        assert!(ts.invalid_message_deliveries < 1.0);
        assert!(ps.behaviour_penalties > -20.0);
        assert!(ps.app_specific_score < 5.0);
    }

    // 11. banned_peers filters by threshold
    #[test]
    fn test_banned_peers() {
        let p = ScoreParameter::default();
        let mut tracker = PeerScoreTracker::new(p);
        {
            let ps = tracker.get_or_create("good");
            ps.app_specific_score = 10.0;
        }
        {
            let ps = tracker.get_or_create("bad");
            // behaviour_penalties * 0.2 must be < -100
            ps.behaviour_penalties = -600.0;
        }
        let banned = tracker.banned_peers();
        assert!(banned.contains(&"bad".to_string()));
        assert!(!banned.contains(&"good".to_string()));
    }

    // 12. greylisted_peers range filter
    #[test]
    fn test_greylisted_peers() {
        let p = ScoreParameter::default();
        let mut tracker = PeerScoreTracker::new(p);
        {
            let ps = tracker.get_or_create("ok");
            ps.app_specific_score = 5.0;
        }
        {
            let ps = tracker.get_or_create("grey");
            // -30 * 0.2 = -6 < -10? No; need > -50 to stay above ban
            // behaviour * 0.2 = -30: that is -6.0, not grey enough
            // Let's target: composite = -15 (grey) => behaviour * 0.2 = -15 => behaviour = -75
            ps.behaviour_penalties = -75.0;
        }
        {
            let ps = tracker.get_or_create("banned");
            ps.behaviour_penalties = -600.0;
        }
        let grey = tracker.greylisted_peers();
        assert!(grey.contains(&"grey".to_string()));
        assert!(!grey.contains(&"ok".to_string()));
        assert!(!grey.contains(&"banned".to_string()));
    }

    // 13. best_peers sorted descending
    #[test]
    fn test_best_peers_sorted() {
        let p = ScoreParameter::default();
        let mut tracker = PeerScoreTracker::new(p);
        for (id, score) in [("p1", 5.0), ("p2", 20.0), ("p3", 1.0), ("p4", 10.0)] {
            let ps = tracker.get_or_create(id);
            ps.app_specific_score = score;
        }
        let best = tracker.best_peers(3);
        assert_eq!(best.len(), 3);
        assert_eq!(best[0], "p2");
        assert_eq!(best[1], "p4");
        assert_eq!(best[2], "p1");
    }

    // 14. prune_peers removes entries
    #[test]
    fn test_prune_peers() {
        let p = ScoreParameter::default();
        let mut tracker = PeerScoreTracker::new(p);
        tracker.get_or_create("a");
        tracker.get_or_create("b");
        tracker.get_or_create("c");
        assert_eq!(tracker.scores.len(), 3);
        tracker.prune_peers(&["a", "c"]);
        assert_eq!(tracker.scores.len(), 1);
        assert!(tracker.scores.contains_key("b"));
        assert!(!tracker.scores.contains_key("a"));
        assert!(!tracker.scores.contains_key("c"));
    }

    // 15. get_or_create creates missing entry
    #[test]
    fn test_get_or_create() {
        let p = ScoreParameter::default();
        let mut tracker = PeerScoreTracker::new(p);
        assert!(!tracker.scores.contains_key("new-peer"));
        tracker.get_or_create("new-peer");
        assert!(tracker.scores.contains_key("new-peer"));
        // Calling again should not duplicate
        tracker.get_or_create("new-peer");
        assert_eq!(tracker.scores.len(), 1);
    }

    // 16. apply_decay_all touches all peers
    #[test]
    fn test_apply_decay_all() {
        let p = ScoreParameter {
            decay_interval_secs: 1,
            ..ScoreParameter::default()
        };
        let mut tracker = PeerScoreTracker::new(p);
        for id in ["x", "y", "z"] {
            let ps = tracker.get_or_create(id);
            ps.behaviour_penalties = -50.0;
            ps.last_decay = Instant::now() - Duration::from_secs(10);
        }
        tracker.apply_decay_all();
        for id in ["x", "y", "z"] {
            let ps = &tracker.scores[id];
            // After decay over 10 intervals: 0.9^10 ≈ 0.349 => -50 * 0.349 ≈ -17.4
            assert!(ps.behaviour_penalties > -50.0, "decay must reduce {id}");
            assert!(ps.behaviour_penalties < 0.0, "{id} still negative");
        }
    }

    // Extra: composite_score returns 0.0 for unknown peers
    #[test]
    fn test_composite_score_unknown_peer() {
        let tracker = PeerScoreTracker::new(ScoreParameter::default());
        assert!((tracker.composite_score("ghost")).abs() < f64::EPSILON);
    }
}
