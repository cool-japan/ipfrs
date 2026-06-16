//! Composite peer scoring system (PeerScoreboard)
//!
//! Combines multiple weighted scoring components into a single composite score
//! per peer, with configurable decay and tick-based aging.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_network::scoreboard::{PeerScoreboard, ScoreboardStats};
//!
//! let mut board = PeerScoreboard::new(0.99);
//! board.register_component("latency", 2.0);
//! board.register_component("uptime", 1.0);
//! board.update_score("peer-a", "latency", 0.8);
//! board.update_score("peer-a", "uptime", 1.0);
//! assert!(board.get_score("peer-a").is_some());
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single scoring dimension for a peer.
#[derive(Debug, Clone)]
pub struct ScoreComponent {
    /// Human-readable name of this component.
    pub name: String,
    /// Weight applied when computing the composite score.
    pub weight: f64,
    /// Normalised value in `[0.0, 1.0]`.
    pub value: f64,
    /// Tick at which this component was last updated.
    pub last_updated_tick: u64,
}

/// Aggregated score entry for one peer.
#[derive(Debug, Clone)]
pub struct SbPeerScore {
    /// Peer identifier.
    pub peer_id: String,
    /// Per-component scores keyed by component name.
    pub components: HashMap<String, ScoreComponent>,
    /// Weighted-sum composite score (recomputed on every mutation).
    pub composite_score: f64,
}

/// Summary statistics for the entire scoreboard.
#[derive(Debug, Clone)]
pub struct ScoreboardStats {
    /// Number of tracked peers.
    pub peer_count: usize,
    /// Number of registered component types.
    pub component_count: usize,
    /// Mean composite score across all peers.
    pub avg_composite: f64,
    /// Maximum composite score across all peers.
    pub max_composite: f64,
}

/// Composite peer scoreboard.
///
/// Maintains per-peer scores across multiple weighted components and provides
/// ranking, decay, and statistics.
#[derive(Debug, Clone)]
pub struct PeerScoreboard {
    peers: HashMap<String, SbPeerScore>,
    component_weights: HashMap<String, f64>,
    current_tick: u64,
    decay_rate: f64,
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl PeerScoreboard {
    /// Create a new scoreboard with the given per-tick decay rate.
    ///
    /// `decay_rate` is clamped to `[0.0, 1.0]`.
    pub fn new(decay_rate: f64) -> Self {
        Self {
            peers: HashMap::new(),
            component_weights: HashMap::new(),
            current_tick: 0,
            decay_rate: decay_rate.clamp(0.0, 1.0),
        }
    }

    /// Register (or update the default weight of) a scoring component.
    pub fn register_component(&mut self, name: &str, default_weight: f64) {
        self.component_weights
            .insert(name.to_string(), default_weight);
    }

    /// Update a single component score for `peer_id`.
    ///
    /// `value` is clamped to `[0.0, 1.0]`. If the component has not been
    /// registered the call is silently ignored.
    pub fn update_score(&mut self, peer_id: &str, component: &str, value: f64) {
        let weight = match self.component_weights.get(component) {
            Some(&w) => w,
            None => return,
        };

        let tick = self.current_tick;
        let entry = self
            .peers
            .entry(peer_id.to_string())
            .or_insert_with(|| SbPeerScore {
                peer_id: peer_id.to_string(),
                components: HashMap::new(),
                composite_score: 0.0,
            });

        let comp = entry
            .components
            .entry(component.to_string())
            .or_insert_with(|| ScoreComponent {
                name: component.to_string(),
                weight,
                value: 0.0,
                last_updated_tick: tick,
            });

        comp.value = value.clamp(0.0, 1.0);
        comp.weight = weight;
        comp.last_updated_tick = tick;

        Self::recompute_composite(entry);
    }

    /// Return the composite score for `peer_id`, if tracked.
    pub fn get_score(&self, peer_id: &str) -> Option<f64> {
        self.peers.get(peer_id).map(|p| p.composite_score)
    }

    /// Return the value of a single component for `peer_id`.
    pub fn get_component(&self, peer_id: &str, component: &str) -> Option<f64> {
        self.peers
            .get(peer_id)
            .and_then(|p| p.components.get(component))
            .map(|c| c.value)
    }

    /// Return all peers ranked by composite score (descending).
    pub fn rank_peers(&self) -> Vec<(String, f64)> {
        let mut ranked: Vec<(String, f64)> = self
            .peers
            .values()
            .map(|p| (p.peer_id.clone(), p.composite_score))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    /// Return the top `n` peers by composite score (descending).
    pub fn top_peers(&self, n: usize) -> Vec<(String, f64)> {
        let ranked = self.rank_peers();
        ranked.into_iter().take(n).collect()
    }

    /// Multiply all component values by `decay_rate` and recompute composites.
    pub fn apply_decay(&mut self) {
        let rate = self.decay_rate;
        for entry in self.peers.values_mut() {
            for comp in entry.components.values_mut() {
                comp.value *= rate;
                // Re-clamp just in case of floating-point drift.
                comp.value = comp.value.clamp(0.0, 1.0);
            }
            Self::recompute_composite(entry);
        }
    }

    /// Remove a peer from the scoreboard. Returns `true` if the peer existed.
    pub fn remove_peer(&mut self, peer_id: &str) -> bool {
        self.peers.remove(peer_id).is_some()
    }

    /// Number of tracked peers.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Advance the internal tick counter by one and apply decay.
    pub fn tick(&mut self) {
        self.current_tick = self.current_tick.saturating_add(1);
        self.apply_decay();
    }

    /// Snapshot statistics for the entire scoreboard.
    pub fn stats(&self) -> ScoreboardStats {
        let peer_count = self.peers.len();
        let component_count = self.component_weights.len();

        let (sum, max) = self.peers.values().fold((0.0_f64, 0.0_f64), |(s, m), p| {
            (s + p.composite_score, m.max(p.composite_score))
        });

        let avg_composite = if peer_count > 0 {
            sum / peer_count as f64
        } else {
            0.0
        };

        ScoreboardStats {
            peer_count,
            component_count,
            avg_composite,
            max_composite: max,
        }
    }

    /// Return the current tick value.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Recompute the composite score for a single peer entry.
    fn recompute_composite(entry: &mut SbPeerScore) {
        let total_weight: f64 = entry.components.values().map(|c| c.weight.abs()).sum();
        if total_weight == 0.0 {
            entry.composite_score = 0.0;
            return;
        }
        let weighted_sum: f64 = entry.components.values().map(|c| c.value * c.weight).sum();
        entry.composite_score = weighted_sum / total_weight;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- basics --------------------------------------------------------------

    #[test]
    fn test_new_scoreboard_empty() {
        let sb = PeerScoreboard::new(0.999);
        assert_eq!(sb.peer_count(), 0);
        assert_eq!(sb.current_tick(), 0);
    }

    #[test]
    fn test_register_component() {
        let mut sb = PeerScoreboard::new(0.999);
        sb.register_component("latency", 2.0);
        sb.register_component("uptime", 1.0);
        let stats = sb.stats();
        assert_eq!(stats.component_count, 2);
    }

    #[test]
    fn test_update_score_creates_peer() {
        let mut sb = PeerScoreboard::new(0.999);
        sb.register_component("latency", 1.0);
        sb.update_score("p1", "latency", 0.5);
        assert_eq!(sb.peer_count(), 1);
    }

    #[test]
    fn test_get_score_known_peer() {
        let mut sb = PeerScoreboard::new(0.999);
        sb.register_component("x", 1.0);
        sb.update_score("p1", "x", 0.7);
        let score = sb.get_score("p1");
        assert!(score.is_some());
        assert!((score.expect("just checked") - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_get_score_unknown_peer() {
        let sb = PeerScoreboard::new(0.999);
        assert!(sb.get_score("missing").is_none());
    }

    #[test]
    fn test_get_component() {
        let mut sb = PeerScoreboard::new(0.999);
        sb.register_component("a", 1.0);
        sb.update_score("p1", "a", 0.42);
        let v = sb.get_component("p1", "a");
        assert!(v.is_some());
        assert!((v.expect("just checked") - 0.42).abs() < 1e-9);
    }

    #[test]
    fn test_get_component_missing_component() {
        let mut sb = PeerScoreboard::new(0.999);
        sb.register_component("a", 1.0);
        sb.update_score("p1", "a", 0.5);
        assert!(sb.get_component("p1", "b").is_none());
    }

    #[test]
    fn test_get_component_missing_peer() {
        let sb = PeerScoreboard::new(0.999);
        assert!(sb.get_component("nope", "x").is_none());
    }

    // -- clamping ------------------------------------------------------------

    #[test]
    fn test_value_clamped_above_one() {
        let mut sb = PeerScoreboard::new(0.999);
        sb.register_component("c", 1.0);
        sb.update_score("p1", "c", 5.0);
        assert!((sb.get_component("p1", "c").expect("exists") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_value_clamped_below_zero() {
        let mut sb = PeerScoreboard::new(0.999);
        sb.register_component("c", 1.0);
        sb.update_score("p1", "c", -3.0);
        assert!((sb.get_component("p1", "c").expect("exists")).abs() < 1e-9);
    }

    #[test]
    fn test_decay_rate_clamped() {
        let sb = PeerScoreboard::new(2.0);
        assert!((sb.decay_rate - 1.0).abs() < 1e-9);
        let sb2 = PeerScoreboard::new(-0.5);
        assert!(sb2.decay_rate.abs() < 1e-9);
    }

    // -- composite calculation -----------------------------------------------

    #[test]
    fn test_composite_single_component() {
        let mut sb = PeerScoreboard::new(0.999);
        sb.register_component("a", 1.0);
        sb.update_score("p1", "a", 0.8);
        assert!((sb.get_score("p1").expect("exists") - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_composite_equal_weights() {
        let mut sb = PeerScoreboard::new(0.999);
        sb.register_component("a", 1.0);
        sb.register_component("b", 1.0);
        sb.update_score("p1", "a", 0.6);
        sb.update_score("p1", "b", 0.4);
        // (0.6*1 + 0.4*1) / (1+1) = 0.5
        assert!((sb.get_score("p1").expect("exists") - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_composite_unequal_weights() {
        let mut sb = PeerScoreboard::new(0.999);
        sb.register_component("a", 3.0);
        sb.register_component("b", 1.0);
        sb.update_score("p1", "a", 1.0);
        sb.update_score("p1", "b", 0.0);
        // (1.0*3 + 0.0*1) / (3+1) = 0.75
        assert!((sb.get_score("p1").expect("exists") - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_composite_updates_on_change() {
        let mut sb = PeerScoreboard::new(0.999);
        sb.register_component("a", 1.0);
        sb.update_score("p1", "a", 0.2);
        assert!((sb.get_score("p1").expect("exists") - 0.2).abs() < 1e-9);
        sb.update_score("p1", "a", 0.9);
        assert!((sb.get_score("p1").expect("exists") - 0.9).abs() < 1e-9);
    }

    // -- ranking -------------------------------------------------------------

    #[test]
    fn test_rank_peers_descending() {
        let mut sb = PeerScoreboard::new(0.999);
        sb.register_component("a", 1.0);
        sb.update_score("low", "a", 0.1);
        sb.update_score("mid", "a", 0.5);
        sb.update_score("high", "a", 0.9);
        let ranked = sb.rank_peers();
        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0].0, "high");
        assert_eq!(ranked[1].0, "mid");
        assert_eq!(ranked[2].0, "low");
    }

    #[test]
    fn test_rank_peers_empty() {
        let sb = PeerScoreboard::new(0.999);
        assert!(sb.rank_peers().is_empty());
    }

    #[test]
    fn test_top_peers() {
        let mut sb = PeerScoreboard::new(0.999);
        sb.register_component("a", 1.0);
        sb.update_score("p1", "a", 0.1);
        sb.update_score("p2", "a", 0.9);
        sb.update_score("p3", "a", 0.5);
        let top = sb.top_peers(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "p2");
        assert_eq!(top[1].0, "p3");
    }

    #[test]
    fn test_top_peers_n_exceeds_count() {
        let mut sb = PeerScoreboard::new(0.999);
        sb.register_component("a", 1.0);
        sb.update_score("p1", "a", 0.5);
        let top = sb.top_peers(100);
        assert_eq!(top.len(), 1);
    }

    // -- decay ---------------------------------------------------------------

    #[test]
    fn test_apply_decay_reduces_scores() {
        let mut sb = PeerScoreboard::new(0.5);
        sb.register_component("a", 1.0);
        sb.update_score("p1", "a", 1.0);
        sb.apply_decay();
        let v = sb.get_component("p1", "a").expect("exists");
        assert!((v - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_apply_decay_multiple_times() {
        let mut sb = PeerScoreboard::new(0.5);
        sb.register_component("a", 1.0);
        sb.update_score("p1", "a", 1.0);
        sb.apply_decay();
        sb.apply_decay();
        let v = sb.get_component("p1", "a").expect("exists");
        assert!((v - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_tick_advances_and_decays() {
        let mut sb = PeerScoreboard::new(0.5);
        sb.register_component("a", 1.0);
        sb.update_score("p1", "a", 1.0);
        sb.tick();
        assert_eq!(sb.current_tick(), 1);
        let v = sb.get_component("p1", "a").expect("exists");
        assert!((v - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_tick_multiple() {
        let mut sb = PeerScoreboard::new(0.9);
        sb.register_component("a", 1.0);
        sb.update_score("p1", "a", 1.0);
        for _ in 0..10 {
            sb.tick();
        }
        assert_eq!(sb.current_tick(), 10);
        let v = sb.get_component("p1", "a").expect("exists");
        let expected = 0.9_f64.powi(10);
        assert!((v - expected).abs() < 1e-9);
    }

    // -- remove peer ---------------------------------------------------------

    #[test]
    fn test_remove_peer_existing() {
        let mut sb = PeerScoreboard::new(0.999);
        sb.register_component("a", 1.0);
        sb.update_score("p1", "a", 0.5);
        assert!(sb.remove_peer("p1"));
        assert_eq!(sb.peer_count(), 0);
        assert!(sb.get_score("p1").is_none());
    }

    #[test]
    fn test_remove_peer_nonexistent() {
        let mut sb = PeerScoreboard::new(0.999);
        assert!(!sb.remove_peer("ghost"));
    }

    // -- stats ---------------------------------------------------------------

    #[test]
    fn test_stats_empty_board() {
        let sb = PeerScoreboard::new(0.999);
        let s = sb.stats();
        assert_eq!(s.peer_count, 0);
        assert_eq!(s.component_count, 0);
        assert!((s.avg_composite).abs() < 1e-9);
        assert!((s.max_composite).abs() < 1e-9);
    }

    #[test]
    fn test_stats_with_peers() {
        let mut sb = PeerScoreboard::new(0.999);
        sb.register_component("a", 1.0);
        sb.update_score("p1", "a", 0.4);
        sb.update_score("p2", "a", 0.8);
        let s = sb.stats();
        assert_eq!(s.peer_count, 2);
        assert_eq!(s.component_count, 1);
        assert!((s.avg_composite - 0.6).abs() < 1e-9);
        assert!((s.max_composite - 0.8).abs() < 1e-9);
    }

    // -- unregistered component is ignored -----------------------------------

    #[test]
    fn test_update_unregistered_component_ignored() {
        let mut sb = PeerScoreboard::new(0.999);
        sb.update_score("p1", "nope", 1.0);
        assert_eq!(sb.peer_count(), 0);
    }

    // -- multiple components with different weights --------------------------

    #[test]
    fn test_multiple_components_weighted() {
        let mut sb = PeerScoreboard::new(0.999);
        sb.register_component("speed", 2.0);
        sb.register_component("reliability", 3.0);
        sb.register_component("age", 1.0);
        sb.update_score("p1", "speed", 0.8);
        sb.update_score("p1", "reliability", 0.6);
        sb.update_score("p1", "age", 1.0);
        // composite = (0.8*2 + 0.6*3 + 1.0*1) / (2+3+1) = (1.6+1.8+1.0)/6 = 4.4/6
        let expected = 4.4 / 6.0;
        assert!((sb.get_score("p1").expect("exists") - expected).abs() < 1e-9);
    }

    // -- register new component after peers exist ----------------------------

    #[test]
    fn test_register_component_after_peers_added() {
        let mut sb = PeerScoreboard::new(0.999);
        sb.register_component("a", 1.0);
        sb.update_score("p1", "a", 0.5);
        // register new component and update existing peer
        sb.register_component("b", 1.0);
        sb.update_score("p1", "b", 1.0);
        // composite = (0.5+1.0)/2 = 0.75
        assert!((sb.get_score("p1").expect("exists") - 0.75).abs() < 1e-9);
    }

    // -- decay with zero rate ------------------------------------------------

    #[test]
    fn test_decay_with_zero_rate() {
        let mut sb = PeerScoreboard::new(0.0);
        sb.register_component("a", 1.0);
        sb.update_score("p1", "a", 1.0);
        sb.apply_decay();
        assert!((sb.get_component("p1", "a").expect("exists")).abs() < 1e-9);
    }

    // -- decay with rate 1.0 (no decay) --------------------------------------

    #[test]
    fn test_decay_with_rate_one() {
        let mut sb = PeerScoreboard::new(1.0);
        sb.register_component("a", 1.0);
        sb.update_score("p1", "a", 0.7);
        sb.apply_decay();
        assert!((sb.get_component("p1", "a").expect("exists") - 0.7).abs() < 1e-9);
    }

    // -- peer_count ----------------------------------------------------------

    #[test]
    fn test_peer_count_multiple() {
        let mut sb = PeerScoreboard::new(0.999);
        sb.register_component("a", 1.0);
        sb.update_score("p1", "a", 0.1);
        sb.update_score("p2", "a", 0.2);
        sb.update_score("p3", "a", 0.3);
        assert_eq!(sb.peer_count(), 3);
    }

    // -- composite with partial components -----------------------------------

    #[test]
    fn test_composite_partial_components() {
        let mut sb = PeerScoreboard::new(0.999);
        sb.register_component("a", 2.0);
        sb.register_component("b", 2.0);
        // Only set component "a"
        sb.update_score("p1", "a", 1.0);
        // composite = 1.0*2 / 2 = 1.0 (only "a" present)
        assert!((sb.get_score("p1").expect("exists") - 1.0).abs() < 1e-9);
    }

    // -- last_updated_tick set correctly -------------------------------------

    #[test]
    fn test_last_updated_tick() {
        let mut sb = PeerScoreboard::new(0.999);
        sb.register_component("a", 1.0);
        sb.tick(); // tick = 1
        sb.tick(); // tick = 2
        sb.update_score("p1", "a", 0.5);
        let entry = sb.peers.get("p1").expect("exists");
        let comp = entry.components.get("a").expect("exists");
        assert_eq!(comp.last_updated_tick, 2);
    }

    // -- decay recomputes composite ------------------------------------------

    #[test]
    fn test_decay_recomputes_composite() {
        let mut sb = PeerScoreboard::new(0.5);
        sb.register_component("a", 1.0);
        sb.register_component("b", 1.0);
        sb.update_score("p1", "a", 1.0);
        sb.update_score("p1", "b", 1.0);
        // composite = 1.0
        assert!((sb.get_score("p1").expect("exists") - 1.0).abs() < 1e-9);
        sb.apply_decay();
        // both halved => composite = 0.5
        assert!((sb.get_score("p1").expect("exists") - 0.5).abs() < 1e-9);
    }

    // -- ranking stability with equal scores ---------------------------------

    #[test]
    fn test_rank_peers_equal_scores() {
        let mut sb = PeerScoreboard::new(0.999);
        sb.register_component("a", 1.0);
        sb.update_score("p1", "a", 0.5);
        sb.update_score("p2", "a", 0.5);
        let ranked = sb.rank_peers();
        assert_eq!(ranked.len(), 2);
        // Both should be 0.5
        assert!((ranked[0].1 - 0.5).abs() < 1e-9);
        assert!((ranked[1].1 - 0.5).abs() < 1e-9);
    }

    // -- weight update via re-register ---------------------------------------

    #[test]
    fn test_reregister_component_updates_weight() {
        let mut sb = PeerScoreboard::new(0.999);
        sb.register_component("a", 1.0);
        sb.update_score("p1", "a", 1.0);
        assert!((sb.get_score("p1").expect("exists") - 1.0).abs() < 1e-9);
        // Change weight and re-update
        sb.register_component("a", 3.0);
        sb.update_score("p1", "a", 1.0);
        // Still 1.0 because only one component
        assert!((sb.get_score("p1").expect("exists") - 1.0).abs() < 1e-9);
    }
}
