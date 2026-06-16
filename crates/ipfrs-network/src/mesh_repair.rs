//! GossipSub mesh repair coordination
//!
//! [`MeshRepairCoordinator`] tracks the current peer count for every
//! subscribed topic and decides when the GossipSub mesh needs to be *repaired*
//! (too few peers) or *pruned* (too many peers), subject to a configurable
//! cooldown between consecutive repair attempts.

use parking_lot::RwLock;
use std::collections::HashMap;

// ─── Configuration ───────────────────────────────────────────────────────────

/// Parameters controlling when mesh repairs and prunes are triggered.
#[derive(Debug, Clone)]
pub struct MeshRepairConfig {
    /// Minimum peer count below which a mesh repair is triggered (`D_low`).
    pub d_low: usize,
    /// Target peer count after a successful repair (`D`).
    pub d_target: usize,
    /// Maximum peer count above which a prune is triggered (`D_high`).
    pub d_high: usize,
    /// Minimum milliseconds that must elapse between consecutive repair
    /// attempts for the same topic.
    pub repair_interval_ms: u64,
}

impl Default for MeshRepairConfig {
    fn default() -> Self {
        Self {
            d_low: 6,
            d_target: 8,
            d_high: 12,
            repair_interval_ms: 5_000,
        }
    }
}

// ─── Per-topic state ─────────────────────────────────────────────────────────

/// Repair / prune bookkeeping for a single GossipSub topic.
#[derive(Debug, Clone)]
pub struct MeshRepairState {
    /// Topic identifier string (e.g. `/ipfrs/content/announce/1.0.0`).
    pub topic: String,
    /// Most recently recorded peer count for this topic.
    pub current_peers: usize,
    /// Unix timestamp (ms) of the last repair action, or `0` if no repair has
    /// occurred yet.
    pub last_repair_ms: u64,
    /// Total number of repair actions recorded for this topic.
    pub repair_count: u64,
    /// Total number of prune actions recorded for this topic.
    pub prune_count: u64,
}

impl MeshRepairState {
    fn new(topic: impl Into<String>) -> Self {
        Self {
            topic: topic.into(),
            current_peers: 0,
            last_repair_ms: 0,
            repair_count: 0,
            prune_count: 0,
        }
    }
}

// ─── Coordinator ─────────────────────────────────────────────────────────────

/// Coordinates mesh repair and prune decisions across all subscribed topics.
///
/// The coordinator is **policy-only**: it records peer counts and timestamps,
/// evaluates thresholds, and signals when action is needed.  The actual libp2p
/// swarm interactions are the caller's responsibility.
pub struct MeshRepairCoordinator {
    config: MeshRepairConfig,
    /// Per-topic repair state, keyed by topic string.
    topic_states: RwLock<HashMap<String, MeshRepairState>>,
}

impl MeshRepairCoordinator {
    /// Create a new coordinator with the supplied configuration.
    pub fn new(config: MeshRepairConfig) -> Self {
        Self {
            config,
            topic_states: RwLock::new(HashMap::new()),
        }
    }

    // ── Mutation helpers ──────────────────────────────────────────────────

    /// Update the tracked peer count for `topic` at time `now_ms`.
    ///
    /// If the topic has not been seen before it is automatically registered.
    pub fn record_peer_count(&self, topic: &str, count: usize, now_ms: u64) {
        let mut map = self.topic_states.write();
        let state = map
            .entry(topic.to_string())
            .or_insert_with(|| MeshRepairState::new(topic));
        state.current_peers = count;
        // Update timestamp so freshness checks work even before first repair.
        // We do NOT reset last_repair_ms here; that is only set by
        // record_repair() so the cooldown is measured from the actual repair,
        // not from every peer-count update.
        let _ = now_ms; // parameter reserved for future use / logging
    }

    /// Record that a mesh repair was performed for `topic` at `now_ms`.
    pub fn record_repair(&self, topic: &str, now_ms: u64) {
        let mut map = self.topic_states.write();
        let state = map
            .entry(topic.to_string())
            .or_insert_with(|| MeshRepairState::new(topic));
        state.last_repair_ms = now_ms;
        state.repair_count = state.repair_count.saturating_add(1);
    }

    /// Record that a mesh prune was performed for `topic`.
    pub fn record_prune(&self, topic: &str) {
        let mut map = self.topic_states.write();
        let state = map
            .entry(topic.to_string())
            .or_insert_with(|| MeshRepairState::new(topic));
        state.prune_count = state.prune_count.saturating_add(1);
    }

    // ── Query helpers ─────────────────────────────────────────────────────

    /// Returns `true` when the topic's mesh needs to be repaired.
    ///
    /// Both conditions must hold:
    /// 1. `current_peers < d_low`
    /// 2. At least `repair_interval_ms` has elapsed since the last repair
    ///    (or no repair has occurred yet, i.e. `last_repair_ms == 0`).
    pub fn needs_repair(&self, topic: &str, now_ms: u64) -> bool {
        let map = self.topic_states.read();
        match map.get(topic) {
            None => false,
            Some(state) => {
                let below_threshold = state.current_peers < self.config.d_low;
                let cooldown_elapsed = state.last_repair_ms == 0
                    || now_ms.saturating_sub(state.last_repair_ms)
                        >= self.config.repair_interval_ms;
                below_threshold && cooldown_elapsed
            }
        }
    }

    /// Returns `true` when the topic's mesh is over-populated and should be
    /// pruned.
    pub fn needs_prune(&self, topic: &str) -> bool {
        let map = self.topic_states.read();
        match map.get(topic) {
            None => false,
            Some(state) => state.current_peers > self.config.d_high,
        }
    }

    /// Return a snapshot of the repair state for `topic`, or `None` if the
    /// topic has not been registered yet.
    pub fn repair_state(&self, topic: &str) -> Option<MeshRepairState> {
        self.topic_states.read().get(topic).cloned()
    }

    /// Return the list of topic names that currently need repair.
    pub fn topics_needing_repair(&self, now_ms: u64) -> Vec<String> {
        let map = self.topic_states.read();
        map.values()
            .filter(|s| {
                let below = s.current_peers < self.config.d_low;
                let cooldown = s.last_repair_ms == 0
                    || now_ms.saturating_sub(s.last_repair_ms) >= self.config.repair_interval_ms;
                below && cooldown
            })
            .map(|s| s.topic.clone())
            .collect()
    }

    /// Return the list of topic names that currently need pruning.
    pub fn topics_needing_prune(&self) -> Vec<String> {
        let map = self.topic_states.read();
        map.values()
            .filter(|s| s.current_peers > self.config.d_high)
            .map(|s| s.topic.clone())
            .collect()
    }

    /// Return snapshots of all tracked topic states.
    pub fn all_states(&self) -> Vec<MeshRepairState> {
        self.topic_states.read().values().cloned().collect()
    }

    /// Reference to the active configuration.
    pub fn config(&self) -> &MeshRepairConfig {
        &self.config
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_coordinator() -> MeshRepairCoordinator {
        MeshRepairCoordinator::new(MeshRepairConfig::default())
    }

    const T: &str = "/ipfrs/test/1.0.0";

    // ── RelayManager tests are in relay.rs; these focus on mesh repair ────

    #[test]
    fn test_initial_no_repair_needed() {
        let coord = default_coordinator();
        // No topic registered → needs_repair must return false.
        assert!(!coord.needs_repair(T, 10_000));
    }

    #[test]
    fn test_needs_repair_below_d_low() {
        let coord = default_coordinator();
        // d_low = 6; set count to 5 (below threshold).
        // last_repair_ms = 0 → cooldown trivially satisfied.
        coord.record_peer_count(T, 5, 0);
        assert!(coord.needs_repair(T, 10_000));
    }

    #[test]
    fn test_no_repair_if_too_recent() {
        let coord = default_coordinator();
        // Record a low peer count.
        coord.record_peer_count(T, 3, 0);
        // Record a repair at t=1000 ms.
        coord.record_repair(T, 1_000);
        // At t=5999 ms the 5000 ms cooldown has NOT elapsed yet.
        assert!(!coord.needs_repair(T, 5_999));
        // At t=6000 ms the cooldown HAS elapsed.
        assert!(coord.needs_repair(T, 6_000));
    }

    #[test]
    fn test_needs_prune_above_d_high() {
        let coord = default_coordinator();
        // d_high = 12; set count to 13.
        coord.record_peer_count(T, 13, 0);
        assert!(coord.needs_prune(T));
    }

    #[test]
    fn test_record_repair_updates_state() {
        let coord = default_coordinator();
        coord.record_peer_count(T, 2, 0);
        coord.record_repair(T, 3_000);
        coord.record_repair(T, 8_000);

        let state = coord.repair_state(T).expect("state present");
        assert_eq!(state.last_repair_ms, 8_000);
        assert_eq!(state.repair_count, 2);
    }

    #[test]
    fn test_topics_needing_repair_multiple_topics() {
        let coord = default_coordinator();
        let t1 = "/ipfrs/t1/1.0.0";
        let t2 = "/ipfrs/t2/1.0.0";
        let t3 = "/ipfrs/t3/1.0.0";

        // t1: 3 peers (needs repair, never repaired)
        coord.record_peer_count(t1, 3, 0);
        // t2: 8 peers (healthy, no repair needed)
        coord.record_peer_count(t2, 8, 0);
        // t3: 2 peers but repaired very recently → cooldown blocks
        coord.record_peer_count(t3, 2, 0);
        coord.record_repair(t3, 99_000);

        let needing = coord.topics_needing_repair(100_000); // only 1 ms elapsed for t3
        assert_eq!(needing.len(), 1);
        assert_eq!(needing[0], t1);
    }

    #[test]
    fn test_topics_needing_prune() {
        let coord = default_coordinator();
        let t1 = "/ipfrs/t1/1.0.0";
        let t2 = "/ipfrs/t2/1.0.0";

        coord.record_peer_count(t1, 15, 0); // > d_high (12) → prune
        coord.record_peer_count(t2, 10, 0); // ≤ d_high → fine

        let needing = coord.topics_needing_prune();
        assert_eq!(needing.len(), 1);
        assert_eq!(needing[0], t1);
    }

    // ── Additional correctness checks ─────────────────────────────────────

    #[test]
    fn test_no_prune_when_at_threshold() {
        let coord = default_coordinator();
        coord.record_peer_count(T, 12, 0); // exactly d_high → no prune
        assert!(!coord.needs_prune(T));
    }

    #[test]
    fn test_no_repair_at_d_low_exactly() {
        let coord = default_coordinator();
        coord.record_peer_count(T, 6, 0); // exactly d_low → no repair
        assert!(!coord.needs_repair(T, 10_000));
    }

    #[test]
    fn test_record_prune_increments_count() {
        let coord = default_coordinator();
        coord.record_peer_count(T, 14, 0);
        coord.record_prune(T);
        coord.record_prune(T);
        let state = coord.repair_state(T).expect("state present");
        assert_eq!(state.prune_count, 2);
    }

    #[test]
    fn test_all_states_returns_all_topics() {
        let coord = default_coordinator();
        coord.record_peer_count("/t1", 5, 0);
        coord.record_peer_count("/t2", 8, 0);
        coord.record_peer_count("/t3", 14, 0);
        assert_eq!(coord.all_states().len(), 3);
    }
}
