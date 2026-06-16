//! Peer Health Checker
//!
//! Tracks peer health through periodic heartbeat monitoring, computes health scores,
//! and manages peer promotion/demotion between health tiers.
//!
//! # Examples
//!
//! ```rust
//! use ipfrs_network::health_checker::{PeerHealthChecker, PeerHealthCheckerConfig, HealthTier};
//!
//! let config = PeerHealthCheckerConfig::default();
//! let mut checker = PeerHealthChecker::new(config);
//!
//! checker.record_heartbeat("peer-1", 100);
//! assert_eq!(checker.tier("peer-1"), HealthTier::Healthy);
//!
//! checker.record_miss("peer-1");
//! checker.record_miss("peer-1");
//! assert_eq!(checker.tier("peer-1"), HealthTier::Degraded);
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// HealthTier
// ---------------------------------------------------------------------------

/// Classification of a peer's health based on consecutive missed heartbeats.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HealthTier {
    /// Peer is responding normally.
    Healthy = 0,
    /// Peer has missed a small number of heartbeats.
    Degraded = 1,
    /// Peer has missed many heartbeats but is not yet considered dead.
    Unhealthy = 2,
    /// Peer has not sent a heartbeat for too long; assumed offline.
    Dead = 3,
}

// ---------------------------------------------------------------------------
// HeartbeatRecord
// ---------------------------------------------------------------------------

/// Accumulated heartbeat statistics for a single peer.
#[derive(Clone, Debug)]
pub struct HeartbeatRecord {
    /// Peer identifier.
    pub peer_id: String,
    /// Logical tick at which the most recent heartbeat was received.
    pub last_seen_tick: u64,
    /// Number of consecutive heartbeat periods with no response.
    pub consecutive_misses: u32,
    /// Total heartbeats successfully recorded.
    pub total_heartbeats: u64,
    /// Total heartbeat periods with no response.
    pub total_misses: u64,
}

impl HeartbeatRecord {
    /// Creates a new record for `peer_id` with zero statistics.
    fn new(peer_id: &str) -> Self {
        Self {
            peer_id: peer_id.to_owned(),
            last_seen_tick: 0,
            consecutive_misses: 0,
            total_heartbeats: 0,
            total_misses: 0,
        }
    }

    /// Fraction of periods in which no heartbeat was observed.
    ///
    /// Returns `0.0` when no data has been collected yet.
    pub fn miss_rate(&self) -> f64 {
        let total = self.total_heartbeats + self.total_misses;
        if total == 0 {
            0.0
        } else {
            self.total_misses as f64 / total as f64
        }
    }

    /// Composite health score in `[0.0, 1.0]`; higher means healthier.
    ///
    /// Defined as `1.0 - miss_rate()`.
    pub fn health_score(&self) -> f64 {
        1.0 - self.miss_rate()
    }
}

// ---------------------------------------------------------------------------
// PeerHealthCheckerConfig
// ---------------------------------------------------------------------------

/// Configuration for the [`PeerHealthChecker`].
#[derive(Clone, Debug)]
pub struct PeerHealthCheckerConfig {
    /// Expected number of ticks between successive heartbeats from a peer.
    pub heartbeat_interval_ticks: u64,
    /// Consecutive misses threshold for transitioning to [`HealthTier::Degraded`].
    pub degraded_miss_threshold: u32,
    /// Consecutive misses threshold for transitioning to [`HealthTier::Unhealthy`].
    pub unhealthy_miss_threshold: u32,
    /// Consecutive misses threshold for transitioning to [`HealthTier::Dead`].
    pub dead_miss_threshold: u32,
}

impl Default for PeerHealthCheckerConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval_ticks: 30,
            degraded_miss_threshold: 2,
            unhealthy_miss_threshold: 5,
            dead_miss_threshold: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// HealthCheckerStats
// ---------------------------------------------------------------------------

/// Aggregate statistics reported by [`PeerHealthChecker::stats`].
#[derive(Clone, Debug)]
pub struct HealthCheckerStats {
    /// Total number of tracked peers.
    pub total_peers: usize,
    /// Number of peers in each health tier.
    pub by_tier: HashMap<HealthTier, usize>,
    /// Cumulative heartbeats received across all peers.
    pub total_heartbeats_received: u64,
    /// Cumulative missed heartbeat periods across all peers.
    pub total_misses_recorded: u64,
}

// ---------------------------------------------------------------------------
// PeerHealthChecker
// ---------------------------------------------------------------------------

/// Tracks peer health through periodic heartbeat monitoring.
///
/// Maintains a [`HeartbeatRecord`] for every known peer and classifies each
/// peer into a [`HealthTier`] based on the number of consecutive missed
/// heartbeats.  The [`tick_check`](PeerHealthChecker::tick_check) method
/// should be called once per tick to automatically promote peers that have
/// gone silent.
pub struct PeerHealthChecker {
    /// Heartbeat records keyed by peer identifier.
    pub records: HashMap<String, HeartbeatRecord>,
    /// Checker configuration.
    pub config: PeerHealthCheckerConfig,
}

impl PeerHealthChecker {
    /// Creates a new checker with the supplied `config`.
    pub fn new(config: PeerHealthCheckerConfig) -> Self {
        Self {
            records: HashMap::new(),
            config,
        }
    }

    /// Records a successful heartbeat from `peer_id` at `current_tick`.
    ///
    /// If no record exists for the peer, one is created automatically.  The
    /// record's `last_seen_tick` is updated to `current_tick`,
    /// `consecutive_misses` is reset to `0`, and `total_heartbeats` is
    /// incremented.
    pub fn record_heartbeat(&mut self, peer_id: &str, current_tick: u64) {
        let record = self
            .records
            .entry(peer_id.to_owned())
            .or_insert_with(|| HeartbeatRecord::new(peer_id));
        record.last_seen_tick = current_tick;
        record.consecutive_misses = 0;
        record.total_heartbeats += 1;
    }

    /// Records a missed heartbeat period for `peer_id`.
    ///
    /// If no record exists for the peer, one is created automatically.
    /// `consecutive_misses` and `total_misses` are both incremented.
    pub fn record_miss(&mut self, peer_id: &str) {
        let record = self
            .records
            .entry(peer_id.to_owned())
            .or_insert_with(|| HeartbeatRecord::new(peer_id));
        record.consecutive_misses += 1;
        record.total_misses += 1;
    }

    /// Returns the current [`HealthTier`] for `peer_id`.
    ///
    /// A peer that has never been seen (i.e. has no record) is considered
    /// [`HealthTier::Dead`].
    pub fn tier(&self, peer_id: &str) -> HealthTier {
        match self.records.get(peer_id) {
            None => HealthTier::Dead,
            Some(rec) => {
                if rec.consecutive_misses >= self.config.dead_miss_threshold {
                    HealthTier::Dead
                } else if rec.consecutive_misses >= self.config.unhealthy_miss_threshold {
                    HealthTier::Unhealthy
                } else if rec.consecutive_misses >= self.config.degraded_miss_threshold {
                    HealthTier::Degraded
                } else {
                    HealthTier::Healthy
                }
            }
        }
    }

    /// Advances the internal clock to `current_tick` and records a miss for
    /// every peer whose `last_seen_tick` is more than
    /// `heartbeat_interval_ticks` ticks in the past.
    ///
    /// Returns the peer IDs that had a miss recorded, sorted alphabetically.
    pub fn tick_check(&mut self, current_tick: u64) -> Vec<String> {
        let interval = self.config.heartbeat_interval_ticks;

        // Collect the IDs of peers that are stale.
        let stale: Vec<String> = self
            .records
            .values()
            .filter(|rec| current_tick.saturating_sub(rec.last_seen_tick) > interval)
            .map(|rec| rec.peer_id.clone())
            .collect();

        for peer_id in &stale {
            // SAFETY: key was collected from self.records so it exists.
            if let Some(rec) = self.records.get_mut(peer_id) {
                rec.consecutive_misses += 1;
                rec.total_misses += 1;
            }
        }

        let mut sorted = stale;
        sorted.sort_unstable();
        sorted
    }

    /// Removes the record for `peer_id`.
    ///
    /// Returns `true` if a record existed and was removed, `false` otherwise.
    pub fn remove_peer(&mut self, peer_id: &str) -> bool {
        self.records.remove(peer_id).is_some()
    }

    /// Returns the health score for `peer_id`, or `0.0` if the peer is
    /// unknown.
    pub fn health_score(&self, peer_id: &str) -> f64 {
        self.records
            .get(peer_id)
            .map(|rec| rec.health_score())
            .unwrap_or(0.0)
    }

    /// Returns the IDs of all peers currently in the [`HealthTier::Healthy`]
    /// tier, sorted alphabetically.
    pub fn healthy_peers(&self) -> Vec<&str> {
        let mut peers: Vec<&str> = self
            .records
            .keys()
            .filter(|id| self.tier(id) == HealthTier::Healthy)
            .map(|s| s.as_str())
            .collect();
        peers.sort_unstable();
        peers
    }

    /// Returns aggregate statistics for the checker.
    pub fn stats(&self) -> HealthCheckerStats {
        let mut by_tier: HashMap<HealthTier, usize> = HashMap::new();
        let mut total_heartbeats_received: u64 = 0;
        let mut total_misses_recorded: u64 = 0;

        for rec in self.records.values() {
            *by_tier.entry(self.tier(&rec.peer_id)).or_insert(0) += 1;
            total_heartbeats_received += rec.total_heartbeats;
            total_misses_recorded += rec.total_misses;
        }

        HealthCheckerStats {
            total_peers: self.records.len(),
            by_tier,
            total_heartbeats_received,
            total_misses_recorded,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_checker() -> PeerHealthChecker {
        PeerHealthChecker::new(PeerHealthCheckerConfig::default())
    }

    // --- HeartbeatRecord ---

    #[test]
    fn miss_rate_zero_when_no_data() {
        let rec = HeartbeatRecord::new("p");
        assert_eq!(rec.miss_rate(), 0.0);
    }

    #[test]
    fn miss_rate_calculation() {
        let mut rec = HeartbeatRecord::new("p");
        rec.total_heartbeats = 3;
        rec.total_misses = 1;
        // 1 / (3 + 1) = 0.25
        let diff = (rec.miss_rate() - 0.25_f64).abs();
        assert!(
            diff < 1e-10,
            "miss_rate should be 0.25, got {}",
            rec.miss_rate()
        );
    }

    #[test]
    fn health_score_equals_one_minus_miss_rate() {
        let mut rec = HeartbeatRecord::new("p");
        rec.total_heartbeats = 3;
        rec.total_misses = 1;
        let expected = 1.0 - rec.miss_rate();
        let diff = (rec.health_score() - expected).abs();
        assert!(diff < 1e-10);
    }

    #[test]
    fn health_score_is_one_when_no_misses() {
        let mut rec = HeartbeatRecord::new("p");
        rec.total_heartbeats = 10;
        let diff = (rec.health_score() - 1.0_f64).abs();
        assert!(diff < 1e-10);
    }

    #[test]
    fn health_score_is_one_when_no_data() {
        let rec = HeartbeatRecord::new("p");
        let diff = (rec.health_score() - 1.0_f64).abs();
        assert!(
            diff < 1e-10,
            "empty record should have score 1.0, got {}",
            rec.health_score()
        );
    }

    // --- record_heartbeat ---

    #[test]
    fn record_heartbeat_creates_record() {
        let mut checker = default_checker();
        checker.record_heartbeat("peer-1", 50);
        assert!(checker.records.contains_key("peer-1"));
    }

    #[test]
    fn record_heartbeat_updates_last_seen_tick() {
        let mut checker = default_checker();
        checker.record_heartbeat("peer-1", 42);
        assert_eq!(checker.records["peer-1"].last_seen_tick, 42);
    }

    #[test]
    fn record_heartbeat_resets_consecutive_misses() {
        let mut checker = default_checker();
        checker.record_miss("peer-1");
        checker.record_miss("peer-1");
        checker.record_heartbeat("peer-1", 100);
        assert_eq!(checker.records["peer-1"].consecutive_misses, 0);
    }

    #[test]
    fn record_heartbeat_increments_total_heartbeats() {
        let mut checker = default_checker();
        checker.record_heartbeat("peer-1", 10);
        checker.record_heartbeat("peer-1", 20);
        assert_eq!(checker.records["peer-1"].total_heartbeats, 2);
    }

    // --- record_miss ---

    #[test]
    fn record_miss_creates_record() {
        let mut checker = default_checker();
        checker.record_miss("peer-2");
        assert!(checker.records.contains_key("peer-2"));
    }

    #[test]
    fn record_miss_increments_consecutive_misses() {
        let mut checker = default_checker();
        checker.record_miss("peer-2");
        checker.record_miss("peer-2");
        assert_eq!(checker.records["peer-2"].consecutive_misses, 2);
    }

    #[test]
    fn record_miss_increments_total_misses() {
        let mut checker = default_checker();
        checker.record_miss("peer-2");
        checker.record_miss("peer-2");
        assert_eq!(checker.records["peer-2"].total_misses, 2);
    }

    // --- tier ---

    #[test]
    fn tier_dead_for_missing_peer() {
        let checker = default_checker();
        assert_eq!(checker.tier("unknown"), HealthTier::Dead);
    }

    #[test]
    fn tier_healthy_on_zero_misses() {
        let mut checker = default_checker();
        checker.record_heartbeat("peer-1", 10);
        assert_eq!(checker.tier("peer-1"), HealthTier::Healthy);
    }

    #[test]
    fn tier_degraded_at_threshold() {
        let mut checker = default_checker();
        checker.record_heartbeat("peer-1", 10);
        // default degraded_miss_threshold = 2
        checker.record_miss("peer-1");
        checker.record_miss("peer-1");
        assert_eq!(checker.tier("peer-1"), HealthTier::Degraded);
    }

    #[test]
    fn tier_unhealthy_at_threshold() {
        let mut checker = default_checker();
        checker.record_heartbeat("peer-1", 10);
        // default unhealthy_miss_threshold = 5
        for _ in 0..5 {
            checker.record_miss("peer-1");
        }
        assert_eq!(checker.tier("peer-1"), HealthTier::Unhealthy);
    }

    #[test]
    fn tier_dead_at_threshold() {
        let mut checker = default_checker();
        checker.record_heartbeat("peer-1", 10);
        // default dead_miss_threshold = 10
        for _ in 0..10 {
            checker.record_miss("peer-1");
        }
        assert_eq!(checker.tier("peer-1"), HealthTier::Dead);
    }

    #[test]
    fn tier_healthy_after_heartbeat_resets_misses() {
        let mut checker = default_checker();
        for _ in 0..8 {
            checker.record_miss("peer-1");
        }
        checker.record_heartbeat("peer-1", 500);
        assert_eq!(checker.tier("peer-1"), HealthTier::Healthy);
    }

    // --- tick_check ---

    #[test]
    fn tick_check_detects_stale_peers() {
        let mut checker = default_checker();
        checker.record_heartbeat("peer-a", 0);
        // tick 0 + interval(30) + 1 = 31 → stale
        let missed = checker.tick_check(31);
        assert!(missed.contains(&"peer-a".to_owned()));
    }

    #[test]
    fn tick_check_does_not_flag_recent_peers() {
        let mut checker = default_checker();
        checker.record_heartbeat("peer-b", 20);
        // Only 10 ticks elapsed, interval is 30 → not stale
        let missed = checker.tick_check(30);
        assert!(!missed.contains(&"peer-b".to_owned()));
    }

    #[test]
    fn tick_check_returns_sorted_alphabetically() {
        let mut checker = default_checker();
        checker.record_heartbeat("peer-z", 0);
        checker.record_heartbeat("peer-a", 0);
        checker.record_heartbeat("peer-m", 0);
        let missed = checker.tick_check(100);
        let mut expected = missed.clone();
        expected.sort();
        assert_eq!(missed, expected);
    }

    #[test]
    fn tick_check_increments_consecutive_misses_for_stale_peer() {
        let mut checker = default_checker();
        checker.record_heartbeat("peer-c", 0);
        checker.tick_check(100);
        assert!(checker.records["peer-c"].consecutive_misses > 0);
    }

    #[test]
    fn tick_check_returns_empty_when_no_peers_are_stale() {
        let mut checker = default_checker();
        checker.record_heartbeat("peer-d", 100);
        let missed = checker.tick_check(110);
        assert!(missed.is_empty());
    }

    // --- remove_peer ---

    #[test]
    fn remove_peer_returns_true_when_present() {
        let mut checker = default_checker();
        checker.record_heartbeat("peer-1", 10);
        assert!(checker.remove_peer("peer-1"));
    }

    #[test]
    fn remove_peer_returns_false_when_absent() {
        let mut checker = default_checker();
        assert!(!checker.remove_peer("ghost"));
    }

    #[test]
    fn remove_peer_actually_removes_record() {
        let mut checker = default_checker();
        checker.record_heartbeat("peer-1", 10);
        checker.remove_peer("peer-1");
        assert!(!checker.records.contains_key("peer-1"));
    }

    // --- health_score ---

    #[test]
    fn health_score_returns_zero_for_unknown_peer() {
        let checker = default_checker();
        let diff = checker.health_score("unknown").abs();
        assert!(diff < 1e-10);
    }

    #[test]
    fn health_score_returns_record_score() {
        let mut checker = default_checker();
        checker.record_heartbeat("peer-1", 10);
        checker.record_heartbeat("peer-1", 20);
        checker.record_miss("peer-1");
        // 1 miss out of 3 total → miss_rate = 1/3 → score ≈ 0.667
        let score = checker.health_score("peer-1");
        let expected = checker.records["peer-1"].health_score();
        let diff = (score - expected).abs();
        assert!(diff < 1e-10);
    }

    // --- healthy_peers ---

    #[test]
    fn healthy_peers_sorted_alphabetically() {
        let mut checker = default_checker();
        checker.record_heartbeat("peer-z", 10);
        checker.record_heartbeat("peer-a", 10);
        checker.record_heartbeat("peer-m", 10);
        let peers = checker.healthy_peers();
        let mut expected = peers.clone();
        expected.sort_unstable();
        assert_eq!(peers, expected);
    }

    #[test]
    fn healthy_peers_excludes_degraded_peers() {
        let mut checker = default_checker();
        checker.record_heartbeat("peer-good", 10);
        checker.record_heartbeat("peer-bad", 10);
        checker.record_miss("peer-bad");
        checker.record_miss("peer-bad");
        let peers = checker.healthy_peers();
        assert!(peers.contains(&"peer-good"));
        assert!(!peers.contains(&"peer-bad"));
    }

    // --- stats ---

    #[test]
    fn stats_total_peers_count() {
        let mut checker = default_checker();
        checker.record_heartbeat("p1", 1);
        checker.record_heartbeat("p2", 1);
        let s = checker.stats();
        assert_eq!(s.total_peers, 2);
    }

    #[test]
    fn stats_by_tier_counts_healthy_and_degraded() {
        let mut checker = default_checker();
        checker.record_heartbeat("healthy", 10);
        checker.record_heartbeat("degraded", 10);
        checker.record_miss("degraded");
        checker.record_miss("degraded");
        let s = checker.stats();
        assert_eq!(s.by_tier.get(&HealthTier::Healthy).copied().unwrap_or(0), 1);
        assert_eq!(
            s.by_tier.get(&HealthTier::Degraded).copied().unwrap_or(0),
            1
        );
    }

    #[test]
    fn stats_total_heartbeats_received() {
        let mut checker = default_checker();
        checker.record_heartbeat("p1", 1);
        checker.record_heartbeat("p1", 2);
        checker.record_heartbeat("p2", 1);
        let s = checker.stats();
        assert_eq!(s.total_heartbeats_received, 3);
    }

    #[test]
    fn stats_total_misses_recorded() {
        let mut checker = default_checker();
        checker.record_heartbeat("p1", 1);
        checker.record_miss("p1");
        checker.record_miss("p1");
        checker.record_miss("p2");
        let s = checker.stats();
        assert_eq!(s.total_misses_recorded, 3);
    }

    #[test]
    fn stats_by_tier_dead_count() {
        let mut checker = default_checker();
        checker.record_heartbeat("p1", 10);
        for _ in 0..10 {
            checker.record_miss("p1");
        }
        let s = checker.stats();
        assert_eq!(s.by_tier.get(&HealthTier::Dead).copied().unwrap_or(0), 1);
    }
}
