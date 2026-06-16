//! Peer churn tracking and stability scoring for DHT and routing decisions.
//!
//! This module provides [`PeerChurnManager`] which tracks peer join/leave events,
//! computes churn rate over sliding windows, and maintains stability scores to
//! inform DHT and routing decisions.

use std::collections::HashMap;

/// An event representing a peer joining or leaving the network.
#[derive(Clone, Debug, PartialEq)]
pub enum ChurnEvent {
    /// A peer joined at the given tick.
    Joined {
        /// The peer's identifier.
        peer_id: String,
        /// The logical clock tick when the peer joined.
        tick: u64,
    },
    /// A peer left at the given tick.
    Left {
        /// The peer's identifier.
        peer_id: String,
        /// The logical clock tick when the peer left.
        tick: u64,
    },
}

impl ChurnEvent {
    /// Returns the tick associated with this event.
    pub fn tick(&self) -> u64 {
        match self {
            ChurnEvent::Joined { tick, .. } => *tick,
            ChurnEvent::Left { tick, .. } => *tick,
        }
    }

    /// Returns the peer_id associated with this event.
    pub fn peer_id(&self) -> &str {
        match self {
            ChurnEvent::Joined { peer_id, .. } => peer_id,
            ChurnEvent::Left { peer_id, .. } => peer_id,
        }
    }
}

/// Records the lifetime of a single peer in the network.
#[derive(Clone, Debug, PartialEq)]
pub struct PeerLifetime {
    /// The peer's unique identifier.
    pub peer_id: String,
    /// The tick at which this peer joined.
    pub joined_tick: u64,
    /// The tick at which this peer left, or `None` if still online.
    pub left_tick: Option<u64>,
}

impl PeerLifetime {
    /// Returns how long this peer has been (or was) connected.
    ///
    /// If the peer is still online, `current_tick` is used as the end boundary.
    pub fn duration(&self, current_tick: u64) -> u64 {
        self.left_tick
            .unwrap_or(current_tick)
            .saturating_sub(self.joined_tick)
    }

    /// Returns `true` if the peer is currently online (has not left).
    pub fn is_online(&self) -> bool {
        self.left_tick.is_none()
    }
}

/// A sliding time window used to compute churn rate.
#[derive(Clone, Debug, PartialEq)]
pub struct ChurnWindow {
    /// The inclusive start tick of this window.
    pub tick_start: u64,
    /// The inclusive end tick of this window.
    pub tick_end: u64,
    /// Number of join events within the window.
    pub joins: u32,
    /// Number of leave events within the window.
    pub leaves: u32,
}

impl ChurnWindow {
    /// Computes the churn rate as `(joins + leaves) / window_width`.
    ///
    /// Returns `0.0` for zero-width windows.
    pub fn churn_rate(&self) -> f64 {
        if self.tick_end == self.tick_start {
            return 0.0;
        }
        let events = (self.joins as f64) + (self.leaves as f64);
        let width = (self.tick_end - self.tick_start + 1) as f64;
        events / width
    }
}

/// Configuration for the [`PeerChurnManager`].
#[derive(Clone, Debug, PartialEq)]
pub struct ChurnManagerConfig {
    /// Width of the sliding window in ticks (default: 100).
    pub window_size_ticks: u64,
    /// Churn rate below this threshold is considered stable (default: 0.1).
    pub stability_threshold: f64,
}

impl Default for ChurnManagerConfig {
    fn default() -> Self {
        Self {
            window_size_ticks: 100,
            stability_threshold: 0.1,
        }
    }
}

/// Aggregated statistics for the churn manager at a given tick.
#[derive(Clone, Debug, PartialEq)]
pub struct ChurnStats {
    /// Total number of join events recorded since creation.
    pub total_joins: u64,
    /// Total number of leave events recorded since creation.
    pub total_leaves: u64,
    /// Number of peers currently online.
    pub current_online: usize,
    /// Mean lifetime of all peers (online peers measured to `current_tick`).
    pub avg_lifetime_ticks: f64,
    /// Churn rate from the most recent sliding window.
    pub churn_rate: f64,
    /// Whether the churn rate is below the stability threshold.
    pub is_stable: bool,
}

/// Manages peer join/leave events, computes churn rate over sliding windows,
/// and provides stability scores for DHT and routing decisions.
pub struct PeerChurnManager {
    /// Per-peer lifetime records, keyed by peer ID.
    pub lifetimes: HashMap<String, PeerLifetime>,
    /// All events recorded in chronological order.
    pub events: Vec<ChurnEvent>,
    /// Configuration governing window size and stability threshold.
    pub config: ChurnManagerConfig,
}

impl PeerChurnManager {
    /// Creates a new [`PeerChurnManager`] with the given configuration.
    pub fn new(config: ChurnManagerConfig) -> Self {
        Self {
            lifetimes: HashMap::new(),
            events: Vec::new(),
            config,
        }
    }

    /// Records a churn event and updates internal state accordingly.
    ///
    /// - `Joined`: Creates (or replaces, if the peer was previously offline) a
    ///   [`PeerLifetime`] record.
    /// - `Left`: If the peer is currently online, marks the peer as offline by
    ///   setting `left_tick`. If the peer is unknown or already offline, this is
    ///   a no-op beyond storing the event.
    pub fn record_event(&mut self, event: ChurnEvent) {
        match &event {
            ChurnEvent::Joined { peer_id, tick } => {
                let lifetime = PeerLifetime {
                    peer_id: peer_id.clone(),
                    joined_tick: *tick,
                    left_tick: None,
                };
                self.lifetimes.insert(peer_id.clone(), lifetime);
            }
            ChurnEvent::Left { peer_id, tick } => {
                if let Some(lifetime) = self.lifetimes.get_mut(peer_id) {
                    if lifetime.is_online() {
                        lifetime.left_tick = Some(*tick);
                    }
                }
            }
        }
        self.events.push(event);
    }

    /// Builds the current sliding window ending at `current_tick`.
    ///
    /// The window spans `[current_tick - window_size_ticks, current_tick]`
    /// (saturating at 0), and counts join and leave events within that range.
    pub fn current_window(&self, current_tick: u64) -> ChurnWindow {
        let tick_start = current_tick.saturating_sub(self.config.window_size_ticks);
        let tick_end = current_tick;

        let mut joins: u32 = 0;
        let mut leaves: u32 = 0;

        for event in &self.events {
            let t = event.tick();
            if t >= tick_start && t <= tick_end {
                match event {
                    ChurnEvent::Joined { .. } => joins += 1,
                    ChurnEvent::Left { .. } => leaves += 1,
                }
            }
        }

        ChurnWindow {
            tick_start,
            tick_end,
            joins,
            leaves,
        }
    }

    /// Computes a stability score in `[0.0, 1.0]`.
    ///
    /// `1.0` means no churn; `0.0` means maximum churn (rate ≥ 1.0).
    pub fn stability_score(&self, current_tick: u64) -> f64 {
        let window = self.current_window(current_tick);
        let rate = window.churn_rate();
        1.0 - rate.min(1.0)
    }

    /// Returns the peer IDs of all currently online peers, sorted alphabetically.
    pub fn online_peers(&self) -> Vec<&str> {
        let mut peers: Vec<&str> = self
            .lifetimes
            .values()
            .filter(|l| l.is_online())
            .map(|l| l.peer_id.as_str())
            .collect();
        peers.sort_unstable();
        peers
    }

    /// Returns a reference to the [`PeerLifetime`] for a given peer ID, if any.
    pub fn peer_lifetime(&self, peer_id: &str) -> Option<&PeerLifetime> {
        self.lifetimes.get(peer_id)
    }

    /// Computes aggregated [`ChurnStats`] at the given tick.
    pub fn stats(&self, current_tick: u64) -> ChurnStats {
        let mut total_joins: u64 = 0;
        let mut total_leaves: u64 = 0;

        for event in &self.events {
            match event {
                ChurnEvent::Joined { .. } => total_joins += 1,
                ChurnEvent::Left { .. } => total_leaves += 1,
            }
        }

        let current_online = self.lifetimes.values().filter(|l| l.is_online()).count();

        let avg_lifetime_ticks = if self.lifetimes.is_empty() {
            0.0
        } else {
            let total: u64 = self
                .lifetimes
                .values()
                .map(|l| l.duration(current_tick))
                .sum();
            total as f64 / self.lifetimes.len() as f64
        };

        let window = self.current_window(current_tick);
        let churn_rate = window.churn_rate();
        let is_stable = churn_rate < self.config.stability_threshold;

        ChurnStats {
            total_joins,
            total_leaves,
            current_online,
            avg_lifetime_ticks,
            churn_rate,
            is_stable,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_manager() -> PeerChurnManager {
        PeerChurnManager::new(ChurnManagerConfig::default())
    }

    // ── ChurnEvent helpers ────────────────────────────────────────────────────

    #[test]
    fn test_churn_event_tick_joined() {
        let e = ChurnEvent::Joined {
            peer_id: "peer1".into(),
            tick: 42,
        };
        assert_eq!(e.tick(), 42);
    }

    #[test]
    fn test_churn_event_tick_left() {
        let e = ChurnEvent::Left {
            peer_id: "peer1".into(),
            tick: 77,
        };
        assert_eq!(e.tick(), 77);
    }

    #[test]
    fn test_churn_event_peer_id() {
        let e = ChurnEvent::Joined {
            peer_id: "alpha".into(),
            tick: 1,
        };
        assert_eq!(e.peer_id(), "alpha");
    }

    #[test]
    fn test_churn_event_clone_and_eq() {
        let e = ChurnEvent::Left {
            peer_id: "x".into(),
            tick: 5,
        };
        assert_eq!(e.clone(), e);
    }

    // ── PeerLifetime ──────────────────────────────────────────────────────────

    #[test]
    fn test_peer_lifetime_is_online_when_left_tick_none() {
        let l = PeerLifetime {
            peer_id: "p".into(),
            joined_tick: 10,
            left_tick: None,
        };
        assert!(l.is_online());
    }

    #[test]
    fn test_peer_lifetime_is_not_online_when_left_tick_set() {
        let l = PeerLifetime {
            peer_id: "p".into(),
            joined_tick: 10,
            left_tick: Some(20),
        };
        assert!(!l.is_online());
    }

    #[test]
    fn test_peer_lifetime_duration_online() {
        let l = PeerLifetime {
            peer_id: "p".into(),
            joined_tick: 5,
            left_tick: None,
        };
        assert_eq!(l.duration(15), 10);
    }

    #[test]
    fn test_peer_lifetime_duration_offline() {
        let l = PeerLifetime {
            peer_id: "p".into(),
            joined_tick: 5,
            left_tick: Some(12),
        };
        assert_eq!(l.duration(999), 7);
    }

    #[test]
    fn test_peer_lifetime_duration_zero_when_same_tick() {
        let l = PeerLifetime {
            peer_id: "p".into(),
            joined_tick: 10,
            left_tick: Some(10),
        };
        assert_eq!(l.duration(10), 0);
    }

    // ── ChurnWindow ───────────────────────────────────────────────────────────

    #[test]
    fn test_churn_window_churn_rate_basic() {
        let w = ChurnWindow {
            tick_start: 0,
            tick_end: 9,
            joins: 3,
            leaves: 2,
        };
        // (3+2) / (9-0+1) = 5/10 = 0.5
        assert!((w.churn_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_churn_window_churn_rate_zero_width() {
        let w = ChurnWindow {
            tick_start: 5,
            tick_end: 5,
            joins: 10,
            leaves: 10,
        };
        assert_eq!(w.churn_rate(), 0.0);
    }

    #[test]
    fn test_churn_window_churn_rate_no_events() {
        let w = ChurnWindow {
            tick_start: 0,
            tick_end: 99,
            joins: 0,
            leaves: 0,
        };
        assert_eq!(w.churn_rate(), 0.0);
    }

    // ── ChurnManagerConfig ────────────────────────────────────────────────────

    #[test]
    fn test_churn_manager_config_default() {
        let cfg = ChurnManagerConfig::default();
        assert_eq!(cfg.window_size_ticks, 100);
        assert!((cfg.stability_threshold - 0.1).abs() < f64::EPSILON);
    }

    // ── record_event: Joined ──────────────────────────────────────────────────

    #[test]
    fn test_record_event_joined_creates_lifetime() {
        let mut mgr = default_manager();
        mgr.record_event(ChurnEvent::Joined {
            peer_id: "peer1".into(),
            tick: 10,
        });
        let l = mgr.peer_lifetime("peer1").expect("lifetime should exist");
        assert_eq!(l.joined_tick, 10);
        assert!(l.is_online());
    }

    #[test]
    fn test_record_event_joined_stores_event() {
        let mut mgr = default_manager();
        mgr.record_event(ChurnEvent::Joined {
            peer_id: "peer1".into(),
            tick: 1,
        });
        assert_eq!(mgr.events.len(), 1);
    }

    #[test]
    fn test_record_event_joined_replaces_offline_peer() {
        let mut mgr = default_manager();
        mgr.record_event(ChurnEvent::Joined {
            peer_id: "peer1".into(),
            tick: 5,
        });
        mgr.record_event(ChurnEvent::Left {
            peer_id: "peer1".into(),
            tick: 10,
        });
        // Peer rejoins
        mgr.record_event(ChurnEvent::Joined {
            peer_id: "peer1".into(),
            tick: 20,
        });
        let l = mgr.peer_lifetime("peer1").expect("lifetime should exist");
        assert_eq!(l.joined_tick, 20);
        assert!(l.is_online());
    }

    // ── record_event: Left ────────────────────────────────────────────────────

    #[test]
    fn test_record_event_left_sets_left_tick() {
        let mut mgr = default_manager();
        mgr.record_event(ChurnEvent::Joined {
            peer_id: "peer1".into(),
            tick: 0,
        });
        mgr.record_event(ChurnEvent::Left {
            peer_id: "peer1".into(),
            tick: 50,
        });
        let l = mgr.peer_lifetime("peer1").expect("lifetime should exist");
        assert_eq!(l.left_tick, Some(50));
        assert!(!l.is_online());
    }

    #[test]
    fn test_record_event_left_unknown_peer_is_noop() {
        let mut mgr = default_manager();
        // No join first — Left for unknown peer should not panic or insert an entry
        mgr.record_event(ChurnEvent::Left {
            peer_id: "ghost".into(),
            tick: 5,
        });
        assert!(mgr.peer_lifetime("ghost").is_none());
        // Event is still stored
        assert_eq!(mgr.events.len(), 1);
    }

    #[test]
    fn test_record_event_left_already_offline_noop() {
        let mut mgr = default_manager();
        mgr.record_event(ChurnEvent::Joined {
            peer_id: "p".into(),
            tick: 0,
        });
        mgr.record_event(ChurnEvent::Left {
            peer_id: "p".into(),
            tick: 10,
        });
        // Second leave should not change left_tick
        mgr.record_event(ChurnEvent::Left {
            peer_id: "p".into(),
            tick: 99,
        });
        let l = mgr.peer_lifetime("p").expect("lifetime should exist");
        assert_eq!(l.left_tick, Some(10));
    }

    // ── current_window ────────────────────────────────────────────────────────

    #[test]
    fn test_current_window_counts_joins_and_leaves() {
        let mut mgr = default_manager();
        // Events inside window [900, 1000]
        mgr.record_event(ChurnEvent::Joined {
            peer_id: "a".into(),
            tick: 950,
        });
        mgr.record_event(ChurnEvent::Joined {
            peer_id: "b".into(),
            tick: 960,
        });
        mgr.record_event(ChurnEvent::Left {
            peer_id: "a".into(),
            tick: 970,
        });
        // Event outside window (before start)
        mgr.record_event(ChurnEvent::Joined {
            peer_id: "c".into(),
            tick: 899,
        });

        let w = mgr.current_window(1000);
        assert_eq!(w.joins, 2);
        assert_eq!(w.leaves, 1);
        assert_eq!(w.tick_start, 900);
        assert_eq!(w.tick_end, 1000);
    }

    #[test]
    fn test_current_window_includes_boundaries() {
        let mut mgr = default_manager();
        mgr.record_event(ChurnEvent::Joined {
            peer_id: "x".into(),
            tick: 0,
        });
        mgr.record_event(ChurnEvent::Left {
            peer_id: "x".into(),
            tick: 100,
        });

        // window_size_ticks=100, so window = [0, 100]
        let w = mgr.current_window(100);
        assert_eq!(w.joins, 1);
        assert_eq!(w.leaves, 1);
    }

    #[test]
    fn test_current_window_saturates_at_zero() {
        let mgr = default_manager();
        let w = mgr.current_window(50);
        assert_eq!(w.tick_start, 0); // 50 - 100 saturates to 0
    }

    // ── stability_score ───────────────────────────────────────────────────────

    #[test]
    fn test_stability_score_no_events() {
        let mgr = default_manager();
        // No events → churn_rate = 0 → stability = 1.0
        assert!((mgr.stability_score(100) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stability_score_high_churn_clamped_to_zero() {
        let mut mgr = PeerChurnManager::new(ChurnManagerConfig {
            window_size_ticks: 1,
            stability_threshold: 0.1,
        });
        // 4 events in a 2-tick window → rate = 4/2 = 2.0, clamped to 1.0 → score = 0.0
        mgr.record_event(ChurnEvent::Joined {
            peer_id: "a".into(),
            tick: 0,
        });
        mgr.record_event(ChurnEvent::Left {
            peer_id: "a".into(),
            tick: 0,
        });
        mgr.record_event(ChurnEvent::Joined {
            peer_id: "b".into(),
            tick: 1,
        });
        mgr.record_event(ChurnEvent::Left {
            peer_id: "b".into(),
            tick: 1,
        });
        let score = mgr.stability_score(1);
        assert!((score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stability_score_partial() {
        let cfg = ChurnManagerConfig {
            window_size_ticks: 9,
            stability_threshold: 0.1,
        };
        let mut mgr = PeerChurnManager::new(cfg);
        // 5 events in window [0, 9] (width=10) → rate = 5/10 = 0.5 → score = 0.5
        for i in 0u64..5 {
            mgr.record_event(ChurnEvent::Joined {
                peer_id: format!("p{i}"),
                tick: i,
            });
        }
        let score = mgr.stability_score(9);
        assert!((score - 0.5).abs() < 1e-10);
    }

    // ── online_peers ──────────────────────────────────────────────────────────

    #[test]
    fn test_online_peers_sorted_alphabetically() {
        let mut mgr = default_manager();
        mgr.record_event(ChurnEvent::Joined {
            peer_id: "charlie".into(),
            tick: 1,
        });
        mgr.record_event(ChurnEvent::Joined {
            peer_id: "alice".into(),
            tick: 2,
        });
        mgr.record_event(ChurnEvent::Joined {
            peer_id: "bob".into(),
            tick: 3,
        });

        let peers = mgr.online_peers();
        assert_eq!(peers, vec!["alice", "bob", "charlie"]);
    }

    #[test]
    fn test_online_peers_excludes_offline() {
        let mut mgr = default_manager();
        mgr.record_event(ChurnEvent::Joined {
            peer_id: "alice".into(),
            tick: 0,
        });
        mgr.record_event(ChurnEvent::Joined {
            peer_id: "bob".into(),
            tick: 1,
        });
        mgr.record_event(ChurnEvent::Left {
            peer_id: "alice".into(),
            tick: 5,
        });

        let peers = mgr.online_peers();
        assert_eq!(peers, vec!["bob"]);
    }

    #[test]
    fn test_online_peers_empty_when_all_left() {
        let mut mgr = default_manager();
        mgr.record_event(ChurnEvent::Joined {
            peer_id: "x".into(),
            tick: 0,
        });
        mgr.record_event(ChurnEvent::Left {
            peer_id: "x".into(),
            tick: 1,
        });
        assert!(mgr.online_peers().is_empty());
    }

    // ── peer_lifetime ─────────────────────────────────────────────────────────

    #[test]
    fn test_peer_lifetime_returns_some() {
        let mut mgr = default_manager();
        mgr.record_event(ChurnEvent::Joined {
            peer_id: "z".into(),
            tick: 7,
        });
        assert!(mgr.peer_lifetime("z").is_some());
    }

    #[test]
    fn test_peer_lifetime_returns_none_for_unknown() {
        let mgr = default_manager();
        assert!(mgr.peer_lifetime("nobody").is_none());
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_current_online() {
        let mut mgr = default_manager();
        mgr.record_event(ChurnEvent::Joined {
            peer_id: "a".into(),
            tick: 0,
        });
        mgr.record_event(ChurnEvent::Joined {
            peer_id: "b".into(),
            tick: 0,
        });
        mgr.record_event(ChurnEvent::Left {
            peer_id: "a".into(),
            tick: 5,
        });
        let s = mgr.stats(10);
        assert_eq!(s.current_online, 1);
    }

    #[test]
    fn test_stats_total_joins_and_leaves() {
        let mut mgr = default_manager();
        mgr.record_event(ChurnEvent::Joined {
            peer_id: "a".into(),
            tick: 0,
        });
        mgr.record_event(ChurnEvent::Joined {
            peer_id: "b".into(),
            tick: 1,
        });
        mgr.record_event(ChurnEvent::Left {
            peer_id: "a".into(),
            tick: 5,
        });
        let s = mgr.stats(10);
        assert_eq!(s.total_joins, 2);
        assert_eq!(s.total_leaves, 1);
    }

    #[test]
    fn test_stats_avg_lifetime_ticks() {
        let mut mgr = default_manager();
        // peer_a: online from 0, left at 10 → duration at tick=20 is 10
        mgr.record_event(ChurnEvent::Joined {
            peer_id: "a".into(),
            tick: 0,
        });
        mgr.record_event(ChurnEvent::Left {
            peer_id: "a".into(),
            tick: 10,
        });
        // peer_b: online from 10, still online → duration at tick=20 is 10
        mgr.record_event(ChurnEvent::Joined {
            peer_id: "b".into(),
            tick: 10,
        });
        let s = mgr.stats(20);
        assert!((s.avg_lifetime_ticks - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_is_stable_true() {
        let cfg = ChurnManagerConfig {
            window_size_ticks: 100,
            stability_threshold: 0.5,
        };
        let mgr = PeerChurnManager::new(cfg);
        // No events → churn_rate = 0 < 0.5 → stable
        let s = mgr.stats(100);
        assert!(s.is_stable);
    }

    #[test]
    fn test_stats_is_stable_false() {
        let cfg = ChurnManagerConfig {
            window_size_ticks: 9,
            stability_threshold: 0.1,
        };
        let mut mgr = PeerChurnManager::new(cfg);
        // 5 joins in [0..9] window → rate = 5/10 = 0.5 > 0.1 → not stable
        for i in 0u64..5 {
            mgr.record_event(ChurnEvent::Joined {
                peer_id: format!("p{i}"),
                tick: i,
            });
        }
        let s = mgr.stats(9);
        assert!(!s.is_stable);
    }
}
