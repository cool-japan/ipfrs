//! Peer Connection Tracker
//!
//! Tracks the full lifecycle of peer connections: establishment, protocol negotiation,
//! failure, graceful close, and per-peer statistics.
//!
//! # Overview
//!
//! `PeerConnectionTracker` maintains a registry of all known peers and their current
//! connection state, aggregated statistics, and a bounded event log of the 500 most
//! recent connection events.
//!
//! # Example
//!
//! ```
//! use ipfrs_network::connection_tracker::{
//!     ConnectionEvent, PeerConnectionTracker,
//! };
//!
//! let mut tracker = PeerConnectionTracker::new();
//!
//! tracker.record_event(ConnectionEvent::Connected {
//!     peer_id: "peer-1".to_string(),
//!     address: "/ip4/127.0.0.1/tcp/4001".to_string(),
//!     at_secs: 1_000,
//! });
//!
//! let peers = tracker.connected_peers();
//! assert_eq!(peers.len(), 1);
//! ```

use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of events retained in the event log.
const MAX_EVENT_LOG: usize = 500;

// ---------------------------------------------------------------------------
// ConnectionEvent
// ---------------------------------------------------------------------------

/// An event that describes a state change or observation on a peer connection.
#[derive(Clone, Debug)]
pub enum ConnectionEvent {
    /// A peer successfully established a connection.
    Connected {
        peer_id: String,
        address: String,
        at_secs: u64,
    },
    /// A peer's connection was terminated.
    Disconnected {
        peer_id: String,
        reason: String,
        at_secs: u64,
    },
    /// A protocol was negotiated with a peer.
    ProtocolNegotiated {
        peer_id: String,
        protocol: String,
        at_secs: u64,
    },
    /// A ping round-trip succeeded.
    PingSuccess {
        peer_id: String,
        rtt_ms: f64,
        at_secs: u64,
    },
    /// A ping attempt failed.
    PingFailed { peer_id: String, at_secs: u64 },
    /// The address associated with a peer changed.
    AddressChanged {
        peer_id: String,
        new_address: String,
        at_secs: u64,
    },
}

impl ConnectionEvent {
    /// Returns the `peer_id` carried by any variant.
    pub fn peer_id(&self) -> &str {
        match self {
            ConnectionEvent::Connected { peer_id, .. } => peer_id,
            ConnectionEvent::Disconnected { peer_id, .. } => peer_id,
            ConnectionEvent::ProtocolNegotiated { peer_id, .. } => peer_id,
            ConnectionEvent::PingSuccess { peer_id, .. } => peer_id,
            ConnectionEvent::PingFailed { peer_id, .. } => peer_id,
            ConnectionEvent::AddressChanged { peer_id, .. } => peer_id,
        }
    }
}

// ---------------------------------------------------------------------------
// ConnectionState
// ---------------------------------------------------------------------------

/// The current state of a peer's connection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConnectionState {
    /// The connection has been fully established.
    Connected,
    /// The connection has been terminated.
    Disconnected,
    /// A connection attempt is in progress.
    Connecting,
}

// ---------------------------------------------------------------------------
// PeerConnectionInfo
// ---------------------------------------------------------------------------

/// All tracked information for a single peer.
#[derive(Clone, Debug)]
pub struct PeerConnectionInfo {
    /// Unique peer identifier.
    pub peer_id: String,
    /// Most recent address used by the peer.
    pub address: String,
    /// Current connection state.
    pub state: ConnectionState,
    /// Unix timestamp (seconds) when the connection was first established.
    pub connected_at_secs: Option<u64>,
    /// Unix timestamp (seconds) when the connection was last terminated.
    pub disconnected_at_secs: Option<u64>,
    /// Protocols negotiated with this peer.
    pub protocols: Vec<String>,
    /// Total number of ping attempts (successes + failures).
    pub ping_count: u64,
    /// Number of failed ping attempts.
    pub ping_failures: u64,
    /// Running average round-trip time across all successful pings (milliseconds).
    pub avg_rtt_ms: f64,
}

impl PeerConnectionInfo {
    /// Create a new, empty `PeerConnectionInfo` for the given peer.
    fn new(peer_id: String, address: String) -> Self {
        Self {
            peer_id,
            address,
            state: ConnectionState::Connecting,
            connected_at_secs: None,
            disconnected_at_secs: None,
            protocols: Vec::new(),
            ping_count: 0,
            ping_failures: 0,
            avg_rtt_ms: 0.0,
        }
    }

    /// Seconds the peer has been connected, measured from `now_secs`.
    ///
    /// Returns `0` if the peer is not currently connected.
    pub fn uptime_secs(&self, now_secs: u64) -> u64 {
        if self.state != ConnectionState::Connected {
            return 0;
        }
        match self.connected_at_secs {
            Some(connected_at) if now_secs >= connected_at => now_secs - connected_at,
            _ => 0,
        }
    }

    /// Fraction of pings that succeeded: `1.0 - (failures / total.max(1))`.
    pub fn reliability(&self) -> f64 {
        1.0 - (self.ping_failures as f64 / self.ping_count.max(1) as f64)
    }
}

// ---------------------------------------------------------------------------
// TrackerStats
// ---------------------------------------------------------------------------

/// Aggregated statistics across all tracked peers.
#[derive(Clone, Debug, Default)]
pub struct TrackerStats {
    /// Total number of peers that have ever connected (cumulative).
    pub total_connections: u64,
    /// Number of peers currently in the `Connected` state.
    pub current_connections: usize,
    /// Total number of disconnection events recorded.
    pub total_disconnections: u64,
    /// Total number of ping attempts (successes + failures) recorded.
    pub total_pings: u64,
    /// Total number of failed ping attempts recorded.
    pub total_ping_failures: u64,
}

// ---------------------------------------------------------------------------
// PeerConnectionTracker
// ---------------------------------------------------------------------------

/// Tracks the lifecycle of all peer connections with per-peer statistics.
pub struct PeerConnectionTracker {
    /// Per-peer connection info, keyed by peer ID.
    pub peers: HashMap<String, PeerConnectionInfo>,
    /// Bounded ring-buffer of recent events (last `MAX_EVENT_LOG`).
    pub event_log: VecDeque<ConnectionEvent>,
    /// Aggregated statistics across all events.
    pub stats: TrackerStats,
}

impl PeerConnectionTracker {
    /// Create a new, empty tracker.
    pub fn new() -> Self {
        Self {
            peers: HashMap::new(),
            event_log: VecDeque::with_capacity(MAX_EVENT_LOG + 1),
            stats: TrackerStats::default(),
        }
    }

    /// Record a `ConnectionEvent`, updating per-peer state and the event log.
    pub fn record_event(&mut self, event: ConnectionEvent) {
        // ------------------------------------------------------------------
        // 1. Update peer state
        // ------------------------------------------------------------------
        match &event {
            ConnectionEvent::Connected {
                peer_id,
                address,
                at_secs,
            } => {
                let entry = self
                    .peers
                    .entry(peer_id.clone())
                    .or_insert_with(|| PeerConnectionInfo::new(peer_id.clone(), address.clone()));
                entry.state = ConnectionState::Connected;
                entry.address = address.clone();
                entry.connected_at_secs = Some(*at_secs);
                entry.disconnected_at_secs = None;

                self.stats.total_connections += 1;
                self.stats.current_connections = self.count_connected();
            }

            ConnectionEvent::Disconnected {
                peer_id, at_secs, ..
            } => {
                if let Some(entry) = self.peers.get_mut(peer_id) {
                    entry.state = ConnectionState::Disconnected;
                    entry.disconnected_at_secs = Some(*at_secs);
                }
                self.stats.total_disconnections += 1;
                self.stats.current_connections = self.count_connected();
            }

            ConnectionEvent::ProtocolNegotiated {
                peer_id, protocol, ..
            } => {
                if let Some(entry) = self.peers.get_mut(peer_id) {
                    if !entry.protocols.contains(protocol) {
                        entry.protocols.push(protocol.clone());
                    }
                }
            }

            ConnectionEvent::PingSuccess {
                peer_id, rtt_ms, ..
            } => {
                if let Some(entry) = self.peers.get_mut(peer_id) {
                    entry.ping_count += 1;
                    let n = entry.ping_count as f64;
                    entry.avg_rtt_ms = (entry.avg_rtt_ms * (n - 1.0) + rtt_ms) / n;
                }
                self.stats.total_pings += 1;
            }

            ConnectionEvent::PingFailed { peer_id, .. } => {
                if let Some(entry) = self.peers.get_mut(peer_id) {
                    entry.ping_count += 1;
                    entry.ping_failures += 1;
                }
                self.stats.total_pings += 1;
                self.stats.total_ping_failures += 1;
            }

            ConnectionEvent::AddressChanged {
                peer_id,
                new_address,
                ..
            } => {
                if let Some(entry) = self.peers.get_mut(peer_id) {
                    entry.address = new_address.clone();
                }
            }
        }

        // ------------------------------------------------------------------
        // 2. Append to event log, trimming to MAX_EVENT_LOG
        // ------------------------------------------------------------------
        self.event_log.push_back(event);
        while self.event_log.len() > MAX_EVENT_LOG {
            self.event_log.pop_front();
        }
    }

    /// Return references to all peers currently in the `Connected` state.
    pub fn connected_peers(&self) -> Vec<&PeerConnectionInfo> {
        self.peers
            .values()
            .filter(|p| p.state == ConnectionState::Connected)
            .collect()
    }

    /// Look up a single peer by ID.
    pub fn peer_info(&self, peer_id: &str) -> Option<&PeerConnectionInfo> {
        self.peers.get(peer_id)
    }

    /// Remove a peer from the registry entirely.
    ///
    /// Returns `true` if the peer was present, `false` otherwise.
    pub fn remove_peer(&mut self, peer_id: &str) -> bool {
        let removed = self.peers.remove(peer_id).is_some();
        if removed {
            self.stats.current_connections = self.count_connected();
        }
        removed
    }

    /// Return peers whose `reliability()` is strictly below `threshold`.
    pub fn unreliable_peers(&self, threshold: f64) -> Vec<&PeerConnectionInfo> {
        self.peers
            .values()
            .filter(|p| p.reliability() < threshold)
            .collect()
    }

    /// Return the top-`n` peers sorted by `uptime_secs(now_secs)` in descending order.
    pub fn top_peers_by_uptime(&self, n: usize, now_secs: u64) -> Vec<&PeerConnectionInfo> {
        let mut peers: Vec<&PeerConnectionInfo> = self.peers.values().collect();
        peers.sort_by_key(|b| std::cmp::Reverse(b.uptime_secs(now_secs)));
        peers.truncate(n);
        peers
    }

    /// Return a reference to the current aggregated statistics.
    pub fn stats(&self) -> &TrackerStats {
        &self.stats
    }

    /// Return the last `count` events from the event log.
    pub fn recent_events(&self, count: usize) -> Vec<&ConnectionEvent> {
        let len = self.event_log.len();
        let skip = len.saturating_sub(count);
        self.event_log.iter().skip(skip).collect()
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    fn count_connected(&self) -> usize {
        self.peers
            .values()
            .filter(|p| p.state == ConnectionState::Connected)
            .count()
    }
}

impl Default for PeerConnectionTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper
    // -----------------------------------------------------------------------

    fn connected_event(peer_id: &str, addr: &str, at: u64) -> ConnectionEvent {
        ConnectionEvent::Connected {
            peer_id: peer_id.to_string(),
            address: addr.to_string(),
            at_secs: at,
        }
    }

    fn disconnected_event(peer_id: &str, reason: &str, at: u64) -> ConnectionEvent {
        ConnectionEvent::Disconnected {
            peer_id: peer_id.to_string(),
            reason: reason.to_string(),
            at_secs: at,
        }
    }

    fn ping_success(peer_id: &str, rtt: f64, at: u64) -> ConnectionEvent {
        ConnectionEvent::PingSuccess {
            peer_id: peer_id.to_string(),
            rtt_ms: rtt,
            at_secs: at,
        }
    }

    fn ping_failed(peer_id: &str, at: u64) -> ConnectionEvent {
        ConnectionEvent::PingFailed {
            peer_id: peer_id.to_string(),
            at_secs: at,
        }
    }

    // -----------------------------------------------------------------------
    // 1. new() produces empty state
    // -----------------------------------------------------------------------
    #[test]
    fn test_new_empty_state() {
        let tracker = PeerConnectionTracker::new();
        assert!(tracker.peers.is_empty());
        assert!(tracker.event_log.is_empty());
        assert_eq!(tracker.stats.total_connections, 0);
        assert_eq!(tracker.stats.current_connections, 0);
    }

    // -----------------------------------------------------------------------
    // 2. record Connected updates peer state
    // -----------------------------------------------------------------------
    #[test]
    fn test_record_connected_updates_state() {
        let mut tracker = PeerConnectionTracker::new();
        tracker.record_event(connected_event("peer-1", "/ip4/1.2.3.4/tcp/4001", 1000));

        let info = tracker.peer_info("peer-1").expect("peer should exist");
        assert_eq!(info.state, ConnectionState::Connected);
        assert_eq!(info.address, "/ip4/1.2.3.4/tcp/4001");
        assert_eq!(info.connected_at_secs, Some(1000));
        assert_eq!(tracker.stats.total_connections, 1);
        assert_eq!(tracker.stats.current_connections, 1);
    }

    // -----------------------------------------------------------------------
    // 3. record Disconnected updates state and timestamp
    // -----------------------------------------------------------------------
    #[test]
    fn test_record_disconnected_updates_state() {
        let mut tracker = PeerConnectionTracker::new();
        tracker.record_event(connected_event("peer-1", "/ip4/1.2.3.4/tcp/4001", 1000));
        tracker.record_event(disconnected_event("peer-1", "timeout", 2000));

        let info = tracker.peer_info("peer-1").expect("peer should exist");
        assert_eq!(info.state, ConnectionState::Disconnected);
        assert_eq!(info.disconnected_at_secs, Some(2000));
        assert_eq!(tracker.stats.total_disconnections, 1);
        assert_eq!(tracker.stats.current_connections, 0);
    }

    // -----------------------------------------------------------------------
    // 4. record ProtocolNegotiated appends to protocols (no duplicates)
    // -----------------------------------------------------------------------
    #[test]
    fn test_protocol_negotiated_appends() {
        let mut tracker = PeerConnectionTracker::new();
        tracker.record_event(connected_event("peer-1", "/ip4/1.2.3.4/tcp/4001", 1000));
        tracker.record_event(ConnectionEvent::ProtocolNegotiated {
            peer_id: "peer-1".to_string(),
            protocol: "/ipfs/kad/1.0.0".to_string(),
            at_secs: 1001,
        });
        tracker.record_event(ConnectionEvent::ProtocolNegotiated {
            peer_id: "peer-1".to_string(),
            protocol: "/ipfs/bitswap/1.2.0".to_string(),
            at_secs: 1002,
        });
        // Duplicate – must not be added twice
        tracker.record_event(ConnectionEvent::ProtocolNegotiated {
            peer_id: "peer-1".to_string(),
            protocol: "/ipfs/kad/1.0.0".to_string(),
            at_secs: 1003,
        });

        let info = tracker.peer_info("peer-1").expect("peer should exist");
        assert_eq!(info.protocols.len(), 2);
        assert!(info.protocols.contains(&"/ipfs/kad/1.0.0".to_string()));
        assert!(info.protocols.contains(&"/ipfs/bitswap/1.2.0".to_string()));
    }

    // -----------------------------------------------------------------------
    // 5. record PingSuccess updates avg_rtt_ms correctly (3 pings)
    // -----------------------------------------------------------------------
    #[test]
    fn test_ping_success_avg_rtt() {
        let mut tracker = PeerConnectionTracker::new();
        tracker.record_event(connected_event("peer-1", "/ip4/1.2.3.4/tcp/4001", 1000));

        tracker.record_event(ping_success("peer-1", 10.0, 1001));
        tracker.record_event(ping_success("peer-1", 20.0, 1002));
        tracker.record_event(ping_success("peer-1", 30.0, 1003));

        let info = tracker.peer_info("peer-1").expect("peer should exist");
        assert_eq!(info.ping_count, 3);
        // Expected: (10 + 20 + 30) / 3 = 20.0
        let expected = 20.0_f64;
        assert!(
            (info.avg_rtt_ms - expected).abs() < 1e-9,
            "avg_rtt_ms = {} expected {}",
            info.avg_rtt_ms,
            expected
        );
    }

    // -----------------------------------------------------------------------
    // 6. record PingFailed increments ping_failures
    // -----------------------------------------------------------------------
    #[test]
    fn test_ping_failed_increments_failures() {
        let mut tracker = PeerConnectionTracker::new();
        tracker.record_event(connected_event("peer-1", "/ip4/1.2.3.4/tcp/4001", 1000));
        tracker.record_event(ping_failed("peer-1", 1001));
        tracker.record_event(ping_failed("peer-1", 1002));

        let info = tracker.peer_info("peer-1").expect("peer should exist");
        assert_eq!(info.ping_count, 2);
        assert_eq!(info.ping_failures, 2);
        assert_eq!(tracker.stats.total_ping_failures, 2);
    }

    // -----------------------------------------------------------------------
    // 7. record AddressChanged updates address
    // -----------------------------------------------------------------------
    #[test]
    fn test_address_changed_updates_address() {
        let mut tracker = PeerConnectionTracker::new();
        tracker.record_event(connected_event("peer-1", "/ip4/1.2.3.4/tcp/4001", 1000));
        tracker.record_event(ConnectionEvent::AddressChanged {
            peer_id: "peer-1".to_string(),
            new_address: "/ip4/5.6.7.8/tcp/4001".to_string(),
            at_secs: 1010,
        });

        let info = tracker.peer_info("peer-1").expect("peer should exist");
        assert_eq!(info.address, "/ip4/5.6.7.8/tcp/4001");
    }

    // -----------------------------------------------------------------------
    // 8. event_log is bounded at MAX_EVENT_LOG (500)
    // -----------------------------------------------------------------------
    #[test]
    fn test_event_log_bounded_at_500() {
        let mut tracker = PeerConnectionTracker::new();
        // Record 600 events – only the last 500 should be retained.
        for i in 0u64..600 {
            tracker.record_event(ping_success("peer-1", i as f64, i));
        }
        assert_eq!(tracker.event_log.len(), 500);
    }

    // -----------------------------------------------------------------------
    // 9. connected_peers filters correctly
    // -----------------------------------------------------------------------
    #[test]
    fn test_connected_peers_filters() {
        let mut tracker = PeerConnectionTracker::new();
        tracker.record_event(connected_event("peer-1", "/ip4/1.2.3.4/tcp/4001", 1000));
        tracker.record_event(connected_event("peer-2", "/ip4/2.2.2.2/tcp/4001", 1000));
        tracker.record_event(disconnected_event("peer-2", "bye", 2000));

        let connected = tracker.connected_peers();
        assert_eq!(connected.len(), 1);
        assert_eq!(connected[0].peer_id, "peer-1");
    }

    // -----------------------------------------------------------------------
    // 10. peer_info returns Some / None
    // -----------------------------------------------------------------------
    #[test]
    fn test_peer_info_some_none() {
        let mut tracker = PeerConnectionTracker::new();
        tracker.record_event(connected_event("peer-1", "/ip4/1.2.3.4/tcp/4001", 1000));

        assert!(tracker.peer_info("peer-1").is_some());
        assert!(tracker.peer_info("unknown").is_none());
    }

    // -----------------------------------------------------------------------
    // 11. remove_peer returns true / false
    // -----------------------------------------------------------------------
    #[test]
    fn test_remove_peer_true_false() {
        let mut tracker = PeerConnectionTracker::new();
        tracker.record_event(connected_event("peer-1", "/ip4/1.2.3.4/tcp/4001", 1000));

        assert!(tracker.remove_peer("peer-1"));
        assert!(!tracker.remove_peer("peer-1")); // already removed
        assert!(!tracker.remove_peer("ghost")); // never existed
    }

    // -----------------------------------------------------------------------
    // 12. unreliable_peers threshold filter
    // -----------------------------------------------------------------------
    #[test]
    fn test_unreliable_peers_threshold() {
        let mut tracker = PeerConnectionTracker::new();
        // peer-1: 1 success, 1 failure → reliability = 0.5
        tracker.record_event(connected_event("peer-1", "/ip4/1.1.1.1/tcp/4001", 1000));
        tracker.record_event(ping_success("peer-1", 10.0, 1001));
        tracker.record_event(ping_failed("peer-1", 1002));

        // peer-2: 2 successes → reliability = 1.0
        tracker.record_event(connected_event("peer-2", "/ip4/2.2.2.2/tcp/4001", 1000));
        tracker.record_event(ping_success("peer-2", 10.0, 1001));
        tracker.record_event(ping_success("peer-2", 10.0, 1002));

        let unreliable = tracker.unreliable_peers(0.8);
        assert_eq!(unreliable.len(), 1);
        assert_eq!(unreliable[0].peer_id, "peer-1");
    }

    // -----------------------------------------------------------------------
    // 13. top_peers_by_uptime sorted descending
    // -----------------------------------------------------------------------
    #[test]
    fn test_top_peers_by_uptime_sorted() {
        let mut tracker = PeerConnectionTracker::new();
        // peer-1 connected at t=100, peer-2 at t=200, peer-3 at t=300
        tracker.record_event(connected_event("peer-1", "/ip4/1.1.1.1/tcp/4001", 100));
        tracker.record_event(connected_event("peer-2", "/ip4/2.2.2.2/tcp/4001", 200));
        tracker.record_event(connected_event("peer-3", "/ip4/3.3.3.3/tcp/4001", 300));

        let now = 1000u64;
        let top = tracker.top_peers_by_uptime(3, now);

        // peer-1 uptime=900, peer-2 uptime=800, peer-3 uptime=700
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].peer_id, "peer-1");
        assert_eq!(top[1].peer_id, "peer-2");
        assert_eq!(top[2].peer_id, "peer-3");
    }

    // -----------------------------------------------------------------------
    // 14. top_peers_by_uptime respects n limit
    // -----------------------------------------------------------------------
    #[test]
    fn test_top_peers_by_uptime_limit() {
        let mut tracker = PeerConnectionTracker::new();
        for i in 0u64..10 {
            tracker.record_event(connected_event(
                &format!("peer-{i}"),
                "/ip4/1.1.1.1/tcp/4001",
                i * 10,
            ));
        }
        let top = tracker.top_peers_by_uptime(3, 1000);
        assert_eq!(top.len(), 3);
    }

    // -----------------------------------------------------------------------
    // 15. reliability() calculation
    // -----------------------------------------------------------------------
    #[test]
    fn test_reliability_calculation() {
        let mut tracker = PeerConnectionTracker::new();
        tracker.record_event(connected_event("peer-1", "/ip4/1.2.3.4/tcp/4001", 1000));

        // No pings yet: reliability = 1.0 - (0 / max(0,1)) = 1.0
        let info = tracker.peer_info("peer-1").expect("peer should exist");
        assert!((info.reliability() - 1.0).abs() < 1e-9);

        // 3 successes, 1 failure → ping_count=4, failures=1, reliability = 0.75
        tracker.record_event(ping_success("peer-1", 10.0, 1001));
        tracker.record_event(ping_success("peer-1", 10.0, 1002));
        tracker.record_event(ping_success("peer-1", 10.0, 1003));
        tracker.record_event(ping_failed("peer-1", 1004));

        let info = tracker.peer_info("peer-1").expect("peer should exist");
        let expected = 1.0 - (1.0 / 4.0);
        assert!(
            (info.reliability() - expected).abs() < 1e-9,
            "got {}",
            info.reliability()
        );
    }

    // -----------------------------------------------------------------------
    // 16. uptime_secs when connected / not connected
    // -----------------------------------------------------------------------
    #[test]
    fn test_uptime_secs_connected_and_not() {
        let mut tracker = PeerConnectionTracker::new();
        tracker.record_event(connected_event("peer-1", "/ip4/1.2.3.4/tcp/4001", 500));

        let info = tracker.peer_info("peer-1").expect("peer should exist");
        assert_eq!(info.uptime_secs(1000), 500);

        tracker.record_event(disconnected_event("peer-1", "bye", 800));
        let info = tracker.peer_info("peer-1").expect("peer should exist");
        assert_eq!(info.uptime_secs(1000), 0);
    }

    // -----------------------------------------------------------------------
    // 17. stats() totals updated correctly
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_totals_updated() {
        let mut tracker = PeerConnectionTracker::new();
        tracker.record_event(connected_event("peer-1", "/ip4/1.1.1.1/tcp/4001", 1000));
        tracker.record_event(connected_event("peer-2", "/ip4/2.2.2.2/tcp/4001", 1001));
        tracker.record_event(ping_success("peer-1", 15.0, 1002));
        tracker.record_event(ping_failed("peer-2", 1003));
        tracker.record_event(disconnected_event("peer-1", "timeout", 2000));

        let s = tracker.stats();
        assert_eq!(s.total_connections, 2);
        assert_eq!(s.current_connections, 1); // peer-2 still connected
        assert_eq!(s.total_disconnections, 1);
        assert_eq!(s.total_pings, 2);
        assert_eq!(s.total_ping_failures, 1);
    }

    // -----------------------------------------------------------------------
    // 18. recent_events returns last N
    // -----------------------------------------------------------------------
    #[test]
    fn test_recent_events_last_n() {
        let mut tracker = PeerConnectionTracker::new();
        tracker.record_event(connected_event("peer-1", "/ip4/1.2.3.4/tcp/4001", 1000));
        tracker.record_event(ping_success("peer-1", 10.0, 1001));
        tracker.record_event(ping_success("peer-1", 20.0, 1002));
        tracker.record_event(ping_failed("peer-1", 1003));
        tracker.record_event(disconnected_event("peer-1", "done", 1004));

        // Last 3 events
        let recent = tracker.recent_events(3);
        assert_eq!(recent.len(), 3);
        // The last event should be Disconnected
        assert!(matches!(recent[2], ConnectionEvent::Disconnected { .. }));
        // The second-to-last should be PingFailed
        assert!(matches!(recent[1], ConnectionEvent::PingFailed { .. }));

        // Requesting more than available returns all
        let all = tracker.recent_events(100);
        assert_eq!(all.len(), 5);
    }
}
