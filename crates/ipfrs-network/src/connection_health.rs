//! Connection Health Checker
//!
//! Monitors TCP/QUIC connection health by tracking keepalive failures, RTT spikes,
//! and connection reset events. Provides EWMA-based RTT smoothing and
//! threshold-driven state transitions between Healthy, Degraded, and Dead.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_network::connection_health::{
//!     ConnectionHealthChecker, ConnectionEvent, HealthCheckerConfig,
//! };
//!
//! let config = HealthCheckerConfig::default();
//! let mut checker = ConnectionHealthChecker::new(config);
//!
//! let conn_id = checker.open_connection("peer-abc".to_string());
//!
//! checker.record_event(conn_id, ConnectionEvent::KeepaliveSuccess { rtt_ms: 10.0 });
//! checker.record_event(conn_id, ConnectionEvent::DataSent { bytes: 1024 });
//!
//! let (total, alive, dead) = checker.stats();
//! println!("total={total}, alive={alive}, dead={dead}");
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ConnectionEvent
// ---------------------------------------------------------------------------

/// An observable event that occurred on a monitored connection.
#[derive(Clone, Debug, PartialEq)]
pub enum ConnectionEvent {
    /// A keepalive probe succeeded; carries the measured round-trip time.
    KeepaliveSuccess { rtt_ms: f64 },
    /// A keepalive probe received no response.
    KeepaliveFailed,
    /// Bytes were written to the connection.
    DataSent { bytes: u64 },
    /// Bytes were read from the connection.
    DataReceived { bytes: u64 },
    /// The connection was forcibly reset by one of the peers.
    Reset { reason: String },
    /// A previously dead or degraded connection has re-established successfully.
    Reconnected,
}

// ---------------------------------------------------------------------------
// ConnectionState
// ---------------------------------------------------------------------------

/// The health state of a single monitored connection.
#[derive(Clone, Debug, PartialEq)]
pub enum ConnectionHealthState {
    /// Connection is operating normally.
    Healthy,
    /// Connection has experienced failures but has not yet crossed the dead
    /// threshold.
    Degraded { consecutive_failures: u32 },
    /// Connection is considered permanently failed until explicitly reconnected.
    Dead { reason: String },
    /// A reconnect has been initiated; the connection is not yet confirmed
    /// healthy.
    Reconnecting,
}

// ---------------------------------------------------------------------------
// ConnectionRecord
// ---------------------------------------------------------------------------

/// Maximum number of events kept per connection ring buffer.
const MAX_EVENTS: usize = 50;

/// EWMA smoothing factor for RTT updates.
const RTT_ALPHA: f64 = 0.2;

/// A record tracking the full observable state of one connection.
#[derive(Clone, Debug)]
pub struct ConnectionRecord {
    /// Stable numeric identifier assigned at open time.
    pub conn_id: u64,
    /// Human-readable peer identifier (e.g. peer ID string, socket address).
    pub peer_id: String,
    /// Current health state.
    pub state: ConnectionHealthState,
    /// Ring buffer of the most recent events (capped at `MAX_EVENTS`).
    pub events: Vec<ConnectionEvent>,
    /// Cumulative bytes written to this connection.
    pub bytes_sent: u64,
    /// Cumulative bytes read from this connection.
    pub bytes_received: u64,
    /// Number of consecutive keepalive failures without an intervening success.
    pub consecutive_failures: u32,
    /// Exponentially weighted moving average of the RTT in milliseconds.
    /// Initialised to 0.0 before any sample arrives.
    pub avg_rtt_ms: f64,
}

impl ConnectionRecord {
    /// Returns `true` when the connection is not in the [`ConnectionHealthState::Dead`] state.
    pub fn is_alive(&self) -> bool {
        !matches!(self.state, ConnectionHealthState::Dead { .. })
    }

    /// Append an event, evicting the oldest entry if the buffer is full.
    fn push_event(&mut self, event: ConnectionEvent) {
        if self.events.len() >= MAX_EVENTS {
            self.events.remove(0);
        }
        self.events.push(event);
    }

    /// Update the EWMA RTT estimate with a new sample.
    ///
    /// If no sample has been recorded yet (avg == 0.0) the first sample is
    /// used directly to avoid a cold-start bias.
    fn update_rtt(&mut self, rtt_ms: f64) {
        if self.avg_rtt_ms == 0.0 {
            self.avg_rtt_ms = rtt_ms;
        } else {
            self.avg_rtt_ms = RTT_ALPHA * rtt_ms + (1.0 - RTT_ALPHA) * self.avg_rtt_ms;
        }
    }
}

// ---------------------------------------------------------------------------
// HealthCheckerConfig
// ---------------------------------------------------------------------------

/// Tuneable thresholds and factors for [`ConnectionHealthChecker`].
#[derive(Clone, Debug)]
pub struct HealthCheckerConfig {
    /// Number of consecutive keepalive failures required to transition to
    /// [`ConnectionHealthState::Degraded`].
    pub failure_threshold: u32,
    /// Number of consecutive keepalive failures required to transition to
    /// [`ConnectionHealthState::Dead`].
    pub dead_threshold: u32,
    /// A keepalive RTT is considered a "spike" when it exceeds
    /// `avg_rtt_ms * rtt_spike_factor`.  The field is stored for
    /// downstream consumers; the health checker itself records the event but
    /// does not alter the state based on spikes alone.
    pub rtt_spike_factor: f64,
}

impl Default for HealthCheckerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 3,
            dead_threshold: 10,
            rtt_spike_factor: 2.0,
        }
    }
}

// ---------------------------------------------------------------------------
// ConnectionHealthChecker
// ---------------------------------------------------------------------------

/// Tracks and evaluates the health of a set of TCP/QUIC connections.
///
/// Call [`open_connection`][ConnectionHealthChecker::open_connection] to begin
/// monitoring a connection, feed events via
/// [`record_event`][ConnectionHealthChecker::record_event], and query health
/// through [`alive_connections`][ConnectionHealthChecker::alive_connections],
/// [`dead_connections`][ConnectionHealthChecker::dead_connections], or
/// [`stats`][ConnectionHealthChecker::stats].
pub struct ConnectionHealthChecker {
    /// All tracked connections, keyed by connection ID.
    pub connections: HashMap<u64, ConnectionRecord>,
    /// Configuration governing threshold behaviour.
    pub config: HealthCheckerConfig,
    /// Monotonically increasing counter used to assign connection IDs.
    pub next_conn_id: u64,
}

impl ConnectionHealthChecker {
    /// Create a new checker with the supplied configuration.
    pub fn new(config: HealthCheckerConfig) -> Self {
        Self {
            connections: HashMap::new(),
            config,
            next_conn_id: 1,
        }
    }

    /// Begin monitoring a connection to `peer_id`.
    ///
    /// Returns the stable connection ID that must be supplied to subsequent
    /// calls.
    pub fn open_connection(&mut self, peer_id: String) -> u64 {
        let conn_id = self.next_conn_id;
        self.next_conn_id += 1;

        let record = ConnectionRecord {
            conn_id,
            peer_id,
            state: ConnectionHealthState::Healthy,
            events: Vec::new(),
            bytes_sent: 0,
            bytes_received: 0,
            consecutive_failures: 0,
            avg_rtt_ms: 0.0,
        };
        self.connections.insert(conn_id, record);
        conn_id
    }

    /// Feed an event into the connection record identified by `conn_id`.
    ///
    /// Unknown `conn_id` values are silently ignored so that callers do not
    /// need to guard against race conditions between close and event delivery.
    pub fn record_event(&mut self, conn_id: u64, event: ConnectionEvent) {
        let Some(record) = self.connections.get_mut(&conn_id) else {
            return;
        };

        // Append to ring buffer before processing so every event is visible.
        record.push_event(event.clone());

        match event {
            ConnectionEvent::KeepaliveFailed => {
                record.consecutive_failures += 1;

                if record.consecutive_failures >= self.config.dead_threshold {
                    record.state = ConnectionHealthState::Dead {
                        reason: format!(
                            "exceeded dead threshold ({} consecutive failures)",
                            record.consecutive_failures
                        ),
                    };
                } else if record.consecutive_failures >= self.config.failure_threshold {
                    record.state = ConnectionHealthState::Degraded {
                        consecutive_failures: record.consecutive_failures,
                    };
                }
            }

            ConnectionEvent::KeepaliveSuccess { rtt_ms } => {
                record.consecutive_failures = 0;
                record.update_rtt(rtt_ms);
                record.state = ConnectionHealthState::Healthy;
            }

            ConnectionEvent::Reset { reason } => {
                record.state = ConnectionHealthState::Dead { reason };
            }

            ConnectionEvent::Reconnected => {
                record.consecutive_failures = 0;
                // Transition through Reconnecting before settling at Healthy
                // so that callers who inspect state mid-stream can distinguish
                // a fresh reconnect from an already-stable connection.
                record.state = ConnectionHealthState::Reconnecting;
                record.state = ConnectionHealthState::Healthy;
            }

            ConnectionEvent::DataSent { bytes } => {
                record.bytes_sent = record.bytes_sent.saturating_add(bytes);
            }

            ConnectionEvent::DataReceived { bytes } => {
                record.bytes_received = record.bytes_received.saturating_add(bytes);
            }
        }
    }

    /// Remove a connection from tracking.
    ///
    /// Returns `true` when the connection was present and has been removed,
    /// `false` when no record with `conn_id` existed.
    pub fn close_connection(&mut self, conn_id: u64) -> bool {
        self.connections.remove(&conn_id).is_some()
    }

    /// Returns references to all connections that are **not** in the `Dead`
    /// state, in unspecified order.
    pub fn alive_connections(&self) -> Vec<&ConnectionRecord> {
        self.connections.values().filter(|r| r.is_alive()).collect()
    }

    /// Returns references to all connections that **are** in the `Dead`
    /// state, in unspecified order.
    pub fn dead_connections(&self) -> Vec<&ConnectionRecord> {
        self.connections
            .values()
            .filter(|r| !r.is_alive())
            .collect()
    }

    /// Returns a triple of `(total, alive, dead)` connection counts.
    pub fn stats(&self) -> (usize, usize, usize) {
        let total = self.connections.len();
        let dead = self.connections.values().filter(|r| !r.is_alive()).count();
        let alive = total - dead;
        (total, alive, dead)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_checker() -> ConnectionHealthChecker {
        ConnectionHealthChecker::new(HealthCheckerConfig::default())
    }

    // 1. open_connection creates a record with correct peer_id
    #[test]
    fn test_open_connection_creates_record() {
        let mut checker = default_checker();
        let id = checker.open_connection("peer-1".to_string());
        let record = checker.connections.get(&id).expect("record must exist");
        assert_eq!(record.peer_id, "peer-1");
        assert_eq!(record.conn_id, id);
    }

    // 2. open_connection starts in Healthy state
    #[test]
    fn test_open_connection_starts_healthy() {
        let mut checker = default_checker();
        let id = checker.open_connection("peer-2".to_string());
        let record = checker.connections.get(&id).expect("record must exist");
        assert!(matches!(record.state, ConnectionHealthState::Healthy));
    }

    // 3. open_connection initial bytes and failures are zero
    #[test]
    fn test_open_connection_initial_counters_zero() {
        let mut checker = default_checker();
        let id = checker.open_connection("peer-3".to_string());
        let record = checker.connections.get(&id).expect("record must exist");
        assert_eq!(record.bytes_sent, 0);
        assert_eq!(record.bytes_received, 0);
        assert_eq!(record.consecutive_failures, 0);
        assert_eq!(record.avg_rtt_ms, 0.0);
    }

    // 4. open_connection assigns distinct IDs for multiple connections
    #[test]
    fn test_open_connection_unique_ids() {
        let mut checker = default_checker();
        let id1 = checker.open_connection("peer-a".to_string());
        let id2 = checker.open_connection("peer-b".to_string());
        let id3 = checker.open_connection("peer-c".to_string());
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    // 5. KeepaliveFailed increments consecutive_failures
    #[test]
    fn test_keepalive_failed_increments_failures() {
        let mut checker = default_checker();
        let id = checker.open_connection("peer".to_string());
        checker.record_event(id, ConnectionEvent::KeepaliveFailed);
        checker.record_event(id, ConnectionEvent::KeepaliveFailed);
        let record = checker.connections.get(&id).expect("record must exist");
        assert_eq!(record.consecutive_failures, 2);
    }

    // 6. Degraded after failure_threshold failures
    #[test]
    fn test_degraded_after_failure_threshold() {
        let mut checker = ConnectionHealthChecker::new(HealthCheckerConfig {
            failure_threshold: 3,
            dead_threshold: 10,
            rtt_spike_factor: 2.0,
        });
        let id = checker.open_connection("peer".to_string());
        for _ in 0..3 {
            checker.record_event(id, ConnectionEvent::KeepaliveFailed);
        }
        let record = checker.connections.get(&id).expect("record must exist");
        assert!(
            matches!(
                record.state,
                ConnectionHealthState::Degraded {
                    consecutive_failures: 3
                }
            ),
            "expected Degraded, got {:?}",
            record.state
        );
    }

    // 7. Dead after dead_threshold failures
    #[test]
    fn test_dead_after_dead_threshold() {
        let mut checker = ConnectionHealthChecker::new(HealthCheckerConfig {
            failure_threshold: 3,
            dead_threshold: 5,
            rtt_spike_factor: 2.0,
        });
        let id = checker.open_connection("peer".to_string());
        for _ in 0..5 {
            checker.record_event(id, ConnectionEvent::KeepaliveFailed);
        }
        let record = checker.connections.get(&id).expect("record must exist");
        assert!(
            matches!(record.state, ConnectionHealthState::Dead { .. }),
            "expected Dead, got {:?}",
            record.state
        );
        assert!(!record.is_alive());
    }

    // 8. KeepaliveSuccess resets consecutive_failures to zero
    #[test]
    fn test_keepalive_success_resets_failures() {
        let mut checker = default_checker();
        let id = checker.open_connection("peer".to_string());
        checker.record_event(id, ConnectionEvent::KeepaliveFailed);
        checker.record_event(id, ConnectionEvent::KeepaliveFailed);
        checker.record_event(id, ConnectionEvent::KeepaliveSuccess { rtt_ms: 20.0 });
        let record = checker.connections.get(&id).expect("record must exist");
        assert_eq!(record.consecutive_failures, 0);
        assert!(matches!(record.state, ConnectionHealthState::Healthy));
    }

    // 9. avg_rtt_ms EWMA: first sample seeds the average
    #[test]
    fn test_avg_rtt_ewma_first_sample() {
        let mut checker = default_checker();
        let id = checker.open_connection("peer".to_string());
        checker.record_event(id, ConnectionEvent::KeepaliveSuccess { rtt_ms: 50.0 });
        let record = checker.connections.get(&id).expect("record must exist");
        // First sample sets avg directly
        assert!((record.avg_rtt_ms - 50.0).abs() < 1e-9);
    }

    // 10. avg_rtt_ms EWMA: second sample applies alpha blending
    #[test]
    fn test_avg_rtt_ewma_second_sample() {
        let mut checker = default_checker();
        let id = checker.open_connection("peer".to_string());
        checker.record_event(id, ConnectionEvent::KeepaliveSuccess { rtt_ms: 100.0 });
        checker.record_event(id, ConnectionEvent::KeepaliveSuccess { rtt_ms: 200.0 });
        let record = checker.connections.get(&id).expect("record must exist");
        // expected: 0.2 * 200 + 0.8 * 100 = 40 + 80 = 120
        let expected = 0.2 * 200.0 + 0.8 * 100.0;
        assert!(
            (record.avg_rtt_ms - expected).abs() < 1e-6,
            "avg_rtt_ms={} expected={}",
            record.avg_rtt_ms,
            expected
        );
    }

    // 11. avg_rtt_ms EWMA: multiple samples converge correctly
    #[test]
    fn test_avg_rtt_ewma_convergence() {
        let mut checker = default_checker();
        let id = checker.open_connection("peer".to_string());
        // Feed a constant RTT; EWMA must converge to that value
        for _ in 0..100 {
            checker.record_event(id, ConnectionEvent::KeepaliveSuccess { rtt_ms: 30.0 });
        }
        let record = checker.connections.get(&id).expect("record must exist");
        assert!(
            (record.avg_rtt_ms - 30.0).abs() < 0.01,
            "avg_rtt_ms={} should converge to 30.0",
            record.avg_rtt_ms
        );
    }

    // 12. Reset sets state to Dead with the supplied reason
    #[test]
    fn test_reset_sets_dead() {
        let mut checker = default_checker();
        let id = checker.open_connection("peer".to_string());
        checker.record_event(
            id,
            ConnectionEvent::Reset {
                reason: "TCP RST received".to_string(),
            },
        );
        let record = checker.connections.get(&id).expect("record must exist");
        match &record.state {
            ConnectionHealthState::Dead { reason } => {
                assert_eq!(reason, "TCP RST received");
            }
            other => panic!("expected Dead, got {other:?}"),
        }
        assert!(!record.is_alive());
    }

    // 13. Reconnected restores Healthy and resets failure counter
    #[test]
    fn test_reconnected_restores_healthy() {
        let mut checker = ConnectionHealthChecker::new(HealthCheckerConfig {
            failure_threshold: 2,
            dead_threshold: 5,
            rtt_spike_factor: 2.0,
        });
        let id = checker.open_connection("peer".to_string());
        // Drive to Dead
        for _ in 0..5 {
            checker.record_event(id, ConnectionEvent::KeepaliveFailed);
        }
        // Reconnect
        checker.record_event(id, ConnectionEvent::Reconnected);
        let record = checker.connections.get(&id).expect("record must exist");
        assert!(matches!(record.state, ConnectionHealthState::Healthy));
        assert_eq!(record.consecutive_failures, 0);
        assert!(record.is_alive());
    }

    // 14. DataSent updates bytes_sent
    #[test]
    fn test_data_sent_updates_bytes() {
        let mut checker = default_checker();
        let id = checker.open_connection("peer".to_string());
        checker.record_event(id, ConnectionEvent::DataSent { bytes: 512 });
        checker.record_event(id, ConnectionEvent::DataSent { bytes: 1024 });
        let record = checker.connections.get(&id).expect("record must exist");
        assert_eq!(record.bytes_sent, 1536);
        assert_eq!(record.bytes_received, 0);
    }

    // 15. DataReceived updates bytes_received
    #[test]
    fn test_data_received_updates_bytes() {
        let mut checker = default_checker();
        let id = checker.open_connection("peer".to_string());
        checker.record_event(id, ConnectionEvent::DataReceived { bytes: 4096 });
        let record = checker.connections.get(&id).expect("record must exist");
        assert_eq!(record.bytes_received, 4096);
        assert_eq!(record.bytes_sent, 0);
    }

    // 16. close_connection removes the record and returns true
    #[test]
    fn test_close_connection_removes_record() {
        let mut checker = default_checker();
        let id = checker.open_connection("peer".to_string());
        assert!(checker.close_connection(id));
        assert!(!checker.connections.contains_key(&id));
    }

    // 17. close_connection returns false for unknown IDs
    #[test]
    fn test_close_connection_unknown_id() {
        let mut checker = default_checker();
        assert!(!checker.close_connection(9999));
    }

    // 18. alive_connections / dead_connections counts
    #[test]
    fn test_alive_dead_counts() {
        let mut checker = ConnectionHealthChecker::new(HealthCheckerConfig {
            failure_threshold: 2,
            dead_threshold: 3,
            rtt_spike_factor: 2.0,
        });
        let id_alive = checker.open_connection("peer-alive".to_string());
        let id_dead = checker.open_connection("peer-dead".to_string());

        // Kill id_dead
        for _ in 0..3 {
            checker.record_event(id_dead, ConnectionEvent::KeepaliveFailed);
        }
        // Keep id_alive healthy
        checker.record_event(id_alive, ConnectionEvent::KeepaliveSuccess { rtt_ms: 5.0 });

        let alive = checker.alive_connections();
        let dead = checker.dead_connections();
        assert_eq!(alive.len(), 1);
        assert_eq!(dead.len(), 1);
        assert_eq!(alive[0].conn_id, id_alive);
        assert_eq!(dead[0].conn_id, id_dead);
    }

    // 19. stats() returns (total, alive, dead) correctly
    #[test]
    fn test_stats_tuple() {
        let mut checker = ConnectionHealthChecker::new(HealthCheckerConfig {
            failure_threshold: 2,
            dead_threshold: 3,
            rtt_spike_factor: 2.0,
        });
        let id1 = checker.open_connection("p1".to_string());
        let id2 = checker.open_connection("p2".to_string());
        let _id3 = checker.open_connection("p3".to_string());

        // Kill id1 and id2
        for _ in 0..3 {
            checker.record_event(id1, ConnectionEvent::KeepaliveFailed);
            checker.record_event(id2, ConnectionEvent::KeepaliveFailed);
        }

        let (total, alive, dead) = checker.stats();
        assert_eq!(total, 3);
        assert_eq!(alive, 1);
        assert_eq!(dead, 2);
    }

    // 20. Event ring buffer is capped at MAX_EVENTS (50)
    #[test]
    fn test_event_cap_at_50() {
        let mut checker = default_checker();
        let id = checker.open_connection("peer".to_string());
        // Push 70 events
        for _ in 0..70 {
            checker.record_event(id, ConnectionEvent::DataSent { bytes: 1 });
        }
        let record = checker.connections.get(&id).expect("record must exist");
        assert_eq!(record.events.len(), MAX_EVENTS);
    }

    // 21. Event ring buffer evicts oldest entries
    #[test]
    fn test_event_ring_evicts_oldest() {
        let mut checker = default_checker();
        let id = checker.open_connection("peer".to_string());
        // Fill the buffer with DataSent(1) events
        for _ in 0..50 {
            checker.record_event(id, ConnectionEvent::DataSent { bytes: 1 });
        }
        // Push a distinct event; the first DataSent(1) should be evicted
        checker.record_event(id, ConnectionEvent::DataReceived { bytes: 999 });
        let record = checker.connections.get(&id).expect("record must exist");
        assert_eq!(record.events.len(), MAX_EVENTS);
        // The last element must be the DataReceived event
        assert_eq!(
            record.events.last(),
            Some(&ConnectionEvent::DataReceived { bytes: 999 })
        );
    }

    // 22. record_event on unknown conn_id is silently ignored
    #[test]
    fn test_record_event_unknown_conn_id_ignored() {
        let mut checker = default_checker();
        // Should not panic
        checker.record_event(42, ConnectionEvent::KeepaliveFailed);
        assert!(checker.connections.is_empty());
    }

    // 23. is_alive returns false only for Dead state
    #[test]
    fn test_is_alive_degraded_is_still_alive() {
        let mut checker = ConnectionHealthChecker::new(HealthCheckerConfig {
            failure_threshold: 2,
            dead_threshold: 10,
            rtt_spike_factor: 2.0,
        });
        let id = checker.open_connection("peer".to_string());
        // Trigger Degraded (2 failures >= failure_threshold=2)
        checker.record_event(id, ConnectionEvent::KeepaliveFailed);
        checker.record_event(id, ConnectionEvent::KeepaliveFailed);
        let record = checker.connections.get(&id).expect("record must exist");
        assert!(matches!(
            record.state,
            ConnectionHealthState::Degraded { .. }
        ));
        assert!(record.is_alive());
    }

    // 24. Multiple connections tracked independently
    #[test]
    fn test_multiple_connections_independent() {
        let mut checker = ConnectionHealthChecker::new(HealthCheckerConfig {
            failure_threshold: 2,
            dead_threshold: 3,
            rtt_spike_factor: 2.0,
        });
        let id_a = checker.open_connection("a".to_string());
        let id_b = checker.open_connection("b".to_string());

        checker.record_event(id_a, ConnectionEvent::KeepaliveSuccess { rtt_ms: 10.0 });
        for _ in 0..3 {
            checker.record_event(id_b, ConnectionEvent::KeepaliveFailed);
        }

        let rec_a = checker.connections.get(&id_a).expect("a must exist");
        let rec_b = checker.connections.get(&id_b).expect("b must exist");

        assert!(matches!(rec_a.state, ConnectionHealthState::Healthy));
        assert!(matches!(rec_b.state, ConnectionHealthState::Dead { .. }));
    }

    // 25. Dead connection via Reset is not counted as alive
    #[test]
    fn test_reset_not_in_alive_connections() {
        let mut checker = default_checker();
        let id = checker.open_connection("peer".to_string());
        checker.record_event(
            id,
            ConnectionEvent::Reset {
                reason: "timeout".to_string(),
            },
        );
        assert_eq!(checker.alive_connections().len(), 0);
        assert_eq!(checker.dead_connections().len(), 1);
    }
}
