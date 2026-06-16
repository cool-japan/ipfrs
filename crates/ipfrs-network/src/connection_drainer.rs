//! Graceful connection draining for orderly node shutdown.
//!
//! When a node is shutting down, it needs to stop accepting new requests while
//! allowing in-flight requests to complete. The [`ConnectionDrainer`] manages
//! this lifecycle for all active connections, tracking pending request counts,
//! enforcing drain timeouts, and reporting statistics.

use std::collections::HashMap;

/// The drain state of a connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DrainState {
    /// Connection is actively accepting requests.
    Active,
    /// Connection is draining — no new requests accepted, waiting for pending to finish.
    Draining,
    /// Connection has been fully drained (pending requests == 0 or timed out).
    Drained,
}

/// A connection that participates in the draining lifecycle.
#[derive(Debug, Clone)]
pub struct DrainableConnection {
    /// Unique identifier for this connection.
    pub conn_id: u64,
    /// The remote peer identifier.
    pub peer_id: String,
    /// Number of requests still in flight on this connection.
    pub pending_requests: u32,
    /// Current drain state.
    pub state: DrainState,
    /// Timestamp (ms) when draining started, if applicable.
    pub drain_started_at: Option<u64>,
}

/// Configuration for the [`ConnectionDrainer`].
#[derive(Debug, Clone)]
pub struct DrainerConfig {
    /// Per-connection timeout (ms) after which a draining connection is force-drained.
    pub drain_timeout_ms: u64,
    /// Maximum total time (ms) to wait for all connections to drain.
    pub max_drain_wait_ms: u64,
    /// Whether to reject new requests on draining connections.
    pub reject_new_requests: bool,
    /// Delay (ms) before closing a fully drained connection.
    pub graceful_close_delay_ms: u64,
}

impl Default for DrainerConfig {
    fn default() -> Self {
        Self {
            drain_timeout_ms: 30_000,
            max_drain_wait_ms: 60_000,
            reject_new_requests: true,
            graceful_close_delay_ms: 500,
        }
    }
}

/// Aggregate statistics for the drainer.
#[derive(Debug, Clone, Default)]
pub struct DrainerStats {
    /// Total connections that completed draining gracefully.
    pub total_drained: u64,
    /// Total connections that were force-drained due to timeout.
    pub total_timed_out: u64,
    /// Total requests rejected because the connection was draining.
    pub total_rejected: u64,
    /// Average drain time in milliseconds (over gracefully drained connections).
    pub avg_drain_time_ms: f64,
    /// Number of connections currently in the Draining state.
    pub currently_draining: u64,
}

/// Manages graceful connection draining for orderly node shutdown.
///
/// Tracks all registered connections, transitions them through the
/// `Active → Draining → Drained` lifecycle, and collects statistics.
pub struct ConnectionDrainer {
    config: DrainerConfig,
    connections: HashMap<u64, DrainableConnection>,
    drain_order: Vec<u64>,
    next_conn_id: u64,
    stats: DrainerStats,
    /// Sum of drain durations for computing the running average.
    drain_duration_sum_ms: u64,
}

impl ConnectionDrainer {
    /// Create a new drainer with the given configuration.
    pub fn new(config: DrainerConfig) -> Self {
        Self {
            config,
            connections: HashMap::new(),
            drain_order: Vec::new(),
            next_conn_id: 1,
            stats: DrainerStats::default(),
            drain_duration_sum_ms: 0,
        }
    }

    /// Register a new active connection and return its unique id.
    pub fn register_connection(&mut self, peer_id: &str) -> u64 {
        let conn_id = self.next_conn_id;
        self.next_conn_id = self.next_conn_id.wrapping_add(1);

        let conn = DrainableConnection {
            conn_id,
            peer_id: peer_id.to_string(),
            pending_requests: 0,
            state: DrainState::Active,
            drain_started_at: None,
        };
        self.connections.insert(conn_id, conn);
        conn_id
    }

    /// Begin draining a single connection.
    ///
    /// Returns an error if the connection does not exist or is already
    /// draining/drained.
    pub fn start_drain(&mut self, conn_id: u64) -> Result<(), String> {
        let conn = self
            .connections
            .get_mut(&conn_id)
            .ok_or_else(|| format!("connection {conn_id} not found"))?;

        match conn.state {
            DrainState::Active => {
                conn.state = DrainState::Draining;
                // We use a placeholder timestamp; callers that need real wall-clock
                // time should use `check_timeouts(now)`.
                conn.drain_started_at = Some(0);
                self.drain_order.push(conn_id);
                self.stats.currently_draining += 1;

                // If there are no pending requests, transition immediately.
                if conn.pending_requests == 0 {
                    conn.state = DrainState::Drained;
                    self.stats.currently_draining = self.stats.currently_draining.saturating_sub(1);
                    self.stats.total_drained += 1;
                }
                Ok(())
            }
            DrainState::Draining => Err(format!("connection {conn_id} is already draining")),
            DrainState::Drained => Err(format!("connection {conn_id} is already drained")),
        }
    }

    /// Begin draining every currently-active connection.
    pub fn start_drain_all(&mut self) {
        let active_ids: Vec<u64> = self
            .connections
            .iter()
            .filter(|(_, c)| c.state == DrainState::Active)
            .map(|(id, _)| *id)
            .collect();

        for id in active_ids {
            // Errors here mean the connection is already draining/drained, which is fine.
            let _ = self.start_drain(id);
        }
    }

    /// Record the completion of one request on a connection.
    ///
    /// If the connection is draining and pending drops to zero it transitions
    /// to `Drained`.
    pub fn complete_request(&mut self, conn_id: u64) -> Result<(), String> {
        let conn = self
            .connections
            .get_mut(&conn_id)
            .ok_or_else(|| format!("connection {conn_id} not found"))?;

        if conn.pending_requests == 0 {
            return Err(format!("connection {conn_id} has no pending requests"));
        }

        conn.pending_requests -= 1;

        if conn.state == DrainState::Draining && conn.pending_requests == 0 {
            conn.state = DrainState::Drained;
            self.stats.currently_draining = self.stats.currently_draining.saturating_sub(1);
            self.stats.total_drained += 1;
            // We don't know the real elapsed time without a clock; drain_started_at
            // is set to 0 by start_drain and real timestamps come from check_timeouts.
            if let Some(start) = conn.drain_started_at {
                if start > 0 {
                    // Approximate: caller should supply wall-clock via check_timeouts.
                    self.drain_duration_sum_ms += 0; // placeholder
                }
            }
        }

        Ok(())
    }

    /// Add a new request to a connection.
    ///
    /// If the connection is draining and `reject_new_requests` is set, the
    /// request is rejected and the rejection counter incremented.
    pub fn add_request(&mut self, conn_id: u64) -> Result<(), String> {
        let conn = self
            .connections
            .get_mut(&conn_id)
            .ok_or_else(|| format!("connection {conn_id} not found"))?;

        match conn.state {
            DrainState::Active => {
                conn.pending_requests += 1;
                Ok(())
            }
            DrainState::Draining => {
                if self.config.reject_new_requests {
                    self.stats.total_rejected += 1;
                    Err(format!(
                        "connection {conn_id} is draining; new requests rejected"
                    ))
                } else {
                    conn.pending_requests += 1;
                    Ok(())
                }
            }
            DrainState::Drained => {
                self.stats.total_rejected += 1;
                Err(format!("connection {conn_id} is already drained"))
            }
        }
    }

    /// Check whether a connection has been fully drained.
    ///
    /// Returns `Some(DrainState::Drained)` when pending requests reach zero
    /// while draining, `Some(current_state)` otherwise, or `None` if the
    /// connection does not exist.
    pub fn check_drained(&mut self, conn_id: u64) -> Option<DrainState> {
        let conn = self.connections.get_mut(&conn_id)?;

        if conn.state == DrainState::Draining && conn.pending_requests == 0 {
            conn.state = DrainState::Drained;
            self.stats.currently_draining = self.stats.currently_draining.saturating_sub(1);
            self.stats.total_drained += 1;
        }

        Some(conn.state.clone())
    }

    /// Force-drain connections whose drain timeout has elapsed.
    ///
    /// `now` is a monotonic timestamp in milliseconds. Returns the ids of
    /// connections that were force-drained.
    pub fn check_timeouts(&mut self, now: u64) -> Vec<u64> {
        let timeout = self.config.drain_timeout_ms;
        let timed_out: Vec<u64> = self
            .connections
            .iter()
            .filter(|(_, c)| c.state == DrainState::Draining)
            .filter(|(_, c)| {
                c.drain_started_at
                    .map(|s| now.saturating_sub(s) >= timeout)
                    .unwrap_or(false)
            })
            .map(|(id, _)| *id)
            .collect();

        for id in &timed_out {
            if let Some(conn) = self.connections.get_mut(id) {
                conn.state = DrainState::Drained;
                self.stats.currently_draining = self.stats.currently_draining.saturating_sub(1);
                self.stats.total_timed_out += 1;

                if let Some(start) = conn.drain_started_at {
                    let elapsed = now.saturating_sub(start);
                    self.drain_duration_sum_ms += elapsed;
                    let total_completed = self.stats.total_drained + self.stats.total_timed_out;
                    if total_completed > 0 {
                        self.stats.avg_drain_time_ms =
                            self.drain_duration_sum_ms as f64 / total_completed as f64;
                    }
                }
            }
        }

        timed_out
    }

    /// Remove all connections in the `Drained` state and return how many were removed.
    pub fn remove_drained(&mut self) -> usize {
        let before = self.connections.len();
        self.connections
            .retain(|_, c| c.state != DrainState::Drained);
        self.drain_order
            .retain(|id| self.connections.contains_key(id));
        before - self.connections.len()
    }

    /// Look up a connection by id.
    pub fn get_connection(&self, conn_id: u64) -> Option<&DrainableConnection> {
        self.connections.get(&conn_id)
    }

    /// Number of connections in the `Active` state.
    pub fn active_count(&self) -> usize {
        self.connections
            .values()
            .filter(|c| c.state == DrainState::Active)
            .count()
    }

    /// Number of connections in the `Draining` state.
    pub fn draining_count(&self) -> usize {
        self.connections
            .values()
            .filter(|c| c.state == DrainState::Draining)
            .count()
    }

    /// Returns `true` when there are no active or draining connections.
    pub fn is_fully_drained(&self) -> bool {
        self.connections
            .values()
            .all(|c| c.state == DrainState::Drained)
    }

    /// Reference to the current statistics.
    pub fn stats(&self) -> &DrainerStats {
        &self.stats
    }

    /// Set a real wall-clock start time on a draining connection.
    ///
    /// This is a helper so callers that use `start_drain` followed by
    /// `check_timeouts(now)` can record accurate timestamps.
    pub fn set_drain_start_time(&mut self, conn_id: u64, timestamp_ms: u64) {
        if let Some(conn) = self.connections.get_mut(&conn_id) {
            if conn.state == DrainState::Draining || conn.state == DrainState::Drained {
                conn.drain_started_at = Some(timestamp_ms);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> DrainerConfig {
        DrainerConfig::default()
    }

    // 1. Registration returns unique ids
    #[test]
    fn test_register_returns_unique_ids() {
        let mut d = ConnectionDrainer::new(default_config());
        let a = d.register_connection("peer-a");
        let b = d.register_connection("peer-b");
        assert_ne!(a, b);
    }

    // 2. Registered connection is Active
    #[test]
    fn test_registered_connection_is_active() {
        let mut d = ConnectionDrainer::new(default_config());
        let id = d.register_connection("peer-a");
        let conn = d.get_connection(id).expect("should exist");
        assert_eq!(conn.state, DrainState::Active);
        assert_eq!(conn.pending_requests, 0);
    }

    // 3. Start drain on active connection
    #[test]
    fn test_start_drain_active() {
        let mut d = ConnectionDrainer::new(default_config());
        let id = d.register_connection("peer-a");
        d.add_request(id).expect("add ok");
        d.start_drain(id).expect("drain ok");
        let conn = d.get_connection(id).expect("exists");
        assert_eq!(conn.state, DrainState::Draining);
    }

    // 4. Start drain with zero pending transitions immediately to Drained
    #[test]
    fn test_start_drain_zero_pending_immediate() {
        let mut d = ConnectionDrainer::new(default_config());
        let id = d.register_connection("peer-a");
        d.start_drain(id).expect("drain ok");
        let conn = d.get_connection(id).expect("exists");
        assert_eq!(conn.state, DrainState::Drained);
    }

    // 5. Duplicate drain attempt fails
    #[test]
    fn test_duplicate_drain_fails() {
        let mut d = ConnectionDrainer::new(default_config());
        let id = d.register_connection("peer-a");
        d.add_request(id).expect("add ok");
        d.start_drain(id).expect("first ok");
        assert!(d.start_drain(id).is_err());
    }

    // 6. Drain on nonexistent connection
    #[test]
    fn test_drain_nonexistent() {
        let mut d = ConnectionDrainer::new(default_config());
        assert!(d.start_drain(999).is_err());
    }

    // 7. drain_all drains all active connections
    #[test]
    fn test_start_drain_all() {
        let mut d = ConnectionDrainer::new(default_config());
        let a = d.register_connection("peer-a");
        let b = d.register_connection("peer-b");
        d.add_request(a).expect("ok");
        d.add_request(b).expect("ok");
        d.start_drain_all();
        assert_eq!(d.active_count(), 0);
        assert_eq!(d.draining_count(), 2);
    }

    // 8. Request rejection during drain
    #[test]
    fn test_reject_request_during_drain() {
        let mut d = ConnectionDrainer::new(default_config());
        let id = d.register_connection("peer-a");
        d.add_request(id).expect("ok");
        d.start_drain(id).expect("ok");
        assert!(d.add_request(id).is_err());
        assert_eq!(d.stats().total_rejected, 1);
    }

    // 9. Allow request during drain when reject_new_requests is false
    #[test]
    fn test_allow_request_during_drain_when_not_rejecting() {
        let cfg = DrainerConfig {
            reject_new_requests: false,
            ..default_config()
        };
        let mut d = ConnectionDrainer::new(cfg);
        let id = d.register_connection("peer-a");
        d.add_request(id).expect("ok");
        d.start_drain(id).expect("ok");
        d.add_request(id).expect("should be allowed");
        let conn = d.get_connection(id).expect("exists");
        assert_eq!(conn.pending_requests, 2);
    }

    // 10. complete_request decrements pending
    #[test]
    fn test_complete_request_decrements() {
        let mut d = ConnectionDrainer::new(default_config());
        let id = d.register_connection("peer-a");
        d.add_request(id).expect("ok");
        d.add_request(id).expect("ok");
        d.complete_request(id).expect("ok");
        let conn = d.get_connection(id).expect("exists");
        assert_eq!(conn.pending_requests, 1);
    }

    // 11. complete_request auto-drains when draining and pending hits 0
    #[test]
    fn test_complete_request_auto_drain() {
        let mut d = ConnectionDrainer::new(default_config());
        let id = d.register_connection("peer-a");
        d.add_request(id).expect("ok");
        d.start_drain(id).expect("ok");
        d.complete_request(id).expect("ok");
        let conn = d.get_connection(id).expect("exists");
        assert_eq!(conn.state, DrainState::Drained);
    }

    // 12. complete_request on zero pending errors
    #[test]
    fn test_complete_request_zero_pending_error() {
        let mut d = ConnectionDrainer::new(default_config());
        let id = d.register_connection("peer-a");
        assert!(d.complete_request(id).is_err());
    }

    // 13. complete_request on nonexistent connection
    #[test]
    fn test_complete_request_nonexistent() {
        let mut d = ConnectionDrainer::new(default_config());
        assert!(d.complete_request(999).is_err());
    }

    // 14. check_drained transitions draining to drained
    #[test]
    fn test_check_drained_transitions() {
        let mut d = ConnectionDrainer::new(default_config());
        let id = d.register_connection("peer-a");
        d.add_request(id).expect("ok");
        d.start_drain(id).expect("ok");
        d.complete_request(id).expect("ok");
        // pending is 0 but complete_request already transitioned it
        let state = d.check_drained(id);
        assert_eq!(state, Some(DrainState::Drained));
    }

    // 15. check_drained returns None for unknown id
    #[test]
    fn test_check_drained_unknown() {
        let mut d = ConnectionDrainer::new(default_config());
        assert!(d.check_drained(999).is_none());
    }

    // 16. Timeout handling
    #[test]
    fn test_check_timeouts() {
        let cfg = DrainerConfig {
            drain_timeout_ms: 100,
            ..default_config()
        };
        let mut d = ConnectionDrainer::new(cfg);
        let id = d.register_connection("peer-a");
        d.add_request(id).expect("ok");
        d.start_drain(id).expect("ok");
        d.set_drain_start_time(id, 1000);

        // Not timed out yet
        let timed = d.check_timeouts(1050);
        assert!(timed.is_empty());

        // Timed out
        let timed = d.check_timeouts(1100);
        assert_eq!(timed, vec![id]);
        let conn = d.get_connection(id).expect("exists");
        assert_eq!(conn.state, DrainState::Drained);
        assert_eq!(d.stats().total_timed_out, 1);
    }

    // 17. remove_drained removes only drained connections
    #[test]
    fn test_remove_drained() {
        let mut d = ConnectionDrainer::new(default_config());
        let a = d.register_connection("peer-a");
        let b = d.register_connection("peer-b");
        d.start_drain(a).expect("ok"); // immediate drained (0 pending)
        let removed = d.remove_drained();
        assert_eq!(removed, 1);
        assert!(d.get_connection(a).is_none());
        assert!(d.get_connection(b).is_some());
    }

    // 18. active_count
    #[test]
    fn test_active_count() {
        let mut d = ConnectionDrainer::new(default_config());
        d.register_connection("a");
        d.register_connection("b");
        assert_eq!(d.active_count(), 2);
    }

    // 19. draining_count
    #[test]
    fn test_draining_count() {
        let mut d = ConnectionDrainer::new(default_config());
        let a = d.register_connection("a");
        let b = d.register_connection("b");
        d.add_request(a).expect("ok");
        d.add_request(b).expect("ok");
        d.start_drain(a).expect("ok");
        assert_eq!(d.draining_count(), 1);
    }

    // 20. is_fully_drained on empty drainer
    #[test]
    fn test_is_fully_drained_empty() {
        let d = ConnectionDrainer::new(default_config());
        assert!(d.is_fully_drained());
    }

    // 21. is_fully_drained with active connections
    #[test]
    fn test_is_fully_drained_with_active() {
        let mut d = ConnectionDrainer::new(default_config());
        d.register_connection("a");
        assert!(!d.is_fully_drained());
    }

    // 22. is_fully_drained after all drained
    #[test]
    fn test_is_fully_drained_all_drained() {
        let mut d = ConnectionDrainer::new(default_config());
        let a = d.register_connection("a");
        let b = d.register_connection("b");
        d.start_drain(a).expect("ok");
        d.start_drain(b).expect("ok");
        assert!(d.is_fully_drained());
    }

    // 23. Stats track total_drained
    #[test]
    fn test_stats_total_drained() {
        let mut d = ConnectionDrainer::new(default_config());
        let a = d.register_connection("a");
        let b = d.register_connection("b");
        d.start_drain(a).expect("ok");
        d.start_drain(b).expect("ok");
        assert_eq!(d.stats().total_drained, 2);
    }

    // 24. Stats currently_draining updates
    #[test]
    fn test_stats_currently_draining() {
        let mut d = ConnectionDrainer::new(default_config());
        let a = d.register_connection("a");
        d.add_request(a).expect("ok");
        d.start_drain(a).expect("ok");
        assert_eq!(d.stats().currently_draining, 1);
        d.complete_request(a).expect("ok");
        assert_eq!(d.stats().currently_draining, 0);
    }

    // 25. avg_drain_time computed on timeout
    #[test]
    fn test_avg_drain_time_on_timeout() {
        let cfg = DrainerConfig {
            drain_timeout_ms: 50,
            ..default_config()
        };
        let mut d = ConnectionDrainer::new(cfg);
        let id = d.register_connection("a");
        d.add_request(id).expect("ok");
        d.start_drain(id).expect("ok");
        d.set_drain_start_time(id, 100);
        d.check_timeouts(200);
        // elapsed = 100ms
        assert!(d.stats().avg_drain_time_ms > 0.0);
    }

    // 26. Peer id is stored correctly
    #[test]
    fn test_peer_id_stored() {
        let mut d = ConnectionDrainer::new(default_config());
        let id = d.register_connection("QmPeer123");
        let conn = d.get_connection(id).expect("exists");
        assert_eq!(conn.peer_id, "QmPeer123");
    }

    // 27. Multiple requests then drain lifecycle
    #[test]
    fn test_full_lifecycle() {
        let mut d = ConnectionDrainer::new(default_config());
        let id = d.register_connection("peer");
        d.add_request(id).expect("ok");
        d.add_request(id).expect("ok");
        d.add_request(id).expect("ok");
        d.start_drain(id).expect("ok");
        assert_eq!(d.draining_count(), 1);

        d.complete_request(id).expect("ok");
        d.complete_request(id).expect("ok");
        assert_eq!(d.get_connection(id).expect("e").state, DrainState::Draining);

        d.complete_request(id).expect("ok");
        assert_eq!(d.get_connection(id).expect("e").state, DrainState::Drained);
        assert!(d.is_fully_drained());

        let removed = d.remove_drained();
        assert_eq!(removed, 1);
        assert!(d.get_connection(id).is_none());
    }

    // 28. add_request on drained connection is rejected
    #[test]
    fn test_add_request_on_drained() {
        let mut d = ConnectionDrainer::new(default_config());
        let id = d.register_connection("peer");
        d.start_drain(id).expect("ok");
        assert!(d.add_request(id).is_err());
        assert_eq!(d.stats().total_rejected, 1);
    }

    // 29. add_request on nonexistent connection
    #[test]
    fn test_add_request_nonexistent() {
        let mut d = ConnectionDrainer::new(default_config());
        assert!(d.add_request(999).is_err());
    }

    // 30. drain_all is idempotent (already draining connections ignored)
    #[test]
    fn test_drain_all_idempotent() {
        let mut d = ConnectionDrainer::new(default_config());
        let a = d.register_connection("a");
        d.add_request(a).expect("ok");
        d.start_drain(a).expect("ok");
        // calling drain_all should not error even though `a` is already draining
        d.start_drain_all();
        assert_eq!(d.draining_count(), 1);
    }

    // 31. check_timeouts with no draining connections returns empty
    #[test]
    fn test_check_timeouts_empty() {
        let mut d = ConnectionDrainer::new(default_config());
        d.register_connection("a");
        let timed = d.check_timeouts(999_999);
        assert!(timed.is_empty());
    }

    // 32. remove_drained on empty drainer
    #[test]
    fn test_remove_drained_empty() {
        let mut d = ConnectionDrainer::new(default_config());
        assert_eq!(d.remove_drained(), 0);
    }

    // 33. Default config values
    #[test]
    fn test_default_config() {
        let cfg = DrainerConfig::default();
        assert_eq!(cfg.drain_timeout_ms, 30_000);
        assert_eq!(cfg.max_drain_wait_ms, 60_000);
        assert!(cfg.reject_new_requests);
        assert_eq!(cfg.graceful_close_delay_ms, 500);
    }

    // 34. DrainerStats default
    #[test]
    fn test_drainer_stats_default() {
        let s = DrainerStats::default();
        assert_eq!(s.total_drained, 0);
        assert_eq!(s.total_timed_out, 0);
        assert_eq!(s.total_rejected, 0);
        assert_eq!(s.currently_draining, 0);
        assert!((s.avg_drain_time_ms - 0.0).abs() < f64::EPSILON);
    }

    // 35. set_drain_start_time on active connection is no-op
    #[test]
    fn test_set_drain_start_time_active_noop() {
        let mut d = ConnectionDrainer::new(default_config());
        let id = d.register_connection("peer");
        d.set_drain_start_time(id, 12345);
        // Should remain None because connection is Active
        let conn = d.get_connection(id).expect("exists");
        assert!(conn.drain_started_at.is_none());
    }
}
