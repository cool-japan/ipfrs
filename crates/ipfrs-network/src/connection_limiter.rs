//! Peer Connection Limiter
//!
//! Rate-based connection limiting per peer and globally. Enforces per-peer
//! connection caps, a global connection ceiling, and a cooldown period
//! (measured in ticks) between successive connections from the same peer.

use std::collections::HashMap;

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration for [`PeerConnectionLimiter`].
#[derive(Debug, Clone)]
pub struct LimiterConfig {
    /// Maximum simultaneous connections allowed from a single peer.
    pub max_connections_per_peer: usize,
    /// Maximum total active connections across all peers.
    pub max_total_connections: usize,
    /// Minimum number of ticks that must elapse between connections from the
    /// same peer.
    pub cooldown_ticks: u64,
}

impl Default for LimiterConfig {
    fn default() -> Self {
        Self {
            max_connections_per_peer: 5,
            max_total_connections: 200,
            cooldown_ticks: 10,
        }
    }
}

// ── Per-peer tracking ────────────────────────────────────────────────────────

/// Per-peer connection information tracked by the limiter.
#[derive(Debug, Clone)]
pub struct PeerConnectionInfo {
    /// Identifier of the peer.
    pub peer_id: String,
    /// Number of currently active connections from this peer.
    pub active_connections: usize,
    /// Tick at which the most recent connection was accepted.
    pub last_connection_tick: u64,
    /// Lifetime count of accepted connections for this peer.
    pub total_connections: u64,
    /// Lifetime count of rejected connection attempts for this peer.
    pub total_rejections: u64,
}

impl PeerConnectionInfo {
    fn new(peer_id: String) -> Self {
        Self {
            peer_id,
            active_connections: 0,
            last_connection_tick: 0,
            total_connections: 0,
            total_rejections: 0,
        }
    }
}

// ── Aggregate statistics ─────────────────────────────────────────────────────

/// Aggregate statistics snapshot returned by [`PeerConnectionLimiter::stats`].
#[derive(Debug, Clone)]
pub struct LimiterStats {
    /// Total active connections across all peers.
    pub total_active: usize,
    /// Number of distinct peers being tracked.
    pub tracked_peers: usize,
    /// Lifetime count of accepted connections.
    pub total_accepted: u64,
    /// Lifetime count of rejected connection attempts.
    pub total_rejected: u64,
}

// ── Limiter ──────────────────────────────────────────────────────────────────

/// Rate-based connection limiter that enforces per-peer limits, a global cap,
/// and a cooldown period between connections from the same peer.
pub struct PeerConnectionLimiter {
    config: LimiterConfig,
    peers: HashMap<String, PeerConnectionInfo>,
    current_tick: u64,
    total_active: usize,
    total_accepted: u64,
    total_rejected: u64,
}

impl PeerConnectionLimiter {
    /// Create a new limiter with the given configuration.
    pub fn new(config: LimiterConfig) -> Self {
        Self {
            config,
            peers: HashMap::new(),
            current_tick: 0,
            total_active: 0,
            total_accepted: 0,
            total_rejected: 0,
        }
    }

    /// Attempt to accept a connection from `peer_id`.
    ///
    /// Returns `Ok(())` if the connection is accepted, or `Err` with a
    /// human-readable reason if rejected.
    pub fn try_connect(&mut self, peer_id: &str) -> Result<(), String> {
        // Global limit
        if self.total_active >= self.config.max_total_connections {
            self.record_rejection(peer_id);
            return Err(format!(
                "global connection limit reached ({}/{})",
                self.total_active, self.config.max_total_connections
            ));
        }

        let info = self
            .peers
            .entry(peer_id.to_string())
            .or_insert_with(|| PeerConnectionInfo::new(peer_id.to_string()));

        // Per-peer limit
        if info.active_connections >= self.config.max_connections_per_peer {
            info.total_rejections += 1;
            self.total_rejected += 1;
            return Err(format!(
                "per-peer connection limit reached for {} ({}/{})",
                peer_id, info.active_connections, self.config.max_connections_per_peer
            ));
        }

        // Cooldown check — only applies if the peer has had at least one
        // previous connection (total_connections > 0).
        if info.total_connections > 0 {
            let elapsed = self.current_tick.saturating_sub(info.last_connection_tick);
            if elapsed < self.config.cooldown_ticks {
                info.total_rejections += 1;
                self.total_rejected += 1;
                return Err(format!(
                    "cooldown active for {} ({} ticks remaining)",
                    peer_id,
                    self.config.cooldown_ticks - elapsed
                ));
            }
        }

        // Accept
        info.active_connections += 1;
        info.last_connection_tick = self.current_tick;
        info.total_connections += 1;
        self.total_active += 1;
        self.total_accepted += 1;

        Ok(())
    }

    /// Record a disconnection for `peer_id`.
    ///
    /// Returns `Err` if the peer has no active connections or is unknown.
    pub fn disconnect(&mut self, peer_id: &str) -> Result<(), String> {
        let info = self
            .peers
            .get_mut(peer_id)
            .ok_or_else(|| format!("unknown peer: {peer_id}"))?;

        if info.active_connections == 0 {
            return Err(format!("no active connections for peer: {peer_id}"));
        }

        info.active_connections -= 1;
        self.total_active -= 1;
        Ok(())
    }

    /// Check whether a connection from `peer_id` would be allowed **without**
    /// modifying any state.
    pub fn is_allowed(&self, peer_id: &str) -> bool {
        if self.total_active >= self.config.max_total_connections {
            return false;
        }

        if let Some(info) = self.peers.get(peer_id) {
            if info.active_connections >= self.config.max_connections_per_peer {
                return false;
            }
            if info.total_connections > 0 {
                let elapsed = self.current_tick.saturating_sub(info.last_connection_tick);
                if elapsed < self.config.cooldown_ticks {
                    return false;
                }
            }
        }

        true
    }

    /// Return the number of active connections for the given peer, or `0` if
    /// the peer is not tracked.
    pub fn active_for_peer(&self, peer_id: &str) -> usize {
        self.peers.get(peer_id).map_or(0, |i| i.active_connections)
    }

    /// Return the total number of active connections across all peers.
    pub fn total_active(&self) -> usize {
        self.total_active
    }

    /// Advance the internal clock by one tick.
    pub fn tick(&mut self) {
        self.current_tick += 1;
    }

    /// Clear all connection info for the specified peer. The peer is removed
    /// from the internal tracking map and the global active count is adjusted.
    pub fn reset_peer(&mut self, peer_id: &str) {
        if let Some(info) = self.peers.remove(peer_id) {
            self.total_active = self.total_active.saturating_sub(info.active_connections);
        }
    }

    /// Return an aggregate statistics snapshot.
    pub fn stats(&self) -> LimiterStats {
        LimiterStats {
            total_active: self.total_active,
            tracked_peers: self.peers.len(),
            total_accepted: self.total_accepted,
            total_rejected: self.total_rejected,
        }
    }

    /// Return a reference to the current configuration.
    pub fn config(&self) -> &LimiterConfig {
        &self.config
    }

    /// Return the current tick value.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// Return a reference to a peer's info, if tracked.
    pub fn peer_info(&self, peer_id: &str) -> Option<&PeerConnectionInfo> {
        self.peers.get(peer_id)
    }

    // ── helpers ──────────────────────────────────────────────────────────

    /// Record a rejection for global-limit hits (peer may not exist yet).
    fn record_rejection(&mut self, peer_id: &str) {
        self.peers
            .entry(peer_id.to_string())
            .or_insert_with(|| PeerConnectionInfo::new(peer_id.to_string()))
            .total_rejections += 1;
        self.total_rejected += 1;
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_limiter() -> PeerConnectionLimiter {
        PeerConnectionLimiter::new(LimiterConfig::default())
    }

    // --- per-peer limit ---

    #[test]
    fn per_peer_limit_allows_up_to_max() {
        let mut lim = default_limiter();
        // Advance past cooldown for each connection
        for _ in 0..5 {
            lim.tick(); // ensure cooldown passes between each
            for _ in 0..lim.config.cooldown_ticks {
                lim.tick();
            }
            assert!(lim.try_connect("peer-a").is_ok());
        }
        assert_eq!(lim.active_for_peer("peer-a"), 5);
    }

    #[test]
    fn per_peer_limit_rejects_excess() {
        let mut lim = PeerConnectionLimiter::new(LimiterConfig {
            max_connections_per_peer: 2,
            cooldown_ticks: 0,
            ..LimiterConfig::default()
        });
        assert!(lim.try_connect("p").is_ok());
        assert!(lim.try_connect("p").is_ok());
        let err = lim.try_connect("p").unwrap_err();
        assert!(err.contains("per-peer"));
    }

    #[test]
    fn per_peer_limit_after_disconnect() {
        let mut lim = PeerConnectionLimiter::new(LimiterConfig {
            max_connections_per_peer: 1,
            cooldown_ticks: 0,
            ..LimiterConfig::default()
        });
        assert!(lim.try_connect("p").is_ok());
        assert!(lim.try_connect("p").is_err());
        assert!(lim.disconnect("p").is_ok());
        assert!(lim.try_connect("p").is_ok());
    }

    // --- global limit ---

    #[test]
    fn global_limit_rejects_excess() {
        let mut lim = PeerConnectionLimiter::new(LimiterConfig {
            max_connections_per_peer: 100,
            max_total_connections: 3,
            cooldown_ticks: 0,
        });
        assert!(lim.try_connect("a").is_ok());
        assert!(lim.try_connect("b").is_ok());
        assert!(lim.try_connect("c").is_ok());
        let err = lim.try_connect("d").unwrap_err();
        assert!(err.contains("global"));
    }

    #[test]
    fn global_limit_allows_after_disconnect() {
        let mut lim = PeerConnectionLimiter::new(LimiterConfig {
            max_connections_per_peer: 100,
            max_total_connections: 2,
            cooldown_ticks: 0,
        });
        assert!(lim.try_connect("a").is_ok());
        assert!(lim.try_connect("b").is_ok());
        assert!(lim.try_connect("c").is_err());
        assert!(lim.disconnect("a").is_ok());
        assert!(lim.try_connect("c").is_ok());
    }

    // --- cooldown ---

    #[test]
    fn cooldown_enforced() {
        let mut lim = PeerConnectionLimiter::new(LimiterConfig {
            max_connections_per_peer: 10,
            max_total_connections: 100,
            cooldown_ticks: 5,
        });
        assert!(lim.try_connect("p").is_ok());
        // Immediately try again — should be rejected due to cooldown
        let err = lim.try_connect("p").unwrap_err();
        assert!(err.contains("cooldown"));
    }

    #[test]
    fn cooldown_passes_after_enough_ticks() {
        let mut lim = PeerConnectionLimiter::new(LimiterConfig {
            max_connections_per_peer: 10,
            max_total_connections: 100,
            cooldown_ticks: 3,
        });
        assert!(lim.try_connect("p").is_ok());
        for _ in 0..3 {
            lim.tick();
        }
        assert!(lim.try_connect("p").is_ok());
    }

    #[test]
    fn cooldown_still_active_one_tick_short() {
        let mut lim = PeerConnectionLimiter::new(LimiterConfig {
            max_connections_per_peer: 10,
            max_total_connections: 100,
            cooldown_ticks: 5,
        });
        assert!(lim.try_connect("p").is_ok());
        for _ in 0..4 {
            lim.tick();
        }
        assert!(lim.try_connect("p").is_err());
        lim.tick();
        assert!(lim.try_connect("p").is_ok());
    }

    #[test]
    fn cooldown_zero_allows_immediate() {
        let mut lim = PeerConnectionLimiter::new(LimiterConfig {
            cooldown_ticks: 0,
            ..LimiterConfig::default()
        });
        assert!(lim.try_connect("p").is_ok());
        assert!(lim.try_connect("p").is_ok());
    }

    // --- disconnect ---

    #[test]
    fn disconnect_decrements_active() {
        let mut lim = PeerConnectionLimiter::new(LimiterConfig {
            cooldown_ticks: 0,
            ..LimiterConfig::default()
        });
        assert!(lim.try_connect("p").is_ok());
        assert!(lim.try_connect("p").is_ok());
        assert_eq!(lim.active_for_peer("p"), 2);
        assert!(lim.disconnect("p").is_ok());
        assert_eq!(lim.active_for_peer("p"), 1);
    }

    #[test]
    fn disconnect_unknown_peer_errors() {
        let mut lim = default_limiter();
        assert!(lim.disconnect("ghost").is_err());
    }

    #[test]
    fn disconnect_zero_active_errors() {
        let mut lim = PeerConnectionLimiter::new(LimiterConfig {
            cooldown_ticks: 0,
            ..LimiterConfig::default()
        });
        assert!(lim.try_connect("p").is_ok());
        assert!(lim.disconnect("p").is_ok());
        let err = lim.disconnect("p").unwrap_err();
        assert!(err.contains("no active"));
    }

    #[test]
    fn disconnect_updates_total_active() {
        let mut lim = PeerConnectionLimiter::new(LimiterConfig {
            cooldown_ticks: 0,
            ..LimiterConfig::default()
        });
        assert!(lim.try_connect("a").is_ok());
        assert!(lim.try_connect("b").is_ok());
        assert_eq!(lim.total_active(), 2);
        assert!(lim.disconnect("a").is_ok());
        assert_eq!(lim.total_active(), 1);
    }

    // --- is_allowed ---

    #[test]
    fn is_allowed_true_for_new_peer() {
        let lim = default_limiter();
        assert!(lim.is_allowed("new-peer"));
    }

    #[test]
    fn is_allowed_false_when_per_peer_full() {
        let mut lim = PeerConnectionLimiter::new(LimiterConfig {
            max_connections_per_peer: 1,
            cooldown_ticks: 0,
            ..LimiterConfig::default()
        });
        assert!(lim.try_connect("p").is_ok());
        assert!(!lim.is_allowed("p"));
    }

    #[test]
    fn is_allowed_false_when_global_full() {
        let mut lim = PeerConnectionLimiter::new(LimiterConfig {
            max_total_connections: 1,
            cooldown_ticks: 0,
            ..LimiterConfig::default()
        });
        assert!(lim.try_connect("a").is_ok());
        assert!(!lim.is_allowed("b"));
    }

    #[test]
    fn is_allowed_false_during_cooldown() {
        let mut lim = PeerConnectionLimiter::new(LimiterConfig {
            cooldown_ticks: 5,
            ..LimiterConfig::default()
        });
        assert!(lim.try_connect("p").is_ok());
        assert!(!lim.is_allowed("p"));
    }

    #[test]
    fn is_allowed_does_not_mutate_state() {
        let mut lim = PeerConnectionLimiter::new(LimiterConfig {
            cooldown_ticks: 0,
            ..LimiterConfig::default()
        });
        assert!(lim.try_connect("p").is_ok());
        let before = lim.stats();
        let _ = lim.is_allowed("p");
        let after = lim.stats();
        assert_eq!(before.total_accepted, after.total_accepted);
        assert_eq!(before.total_rejected, after.total_rejected);
    }

    // --- stats ---

    #[test]
    fn stats_tracking_accepted() {
        let mut lim = PeerConnectionLimiter::new(LimiterConfig {
            cooldown_ticks: 0,
            ..LimiterConfig::default()
        });
        assert!(lim.try_connect("a").is_ok());
        assert!(lim.try_connect("b").is_ok());
        let s = lim.stats();
        assert_eq!(s.total_accepted, 2);
        assert_eq!(s.total_rejected, 0);
        assert_eq!(s.total_active, 2);
        assert_eq!(s.tracked_peers, 2);
    }

    #[test]
    fn stats_tracking_rejected() {
        let mut lim = PeerConnectionLimiter::new(LimiterConfig {
            max_connections_per_peer: 1,
            cooldown_ticks: 0,
            ..LimiterConfig::default()
        });
        assert!(lim.try_connect("p").is_ok());
        let _ = lim.try_connect("p"); // rejected
        let s = lim.stats();
        assert_eq!(s.total_accepted, 1);
        assert_eq!(s.total_rejected, 1);
    }

    #[test]
    fn stats_global_rejection_counted() {
        let mut lim = PeerConnectionLimiter::new(LimiterConfig {
            max_total_connections: 1,
            cooldown_ticks: 0,
            ..LimiterConfig::default()
        });
        assert!(lim.try_connect("a").is_ok());
        let _ = lim.try_connect("b");
        let s = lim.stats();
        assert_eq!(s.total_rejected, 1);
    }

    // --- reset_peer ---

    #[test]
    fn reset_peer_removes_tracking() {
        let mut lim = PeerConnectionLimiter::new(LimiterConfig {
            cooldown_ticks: 0,
            ..LimiterConfig::default()
        });
        assert!(lim.try_connect("p").is_ok());
        assert!(lim.try_connect("p").is_ok());
        lim.reset_peer("p");
        assert_eq!(lim.active_for_peer("p"), 0);
        assert_eq!(lim.total_active(), 0);
        assert!(lim.peer_info("p").is_none());
    }

    #[test]
    fn reset_peer_unknown_is_noop() {
        let mut lim = default_limiter();
        lim.reset_peer("ghost"); // should not panic
        assert_eq!(lim.total_active(), 0);
    }

    // --- tick ---

    #[test]
    fn tick_advances_clock() {
        let mut lim = default_limiter();
        assert_eq!(lim.current_tick(), 0);
        lim.tick();
        assert_eq!(lim.current_tick(), 1);
        for _ in 0..9 {
            lim.tick();
        }
        assert_eq!(lim.current_tick(), 10);
    }

    // --- multiple peers ---

    #[test]
    fn multiple_peers_independent_limits() {
        let mut lim = PeerConnectionLimiter::new(LimiterConfig {
            max_connections_per_peer: 2,
            cooldown_ticks: 0,
            ..LimiterConfig::default()
        });
        assert!(lim.try_connect("a").is_ok());
        assert!(lim.try_connect("a").is_ok());
        assert!(lim.try_connect("b").is_ok());
        assert!(lim.try_connect("b").is_ok());
        assert!(lim.try_connect("a").is_err());
        assert!(lim.try_connect("b").is_err());
        assert_eq!(lim.total_active(), 4);
    }

    #[test]
    fn multiple_peers_share_global_limit() {
        let mut lim = PeerConnectionLimiter::new(LimiterConfig {
            max_connections_per_peer: 10,
            max_total_connections: 3,
            cooldown_ticks: 0,
        });
        assert!(lim.try_connect("a").is_ok());
        assert!(lim.try_connect("b").is_ok());
        assert!(lim.try_connect("c").is_ok());
        assert!(lim.try_connect("d").is_err());
    }

    // --- edge cases ---

    #[test]
    fn zero_max_per_peer_always_rejects() {
        let mut lim = PeerConnectionLimiter::new(LimiterConfig {
            max_connections_per_peer: 0,
            cooldown_ticks: 0,
            ..LimiterConfig::default()
        });
        assert!(lim.try_connect("p").is_err());
    }

    #[test]
    fn zero_max_global_always_rejects() {
        let mut lim = PeerConnectionLimiter::new(LimiterConfig {
            max_total_connections: 0,
            cooldown_ticks: 0,
            ..LimiterConfig::default()
        });
        assert!(lim.try_connect("p").is_err());
    }

    #[test]
    fn default_config_values() {
        let cfg = LimiterConfig::default();
        assert_eq!(cfg.max_connections_per_peer, 5);
        assert_eq!(cfg.max_total_connections, 200);
        assert_eq!(cfg.cooldown_ticks, 10);
    }

    #[test]
    fn peer_info_returns_none_for_unknown() {
        let lim = default_limiter();
        assert!(lim.peer_info("unknown").is_none());
    }

    #[test]
    fn peer_info_returns_correct_data() {
        let mut lim = PeerConnectionLimiter::new(LimiterConfig {
            cooldown_ticks: 0,
            ..LimiterConfig::default()
        });
        assert!(lim.try_connect("p").is_ok());
        let info = lim.peer_info("p").expect("peer should be tracked");
        assert_eq!(info.active_connections, 1);
        assert_eq!(info.total_connections, 1);
        assert_eq!(info.total_rejections, 0);
    }

    #[test]
    fn mixed_accept_reject_stats() {
        let mut lim = PeerConnectionLimiter::new(LimiterConfig {
            max_connections_per_peer: 1,
            max_total_connections: 10,
            cooldown_ticks: 0,
        });
        // accept 3
        assert!(lim.try_connect("a").is_ok());
        assert!(lim.try_connect("b").is_ok());
        assert!(lim.try_connect("c").is_ok());
        // reject 2
        let _ = lim.try_connect("a");
        let _ = lim.try_connect("b");
        let s = lim.stats();
        assert_eq!(s.total_accepted, 3);
        assert_eq!(s.total_rejected, 2);
        assert_eq!(s.total_active, 3);
        assert_eq!(s.tracked_peers, 3);
    }

    #[test]
    fn cooldown_independent_per_peer() {
        let mut lim = PeerConnectionLimiter::new(LimiterConfig {
            max_connections_per_peer: 10,
            max_total_connections: 100,
            cooldown_ticks: 3,
        });
        assert!(lim.try_connect("a").is_ok());
        // Advance 3 ticks
        for _ in 0..3 {
            lim.tick();
        }
        // "a" cooldown is over, "b" is fresh
        assert!(lim.try_connect("a").is_ok());
        assert!(lim.try_connect("b").is_ok());
        // "b" is now on cooldown, "a" is too
        assert!(lim.try_connect("a").is_err());
        assert!(lim.try_connect("b").is_err());
    }

    #[test]
    fn reset_then_reconnect() {
        let mut lim = PeerConnectionLimiter::new(LimiterConfig {
            cooldown_ticks: 100,
            ..LimiterConfig::default()
        });
        assert!(lim.try_connect("p").is_ok());
        // Cooldown would block reconnection
        assert!(lim.try_connect("p").is_err());
        // Reset clears everything — including cooldown history
        lim.reset_peer("p");
        assert!(lim.try_connect("p").is_ok());
    }
}
