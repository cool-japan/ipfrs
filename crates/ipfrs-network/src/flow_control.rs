//! Window-based flow control for peer data transfer.
//!
//! Provides per-peer sliding window flow control to prevent overwhelming
//! peers with data faster than they can process. Each peer gets an independent
//! window that grows on successful transfers and shrinks on congestion.

use std::collections::HashMap;

/// Configuration for the flow control system.
#[derive(Debug, Clone)]
pub struct FlowControlConfig {
    /// Initial window size in bytes for new peers (default: 65536 = 64KB).
    pub initial_window: u64,
    /// Maximum window size in bytes (default: 1_048_576 = 1MB).
    pub max_window: u64,
    /// Minimum window size in bytes (default: 4096).
    pub min_window: u64,
    /// Multiplicative factor for growing the window (default: 1.5).
    pub growth_factor: f64,
    /// Multiplicative factor for shrinking the window (default: 0.5).
    pub shrink_factor: f64,
}

impl Default for FlowControlConfig {
    fn default() -> Self {
        Self {
            initial_window: 65_536,
            max_window: 1_048_576,
            min_window: 4096,
            growth_factor: 1.5,
            shrink_factor: 0.5,
        }
    }
}

/// Per-peer flow window tracking bytes in flight and acknowledgements.
#[derive(Debug, Clone)]
pub struct FlowWindow {
    /// Peer identifier.
    pub peer_id: String,
    /// Maximum bytes allowed in flight for this peer.
    pub window_size: u64,
    /// Bytes currently in flight (sent but not yet acknowledged).
    pub bytes_in_flight: u64,
    /// Total bytes acknowledged by this peer.
    pub bytes_acked: u64,
    /// Total bytes sent to this peer.
    pub bytes_sent: u64,
    /// Number of times a send was rejected because the window was full.
    pub stall_count: u64,
}

/// Aggregate statistics across all peers.
#[derive(Debug, Clone)]
pub struct FlowControlStats {
    /// Number of peers with active flow windows.
    pub peer_count: usize,
    /// Total stalls across all peers.
    pub total_stalls: u64,
    /// Average utilization (bytes_in_flight / window_size) across all peers.
    pub avg_utilization: f64,
}

/// Window-based flow control manager for multiple peers.
///
/// Tracks per-peer send windows, automatically creates windows for new peers,
/// and provides grow/shrink operations for adaptive congestion control.
pub struct PeerFlowControl {
    config: FlowControlConfig,
    windows: HashMap<String, FlowWindow>,
    total_stalls: u64,
}

impl PeerFlowControl {
    /// Create a new `PeerFlowControl` with the given configuration.
    pub fn new(config: FlowControlConfig) -> Self {
        Self {
            config,
            windows: HashMap::new(),
            total_stalls: 0,
        }
    }

    /// Check whether `bytes` can be sent to `peer_id` without exceeding
    /// the window. Returns `true` even if the peer has no window yet
    /// (it would be auto-created on `send`).
    pub fn can_send(&self, peer_id: &str, bytes: u64) -> bool {
        match self.windows.get(peer_id) {
            Some(w) => w.bytes_in_flight.saturating_add(bytes) <= w.window_size,
            None => bytes <= self.config.initial_window,
        }
    }

    /// Record `bytes` as sent to `peer_id`.
    ///
    /// Auto-creates a window for new peers. Returns an error if sending
    /// `bytes` would exceed the peer's window size and increments the
    /// stall counter.
    pub fn send(&mut self, peer_id: &str, bytes: u64) -> Result<(), String> {
        let window = self
            .windows
            .entry(peer_id.to_string())
            .or_insert_with(|| FlowWindow {
                peer_id: peer_id.to_string(),
                window_size: self.config.initial_window,
                bytes_in_flight: 0,
                bytes_acked: 0,
                bytes_sent: 0,
                stall_count: 0,
            });

        if window.bytes_in_flight.saturating_add(bytes) > window.window_size {
            window.stall_count += 1;
            self.total_stalls += 1;
            return Err(format!(
                "window full for peer {}: in_flight={} + bytes={} > window={}",
                peer_id, window.bytes_in_flight, bytes, window.window_size
            ));
        }

        window.bytes_in_flight = window.bytes_in_flight.saturating_add(bytes);
        window.bytes_sent = window.bytes_sent.saturating_add(bytes);
        Ok(())
    }

    /// Acknowledge `bytes` from `peer_id`, reducing bytes in flight.
    ///
    /// If the peer has no window this is a no-op. The in-flight counter
    /// is clamped to zero (will not underflow).
    pub fn ack(&mut self, peer_id: &str, bytes: u64) {
        if let Some(w) = self.windows.get_mut(peer_id) {
            w.bytes_in_flight = w.bytes_in_flight.saturating_sub(bytes);
            w.bytes_acked = w.bytes_acked.saturating_add(bytes);
        }
    }

    /// Grow the window for `peer_id` by `growth_factor`, clamped to `max_window`.
    ///
    /// No-op if the peer has no window.
    pub fn grow_window(&mut self, peer_id: &str) {
        if let Some(w) = self.windows.get_mut(peer_id) {
            let new_size = (w.window_size as f64 * self.config.growth_factor) as u64;
            w.window_size = new_size.min(self.config.max_window);
        }
    }

    /// Shrink the window for `peer_id` by `shrink_factor`, clamped to `min_window`.
    ///
    /// No-op if the peer has no window.
    pub fn shrink_window(&mut self, peer_id: &str) {
        if let Some(w) = self.windows.get_mut(peer_id) {
            let new_size = (w.window_size as f64 * self.config.shrink_factor) as u64;
            w.window_size = new_size.max(self.config.min_window);
        }
    }

    /// Get a reference to the flow window for `peer_id`, if it exists.
    pub fn get_window(&self, peer_id: &str) -> Option<&FlowWindow> {
        self.windows.get(peer_id)
    }

    /// Compute the utilization ratio (bytes_in_flight / window_size) for
    /// `peer_id`. Returns `None` if the peer has no window.
    pub fn utilization(&self, peer_id: &str) -> Option<f64> {
        self.windows.get(peer_id).map(|w| {
            if w.window_size == 0 {
                0.0
            } else {
                w.bytes_in_flight as f64 / w.window_size as f64
            }
        })
    }

    /// Remove the flow window for `peer_id`. Returns `true` if the peer
    /// was present.
    pub fn remove_peer(&mut self, peer_id: &str) -> bool {
        self.windows.remove(peer_id).is_some()
    }

    /// Return the number of peers with active flow windows.
    pub fn peer_count(&self) -> usize {
        self.windows.len()
    }

    /// Compute aggregate statistics across all peers.
    pub fn stats(&self) -> FlowControlStats {
        let peer_count = self.windows.len();
        let avg_utilization = if peer_count == 0 {
            0.0
        } else {
            let total: f64 = self
                .windows
                .values()
                .map(|w| {
                    if w.window_size == 0 {
                        0.0
                    } else {
                        w.bytes_in_flight as f64 / w.window_size as f64
                    }
                })
                .sum();
            total / peer_count as f64
        };

        FlowControlStats {
            peer_count,
            total_stalls: self.total_stalls,
            avg_utilization,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_ctrl() -> PeerFlowControl {
        PeerFlowControl::new(FlowControlConfig::default())
    }

    // ---- Basic send / ack lifecycle ----

    #[test]
    fn test_send_ack_lifecycle() {
        let mut fc = default_ctrl();
        fc.send("peer1", 1000).expect("send should succeed");
        let w = fc.get_window("peer1").expect("window should exist");
        assert_eq!(w.bytes_in_flight, 1000);
        assert_eq!(w.bytes_sent, 1000);

        fc.ack("peer1", 500);
        let w = fc.get_window("peer1").expect("window should exist");
        assert_eq!(w.bytes_in_flight, 500);
        assert_eq!(w.bytes_acked, 500);
    }

    #[test]
    fn test_send_ack_full_cycle() {
        let mut fc = default_ctrl();
        fc.send("p", 2000).expect("ok");
        fc.ack("p", 2000);
        let w = fc.get_window("p").expect("exists");
        assert_eq!(w.bytes_in_flight, 0);
        assert_eq!(w.bytes_acked, 2000);
        assert_eq!(w.bytes_sent, 2000);
    }

    // ---- can_send ----

    #[test]
    fn test_can_send_new_peer() {
        let fc = default_ctrl();
        assert!(fc.can_send("unknown", 65_536));
        assert!(!fc.can_send("unknown", 65_537));
    }

    #[test]
    fn test_can_send_existing_peer() {
        let mut fc = default_ctrl();
        fc.send("p", 60_000).expect("ok");
        assert!(fc.can_send("p", 5_536));
        assert!(!fc.can_send("p", 5_537));
    }

    #[test]
    fn test_can_send_zero_bytes() {
        let fc = default_ctrl();
        assert!(fc.can_send("any", 0));
    }

    // ---- Window full / stall ----

    #[test]
    fn test_window_full_stall() {
        let mut fc = default_ctrl();
        fc.send("p", 65_536).expect("ok");
        let res = fc.send("p", 1);
        assert!(res.is_err());
        let w = fc.get_window("p").expect("exists");
        assert_eq!(w.stall_count, 1);
        assert_eq!(fc.total_stalls, 1);
    }

    #[test]
    fn test_multiple_stalls() {
        let mut fc = default_ctrl();
        fc.send("p", 65_536).expect("ok");
        for _ in 0..5 {
            let _ = fc.send("p", 1);
        }
        let w = fc.get_window("p").expect("exists");
        assert_eq!(w.stall_count, 5);
        assert_eq!(fc.total_stalls, 5);
    }

    // ---- Grow window ----

    #[test]
    fn test_grow_window() {
        let mut fc = default_ctrl();
        fc.send("p", 0).expect("ok");
        fc.grow_window("p");
        let w = fc.get_window("p").expect("exists");
        assert_eq!(w.window_size, 98_304); // 65536 * 1.5
    }

    #[test]
    fn test_grow_window_clamp_max() {
        let mut fc = PeerFlowControl::new(FlowControlConfig {
            initial_window: 900_000,
            max_window: 1_000_000,
            ..Default::default()
        });
        fc.send("p", 0).expect("ok");
        fc.grow_window("p"); // 900000 * 1.5 = 1350000, clamped to 1000000
        let w = fc.get_window("p").expect("exists");
        assert_eq!(w.window_size, 1_000_000);
    }

    #[test]
    fn test_grow_window_nonexistent_peer_noop() {
        let mut fc = default_ctrl();
        fc.grow_window("ghost"); // should not panic
        assert!(fc.get_window("ghost").is_none());
    }

    // ---- Shrink window ----

    #[test]
    fn test_shrink_window() {
        let mut fc = default_ctrl();
        fc.send("p", 0).expect("ok");
        fc.shrink_window("p");
        let w = fc.get_window("p").expect("exists");
        assert_eq!(w.window_size, 32_768); // 65536 * 0.5
    }

    #[test]
    fn test_shrink_window_clamp_min() {
        let mut fc = PeerFlowControl::new(FlowControlConfig {
            initial_window: 5_000,
            min_window: 4_096,
            ..Default::default()
        });
        fc.send("p", 0).expect("ok");
        fc.shrink_window("p"); // 5000 * 0.5 = 2500, clamped to 4096
        let w = fc.get_window("p").expect("exists");
        assert_eq!(w.window_size, 4_096);
    }

    #[test]
    fn test_shrink_window_nonexistent_peer_noop() {
        let mut fc = default_ctrl();
        fc.shrink_window("ghost");
        assert!(fc.get_window("ghost").is_none());
    }

    // ---- Utilization ----

    #[test]
    fn test_utilization_no_peer() {
        let fc = default_ctrl();
        assert!(fc.utilization("nope").is_none());
    }

    #[test]
    fn test_utilization_zero_in_flight() {
        let mut fc = default_ctrl();
        fc.send("p", 0).expect("ok");
        let u = fc.utilization("p").expect("exists");
        assert!((u - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_utilization_half() {
        let mut fc = default_ctrl();
        fc.send("p", 32_768).expect("ok"); // half of 65536
        let u = fc.utilization("p").expect("exists");
        assert!((u - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_utilization_full() {
        let mut fc = default_ctrl();
        fc.send("p", 65_536).expect("ok");
        let u = fc.utilization("p").expect("exists");
        assert!((u - 1.0).abs() < f64::EPSILON);
    }

    // ---- Remove peer ----

    #[test]
    fn test_remove_peer_present() {
        let mut fc = default_ctrl();
        fc.send("p", 100).expect("ok");
        assert!(fc.remove_peer("p"));
        assert!(fc.get_window("p").is_none());
        assert_eq!(fc.peer_count(), 0);
    }

    #[test]
    fn test_remove_peer_absent() {
        let mut fc = default_ctrl();
        assert!(!fc.remove_peer("ghost"));
    }

    // ---- Auto-create window ----

    #[test]
    fn test_auto_create_window_on_send() {
        let mut fc = default_ctrl();
        assert!(fc.get_window("p").is_none());
        fc.send("p", 10).expect("ok");
        let w = fc.get_window("p").expect("exists");
        assert_eq!(w.window_size, 65_536);
        assert_eq!(w.peer_id, "p");
    }

    // ---- Peer count ----

    #[test]
    fn test_peer_count_empty() {
        let fc = default_ctrl();
        assert_eq!(fc.peer_count(), 0);
    }

    #[test]
    fn test_peer_count_multiple() {
        let mut fc = default_ctrl();
        fc.send("a", 1).expect("ok");
        fc.send("b", 1).expect("ok");
        fc.send("c", 1).expect("ok");
        assert_eq!(fc.peer_count(), 3);
    }

    // ---- Stats ----

    #[test]
    fn test_stats_empty() {
        let fc = default_ctrl();
        let s = fc.stats();
        assert_eq!(s.peer_count, 0);
        assert_eq!(s.total_stalls, 0);
        assert!((s.avg_utilization - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_with_peers() {
        let mut fc = default_ctrl();
        fc.send("a", 65_536).expect("ok"); // util = 1.0
        fc.send("b", 0).expect("ok"); // util = 0.0
        let _ = fc.send("a", 1); // stall
        let s = fc.stats();
        assert_eq!(s.peer_count, 2);
        assert_eq!(s.total_stalls, 1);
        assert!((s.avg_utilization - 0.5).abs() < f64::EPSILON);
    }

    // ---- Ack on nonexistent peer ----

    #[test]
    fn test_ack_nonexistent_peer_noop() {
        let mut fc = default_ctrl();
        fc.ack("ghost", 1000); // should not panic
    }

    // ---- Ack more than in flight (saturating) ----

    #[test]
    fn test_ack_more_than_in_flight() {
        let mut fc = default_ctrl();
        fc.send("p", 100).expect("ok");
        fc.ack("p", 500); // over-ack
        let w = fc.get_window("p").expect("exists");
        assert_eq!(w.bytes_in_flight, 0);
        assert_eq!(w.bytes_acked, 500);
    }

    // ---- Multiple sends ----

    #[test]
    fn test_multiple_sends_accumulate() {
        let mut fc = default_ctrl();
        fc.send("p", 10_000).expect("ok");
        fc.send("p", 20_000).expect("ok");
        fc.send("p", 30_000).expect("ok");
        let w = fc.get_window("p").expect("exists");
        assert_eq!(w.bytes_in_flight, 60_000);
        assert_eq!(w.bytes_sent, 60_000);
    }

    // ---- Grow then send ----

    #[test]
    fn test_grow_allows_more_sends() {
        let mut fc = default_ctrl();
        fc.send("p", 65_536).expect("fill window");
        assert!(fc.send("p", 1).is_err());
        fc.ack("p", 65_536);
        fc.grow_window("p"); // now 98304
        fc.send("p", 98_304).expect("ok with grown window");
        let w = fc.get_window("p").expect("exists");
        assert_eq!(w.bytes_in_flight, 98_304);
    }

    // ---- Shrink then send fails ----

    #[test]
    fn test_shrink_reduces_capacity() {
        let mut fc = default_ctrl();
        fc.send("p", 0).expect("ok");
        fc.shrink_window("p"); // 32768
        assert!(fc.can_send("p", 32_768));
        assert!(!fc.can_send("p", 32_769));
    }

    // ---- Custom config ----

    #[test]
    fn test_custom_config() {
        let cfg = FlowControlConfig {
            initial_window: 1000,
            max_window: 5000,
            min_window: 100,
            growth_factor: 2.0,
            shrink_factor: 0.25,
        };
        let mut fc = PeerFlowControl::new(cfg);
        fc.send("p", 0).expect("ok");
        assert_eq!(fc.get_window("p").expect("e").window_size, 1000);

        fc.grow_window("p");
        assert_eq!(fc.get_window("p").expect("e").window_size, 2000);

        fc.grow_window("p");
        assert_eq!(fc.get_window("p").expect("e").window_size, 4000);

        fc.grow_window("p"); // 8000, clamped to 5000
        assert_eq!(fc.get_window("p").expect("e").window_size, 5000);

        fc.shrink_window("p"); // 5000 * 0.25 = 1250
        assert_eq!(fc.get_window("p").expect("e").window_size, 1250);

        fc.shrink_window("p"); // 1250 * 0.25 = 312
        fc.shrink_window("p"); // 312 * 0.25 = 78, clamped to 100
        assert_eq!(fc.get_window("p").expect("e").window_size, 100);
    }

    // ---- Stats avg_utilization across many peers ----

    #[test]
    fn test_stats_avg_utilization_three_peers() {
        let mut fc = default_ctrl();
        // 65536 window each
        fc.send("a", 65_536).expect("ok"); // util=1.0
        fc.send("b", 32_768).expect("ok"); // util=0.5
        fc.send("c", 16_384).expect("ok"); // util=0.25
        let s = fc.stats();
        let expected = (1.0 + 0.5 + 0.25) / 3.0;
        assert!((s.avg_utilization - expected).abs() < 1e-10);
    }

    // ---- Stalls across multiple peers ----

    #[test]
    fn test_stalls_across_multiple_peers() {
        let mut fc = default_ctrl();
        fc.send("a", 65_536).expect("ok");
        fc.send("b", 65_536).expect("ok");
        let _ = fc.send("a", 1);
        let _ = fc.send("b", 1);
        let _ = fc.send("a", 1);
        assert_eq!(fc.total_stalls, 3);
        assert_eq!(fc.stats().total_stalls, 3);
    }

    // ---- Default config values ----

    #[test]
    fn test_default_config() {
        let cfg = FlowControlConfig::default();
        assert_eq!(cfg.initial_window, 65_536);
        assert_eq!(cfg.max_window, 1_048_576);
        assert_eq!(cfg.min_window, 4096);
        assert!((cfg.growth_factor - 1.5).abs() < f64::EPSILON);
        assert!((cfg.shrink_factor - 0.5).abs() < f64::EPSILON);
    }

    // ---- Remove then re-add ----

    #[test]
    fn test_remove_and_readd_peer() {
        let mut fc = default_ctrl();
        fc.send("p", 100).expect("ok");
        fc.grow_window("p");
        fc.remove_peer("p");

        fc.send("p", 50).expect("ok");
        let w = fc.get_window("p").expect("exists");
        // Fresh window after removal
        assert_eq!(w.window_size, 65_536);
        assert_eq!(w.bytes_in_flight, 50);
        assert_eq!(w.bytes_sent, 50);
    }
}
