//! Gossip protocol efficiency metrics.
//!
//! Tracks message propagation rates, duplicate detection, and coverage estimation
//! across the peer gossip overlay network.

use std::collections::HashMap;

/// Events emitted by the gossip protocol layer.
#[derive(Clone, Debug, PartialEq)]
pub enum GossipEvent {
    /// A message was received from a peer.
    Received {
        /// Unique message identifier.
        msg_id: u64,
        /// Peer that sent this message.
        from_peer: String,
        /// Number of hops this message has traversed.
        hop_count: u32,
    },
    /// A message was forwarded to one or more peers.
    Forwarded {
        /// Unique message identifier.
        msg_id: u64,
        /// Peers the message was forwarded to.
        to_peers: Vec<String>,
    },
    /// A duplicate message was received (already seen).
    Duplicate {
        /// Unique message identifier.
        msg_id: u64,
        /// Peer that sent the duplicate.
        from_peer: String,
    },
    /// A message has expired from the gossip cache.
    Expired {
        /// Unique message identifier.
        msg_id: u64,
    },
}

/// Per-message trace accumulating all observations of a given message.
#[derive(Clone, Debug, PartialEq)]
pub struct MessageTrace {
    /// Unique message identifier.
    pub msg_id: u64,
    /// Logical tick at which this message was first seen.
    pub first_seen_tick: u64,
    /// Total times the message was received (including duplicates).
    pub received_count: u32,
    /// Number of times this message was forwarded.
    pub forward_count: u32,
    /// Number of duplicate receptions detected.
    pub duplicate_count: u32,
    /// Minimum hop count observed across all receptions.
    pub min_hop_count: u32,
}

impl MessageTrace {
    /// Fraction of receptions that were duplicates.
    ///
    /// Returns `duplicate_count / received_count.max(1)` as an `f64`.
    pub fn redundancy_ratio(&self) -> f64 {
        self.duplicate_count as f64 / self.received_count.max(1) as f64
    }
}

/// Point-in-time snapshot of aggregated gossip metrics.
#[derive(Clone, Debug, PartialEq)]
pub struct GossipMetricsSnapshot {
    /// Total distinct messages seen.
    pub total_messages: u64,
    /// Total duplicate receptions across all messages.
    pub total_duplicates: u64,
    /// Total forward operations across all messages.
    pub total_forwarded: u64,
    /// Average minimum hop count across all message traces.
    pub avg_hop_count: f64,
    /// Ratio of duplicates to total messages (`total_duplicates / total_messages.max(1)`).
    pub duplicate_rate: f64,
}

impl GossipMetricsSnapshot {
    /// Simple coverage approximation: `1.0 - duplicate_rate`.
    ///
    /// A higher duplicate rate implies more redundant paths, suggesting lower
    /// marginal coverage gain per message.
    pub fn estimated_coverage(&self) -> f64 {
        1.0 - self.duplicate_rate
    }
}

/// Tracks gossip protocol efficiency across all observed messages.
#[derive(Debug)]
pub struct PeerGossipMetrics {
    /// Per-message traces, keyed by `msg_id`.
    pub traces: HashMap<u64, MessageTrace>,
    /// Current logical tick (monotonically increasing).
    pub current_tick: u64,
}

impl PeerGossipMetrics {
    /// Create a new, empty `PeerGossipMetrics` instance.
    pub fn new() -> Self {
        Self {
            traces: HashMap::new(),
            current_tick: 0,
        }
    }

    /// Record a gossip event, updating the relevant message trace.
    ///
    /// # Behaviour by variant
    ///
    /// - `Received`: Upsert the trace.  Increment `received_count` and update
    ///   `min_hop_count`.  If this is **not** the first reception (i.e.
    ///   `received_count` was already ≥ 1 before this event), also increment
    ///   `duplicate_count`.
    /// - `Forwarded`: Increment `forward_count` for the trace (no-op if the
    ///   trace does not yet exist).
    /// - `Duplicate`: Increment `duplicate_count` for the trace (no-op if the
    ///   trace does not yet exist).
    /// - `Expired`: No-op; the trace is retained for historical queries.
    pub fn record_event(&mut self, event: GossipEvent) {
        match event {
            GossipEvent::Received {
                msg_id,
                from_peer: _,
                hop_count,
            } => {
                let tick = self.current_tick;
                let trace = self.traces.entry(msg_id).or_insert_with(|| MessageTrace {
                    msg_id,
                    first_seen_tick: tick,
                    received_count: 0,
                    forward_count: 0,
                    duplicate_count: 0,
                    min_hop_count: hop_count,
                });

                // If already received at least once this is a duplicate reception.
                if trace.received_count >= 1 {
                    trace.duplicate_count += 1;
                }

                trace.received_count += 1;

                if hop_count < trace.min_hop_count {
                    trace.min_hop_count = hop_count;
                }
            }

            GossipEvent::Forwarded { msg_id, to_peers } => {
                if let Some(trace) = self.traces.get_mut(&msg_id) {
                    trace.forward_count += to_peers.len() as u32;
                }
            }

            GossipEvent::Duplicate {
                msg_id,
                from_peer: _,
            } => {
                if let Some(trace) = self.traces.get_mut(&msg_id) {
                    trace.duplicate_count += 1;
                }
            }

            GossipEvent::Expired { msg_id: _ } => {
                // Intentionally a no-op; traces are kept for historical access.
            }
        }
    }

    /// Advance the logical tick by one.
    pub fn advance_tick(&mut self) {
        self.current_tick += 1;
    }

    /// Build an aggregated snapshot of current metrics.
    pub fn snapshot(&self) -> GossipMetricsSnapshot {
        let total_messages = self.traces.len() as u64;
        let total_duplicates: u64 = self.traces.values().map(|t| t.duplicate_count as u64).sum();
        let total_forwarded: u64 = self.traces.values().map(|t| t.forward_count as u64).sum();

        let avg_hop_count = if self.traces.is_empty() {
            0.0
        } else {
            let sum: u64 = self.traces.values().map(|t| t.min_hop_count as u64).sum();
            sum as f64 / total_messages as f64
        };

        let duplicate_rate = total_duplicates as f64 / total_messages.max(1) as f64;

        GossipMetricsSnapshot {
            total_messages,
            total_duplicates,
            total_forwarded,
            avg_hop_count,
            duplicate_rate,
        }
    }

    /// Return up to `n` message traces sorted descending by `redundancy_ratio`.
    pub fn top_messages_by_redundancy(&self, n: usize) -> Vec<&MessageTrace> {
        let mut traces: Vec<&MessageTrace> = self.traces.values().collect();
        traces.sort_by(|a, b| {
            b.redundancy_ratio()
                .partial_cmp(&a.redundancy_ratio())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        traces.truncate(n);
        traces
    }

    /// Look up the trace for a specific message, if one exists.
    pub fn message_trace(&self, msg_id: u64) -> Option<&MessageTrace> {
        self.traces.get(&msg_id)
    }
}

impl Default for PeerGossipMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn recv(msg_id: u64, from: &str, hop: u32) -> GossipEvent {
        GossipEvent::Received {
            msg_id,
            from_peer: from.to_string(),
            hop_count: hop,
        }
    }

    fn fwd(msg_id: u64, peers: &[&str]) -> GossipEvent {
        GossipEvent::Forwarded {
            msg_id,
            to_peers: peers.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn dup(msg_id: u64, from: &str) -> GossipEvent {
        GossipEvent::Duplicate {
            msg_id,
            from_peer: from.to_string(),
        }
    }

    fn exp(msg_id: u64) -> GossipEvent {
        GossipEvent::Expired { msg_id }
    }

    // ── test 1: Received creates a trace ─────────────────────────────────────

    #[test]
    fn received_creates_trace() {
        let mut m = PeerGossipMetrics::new();
        m.record_event(recv(1, "peer-A", 2));

        let trace = m.message_trace(1).expect("trace should exist");
        assert_eq!(trace.msg_id, 1);
        assert_eq!(trace.received_count, 1);
        assert_eq!(trace.duplicate_count, 0);
        assert_eq!(trace.min_hop_count, 2);
        assert_eq!(trace.first_seen_tick, 0);
    }

    // ── test 2: second Received increments duplicate_count ───────────────────

    #[test]
    fn second_received_increments_duplicate_count() {
        let mut m = PeerGossipMetrics::new();
        m.record_event(recv(1, "peer-A", 2));
        m.record_event(recv(1, "peer-B", 3));

        let trace = m.message_trace(1).expect("trace should exist");
        assert_eq!(trace.received_count, 2);
        assert_eq!(trace.duplicate_count, 1);
    }

    // ── test 3: third Received further increments duplicate_count ────────────

    #[test]
    fn third_received_increments_duplicate_count_again() {
        let mut m = PeerGossipMetrics::new();
        m.record_event(recv(1, "peer-A", 2));
        m.record_event(recv(1, "peer-B", 3));
        m.record_event(recv(1, "peer-C", 1));

        let trace = m.message_trace(1).expect("trace should exist");
        assert_eq!(trace.received_count, 3);
        assert_eq!(trace.duplicate_count, 2);
    }

    // ── test 4: Forwarded updates forward_count ───────────────────────────────

    #[test]
    fn forwarded_updates_forward_count() {
        let mut m = PeerGossipMetrics::new();
        m.record_event(recv(2, "peer-A", 1));
        m.record_event(fwd(2, &["peer-B", "peer-C"]));

        let trace = m.message_trace(2).expect("trace should exist");
        assert_eq!(trace.forward_count, 2);
    }

    // ── test 5: Forwarded to single peer ─────────────────────────────────────

    #[test]
    fn forwarded_single_peer() {
        let mut m = PeerGossipMetrics::new();
        m.record_event(recv(3, "peer-A", 0));
        m.record_event(fwd(3, &["peer-B"]));

        let trace = m.message_trace(3).expect("trace should exist");
        assert_eq!(trace.forward_count, 1);
    }

    // ── test 6: explicit Duplicate increments duplicate_count ─────────────────

    #[test]
    fn explicit_duplicate_increments_count() {
        let mut m = PeerGossipMetrics::new();
        m.record_event(recv(4, "peer-A", 1));
        m.record_event(dup(4, "peer-B"));

        let trace = m.message_trace(4).expect("trace should exist");
        assert_eq!(trace.duplicate_count, 1);
    }

    // ── test 7: multiple explicit Duplicates ──────────────────────────────────

    #[test]
    fn multiple_explicit_duplicates() {
        let mut m = PeerGossipMetrics::new();
        m.record_event(recv(5, "peer-A", 2));
        m.record_event(dup(5, "peer-B"));
        m.record_event(dup(5, "peer-C"));

        let trace = m.message_trace(5).expect("trace should exist");
        assert_eq!(trace.duplicate_count, 2);
    }

    // ── test 8: min_hop_count tracks smallest value ───────────────────────────

    #[test]
    fn min_hop_count_tracks_minimum() {
        let mut m = PeerGossipMetrics::new();
        m.record_event(recv(6, "peer-A", 5));
        m.record_event(recv(6, "peer-B", 3));
        m.record_event(recv(6, "peer-C", 7));

        let trace = m.message_trace(6).expect("trace should exist");
        assert_eq!(trace.min_hop_count, 3);
    }

    // ── test 9: min_hop_count set on first reception ──────────────────────────

    #[test]
    fn min_hop_count_initial_value() {
        let mut m = PeerGossipMetrics::new();
        m.record_event(recv(7, "peer-A", 4));

        let trace = m.message_trace(7).expect("trace should exist");
        assert_eq!(trace.min_hop_count, 4);
    }

    // ── test 10: redundancy_ratio ─────────────────────────────────────────────

    #[test]
    fn redundancy_ratio_correct() {
        let mut m = PeerGossipMetrics::new();
        m.record_event(recv(8, "peer-A", 1));
        m.record_event(recv(8, "peer-B", 1));
        m.record_event(recv(8, "peer-C", 1));
        // received_count = 3, duplicate_count = 2
        let trace = m.message_trace(8).expect("trace should exist");
        let expected = 2.0 / 3.0;
        assert!((trace.redundancy_ratio() - expected).abs() < 1e-10);
    }

    // ── test 11: redundancy_ratio zero when no duplicates ────────────────────

    #[test]
    fn redundancy_ratio_zero_no_duplicates() {
        let mut m = PeerGossipMetrics::new();
        m.record_event(recv(9, "peer-A", 1));

        let trace = m.message_trace(9).expect("trace should exist");
        assert_eq!(trace.redundancy_ratio(), 0.0);
    }

    // ── test 12: snapshot totals ──────────────────────────────────────────────

    #[test]
    fn snapshot_totals() {
        let mut m = PeerGossipMetrics::new();
        // msg 10: received twice (1 dup), forwarded to 3
        m.record_event(recv(10, "peer-A", 1));
        m.record_event(recv(10, "peer-B", 2));
        m.record_event(fwd(10, &["peer-C", "peer-D", "peer-E"]));
        // msg 11: received once, forwarded to 1
        m.record_event(recv(11, "peer-A", 2));
        m.record_event(fwd(11, &["peer-F"]));

        let snap = m.snapshot();
        assert_eq!(snap.total_messages, 2);
        assert_eq!(snap.total_duplicates, 1);
        assert_eq!(snap.total_forwarded, 4);
    }

    // ── test 13: snapshot duplicate_rate ─────────────────────────────────────

    #[test]
    fn snapshot_duplicate_rate() {
        let mut m = PeerGossipMetrics::new();
        m.record_event(recv(12, "peer-A", 1));
        m.record_event(recv(12, "peer-B", 1));
        m.record_event(recv(13, "peer-A", 1));

        let snap = m.snapshot();
        // total_messages = 2, total_duplicates = 1
        let expected = 1.0 / 2.0;
        assert!((snap.duplicate_rate - expected).abs() < 1e-10);
    }

    // ── test 14: estimated_coverage ──────────────────────────────────────────

    #[test]
    fn estimated_coverage() {
        let mut m = PeerGossipMetrics::new();
        m.record_event(recv(14, "peer-A", 1));
        m.record_event(recv(14, "peer-B", 1)); // 1 dup in 1 msg → dup_rate = 1.0

        let snap = m.snapshot();
        // coverage = 1.0 - 1.0 = 0.0
        assert!((snap.estimated_coverage() - 0.0).abs() < 1e-10);
    }

    // ── test 15: estimated_coverage no duplicates ─────────────────────────────

    #[test]
    fn estimated_coverage_no_duplicates() {
        let mut m = PeerGossipMetrics::new();
        m.record_event(recv(15, "peer-A", 1));
        m.record_event(recv(16, "peer-B", 1));

        let snap = m.snapshot();
        assert!((snap.estimated_coverage() - 1.0).abs() < 1e-10);
    }

    // ── test 16: top_messages_by_redundancy ordering ──────────────────────────

    #[test]
    fn top_messages_by_redundancy_ordered() {
        let mut m = PeerGossipMetrics::new();
        // msg 20: received once  → redundancy 0/1 = 0.0
        m.record_event(recv(20, "peer-A", 1));
        // msg 21: received twice → redundancy 1/2 = 0.5
        m.record_event(recv(21, "peer-A", 1));
        m.record_event(recv(21, "peer-B", 1));
        // msg 22: received three times → redundancy 2/3 ≈ 0.667
        m.record_event(recv(22, "peer-A", 1));
        m.record_event(recv(22, "peer-B", 1));
        m.record_event(recv(22, "peer-C", 1));

        let top = m.top_messages_by_redundancy(3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].msg_id, 22);
        assert_eq!(top[1].msg_id, 21);
        assert_eq!(top[2].msg_id, 20);
    }

    // ── test 17: top_messages_by_redundancy respects n ────────────────────────

    #[test]
    fn top_messages_by_redundancy_truncates() {
        let mut m = PeerGossipMetrics::new();
        for id in 30..36_u64 {
            m.record_event(recv(id, "peer-A", 1));
        }

        let top = m.top_messages_by_redundancy(3);
        assert_eq!(top.len(), 3);
    }

    // ── test 18: message_trace Some ───────────────────────────────────────────

    #[test]
    fn message_trace_some() {
        let mut m = PeerGossipMetrics::new();
        m.record_event(recv(40, "peer-A", 1));
        assert!(m.message_trace(40).is_some());
    }

    // ── test 19: message_trace None ───────────────────────────────────────────

    #[test]
    fn message_trace_none() {
        let m = PeerGossipMetrics::new();
        assert!(m.message_trace(999).is_none());
    }

    // ── test 20: Expired is a no-op but trace survives ───────────────────────

    #[test]
    fn expired_is_noop_trace_survives() {
        let mut m = PeerGossipMetrics::new();
        m.record_event(recv(50, "peer-A", 1));
        m.record_event(exp(50));

        let trace = m.message_trace(50).expect("trace should survive expiry");
        assert_eq!(trace.received_count, 1);
        assert_eq!(trace.duplicate_count, 0);
    }

    // ── test 21: advance_tick increments current_tick ────────────────────────

    #[test]
    fn advance_tick_increments() {
        let mut m = PeerGossipMetrics::new();
        assert_eq!(m.current_tick, 0);
        m.advance_tick();
        assert_eq!(m.current_tick, 1);
        m.advance_tick();
        assert_eq!(m.current_tick, 2);
    }

    // ── test 22: first_seen_tick reflects tick at insertion ──────────────────

    #[test]
    fn first_seen_tick_correct() {
        let mut m = PeerGossipMetrics::new();
        m.advance_tick(); // tick = 1
        m.advance_tick(); // tick = 2
        m.record_event(recv(60, "peer-A", 1));

        let trace = m.message_trace(60).expect("trace should exist");
        assert_eq!(trace.first_seen_tick, 2);
    }

    // ── test 23: snapshot avg_hop_count ──────────────────────────────────────

    #[test]
    fn snapshot_avg_hop_count() {
        let mut m = PeerGossipMetrics::new();
        m.record_event(recv(70, "peer-A", 2));
        m.record_event(recv(71, "peer-A", 4));

        let snap = m.snapshot();
        // avg of min_hop_counts 2 and 4 = 3.0
        assert!((snap.avg_hop_count - 3.0).abs() < 1e-10);
    }

    // ── test 24: empty snapshot ───────────────────────────────────────────────

    #[test]
    fn empty_snapshot() {
        let m = PeerGossipMetrics::new();
        let snap = m.snapshot();
        assert_eq!(snap.total_messages, 0);
        assert_eq!(snap.total_duplicates, 0);
        assert_eq!(snap.total_forwarded, 0);
        assert_eq!(snap.avg_hop_count, 0.0);
        assert_eq!(snap.duplicate_rate, 0.0);
        assert_eq!(snap.estimated_coverage(), 1.0);
    }
}
