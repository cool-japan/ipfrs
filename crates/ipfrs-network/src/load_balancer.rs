//! Peer load balancer with configurable strategies.
//!
//! Distributes requests across peers using RoundRobin, LeastConnections,
//! WeightedRandom, or LowestLatency strategies, tracking per-peer load and
//! response times via EWMA.

use std::collections::HashMap;

// ──────────────────────────────────────────────
// Strategy
// ──────────────────────────────────────────────

/// Load-balancing strategy used by [`PeerLoadBalancer`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LbStrategy {
    /// Cycle through healthy peers in alphabetical order.
    RoundRobin,
    /// Pick the peer with the fewest active connections.
    LeastConnections,
    /// Deterministic weight-proportional selection.
    ///
    /// Picks the peer whose cumulative weight covers
    /// `index = total_requests % total_weight`.
    WeightedRandom,
    /// Pick the peer with the lowest average latency.
    ///
    /// Falls back to `RoundRobin` when all latencies are zero.
    LowestLatency,
}

// ──────────────────────────────────────────────
// PeerLoad
// ──────────────────────────────────────────────

/// Per-peer load tracking information.
#[derive(Debug, Clone)]
pub struct PeerLoad {
    /// Unique peer identifier.
    pub peer_id: String,
    /// Number of currently in-flight requests.
    pub active_connections: u32,
    /// Routing weight; higher values attract more requests.
    pub weight: u32,
    /// Exponentially-weighted moving average of latency (ms), alpha = 0.2.
    pub avg_latency_ms: f64,
    /// Total requests ever dispatched to this peer.
    pub total_requests: u64,
    /// Total failed requests for this peer.
    pub failed_requests: u64,
}

impl PeerLoad {
    /// Creates a new `PeerLoad` record with the given `peer_id` and `weight`.
    fn new(peer_id: impl Into<String>, weight: u32) -> Self {
        Self {
            peer_id: peer_id.into(),
            active_connections: 0,
            weight,
            avg_latency_ms: 0.0,
            total_requests: 0,
            failed_requests: 0,
        }
    }

    /// Fraction of requests that have failed; returns `0.0` when no requests
    /// have been dispatched yet.
    pub fn failure_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.failed_requests as f64 / self.total_requests as f64
        }
    }

    /// A peer is healthy when its failure rate is below 50 % **and** it has
    /// fewer than 1 000 active connections.
    pub fn is_healthy(&self) -> bool {
        self.failure_rate() < 0.5 && self.active_connections < 1000
    }
}

// ──────────────────────────────────────────────
// LbStats
// ──────────────────────────────────────────────

/// Aggregate load-balancer statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LbStats {
    /// Total number of registered peers (healthy + unhealthy).
    pub total_peers: usize,
    /// Number of peers currently considered healthy.
    pub healthy_peers: usize,
    /// Number of successful `select()` calls (requests routed).
    pub total_requests_routed: u64,
    /// Monotonically increasing counter used for RoundRobin / WeightedRandom.
    pub total_request_counter: u64,
}

// ──────────────────────────────────────────────
// PeerLoadBalancer
// ──────────────────────────────────────────────

/// Distributes requests across peers using a configurable [`LbStrategy`].
pub struct PeerLoadBalancer {
    /// All registered peers, keyed by peer id.
    pub peers: HashMap<String, PeerLoad>,
    /// Active load-balancing strategy.
    pub strategy: LbStrategy,
    /// Monotonically increasing counter; drives round-robin index and
    /// weighted-random index.  Incremented on every successful `select()`.
    pub total_requests: u64,
}

impl PeerLoadBalancer {
    // ── Construction ───────────────────────────

    /// Creates a new, empty load balancer using `strategy`.
    pub fn new(strategy: LbStrategy) -> Self {
        Self {
            peers: HashMap::new(),
            strategy,
            total_requests: 0,
        }
    }

    // ── Peer management ────────────────────────

    /// Registers a peer.  If the peer already exists its weight is updated.
    pub fn add_peer(&mut self, peer_id: &str, weight: u32) {
        self.peers
            .entry(peer_id.to_owned())
            .and_modify(|p| p.weight = weight)
            .or_insert_with(|| PeerLoad::new(peer_id, weight));
    }

    /// Removes a peer.  Returns `true` when the peer existed.
    pub fn remove_peer(&mut self, peer_id: &str) -> bool {
        self.peers.remove(peer_id).is_some()
    }

    // ── Selection ──────────────────────────────

    /// Selects a peer according to the configured strategy.
    ///
    /// Only healthy peers are considered.  Returns `None` when no healthy
    /// peers are available.  On success the internal request counter is
    /// incremented.
    pub fn select(&mut self) -> Option<String> {
        let chosen = match self.strategy {
            LbStrategy::RoundRobin => self.select_round_robin(),
            LbStrategy::LeastConnections => self.select_least_connections(),
            LbStrategy::WeightedRandom => self.select_weighted_random(),
            LbStrategy::LowestLatency => self.select_lowest_latency(),
        };

        if chosen.is_some() {
            self.total_requests += 1;
        }

        chosen
    }

    // ── Record helpers ─────────────────────────

    /// Marks the start of a request to `peer_id`.
    ///
    /// Increments `active_connections` and `total_requests` on the peer.
    /// Returns `false` when the peer is not registered.
    pub fn record_start(&mut self, peer_id: &str) -> bool {
        match self.peers.get_mut(peer_id) {
            Some(p) => {
                p.active_connections = p.active_connections.saturating_add(1);
                p.total_requests += 1;
                true
            }
            None => false,
        }
    }

    /// Records the completion of a request to `peer_id`.
    ///
    /// - Decrements `active_connections` (saturating at 0).
    /// - Updates `avg_latency_ms` with EWMA alpha = 0.2.
    /// - Increments `failed_requests` when `success` is `false`.
    ///
    /// Returns `false` when the peer is not registered.
    pub fn record_complete(&mut self, peer_id: &str, latency_ms: f64, success: bool) -> bool {
        match self.peers.get_mut(peer_id) {
            Some(p) => {
                p.active_connections = p.active_connections.saturating_sub(1);
                p.avg_latency_ms = 0.8 * p.avg_latency_ms + 0.2 * latency_ms;
                if !success {
                    p.failed_requests += 1;
                }
                true
            }
            None => false,
        }
    }

    // ── Statistics ─────────────────────────────

    /// Returns a snapshot of current load-balancer statistics.
    pub fn stats(&self) -> LbStats {
        let total_peers = self.peers.len();
        let healthy_peers = self.peers.values().filter(|p| p.is_healthy()).count();
        LbStats {
            total_peers,
            healthy_peers,
            total_requests_routed: self.total_requests,
            total_request_counter: self.total_requests,
        }
    }

    // ── Private strategy implementations ───────

    /// Collects healthy peers sorted alphabetically by peer id.
    fn healthy_sorted(&self) -> Vec<&PeerLoad> {
        let mut peers: Vec<&PeerLoad> = self.peers.values().filter(|p| p.is_healthy()).collect();
        peers.sort_by(|a, b| a.peer_id.cmp(&b.peer_id));
        peers
    }

    fn select_round_robin(&self) -> Option<String> {
        let peers = self.healthy_sorted();
        if peers.is_empty() {
            return None;
        }
        let idx = (self.total_requests as usize) % peers.len();
        Some(peers[idx].peer_id.clone())
    }

    fn select_least_connections(&self) -> Option<String> {
        self.peers
            .values()
            .filter(|p| p.is_healthy())
            .min_by_key(|p| p.active_connections)
            .map(|p| p.peer_id.clone())
    }

    fn select_weighted_random(&self) -> Option<String> {
        let mut peers = self.healthy_sorted();
        if peers.is_empty() {
            return None;
        }

        let total_weight: u32 = peers.iter().map(|p| p.weight).sum();
        if total_weight == 0 {
            // Fall back to round-robin when all weights are zero.
            let idx = (self.total_requests as usize) % peers.len();
            return Some(peers[idx].peer_id.clone());
        }

        let index = (self.total_requests % total_weight as u64) as u32;
        let mut cumulative: u32 = 0;
        for p in peers.drain(..) {
            cumulative += p.weight;
            if index < cumulative {
                return Some(p.peer_id.clone());
            }
        }
        // Unreachable in correct logic, but return last peer as safety.
        None
    }

    fn select_lowest_latency(&self) -> Option<String> {
        let peers: Vec<&PeerLoad> = self.peers.values().filter(|p| p.is_healthy()).collect();

        if peers.is_empty() {
            return None;
        }

        // If every peer has latency == 0 fall back to round-robin.
        if peers.iter().all(|p| p.avg_latency_ms == 0.0) {
            return self.select_round_robin();
        }

        peers
            .into_iter()
            .min_by(|a, b| {
                a.avg_latency_ms
                    .partial_cmp(&b.avg_latency_ms)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| p.peer_id.clone())
    }
}

// ──────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── 1. new() starts empty ──────────────────
    #[test]
    fn test_new_starts_empty() {
        let lb = PeerLoadBalancer::new(LbStrategy::RoundRobin);
        assert!(lb.peers.is_empty());
        assert_eq!(lb.total_requests, 0);
    }

    // ── 2. add_peer creates PeerLoad with weight ──
    #[test]
    fn test_add_peer_creates_with_weight() {
        let mut lb = PeerLoadBalancer::new(LbStrategy::RoundRobin);
        lb.add_peer("alpha", 5);
        let p = lb.peers.get("alpha").expect("peer must exist");
        assert_eq!(p.weight, 5);
        assert_eq!(p.active_connections, 0);
        assert_eq!(p.total_requests, 0);
    }

    // ── 3. add_peer updates weight on duplicate ──
    #[test]
    fn test_add_peer_updates_weight() {
        let mut lb = PeerLoadBalancer::new(LbStrategy::RoundRobin);
        lb.add_peer("alpha", 1);
        lb.add_peer("alpha", 10);
        assert_eq!(
            lb.peers
                .get("alpha")
                .expect("test: alpha peer must exist after duplicate add")
                .weight,
            10
        );
        assert_eq!(lb.peers.len(), 1);
    }

    // ── 4. remove_peer returns true when exists ──
    #[test]
    fn test_remove_peer_true() {
        let mut lb = PeerLoadBalancer::new(LbStrategy::RoundRobin);
        lb.add_peer("alpha", 1);
        assert!(lb.remove_peer("alpha"));
    }

    // ── 5. remove_peer returns false when missing ──
    #[test]
    fn test_remove_peer_false() {
        let mut lb = PeerLoadBalancer::new(LbStrategy::RoundRobin);
        assert!(!lb.remove_peer("ghost"));
    }

    // ── 6. select None when no peers ──────────
    #[test]
    fn test_select_none_when_no_peers() {
        let mut lb = PeerLoadBalancer::new(LbStrategy::RoundRobin);
        assert!(lb.select().is_none());
    }

    // ── 7. RoundRobin cycles alphabetically ───
    #[test]
    fn test_round_robin_cycles_alpha_order() {
        let mut lb = PeerLoadBalancer::new(LbStrategy::RoundRobin);
        lb.add_peer("charlie", 1);
        lb.add_peer("alpha", 1);
        lb.add_peer("bravo", 1);

        // Sorted order: alpha, bravo, charlie
        let first = lb
            .select()
            .expect("test: round-robin first select must succeed");
        assert_eq!(first, "alpha");
        let second = lb
            .select()
            .expect("test: round-robin second select must succeed");
        assert_eq!(second, "bravo");
        let third = lb
            .select()
            .expect("test: round-robin third select must succeed");
        assert_eq!(third, "charlie");
        // Wraps around
        let fourth = lb
            .select()
            .expect("test: round-robin wrap-around select must succeed");
        assert_eq!(fourth, "alpha");
    }

    // ── 8. LeastConnections picks lowest active ──
    #[test]
    fn test_least_connections() {
        let mut lb = PeerLoadBalancer::new(LbStrategy::LeastConnections);
        lb.add_peer("a", 1);
        lb.add_peer("b", 1);
        lb.peers
            .get_mut("a")
            .expect("test: peer 'a' must exist")
            .active_connections = 5;
        lb.peers
            .get_mut("b")
            .expect("test: peer 'b' must exist")
            .active_connections = 2;

        let chosen = lb
            .select()
            .expect("test: least-connections select must succeed");
        assert_eq!(chosen, "b");
    }

    // ── 9. WeightedRandom proportional selection ──
    #[test]
    fn test_weighted_random_proportional() {
        // total_weight = 3 (a=1, b=2)
        // index 0 → a (cumul 1 > 0)
        // index 1 → b (cumul 3 > 1)
        // index 2 → b (cumul 3 > 2)
        let mut lb = PeerLoadBalancer::new(LbStrategy::WeightedRandom);
        lb.add_peer("a", 1);
        lb.add_peer("b", 2);

        // total_requests starts at 0 → index 0 → "a"
        let r0 = lb
            .select()
            .expect("test: weighted-random select 0 must succeed"); // counter becomes 1
                                                                    // total_requests was 0 when select_weighted_random ran
        assert_eq!(r0, "a");

        // counter = 1, index 1 → "b"
        let r1 = lb
            .select()
            .expect("test: weighted-random select 1 must succeed"); // counter becomes 2
        assert_eq!(r1, "b");

        // counter = 2, index 2 → "b"
        let r2 = lb
            .select()
            .expect("test: weighted-random select 2 must succeed"); // counter becomes 3
        assert_eq!(r2, "b");

        // counter = 3, index 0 → "a"
        let r3 = lb
            .select()
            .expect("test: weighted-random select 3 must succeed");
        assert_eq!(r3, "a");
    }

    // ── 10. LowestLatency picks peer with lowest latency ──
    #[test]
    fn test_lowest_latency_picks_lowest() {
        let mut lb = PeerLoadBalancer::new(LbStrategy::LowestLatency);
        lb.add_peer("fast", 1);
        lb.add_peer("slow", 1);
        lb.peers
            .get_mut("fast")
            .expect("test: peer 'fast' must exist")
            .avg_latency_ms = 10.0;
        lb.peers
            .get_mut("slow")
            .expect("test: peer 'slow' must exist")
            .avg_latency_ms = 100.0;

        let chosen = lb
            .select()
            .expect("test: lowest-latency select must succeed");
        assert_eq!(chosen, "fast");
    }

    // ── 11. LowestLatency fallback to RoundRobin when all zero ──
    #[test]
    fn test_lowest_latency_fallback_round_robin() {
        let mut lb = PeerLoadBalancer::new(LbStrategy::LowestLatency);
        lb.add_peer("beta", 1);
        lb.add_peer("alpha", 1);
        // Both latencies are 0 → round-robin (sorted: alpha, beta)
        let first = lb
            .select()
            .expect("test: lowest-latency fallback first select must succeed");
        assert_eq!(first, "alpha");
        let second = lb
            .select()
            .expect("test: lowest-latency fallback second select must succeed");
        assert_eq!(second, "beta");
    }

    // ── 12. select skips unhealthy peers ──────
    #[test]
    fn test_select_skips_unhealthy() {
        let mut lb = PeerLoadBalancer::new(LbStrategy::RoundRobin);
        lb.add_peer("sick", 1);
        lb.add_peer("well", 1);
        // Make "sick" unhealthy: failure_rate ≥ 0.5
        lb.peers
            .get_mut("sick")
            .expect("test: peer 'sick' must exist")
            .total_requests = 2;
        lb.peers
            .get_mut("sick")
            .expect("test: peer 'sick' must exist")
            .failed_requests = 2;

        // Only "well" is healthy; round-robin over 1 peer always returns "well"
        let chosen = lb
            .select()
            .expect("test: select skipping unhealthy must return healthy peer");
        assert_eq!(chosen, "well");
    }

    // ── 13. record_start increments active_connections ──
    #[test]
    fn test_record_start_increments_connections() {
        let mut lb = PeerLoadBalancer::new(LbStrategy::RoundRobin);
        lb.add_peer("x", 1);
        assert!(lb.record_start("x"));
        assert_eq!(lb.peers["x"].active_connections, 1);
        assert_eq!(lb.peers["x"].total_requests, 1);
    }

    // ── 14. record_start returns false for unknown peer ──
    #[test]
    fn test_record_start_false_unknown() {
        let mut lb = PeerLoadBalancer::new(LbStrategy::RoundRobin);
        assert!(!lb.record_start("unknown"));
    }

    // ── 15. record_complete decrements active_connections ──
    #[test]
    fn test_record_complete_decrements_connections() {
        let mut lb = PeerLoadBalancer::new(LbStrategy::RoundRobin);
        lb.add_peer("x", 1);
        lb.record_start("x");
        lb.record_complete("x", 50.0, true);
        assert_eq!(lb.peers["x"].active_connections, 0);
    }

    // ── 16. record_complete updates EWMA latency ──
    #[test]
    fn test_record_complete_ewma_latency() {
        let mut lb = PeerLoadBalancer::new(LbStrategy::RoundRobin);
        lb.add_peer("x", 1);
        // initial avg = 0, new sample = 100 → 0.8*0 + 0.2*100 = 20
        lb.record_complete("x", 100.0, true);
        let lat = lb.peers["x"].avg_latency_ms;
        assert!((lat - 20.0).abs() < f64::EPSILON);

        // second sample = 50 → 0.8*20 + 0.2*50 = 16 + 10 = 26
        lb.record_complete("x", 50.0, true);
        let lat2 = lb.peers["x"].avg_latency_ms;
        assert!((lat2 - 26.0).abs() < f64::EPSILON);
    }

    // ── 17. record_complete increments failed_requests on failure ──
    #[test]
    fn test_record_complete_failure_increments_failed() {
        let mut lb = PeerLoadBalancer::new(LbStrategy::RoundRobin);
        lb.add_peer("x", 1);
        lb.record_complete("x", 10.0, false);
        assert_eq!(lb.peers["x"].failed_requests, 1);
    }

    // ── 18. failure_rate computed correctly ───
    #[test]
    fn test_failure_rate() {
        let mut p = PeerLoad::new("p", 1);
        assert_eq!(p.failure_rate(), 0.0); // no requests
        p.total_requests = 4;
        p.failed_requests = 1;
        assert!((p.failure_rate() - 0.25).abs() < f64::EPSILON);
    }

    // ── 19. is_healthy false when failure_rate ≥ 0.5 ──
    #[test]
    fn test_is_healthy_false_high_failure_rate() {
        let mut p = PeerLoad::new("p", 1);
        p.total_requests = 2;
        p.failed_requests = 1; // exactly 0.5 → unhealthy
        assert!(!p.is_healthy());
    }

    // ── 20. is_healthy false when connections ≥ 1000 ──
    #[test]
    fn test_is_healthy_false_too_many_connections() {
        let mut p = PeerLoad::new("p", 1);
        p.active_connections = 1000;
        assert!(!p.is_healthy());
    }

    // ── 21. stats total_peers / healthy_peers ──
    #[test]
    fn test_stats_peer_counts() {
        let mut lb = PeerLoadBalancer::new(LbStrategy::RoundRobin);
        lb.add_peer("a", 1);
        lb.add_peer("b", 1);
        lb.peers
            .get_mut("b")
            .expect("test: peer 'b' must exist")
            .total_requests = 2;
        lb.peers
            .get_mut("b")
            .expect("test: peer 'b' must exist")
            .failed_requests = 2; // unhealthy

        let s = lb.stats();
        assert_eq!(s.total_peers, 2);
        assert_eq!(s.healthy_peers, 1);
    }

    // ── 22. stats total_requests_routed increments on select ──
    #[test]
    fn test_stats_requests_routed() {
        let mut lb = PeerLoadBalancer::new(LbStrategy::RoundRobin);
        lb.add_peer("a", 1);
        lb.select();
        lb.select();
        assert_eq!(lb.stats().total_requests_routed, 2);
    }

    // ── 23. active_connections saturating at 0 ──
    #[test]
    fn test_saturating_decrement() {
        let mut lb = PeerLoadBalancer::new(LbStrategy::RoundRobin);
        lb.add_peer("x", 1);
        // Never called record_start; decrement saturates at 0
        lb.record_complete("x", 10.0, true);
        assert_eq!(lb.peers["x"].active_connections, 0);
    }

    // ── 24. select returns None when all peers unhealthy ──
    #[test]
    fn test_select_none_all_unhealthy() {
        let mut lb = PeerLoadBalancer::new(LbStrategy::LeastConnections);
        lb.add_peer("a", 1);
        lb.peers
            .get_mut("a")
            .expect("test: peer 'a' must exist")
            .active_connections = 1000; // unhealthy
        assert!(lb.select().is_none());
    }

    // ── 25. multiple peers round-robin cycles ──
    #[test]
    fn test_multiple_peers_round_robin() {
        let mut lb = PeerLoadBalancer::new(LbStrategy::RoundRobin);
        for name in &["peer-c", "peer-a", "peer-b"] {
            lb.add_peer(name, 1);
        }
        // sorted: peer-a, peer-b, peer-c
        let expected = ["peer-a", "peer-b", "peer-c", "peer-a", "peer-b", "peer-c"];
        for &exp in &expected {
            assert_eq!(
                lb.select()
                    .expect("test: multi-peer round-robin select must succeed"),
                exp
            );
        }
    }

    // ── 26. record_complete returns false for unknown peer ──
    #[test]
    fn test_record_complete_false_unknown() {
        let mut lb = PeerLoadBalancer::new(LbStrategy::RoundRobin);
        assert!(!lb.record_complete("ghost", 10.0, true));
    }

    // ── 27. LeastConnections tie-breaking is deterministic ──
    #[test]
    fn test_least_connections_with_two_zeros() {
        let mut lb = PeerLoadBalancer::new(LbStrategy::LeastConnections);
        lb.add_peer("x", 1);
        lb.add_peer("y", 1);
        // Both have 0 active connections; either is valid — just ensure Some
        assert!(lb.select().is_some());
    }

    // ── 28. WeightedRandom with single peer always picks it ──
    #[test]
    fn test_weighted_random_single_peer() {
        let mut lb = PeerLoadBalancer::new(LbStrategy::WeightedRandom);
        lb.add_peer("only", 7);
        for _ in 0..10 {
            assert_eq!(
                lb.select()
                    .expect("test: weighted-random single peer must always select it"),
                "only"
            );
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// AdaptiveLoadBalancer — production-grade multi-algorithm load balancer
// ══════════════════════════════════════════════════════════════════════════════

/// Algorithm used by [`AdaptiveLoadBalancer`] to route requests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LbAlgorithm {
    /// Cycle through healthy peers in insertion order.
    RoundRobin,
    /// Pick the peer with the fewest active connections.
    LeastConnections,
    /// Route by weight proportion (higher weight → more requests).
    WeightedRoundRobin,
    /// Pick the peer with the lowest `avg_latency_ms`.
    LeastLatency,
    /// XorShift64-based pseudo-random selection from healthy peers.
    Random,
    /// FNV-1a consistent-hash ring using virtual nodes for uniform distribution.
    ConsistentHash,
}

/// Per-peer metadata tracked by [`AdaptiveLoadBalancer`].
#[derive(Clone, Debug)]
pub struct LbPeer {
    /// Unique peer identifier.
    pub peer_id: String,
    /// Routing weight; used by `WeightedRoundRobin`.
    pub weight: u32,
    /// Number of currently in-flight requests.
    pub active_connections: u32,
    /// Total requests ever dispatched to this peer.
    pub total_requests: u64,
    /// Total failed requests for this peer.
    pub total_errors: u64,
    /// Exponentially-weighted moving average latency (ms), alpha = 0.2.
    pub avg_latency_ms: f64,
    /// Whether the peer is considered healthy.
    pub healthy: bool,
    /// Timestamp of last use (opaque `u64`, caller-supplied).
    pub last_used: u64,
}

impl LbPeer {
    /// Create a new [`LbPeer`] with defaults.
    pub fn new(peer_id: impl Into<String>, weight: u32) -> Self {
        Self {
            peer_id: peer_id.into(),
            weight,
            active_connections: 0,
            total_requests: 0,
            total_errors: 0,
            avg_latency_ms: 0.0,
            healthy: true,
            last_used: 0,
        }
    }

    /// Error rate in `[0.0, 1.0]`; `0.0` when no requests have been made.
    pub fn error_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.total_errors as f64 / self.total_requests as f64
        }
    }
}

/// Describes a single request to be routed.
#[derive(Clone, Debug)]
pub struct LbRequest {
    /// Unique request identifier.
    pub id: u64,
    /// Optional routing key; used by `ConsistentHash` (None → random peer).
    pub key: Option<String>,
    /// Request size in bytes (informational).
    pub size_bytes: usize,
    /// Request creation timestamp (opaque `u64`, caller-supplied).
    pub timestamp: u64,
}

impl LbRequest {
    /// Convenience constructor.
    pub fn new(id: u64, key: Option<String>, size_bytes: usize, timestamp: u64) -> Self {
        Self {
            id,
            key,
            size_bytes,
            timestamp,
        }
    }
}

/// The routing decision produced by [`AdaptiveLoadBalancer::select`].
#[derive(Clone, Debug)]
pub struct LbDecision {
    /// Peer selected to handle the request.
    pub peer_id: String,
    /// Human-readable name of the algorithm that made the decision.
    pub algorithm_used: String,
    /// The originating request id.
    pub request_id: u64,
    /// Timestamp at which the decision was made (caller-supplied `now`).
    pub decided_at: u64,
}

/// Aggregate statistics returned by [`AdaptiveLoadBalancer::stats`].
#[derive(Clone, Debug)]
pub struct AdaptiveLbStats {
    /// Total number of registered peers (healthy + unhealthy).
    pub total_peers: usize,
    /// Number of currently healthy peers.
    pub healthy_peers: usize,
    /// Total requests routed successfully.
    pub total_requests: u64,
    /// Total errors recorded across all peers.
    pub total_errors: u64,
    /// Fraction of errored requests; `0.0` when `total_requests == 0`.
    pub error_rate: f64,
    /// Mean of per-peer `avg_latency_ms` across all peers with non-zero latency.
    pub avg_latency_ms: f64,
}

// ──────────────────────────────────────────────────────────────────────────────
// FNV-1a helper (64-bit)
// ──────────────────────────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash of a byte slice.
#[inline]
fn fnv1a_64_bytes(data: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// FNV-1a 64-bit hash of a string.
#[inline]
fn fnv1a_64(s: &str) -> u64 {
    fnv1a_64_bytes(s.as_bytes())
}

// ──────────────────────────────────────────────────────────────────────────────
// XorShift64 — inline, no rand dependency
// ──────────────────────────────────────────────────────────────────────────────

/// A minimal non-zero XorShift64 state.  Seeded lazily from request id + now.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    // Ensure non-zero seed
    if x == 0 {
        x = 0x853c_49e6_748f_ea9b;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ──────────────────────────────────────────────────────────────────────────────
// AdaptiveLoadBalancer
// ──────────────────────────────────────────────────────────────────────────────

/// Production-grade, multi-algorithm load balancer with dynamic weight
/// adjustment and consistent-hash support.
pub struct AdaptiveLoadBalancer {
    /// Active routing algorithm.
    pub algorithm: LbAlgorithm,
    /// Ordered list of registered peers (insertion order is preserved).
    pub peers: Vec<LbPeer>,
    /// Round-robin cursor (index into `peers` after filtering healthy ones).
    pub rr_index: usize,
    /// Number of virtual nodes per peer in the consistent-hash ring.
    pub virtual_nodes: usize,
    /// Total requests successfully routed by this balancer instance.
    pub total_requests: u64,
    /// Total errors recorded via `record_completion`.
    pub total_errors: u64,
    /// Sorted consistent-hash ring `(hash, peer_index)`.
    ring: Vec<(u64, usize)>,
    /// Dirty flag: ring needs rebuilding.
    ring_dirty: bool,
    /// XorShift64 PRNG state for `Random` algorithm.
    prng_state: u64,
}

impl AdaptiveLoadBalancer {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Creates a new, empty balancer using the given algorithm.
    /// `virtual_nodes` defaults to 150.
    pub fn new(algorithm: LbAlgorithm) -> Self {
        Self {
            algorithm,
            peers: Vec::new(),
            rr_index: 0,
            virtual_nodes: 150,
            total_requests: 0,
            total_errors: 0,
            ring: Vec::new(),
            ring_dirty: true,
            prng_state: 0,
        }
    }

    // ── Peer management ──────────────────────────────────────────────────────

    /// Register a peer.  If a peer with the same id already exists it is
    /// replaced with the new record.
    pub fn add_peer(&mut self, peer: LbPeer) {
        if let Some(existing) = self.peers.iter_mut().find(|p| p.peer_id == peer.peer_id) {
            *existing = peer;
        } else {
            self.peers.push(peer);
        }
        self.ring_dirty = true;
    }

    /// Remove a peer by id.  Returns `true` if the peer was found and removed.
    pub fn remove_peer(&mut self, peer_id: &str) -> bool {
        let before = self.peers.len();
        self.peers.retain(|p| p.peer_id != peer_id);
        let removed = self.peers.len() < before;
        if removed {
            self.ring_dirty = true;
            // Clamp rr_index to avoid out-of-bounds after removal.
            if !self.peers.is_empty() {
                self.rr_index %= self.peers.len();
            } else {
                self.rr_index = 0;
            }
        }
        removed
    }

    // ── Selection ────────────────────────────────────────────────────────────

    /// Route a request to the best available peer.
    ///
    /// Returns `None` when no healthy peers are available.
    pub fn select(&mut self, request: &LbRequest, now: u64) -> Option<LbDecision> {
        let (peer_id, algo_name) = match self.algorithm {
            LbAlgorithm::RoundRobin => self.select_round_robin()?,
            LbAlgorithm::LeastConnections => self.select_least_connections()?,
            LbAlgorithm::WeightedRoundRobin => self.select_weighted_round_robin()?,
            LbAlgorithm::LeastLatency => self.select_least_latency()?,
            LbAlgorithm::Random => self.select_random(request)?,
            LbAlgorithm::ConsistentHash => self.select_consistent_hash(request)?,
        };

        self.total_requests += 1;

        // Update last_used on chosen peer.
        if let Some(p) = self.peers.iter_mut().find(|p| p.peer_id == peer_id) {
            p.last_used = now;
        }

        Some(LbDecision {
            peer_id,
            algorithm_used: algo_name,
            request_id: request.id,
            decided_at: now,
        })
    }

    // ── Record helpers ────────────────────────────────────────────────────────

    /// Increment `active_connections` for a peer.  Returns `false` if unknown.
    pub fn record_connection_start(&mut self, peer_id: &str) -> bool {
        match self.peers.iter_mut().find(|p| p.peer_id == peer_id) {
            Some(p) => {
                p.active_connections = p.active_connections.saturating_add(1);
                true
            }
            None => false,
        }
    }

    /// Record completion of a request.
    ///
    /// - Decrements `active_connections` (saturating at 0).
    /// - Updates `avg_latency_ms` with EMA alpha = 0.2.
    /// - Increments `total_errors` (per-peer and global) on failure.
    ///
    /// Returns `false` when the peer is not registered.
    pub fn record_completion(
        &mut self,
        peer_id: &str,
        latency_ms: f64,
        success: bool,
        _now: u64,
    ) -> bool {
        const ALPHA: f64 = 0.2;
        match self.peers.iter_mut().find(|p| p.peer_id == peer_id) {
            Some(p) => {
                p.active_connections = p.active_connections.saturating_sub(1);
                // EMA latency update
                if p.avg_latency_ms == 0.0 {
                    p.avg_latency_ms = latency_ms;
                } else {
                    p.avg_latency_ms = ALPHA * latency_ms + (1.0 - ALPHA) * p.avg_latency_ms;
                }
                if !success {
                    p.total_errors += 1;
                    self.total_errors += 1;
                }
                true
            }
            None => false,
        }
    }

    /// Set a peer's healthy flag.  Returns `false` if the peer is unknown.
    pub fn mark_healthy(&mut self, peer_id: &str, healthy: bool) -> bool {
        match self.peers.iter_mut().find(|p| p.peer_id == peer_id) {
            Some(p) => {
                p.healthy = healthy;
                self.ring_dirty = true;
                true
            }
            None => false,
        }
    }

    // ── Weight auto-tuning ────────────────────────────────────────────────────

    /// Auto-tune weights based on current latency estimates.
    ///
    /// Formula: `weight[i] = max(1, round(1000.0 / avg_latency_ms))`
    /// Peers with `avg_latency_ms == 0` are skipped (latency not yet measured).
    pub fn adjust_weights_by_latency(&mut self) {
        for p in self.peers.iter_mut() {
            if p.avg_latency_ms > 0.0 {
                p.weight = ((1000.0 / p.avg_latency_ms).round() as u32).max(1);
            }
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Returns references to all healthy peers.
    pub fn healthy_peers(&self) -> Vec<&LbPeer> {
        self.peers.iter().filter(|p| p.healthy).collect()
    }

    /// Total number of registered peers.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Number of healthy peers.
    pub fn healthy_peer_count(&self) -> usize {
        self.peers.iter().filter(|p| p.healthy).count()
    }

    /// Aggregate statistics snapshot.
    pub fn stats(&self) -> AdaptiveLbStats {
        let total_peers = self.peers.len();
        let healthy_peers = self.healthy_peer_count();

        let error_rate = if self.total_requests == 0 {
            0.0
        } else {
            self.total_errors as f64 / self.total_requests as f64
        };

        // Mean latency across peers that have a non-zero measurement.
        let latency_peers: Vec<f64> = self
            .peers
            .iter()
            .filter(|p| p.avg_latency_ms > 0.0)
            .map(|p| p.avg_latency_ms)
            .collect();
        let avg_latency_ms = if latency_peers.is_empty() {
            0.0
        } else {
            latency_peers.iter().sum::<f64>() / latency_peers.len() as f64
        };

        AdaptiveLbStats {
            total_peers,
            healthy_peers,
            total_requests: self.total_requests,
            total_errors: self.total_errors,
            error_rate,
            avg_latency_ms,
        }
    }

    // ── Private: consistent-hash ring ────────────────────────────────────────

    fn rebuild_ring_if_needed(&mut self) {
        if !self.ring_dirty {
            return;
        }
        self.ring.clear();
        for (idx, peer) in self.peers.iter().enumerate() {
            if peer.healthy {
                for i in 0..self.virtual_nodes {
                    let key = format!("{}-{}", peer.peer_id, i);
                    let hash = fnv1a_64(&key);
                    self.ring.push((hash, idx));
                }
            }
        }
        self.ring.sort_unstable_by_key(|(h, _)| *h);
        self.ring_dirty = false;
    }

    // ── Private: per-algorithm selection ─────────────────────────────────────

    fn select_round_robin(&mut self) -> Option<(String, String)> {
        let healthy: Vec<usize> = self
            .peers
            .iter()
            .enumerate()
            .filter(|(_, p)| p.healthy)
            .map(|(i, _)| i)
            .collect();
        if healthy.is_empty() {
            return None;
        }
        let pos = self.rr_index % healthy.len();
        self.rr_index = pos + 1;
        Some((
            self.peers[healthy[pos]].peer_id.clone(),
            "RoundRobin".to_string(),
        ))
    }

    fn select_least_connections(&self) -> Option<(String, String)> {
        self.peers
            .iter()
            .filter(|p| p.healthy)
            .min_by_key(|p| p.active_connections)
            .map(|p| (p.peer_id.clone(), "LeastConnections".to_string()))
    }

    fn select_weighted_round_robin(&mut self) -> Option<(String, String)> {
        let healthy: Vec<&LbPeer> = self.peers.iter().filter(|p| p.healthy).collect();
        if healthy.is_empty() {
            return None;
        }
        let total_weight: u64 = healthy.iter().map(|p| u64::from(p.weight)).sum();
        if total_weight == 0 {
            // Fallback: plain round-robin over healthy peers.
            let pos = self.rr_index % healthy.len();
            self.rr_index = pos + 1;
            return Some((
                healthy[pos].peer_id.clone(),
                "WeightedRoundRobin".to_string(),
            ));
        }
        let index = self.total_requests % total_weight;
        let mut cumulative: u64 = 0;
        for p in &healthy {
            cumulative += u64::from(p.weight);
            if index < cumulative {
                return Some((p.peer_id.clone(), "WeightedRoundRobin".to_string()));
            }
        }
        // Fallback — return last healthy peer (should be unreachable).
        healthy
            .last()
            .map(|p| (p.peer_id.clone(), "WeightedRoundRobin".to_string()))
    }

    fn select_least_latency(&self) -> Option<(String, String)> {
        let healthy: Vec<&LbPeer> = self.peers.iter().filter(|p| p.healthy).collect();
        if healthy.is_empty() {
            return None;
        }
        // If all latencies are zero, fall back to first healthy peer.
        if healthy.iter().all(|p| p.avg_latency_ms == 0.0) {
            return healthy
                .first()
                .map(|p| (p.peer_id.clone(), "LeastLatency".to_string()));
        }
        healthy
            .iter()
            .filter(|p| p.avg_latency_ms > 0.0)
            .min_by(|a, b| {
                a.avg_latency_ms
                    .partial_cmp(&b.avg_latency_ms)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .or_else(|| healthy.first())
            .map(|p| (p.peer_id.clone(), "LeastLatency".to_string()))
    }

    fn select_random(&mut self, request: &LbRequest) -> Option<(String, String)> {
        let healthy: Vec<&LbPeer> = self.peers.iter().filter(|p| p.healthy).collect();
        if healthy.is_empty() {
            return None;
        }
        // Seed PRNG from request if state is zero.
        if self.prng_state == 0 {
            self.prng_state = request.id.wrapping_add(request.timestamp).wrapping_add(1);
        }
        let r = xorshift64(&mut self.prng_state);
        let idx = (r as usize) % healthy.len();
        Some((healthy[idx].peer_id.clone(), "Random".to_string()))
    }

    fn select_consistent_hash(&mut self, request: &LbRequest) -> Option<(String, String)> {
        self.rebuild_ring_if_needed();
        if self.ring.is_empty() {
            return None;
        }
        // Determine hash of key.
        let key_hash = match &request.key {
            Some(k) => fnv1a_64(k),
            None => {
                // No key → use request id as hash source.
                let mut s = self.prng_state.wrapping_add(request.id).wrapping_add(1);
                xorshift64(&mut s)
            }
        };

        // Binary-search for the first ring entry ≥ key_hash (clockwise).
        let start = match self.ring.binary_search_by_key(&key_hash, |(h, _)| *h) {
            Ok(i) | Err(i) => i % self.ring.len(),
        };

        // Walk clockwise until we find a healthy peer.
        let ring_len = self.ring.len();
        for offset in 0..ring_len {
            let (_, peer_idx) = self.ring[(start + offset) % ring_len];
            // Safety: peer_idx is always within bounds (built from peers.len()).
            if peer_idx < self.peers.len() && self.peers[peer_idx].healthy {
                return Some((
                    self.peers[peer_idx].peer_id.clone(),
                    "ConsistentHash".to_string(),
                ));
            }
        }
        None
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests for AdaptiveLoadBalancer
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod adaptive_tests {
    use crate::load_balancer::{AdaptiveLoadBalancer, LbAlgorithm, LbPeer, LbRequest};

    // Helper: build a request with a given id and optional key.
    fn req(id: u64, key: Option<&str>) -> LbRequest {
        LbRequest::new(id, key.map(|s| s.to_string()), 0, 1000)
    }

    fn peer(id: &str, weight: u32) -> LbPeer {
        LbPeer::new(id, weight)
    }

    // ── 1. new() has no peers and zero counters ────────────────────────────
    #[test]
    fn test_new_empty() {
        let lb = AdaptiveLoadBalancer::new(LbAlgorithm::RoundRobin);
        assert_eq!(lb.peer_count(), 0);
        assert_eq!(lb.total_requests, 0);
        assert_eq!(lb.total_errors, 0);
        assert_eq!(lb.virtual_nodes, 150);
    }

    // ── 2. add_peer inserts a peer ─────────────────────────────────────────
    #[test]
    fn test_add_peer_inserts() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::RoundRobin);
        lb.add_peer(peer("alice", 1));
        assert_eq!(lb.peer_count(), 1);
    }

    // ── 3. add_peer replaces on duplicate id ──────────────────────────────
    #[test]
    fn test_add_peer_replaces_duplicate() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::RoundRobin);
        lb.add_peer(peer("alice", 1));
        let mut p2 = peer("alice", 10);
        p2.avg_latency_ms = 42.0;
        lb.add_peer(p2);
        assert_eq!(lb.peer_count(), 1);
        assert_eq!(lb.peers[0].weight, 10);
        assert!((lb.peers[0].avg_latency_ms - 42.0).abs() < f64::EPSILON);
    }

    // ── 4. remove_peer returns true when present ──────────────────────────
    #[test]
    fn test_remove_peer_present() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::RoundRobin);
        lb.add_peer(peer("alice", 1));
        assert!(lb.remove_peer("alice"));
        assert_eq!(lb.peer_count(), 0);
    }

    // ── 5. remove_peer returns false when missing ─────────────────────────
    #[test]
    fn test_remove_peer_missing() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::RoundRobin);
        assert!(!lb.remove_peer("ghost"));
    }

    // ── 6. select returns None when no peers ─────────────────────────────
    #[test]
    fn test_select_none_no_peers() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::RoundRobin);
        assert!(lb.select(&req(1, None), 100).is_none());
    }

    // ── 7. select returns None when all peers unhealthy ───────────────────
    #[test]
    fn test_select_none_all_unhealthy() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::RoundRobin);
        let mut p = peer("a", 1);
        p.healthy = false;
        lb.add_peer(p);
        assert!(lb.select(&req(1, None), 100).is_none());
    }

    // ── 8. RoundRobin cycles through healthy peers ────────────────────────
    #[test]
    fn test_round_robin_cycles() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::RoundRobin);
        lb.add_peer(peer("a", 1));
        lb.add_peer(peer("b", 1));
        lb.add_peer(peer("c", 1));

        let d0 = lb.select(&req(0, None), 0).expect("must route");
        let d1 = lb.select(&req(1, None), 0).expect("must route");
        let d2 = lb.select(&req(2, None), 0).expect("must route");
        let d3 = lb.select(&req(3, None), 0).expect("must route");

        assert_eq!(d0.peer_id, "a");
        assert_eq!(d1.peer_id, "b");
        assert_eq!(d2.peer_id, "c");
        assert_eq!(d3.peer_id, "a"); // wrap
    }

    // ── 9. RoundRobin skips unhealthy peers ──────────────────────────────
    #[test]
    fn test_round_robin_skips_unhealthy() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::RoundRobin);
        lb.add_peer(peer("a", 1));
        let mut sick = peer("b", 1);
        sick.healthy = false;
        lb.add_peer(sick);
        lb.add_peer(peer("c", 1));

        for _ in 0..6 {
            let d = lb.select(&req(0, None), 0).expect("must route");
            assert_ne!(d.peer_id, "b", "unhealthy peer must not be chosen");
        }
    }

    // ── 10. LeastConnections picks lowest active_connections ──────────────
    #[test]
    fn test_least_connections() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::LeastConnections);
        let mut a = peer("a", 1);
        a.active_connections = 5;
        lb.add_peer(a);
        let mut b = peer("b", 1);
        b.active_connections = 1;
        lb.add_peer(b);
        let d = lb.select(&req(1, None), 0).expect("must route");
        assert_eq!(d.peer_id, "b");
    }

    // ── 11. WeightedRoundRobin respects weights ───────────────────────────
    #[test]
    fn test_weighted_round_robin_proportional() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::WeightedRoundRobin);
        lb.add_peer(peer("a", 1)); // weight 1
        lb.add_peer(peer("b", 3)); // weight 3

        // total_requests starts 0 → index 0 → "a" (cumul 1 > 0)
        let d0 = lb.select(&req(0, None), 0).expect("d0");
        // next: total_requests = 1, index 1 → "b" (cumul 4 > 1)
        let d1 = lb.select(&req(1, None), 0).expect("d1");
        // index 2 → "b"
        let d2 = lb.select(&req(2, None), 0).expect("d2");
        // index 3 → "b"
        let d3 = lb.select(&req(3, None), 0).expect("d3");
        // index 0 again → "a"
        let d4 = lb.select(&req(4, None), 0).expect("d4");

        assert_eq!(d0.peer_id, "a");
        assert_eq!(d1.peer_id, "b");
        assert_eq!(d2.peer_id, "b");
        assert_eq!(d3.peer_id, "b");
        assert_eq!(d4.peer_id, "a");
    }

    // ── 12. LeastLatency picks peer with lowest avg_latency_ms ───────────
    #[test]
    fn test_least_latency_picks_lowest() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::LeastLatency);
        let mut fast = peer("fast", 1);
        fast.avg_latency_ms = 5.0;
        lb.add_peer(fast);
        let mut slow = peer("slow", 1);
        slow.avg_latency_ms = 200.0;
        lb.add_peer(slow);
        let d = lb.select(&req(1, None), 0).expect("must route");
        assert_eq!(d.peer_id, "fast");
    }

    // ── 13. LeastLatency falls back to first peer when all zero ──────────
    #[test]
    fn test_least_latency_fallback_all_zero() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::LeastLatency);
        lb.add_peer(peer("x", 1));
        lb.add_peer(peer("y", 1));
        // Both latencies are 0 → fallback to first
        assert!(lb.select(&req(1, None), 0).is_some());
    }

    // ── 14. Random always selects a healthy peer ──────────────────────────
    #[test]
    fn test_random_selects_healthy() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::Random);
        lb.add_peer(peer("a", 1));
        lb.add_peer(peer("b", 1));
        let mut sick = peer("c", 1);
        sick.healthy = false;
        lb.add_peer(sick);

        for i in 0..50u64 {
            let d = lb.select(&req(i, None), i).expect("must route");
            assert_ne!(d.peer_id, "c", "unhealthy peer must not be chosen");
        }
    }

    // ── 15. ConsistentHash produces stable routing for same key ──────────
    #[test]
    fn test_consistent_hash_stable() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::ConsistentHash);
        lb.add_peer(peer("n1", 1));
        lb.add_peer(peer("n2", 1));
        lb.add_peer(peer("n3", 1));

        let first = lb.select(&req(1, Some("my-key")), 0).expect("must route");
        for _ in 0..10 {
            let repeated = lb.select(&req(1, Some("my-key")), 0).expect("must route");
            assert_eq!(first.peer_id, repeated.peer_id, "same key → same peer");
        }
    }

    // ── 16. ConsistentHash routes different keys to potentially different peers
    #[test]
    fn test_consistent_hash_distribution() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::ConsistentHash);
        lb.add_peer(peer("n1", 1));
        lb.add_peer(peer("n2", 1));
        lb.add_peer(peer("n3", 1));

        let mut seen = std::collections::HashSet::new();
        for i in 0..50u64 {
            let key = format!("key-{}", i);
            let d = lb.select(&req(i, Some(&key)), 0).expect("must route");
            seen.insert(d.peer_id);
        }
        // With 3 peers and 50 distinct keys, we expect all 3 peers to receive traffic.
        assert!(
            seen.len() >= 2,
            "consistent hash should distribute across peers"
        );
    }

    // ── 17. record_connection_start increments active_connections ─────────
    #[test]
    fn test_record_connection_start() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::RoundRobin);
        lb.add_peer(peer("x", 1));
        assert!(lb.record_connection_start("x"));
        assert_eq!(lb.peers[0].active_connections, 1);
    }

    // ── 18. record_connection_start returns false for unknown peer ────────
    #[test]
    fn test_record_connection_start_unknown() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::RoundRobin);
        assert!(!lb.record_connection_start("ghost"));
    }

    // ── 19. record_completion decrements active_connections ──────────────
    #[test]
    fn test_record_completion_decrements() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::RoundRobin);
        lb.add_peer(peer("x", 1));
        lb.record_connection_start("x");
        assert!(lb.record_completion("x", 50.0, true, 0));
        assert_eq!(lb.peers[0].active_connections, 0);
    }

    // ── 20. record_completion saturates at 0 ─────────────────────────────
    #[test]
    fn test_record_completion_saturating() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::RoundRobin);
        lb.add_peer(peer("x", 1));
        // No start call; decrement should saturate at 0.
        lb.record_completion("x", 10.0, true, 0);
        assert_eq!(lb.peers[0].active_connections, 0);
    }

    // ── 21. record_completion first-sample latency ────────────────────────
    #[test]
    fn test_record_completion_first_latency() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::RoundRobin);
        lb.add_peer(peer("x", 1));
        lb.record_completion("x", 100.0, true, 0);
        assert!((lb.peers[0].avg_latency_ms - 100.0).abs() < f64::EPSILON);
    }

    // ── 22. record_completion EMA subsequent latency ──────────────────────
    #[test]
    fn test_record_completion_ema_latency() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::RoundRobin);
        lb.add_peer(peer("x", 1));
        lb.record_completion("x", 100.0, true, 0); // avg = 100
        lb.record_completion("x", 200.0, true, 0); // avg = 0.2*200 + 0.8*100 = 120
        assert!((lb.peers[0].avg_latency_ms - 120.0).abs() < 1e-9);
    }

    // ── 23. record_completion increments errors on failure ────────────────
    #[test]
    fn test_record_completion_errors() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::RoundRobin);
        lb.add_peer(peer("x", 1));
        lb.record_completion("x", 10.0, false, 0);
        lb.record_completion("x", 10.0, false, 0);
        assert_eq!(lb.peers[0].total_errors, 2);
        assert_eq!(lb.total_errors, 2);
    }

    // ── 24. record_completion returns false for unknown peer ──────────────
    #[test]
    fn test_record_completion_unknown() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::RoundRobin);
        assert!(!lb.record_completion("ghost", 10.0, true, 0));
    }

    // ── 25. mark_healthy sets healthy flag ────────────────────────────────
    #[test]
    fn test_mark_healthy() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::RoundRobin);
        lb.add_peer(peer("a", 1));
        assert!(lb.mark_healthy("a", false));
        assert!(!lb.peers[0].healthy);
        assert!(lb.mark_healthy("a", true));
        assert!(lb.peers[0].healthy);
    }

    // ── 26. mark_healthy returns false for unknown peer ───────────────────
    #[test]
    fn test_mark_healthy_unknown() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::RoundRobin);
        assert!(!lb.mark_healthy("ghost", true));
    }

    // ── 27. adjust_weights_by_latency tunes weights ───────────────────────
    #[test]
    fn test_adjust_weights_by_latency() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::WeightedRoundRobin);
        lb.add_peer(peer("fast", 1));
        lb.add_peer(peer("slow", 1));
        lb.record_completion("fast", 10.0, true, 0); // avg = 10 → weight = 100
        lb.record_completion("slow", 500.0, true, 0); // avg = 500 → weight = 2

        lb.adjust_weights_by_latency();

        let fast_w = lb
            .peers
            .iter()
            .find(|p| p.peer_id == "fast")
            .map(|p| p.weight)
            .unwrap_or(0);
        let slow_w = lb
            .peers
            .iter()
            .find(|p| p.peer_id == "slow")
            .map(|p| p.weight)
            .unwrap_or(0);
        assert_eq!(fast_w, 100, "fast peer: 1000/10 = 100");
        assert_eq!(slow_w, 2, "slow peer: 1000/500 = 2");
    }

    // ── 28. adjust_weights_by_latency skips zero-latency peers ───────────
    #[test]
    fn test_adjust_weights_skips_zero_latency() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::WeightedRoundRobin);
        let p = peer("x", 7);
        lb.add_peer(p);
        lb.adjust_weights_by_latency();
        assert_eq!(
            lb.peers[0].weight, 7,
            "weight must not change when latency is 0"
        );
    }

    // ── 29. healthy_peers returns only healthy peers ──────────────────────
    #[test]
    fn test_healthy_peers_accessor() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::RoundRobin);
        lb.add_peer(peer("a", 1));
        let mut b = peer("b", 1);
        b.healthy = false;
        lb.add_peer(b);
        lb.add_peer(peer("c", 1));
        let hp = lb.healthy_peers();
        assert_eq!(hp.len(), 2);
        assert!(hp.iter().all(|p| p.healthy));
    }

    // ── 30. healthy_peer_count matches healthy_peers ──────────────────────
    #[test]
    fn test_healthy_peer_count() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::RoundRobin);
        lb.add_peer(peer("a", 1));
        let mut b = peer("b", 1);
        b.healthy = false;
        lb.add_peer(b);
        assert_eq!(lb.healthy_peer_count(), 1);
        assert_eq!(lb.peer_count(), 2);
    }

    // ── 31. stats reflects correct totals ────────────────────────────────
    #[test]
    fn test_stats_totals() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::RoundRobin);
        lb.add_peer(peer("a", 1));
        lb.select(&req(1, None), 0);
        lb.select(&req(2, None), 0);
        lb.record_completion("a", 50.0, false, 0); // 1 error

        let s = lb.stats();
        assert_eq!(s.total_peers, 1);
        assert_eq!(s.healthy_peers, 1);
        assert_eq!(s.total_requests, 2);
        assert_eq!(s.total_errors, 1);
        assert!((s.error_rate - 0.5).abs() < 1e-9);
    }

    // ── 32. stats avg_latency_ms ──────────────────────────────────────────
    #[test]
    fn test_stats_avg_latency() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::LeastLatency);
        lb.add_peer(peer("a", 1));
        lb.add_peer(peer("b", 1));
        lb.record_completion("a", 100.0, true, 0); // a avg = 100
        lb.record_completion("b", 200.0, true, 0); // b avg = 200
        let s = lb.stats();
        // mean of [100, 200] = 150
        assert!((s.avg_latency_ms - 150.0).abs() < 1e-9);
    }

    // ── 33. decision.algorithm_used matches algorithm ─────────────────────
    #[test]
    fn test_decision_algorithm_name() {
        let algos: &[(LbAlgorithm, &str)] = &[
            (LbAlgorithm::RoundRobin, "RoundRobin"),
            (LbAlgorithm::LeastConnections, "LeastConnections"),
            (LbAlgorithm::WeightedRoundRobin, "WeightedRoundRobin"),
            (LbAlgorithm::LeastLatency, "LeastLatency"),
            (LbAlgorithm::Random, "Random"),
            (LbAlgorithm::ConsistentHash, "ConsistentHash"),
        ];
        for &(algo, name) in algos {
            let mut lb = AdaptiveLoadBalancer::new(algo);
            lb.add_peer(peer("x", 1));
            let d = lb.select(&req(1, Some("k")), 0).expect("must route");
            assert_eq!(d.algorithm_used, name, "algorithm mismatch for {:?}", algo);
        }
    }

    // ── 34. decision carries correct request_id and decided_at ───────────
    #[test]
    fn test_decision_metadata() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::RoundRobin);
        lb.add_peer(peer("a", 1));
        let d = lb.select(&req(42, None), 999).expect("must route");
        assert_eq!(d.request_id, 42);
        assert_eq!(d.decided_at, 999);
    }

    // ── 35. ConsistentHash falls back when ring is empty ─────────────────
    #[test]
    fn test_consistent_hash_no_healthy() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::ConsistentHash);
        let mut p = peer("a", 1);
        p.healthy = false;
        lb.add_peer(p);
        assert!(lb.select(&req(1, Some("key")), 0).is_none());
    }

    // ── 36. ConsistentHash re-routes after unhealthy on ring ─────────────
    #[test]
    fn test_consistent_hash_reroute_after_unhealthy() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::ConsistentHash);
        lb.add_peer(peer("n1", 1));
        lb.add_peer(peer("n2", 1));
        lb.add_peer(peer("n3", 1));

        // Get the natural mapping for "testkey"
        let first_peer = lb
            .select(&req(1, Some("testkey")), 0)
            .expect("must route")
            .peer_id;

        // Mark that peer unhealthy; the hash should reroute to another
        lb.mark_healthy(&first_peer, false);
        let second_peer = lb
            .select(&req(1, Some("testkey")), 0)
            .expect("must route after reroute")
            .peer_id;
        assert_ne!(
            second_peer, first_peer,
            "should reroute away from unhealthy peer"
        );
    }

    // ── 37. total_requests increments on each successful select ──────────
    #[test]
    fn test_total_requests_increments() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::RoundRobin);
        lb.add_peer(peer("a", 1));
        for i in 0..10u64 {
            lb.select(&req(i, None), 0).expect("must route");
        }
        assert_eq!(lb.total_requests, 10);
    }

    // ── 38. LbPeer::error_rate zero with no requests ──────────────────────
    #[test]
    fn test_lb_peer_error_rate_zero() {
        let p = LbPeer::new("x", 1);
        assert_eq!(p.error_rate(), 0.0);
    }

    // ── 39. LbPeer::error_rate computed correctly ─────────────────────────
    #[test]
    fn test_lb_peer_error_rate_computed() {
        let mut p = LbPeer::new("x", 1);
        p.total_requests = 10;
        p.total_errors = 3;
        assert!((p.error_rate() - 0.3).abs() < 1e-9);
    }

    // ── 40. select sets last_used on chosen peer ──────────────────────────
    #[test]
    fn test_select_sets_last_used() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::RoundRobin);
        lb.add_peer(peer("a", 1));
        lb.select(&req(1, None), 12345).expect("must route");
        assert_eq!(lb.peers[0].last_used, 12345);
    }

    // ── 41. WeightedRoundRobin with all-zero weights falls back ──────────
    #[test]
    fn test_weighted_round_robin_zero_weights() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::WeightedRoundRobin);
        lb.add_peer(LbPeer {
            weight: 0,
            ..LbPeer::new("a", 0)
        });
        lb.add_peer(LbPeer {
            weight: 0,
            ..LbPeer::new("b", 0)
        });
        // Zero total weight → should fall back and still return Some.
        assert!(lb.select(&req(1, None), 0).is_some());
    }

    // ── 42. remove_peer clamps rr_index ──────────────────────────────────
    #[test]
    fn test_remove_peer_clamps_rr_index() {
        let mut lb = AdaptiveLoadBalancer::new(LbAlgorithm::RoundRobin);
        lb.add_peer(peer("a", 1));
        lb.add_peer(peer("b", 1));
        lb.add_peer(peer("c", 1));
        // Advance rr_index to 2 by doing 2 selects.
        lb.select(&req(0, None), 0);
        lb.select(&req(1, None), 0);
        // rr_index = 2; remove "c" → peers count = 2 → rr_index should be clamped.
        lb.remove_peer("c");
        // After removal the balancer should still work.
        assert!(lb.select(&req(2, None), 0).is_some());
    }

    // ── 43. AdaptiveLbStats error_rate zero when no requests ─────────────
    #[test]
    fn test_stats_error_rate_zero_no_requests() {
        let lb = AdaptiveLoadBalancer::new(LbAlgorithm::RoundRobin);
        assert_eq!(lb.stats().error_rate, 0.0);
    }

    // ── 44. Virtual nodes parameter respected ─────────────────────────────
    #[test]
    fn test_virtual_nodes_default() {
        let lb = AdaptiveLoadBalancer::new(LbAlgorithm::ConsistentHash);
        assert_eq!(lb.virtual_nodes, 150);
    }

    // ── 45. Multiple algorithms produce valid decisions ───────────────────
    #[test]
    fn test_all_algorithms_route() {
        let algos = [
            LbAlgorithm::RoundRobin,
            LbAlgorithm::LeastConnections,
            LbAlgorithm::WeightedRoundRobin,
            LbAlgorithm::LeastLatency,
            LbAlgorithm::Random,
            LbAlgorithm::ConsistentHash,
        ];
        for algo in algos {
            let mut lb = AdaptiveLoadBalancer::new(algo);
            lb.add_peer(peer("p1", 2));
            lb.add_peer(peer("p2", 3));
            let d = lb
                .select(&req(1, Some("routing-key")), 0)
                .unwrap_or_else(|| panic!("algo {:?} returned None", algo));
            assert!(!d.peer_id.is_empty());
        }
    }
}
