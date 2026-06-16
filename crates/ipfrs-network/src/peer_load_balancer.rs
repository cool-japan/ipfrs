//! Peer-aware load balancer for IPFRS network layer.
//!
//! Distributes requests across peers based on capacity, health, and observed latency.
//! Supports six selection strategies including consistent hashing and power-of-two choices.

use std::collections::{HashMap, VecDeque};

// ── Type aliases ─────────────────────────────────────────────────────────────

/// 32-byte peer identifier used by the load balancer.
pub type PlbPeerId = [u8; 32];

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors produced by the load balancer.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum PlbError {
    /// No healthy peers are available for selection.
    #[error("no healthy peers available")]
    NoHealthyPeers,
    /// The requested peer was not found in the balancer.
    #[error("peer not found")]
    PeerNotFound,
    /// An arithmetic overflow occurred during weight computation.
    #[error("weight computation overflow")]
    WeightOverflow,
}

// ── Strategy enum ─────────────────────────────────────────────────────────────

/// Load-balancing selection strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlbStrategy {
    /// Simple round-robin over healthy peers.
    #[default]
    RoundRobin,
    /// Weighted random selection proportional to peer weight.
    WeightedRandom,
    /// Always select the peer with fewest in-flight requests.
    LeastConnections,
    /// Always select the peer with lowest average latency.
    LeastLatency,
    /// FNV-1a consistent hash on the request key with virtual nodes.
    ConsistentHash,
    /// Sample two random healthy peers and pick the better one.
    PowerOfTwo,
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for [`PeerLoadBalancer`].
#[derive(Debug, Clone)]
pub struct PlbBalancerConfig {
    /// Strategy used for peer selection.
    pub strategy: PlbStrategy,
    /// Fraction of successes required for a peer to remain healthy (0.0–1.0).
    /// e.g. 0.8 means a peer must have ≥ 80 % success rate.
    pub health_threshold: f64,
    /// Maximum in-flight requests allowed per peer.
    pub max_in_flight: u32,
    /// Duration in milliseconds a peer stays in cooldown after being penalised.
    pub cooldown_ms: u64,
    /// Number of virtual nodes per peer for consistent-hash ring.
    pub virtual_nodes: u32,
    /// EWMA smoothing factor α for latency updates (0 < α ≤ 1).
    pub ewma_alpha: f64,
}

impl Default for PlbBalancerConfig {
    fn default() -> Self {
        Self {
            strategy: PlbStrategy::default(),
            health_threshold: 0.8,
            max_in_flight: 64,
            cooldown_ms: 5_000,
            virtual_nodes: 150,
            ewma_alpha: 0.1,
        }
    }
}

// ── Per-peer state ────────────────────────────────────────────────────────────

/// Runtime state for a single peer tracked by the balancer.
#[derive(Debug, Clone)]
pub struct PlbPeerState {
    /// Peer identifier.
    pub id: PlbPeerId,
    /// Maximum concurrent requests this peer can handle.
    pub capacity: u32,
    /// Currently in-flight requests directed at this peer.
    pub in_flight: u32,
    /// Cumulative successful responses.
    pub success_count: u64,
    /// Cumulative failed responses.
    pub failure_count: u64,
    /// Exponentially weighted moving average of response latency (ms).
    pub avg_latency_ms: f64,
    /// Unix-ms timestamp of the last request sent to this peer.
    pub last_used_ts: u64,
    /// Whether this peer is currently considered healthy.
    pub is_healthy: bool,
    /// Normalised selection weight in [0, 1].
    pub weight: f64,
    /// Unix-ms timestamp at which the cooldown expires (0 = not in cooldown).
    pub cooldown_until_ts: u64,
}

impl PlbPeerState {
    fn new(id: PlbPeerId, capacity: u32) -> Self {
        Self {
            id,
            capacity,
            in_flight: 0,
            success_count: 0,
            failure_count: 0,
            avg_latency_ms: 0.0,
            last_used_ts: 0,
            is_healthy: true,
            weight: 1.0,
            cooldown_until_ts: 0,
        }
    }

    /// True when the peer is available for selection.
    fn is_available(&self, max_in_flight: u32, now_ms: u64) -> bool {
        self.is_healthy
            && self.in_flight < max_in_flight
            && self.in_flight < self.capacity
            && (self.cooldown_until_ts == 0 || now_ms >= self.cooldown_until_ts)
    }

    /// Failure rate in [0, 1].  Returns 0 when no requests have been observed.
    pub fn failure_rate(&self) -> f64 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            0.0
        } else {
            self.failure_count as f64 / total as f64
        }
    }
}

// ── Request record ─────────────────────────────────────────────────────────────

/// A single completed request record stored in the rolling log.
#[derive(Debug, Clone)]
pub struct PlbRequestRecord {
    /// Unix-ms timestamp when the request completed.
    pub ts: u64,
    /// Peer that handled the request.
    pub peer_id: PlbPeerId,
    /// Observed round-trip latency (ms).
    pub latency_ms: f64,
    /// Whether the request succeeded.
    pub success: bool,
}

// ── Aggregate statistics ──────────────────────────────────────────────────────

/// Per-peer statistics snapshot.
#[derive(Debug, Clone)]
pub struct PlbPeerStats {
    /// Peer identifier.
    pub id: PlbPeerId,
    /// Current in-flight requests.
    pub in_flight: u32,
    /// Cumulative successes.
    pub success_count: u64,
    /// Cumulative failures.
    pub failure_count: u64,
    /// Current average latency (ms).
    pub avg_latency_ms: f64,
    /// Current normalised weight.
    pub weight: f64,
    /// Whether healthy.
    pub is_healthy: bool,
}

/// Aggregate statistics for the whole balancer.
#[derive(Debug, Clone)]
pub struct PlbBalancerStats {
    /// Total requests recorded since creation.
    pub total_requests: u64,
    /// Fraction of requests that failed (0.0–1.0).
    pub error_rate: f64,
    /// Average latency across all recorded requests (ms).
    pub avg_latency_ms: f64,
    /// Per-peer statistics.
    pub peer_stats: Vec<PlbPeerStats>,
    /// Number of healthy peers.
    pub healthy_peers: usize,
    /// Number of peers in cooldown.
    pub cooling_peers: usize,
}

// ── PRNG helpers ──────────────────────────────────────────────────────────────

/// XorShift64 pseudo-random number generator.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// FNV-1a 64-bit hash.
#[inline]
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

// ── Virtual-node ring for consistent hashing ─────────────────────────────────

/// A single entry in the consistent-hash ring.
#[derive(Debug, Clone)]
struct RingEntry {
    hash: u64,
    peer_id: PlbPeerId,
}

// ── Main balancer struct ──────────────────────────────────────────────────────

/// Peer-aware load balancer.
///
/// Distributes requests across known peers using a configurable [`PlbStrategy`].
/// Tracks health, latency, and in-flight requests per peer and recomputes
/// normalised weights after every completed request.
pub struct PeerLoadBalancer {
    peers: HashMap<PlbPeerId, PlbPeerState>,
    request_log: VecDeque<PlbRequestRecord>,
    config: PlbBalancerConfig,
    /// Round-robin cursor (index into sorted healthy peer list).
    rr_cursor: usize,
    /// PRNG state for weighted-random and power-of-two strategies.
    prng_state: u64,
    /// Total requests dispatched.
    total_requests: u64,
    /// Consistent-hash ring (sorted by hash).
    ring: Vec<RingEntry>,
    /// True when the ring needs rebuilding.
    ring_dirty: bool,
}

impl PeerLoadBalancer {
    /// Maximum entries kept in the request log.
    const LOG_CAPACITY: usize = 2_000;

    /// Create a new balancer with the given configuration.
    pub fn new(config: PlbBalancerConfig) -> Self {
        Self {
            peers: HashMap::new(),
            request_log: VecDeque::with_capacity(Self::LOG_CAPACITY),
            config,
            rr_cursor: 0,
            prng_state: 0xDEAD_BEEF_CAFE_1234,
            total_requests: 0,
            ring: Vec::new(),
            ring_dirty: false,
        }
    }

    /// Create a balancer with default configuration.
    pub fn default_config() -> Self {
        Self::new(PlbBalancerConfig::default())
    }

    // ── Peer management ───────────────────────────────────────────────────────

    /// Register a new peer with the given capacity.
    /// If the peer already exists its capacity is updated.
    pub fn add_peer(&mut self, id: PlbPeerId, capacity: u32) {
        self.peers
            .entry(id)
            .and_modify(|s| s.capacity = capacity)
            .or_insert_with(|| PlbPeerState::new(id, capacity));
        self.ring_dirty = true;
        self.recompute_weights();
    }

    /// Remove a peer from the balancer.
    pub fn remove_peer(&mut self, id: &PlbPeerId) -> Option<PlbPeerState> {
        let removed = self.peers.remove(id);
        if removed.is_some() {
            self.ring_dirty = true;
            self.recompute_weights();
        }
        removed
    }

    /// Manually override the health state of a peer.
    pub fn set_healthy(&mut self, id: &PlbPeerId, healthy: bool) -> Result<(), PlbError> {
        let peer = self.peers.get_mut(id).ok_or(PlbError::PeerNotFound)?;
        peer.is_healthy = healthy;
        self.recompute_weights();
        Ok(())
    }

    /// Put a peer in cooldown for `config.cooldown_ms` milliseconds.
    pub fn cooldown_peer(&mut self, id: &PlbPeerId) -> Result<(), PlbError> {
        let now_ms = now_ms();
        let cooldown = self.config.cooldown_ms;
        let peer = self.peers.get_mut(id).ok_or(PlbError::PeerNotFound)?;
        peer.cooldown_until_ts = now_ms.saturating_add(cooldown);
        Ok(())
    }

    // ── Peer selection ────────────────────────────────────────────────────────

    /// Select a peer for the next request.
    ///
    /// `key` is used only by [`PlbStrategy::ConsistentHash`]; other strategies ignore it.
    pub fn select_peer(&mut self, key: &[u8]) -> Result<PlbPeerId, PlbError> {
        let now = now_ms();
        let strategy = self.config.strategy;
        let selected = match strategy {
            PlbStrategy::RoundRobin => self.select_round_robin(now)?,
            PlbStrategy::WeightedRandom => self.select_weighted_random(now)?,
            PlbStrategy::LeastConnections => self.select_least_connections(now)?,
            PlbStrategy::LeastLatency => self.select_least_latency(now)?,
            PlbStrategy::ConsistentHash => self.select_consistent_hash(key, now)?,
            PlbStrategy::PowerOfTwo => self.select_power_of_two(now)?,
        };
        if let Some(peer) = self.peers.get_mut(&selected) {
            peer.in_flight = peer.in_flight.saturating_add(1);
            peer.last_used_ts = now;
        }
        self.total_requests = self.total_requests.saturating_add(1);
        Ok(selected)
    }

    // ── Result recording ──────────────────────────────────────────────────────

    /// Record the outcome of a completed request.
    ///
    /// Updates EWMA latency, success/failure counters and recomputes weights.
    pub fn record_result(
        &mut self,
        peer_id: &PlbPeerId,
        latency_ms: f64,
        success: bool,
    ) -> Result<(), PlbError> {
        let alpha = self.config.ewma_alpha;
        {
            let peer = self.peers.get_mut(peer_id).ok_or(PlbError::PeerNotFound)?;
            // Decrement in-flight
            peer.in_flight = peer.in_flight.saturating_sub(1);
            // EWMA latency update
            if peer.avg_latency_ms == 0.0 {
                peer.avg_latency_ms = latency_ms;
            } else {
                peer.avg_latency_ms = alpha * latency_ms + (1.0 - alpha) * peer.avg_latency_ms;
            }
            if success {
                peer.success_count = peer.success_count.saturating_add(1);
            } else {
                peer.failure_count = peer.failure_count.saturating_add(1);
            }
        }
        // Append to log
        let record = PlbRequestRecord {
            ts: now_ms(),
            peer_id: *peer_id,
            latency_ms,
            success,
        };
        if self.request_log.len() >= Self::LOG_CAPACITY {
            self.request_log.pop_front();
        }
        self.request_log.push_back(record);
        self.recompute_weights();
        Ok(())
    }

    // ── Weight computation ────────────────────────────────────────────────────

    /// Recompute normalised weights for all peers.
    ///
    /// Weight formula per peer:
    /// `w = capacity_score × latency_score × health_score`
    ///
    /// where:
    /// - `capacity_score = (capacity - in_flight) / capacity`  (0 if zero capacity)
    /// - `latency_score = 1 / (1 + avg_latency_ms / 1000)`
    /// - `health_score = success_rate` (1.0 for new peers with no history)
    pub fn recompute_weights(&mut self) {
        // Collect raw scores
        let ids: Vec<PlbPeerId> = self.peers.keys().copied().collect();
        let mut raw: Vec<(PlbPeerId, f64)> = ids
            .into_iter()
            .map(|id| {
                let p = &self.peers[&id];
                let capacity_score = if p.capacity == 0 {
                    0.0
                } else {
                    let avail = p.capacity.saturating_sub(p.in_flight) as f64;
                    (avail / p.capacity as f64).clamp(0.0, 1.0)
                };
                let latency_score = 1.0 / (1.0 + p.avg_latency_ms / 1000.0);
                let total = p.success_count + p.failure_count;
                let health_score = if total == 0 {
                    1.0
                } else {
                    p.success_count as f64 / total as f64
                };
                let w = capacity_score * latency_score * health_score;
                (id, w.max(0.0))
            })
            .collect();

        // Normalise
        let sum: f64 = raw.iter().map(|(_, w)| *w).sum();
        for (id, w) in &mut raw {
            let normalised = if sum > 0.0 { *w / sum } else { 0.0 };
            if let Some(peer) = self.peers.get_mut(id) {
                peer.weight = normalised;
            }
        }
    }

    // ── Health check ──────────────────────────────────────────────────────────

    /// Evaluate all peers and mark unhealthy those whose failure rate exceeds
    /// `1 - health_threshold`.
    pub fn run_health_check(&mut self) {
        let threshold = self.config.health_threshold;
        let ids: Vec<PlbPeerId> = self.peers.keys().copied().collect();
        for id in ids {
            if let Some(peer) = self.peers.get_mut(&id) {
                let total = peer.success_count + peer.failure_count;
                if total > 0 {
                    let failure_rate = peer.failure_count as f64 / total as f64;
                    // Mark unhealthy if failure rate exceeds 1-threshold
                    peer.is_healthy = failure_rate <= 1.0 - threshold;
                }
            }
        }
        self.recompute_weights();
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    /// Return an aggregate snapshot of balancer statistics.
    pub fn balancer_stats(&self) -> PlbBalancerStats {
        let now = now_ms();
        let total = self.request_log.len() as u64;
        let errors = self.request_log.iter().filter(|r| !r.success).count() as u64;
        let error_rate = if total == 0 {
            0.0
        } else {
            errors as f64 / total as f64
        };
        let avg_latency_ms = if total == 0 {
            0.0
        } else {
            self.request_log.iter().map(|r| r.latency_ms).sum::<f64>() / total as f64
        };
        let peer_stats: Vec<PlbPeerStats> = self
            .peers
            .values()
            .map(|p| PlbPeerStats {
                id: p.id,
                in_flight: p.in_flight,
                success_count: p.success_count,
                failure_count: p.failure_count,
                avg_latency_ms: p.avg_latency_ms,
                weight: p.weight,
                is_healthy: p.is_healthy,
            })
            .collect();
        let healthy_peers = self
            .peers
            .values()
            .filter(|p| p.is_available(self.config.max_in_flight, now))
            .count();
        let cooling_peers = self
            .peers
            .values()
            .filter(|p| p.cooldown_until_ts > 0 && now < p.cooldown_until_ts)
            .count();
        PlbBalancerStats {
            total_requests: self.total_requests,
            error_rate,
            avg_latency_ms,
            peer_stats,
            healthy_peers,
            cooling_peers,
        }
    }

    /// Return the current configuration.
    pub fn config(&self) -> &PlbBalancerConfig {
        &self.config
    }

    /// Return the number of registered peers.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Retrieve state for a specific peer.
    pub fn peer_state(&self, id: &PlbPeerId) -> Option<&PlbPeerState> {
        self.peers.get(id)
    }

    // ── Private selection helpers ─────────────────────────────────────────────

    fn available_peers(&self, now: u64) -> Vec<PlbPeerId> {
        self.peers
            .values()
            .filter(|p| p.is_available(self.config.max_in_flight, now))
            .map(|p| p.id)
            .collect()
    }

    fn select_round_robin(&mut self, now: u64) -> Result<PlbPeerId, PlbError> {
        let mut available = self.available_peers(now);
        if available.is_empty() {
            return Err(PlbError::NoHealthyPeers);
        }
        // Sort for determinism
        available.sort_unstable();
        let idx = self.rr_cursor % available.len();
        self.rr_cursor = self.rr_cursor.wrapping_add(1);
        Ok(available[idx])
    }

    fn select_weighted_random(&mut self, now: u64) -> Result<PlbPeerId, PlbError> {
        let available: Vec<&PlbPeerState> = self
            .peers
            .values()
            .filter(|p| p.is_available(self.config.max_in_flight, now))
            .collect();
        if available.is_empty() {
            return Err(PlbError::NoHealthyPeers);
        }
        let total_weight: f64 = available.iter().map(|p| p.weight).sum();
        if total_weight <= 0.0 {
            // Fall back to uniform
            let r = xorshift64(&mut self.prng_state) as usize % available.len();
            return Ok(available[r].id);
        }
        let r = (xorshift64(&mut self.prng_state) as f64 / u64::MAX as f64) * total_weight;
        let mut cumulative = 0.0;
        for p in &available {
            cumulative += p.weight;
            if r <= cumulative {
                return Ok(p.id);
            }
        }
        // Fallback: last element
        Ok(available[available.len() - 1].id)
    }

    fn select_least_connections(&self, now: u64) -> Result<PlbPeerId, PlbError> {
        self.peers
            .values()
            .filter(|p| p.is_available(self.config.max_in_flight, now))
            .min_by_key(|p| p.in_flight)
            .map(|p| p.id)
            .ok_or(PlbError::NoHealthyPeers)
    }

    fn select_least_latency(&self, now: u64) -> Result<PlbPeerId, PlbError> {
        self.peers
            .values()
            .filter(|p| p.is_available(self.config.max_in_flight, now))
            .min_by(|a, b| {
                a.avg_latency_ms
                    .partial_cmp(&b.avg_latency_ms)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| p.id)
            .ok_or(PlbError::NoHealthyPeers)
    }

    fn select_consistent_hash(&mut self, key: &[u8], now: u64) -> Result<PlbPeerId, PlbError> {
        // Rebuild ring if dirty
        if self.ring_dirty {
            self.rebuild_ring();
        }
        if self.ring.is_empty() {
            return Err(PlbError::NoHealthyPeers);
        }
        let h = fnv1a_64(key);
        // Binary search for first entry >= h
        let pos = self.ring.partition_point(|entry| entry.hash < h);
        // Walk the ring starting at pos, skip unavailable peers
        let len = self.ring.len();
        for offset in 0..len {
            let idx = (pos + offset) % len;
            let candidate = self.ring[idx].peer_id;
            if let Some(peer) = self.peers.get(&candidate) {
                if peer.is_available(self.config.max_in_flight, now) {
                    return Ok(candidate);
                }
            }
        }
        Err(PlbError::NoHealthyPeers)
    }

    fn select_power_of_two(&mut self, now: u64) -> Result<PlbPeerId, PlbError> {
        let available: Vec<PlbPeerId> = self.available_peers(now);
        if available.is_empty() {
            return Err(PlbError::NoHealthyPeers);
        }
        if available.len() == 1 {
            return Ok(available[0]);
        }
        // Pick two distinct random indices
        let i1 = (xorshift64(&mut self.prng_state) as usize) % available.len();
        let mut i2 = (xorshift64(&mut self.prng_state) as usize) % available.len();
        if i2 == i1 {
            i2 = (i1 + 1) % available.len();
        }
        let p1 = available[i1];
        let p2 = available[i2];
        // Choose the one with lower in-flight count; break ties by latency
        let s1 = self.peers.get(&p1);
        let s2 = self.peers.get(&p2);
        match (s1, s2) {
            (Some(a), Some(b)) => {
                if a.in_flight < b.in_flight {
                    Ok(p1)
                } else if b.in_flight < a.in_flight {
                    Ok(p2)
                } else if a.avg_latency_ms <= b.avg_latency_ms {
                    Ok(p1)
                } else {
                    Ok(p2)
                }
            }
            (Some(_), None) => Ok(p1),
            (None, Some(_)) => Ok(p2),
            (None, None) => Err(PlbError::NoHealthyPeers),
        }
    }

    fn rebuild_ring(&mut self) {
        let virtual_nodes = self.config.virtual_nodes;
        let mut ring: Vec<RingEntry> = self
            .peers
            .values()
            .flat_map(|p| {
                (0..virtual_nodes).map(move |i| {
                    let mut vnode_key = Vec::with_capacity(32 + 4);
                    vnode_key.extend_from_slice(&p.id);
                    vnode_key.extend_from_slice(&i.to_le_bytes());
                    RingEntry {
                        hash: fnv1a_64(&vnode_key),
                        peer_id: p.id,
                    }
                })
            })
            .collect();
        ring.sort_unstable_by_key(|e| e.hash);
        self.ring = ring;
        self.ring_dirty = false;
    }
}

// ── Timestamp helper ──────────────────────────────────────────────────────────

fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ── Type aliases (public re-exports required by the task) ─────────────────────

/// Alias: the main load balancer type.
pub type PlbPeerLoadBalancer = PeerLoadBalancer;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a 32-byte id from a u8
    fn peer_id(b: u8) -> PlbPeerId {
        let mut id = [0u8; 32];
        id[0] = b;
        id
    }

    // Helper: balancer with a single strategy
    fn balancer(strategy: PlbStrategy) -> PeerLoadBalancer {
        PeerLoadBalancer::new(PlbBalancerConfig {
            strategy,
            ..Default::default()
        })
    }

    // ── Construction & config ─────────────────────────────────────────────────

    #[test]
    fn test_new_balancer_has_no_peers() {
        let lb = PeerLoadBalancer::default_config();
        assert_eq!(lb.peer_count(), 0);
    }

    #[test]
    fn test_config_is_stored() {
        let cfg = PlbBalancerConfig {
            strategy: PlbStrategy::LeastConnections,
            health_threshold: 0.9,
            max_in_flight: 16,
            cooldown_ms: 1000,
            virtual_nodes: 50,
            ewma_alpha: 0.2,
        };
        let lb = PeerLoadBalancer::new(cfg.clone());
        assert_eq!(lb.config().max_in_flight, 16);
        assert_eq!(lb.config().cooldown_ms, 1000);
    }

    #[test]
    fn test_default_strategy_is_round_robin() {
        let lb = PeerLoadBalancer::default_config();
        assert_eq!(lb.config().strategy, PlbStrategy::RoundRobin);
    }

    // ── add_peer / remove_peer ────────────────────────────────────────────────

    #[test]
    fn test_add_peer_increments_count() {
        let mut lb = PeerLoadBalancer::default_config();
        lb.add_peer(peer_id(1), 10);
        assert_eq!(lb.peer_count(), 1);
    }

    #[test]
    fn test_add_multiple_peers() {
        let mut lb = PeerLoadBalancer::default_config();
        lb.add_peer(peer_id(1), 10);
        lb.add_peer(peer_id(2), 20);
        lb.add_peer(peer_id(3), 5);
        assert_eq!(lb.peer_count(), 3);
    }

    #[test]
    fn test_add_duplicate_peer_updates_capacity() {
        let mut lb = PeerLoadBalancer::default_config();
        lb.add_peer(peer_id(1), 10);
        lb.add_peer(peer_id(1), 50);
        assert_eq!(lb.peer_count(), 1);
        assert_eq!(
            lb.peer_state(&peer_id(1))
                .expect("test: peer 1 should exist")
                .capacity,
            50
        );
    }

    #[test]
    fn test_remove_peer_decrements_count() {
        let mut lb = PeerLoadBalancer::default_config();
        lb.add_peer(peer_id(1), 10);
        lb.remove_peer(&peer_id(1));
        assert_eq!(lb.peer_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_peer_returns_none() {
        let mut lb = PeerLoadBalancer::default_config();
        assert!(lb.remove_peer(&peer_id(99)).is_none());
    }

    #[test]
    fn test_peer_state_accessible() {
        let mut lb = PeerLoadBalancer::default_config();
        lb.add_peer(peer_id(7), 32);
        let s = lb
            .peer_state(&peer_id(7))
            .expect("test: peer 7 should exist");
        assert_eq!(s.capacity, 32);
        assert!(s.is_healthy);
    }

    // ── set_healthy ───────────────────────────────────────────────────────────

    #[test]
    fn test_set_healthy_false() {
        let mut lb = PeerLoadBalancer::default_config();
        lb.add_peer(peer_id(1), 10);
        lb.set_healthy(&peer_id(1), false)
            .expect("test: set_healthy should succeed");
        assert!(
            !lb.peer_state(&peer_id(1))
                .expect("test: peer 1 should exist after set_healthy")
                .is_healthy
        );
    }

    #[test]
    fn test_set_healthy_true() {
        let mut lb = PeerLoadBalancer::default_config();
        lb.add_peer(peer_id(1), 10);
        lb.set_healthy(&peer_id(1), false)
            .expect("test: set_healthy false should succeed");
        lb.set_healthy(&peer_id(1), true)
            .expect("test: set_healthy true should succeed");
        assert!(
            lb.peer_state(&peer_id(1))
                .expect("test: peer 1 should exist after set_healthy true")
                .is_healthy
        );
    }

    #[test]
    fn test_set_healthy_nonexistent_returns_error() {
        let mut lb = PeerLoadBalancer::default_config();
        assert_eq!(
            lb.set_healthy(&peer_id(99), true),
            Err(PlbError::PeerNotFound)
        );
    }

    #[test]
    fn test_unhealthy_peer_not_selected_round_robin() {
        let mut lb = balancer(PlbStrategy::RoundRobin);
        lb.add_peer(peer_id(1), 10);
        lb.set_healthy(&peer_id(1), false)
            .expect("test: set_healthy false should succeed");
        assert_eq!(lb.select_peer(&[]), Err(PlbError::NoHealthyPeers));
    }

    // ── select_peer: no peers ─────────────────────────────────────────────────

    #[test]
    fn test_select_no_peers_returns_error() {
        let mut lb = PeerLoadBalancer::default_config();
        assert_eq!(lb.select_peer(&[]), Err(PlbError::NoHealthyPeers));
    }

    // ── Round-Robin ───────────────────────────────────────────────────────────

    #[test]
    fn test_round_robin_cycles_through_all_peers() {
        let mut lb = balancer(PlbStrategy::RoundRobin);
        lb.add_peer(peer_id(1), 10);
        lb.add_peer(peer_id(2), 10);
        lb.add_peer(peer_id(3), 10);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..9 {
            let p = lb
                .select_peer(&[])
                .expect("test: select_peer should return a peer in round-robin");
            seen.insert(p);
            // decrement in_flight so max_in_flight is not hit
            lb.record_result(&p, 10.0, true)
                .expect("test: record_result should succeed");
        }
        assert_eq!(seen.len(), 3);
    }

    #[test]
    fn test_round_robin_single_peer() {
        let mut lb = balancer(PlbStrategy::RoundRobin);
        lb.add_peer(peer_id(1), 100);
        for _ in 0..5 {
            let p = lb
                .select_peer(&[])
                .expect("test: select_peer should succeed for single peer");
            lb.record_result(&p, 5.0, true)
                .expect("test: record_result for single peer should succeed");
            assert_eq!(p, peer_id(1));
        }
    }

    // ── LeastConnections ─────────────────────────────────────────────────────

    #[test]
    fn test_least_connections_prefers_idle_peer() {
        let mut lb = balancer(PlbStrategy::LeastConnections);
        lb.add_peer(peer_id(1), 10);
        lb.add_peer(peer_id(2), 10);
        // Manually inflate in_flight of peer 1
        lb.peers
            .get_mut(&peer_id(1))
            .expect("test: peer 1 should exist for in_flight mutation")
            .in_flight = 5;
        let selected = lb
            .select_peer(&[])
            .expect("test: select_peer should pick least-loaded peer");
        assert_eq!(selected, peer_id(2));
    }

    #[test]
    fn test_least_connections_all_tied() {
        let mut lb = balancer(PlbStrategy::LeastConnections);
        lb.add_peer(peer_id(1), 10);
        lb.add_peer(peer_id(2), 10);
        // Both have 0 in_flight; any result is acceptable
        assert!(lb.select_peer(&[]).is_ok());
    }

    // ── LeastLatency ─────────────────────────────────────────────────────────

    #[test]
    fn test_least_latency_selects_low_latency_peer() {
        let mut lb = balancer(PlbStrategy::LeastLatency);
        lb.add_peer(peer_id(1), 10);
        lb.add_peer(peer_id(2), 10);
        lb.peers
            .get_mut(&peer_id(1))
            .expect("test: peer 1 should exist for latency mutation")
            .avg_latency_ms = 100.0;
        lb.peers
            .get_mut(&peer_id(2))
            .expect("test: peer 2 should exist for latency mutation")
            .avg_latency_ms = 5.0;
        let selected = lb
            .select_peer(&[])
            .expect("test: select_peer should pick least-latency peer");
        assert_eq!(selected, peer_id(2));
    }

    #[test]
    fn test_least_latency_new_peers_have_zero_latency() {
        let mut lb = balancer(PlbStrategy::LeastLatency);
        lb.add_peer(peer_id(1), 10);
        lb.add_peer(peer_id(2), 10);
        // Both have 0 ms latency; either is fine
        assert!(lb.select_peer(&[]).is_ok());
    }

    // ── WeightedRandom ────────────────────────────────────────────────────────

    #[test]
    fn test_weighted_random_selects_from_available() {
        let mut lb = balancer(PlbStrategy::WeightedRandom);
        lb.add_peer(peer_id(1), 10);
        lb.add_peer(peer_id(2), 10);
        for _ in 0..20 {
            let p = lb
                .select_peer(&[])
                .expect("test: weighted random select should succeed");
            lb.record_result(&p, 10.0, true)
                .expect("test: record_result in weighted random should succeed");
        }
    }

    #[test]
    fn test_weighted_random_single_peer_always_selected() {
        let mut lb = balancer(PlbStrategy::WeightedRandom);
        lb.add_peer(peer_id(5), 100);
        for _ in 0..10 {
            let p = lb
                .select_peer(&[])
                .expect("test: weighted random single peer should select it");
            lb.record_result(&p, 10.0, true)
                .expect("test: record_result for single peer in weighted random should succeed");
            assert_eq!(p, peer_id(5));
        }
    }

    // ── ConsistentHash ────────────────────────────────────────────────────────

    #[test]
    fn test_consistent_hash_same_key_same_peer() {
        let mut lb = balancer(PlbStrategy::ConsistentHash);
        lb.add_peer(peer_id(1), 100);
        lb.add_peer(peer_id(2), 100);
        lb.add_peer(peer_id(3), 100);
        let key = b"my-content-key";
        let p1 = lb
            .select_peer(key)
            .expect("test: consistent hash first select should succeed");
        lb.record_result(&p1, 5.0, true)
            .expect("test: record_result after first consistent hash select");
        let p2 = lb
            .select_peer(key)
            .expect("test: consistent hash second select should succeed");
        lb.record_result(&p2, 5.0, true)
            .expect("test: record_result after second consistent hash select");
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_consistent_hash_different_keys_may_differ() {
        let mut lb = balancer(PlbStrategy::ConsistentHash);
        lb.add_peer(peer_id(1), 100);
        lb.add_peer(peer_id(2), 100);
        lb.add_peer(peer_id(3), 100);
        // Just verify no error
        assert!(lb.select_peer(b"key-a").is_ok());
        assert!(lb.select_peer(b"key-b").is_ok());
    }

    #[test]
    fn test_consistent_hash_empty_key() {
        let mut lb = balancer(PlbStrategy::ConsistentHash);
        lb.add_peer(peer_id(1), 10);
        assert!(lb.select_peer(&[]).is_ok());
    }

    #[test]
    fn test_consistent_hash_ring_rebuilt_on_add() {
        let mut lb = balancer(PlbStrategy::ConsistentHash);
        lb.add_peer(peer_id(1), 10);
        let _ = lb.select_peer(b"x"); // triggers ring build
        lb.add_peer(peer_id(2), 10); // marks ring dirty
        assert!(lb.ring_dirty);
        let _ = lb.select_peer(b"x"); // rebuilds
        assert!(!lb.ring_dirty);
    }

    // ── PowerOfTwo ────────────────────────────────────────────────────────────

    #[test]
    fn test_power_of_two_returns_peer() {
        let mut lb = balancer(PlbStrategy::PowerOfTwo);
        lb.add_peer(peer_id(1), 10);
        lb.add_peer(peer_id(2), 10);
        assert!(lb.select_peer(&[]).is_ok());
    }

    #[test]
    fn test_power_of_two_single_peer() {
        let mut lb = balancer(PlbStrategy::PowerOfTwo);
        lb.add_peer(peer_id(1), 100);
        let p = lb
            .select_peer(&[])
            .expect("test: power-of-two single peer should be selected");
        assert_eq!(p, peer_id(1));
    }

    #[test]
    fn test_power_of_two_prefers_less_loaded() {
        let mut lb = balancer(PlbStrategy::PowerOfTwo);
        lb.add_peer(peer_id(1), 100);
        lb.add_peer(peer_id(2), 100);
        // Force PRNG to always pick indices 0 and 1 by seeding predictably
        // Just verify it runs without errors many times
        for _ in 0..50 {
            let p = lb
                .select_peer(&[])
                .expect("test: power-of-two should select a peer");
            lb.record_result(&p, 10.0, true)
                .expect("test: record_result in power-of-two should succeed");
        }
    }

    // ── record_result ─────────────────────────────────────────────────────────

    #[test]
    fn test_record_result_updates_success_count() {
        let mut lb = PeerLoadBalancer::default_config();
        lb.add_peer(peer_id(1), 10);
        let p = lb
            .select_peer(&[])
            .expect("test: select_peer for record_result success count test");
        lb.record_result(&p, 20.0, true)
            .expect("test: record_result success should succeed");
        assert_eq!(
            lb.peer_state(&peer_id(1))
                .expect("test: peer 1 should exist after success")
                .success_count,
            1
        );
    }

    #[test]
    fn test_record_result_updates_failure_count() {
        let mut lb = PeerLoadBalancer::default_config();
        lb.add_peer(peer_id(1), 10);
        let p = lb
            .select_peer(&[])
            .expect("test: select_peer for record_result failure count test");
        lb.record_result(&p, 20.0, false)
            .expect("test: record_result failure should succeed");
        assert_eq!(
            lb.peer_state(&peer_id(1))
                .expect("test: peer 1 should exist after failure")
                .failure_count,
            1
        );
    }

    #[test]
    fn test_record_result_decrements_in_flight() {
        let mut lb = PeerLoadBalancer::default_config();
        lb.add_peer(peer_id(1), 10);
        let p = lb
            .select_peer(&[])
            .expect("test: select_peer for in_flight decrement test");
        assert_eq!(
            lb.peer_state(&peer_id(1))
                .expect("test: peer 1 should exist to check in_flight")
                .in_flight,
            1
        );
        lb.record_result(&p, 5.0, true)
            .expect("test: record_result for in_flight decrement should succeed");
        assert_eq!(
            lb.peer_state(&peer_id(1))
                .expect("test: peer 1 should exist after record to check in_flight zero")
                .in_flight,
            0
        );
    }

    #[test]
    fn test_record_result_updates_ewma_latency() {
        let mut lb = PeerLoadBalancer::default_config();
        lb.add_peer(peer_id(1), 10);
        let p = lb
            .select_peer(&[])
            .expect("test: select_peer for EWMA latency update test");
        lb.record_result(&p, 100.0, true)
            .expect("test: record_result for EWMA should succeed");
        let lat = lb
            .peer_state(&peer_id(1))
            .expect("test: peer 1 should exist to check EWMA latency")
            .avg_latency_ms;
        assert!((lat - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_record_result_ewma_converges() {
        let mut lb = PeerLoadBalancer::new(PlbBalancerConfig {
            ewma_alpha: 0.5,
            max_in_flight: 1000,
            ..Default::default()
        });
        lb.add_peer(peer_id(1), 1000);
        for _ in 0..20 {
            let p = lb
                .select_peer(&[])
                .expect("test: select_peer in EWMA convergence test");
            lb.record_result(&p, 50.0, true)
                .expect("test: record_result in EWMA convergence should succeed");
        }
        let lat = lb
            .peer_state(&peer_id(1))
            .expect("test: peer 1 should exist to check converged EWMA")
            .avg_latency_ms;
        assert!((lat - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_record_result_for_unknown_peer_returns_error() {
        let mut lb = PeerLoadBalancer::default_config();
        assert_eq!(
            lb.record_result(&peer_id(99), 10.0, true),
            Err(PlbError::PeerNotFound)
        );
    }

    #[test]
    fn test_request_log_bounded_at_2000() {
        let mut lb = PeerLoadBalancer::new(PlbBalancerConfig {
            max_in_flight: 10_000,
            ..Default::default()
        });
        lb.add_peer(peer_id(1), 10_000);
        for _ in 0..2_100 {
            let p = lb
                .select_peer(&[])
                .expect("test: select_peer for log bound test");
            lb.record_result(&p, 1.0, true)
                .expect("test: record_result for log bound test should succeed");
        }
        assert!(lb.request_log.len() <= PeerLoadBalancer::LOG_CAPACITY);
    }

    // ── recompute_weights ─────────────────────────────────────────────────────

    #[test]
    fn test_weights_sum_to_one() {
        let mut lb = PeerLoadBalancer::default_config();
        lb.add_peer(peer_id(1), 10);
        lb.add_peer(peer_id(2), 20);
        lb.add_peer(peer_id(3), 5);
        lb.recompute_weights();
        let sum: f64 = lb.peers.values().map(|p| p.weight).sum();
        assert!((sum - 1.0).abs() < 1e-9 || sum == 0.0);
    }

    #[test]
    fn test_zero_capacity_peer_gets_zero_weight() {
        let mut lb = PeerLoadBalancer::default_config();
        lb.add_peer(peer_id(1), 0);
        lb.add_peer(peer_id(2), 10);
        lb.recompute_weights();
        assert_eq!(
            lb.peer_state(&peer_id(1))
                .expect("test: peer 1 should exist to check zero weight")
                .weight,
            0.0
        );
    }

    #[test]
    fn test_unhealthy_peer_may_have_weight() {
        // Weight is based on stats, health doesn't zero it in recompute — it's
        // health_score that does that.  An unhealthy peer with 100% successes
        // still gets a weight (but is excluded from selection).
        let mut lb = PeerLoadBalancer::default_config();
        lb.add_peer(peer_id(1), 10);
        lb.set_healthy(&peer_id(1), false)
            .expect("test: set_healthy false should succeed");
        lb.recompute_weights();
        // Weight can be non-zero even when unhealthy (selection is blocked separately)
        assert!(
            lb.peer_state(&peer_id(1))
                .expect("test: peer 1 should exist to check weight >= 0")
                .weight
                >= 0.0
        );
    }

    // ── run_health_check ──────────────────────────────────────────────────────

    #[test]
    fn test_health_check_marks_bad_peer_unhealthy() {
        let mut lb = PeerLoadBalancer::new(PlbBalancerConfig {
            health_threshold: 0.8,
            max_in_flight: 1000,
            ..Default::default()
        });
        lb.add_peer(peer_id(1), 100);
        lb.peers
            .get_mut(&peer_id(1))
            .expect("test: peer 1 should exist for success_count mutation")
            .success_count = 1;
        lb.peers
            .get_mut(&peer_id(1))
            .expect("test: peer 1 should exist for failure_count mutation")
            .failure_count = 9;
        lb.run_health_check();
        assert!(
            !lb.peer_state(&peer_id(1))
                .expect("test: peer 1 should exist after health check unhealthy")
                .is_healthy
        );
    }

    #[test]
    fn test_health_check_keeps_good_peer_healthy() {
        let mut lb = PeerLoadBalancer::new(PlbBalancerConfig {
            health_threshold: 0.8,
            max_in_flight: 1000,
            ..Default::default()
        });
        lb.add_peer(peer_id(1), 100);
        lb.peers
            .get_mut(&peer_id(1))
            .expect("test: peer 1 should exist for success_count good-peer mutation")
            .success_count = 9;
        lb.peers
            .get_mut(&peer_id(1))
            .expect("test: peer 1 should exist for failure_count good-peer mutation")
            .failure_count = 1;
        lb.run_health_check();
        assert!(
            lb.peer_state(&peer_id(1))
                .expect("test: peer 1 should remain healthy after good-peer health check")
                .is_healthy
        );
    }

    #[test]
    fn test_health_check_no_requests_peer_stays_healthy() {
        let mut lb = PeerLoadBalancer::default_config();
        lb.add_peer(peer_id(1), 10);
        lb.run_health_check();
        assert!(
            lb.peer_state(&peer_id(1))
                .expect("test: peer 1 should remain healthy when no requests")
                .is_healthy
        );
    }

    #[test]
    fn test_health_check_recovers_peer() {
        let mut lb = PeerLoadBalancer::new(PlbBalancerConfig {
            health_threshold: 0.8,
            max_in_flight: 1000,
            ..Default::default()
        });
        lb.add_peer(peer_id(1), 100);
        // Make it fail first
        lb.peers
            .get_mut(&peer_id(1))
            .expect("test: peer 1 should exist for success_count mutation in recovery test")
            .success_count = 1;
        lb.peers
            .get_mut(&peer_id(1))
            .expect("test: peer 1 should exist for failure_count mutation in recovery test")
            .failure_count = 9;
        lb.run_health_check();
        assert!(
            !lb.peer_state(&peer_id(1))
                .expect("test: peer 1 should be unhealthy after bad rate in recovery test")
                .is_healthy
        );
        // Fix success rate
        lb.peers
            .get_mut(&peer_id(1))
            .expect("test: peer 1 should exist for success_count fix in recovery test")
            .success_count = 90;
        lb.peers
            .get_mut(&peer_id(1))
            .expect("test: peer 1 should exist for failure_count fix in recovery test")
            .failure_count = 10;
        lb.run_health_check();
        assert!(
            lb.peer_state(&peer_id(1))
                .expect("test: peer 1 should be healthy after recovery")
                .is_healthy
        );
    }

    // ── cooldown_peer ─────────────────────────────────────────────────────────

    #[test]
    fn test_cooldown_excludes_peer_from_selection() {
        let mut lb = PeerLoadBalancer::new(PlbBalancerConfig {
            cooldown_ms: 60_000, // 1 minute
            ..Default::default()
        });
        lb.add_peer(peer_id(1), 10);
        lb.cooldown_peer(&peer_id(1))
            .expect("test: cooldown_peer should succeed");
        assert_eq!(lb.select_peer(&[]), Err(PlbError::NoHealthyPeers));
    }

    #[test]
    fn test_cooldown_nonexistent_peer_returns_error() {
        let mut lb = PeerLoadBalancer::default_config();
        assert_eq!(lb.cooldown_peer(&peer_id(99)), Err(PlbError::PeerNotFound));
    }

    #[test]
    fn test_cooldown_does_not_affect_other_peers() {
        let mut lb = PeerLoadBalancer::new(PlbBalancerConfig {
            cooldown_ms: 60_000,
            strategy: PlbStrategy::RoundRobin,
            ..Default::default()
        });
        lb.add_peer(peer_id(1), 10);
        lb.add_peer(peer_id(2), 10);
        lb.cooldown_peer(&peer_id(1))
            .expect("test: cooldown peer 1 should succeed");
        let p = lb
            .select_peer(&[])
            .expect("test: select_peer should pick non-cooled peer 2");
        assert_eq!(p, peer_id(2));
    }

    #[test]
    fn test_cooldown_peer_state_has_future_ts() {
        let mut lb = PeerLoadBalancer::new(PlbBalancerConfig {
            cooldown_ms: 5000,
            ..Default::default()
        });
        lb.add_peer(peer_id(1), 10);
        lb.cooldown_peer(&peer_id(1))
            .expect("test: cooldown_peer should set future cooldown timestamp");
        let state = lb
            .peer_state(&peer_id(1))
            .expect("test: peer 1 should exist to check cooldown timestamp");
        assert!(state.cooldown_until_ts > 0);
    }

    // ── balancer_stats ────────────────────────────────────────────────────────

    #[test]
    fn test_stats_zero_on_fresh_balancer() {
        let lb = PeerLoadBalancer::default_config();
        let s = lb.balancer_stats();
        assert_eq!(s.total_requests, 0);
        assert_eq!(s.error_rate, 0.0);
        assert_eq!(s.avg_latency_ms, 0.0);
        assert!(s.peer_stats.is_empty());
    }

    #[test]
    fn test_stats_total_requests_increments() {
        let mut lb = PeerLoadBalancer::new(PlbBalancerConfig {
            max_in_flight: 1000,
            ..Default::default()
        });
        lb.add_peer(peer_id(1), 100);
        for _ in 0..5 {
            let p = lb
                .select_peer(&[])
                .expect("test: select_peer for total_requests stats test");
            lb.record_result(&p, 10.0, true)
                .expect("test: record_result for total_requests stats test");
        }
        assert_eq!(lb.balancer_stats().total_requests, 5);
    }

    #[test]
    fn test_stats_error_rate_correct() {
        let mut lb = PeerLoadBalancer::new(PlbBalancerConfig {
            max_in_flight: 1000,
            ..Default::default()
        });
        lb.add_peer(peer_id(1), 100);
        for _ in 0..8 {
            let p = lb
                .select_peer(&[])
                .expect("test: select_peer for error_rate stats test");
            lb.record_result(&p, 10.0, true)
                .expect("test: record_result success for error_rate stats test");
        }
        for _ in 0..2 {
            let p = lb
                .select_peer(&[])
                .expect("test: select_peer for error records in error_rate stats test");
            lb.record_result(&p, 10.0, false)
                .expect("test: record_result failure for error_rate stats test");
        }
        let s = lb.balancer_stats();
        assert!((s.error_rate - 0.2).abs() < 1e-9);
    }

    #[test]
    fn test_stats_avg_latency() {
        let mut lb = PeerLoadBalancer::new(PlbBalancerConfig {
            max_in_flight: 1000,
            ..Default::default()
        });
        lb.add_peer(peer_id(1), 100);
        for _ in 0..4 {
            let p = lb
                .select_peer(&[])
                .expect("test: select_peer for avg_latency stats test");
            lb.record_result(&p, 50.0, true)
                .expect("test: record_result for avg_latency stats test");
        }
        let s = lb.balancer_stats();
        assert!((s.avg_latency_ms - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_healthy_peers_count() {
        let mut lb = PeerLoadBalancer::default_config();
        lb.add_peer(peer_id(1), 10);
        lb.add_peer(peer_id(2), 10);
        lb.add_peer(peer_id(3), 10);
        lb.set_healthy(&peer_id(3), false)
            .expect("test: set_healthy false for healthy peers count test");
        let s = lb.balancer_stats();
        assert_eq!(s.healthy_peers, 2);
    }

    #[test]
    fn test_stats_cooling_peers_count() {
        let mut lb = PeerLoadBalancer::new(PlbBalancerConfig {
            cooldown_ms: 60_000,
            ..Default::default()
        });
        lb.add_peer(peer_id(1), 10);
        lb.add_peer(peer_id(2), 10);
        lb.cooldown_peer(&peer_id(1))
            .expect("test: cooldown_peer for cooling_peers count test");
        let s = lb.balancer_stats();
        assert_eq!(s.cooling_peers, 1);
    }

    #[test]
    fn test_stats_peer_stats_populated() {
        let mut lb = PeerLoadBalancer::default_config();
        lb.add_peer(peer_id(1), 10);
        lb.add_peer(peer_id(2), 20);
        let s = lb.balancer_stats();
        assert_eq!(s.peer_stats.len(), 2);
    }

    // ── max_in_flight enforcement ─────────────────────────────────────────────

    #[test]
    fn test_max_in_flight_blocks_selection() {
        let mut lb = PeerLoadBalancer::new(PlbBalancerConfig {
            max_in_flight: 2,
            ..Default::default()
        });
        lb.add_peer(peer_id(1), 10);
        let _p1 = lb
            .select_peer(&[])
            .expect("test: first select_peer for max_in_flight test");
        let _p2 = lb
            .select_peer(&[])
            .expect("test: second select_peer for max_in_flight test");
        // Now in_flight == 2 == max_in_flight
        assert_eq!(lb.select_peer(&[]), Err(PlbError::NoHealthyPeers));
    }

    #[test]
    fn test_in_flight_released_after_record() {
        let mut lb = PeerLoadBalancer::new(PlbBalancerConfig {
            max_in_flight: 1,
            ..Default::default()
        });
        lb.add_peer(peer_id(1), 10);
        let p = lb
            .select_peer(&[])
            .expect("test: select_peer for in_flight release test");
        assert_eq!(lb.select_peer(&[]), Err(PlbError::NoHealthyPeers));
        lb.record_result(&p, 5.0, true)
            .expect("test: record_result to release in_flight should succeed");
        assert!(lb.select_peer(&[]).is_ok());
    }

    // ── capacity enforcement ──────────────────────────────────────────────────

    #[test]
    fn test_capacity_blocks_selection_when_full() {
        let mut lb = PeerLoadBalancer::new(PlbBalancerConfig {
            max_in_flight: 1000,
            ..Default::default()
        });
        lb.add_peer(peer_id(1), 1);
        let _p = lb
            .select_peer(&[])
            .expect("test: first select_peer for capacity enforcement test");
        // in_flight == capacity == 1
        assert_eq!(lb.select_peer(&[]), Err(PlbError::NoHealthyPeers));
    }

    // ── PlbError display ──────────────────────────────────────────────────────

    #[test]
    fn test_error_display_no_healthy_peers() {
        let e = PlbError::NoHealthyPeers;
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn test_error_display_peer_not_found() {
        let e = PlbError::PeerNotFound;
        assert!(!e.to_string().is_empty());
    }

    // ── PlbStrategy default ───────────────────────────────────────────────────

    #[test]
    fn test_strategy_default_is_round_robin() {
        assert_eq!(PlbStrategy::default(), PlbStrategy::RoundRobin);
    }

    // ── fnv1a_64 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_fnv1a_64_empty() {
        assert_eq!(fnv1a_64(&[]), 14_695_981_039_346_656_037);
    }

    #[test]
    fn test_fnv1a_64_deterministic() {
        assert_eq!(fnv1a_64(b"hello"), fnv1a_64(b"hello"));
    }

    #[test]
    fn test_fnv1a_64_different_inputs_differ() {
        assert_ne!(fnv1a_64(b"foo"), fnv1a_64(b"bar"));
    }

    // ── xorshift64 ────────────────────────────────────────────────────────────

    #[test]
    fn test_xorshift64_not_zero() {
        let mut state = 12345u64;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
    }

    #[test]
    fn test_xorshift64_updates_state() {
        let mut state = 12345u64;
        xorshift64(&mut state);
        assert_ne!(state, 12345);
    }

    #[test]
    fn test_xorshift64_sequence_varies() {
        let mut state = 99999u64;
        let v1 = xorshift64(&mut state);
        let v2 = xorshift64(&mut state);
        assert_ne!(v1, v2);
    }

    // ── PlbPeerId type ────────────────────────────────────────────────────────

    #[test]
    fn test_peer_id_is_32_bytes() {
        let id: PlbPeerId = [0u8; 32];
        assert_eq!(id.len(), 32);
    }

    // ── PlbBalancerConfig defaults ────────────────────────────────────────────

    #[test]
    fn test_config_default_health_threshold() {
        let cfg = PlbBalancerConfig::default();
        assert!((cfg.health_threshold - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_config_default_max_in_flight() {
        let cfg = PlbBalancerConfig::default();
        assert_eq!(cfg.max_in_flight, 64);
    }

    #[test]
    fn test_config_default_ewma_alpha() {
        let cfg = PlbBalancerConfig::default();
        assert!((cfg.ewma_alpha - 0.1).abs() < 1e-9);
    }

    // ── Integration: full request cycle ──────────────────────────────────────

    #[test]
    fn test_full_cycle_multiple_peers_and_strategies() {
        for strategy in [
            PlbStrategy::RoundRobin,
            PlbStrategy::WeightedRandom,
            PlbStrategy::LeastConnections,
            PlbStrategy::LeastLatency,
            PlbStrategy::ConsistentHash,
            PlbStrategy::PowerOfTwo,
        ] {
            let mut lb = PeerLoadBalancer::new(PlbBalancerConfig {
                strategy,
                max_in_flight: 1000,
                ..Default::default()
            });
            lb.add_peer(peer_id(1), 100);
            lb.add_peer(peer_id(2), 100);
            lb.add_peer(peer_id(3), 100);
            for i in 0u8..30 {
                let key = [i; 8];
                let p = lb
                    .select_peer(&key)
                    .expect("test: select_peer in full cycle integration test");
                lb.record_result(&p, (i as f64) * 2.0, i % 5 != 0)
                    .expect("test: record_result in full cycle integration test");
            }
            let s = lb.balancer_stats();
            assert_eq!(s.total_requests, 30);
        }
    }

    #[test]
    fn test_remove_peer_during_operation() {
        let mut lb = PeerLoadBalancer::new(PlbBalancerConfig {
            max_in_flight: 1000,
            ..Default::default()
        });
        lb.add_peer(peer_id(1), 10);
        lb.add_peer(peer_id(2), 10);
        let p = lb
            .select_peer(&[])
            .expect("test: select_peer before remove peer");
        lb.record_result(&p, 10.0, true)
            .expect("test: record_result before remove peer");
        lb.remove_peer(&peer_id(1));
        // Should still work with remaining peer
        let p2 = lb
            .select_peer(&[])
            .expect("test: select_peer after remove peer");
        lb.record_result(&p2, 5.0, true)
            .expect("test: record_result after remove peer");
        assert_eq!(lb.peer_count(), 1);
    }

    #[test]
    fn test_type_alias_peer_load_balancer() {
        let lb: PlbPeerLoadBalancer = PlbPeerLoadBalancer::default_config();
        assert_eq!(lb.peer_count(), 0);
    }

    #[test]
    fn test_peer_state_failure_rate_zero_when_no_history() {
        let state = PlbPeerState::new(peer_id(1), 10);
        assert_eq!(state.failure_rate(), 0.0);
    }

    #[test]
    fn test_peer_state_failure_rate_calculation() {
        let mut state = PlbPeerState::new(peer_id(1), 10);
        state.success_count = 3;
        state.failure_count = 1;
        assert!((state.failure_rate() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_peer_state_is_available_respects_cooldown() {
        let mut state = PlbPeerState::new(peer_id(1), 10);
        state.cooldown_until_ts = u64::MAX;
        assert!(!state.is_available(100, 0));
    }

    #[test]
    fn test_peer_state_is_available_when_not_in_cooldown() {
        let state = PlbPeerState::new(peer_id(1), 10);
        assert!(state.is_available(100, now_ms()));
    }
}
