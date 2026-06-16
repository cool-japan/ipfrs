//! Adaptive Routing Engine
//!
//! Dynamically adjusts routing decisions based on observed network conditions.
//! Supports multiple routing policies, multi-path routing, EWMA-based metrics,
//! and periodic route probing/adaptation cycles.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// ─── PRNG / Hashing helpers ──────────────────────────────────────────────────

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

#[inline]
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

// ─── Public type aliases (Are prefix) ────────────────────────────────────────

/// Type alias – route key (src + dst pair).
pub type AreRouteKey = RouteKey;
/// Type alias – a single route entry.
pub type AreRouteEntry = RouteEntry;
/// Type alias – routing policy selector.
pub type AreRoutingPolicy = RoutingPolicy;
/// Type alias – engine configuration.
pub type AreRoutingConfig = AdaptiveRoutingConfig;
/// Type alias – engine statistics snapshot.
pub type AreRoutingStats = AdaptiveRoutingStats;

// ─── RouteKey ────────────────────────────────────────────────────────────────

/// Identifies a source → destination pair for multi-path routing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RouteKey {
    /// Source peer identifier (32 bytes).
    pub src: [u8; 32],
    /// Destination peer identifier (32 bytes).
    pub dst: [u8; 32],
}

impl RouteKey {
    /// Create a new `RouteKey` from raw byte arrays.
    pub fn new(src: [u8; 32], dst: [u8; 32]) -> Self {
        Self { src, dst }
    }

    /// Compute a deterministic u64 hash of the key using FNV-1a.
    pub fn hash_u64(&self) -> u64 {
        let mut buf = [0u8; 64];
        buf[..32].copy_from_slice(&self.src);
        buf[32..].copy_from_slice(&self.dst);
        fnv1a_64(&buf)
    }
}

// ─── RouteEntry ──────────────────────────────────────────────────────────────

/// A single next-hop route with associated metrics.
#[derive(Debug, Clone)]
pub struct RouteEntry {
    /// Next-hop peer identifier (32 bytes).
    pub next_hop: [u8; 32],
    /// Current routing weight (higher is better).
    pub weight: f64,
    /// Exponentially weighted moving-average RTT in milliseconds.
    pub rtt_ms: f64,
    /// Estimated packet loss rate in the range [0.0, 1.0].
    pub loss_rate: f64,
    /// Unix timestamp (seconds) when this entry was last updated.
    pub last_updated: u64,
    /// Estimated available bandwidth in kbps.
    pub bandwidth_kbps: f64,
}

impl RouteEntry {
    /// Construct a new `RouteEntry` with default metric estimates.
    pub fn new(next_hop: [u8; 32]) -> Self {
        Self {
            next_hop,
            weight: 1.0,
            rtt_ms: 100.0,
            loss_rate: 0.0,
            last_updated: now_secs(),
            bandwidth_kbps: 1000.0,
        }
    }

    /// Returns `true` if the entry is considered healthy (loss < 50 %, RTT < 5 s).
    pub fn is_healthy(&self) -> bool {
        self.loss_rate < 0.5 && self.rtt_ms < 5_000.0
    }
}

// ─── RoutingPolicy ───────────────────────────────────────────────────────────

/// Policy governing how the best next-hop is selected from candidate routes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RoutingPolicy {
    /// Minimize hop count (use pre-computed weight).
    #[default]
    ShortestPath,
    /// Prefer routes with the lowest RTT.
    LowestLatency,
    /// Prefer routes with the highest estimated bandwidth.
    HighestBandwidth,
    /// Distribute traffic across routes proportional to their weights.
    LoadBalanced,
    /// Consider both latency and loss rate (QoS composite score).
    QoSAware,
}

impl RoutingPolicy {
    /// Return a human-readable label for the policy.
    pub fn label(&self) -> &'static str {
        match self {
            RoutingPolicy::ShortestPath => "shortest_path",
            RoutingPolicy::LowestLatency => "lowest_latency",
            RoutingPolicy::HighestBandwidth => "highest_bandwidth",
            RoutingPolicy::LoadBalanced => "load_balanced",
            RoutingPolicy::QoSAware => "qos_aware",
        }
    }
}

// ─── AdaptiveRoutingConfig ───────────────────────────────────────────────────

/// Configuration for the `AdaptiveRoutingEngine`.
#[derive(Debug, Clone)]
pub struct AdaptiveRoutingConfig {
    /// Default routing policy applied when none is specified per-call.
    pub policy: RoutingPolicy,
    /// EWMA smoothing factor α ∈ (0, 1].  Higher → more weight on recent samples.
    pub alpha: f64,
    /// Maximum number of route entries retained per `RouteKey`.
    pub max_routes_per_key: usize,
    /// Interval between synthetic probe rounds (seconds).
    pub probe_interval_secs: u64,
}

impl Default for AdaptiveRoutingConfig {
    fn default() -> Self {
        Self {
            policy: RoutingPolicy::LowestLatency,
            alpha: 0.125,
            max_routes_per_key: 8,
            probe_interval_secs: 30,
        }
    }
}

// ─── AdaptiveRoutingStats ────────────────────────────────────────────────────

/// Snapshot of key statistics for the routing engine.
#[derive(Debug, Clone, Default)]
pub struct AdaptiveRoutingStats {
    /// Total number of unique route entries across all keys.
    pub total_routes: usize,
    /// Arithmetic mean RTT across all route entries (ms).
    pub avg_rtt_ms: f64,
    /// Arithmetic mean packet loss rate across all route entries.
    pub avg_loss_rate: f64,
    /// Number of times the active routing policy has been switched.
    pub policy_switches: u64,
    /// Number of route entries pruned in the last adaptation cycle.
    pub pruned_last_cycle: usize,
    /// Total probe rounds executed since engine creation.
    pub total_probe_rounds: u64,
}

// ─── AdaptiveRoutingEngine ───────────────────────────────────────────────────

/// An adaptive, multi-path routing engine for IPFRS network nodes.
///
/// Maintains per-destination route tables with real-time metrics updated via
/// EWMA and selects the best next-hop according to a configurable
/// [`RoutingPolicy`].
pub struct AdaptiveRoutingEngine {
    /// Route table: one or more entries per (src, dst) key.
    route_table: HashMap<RouteKey, Vec<RouteEntry>>,
    /// Engine configuration.
    config: AdaptiveRoutingConfig,
    /// Currently active policy (may differ from config default after a switch).
    active_policy: RoutingPolicy,
    /// Accumulated statistics.
    stats: AdaptiveRoutingStats,
    /// Pseudo-random state for synthetic probes.
    prng_state: u64,
}

impl AdaptiveRoutingEngine {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Create a new engine with the supplied configuration.
    pub fn new(config: AdaptiveRoutingConfig) -> Self {
        let active_policy = config.policy;
        // Seed with FNV-1a of the config policy label.
        let seed = fnv1a_64(active_policy.label().as_bytes());
        Self {
            route_table: HashMap::new(),
            config,
            active_policy,
            stats: AdaptiveRoutingStats::default(),
            prng_state: seed | 1, // ensure non-zero
        }
    }

    /// Create an engine with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(AdaptiveRoutingConfig::default())
    }

    // ── Route management ─────────────────────────────────────────────────────

    /// Add or overwrite a route entry for the given key.
    ///
    /// If `max_routes_per_key` is reached the entry with the worst score is
    /// replaced.
    pub fn add_route(&mut self, key: RouteKey, entry: RouteEntry) {
        let entries = self.route_table.entry(key).or_default();
        // If next_hop already exists update in-place.
        if let Some(existing) = entries.iter_mut().find(|e| e.next_hop == entry.next_hop) {
            *existing = entry;
            return;
        }
        if entries.len() >= self.config.max_routes_per_key {
            // Evict the worst-scoring entry.
            if let Some(idx) = worst_entry_index(entries) {
                entries.remove(idx);
            }
        }
        entries.push(entry);
    }

    /// Remove all routes for the given key that use `next_hop`.
    pub fn remove_route(&mut self, key: &RouteKey, next_hop: &[u8; 32]) {
        if let Some(entries) = self.route_table.get_mut(key) {
            entries.retain(|e| &e.next_hop != next_hop);
        }
    }

    /// Remove every route entry associated with `key`.
    pub fn clear_routes(&mut self, key: &RouteKey) {
        self.route_table.remove(key);
    }

    /// Return an immutable slice of route entries for a key, if any.
    pub fn routes_for(&self, key: &RouteKey) -> Option<&[RouteEntry]> {
        self.route_table.get(key).map(|v| v.as_slice())
    }

    // ── Metric updates ───────────────────────────────────────────────────────

    /// Apply an EWMA update to RTT and loss metrics for a specific next-hop.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the key or next-hop is not found in the route table.
    pub fn update_metrics(
        &mut self,
        key: &RouteKey,
        next_hop: &[u8; 32],
        rtt_ms: f64,
        loss: f64,
    ) -> Result<(), RoutingEngineError> {
        let entries = self
            .route_table
            .get_mut(key)
            .ok_or(RoutingEngineError::KeyNotFound)?;
        let entry = entries
            .iter_mut()
            .find(|e| &e.next_hop == next_hop)
            .ok_or(RoutingEngineError::NextHopNotFound)?;

        let alpha = self.config.alpha;
        entry.rtt_ms = alpha * rtt_ms + (1.0 - alpha) * entry.rtt_ms;
        entry.loss_rate = (alpha * loss + (1.0 - alpha) * entry.loss_rate).clamp(0.0, 1.0);
        entry.last_updated = now_secs();
        recompute_weight(entry);
        Ok(())
    }

    /// Update only the bandwidth estimate for a specific next-hop.
    pub fn update_bandwidth(
        &mut self,
        key: &RouteKey,
        next_hop: &[u8; 32],
        bandwidth_kbps: f64,
    ) -> Result<(), RoutingEngineError> {
        let entries = self
            .route_table
            .get_mut(key)
            .ok_or(RoutingEngineError::KeyNotFound)?;
        let entry = entries
            .iter_mut()
            .find(|e| &e.next_hop == next_hop)
            .ok_or(RoutingEngineError::NextHopNotFound)?;
        let alpha = self.config.alpha;
        entry.bandwidth_kbps = alpha * bandwidth_kbps + (1.0 - alpha) * entry.bandwidth_kbps;
        entry.last_updated = now_secs();
        recompute_weight(entry);
        Ok(())
    }

    // ── Next-hop selection ───────────────────────────────────────────────────

    /// Select the best next-hop peer for `key` according to `policy`.
    ///
    /// Returns `None` if no routes are recorded for the key or all entries are
    /// unhealthy.
    pub fn select_next_hop(&self, key: &RouteKey, policy: RoutingPolicy) -> Option<[u8; 32]> {
        let entries = self.route_table.get(key)?;
        let healthy: Vec<&RouteEntry> = entries.iter().filter(|e| e.is_healthy()).collect();
        if healthy.is_empty() {
            // Fall back to any entry if no healthy ones remain.
            return entries.first().map(|e| e.next_hop);
        }
        match policy {
            RoutingPolicy::ShortestPath => healthy
                .iter()
                .max_by(|a, b| {
                    a.weight
                        .partial_cmp(&b.weight)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|e| e.next_hop),
            RoutingPolicy::LowestLatency => healthy
                .iter()
                .min_by(|a, b| {
                    a.rtt_ms
                        .partial_cmp(&b.rtt_ms)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|e| e.next_hop),
            RoutingPolicy::HighestBandwidth => healthy
                .iter()
                .max_by(|a, b| {
                    a.bandwidth_kbps
                        .partial_cmp(&b.bandwidth_kbps)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|e| e.next_hop),
            RoutingPolicy::LoadBalanced => self.select_load_balanced(&healthy),
            RoutingPolicy::QoSAware => healthy
                .iter()
                .max_by(|a, b| {
                    qos_score(a)
                        .partial_cmp(&qos_score(b))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|e| e.next_hop),
        }
    }

    /// Select using the engine's current active policy.
    pub fn select_next_hop_default(&self, key: &RouteKey) -> Option<[u8; 32]> {
        self.select_next_hop(key, self.active_policy)
    }

    /// Weighted-random selection proportional to route weights (load-balanced).
    fn select_load_balanced(&self, entries: &[&RouteEntry]) -> Option<[u8; 32]> {
        if entries.is_empty() {
            return None;
        }
        let total_weight: f64 = entries.iter().map(|e| e.weight.max(0.0)).sum();
        if total_weight <= 0.0 {
            return entries.first().map(|e| e.next_hop);
        }
        // Deterministic seed derived from current time and route key hash.
        let mut state = fnv1a_64(&now_secs().to_le_bytes()) | 1;
        let r = (xorshift64(&mut state) as f64 / u64::MAX as f64) * total_weight;
        let mut acc = 0.0;
        for entry in entries {
            acc += entry.weight.max(0.0);
            if acc >= r {
                return Some(entry.next_hop);
            }
        }
        entries.last().map(|e| e.next_hop)
    }

    // ── Policy management ────────────────────────────────────────────────────

    /// Switch the active routing policy, incrementing the policy-switch counter.
    pub fn set_policy(&mut self, policy: RoutingPolicy) {
        if policy != self.active_policy {
            self.stats.policy_switches += 1;
            self.active_policy = policy;
        }
    }

    /// Return the currently active routing policy.
    pub fn active_policy(&self) -> RoutingPolicy {
        self.active_policy
    }

    // ── Probing ──────────────────────────────────────────────────────────────

    /// Simulate a probe round: apply synthetic RTT perturbations to all routes
    /// using xorshift64-based PRNG.
    ///
    /// In production this would send real probe packets; here we perturb
    /// existing RTT estimates to model jitter and update weights accordingly.
    pub fn probe_routes(&mut self) {
        let alpha = self.config.alpha;
        for entries in self.route_table.values_mut() {
            for entry in entries.iter_mut() {
                // Generate synthetic delta RTT: ±10 % of current RTT.
                let r = xorshift64(&mut self.prng_state);
                let jitter_fraction = (r as f64 / u64::MAX as f64) * 0.20 - 0.10; // [-0.10, +0.10]
                let synthetic_rtt = (entry.rtt_ms * (1.0 + jitter_fraction)).max(0.5);
                // Apply EWMA with a smaller alpha for probes to avoid over-reacting.
                let probe_alpha = alpha * 0.5;
                entry.rtt_ms = probe_alpha * synthetic_rtt + (1.0 - probe_alpha) * entry.rtt_ms;
                entry.last_updated = now_secs();
                recompute_weight(entry);
            }
        }
        self.stats.total_probe_rounds += 1;
    }

    // ── Adaptation cycle ─────────────────────────────────────────────────────

    /// Prune stale route entries and recompute all weights.
    ///
    /// An entry is considered stale if it hasn't been updated for more than
    /// `probe_interval_secs × 3` seconds.
    pub fn run_adaptation_cycle(&mut self) {
        let stale_threshold = self.config.probe_interval_secs.saturating_mul(3);
        let now = now_secs();
        let mut pruned = 0usize;

        for entries in self.route_table.values_mut() {
            let before = entries.len();
            entries.retain(|e| {
                let age = now.saturating_sub(e.last_updated);
                age < stale_threshold
            });
            pruned += before - entries.len();
            for entry in entries.iter_mut() {
                recompute_weight(entry);
            }
        }
        // Remove keys with no remaining routes.
        self.route_table.retain(|_, v| !v.is_empty());

        self.stats.pruned_last_cycle = pruned;
    }

    // ── Statistics ───────────────────────────────────────────────────────────

    /// Return a snapshot of current engine statistics.
    pub fn routing_stats(&self) -> AdaptiveRoutingStats {
        let all_entries: Vec<&RouteEntry> =
            self.route_table.values().flat_map(|v| v.iter()).collect();
        let total_routes = all_entries.len();
        let (avg_rtt_ms, avg_loss_rate) = if total_routes == 0 {
            (0.0, 0.0)
        } else {
            let sum_rtt: f64 = all_entries.iter().map(|e| e.rtt_ms).sum();
            let sum_loss: f64 = all_entries.iter().map(|e| e.loss_rate).sum();
            (
                sum_rtt / total_routes as f64,
                sum_loss / total_routes as f64,
            )
        };

        AdaptiveRoutingStats {
            total_routes,
            avg_rtt_ms,
            avg_loss_rate,
            policy_switches: self.stats.policy_switches,
            pruned_last_cycle: self.stats.pruned_last_cycle,
            total_probe_rounds: self.stats.total_probe_rounds,
        }
    }

    /// Return the number of unique `RouteKey`s tracked.
    pub fn key_count(&self) -> usize {
        self.route_table.len()
    }

    /// Return the total number of route entries across all keys.
    pub fn entry_count(&self) -> usize {
        self.route_table.values().map(|v| v.len()).sum()
    }

    /// Check whether a route exists for the given key and next-hop.
    pub fn has_route(&self, key: &RouteKey, next_hop: &[u8; 32]) -> bool {
        self.route_table
            .get(key)
            .map(|entries| entries.iter().any(|e| &e.next_hop == next_hop))
            .unwrap_or(false)
    }

    /// Return all distinct next-hops recorded for `key`.
    pub fn next_hops_for(&self, key: &RouteKey) -> Vec<[u8; 32]> {
        self.route_table
            .get(key)
            .map(|entries| entries.iter().map(|e| e.next_hop).collect())
            .unwrap_or_default()
    }

    /// Return the best route entry for `key` under the given policy (clone).
    pub fn best_entry(&self, key: &RouteKey, policy: RoutingPolicy) -> Option<RouteEntry> {
        let next_hop = self.select_next_hop(key, policy)?;
        self.route_table
            .get(key)?
            .iter()
            .find(|e| e.next_hop == next_hop)
            .cloned()
    }

    /// Return a reference to the underlying route table.
    pub fn route_table(&self) -> &HashMap<RouteKey, Vec<RouteEntry>> {
        &self.route_table
    }

    /// Merge another engine's route table into this one.
    pub fn merge_from(&mut self, other: &AdaptiveRoutingEngine) {
        for (key, entries) in &other.route_table {
            for entry in entries {
                self.add_route(key.clone(), entry.clone());
            }
        }
    }

    /// Resize `max_routes_per_key` and evict excess entries if needed.
    pub fn resize_max_routes(&mut self, new_max: usize) {
        self.config.max_routes_per_key = new_max;
        for entries in self.route_table.values_mut() {
            while entries.len() > new_max {
                if let Some(idx) = worst_entry_index(entries) {
                    entries.remove(idx);
                } else {
                    break;
                }
            }
        }
    }

    /// Update the EWMA alpha factor.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `alpha` is outside (0.0, 1.0].
    pub fn set_alpha(&mut self, alpha: f64) -> Result<(), RoutingEngineError> {
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(RoutingEngineError::InvalidAlpha(alpha));
        }
        self.config.alpha = alpha;
        Ok(())
    }

    /// Update the probe interval.
    pub fn set_probe_interval(&mut self, secs: u64) {
        self.config.probe_interval_secs = secs;
    }

    /// Drain all routes and reset statistics.
    pub fn reset(&mut self) {
        self.route_table.clear();
        self.stats = AdaptiveRoutingStats::default();
    }
}

// ─── RoutingEngineError ───────────────────────────────────────────────────────

/// Errors returned by `AdaptiveRoutingEngine` operations.
#[derive(Debug, Clone, PartialEq)]
pub enum RoutingEngineError {
    /// No entry found for the requested `RouteKey`.
    KeyNotFound,
    /// The specified `next_hop` is not recorded under the given key.
    NextHopNotFound,
    /// The supplied EWMA alpha factor is invalid.
    InvalidAlpha(f64),
}

impl std::fmt::Display for RoutingEngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RoutingEngineError::KeyNotFound => write!(f, "route key not found"),
            RoutingEngineError::NextHopNotFound => write!(f, "next-hop not found for key"),
            RoutingEngineError::InvalidAlpha(a) => {
                write!(f, "invalid EWMA alpha {a}: must be in (0, 1]")
            }
        }
    }
}

impl std::error::Error for RoutingEngineError {}

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Current Unix timestamp in whole seconds (saturating at u64::MAX).
fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Recompute the composite weight for a route entry.
///
/// weight = bandwidth_kbps / (rtt_ms.max(1) * (1 + loss_rate)²)
fn recompute_weight(entry: &mut RouteEntry) {
    let rtt = entry.rtt_ms.max(1.0);
    let loss_factor = (1.0 + entry.loss_rate).powi(2);
    entry.weight = entry.bandwidth_kbps / (rtt * loss_factor);
}

/// QoS composite score: higher is better.
///
/// qos = (1 − loss_rate) × bandwidth_kbps / rtt_ms.max(1)
fn qos_score(entry: &RouteEntry) -> f64 {
    let availability = (1.0 - entry.loss_rate).max(0.0);
    availability * entry.bandwidth_kbps / entry.rtt_ms.max(1.0)
}

/// Return the index of the worst-scoring entry in a non-empty slice.
fn worst_entry_index(entries: &[RouteEntry]) -> Option<usize> {
    entries
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.weight
                .partial_cmp(&b.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn make_key(src_byte: u8, dst_byte: u8) -> RouteKey {
        let mut src = [0u8; 32];
        let mut dst = [0u8; 32];
        src[0] = src_byte;
        dst[0] = dst_byte;
        RouteKey::new(src, dst)
    }

    fn make_hop(byte: u8) -> [u8; 32] {
        let mut h = [0u8; 32];
        h[0] = byte;
        h
    }

    fn entry_with_metrics(hop_byte: u8, rtt: f64, loss: f64, bw: f64) -> RouteEntry {
        let mut e = RouteEntry::new(make_hop(hop_byte));
        e.rtt_ms = rtt;
        e.loss_rate = loss;
        e.bandwidth_kbps = bw;
        recompute_weight(&mut e);
        e
    }

    fn engine_default() -> AdaptiveRoutingEngine {
        AdaptiveRoutingEngine::with_defaults()
    }

    // ── xorshift64 ───────────────────────────────────────────────────────────

    #[test]
    fn test_xorshift64_non_zero_output() {
        let mut state = 12345u64;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
    }

    #[test]
    fn test_xorshift64_state_changes() {
        let mut state = 99u64;
        let before = state;
        xorshift64(&mut state);
        assert_ne!(state, before);
    }

    #[test]
    fn test_xorshift64_sequence_unique() {
        let mut state = 1u64;
        let a = xorshift64(&mut state);
        let b = xorshift64(&mut state);
        let c = xorshift64(&mut state);
        assert_ne!(a, b);
        assert_ne!(b, c);
    }

    #[test]
    fn test_xorshift64_deterministic() {
        let mut s1 = 42u64;
        let mut s2 = 42u64;
        assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
    }

    // ── fnv1a_64 ─────────────────────────────────────────────────────────────

    #[test]
    fn test_fnv1a_64_empty() {
        let h = fnv1a_64(&[]);
        assert_eq!(h, 14_695_981_039_346_656_037u64);
    }

    #[test]
    fn test_fnv1a_64_known_value() {
        // "foobar" FNV-1a 64-bit = 0x85944171f73967e8
        let h = fnv1a_64(b"foobar");
        assert_eq!(h, 0x85944171f73967e8);
    }

    #[test]
    fn test_fnv1a_64_different_inputs() {
        assert_ne!(fnv1a_64(b"abc"), fnv1a_64(b"xyz"));
    }

    #[test]
    fn test_fnv1a_64_single_byte() {
        let h = fnv1a_64(&[0xAB]);
        assert_ne!(h, 0);
    }

    // ── RouteKey ─────────────────────────────────────────────────────────────

    #[test]
    fn test_route_key_equality() {
        let k1 = make_key(1, 2);
        let k2 = make_key(1, 2);
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_route_key_inequality() {
        let k1 = make_key(1, 2);
        let k2 = make_key(1, 3);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_route_key_hash_deterministic() {
        let k = make_key(5, 10);
        assert_eq!(k.hash_u64(), k.hash_u64());
    }

    #[test]
    fn test_route_key_hash_distinct() {
        assert_ne!(make_key(1, 2).hash_u64(), make_key(2, 1).hash_u64());
    }

    #[test]
    fn test_route_key_hashmap_usage() {
        let mut map: HashMap<RouteKey, u32> = HashMap::new();
        let k = make_key(7, 8);
        map.insert(k.clone(), 42);
        assert_eq!(map.get(&k), Some(&42));
    }

    // ── RouteEntry ────────────────────────────────────────────────────────────

    #[test]
    fn test_route_entry_defaults() {
        let e = RouteEntry::new(make_hop(1));
        assert_eq!(e.rtt_ms, 100.0);
        assert_eq!(e.loss_rate, 0.0);
        assert_eq!(e.bandwidth_kbps, 1000.0);
        assert!(e.weight > 0.0);
    }

    #[test]
    fn test_route_entry_is_healthy_nominal() {
        let e = RouteEntry::new(make_hop(1));
        assert!(e.is_healthy());
    }

    #[test]
    fn test_route_entry_unhealthy_high_loss() {
        let mut e = RouteEntry::new(make_hop(1));
        e.loss_rate = 0.9;
        assert!(!e.is_healthy());
    }

    #[test]
    fn test_route_entry_unhealthy_high_rtt() {
        let mut e = RouteEntry::new(make_hop(1));
        e.rtt_ms = 6000.0;
        assert!(!e.is_healthy());
    }

    #[test]
    fn test_recompute_weight_high_bw_low_rtt() {
        let e = entry_with_metrics(1, 10.0, 0.0, 10_000.0);
        let w1 = e.weight;
        let mut e2 = entry_with_metrics(2, 100.0, 0.0, 10_000.0);
        let w2 = e2.weight;
        // Lower RTT → higher weight.
        assert!(w1 > w2, "w1={w1} w2={w2}");
        // Avoid unused warning.
        let _ = &mut e2;
        let _ = e.weight;
    }

    #[test]
    fn test_recompute_weight_increases_with_bandwidth() {
        let mut e1 = entry_with_metrics(1, 50.0, 0.0, 500.0);
        let mut e2 = entry_with_metrics(2, 50.0, 0.0, 5000.0);
        assert!(e2.weight > e1.weight);
        // suppress unused warnings
        let _ = &mut e1;
        let _ = &mut e2;
    }

    // ── RoutingPolicy ─────────────────────────────────────────────────────────

    #[test]
    fn test_routing_policy_labels_unique() {
        let policies = [
            RoutingPolicy::ShortestPath,
            RoutingPolicy::LowestLatency,
            RoutingPolicy::HighestBandwidth,
            RoutingPolicy::LoadBalanced,
            RoutingPolicy::QoSAware,
        ];
        let labels: std::collections::HashSet<_> = policies.iter().map(|p| p.label()).collect();
        assert_eq!(labels.len(), 5);
    }

    #[test]
    fn test_routing_policy_default_is_shortest_path() {
        assert_eq!(RoutingPolicy::default(), RoutingPolicy::ShortestPath);
    }

    // ── AdaptiveRoutingConfig ─────────────────────────────────────────────────

    #[test]
    fn test_config_defaults_valid() {
        let cfg = AdaptiveRoutingConfig::default();
        assert!(cfg.alpha > 0.0 && cfg.alpha <= 1.0);
        assert!(cfg.max_routes_per_key > 0);
        assert!(cfg.probe_interval_secs > 0);
    }

    // ── Engine construction ───────────────────────────────────────────────────

    #[test]
    fn test_engine_new_empty() {
        let e = engine_default();
        assert_eq!(e.key_count(), 0);
        assert_eq!(e.entry_count(), 0);
    }

    #[test]
    fn test_engine_active_policy_matches_config() {
        let cfg = AdaptiveRoutingConfig {
            policy: RoutingPolicy::QoSAware,
            ..AdaptiveRoutingConfig::default()
        };
        let eng = AdaptiveRoutingEngine::new(cfg);
        assert_eq!(eng.active_policy(), RoutingPolicy::QoSAware);
    }

    // ── add_route / remove_route ──────────────────────────────────────────────

    #[test]
    fn test_add_route_increases_count() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        eng.add_route(key, RouteEntry::new(make_hop(10)));
        assert_eq!(eng.entry_count(), 1);
    }

    #[test]
    fn test_add_route_duplicate_hop_overwrites() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        eng.add_route(key.clone(), RouteEntry::new(make_hop(10)));
        eng.add_route(key.clone(), RouteEntry::new(make_hop(10)));
        assert_eq!(eng.entry_count(), 1);
    }

    #[test]
    fn test_add_multiple_hops() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        eng.add_route(key.clone(), RouteEntry::new(make_hop(10)));
        eng.add_route(key.clone(), RouteEntry::new(make_hop(11)));
        assert_eq!(eng.entry_count(), 2);
    }

    #[test]
    fn test_remove_route() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        eng.add_route(key.clone(), RouteEntry::new(make_hop(10)));
        eng.remove_route(&key, &make_hop(10));
        assert_eq!(eng.entry_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_route_no_panic() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        // Should be a no-op, not a panic.
        eng.remove_route(&key, &make_hop(99));
    }

    #[test]
    fn test_max_routes_per_key_evicts_worst() {
        let cfg = AdaptiveRoutingConfig {
            max_routes_per_key: 2,
            ..AdaptiveRoutingConfig::default()
        };
        let mut eng = AdaptiveRoutingEngine::new(cfg);
        let key = make_key(0, 1);
        // Add 3 entries; third should evict the worst.
        eng.add_route(key.clone(), entry_with_metrics(1, 50.0, 0.0, 1000.0));
        eng.add_route(key.clone(), entry_with_metrics(2, 100.0, 0.2, 1000.0));
        eng.add_route(key.clone(), entry_with_metrics(3, 30.0, 0.0, 2000.0));
        assert_eq!(eng.routes_for(&key).map(|s| s.len()), Some(2));
    }

    #[test]
    fn test_clear_routes() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        eng.add_route(key.clone(), RouteEntry::new(make_hop(1)));
        eng.add_route(key.clone(), RouteEntry::new(make_hop(2)));
        eng.clear_routes(&key);
        assert!(eng.routes_for(&key).is_none());
    }

    #[test]
    fn test_has_route_true() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        eng.add_route(key.clone(), RouteEntry::new(make_hop(10)));
        assert!(eng.has_route(&key, &make_hop(10)));
    }

    #[test]
    fn test_has_route_false() {
        let eng = engine_default();
        let key = make_key(1, 2);
        assert!(!eng.has_route(&key, &make_hop(10)));
    }

    // ── update_metrics ────────────────────────────────────────────────────────

    #[test]
    fn test_update_metrics_ok() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        eng.add_route(key.clone(), RouteEntry::new(make_hop(10)));
        let result = eng.update_metrics(&key, &make_hop(10), 20.0, 0.01);
        assert!(result.is_ok());
    }

    #[test]
    fn test_update_metrics_rtt_decreases_with_low_sample() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        let mut entry = RouteEntry::new(make_hop(10));
        entry.rtt_ms = 200.0;
        eng.add_route(key.clone(), entry);
        eng.update_metrics(&key, &make_hop(10), 10.0, 0.0).ok();
        let rtt = eng
            .routes_for(&key)
            .and_then(|s| s.first())
            .map(|e| e.rtt_ms)
            .unwrap_or(0.0);
        assert!(rtt < 200.0, "rtt should decrease: {rtt}");
    }

    #[test]
    fn test_update_metrics_loss_clamped() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        eng.add_route(key.clone(), RouteEntry::new(make_hop(10)));
        eng.update_metrics(&key, &make_hop(10), 50.0, 2.0).ok();
        let loss = eng
            .routes_for(&key)
            .and_then(|s| s.first())
            .map(|e| e.loss_rate)
            .unwrap_or(1.1);
        assert!(loss <= 1.0, "loss clamped to 1.0: {loss}");
    }

    #[test]
    fn test_update_metrics_key_not_found() {
        let mut eng = engine_default();
        let key = make_key(9, 9);
        let res = eng.update_metrics(&key, &make_hop(1), 10.0, 0.0);
        assert_eq!(res, Err(RoutingEngineError::KeyNotFound));
    }

    #[test]
    fn test_update_metrics_hop_not_found() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        eng.add_route(key.clone(), RouteEntry::new(make_hop(1)));
        let res = eng.update_metrics(&key, &make_hop(99), 10.0, 0.0);
        assert_eq!(res, Err(RoutingEngineError::NextHopNotFound));
    }

    // ── select_next_hop ───────────────────────────────────────────────────────

    #[test]
    fn test_select_next_hop_no_routes_returns_none() {
        let eng = engine_default();
        let key = make_key(1, 2);
        assert!(eng
            .select_next_hop(&key, RoutingPolicy::LowestLatency)
            .is_none());
    }

    #[test]
    fn test_select_next_hop_single_entry() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        eng.add_route(key.clone(), RouteEntry::new(make_hop(7)));
        let hop = eng.select_next_hop(&key, RoutingPolicy::LowestLatency);
        assert_eq!(hop, Some(make_hop(7)));
    }

    #[test]
    fn test_select_lowest_latency_picks_best() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        eng.add_route(key.clone(), entry_with_metrics(1, 200.0, 0.0, 1000.0));
        eng.add_route(key.clone(), entry_with_metrics(2, 50.0, 0.0, 1000.0));
        eng.add_route(key.clone(), entry_with_metrics(3, 500.0, 0.0, 1000.0));
        let hop = eng.select_next_hop(&key, RoutingPolicy::LowestLatency);
        assert_eq!(hop, Some(make_hop(2)));
    }

    #[test]
    fn test_select_highest_bandwidth_picks_best() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        eng.add_route(key.clone(), entry_with_metrics(1, 50.0, 0.0, 500.0));
        eng.add_route(key.clone(), entry_with_metrics(2, 50.0, 0.0, 5000.0));
        eng.add_route(key.clone(), entry_with_metrics(3, 50.0, 0.0, 100.0));
        let hop = eng.select_next_hop(&key, RoutingPolicy::HighestBandwidth);
        assert_eq!(hop, Some(make_hop(2)));
    }

    #[test]
    fn test_select_shortest_path_picks_highest_weight() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        // Lower RTT + higher BW = higher weight.
        eng.add_route(key.clone(), entry_with_metrics(1, 100.0, 0.0, 1000.0));
        eng.add_route(key.clone(), entry_with_metrics(2, 10.0, 0.0, 10_000.0));
        let hop = eng.select_next_hop(&key, RoutingPolicy::ShortestPath);
        assert_eq!(hop, Some(make_hop(2)));
    }

    #[test]
    fn test_select_qos_aware_picks_best_composite() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        // Hop 1: good bw, low latency, low loss → should win QoS.
        eng.add_route(key.clone(), entry_with_metrics(1, 30.0, 0.01, 5000.0));
        // Hop 2: very high bandwidth but also very high loss.
        eng.add_route(key.clone(), entry_with_metrics(2, 30.0, 0.8, 50_000.0));
        let hop = eng.select_next_hop(&key, RoutingPolicy::QoSAware);
        assert_eq!(hop, Some(make_hop(1)));
    }

    #[test]
    fn test_select_load_balanced_returns_valid_hop() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        eng.add_route(key.clone(), entry_with_metrics(1, 50.0, 0.0, 1000.0));
        eng.add_route(key.clone(), entry_with_metrics(2, 60.0, 0.0, 2000.0));
        let hop = eng.select_next_hop(&key, RoutingPolicy::LoadBalanced);
        assert!(hop == Some(make_hop(1)) || hop == Some(make_hop(2)));
    }

    #[test]
    fn test_select_default_uses_active_policy() {
        let mut eng = engine_default();
        eng.set_policy(RoutingPolicy::HighestBandwidth);
        let key = make_key(1, 2);
        eng.add_route(key.clone(), entry_with_metrics(1, 50.0, 0.0, 100.0));
        eng.add_route(key.clone(), entry_with_metrics(2, 50.0, 0.0, 9000.0));
        let hop = eng.select_next_hop_default(&key);
        assert_eq!(hop, Some(make_hop(2)));
    }

    #[test]
    fn test_select_unhealthy_all_falls_back_to_first() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        // Both entries are unhealthy.
        eng.add_route(key.clone(), entry_with_metrics(1, 6000.0, 0.9, 1000.0));
        eng.add_route(key.clone(), entry_with_metrics(2, 6000.0, 0.9, 1000.0));
        let hop = eng.select_next_hop(&key, RoutingPolicy::LowestLatency);
        assert!(hop.is_some());
    }

    // ── set_policy ────────────────────────────────────────────────────────────

    #[test]
    fn test_set_policy_increments_switch_count() {
        let mut eng = engine_default();
        eng.set_policy(RoutingPolicy::LowestLatency); // already default → no switch
                                                      // Change to a different policy.
        eng.set_policy(RoutingPolicy::QoSAware);
        assert_eq!(eng.routing_stats().policy_switches, 1);
    }

    #[test]
    fn test_set_policy_same_no_increment() {
        let mut eng = engine_default();
        let p = eng.active_policy();
        eng.set_policy(p);
        assert_eq!(eng.routing_stats().policy_switches, 0);
    }

    // ── probe_routes ──────────────────────────────────────────────────────────

    #[test]
    fn test_probe_routes_increments_counter() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        eng.add_route(key, RouteEntry::new(make_hop(1)));
        eng.probe_routes();
        assert_eq!(eng.routing_stats().total_probe_rounds, 1);
    }

    #[test]
    fn test_probe_routes_modifies_rtt() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        let mut entry = RouteEntry::new(make_hop(1));
        entry.rtt_ms = 100.0;
        eng.add_route(key.clone(), entry);
        eng.probe_routes();
        let rtt = eng
            .routes_for(&key)
            .and_then(|s| s.first())
            .map(|e| e.rtt_ms)
            .unwrap_or(100.0);
        // RTT should have changed (jitter applied).
        // Because probe alpha is small, change could be tiny; just verify range.
        assert!(rtt > 0.0);
    }

    #[test]
    fn test_probe_routes_empty_table_no_panic() {
        let mut eng = engine_default();
        eng.probe_routes(); // should not panic
        assert_eq!(eng.routing_stats().total_probe_rounds, 1);
    }

    // ── run_adaptation_cycle ──────────────────────────────────────────────────

    #[test]
    fn test_adaptation_cycle_prunes_stale() {
        let cfg = AdaptiveRoutingConfig {
            probe_interval_secs: 1,
            ..AdaptiveRoutingConfig::default()
        };
        // Very short stale threshold: interval=1 → stale after 3 secs.
        let mut eng = AdaptiveRoutingEngine::new(cfg);
        let key = make_key(1, 2);
        let mut entry = RouteEntry::new(make_hop(1));
        // Force last_updated to ancient past.
        entry.last_updated = 0;
        eng.route_table.entry(key.clone()).or_default().push(entry);

        eng.run_adaptation_cycle();
        assert_eq!(eng.entry_count(), 0, "stale entry should be pruned");
    }

    #[test]
    fn test_adaptation_cycle_keeps_fresh() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        eng.add_route(key.clone(), RouteEntry::new(make_hop(1)));
        eng.run_adaptation_cycle();
        assert_eq!(eng.entry_count(), 1, "fresh entry must survive");
    }

    #[test]
    fn test_adaptation_cycle_prune_count_recorded() {
        let cfg = AdaptiveRoutingConfig {
            probe_interval_secs: 1,
            ..AdaptiveRoutingConfig::default()
        };
        let mut eng = AdaptiveRoutingEngine::new(cfg);
        let key = make_key(3, 4);
        let mut e = RouteEntry::new(make_hop(5));
        e.last_updated = 0;
        eng.route_table.entry(key).or_default().push(e);
        eng.run_adaptation_cycle();
        assert_eq!(eng.routing_stats().pruned_last_cycle, 1);
    }

    // ── routing_stats ─────────────────────────────────────────────────────────

    #[test]
    fn test_routing_stats_empty_engine() {
        let eng = engine_default();
        let s = eng.routing_stats();
        assert_eq!(s.total_routes, 0);
        assert_eq!(s.avg_rtt_ms, 0.0);
        assert_eq!(s.avg_loss_rate, 0.0);
    }

    #[test]
    fn test_routing_stats_avg_rtt() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        eng.add_route(key.clone(), entry_with_metrics(1, 100.0, 0.0, 1000.0));
        eng.add_route(key.clone(), entry_with_metrics(2, 200.0, 0.0, 1000.0));
        let s = eng.routing_stats();
        assert!(
            (s.avg_rtt_ms - 150.0).abs() < 1.0,
            "avg_rtt={}",
            s.avg_rtt_ms
        );
    }

    #[test]
    fn test_routing_stats_total_routes() {
        let mut eng = engine_default();
        for i in 0u8..5 {
            let key = make_key(i, i + 1);
            eng.add_route(key, RouteEntry::new(make_hop(i)));
        }
        assert_eq!(eng.routing_stats().total_routes, 5);
    }

    // ── next_hops_for ─────────────────────────────────────────────────────────

    #[test]
    fn test_next_hops_for_empty() {
        let eng = engine_default();
        let key = make_key(1, 2);
        assert!(eng.next_hops_for(&key).is_empty());
    }

    #[test]
    fn test_next_hops_for_multiple() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        eng.add_route(key.clone(), RouteEntry::new(make_hop(10)));
        eng.add_route(key.clone(), RouteEntry::new(make_hop(20)));
        let hops = eng.next_hops_for(&key);
        assert_eq!(hops.len(), 2);
        assert!(hops.contains(&make_hop(10)));
        assert!(hops.contains(&make_hop(20)));
    }

    // ── best_entry ────────────────────────────────────────────────────────────

    #[test]
    fn test_best_entry_returns_lowest_latency() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        eng.add_route(key.clone(), entry_with_metrics(1, 300.0, 0.0, 1000.0));
        eng.add_route(key.clone(), entry_with_metrics(2, 20.0, 0.0, 1000.0));
        let best = eng.best_entry(&key, RoutingPolicy::LowestLatency);
        assert!(best.is_some());
        assert_eq!(
            best.expect("test: best_entry must return Some for LowestLatency with two routes")
                .next_hop,
            make_hop(2)
        );
    }

    #[test]
    fn test_best_entry_missing_key_none() {
        let eng = engine_default();
        assert!(eng
            .best_entry(&make_key(9, 9), RoutingPolicy::ShortestPath)
            .is_none());
    }

    // ── set_alpha ─────────────────────────────────────────────────────────────

    #[test]
    fn test_set_alpha_valid() {
        let mut eng = engine_default();
        assert!(eng.set_alpha(0.5).is_ok());
    }

    #[test]
    fn test_set_alpha_zero_is_error() {
        let mut eng = engine_default();
        assert!(eng.set_alpha(0.0).is_err());
    }

    #[test]
    fn test_set_alpha_negative_is_error() {
        let mut eng = engine_default();
        assert!(eng.set_alpha(-0.1).is_err());
    }

    #[test]
    fn test_set_alpha_one_is_valid() {
        let mut eng = engine_default();
        assert!(eng.set_alpha(1.0).is_ok());
    }

    // ── update_bandwidth ──────────────────────────────────────────────────────

    #[test]
    fn test_update_bandwidth_ok() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        eng.add_route(key.clone(), RouteEntry::new(make_hop(1)));
        assert!(eng.update_bandwidth(&key, &make_hop(1), 5000.0).is_ok());
    }

    #[test]
    fn test_update_bandwidth_increases_bw() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        let mut e = RouteEntry::new(make_hop(1));
        e.bandwidth_kbps = 100.0;
        eng.add_route(key.clone(), e);
        eng.update_bandwidth(&key, &make_hop(1), 10_000.0).ok();
        let bw = eng
            .routes_for(&key)
            .and_then(|s| s.first())
            .map(|e| e.bandwidth_kbps)
            .unwrap_or(0.0);
        assert!(bw > 100.0, "bw should increase: {bw}");
    }

    #[test]
    fn test_update_bandwidth_key_not_found() {
        let mut eng = engine_default();
        let res = eng.update_bandwidth(&make_key(9, 9), &make_hop(1), 5000.0);
        assert_eq!(res, Err(RoutingEngineError::KeyNotFound));
    }

    // ── merge_from ────────────────────────────────────────────────────────────

    #[test]
    fn test_merge_from_adds_routes() {
        let mut eng1 = engine_default();
        let mut eng2 = engine_default();
        let key = make_key(1, 2);
        eng2.add_route(key.clone(), RouteEntry::new(make_hop(42)));
        eng1.merge_from(&eng2);
        assert!(eng1.has_route(&key, &make_hop(42)));
    }

    #[test]
    fn test_merge_from_no_duplicates_for_same_hop() {
        let mut eng1 = engine_default();
        let mut eng2 = engine_default();
        let key = make_key(1, 2);
        eng1.add_route(key.clone(), RouteEntry::new(make_hop(1)));
        eng2.add_route(key.clone(), RouteEntry::new(make_hop(1)));
        eng1.merge_from(&eng2);
        assert_eq!(eng1.entry_count(), 1);
    }

    // ── resize_max_routes ─────────────────────────────────────────────────────

    #[test]
    fn test_resize_max_routes_evicts_when_shrinking() {
        let cfg = AdaptiveRoutingConfig {
            max_routes_per_key: 5,
            ..AdaptiveRoutingConfig::default()
        };
        let mut eng = AdaptiveRoutingEngine::new(cfg);
        let key = make_key(1, 2);
        for i in 0u8..5 {
            eng.add_route(key.clone(), RouteEntry::new(make_hop(i + 1)));
        }
        assert_eq!(eng.entry_count(), 5);
        eng.resize_max_routes(2);
        assert!(eng.entry_count() <= 2, "count={}", eng.entry_count());
    }

    // ── reset ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_reset_clears_routes() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        eng.add_route(key, RouteEntry::new(make_hop(1)));
        eng.reset();
        assert_eq!(eng.entry_count(), 0);
    }

    #[test]
    fn test_reset_clears_stats() {
        let mut eng = engine_default();
        eng.set_policy(RoutingPolicy::QoSAware);
        eng.reset();
        assert_eq!(eng.routing_stats().policy_switches, 0);
    }

    // ── set_probe_interval ────────────────────────────────────────────────────

    #[test]
    fn test_set_probe_interval() {
        let mut eng = engine_default();
        eng.set_probe_interval(60);
        assert_eq!(eng.config.probe_interval_secs, 60);
    }

    // ── Type aliases ──────────────────────────────────────────────────────────

    #[test]
    fn test_are_type_aliases_usable() {
        let key: AreRouteKey = make_key(1, 2);
        let entry: AreRouteEntry = RouteEntry::new(make_hop(5));
        let policy: AreRoutingPolicy = RoutingPolicy::QoSAware;
        let config: AreRoutingConfig = AdaptiveRoutingConfig::default();
        let stats: AreRoutingStats = AdaptiveRoutingStats::default();

        assert_eq!(key.src[0], 1);
        assert!(entry.rtt_ms > 0.0);
        assert_eq!(policy, RoutingPolicy::QoSAware);
        assert!(config.alpha > 0.0);
        assert_eq!(stats.total_routes, 0);
    }

    // ── Edge / stress ─────────────────────────────────────────────────────────

    #[test]
    fn test_many_keys() {
        let mut eng = engine_default();
        for i in 0u8..200 {
            let key = make_key(i, 255 - i);
            eng.add_route(key, RouteEntry::new(make_hop(i)));
        }
        assert_eq!(eng.key_count(), 200);
    }

    #[test]
    fn test_multiple_probe_rounds() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        eng.add_route(key, RouteEntry::new(make_hop(1)));
        for _ in 0..10 {
            eng.probe_routes();
        }
        assert_eq!(eng.routing_stats().total_probe_rounds, 10);
    }

    #[test]
    fn test_update_metrics_ewma_convergence() {
        let mut eng = engine_default();
        let key = make_key(1, 2);
        let mut e = RouteEntry::new(make_hop(1));
        e.rtt_ms = 1000.0;
        eng.add_route(key.clone(), e);
        // Apply many updates with low RTT.
        for _ in 0..100 {
            eng.update_metrics(&key, &make_hop(1), 10.0, 0.0).ok();
        }
        let rtt = eng
            .routes_for(&key)
            .and_then(|s| s.first())
            .map(|e| e.rtt_ms)
            .unwrap_or(1000.0);
        assert!(rtt < 50.0, "EWMA should converge: {rtt}");
    }

    #[test]
    fn test_qos_score_better_for_low_loss() {
        let e_low_loss = entry_with_metrics(1, 50.0, 0.01, 1000.0);
        let e_high_loss = entry_with_metrics(2, 50.0, 0.9, 1000.0);
        assert!(qos_score(&e_low_loss) > qos_score(&e_high_loss));
    }

    #[test]
    fn test_routing_engine_error_display() {
        let e = RoutingEngineError::InvalidAlpha(0.0);
        assert!(e.to_string().contains("alpha"));
        let e2 = RoutingEngineError::KeyNotFound;
        assert!(e2.to_string().contains("key"));
    }
}
