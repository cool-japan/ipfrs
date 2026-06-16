//! Adaptive Peer Scheduler
//!
//! A dynamic peer request scheduler that adapts request rates based on peer
//! performance, network conditions, and backpressure signals. The scheduler
//! continuously recomputes per-peer request allocations by weighting together
//! success rate and average latency, then applies a global backpressure factor
//! to prevent overloading the local node or the network.
//!
//! # Design
//!
//! * Each registered peer has a [`PeerMetrics`] record that accumulates success
//!   counts, failure counts and cumulative latency.
//! * A [`ScheduleSlot`] is derived from those metrics and represents the number
//!   of concurrent/next requests the scheduler is willing to dispatch to that
//!   peer in one scheduling epoch.
//! * [`BackpressureSignal`] allows callers to throttle the whole scheduler when
//!   the network or the application layer is under pressure.
//! * [`AdaptivePeerScheduler::recompute_schedule`] must be called periodically
//!   (or after significant metric changes) to refresh the slots.

use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// A single slot in the scheduling table for one peer.
#[derive(Debug, Clone, PartialEq)]
pub struct ScheduleSlot {
    /// The peer this slot belongs to.
    pub peer_id: String,
    /// Number of requests that may be dispatched to this peer.
    pub allocated_requests: u32,
    /// Computed weight `[0.01, 1.0]` used to derive the allocation.
    pub weight: f64,
    /// Unix-epoch timestamp (ms) when this slot was last recomputed.
    pub last_updated: u64,
}

/// Global backpressure signal reported by the application or transport layer.
#[derive(Debug, Clone, PartialEq)]
pub enum BackpressureSignal {
    /// No backpressure — operate at full rate.
    None,
    /// Mild congestion — reduce rates by the given factor `(0, 1]`.
    Mild {
        /// Multiplicative factor applied to `base_requests_per_peer`.
        factor: f64,
    },
    /// Severe congestion — heavily reduce rates.
    Severe {
        /// Multiplicative factor applied to `base_requests_per_peer`.
        factor: f64,
    },
    /// System is overloaded — allocate zero requests to every peer.
    Overloaded,
}

impl BackpressureSignal {
    /// Returns the multiplicative factor for this signal.
    ///
    /// * `None`      → `1.0`
    /// * `Mild`      → the stored factor
    /// * `Severe`    → the stored factor
    /// * `Overloaded`→ `0.0`
    #[inline]
    pub fn factor(&self) -> f64 {
        match self {
            Self::None => 1.0,
            Self::Mild { factor } => *factor,
            Self::Severe { factor } => *factor,
            Self::Overloaded => 0.0,
        }
    }

    /// Human-readable label used in [`SchedulerStats`].
    pub fn label(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Mild { .. } => "mild",
            Self::Severe { .. } => "severe",
            Self::Overloaded => "overloaded",
        }
    }
}

/// Per-peer performance metrics accumulated over the lifetime of the peer
/// registration (or since last eviction).
#[derive(Debug, Clone, PartialEq)]
pub struct PeerMetrics {
    /// Unique peer identifier.
    pub peer_id: String,
    /// Number of successfully completed requests.
    pub success_count: u64,
    /// Number of failed requests.
    pub failure_count: u64,
    /// Sum of latencies (in ms) for all successful requests.
    pub total_latency_ms: u64,
    /// Unix-epoch timestamp (ms) of the last interaction with this peer.
    pub last_seen: u64,
    /// Number of failures in a row without any intervening success.
    pub consecutive_failures: u32,
}

impl PeerMetrics {
    /// Creates a fresh record for a peer, seeded with the given timestamp.
    pub fn new(peer_id: String, now: u64) -> Self {
        Self {
            peer_id,
            success_count: 0,
            failure_count: 0,
            total_latency_ms: 0,
            last_seen: now,
            consecutive_failures: 0,
        }
    }

    /// Returns the success rate in `[0.0, 1.0]`.
    ///
    /// Returns `1.0` if neither successes nor failures have been recorded yet
    /// (i.e. the peer is brand new and deserves optimistic treatment).
    pub fn success_rate(&self) -> f64 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            1.0
        } else {
            self.success_count as f64 / total as f64
        }
    }

    /// Returns the average request latency in milliseconds.
    ///
    /// Returns [`f64::MAX`] when no successful requests have been recorded,
    /// which causes the latency component of the weight to approach `0`.
    pub fn avg_latency_ms(&self) -> f64 {
        if self.success_count == 0 {
            f64::MAX
        } else {
            self.total_latency_ms as f64 / self.success_count as f64
        }
    }
}

/// Tuning knobs for the scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum number of peers tracked simultaneously.
    pub max_peers: usize,
    /// Baseline requests per peer used as the multiplier in weight → allocation.
    pub base_requests_per_peer: u32,
    /// Hard cap on allocated requests per peer per epoch.
    pub max_requests_per_peer: u32,
    /// Minimum number of requests a live peer always receives (unless Overloaded).
    pub min_requests_per_peer: u32,
    /// When the computed success rate of a peer drops below this value the peer
    /// is still kept but its weight is penalised via `failure_penalty`.
    pub backpressure_threshold: f64,
    /// Multiplicative penalty applied to a peer's weight when its success rate
    /// is below `backpressure_threshold`.
    pub failure_penalty: f64,
    /// Relative importance of latency in the weight formula.
    pub latency_weight: f64,
    /// Relative importance of success rate in the weight formula.
    pub success_weight: f64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_peers: 50,
            base_requests_per_peer: 10,
            max_requests_per_peer: 100,
            min_requests_per_peer: 1,
            backpressure_threshold: 0.8,
            failure_penalty: 0.5,
            latency_weight: 0.3,
            success_weight: 0.7,
        }
    }
}

/// Summary statistics snapshot emitted by [`AdaptivePeerScheduler::scheduler_stats`].
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    /// Number of peers currently registered.
    pub registered_peers: usize,
    /// Number of schedule slots with `allocated_requests > 0`.
    pub active_slots: usize,
    /// Total requests dispatched since construction.
    pub total_dispatched: u64,
    /// Total requests recorded as succeeded since construction.
    pub total_succeeded: u64,
    /// `total_succeeded / total_dispatched` (or `1.0` when `total_dispatched == 0`).
    pub success_rate: f64,
    /// Human-readable label of the current backpressure signal.
    pub backpressure: String,
}

// ──────────────────────────────────────────────────────────────────────────────
// AdaptivePeerScheduler
// ──────────────────────────────────────────────────────────────────────────────

/// A dynamic peer request scheduler that adapts request rates based on peer
/// performance, network conditions, and backpressure signals.
///
/// # Usage
///
/// ```rust
/// use ipfrs_network::{
///     ApsSchedulerConfig, ApsBackpressureSignal, AdaptivePeerScheduler,
/// };
///
/// let config = ApsSchedulerConfig::default();
/// let mut sched = AdaptivePeerScheduler::new(config);
///
/// sched.register_peer("peer-1".to_string(), 0);
/// sched.record_success("peer-1", 50, 100);
/// sched.recompute_schedule(200);
///
/// if let Some(peer) = sched.next_peer() {
///     println!("dispatch to {peer}");
/// }
/// ```
#[derive(Debug)]
pub struct AdaptivePeerScheduler {
    /// Scheduler configuration.
    pub config: SchedulerConfig,
    /// Per-peer performance metrics.
    pub peers: HashMap<String, PeerMetrics>,
    /// Last computed schedule.
    pub schedule: HashMap<String, ScheduleSlot>,
    /// Current global backpressure signal.
    pub global_backpressure: BackpressureSignal,
    /// Monotonically increasing counter of dispatched requests.
    pub total_requests_dispatched: u64,
    /// Monotonically increasing counter of succeeded requests.
    pub total_requests_succeeded: u64,
}

impl AdaptivePeerScheduler {
    // ── Construction ────────────────────────────────────────────────────────

    /// Creates a new scheduler with the supplied configuration.
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            peers: HashMap::new(),
            schedule: HashMap::new(),
            global_backpressure: BackpressureSignal::None,
            total_requests_dispatched: 0,
            total_requests_succeeded: 0,
        }
    }

    // ── Peer management ─────────────────────────────────────────────────────

    /// Registers a peer with the scheduler.
    ///
    /// If the peer is already known this is a no-op (existing metrics are
    /// preserved). When the peer table is full the peer with the oldest
    /// `last_seen` timestamp is evicted to make room.
    pub fn register_peer(&mut self, peer_id: String, now: u64) {
        if self.peers.contains_key(&peer_id) {
            return;
        }

        // Evict the stalest peer when at capacity.
        if self.peers.len() >= self.config.max_peers {
            if let Some(oldest_id) = self.oldest_peer_id() {
                self.peers.remove(&oldest_id);
                self.schedule.remove(&oldest_id);
            }
        }

        self.peers
            .insert(peer_id.clone(), PeerMetrics::new(peer_id, now));
    }

    /// Removes a peer from both the metrics table and the schedule.
    ///
    /// Returns `true` if the peer was present and has been removed.
    pub fn remove_peer(&mut self, peer_id: &str) -> bool {
        let removed = self.peers.remove(peer_id).is_some();
        self.schedule.remove(peer_id);
        removed
    }

    // ── Metric recording ────────────────────────────────────────────────────

    /// Records a successful request completion for the given peer.
    ///
    /// Also increments the global `total_requests_succeeded` counter.
    /// Silently ignores unknown peer IDs.
    pub fn record_success(&mut self, peer_id: &str, latency_ms: u64, now: u64) {
        if let Some(metrics) = self.peers.get_mut(peer_id) {
            metrics.success_count = metrics.success_count.saturating_add(1);
            metrics.total_latency_ms = metrics.total_latency_ms.saturating_add(latency_ms);
            metrics.last_seen = now;
            metrics.consecutive_failures = 0;
            self.total_requests_succeeded = self.total_requests_succeeded.saturating_add(1);
        }
    }

    /// Records a request failure for the given peer.
    ///
    /// Silently ignores unknown peer IDs.
    pub fn record_failure(&mut self, peer_id: &str, now: u64) {
        if let Some(metrics) = self.peers.get_mut(peer_id) {
            metrics.failure_count = metrics.failure_count.saturating_add(1);
            metrics.consecutive_failures = metrics.consecutive_failures.saturating_add(1);
            metrics.last_seen = now;
        }
    }

    // ── Backpressure ─────────────────────────────────────────────────────────

    /// Updates the global backpressure signal.
    ///
    /// The new signal takes effect on the *next* call to
    /// [`recompute_schedule`](Self::recompute_schedule).
    pub fn set_backpressure(&mut self, signal: BackpressureSignal) {
        self.global_backpressure = signal;
    }

    // ── Weight computation ───────────────────────────────────────────────────

    /// Computes the scheduling weight for a single peer.
    ///
    /// Formula:
    /// ```text
    /// weight = success_weight  * success_rate
    ///        + latency_weight  * (1 / (1 + avg_latency_ms / 1000))
    /// ```
    /// The result is clamped to `[0.01, 1.0]`.
    ///
    /// Additionally, if the peer's success rate is below
    /// `config.backpressure_threshold`, the weight is further multiplied by
    /// `config.failure_penalty`.
    pub fn compute_weight(&self, metrics: &PeerMetrics) -> f64 {
        let sr = metrics.success_rate();
        let avg_lat = metrics.avg_latency_ms();

        // Latency component: approaches 1 for very low latency, 0 for very high.
        let lat_component = if avg_lat == f64::MAX {
            0.0
        } else {
            1.0 / (1.0 + avg_lat / 1000.0)
        };

        let mut weight =
            self.config.success_weight * sr + self.config.latency_weight * lat_component;

        // Apply failure penalty when the peer is under-performing.
        if sr < self.config.backpressure_threshold {
            weight *= self.config.failure_penalty;
        }

        weight.clamp(0.01, 1.0)
    }

    // ── Schedule recomputation ───────────────────────────────────────────────

    /// Recomputes the entire schedule based on current peer metrics and the
    /// active backpressure signal.
    ///
    /// For each registered peer:
    /// 1. The weight is computed via [`compute_weight`](Self::compute_weight).
    /// 2. `allocated = clamp(round(weight × base × bp_factor), min, max)`.
    ///    When backpressure is [`BackpressureSignal::Overloaded`] the allocation
    ///    is forced to `0` regardless of the clamp range.
    /// 3. The [`ScheduleSlot`] in `self.schedule` is updated (or inserted).
    pub fn recompute_schedule(&mut self, now: u64) {
        let bp_factor = self.global_backpressure.factor();
        let base = self.config.base_requests_per_peer as f64;
        let min_req = self.config.min_requests_per_peer;
        let max_req = self.config.max_requests_per_peer;
        let overloaded = matches!(self.global_backpressure, BackpressureSignal::Overloaded);

        // Collect peer IDs + pre-computed weights to avoid borrow conflicts.
        let peer_data: Vec<(String, f64)> = self
            .peers
            .values()
            .map(|m| (m.peer_id.clone(), self.compute_weight(m)))
            .collect();

        for (peer_id, weight) in peer_data {
            let allocated = if overloaded {
                0u32
            } else {
                let raw = (weight * base * bp_factor).round() as u32;
                raw.clamp(min_req, max_req)
            };

            let slot = self
                .schedule
                .entry(peer_id.clone())
                .or_insert_with(|| ScheduleSlot {
                    peer_id: peer_id.clone(),
                    allocated_requests: 0,
                    weight: 0.0,
                    last_updated: now,
                });
            slot.allocated_requests = allocated;
            slot.weight = weight;
            slot.last_updated = now;
        }

        // Remove schedule slots whose peer has been evicted.
        self.schedule
            .retain(|pid, _| self.peers.contains_key(pid.as_str()));
    }

    // ── Dispatch helpers ─────────────────────────────────────────────────────

    /// Returns the peer ID with the highest `allocated_requests` in the current
    /// schedule (among peers with at least one allocated request).
    ///
    /// Returns `None` when every slot is at zero (or the schedule is empty).
    ///
    /// Also increments `total_requests_dispatched` when a peer is returned.
    pub fn next_peer(&self) -> Option<&str> {
        self.schedule
            .values()
            .filter(|s| s.allocated_requests > 0)
            .max_by_key(|s| s.allocated_requests)
            .map(|s| s.peer_id.as_str())
    }

    /// Increments the total dispatched counter. Callers should call this once
    /// they have committed to sending a request to the peer returned by
    /// [`next_peer`](Self::next_peer).
    pub fn mark_dispatched(&mut self) {
        self.total_requests_dispatched = self.total_requests_dispatched.saturating_add(1);
    }

    // ── Schedule inspection ──────────────────────────────────────────────────

    /// Returns a snapshot of the schedule sorted by `allocated_requests`
    /// descending.
    ///
    /// Each entry is `(peer_id, allocated_requests, weight)`.
    pub fn peek_schedule(&self) -> Vec<(&str, u32, f64)> {
        let mut entries: Vec<(&str, u32, f64)> = self
            .schedule
            .values()
            .map(|s| (s.peer_id.as_str(), s.allocated_requests, s.weight))
            .collect();
        entries.sort_by_key(|b| std::cmp::Reverse(b.1));
        entries
    }

    // ── Stale peer eviction ───────────────────────────────────────────────────

    /// Removes all peers that have not been seen within the last `max_idle_ms`
    /// milliseconds (i.e. where `now - last_seen > max_idle_ms`).
    pub fn evict_stale_peers(&mut self, now: u64, max_idle_ms: u64) {
        let stale: Vec<String> = self
            .peers
            .values()
            .filter(|m| now.saturating_sub(m.last_seen) > max_idle_ms)
            .map(|m| m.peer_id.clone())
            .collect();

        for pid in stale {
            self.peers.remove(&pid);
            self.schedule.remove(&pid);
        }
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    /// Returns a statistics snapshot.
    pub fn scheduler_stats(&self) -> SchedulerStats {
        let active_slots = self
            .schedule
            .values()
            .filter(|s| s.allocated_requests > 0)
            .count();

        let success_rate = if self.total_requests_dispatched == 0 {
            1.0
        } else {
            self.total_requests_succeeded as f64 / self.total_requests_dispatched as f64
        };

        SchedulerStats {
            registered_peers: self.peers.len(),
            active_slots,
            total_dispatched: self.total_requests_dispatched,
            total_succeeded: self.total_requests_succeeded,
            success_rate,
            backpressure: self.global_backpressure.label().to_string(),
        }
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Returns the ID of the peer with the smallest `last_seen` timestamp.
    fn oldest_peer_id(&self) -> Option<String> {
        self.peers
            .values()
            .min_by_key(|m| m.last_seen)
            .map(|m| m.peer_id.clone())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        AdaptivePeerScheduler, BackpressureSignal, PeerMetrics, ScheduleSlot, SchedulerConfig,
    };

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn default_scheduler() -> AdaptivePeerScheduler {
        AdaptivePeerScheduler::new(SchedulerConfig::default())
    }

    fn scheduler_with_peer(peer_id: &str) -> AdaptivePeerScheduler {
        let mut s = default_scheduler();
        s.register_peer(peer_id.to_string(), 0);
        s
    }

    // ── PeerMetrics unit tests ────────────────────────────────────────────────

    #[test]
    fn test_peer_metrics_success_rate_no_activity() {
        let m = PeerMetrics::new("p1".to_string(), 0);
        assert_eq!(m.success_rate(), 1.0, "brand-new peer should be 1.0");
    }

    #[test]
    fn test_peer_metrics_success_rate_all_success() {
        let mut m = PeerMetrics::new("p1".to_string(), 0);
        m.success_count = 5;
        assert_eq!(m.success_rate(), 1.0);
    }

    #[test]
    fn test_peer_metrics_success_rate_mixed() {
        let mut m = PeerMetrics::new("p1".to_string(), 0);
        m.success_count = 3;
        m.failure_count = 1;
        assert!((m.success_rate() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_peer_metrics_success_rate_all_failure() {
        let mut m = PeerMetrics::new("p1".to_string(), 0);
        m.failure_count = 10;
        assert_eq!(m.success_rate(), 0.0);
    }

    #[test]
    fn test_peer_metrics_avg_latency_no_success() {
        let m = PeerMetrics::new("p1".to_string(), 0);
        assert_eq!(m.avg_latency_ms(), f64::MAX);
    }

    #[test]
    fn test_peer_metrics_avg_latency_computed() {
        let mut m = PeerMetrics::new("p1".to_string(), 0);
        m.success_count = 4;
        m.total_latency_ms = 200;
        assert!((m.avg_latency_ms() - 50.0).abs() < 1e-9);
    }

    // ── SchedulerConfig defaults ──────────────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let c = SchedulerConfig::default();
        assert_eq!(c.max_peers, 50);
        assert_eq!(c.base_requests_per_peer, 10);
        assert_eq!(c.max_requests_per_peer, 100);
        assert_eq!(c.min_requests_per_peer, 1);
        assert!((c.backpressure_threshold - 0.8).abs() < 1e-9);
        assert!((c.failure_penalty - 0.5).abs() < 1e-9);
        assert!((c.latency_weight - 0.3).abs() < 1e-9);
        assert!((c.success_weight - 0.7).abs() < 1e-9);
    }

    // ── BackpressureSignal ────────────────────────────────────────────────────

    #[test]
    fn test_backpressure_none_factor() {
        assert_eq!(BackpressureSignal::None.factor(), 1.0);
    }

    #[test]
    fn test_backpressure_mild_factor() {
        let s = BackpressureSignal::Mild { factor: 0.6 };
        assert!((s.factor() - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_backpressure_severe_factor() {
        let s = BackpressureSignal::Severe { factor: 0.2 };
        assert!((s.factor() - 0.2).abs() < 1e-9);
    }

    #[test]
    fn test_backpressure_overloaded_factor() {
        assert_eq!(BackpressureSignal::Overloaded.factor(), 0.0);
    }

    #[test]
    fn test_backpressure_labels() {
        assert_eq!(BackpressureSignal::None.label(), "none");
        assert_eq!(BackpressureSignal::Mild { factor: 0.5 }.label(), "mild");
        assert_eq!(BackpressureSignal::Severe { factor: 0.1 }.label(), "severe");
        assert_eq!(BackpressureSignal::Overloaded.label(), "overloaded");
    }

    // ── register_peer ─────────────────────────────────────────────────────────

    #[test]
    fn test_register_peer_basic() {
        let mut s = default_scheduler();
        s.register_peer("p1".to_string(), 100);
        assert_eq!(s.peers.len(), 1);
        assert!(s.peers.contains_key("p1"));
    }

    #[test]
    fn test_register_peer_idempotent() {
        let mut s = default_scheduler();
        s.register_peer("p1".to_string(), 100);
        s.register_peer("p1".to_string(), 200); // should be a no-op
        assert_eq!(s.peers.len(), 1);
        // Timestamp should remain at the first registration.
        assert_eq!(s.peers["p1"].last_seen, 100);
    }

    #[test]
    fn test_register_peer_evicts_oldest_when_full() {
        let config = SchedulerConfig {
            max_peers: 2,
            ..SchedulerConfig::default()
        };
        let mut s = AdaptivePeerScheduler::new(config);
        s.register_peer("p1".to_string(), 10);
        s.register_peer("p2".to_string(), 20);
        assert_eq!(s.peers.len(), 2);
        // Adding a third should evict the oldest (p1, last_seen=10).
        s.register_peer("p3".to_string(), 30);
        assert_eq!(s.peers.len(), 2);
        assert!(!s.peers.contains_key("p1"), "oldest peer must be evicted");
        assert!(s.peers.contains_key("p3"));
    }

    // ── remove_peer ───────────────────────────────────────────────────────────

    #[test]
    fn test_remove_peer_known() {
        let mut s = scheduler_with_peer("p1");
        assert!(s.remove_peer("p1"));
        assert!(s.peers.is_empty());
    }

    #[test]
    fn test_remove_peer_unknown() {
        let mut s = default_scheduler();
        assert!(!s.remove_peer("ghost"));
    }

    #[test]
    fn test_remove_peer_clears_schedule_slot() {
        let mut s = scheduler_with_peer("p1");
        s.recompute_schedule(0);
        s.remove_peer("p1");
        assert!(s.schedule.is_empty());
    }

    // ── record_success / record_failure ───────────────────────────────────────

    #[test]
    fn test_record_success_updates_metrics() {
        let mut s = scheduler_with_peer("p1");
        s.record_success("p1", 80, 500);
        let m = &s.peers["p1"];
        assert_eq!(m.success_count, 1);
        assert_eq!(m.total_latency_ms, 80);
        assert_eq!(m.last_seen, 500);
        assert_eq!(m.consecutive_failures, 0);
        assert_eq!(s.total_requests_succeeded, 1);
    }

    #[test]
    fn test_record_success_ignores_unknown_peer() {
        let mut s = default_scheduler();
        s.record_success("ghost", 10, 1000); // must not panic
        assert_eq!(s.total_requests_succeeded, 0);
    }

    #[test]
    fn test_record_failure_updates_metrics() {
        let mut s = scheduler_with_peer("p1");
        s.record_failure("p1", 200);
        let m = &s.peers["p1"];
        assert_eq!(m.failure_count, 1);
        assert_eq!(m.consecutive_failures, 1);
        assert_eq!(m.last_seen, 200);
    }

    #[test]
    fn test_record_failure_resets_after_success() {
        let mut s = scheduler_with_peer("p1");
        s.record_failure("p1", 100);
        s.record_failure("p1", 200);
        s.record_success("p1", 50, 300);
        assert_eq!(s.peers["p1"].consecutive_failures, 0);
    }

    #[test]
    fn test_record_failure_ignores_unknown_peer() {
        let mut s = default_scheduler();
        s.record_failure("ghost", 1000); // must not panic
    }

    // ── compute_weight ────────────────────────────────────────────────────────

    #[test]
    fn test_compute_weight_new_peer() {
        let s = default_scheduler();
        let m = PeerMetrics::new("p1".to_string(), 0);
        // success_rate = 1.0; avg_latency = MAX → lat_component = 0
        // weight = 0.7*1.0 + 0.3*0 = 0.7; no penalty; clamped → 0.7
        let w = s.compute_weight(&m);
        assert!((w - 0.7).abs() < 1e-9, "got {w}");
    }

    #[test]
    fn test_compute_weight_perfect_fast_peer() {
        let s = default_scheduler();
        let mut m = PeerMetrics::new("p1".to_string(), 0);
        m.success_count = 100;
        m.total_latency_ms = 0; // 0 ms average latency
                                // lat_component = 1/(1+0) = 1.0
                                // weight = 0.7*1.0 + 0.3*1.0 = 1.0
        let w = s.compute_weight(&m);
        assert!((w - 1.0).abs() < 1e-9, "got {w}");
    }

    #[test]
    fn test_compute_weight_clamped_below() {
        let s = default_scheduler();
        let mut m = PeerMetrics::new("p1".to_string(), 0);
        m.failure_count = 1000; // success_rate ≈ 0
                                // weight ≈ 0, penalty applied, but clamped to 0.01
        let w = s.compute_weight(&m);
        assert!(w >= 0.01, "weight must be at least 0.01, got {w}");
    }

    #[test]
    fn test_compute_weight_failure_penalty_applied() {
        let s = default_scheduler();
        let mut m = PeerMetrics::new("p1".to_string(), 0);
        // success_rate = 0.5, below threshold 0.8
        m.success_count = 1;
        m.failure_count = 1;
        m.total_latency_ms = 1000; // 1 s avg latency
        let w = s.compute_weight(&m);
        // weight before penalty = 0.7*0.5 + 0.3*(1/(1+1)) = 0.35+0.15 = 0.5
        // after penalty *= 0.5 → 0.25
        assert!((w - 0.25).abs() < 1e-6, "got {w}");
    }

    // ── recompute_schedule ────────────────────────────────────────────────────

    #[test]
    fn test_recompute_schedule_creates_slots() {
        let mut s = default_scheduler();
        s.register_peer("p1".to_string(), 0);
        s.register_peer("p2".to_string(), 0);
        s.recompute_schedule(100);
        assert_eq!(s.schedule.len(), 2);
    }

    #[test]
    fn test_recompute_schedule_overloaded_zeroes_all() {
        let mut s = scheduler_with_peer("p1");
        s.record_success("p1", 10, 50);
        s.set_backpressure(BackpressureSignal::Overloaded);
        s.recompute_schedule(100);
        let slot = &s.schedule["p1"];
        assert_eq!(slot.allocated_requests, 0);
    }

    #[test]
    fn test_recompute_schedule_mild_backpressure() {
        let mut s = scheduler_with_peer("p1");
        // Pump up success rate so we get solid weight.
        for _ in 0..10 {
            s.record_success("p1", 0, 100);
        }
        s.set_backpressure(BackpressureSignal::Mild { factor: 0.5 });
        s.recompute_schedule(200);
        let slot = &s.schedule["p1"];
        // base=10 * factor=0.5 * weight=1.0 = 5
        assert_eq!(
            slot.allocated_requests, 5,
            "got {}",
            slot.allocated_requests
        );
    }

    #[test]
    fn test_recompute_schedule_no_backpressure_base_alloc() {
        let mut s = scheduler_with_peer("p1");
        for _ in 0..10 {
            s.record_success("p1", 0, 100);
        }
        s.recompute_schedule(200);
        let slot = &s.schedule["p1"];
        // weight=1.0, bp=1.0, base=10 → 10
        assert_eq!(slot.allocated_requests, 10);
    }

    #[test]
    fn test_recompute_schedule_respects_min_alloc() {
        let _s = scheduler_with_peer("p1");
        // Don't pump any metrics; weight=0.7 (new peer)
        // 0.7*10 = 7, which is > min=1 so min doesn't kick in naturally.
        // Force min by reducing base.
        let cfg = SchedulerConfig {
            base_requests_per_peer: 0,
            ..SchedulerConfig::default()
        }; // raw = 0, clamp to min=1
        let mut s2 = AdaptivePeerScheduler::new(cfg);
        s2.register_peer("p1".to_string(), 0);
        s2.recompute_schedule(0);
        assert_eq!(s2.schedule["p1"].allocated_requests, 1);
    }

    #[test]
    fn test_recompute_schedule_respects_max_alloc() {
        let cfg = SchedulerConfig {
            base_requests_per_peer: 1000,
            max_requests_per_peer: 50,
            ..SchedulerConfig::default()
        };
        let mut s = AdaptivePeerScheduler::new(cfg);
        s.register_peer("p1".to_string(), 0);
        for _ in 0..100 {
            s.record_success("p1", 0, 100);
        }
        s.recompute_schedule(200);
        assert!(s.schedule["p1"].allocated_requests <= 50);
    }

    // ── next_peer ─────────────────────────────────────────────────────────────

    #[test]
    fn test_next_peer_empty_schedule() {
        let s = default_scheduler();
        assert!(s.next_peer().is_none());
    }

    #[test]
    fn test_next_peer_all_zero() {
        let mut s = scheduler_with_peer("p1");
        s.set_backpressure(BackpressureSignal::Overloaded);
        s.recompute_schedule(0);
        assert!(s.next_peer().is_none());
    }

    #[test]
    fn test_next_peer_returns_highest_allocation() {
        let mut s = default_scheduler();
        s.register_peer("p1".to_string(), 0);
        s.register_peer("p2".to_string(), 0);
        // Give p2 a much better track record.
        for _ in 0..20 {
            s.record_success("p2", 0, 50);
        }
        s.record_failure("p1", 50);
        s.record_failure("p1", 50);
        s.recompute_schedule(100);
        let best = s.next_peer().expect("should have a peer");
        // p2 must have higher or equal allocation.
        let p2_alloc = s.schedule["p2"].allocated_requests;
        let p1_alloc = s.schedule["p1"].allocated_requests;
        assert!(
            p2_alloc >= p1_alloc,
            "p2={p2_alloc} should >= p1={p1_alloc}"
        );
        // next_peer must be the one with the highest allocation.
        assert_eq!(best, "p2");
    }

    // ── peek_schedule ─────────────────────────────────────────────────────────

    #[test]
    fn test_peek_schedule_sorted_desc() {
        let mut s = default_scheduler();
        for i in 1u32..=5 {
            let id = format!("p{i}");
            s.register_peer(id.clone(), 0);
            for _ in 0..i {
                s.record_success(&id, 0, 100);
            }
        }
        s.recompute_schedule(200);
        let view = s.peek_schedule();
        // Check descending order.
        for pair in view.windows(2) {
            assert!(
                pair[0].1 >= pair[1].1,
                "not descending: {} < {}",
                pair[0].1,
                pair[1].1
            );
        }
    }

    #[test]
    fn test_peek_schedule_empty() {
        let s = default_scheduler();
        assert!(s.peek_schedule().is_empty());
    }

    // ── evict_stale_peers ─────────────────────────────────────────────────────

    #[test]
    fn test_evict_stale_peers_removes_old() {
        let mut s = default_scheduler();
        s.register_peer("old".to_string(), 0);
        s.register_peer("fresh".to_string(), 9000);
        // now=10000, max_idle=5000 → old (10000-0=10000 > 5000) is stale
        s.evict_stale_peers(10_000, 5_000);
        assert!(!s.peers.contains_key("old"), "old peer must be evicted");
        assert!(s.peers.contains_key("fresh"), "fresh peer must remain");
    }

    #[test]
    fn test_evict_stale_peers_keeps_recent() {
        let mut s = default_scheduler();
        s.register_peer("p1".to_string(), 9500);
        s.evict_stale_peers(10_000, 5_000);
        assert!(s.peers.contains_key("p1"));
    }

    #[test]
    fn test_evict_stale_peers_clears_schedule_slot() {
        let mut s = default_scheduler();
        s.register_peer("old".to_string(), 0);
        s.recompute_schedule(100);
        s.evict_stale_peers(10_000, 5_000);
        assert!(!s.schedule.contains_key("old"));
    }

    #[test]
    fn test_evict_stale_peers_exact_boundary() {
        let mut s = default_scheduler();
        // last_seen = 5000, now = 10000, max_idle = 5000 → 10000-5000 = 5000, NOT > 5000 → keep
        s.register_peer("boundary".to_string(), 5000);
        s.evict_stale_peers(10_000, 5_000);
        assert!(
            s.peers.contains_key("boundary"),
            "exact boundary must be kept"
        );
    }

    // ── scheduler_stats ───────────────────────────────────────────────────────

    #[test]
    fn test_scheduler_stats_empty() {
        let s = default_scheduler();
        let st = s.scheduler_stats();
        assert_eq!(st.registered_peers, 0);
        assert_eq!(st.active_slots, 0);
        assert_eq!(st.total_dispatched, 0);
        assert_eq!(st.total_succeeded, 0);
        assert!((st.success_rate - 1.0).abs() < 1e-9);
        assert_eq!(st.backpressure, "none");
    }

    #[test]
    fn test_scheduler_stats_after_activity() {
        let mut s = scheduler_with_peer("p1");
        s.record_success("p1", 30, 100);
        s.mark_dispatched();
        s.recompute_schedule(200);
        let st = s.scheduler_stats();
        assert_eq!(st.registered_peers, 1);
        assert_eq!(st.total_succeeded, 1);
        assert_eq!(st.total_dispatched, 1);
        assert!((st.success_rate - 1.0).abs() < 1e-9);
        assert!(st.active_slots >= 1);
    }

    #[test]
    fn test_scheduler_stats_backpressure_label() {
        let mut s = default_scheduler();
        s.set_backpressure(BackpressureSignal::Severe { factor: 0.1 });
        assert_eq!(s.scheduler_stats().backpressure, "severe");
    }

    // ── mark_dispatched ───────────────────────────────────────────────────────

    #[test]
    fn test_mark_dispatched_increments_counter() {
        let mut s = default_scheduler();
        assert_eq!(s.total_requests_dispatched, 0);
        s.mark_dispatched();
        s.mark_dispatched();
        assert_eq!(s.total_requests_dispatched, 2);
    }

    // ── ScheduleSlot fields ───────────────────────────────────────────────────

    #[test]
    fn test_schedule_slot_last_updated_set() {
        let mut s = scheduler_with_peer("p1");
        s.recompute_schedule(9999);
        assert_eq!(s.schedule["p1"].last_updated, 9999);
    }

    #[test]
    fn test_schedule_slot_weight_set() {
        let mut s = scheduler_with_peer("p1");
        s.recompute_schedule(0);
        let w = s.schedule["p1"].weight;
        assert!((0.01..=1.0).contains(&w), "weight {w} out of range");
    }

    // ── Eviction + schedule consistency ──────────────────────────────────────

    #[test]
    fn test_recompute_removes_orphaned_schedule_slots() {
        let mut s = default_scheduler();
        s.register_peer("p1".to_string(), 0);
        s.recompute_schedule(0);
        // Manually add an orphaned slot (simulates a race or previous eviction).
        s.schedule.insert(
            "ghost".to_string(),
            ScheduleSlot {
                peer_id: "ghost".to_string(),
                allocated_requests: 5,
                weight: 0.5,
                last_updated: 0,
            },
        );
        s.recompute_schedule(100);
        assert!(
            !s.schedule.contains_key("ghost"),
            "orphaned slot must be removed"
        );
    }

    // ── Saturation arithmetic ─────────────────────────────────────────────────

    #[test]
    fn test_success_count_saturates() {
        let mut s = scheduler_with_peer("p1");
        s.peers.get_mut("p1").expect("must exist").success_count = u64::MAX;
        s.record_success("p1", 0, 0); // must not overflow
        assert_eq!(s.peers["p1"].success_count, u64::MAX);
    }

    #[test]
    fn test_failure_count_saturates() {
        let mut s = scheduler_with_peer("p1");
        s.peers.get_mut("p1").expect("must exist").failure_count = u64::MAX;
        s.record_failure("p1", 0); // must not overflow
        assert_eq!(s.peers["p1"].failure_count, u64::MAX);
    }

    // ── Multiple peers competitive allocation ─────────────────────────────────

    #[test]
    fn test_competitive_allocation_higher_success_wins() {
        let mut s = default_scheduler();
        s.register_peer("good".to_string(), 0);
        s.register_peer("bad".to_string(), 0);

        for _ in 0..100 {
            s.record_success("good", 10, 100);
        }
        for _ in 0..100 {
            s.record_failure("bad", 100);
        }
        s.recompute_schedule(200);

        let good = s.schedule["good"].allocated_requests;
        let bad = s.schedule["bad"].allocated_requests;
        assert!(good > bad, "good={good} should > bad={bad}");
    }

    #[test]
    fn test_severe_backpressure_reduces_allocation() {
        let mut s = scheduler_with_peer("p1");
        for _ in 0..10 {
            s.record_success("p1", 0, 100);
        }
        s.recompute_schedule(200);
        let full_alloc = s.schedule["p1"].allocated_requests;

        s.set_backpressure(BackpressureSignal::Severe { factor: 0.1 });
        s.recompute_schedule(300);
        let reduced_alloc = s.schedule["p1"].allocated_requests;

        assert!(
            reduced_alloc <= full_alloc,
            "severe bp should reduce: {reduced_alloc} <= {full_alloc}"
        );
    }
}
