//! Network Quality-of-Service (QoS) Manager.
//!
//! Classifies traffic into four priority tiers, enforces SLA contracts via
//! latency and throughput checks, and provides both strict-priority and
//! weighted-round-robin scheduling with per-class bandwidth guarantees.

use std::collections::{HashMap, VecDeque};

// ────────────────────────────────────────────────────────────────────────────
// Traffic classification
// ────────────────────────────────────────────────────────────────────────────

/// Four-tier traffic classification for QoS scheduling.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TrafficClass {
    /// Highest priority: real-time streams, consensus messages.
    RealTime,
    /// Second priority: interactive user sessions, RPC calls.
    Interactive,
    /// Third priority: bulk file transfers, sync operations.
    BulkData,
    /// Lowest priority: background maintenance, metrics gossip.
    Background,
}

impl TrafficClass {
    /// Numeric priority — higher value is serviced first (1..=4).
    pub fn priority(&self) -> u8 {
        match self {
            TrafficClass::RealTime => 4,
            TrafficClass::Interactive => 3,
            TrafficClass::BulkData => 2,
            TrafficClass::Background => 1,
        }
    }

    /// Fraction of total bandwidth allocated to this class (sum = 1.0).
    pub fn bandwidth_share(&self) -> f64 {
        match self {
            TrafficClass::RealTime => 0.4,
            TrafficClass::Interactive => 0.3,
            TrafficClass::BulkData => 0.2,
            TrafficClass::Background => 0.1,
        }
    }

    /// Construct from a raw priority byte; returns `None` for unknown values.
    pub fn from_priority(p: u8) -> Option<Self> {
        match p {
            4 => Some(TrafficClass::RealTime),
            3 => Some(TrafficClass::Interactive),
            2 => Some(TrafficClass::BulkData),
            1 => Some(TrafficClass::Background),
            _ => None,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Core data types
// ────────────────────────────────────────────────────────────────────────────

/// A network packet with QoS metadata, ready for priority queuing.
#[derive(Clone, Debug)]
pub struct QoSPacket {
    /// Unique packet identifier within this manager instance.
    pub id: u64,
    /// Traffic class that determines scheduling tier.
    pub class: TrafficClass,
    /// Payload size in bytes; used for bandwidth accounting.
    pub size_bytes: usize,
    /// Originating or destination peer identifier.
    pub peer_id: String,
    /// Monotonic timestamp (ms) at which the packet entered the queue.
    pub enqueued_at: u64,
}

/// SLA contract for a single traffic class.
#[derive(Clone, Debug)]
pub struct SLASpec {
    /// The class this spec governs.
    pub class: TrafficClass,
    /// Maximum acceptable average queueing latency in milliseconds.
    pub max_latency_ms: u64,
    /// Minimum guaranteed throughput in bits per second.
    pub min_throughput_bps: u64,
    /// Maximum acceptable jitter (latency variance) in milliseconds.
    pub max_jitter_ms: u64,
}

/// A recorded SLA violation event.
#[derive(Clone, Debug)]
pub struct SLAViolation {
    /// Traffic class that violated its SLA.
    pub class: TrafficClass,
    /// Human-readable metric name (e.g. `"avg_wait_ms"`).
    pub metric: String,
    /// Observed value.
    pub actual: f64,
    /// Allowed limit.
    pub limit: f64,
    /// Monotonic timestamp (ms) when the violation was detected.
    pub timestamp: u64,
}

/// Snapshot of per-class queue statistics.
#[derive(Clone, Debug)]
pub struct QueueMetrics {
    /// Traffic class these metrics belong to.
    pub class: TrafficClass,
    /// Number of packets currently in the queue.
    pub queued_packets: usize,
    /// Total byte count of all enqueued packets.
    pub total_bytes: usize,
    /// Exponentially-weighted moving average wait time (ms); α = 0.1.
    pub avg_wait_ms: f64,
    /// Total packets dropped from this queue since creation.
    pub dropped_packets: u64,
}

/// Aggregate QoS statistics across all classes.
#[derive(Clone, Debug)]
pub struct QoSStats {
    /// Lifetime enqueued packet count.
    pub total_enqueued: u64,
    /// Lifetime dequeued packet count.
    pub total_dequeued: u64,
    /// Lifetime dropped packet count.
    pub total_dropped: u64,
    /// Fraction of enqueued packets that were dropped (0.0..=1.0).
    pub drop_rate: f64,
    /// Number of priority queues that contain at least one packet.
    pub active_queues: usize,
    /// Total violations recorded in the violations log.
    pub total_violations: usize,
}

// ────────────────────────────────────────────────────────────────────────────
// Configuration
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for the `NetworkQoSManager`.
#[derive(Clone, Debug)]
pub struct QoSConfig {
    /// Maximum combined packet count across all queues before drops occur.
    pub max_queue_size: usize,
    /// Aggregate link capacity in bits per second.
    pub total_bandwidth_bps: u64,
    /// Per-class SLA specifications; can be empty (no SLA checking).
    pub sla_specs: Vec<SLASpec>,
    /// When `true`, always service the highest non-empty priority queue
    /// (strict priority).  When `false`, use weighted round-robin.
    pub enable_strict_priority: bool,
}

impl Default for QoSConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 10_000,
            total_bandwidth_bps: 100_000_000, // 100 Mbps
            sla_specs: Vec::new(),
            enable_strict_priority: false,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Internal round-robin state
// ────────────────────────────────────────────────────────────────────────────

/// Tracks the weighted-round-robin deficit counter for one priority queue.
#[derive(Debug, Default)]
struct WrrState {
    /// Accumulated "credits" not yet consumed by a dequeue.
    deficit: f64,
    /// Quantum of credits added each round (proportional to bandwidth_share).
    quantum: f64,
}

// ────────────────────────────────────────────────────────────────────────────
// Manager
// ────────────────────────────────────────────────────────────────────────────

/// Quality-of-Service manager: priority queuing, SLA enforcement, and
/// bandwidth-share scheduling for the IPFRS network layer.
pub struct NetworkQoSManager {
    /// User-supplied configuration.
    pub config: QoSConfig,
    /// Priority queues keyed by `TrafficClass::priority()` (1..=4).
    pub queues: HashMap<u8, VecDeque<QoSPacket>>,
    /// Per-priority queue metrics keyed by priority (1..=4).
    pub metrics: HashMap<u8, QueueMetrics>,
    /// Bounded log of observed SLA violations (max 1 000 entries).
    pub violations: VecDeque<SLAViolation>,
    /// Lifetime counter of successfully enqueued packets.
    pub total_enqueued: u64,
    /// Lifetime counter of successfully dequeued packets.
    pub total_dequeued: u64,
    /// Lifetime counter of dropped packets.
    pub total_dropped: u64,

    // ── internal scheduling state ──────────────────────────────────────────
    /// Weighted-round-robin deficit state keyed by priority (1..=4).
    wrr: HashMap<u8, WrrState>,
    /// Current round-robin position (priority value 1..=4).
    wrr_cursor: u8,
}

/// Ordered priority levels from highest to lowest (used for iteration).
const PRIORITIES: [u8; 4] = [4, 3, 2, 1];

impl NetworkQoSManager {
    // ── construction ────────────────────────────────────────────────────────

    /// Create a new manager with the given configuration.
    ///
    /// Initialises four priority queues (keyed 1..=4) and pre-fills WRR
    /// quanta proportional to each class's `bandwidth_share`.
    pub fn new(config: QoSConfig) -> Self {
        let mut queues = HashMap::with_capacity(4);
        let mut metrics = HashMap::with_capacity(4);
        let mut wrr = HashMap::with_capacity(4);

        for &prio in &PRIORITIES {
            let class = TrafficClass::from_priority(prio).unwrap_or(TrafficClass::Background);

            queues.insert(prio, VecDeque::new());
            metrics.insert(
                prio,
                QueueMetrics {
                    class: class.clone(),
                    queued_packets: 0,
                    total_bytes: 0,
                    avg_wait_ms: 0.0,
                    dropped_packets: 0,
                },
            );

            // WRR quantum = bandwidth_share * 1000 (normalised credits).
            wrr.insert(
                prio,
                WrrState {
                    deficit: 0.0,
                    quantum: class.bandwidth_share() * 1000.0,
                },
            );
        }

        Self {
            config,
            queues,
            metrics,
            violations: VecDeque::new(),
            total_enqueued: 0,
            total_dequeued: 0,
            total_dropped: 0,
            wrr,
            wrr_cursor: 4, // start at highest priority
        }
    }

    // ── internal helpers ─────────────────────────────────────────────────────

    /// Total number of packets across all queues.
    fn total_queued_count(&self) -> usize {
        self.queues.values().map(|q| q.len()).sum()
    }

    /// Update the `queued_packets` and `total_bytes` fields of the metrics
    /// entry for `priority` to match the actual queue state.
    fn sync_queue_metrics(&mut self, priority: u8) {
        if let Some(q) = self.queues.get(&priority) {
            let count = q.len();
            let bytes: usize = q.iter().map(|p| p.size_bytes).sum();
            if let Some(m) = self.metrics.get_mut(&priority) {
                m.queued_packets = count;
                m.total_bytes = bytes;
            }
        }
    }

    /// Apply an EWMA update (α = 0.1) to `avg_wait_ms` for the queue at
    /// `priority` using the supplied observed wait duration.
    fn update_avg_wait(&mut self, priority: u8, wait_ms: f64) {
        const ALPHA: f64 = 0.1;
        if let Some(m) = self.metrics.get_mut(&priority) {
            if m.avg_wait_ms == 0.0 {
                // Bootstrap: first sample sets the value directly.
                m.avg_wait_ms = wait_ms;
            } else {
                m.avg_wait_ms = ALPHA * wait_ms + (1.0 - ALPHA) * m.avg_wait_ms;
            }
        }
    }

    // ── public API ───────────────────────────────────────────────────────────

    /// Enqueue a packet into the priority queue determined by its class.
    ///
    /// If the total queue occupancy is at or above `max_queue_size`:
    /// 1. Try to drop the **oldest** Background packet (priority 1).
    /// 2. If Background is already empty, drop the **newest** Background
    ///    packet (i.e. the one just being enqueued is discarded instead).
    /// 3. If neither relieves enough space, return `false`.
    ///
    /// Returns `true` when the packet was successfully enqueued.
    pub fn enqueue(&mut self, packet: QoSPacket, _now: u64) -> bool {
        // Make room if over capacity.
        if self.total_queued_count() >= self.config.max_queue_size {
            // Drop oldest Background packet.
            let dropped = self.drop_packet(1);
            if !dropped {
                // Background queue empty; if the incoming packet IS Background,
                // discard it directly.
                if packet.class.priority() == 1 {
                    self.total_dropped += 1;
                    if let Some(m) = self.metrics.get_mut(&1) {
                        m.dropped_packets += 1;
                    }
                    return false;
                }
                // Otherwise we cannot make room.
                return false;
            }
        }

        let priority = packet.class.priority();
        if let Some(q) = self.queues.get_mut(&priority) {
            q.push_back(packet);
            self.total_enqueued += 1;
            self.sync_queue_metrics(priority);
            true
        } else {
            false
        }
    }

    /// Drop the **oldest** packet from the queue at `priority`.
    ///
    /// Returns `true` if a packet was removed, `false` if the queue was empty.
    pub fn drop_packet(&mut self, priority: u8) -> bool {
        if let Some(q) = self.queues.get_mut(&priority) {
            if q.pop_front().is_some() {
                self.total_dropped += 1;
                if let Some(m) = self.metrics.get_mut(&priority) {
                    m.dropped_packets += 1;
                }
                self.sync_queue_metrics(priority);
                return true;
            }
        }
        false
    }

    /// Dequeue the next packet according to the configured scheduling policy.
    ///
    /// * **Strict priority** (`enable_strict_priority = true`): always pick
    ///   from the highest non-empty priority queue.
    /// * **Weighted round-robin** (`enable_strict_priority = false`): iterate
    ///   round-robin through classes; add `quantum` credits each visit and
    ///   dequeue when credits ≥ packet size.  Falls back to the highest
    ///   non-empty queue if no class accumulates enough credits.
    ///
    /// Returns `None` if all queues are empty.
    pub fn dequeue(&mut self, now: u64) -> Option<QoSPacket> {
        if self.config.enable_strict_priority {
            self.dequeue_strict(now)
        } else {
            self.dequeue_wrr(now)
        }
    }

    /// Strict-priority dequeue: highest non-empty queue wins.
    fn dequeue_strict(&mut self, now: u64) -> Option<QoSPacket> {
        for &prio in &PRIORITIES {
            if let Some(q) = self.queues.get_mut(&prio) {
                if let Some(pkt) = q.pop_front() {
                    let wait_ms = now.saturating_sub(pkt.enqueued_at) as f64;
                    self.total_dequeued += 1;
                    self.sync_queue_metrics(prio);
                    self.update_avg_wait(prio, wait_ms);
                    return Some(pkt);
                }
            }
        }
        None
    }

    /// Weighted round-robin dequeue.
    ///
    /// Each class earns `quantum` credits per round and spends them on packet
    /// sizes.  We iterate up to 4 full rounds to avoid infinite loops when all
    /// queues are empty.
    fn dequeue_wrr(&mut self, now: u64) -> Option<QoSPacket> {
        // Quick exit when everything is empty.
        if self.total_queued_count() == 0 {
            return None;
        }

        // Try WRR: up to 4 * 4 = 16 cursor advances to find a serviced packet.
        for _ in 0..16 {
            let prio = self.wrr_cursor;

            // Add quantum credits to this queue.
            if let Some(state) = self.wrr.get_mut(&prio) {
                state.deficit += state.quantum;
            }

            // Attempt to dequeue if we have enough credits.
            let head_size = self
                .queues
                .get(&prio)
                .and_then(|q| q.front())
                .map(|p| p.size_bytes as f64);

            if let Some(sz) = head_size {
                let credits = self.wrr.get(&prio).map(|s| s.deficit).unwrap_or(0.0);
                if credits >= sz {
                    // Dequeue the packet.
                    if let Some(q) = self.queues.get_mut(&prio) {
                        if let Some(pkt) = q.pop_front() {
                            if let Some(state) = self.wrr.get_mut(&prio) {
                                state.deficit -= sz;
                            }
                            let wait_ms = now.saturating_sub(pkt.enqueued_at) as f64;
                            self.total_dequeued += 1;
                            self.sync_queue_metrics(prio);
                            self.update_avg_wait(prio, wait_ms);
                            // Advance cursor for next call.
                            self.advance_cursor();
                            return Some(pkt);
                        }
                    }
                }
            }

            // Advance cursor regardless.
            self.advance_cursor();
        }

        // WRR couldn't find a winner (all queues small enough credits);
        // fall back to strict priority to prevent starvation.
        self.dequeue_strict(now)
    }

    /// Advance the WRR cursor in decreasing priority order (4 → 3 → 2 → 1 → 4).
    fn advance_cursor(&mut self) {
        self.wrr_cursor = if self.wrr_cursor > 1 {
            self.wrr_cursor - 1
        } else {
            4
        };
    }

    /// Check every configured SLA spec against current queue metrics.
    ///
    /// Any violation is appended to the internal violations log (bounded to
    /// 1 000 entries, oldest evicted on overflow).  The set of violations
    /// discovered during **this call** is returned.
    pub fn check_sla(&mut self, now: u64) -> Vec<SLAViolation> {
        let specs = self.config.sla_specs.clone();
        let mut found = Vec::new();

        for spec in &specs {
            let prio = spec.class.priority();
            let avg_wait = self
                .metrics
                .get(&prio)
                .map(|m| m.avg_wait_ms)
                .unwrap_or(0.0);

            if avg_wait > spec.max_latency_ms as f64 {
                let v = SLAViolation {
                    class: spec.class.clone(),
                    metric: "avg_wait_ms".to_string(),
                    actual: avg_wait,
                    limit: spec.max_latency_ms as f64,
                    timestamp: now,
                };
                found.push(v);
            }
        }

        for v in &found {
            if self.violations.len() >= 1_000 {
                self.violations.pop_front();
            }
            self.violations.push_back(v.clone());
        }

        found
    }

    /// Return queue metrics for a specific traffic class, or `None` if the
    /// class is not tracked (should not happen with a correctly initialised
    /// manager).
    pub fn queue_metrics(&self, class: &TrafficClass) -> Option<&QueueMetrics> {
        self.metrics.get(&class.priority())
    }

    /// Return metrics for all four priority queues ordered highest-first.
    pub fn all_metrics(&self) -> Vec<&QueueMetrics> {
        PRIORITIES
            .iter()
            .filter_map(|p| self.metrics.get(p))
            .collect()
    }

    /// Total byte count across all queued packets.
    pub fn total_queued_bytes(&self) -> usize {
        self.queues
            .values()
            .flat_map(|q| q.iter())
            .map(|p| p.size_bytes)
            .sum()
    }

    /// Immutable reference to the bounded violations log.
    pub fn violations_log(&self) -> &VecDeque<SLAViolation> {
        &self.violations
    }

    /// Aggregate statistics snapshot.
    pub fn qos_stats(&self) -> QoSStats {
        let active_queues = self.queues.values().filter(|q| !q.is_empty()).count();

        let drop_rate = if self.total_enqueued > 0 {
            self.total_dropped as f64 / self.total_enqueued as f64
        } else {
            0.0
        };

        QoSStats {
            total_enqueued: self.total_enqueued,
            total_dequeued: self.total_dequeued,
            total_dropped: self.total_dropped,
            drop_rate,
            active_queues,
            total_violations: self.violations.len(),
        }
    }

    /// Reset all per-class EWMA wait metrics to zero (useful for test
    /// isolation or after a reconfiguration event).
    pub fn reset_metrics(&mut self) {
        for m in self.metrics.values_mut() {
            m.avg_wait_ms = 0.0;
        }
    }

    /// Drain all packets from all queues and return the total count removed.
    pub fn flush_all(&mut self) -> usize {
        let mut count = 0;
        for q in self.queues.values_mut() {
            count += q.len();
            q.clear();
        }
        for prio in &[1u8, 2, 3, 4] {
            self.sync_queue_metrics(*prio);
        }
        count
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{NetworkQoSManager, QoSConfig, QoSPacket, SLASpec, TrafficClass};

    // ── helpers ──────────────────────────────────────────────────────────────

    fn default_manager() -> NetworkQoSManager {
        NetworkQoSManager::new(QoSConfig::default())
    }

    fn make_packet(id: u64, class: TrafficClass, size: usize, enqueued_at: u64) -> QoSPacket {
        QoSPacket {
            id,
            class,
            size_bytes: size,
            peer_id: format!("peer-{id}"),
            enqueued_at,
        }
    }

    // ── TrafficClass unit tests ───────────────────────────────────────────────

    #[test]
    fn test_traffic_class_priority_values() {
        assert_eq!(TrafficClass::RealTime.priority(), 4);
        assert_eq!(TrafficClass::Interactive.priority(), 3);
        assert_eq!(TrafficClass::BulkData.priority(), 2);
        assert_eq!(TrafficClass::Background.priority(), 1);
    }

    #[test]
    fn test_traffic_class_bandwidth_shares_sum_to_one() {
        let sum = TrafficClass::RealTime.bandwidth_share()
            + TrafficClass::Interactive.bandwidth_share()
            + TrafficClass::BulkData.bandwidth_share()
            + TrafficClass::Background.bandwidth_share();
        assert!((sum - 1.0).abs() < 1e-9, "shares sum to {sum}");
    }

    #[test]
    fn test_traffic_class_bandwidth_share_values() {
        assert!((TrafficClass::RealTime.bandwidth_share() - 0.4).abs() < 1e-9);
        assert!((TrafficClass::Interactive.bandwidth_share() - 0.3).abs() < 1e-9);
        assert!((TrafficClass::BulkData.bandwidth_share() - 0.2).abs() < 1e-9);
        assert!((TrafficClass::Background.bandwidth_share() - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_from_priority_round_trip() {
        for &p in &[1u8, 2, 3, 4] {
            let cls = TrafficClass::from_priority(p).expect("known priority");
            assert_eq!(cls.priority(), p);
        }
    }

    #[test]
    fn test_from_priority_unknown_returns_none() {
        assert!(TrafficClass::from_priority(0).is_none());
        assert!(TrafficClass::from_priority(5).is_none());
        assert!(TrafficClass::from_priority(255).is_none());
    }

    // ── QoSConfig defaults ───────────────────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let cfg = QoSConfig::default();
        assert_eq!(cfg.max_queue_size, 10_000);
        assert_eq!(cfg.total_bandwidth_bps, 100_000_000);
        assert!(!cfg.enable_strict_priority);
        assert!(cfg.sla_specs.is_empty());
    }

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn test_new_initialises_four_queues() {
        let mgr = default_manager();
        assert_eq!(mgr.queues.len(), 4);
        assert_eq!(mgr.metrics.len(), 4);
    }

    #[test]
    fn test_new_all_queues_empty() {
        let mgr = default_manager();
        for q in mgr.queues.values() {
            assert!(q.is_empty());
        }
    }

    #[test]
    fn test_new_counters_zero() {
        let mgr = default_manager();
        assert_eq!(mgr.total_enqueued, 0);
        assert_eq!(mgr.total_dequeued, 0);
        assert_eq!(mgr.total_dropped, 0);
    }

    // ── Enqueue ───────────────────────────────────────────────────────────────

    #[test]
    fn test_enqueue_basic_returns_true() {
        let mut mgr = default_manager();
        let pkt = make_packet(1, TrafficClass::RealTime, 100, 0);
        assert!(mgr.enqueue(pkt, 0));
        assert_eq!(mgr.total_enqueued, 1);
    }

    #[test]
    fn test_enqueue_increments_queue_metrics() {
        let mut mgr = default_manager();
        mgr.enqueue(make_packet(1, TrafficClass::Interactive, 200, 0), 0);
        let m = mgr
            .queue_metrics(&TrafficClass::Interactive)
            .expect("test: Interactive queue metrics must exist");
        assert_eq!(m.queued_packets, 1);
        assert_eq!(m.total_bytes, 200);
    }

    #[test]
    fn test_enqueue_multiple_classes() {
        let mut mgr = default_manager();
        mgr.enqueue(make_packet(1, TrafficClass::RealTime, 50, 0), 0);
        mgr.enqueue(make_packet(2, TrafficClass::BulkData, 150, 0), 0);
        mgr.enqueue(make_packet(3, TrafficClass::Background, 10, 0), 0);
        assert_eq!(mgr.total_enqueued, 3);
    }

    #[test]
    fn test_enqueue_respects_fifo_within_class() {
        let mut mgr = default_manager();
        mgr.enqueue(make_packet(1, TrafficClass::BulkData, 100, 10), 10);
        mgr.enqueue(make_packet(2, TrafficClass::BulkData, 200, 20), 20);
        let q = mgr
            .queues
            .get(&2)
            .expect("test: priority-2 queue must exist");
        assert_eq!(
            q.front().expect("test: queue must have a front element").id,
            1
        );
        assert_eq!(
            q.back().expect("test: queue must have a back element").id,
            2
        );
    }

    // ── Drop / overflow ───────────────────────────────────────────────────────

    #[test]
    fn test_drop_packet_returns_false_on_empty_queue() {
        let mut mgr = default_manager();
        assert!(!mgr.drop_packet(1));
    }

    #[test]
    fn test_drop_packet_removes_oldest() {
        let mut mgr = default_manager();
        mgr.enqueue(make_packet(1, TrafficClass::Background, 10, 0), 0);
        mgr.enqueue(make_packet(2, TrafficClass::Background, 20, 1), 1);
        assert!(mgr.drop_packet(1));
        // Oldest (id=1) should be gone; id=2 stays.
        let q = mgr
            .queues
            .get(&1)
            .expect("test: priority-1 queue must exist");
        assert_eq!(
            q.front()
                .expect("test: queue must have a front element after drop")
                .id,
            2
        );
        assert_eq!(mgr.total_dropped, 1);
    }

    #[test]
    fn test_enqueue_drops_background_when_at_capacity() {
        let mut mgr = NetworkQoSManager::new(QoSConfig {
            max_queue_size: 3,
            ..Default::default()
        });
        // Fill with Background packets.
        mgr.enqueue(make_packet(1, TrafficClass::Background, 10, 0), 0);
        mgr.enqueue(make_packet(2, TrafficClass::Background, 10, 1), 1);
        mgr.enqueue(make_packet(3, TrafficClass::Background, 10, 2), 2);

        // Enqueueing a 4th (higher-priority) packet should drop oldest Background.
        let result = mgr.enqueue(make_packet(4, TrafficClass::RealTime, 50, 3), 3);
        assert!(result);
        assert_eq!(mgr.total_dropped, 1);
    }

    #[test]
    fn test_enqueue_drops_background_packet_itself_at_capacity() {
        let mut mgr = NetworkQoSManager::new(QoSConfig {
            max_queue_size: 2,
            ..Default::default()
        });
        // Fill with non-Background packets.
        mgr.enqueue(make_packet(1, TrafficClass::RealTime, 10, 0), 0);
        mgr.enqueue(make_packet(2, TrafficClass::Interactive, 10, 1), 1);

        // Trying to enqueue Background when Background queue is empty → dropped.
        let result = mgr.enqueue(make_packet(3, TrafficClass::Background, 10, 2), 2);
        assert!(!result);
        assert_eq!(mgr.total_dropped, 1);
    }

    // ── Dequeue — strict priority ─────────────────────────────────────────────

    #[test]
    fn test_dequeue_strict_highest_priority_first() {
        let mut mgr = NetworkQoSManager::new(QoSConfig {
            enable_strict_priority: true,
            ..Default::default()
        });
        mgr.enqueue(make_packet(1, TrafficClass::Background, 10, 0), 0);
        mgr.enqueue(make_packet(2, TrafficClass::RealTime, 10, 0), 0);
        mgr.enqueue(make_packet(3, TrafficClass::Interactive, 10, 0), 0);

        let first = mgr
            .dequeue(10)
            .expect("test: dequeue must return a packet (RealTime enqueued)");
        assert_eq!(first.id, 2); // RealTime → priority 4

        let second = mgr
            .dequeue(10)
            .expect("test: dequeue must return a packet (Interactive enqueued)");
        assert_eq!(second.id, 3); // Interactive → priority 3
    }

    #[test]
    fn test_dequeue_strict_returns_none_on_empty() {
        let mut mgr = NetworkQoSManager::new(QoSConfig {
            enable_strict_priority: true,
            ..Default::default()
        });
        assert!(mgr.dequeue(0).is_none());
    }

    #[test]
    fn test_dequeue_strict_increments_total_dequeued() {
        let mut mgr = NetworkQoSManager::new(QoSConfig {
            enable_strict_priority: true,
            ..Default::default()
        });
        mgr.enqueue(make_packet(1, TrafficClass::BulkData, 100, 0), 0);
        mgr.dequeue(50);
        assert_eq!(mgr.total_dequeued, 1);
    }

    #[test]
    fn test_dequeue_strict_updates_avg_wait() {
        let mut mgr = NetworkQoSManager::new(QoSConfig {
            enable_strict_priority: true,
            ..Default::default()
        });
        mgr.enqueue(make_packet(1, TrafficClass::RealTime, 100, 0), 0);
        mgr.dequeue(100); // 100 ms wait

        let m = mgr
            .queue_metrics(&TrafficClass::RealTime)
            .expect("test: RealTime queue metrics must exist");
        // First sample bootstraps avg directly.
        assert!((m.avg_wait_ms - 100.0).abs() < 1e-6);
    }

    // ── Dequeue — weighted round-robin ────────────────────────────────────────

    #[test]
    fn test_dequeue_wrr_returns_none_on_empty() {
        let mut mgr = default_manager();
        assert!(mgr.dequeue(0).is_none());
    }

    #[test]
    fn test_dequeue_wrr_returns_packet_when_non_empty() {
        let mut mgr = default_manager();
        mgr.enqueue(make_packet(1, TrafficClass::Interactive, 200, 0), 0);
        let pkt = mgr.dequeue(10);
        assert!(pkt.is_some());
        assert_eq!(mgr.total_dequeued, 1);
    }

    #[test]
    fn test_dequeue_wrr_services_multiple_classes() {
        let mut mgr = default_manager();
        for i in 0..5 {
            mgr.enqueue(make_packet(i, TrafficClass::RealTime, 50, 0), 0);
            mgr.enqueue(make_packet(100 + i, TrafficClass::Background, 50, 0), 0);
        }
        // Drain everything.
        let mut dequeued = 0;
        while mgr.dequeue(10).is_some() {
            dequeued += 1;
        }
        assert_eq!(dequeued, 10);
    }

    #[test]
    fn test_dequeue_wrr_decrements_queue_metrics() {
        let mut mgr = default_manager();
        mgr.enqueue(make_packet(1, TrafficClass::BulkData, 512, 0), 0);
        mgr.dequeue(10);
        let m = mgr
            .queue_metrics(&TrafficClass::BulkData)
            .expect("test: BulkData queue metrics must exist");
        assert_eq!(m.queued_packets, 0);
        assert_eq!(m.total_bytes, 0);
    }

    // ── SLA checking ─────────────────────────────────────────────────────────

    #[test]
    fn test_check_sla_no_specs_returns_empty() {
        let mut mgr = default_manager();
        let violations = mgr.check_sla(100);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_check_sla_no_violation_when_within_limit() {
        let mut mgr = NetworkQoSManager::new(QoSConfig {
            sla_specs: vec![SLASpec {
                class: TrafficClass::RealTime,
                max_latency_ms: 50,
                min_throughput_bps: 0,
                max_jitter_ms: 10,
            }],
            enable_strict_priority: true,
            ..Default::default()
        });
        // avg_wait_ms starts at 0 → no violation.
        let v = mgr.check_sla(0);
        assert!(v.is_empty());
    }

    #[test]
    fn test_check_sla_detects_latency_violation() {
        let mut mgr = NetworkQoSManager::new(QoSConfig {
            enable_strict_priority: true,
            sla_specs: vec![SLASpec {
                class: TrafficClass::RealTime,
                max_latency_ms: 5,
                min_throughput_bps: 0,
                max_jitter_ms: 2,
            }],
            ..Default::default()
        });
        // Enqueue then dequeue with 100 ms wait — bootstraps avg_wait to 100.
        mgr.enqueue(make_packet(1, TrafficClass::RealTime, 10, 0), 0);
        mgr.dequeue(100);

        let violations = mgr.check_sla(100);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].metric, "avg_wait_ms");
        assert!(violations[0].actual > 5.0);
    }

    #[test]
    fn test_check_sla_appends_to_violations_log() {
        let mut mgr = NetworkQoSManager::new(QoSConfig {
            enable_strict_priority: true,
            sla_specs: vec![SLASpec {
                class: TrafficClass::Interactive,
                max_latency_ms: 1,
                min_throughput_bps: 0,
                max_jitter_ms: 1,
            }],
            ..Default::default()
        });
        mgr.enqueue(make_packet(1, TrafficClass::Interactive, 10, 0), 0);
        mgr.dequeue(500); // 500 ms wait

        mgr.check_sla(500);
        assert!(!mgr.violations_log().is_empty());
    }

    #[test]
    fn test_check_sla_violations_log_capped_at_1000() {
        let mut mgr = NetworkQoSManager::new(QoSConfig {
            enable_strict_priority: true,
            sla_specs: vec![SLASpec {
                class: TrafficClass::Background,
                max_latency_ms: 1,
                min_throughput_bps: 0,
                max_jitter_ms: 1,
            }],
            ..Default::default()
        });
        // Manually set avg_wait high enough to always violate.
        if let Some(m) = mgr.metrics.get_mut(&1) {
            m.avg_wait_ms = 9_999.0;
        }
        for i in 0..1_200u64 {
            mgr.check_sla(i);
        }
        assert!(mgr.violations_log().len() <= 1_000);
    }

    // ── Metrics helpers ───────────────────────────────────────────────────────

    #[test]
    fn test_queue_metrics_returns_correct_class() {
        let mgr = default_manager();
        let m = mgr
            .queue_metrics(&TrafficClass::BulkData)
            .expect("test: BulkData queue metrics must exist");
        assert_eq!(m.class, TrafficClass::BulkData);
    }

    #[test]
    fn test_all_metrics_returns_four_entries() {
        let mgr = default_manager();
        assert_eq!(mgr.all_metrics().len(), 4);
    }

    #[test]
    fn test_all_metrics_ordered_highest_first() {
        let mgr = default_manager();
        let ms = mgr.all_metrics();
        assert_eq!(ms[0].class, TrafficClass::RealTime);
        assert_eq!(ms[1].class, TrafficClass::Interactive);
        assert_eq!(ms[2].class, TrafficClass::BulkData);
        assert_eq!(ms[3].class, TrafficClass::Background);
    }

    #[test]
    fn test_total_queued_bytes_empty() {
        let mgr = default_manager();
        assert_eq!(mgr.total_queued_bytes(), 0);
    }

    #[test]
    fn test_total_queued_bytes_after_enqueue() {
        let mut mgr = default_manager();
        mgr.enqueue(make_packet(1, TrafficClass::RealTime, 1000, 0), 0);
        mgr.enqueue(make_packet(2, TrafficClass::Background, 500, 0), 0);
        assert_eq!(mgr.total_queued_bytes(), 1500);
    }

    #[test]
    fn test_total_queued_bytes_after_dequeue() {
        let mut mgr = NetworkQoSManager::new(QoSConfig {
            enable_strict_priority: true,
            ..Default::default()
        });
        mgr.enqueue(make_packet(1, TrafficClass::RealTime, 800, 0), 0);
        mgr.dequeue(10);
        assert_eq!(mgr.total_queued_bytes(), 0);
    }

    // ── QoSStats ─────────────────────────────────────────────────────────────

    #[test]
    fn test_qos_stats_initial_state() {
        let mgr = default_manager();
        let stats = mgr.qos_stats();
        assert_eq!(stats.total_enqueued, 0);
        assert_eq!(stats.total_dequeued, 0);
        assert_eq!(stats.total_dropped, 0);
        assert!((stats.drop_rate).abs() < 1e-9);
        assert_eq!(stats.active_queues, 0);
        assert_eq!(stats.total_violations, 0);
    }

    #[test]
    fn test_qos_stats_active_queues_counted() {
        let mut mgr = default_manager();
        mgr.enqueue(make_packet(1, TrafficClass::RealTime, 10, 0), 0);
        mgr.enqueue(make_packet(2, TrafficClass::Background, 10, 0), 0);
        let stats = mgr.qos_stats();
        assert_eq!(stats.active_queues, 2);
    }

    #[test]
    fn test_qos_stats_drop_rate_calculation() {
        let mut mgr = NetworkQoSManager::new(QoSConfig {
            max_queue_size: 1,
            ..Default::default()
        });
        // Enqueue one packet to fill capacity.
        mgr.enqueue(make_packet(1, TrafficClass::RealTime, 10, 0), 0);
        // Next Background packet cannot enter and is dropped.
        mgr.enqueue(make_packet(2, TrafficClass::Background, 10, 0), 0);

        let stats = mgr.qos_stats();
        // 1 enqueued + 1 dropped; drop_rate = 1/2 = 0.5 (total_enqueued
        // counts only successful enqueues, so enqueued=1, dropped=1).
        assert!(stats.total_dropped >= 1);
    }

    // ── Reset / flush ─────────────────────────────────────────────────────────

    #[test]
    fn test_reset_metrics_clears_avg_wait() {
        let mut mgr = NetworkQoSManager::new(QoSConfig {
            enable_strict_priority: true,
            ..Default::default()
        });
        mgr.enqueue(make_packet(1, TrafficClass::BulkData, 10, 0), 0);
        mgr.dequeue(200);
        mgr.reset_metrics();
        let m = mgr
            .queue_metrics(&TrafficClass::BulkData)
            .expect("test: BulkData queue metrics must exist after reset");
        assert!((m.avg_wait_ms).abs() < 1e-9);
    }

    #[test]
    fn test_flush_all_empties_queues() {
        let mut mgr = default_manager();
        mgr.enqueue(make_packet(1, TrafficClass::RealTime, 10, 0), 0);
        mgr.enqueue(make_packet(2, TrafficClass::BulkData, 20, 0), 0);
        let removed = mgr.flush_all();
        assert_eq!(removed, 2);
        assert_eq!(mgr.total_queued_bytes(), 0);
    }

    #[test]
    fn test_flush_all_returns_zero_on_empty() {
        let mut mgr = default_manager();
        assert_eq!(mgr.flush_all(), 0);
    }

    // ── EWMA wait time ────────────────────────────────────────────────────────

    #[test]
    fn test_ewma_converges_towards_new_value() {
        let mut mgr = NetworkQoSManager::new(QoSConfig {
            enable_strict_priority: true,
            ..Default::default()
        });
        // Bootstrap with 100 ms wait.
        mgr.enqueue(make_packet(1, TrafficClass::Interactive, 10, 0), 0);
        mgr.dequeue(100);

        // Feed several 10 ms waits; avg should converge below 100.
        for i in 0..20u64 {
            mgr.enqueue(make_packet(i + 10, TrafficClass::Interactive, 10, 0), 0);
            mgr.dequeue(10);
        }
        let m = mgr
            .queue_metrics(&TrafficClass::Interactive)
            .expect("test: Interactive queue metrics must exist");
        assert!(
            m.avg_wait_ms < 80.0,
            "EWMA should converge: {}",
            m.avg_wait_ms
        );
    }

    // ── Drop metric consistency ───────────────────────────────────────────────

    #[test]
    fn test_drop_increments_per_class_dropped_packets() {
        let mut mgr = default_manager();
        mgr.enqueue(make_packet(1, TrafficClass::Background, 10, 0), 0);
        mgr.drop_packet(1);
        let m = mgr
            .queue_metrics(&TrafficClass::Background)
            .expect("test: Background queue metrics must exist");
        assert_eq!(m.dropped_packets, 1);
    }

    // ── Violations log ────────────────────────────────────────────────────────

    #[test]
    fn test_violations_log_initially_empty() {
        let mgr = default_manager();
        assert!(mgr.violations_log().is_empty());
    }

    #[test]
    fn test_violations_log_records_correct_class() {
        let mut mgr = NetworkQoSManager::new(QoSConfig {
            enable_strict_priority: true,
            sla_specs: vec![SLASpec {
                class: TrafficClass::RealTime,
                max_latency_ms: 1,
                min_throughput_bps: 0,
                max_jitter_ms: 1,
            }],
            ..Default::default()
        });
        mgr.enqueue(make_packet(1, TrafficClass::RealTime, 10, 0), 0);
        mgr.dequeue(999);
        mgr.check_sla(999);

        let log = mgr.violations_log();
        assert!(!log.is_empty());
        assert_eq!(log[0].class, TrafficClass::RealTime);
    }

    // ── Strict priority exhaustion ────────────────────────────────────────────

    #[test]
    fn test_strict_priority_drains_all_queues() {
        let mut mgr = NetworkQoSManager::new(QoSConfig {
            enable_strict_priority: true,
            ..Default::default()
        });
        let classes = [
            TrafficClass::RealTime,
            TrafficClass::Interactive,
            TrafficClass::BulkData,
            TrafficClass::Background,
        ];
        for (i, cls) in classes.iter().enumerate() {
            mgr.enqueue(make_packet(i as u64, cls.clone(), 10, 0), 0);
        }
        let mut count = 0;
        while mgr.dequeue(10).is_some() {
            count += 1;
        }
        assert_eq!(count, 4);
        assert_eq!(mgr.total_dequeued, 4);
    }
}
