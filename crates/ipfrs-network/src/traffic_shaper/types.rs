//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque};

use super::functions::{tokens_available, xorshift_f64};
use super::types_3::ShaperConfig;

/// Fair-queuing traffic shaper that manages per-peer token buckets.
#[derive(Debug)]
pub struct PeerTrafficShaper {
    /// Per-peer token buckets keyed by peer_id.
    pub buckets: HashMap<String, PeerTokenBucket>,
    /// Default burst cap applied when creating new buckets.
    pub default_max_burst: u64,
    /// Default rate (bytes per tick) applied when creating new buckets.
    pub default_rate_bps: u64,
}
impl PeerTrafficShaper {
    /// Construct a new shaper with the given defaults.
    pub fn new(default_max_burst: u64, default_rate_bps: u64) -> Self {
        Self {
            buckets: HashMap::new(),
            default_max_burst,
            default_rate_bps,
        }
    }
    /// Enqueue a traffic token for later sending.
    pub fn enqueue(&mut self, token: TrafficToken) {
        let max_burst = self.default_max_burst;
        let rate_bps = self.default_rate_bps;
        let bucket = self
            .buckets
            .entry(token.peer_id.clone())
            .or_insert_with(|| PeerTokenBucket::new(token.peer_id.clone(), max_burst, rate_bps));
        if bucket.tokens == 0 && bucket.queued.is_empty() {
            bucket.dropped_bytes += token.bytes;
            return;
        }
        bucket.queued.push(token);
        bucket
            .queued
            .sort_by_key(|a| std::cmp::Reverse(a.class.weight()));
    }
    /// Refill all peer buckets by their individual `rate_bps`.
    pub fn tick(&mut self) {
        for bucket in self.buckets.values_mut() {
            let rate = bucket.rate_bps;
            bucket.refill(rate);
        }
    }
    /// Round-robin dequeue: take at most one token from each bucket.
    pub fn dequeue_all(&mut self) -> Vec<TrafficToken> {
        let mut out = Vec::with_capacity(self.buckets.len());
        for bucket in self.buckets.values_mut() {
            if let Some(token) = bucket.dequeue() {
                out.push(token);
            }
        }
        out
    }
    /// Snapshot of aggregate statistics.
    pub fn stats(&self) -> PeerShaperStats {
        let mut stats = PeerShaperStats {
            active_peers: self.buckets.len(),
            ..PeerShaperStats::default()
        };
        for bucket in self.buckets.values() {
            stats.total_sent_bytes += bucket.sent_bytes;
            stats.total_dropped_bytes += bucket.dropped_bytes;
            stats.total_queued += bucket.queued.len();
        }
        stats
    }
    /// Remove a peer bucket. Returns `true` if the peer existed.
    pub fn remove_peer(&mut self, peer_id: &str) -> bool {
        self.buckets.remove(peer_id).is_some()
    }
    /// Borrow the bucket for a specific peer if it exists.
    pub fn peer_bucket(&self, peer_id: &str) -> Option<&PeerTokenBucket> {
        self.buckets.get(peer_id)
    }
}
/// Events emitted by [`TrafficShaper`].
#[derive(Clone, Debug)]
pub enum ShaperEvent {
    /// Entry with given id was successfully enqueued.
    PacketEnqueued(u64),
    /// Entry with given id was successfully dequeued.
    PacketDequeued(u64),
    /// Entry was dropped.
    PacketDropped {
        /// Entry id.
        id: u64,
        /// Human-readable reason.
        reason: String,
    },
    /// Enqueue was blocked by the current rate limit.
    RateLimitHit,
    /// Enqueue exceeded burst allowance.
    BurstAllowanceExceeded,
}
/// Classification for flows managed by [`TrafficShaper`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TrafficClass {
    /// Highest priority — real-time flows (e.g. consensus, VoIP).
    RealTime,
    /// High priority — interactive flows (e.g. RPC, interactive queries).
    Interactive,
    /// Normal priority — large block/file transfers.
    BulkData,
    /// Low priority — background syncing, crawling.
    Background,
    /// Management and control-plane traffic.
    Management,
}
impl TrafficClass {
    /// Numeric priority (higher = more urgent).
    pub fn priority(&self) -> u8 {
        match self {
            TrafficClass::RealTime => 5,
            TrafficClass::Interactive => 4,
            TrafficClass::Management => 3,
            TrafficClass::BulkData => 2,
            TrafficClass::Background => 1,
        }
    }
    /// Human-readable label used in stats.
    pub fn label(&self) -> &'static str {
        match self {
            TrafficClass::RealTime => "RealTime",
            TrafficClass::Interactive => "Interactive",
            TrafficClass::BulkData => "BulkData",
            TrafficClass::Background => "Background",
            TrafficClass::Management => "Management",
        }
    }
}
/// Aggregate statistics for the peer shaper.
#[derive(Clone, Debug, Default)]
pub struct PeerShaperStats {
    /// Total bytes sent across all peers.
    pub total_sent_bytes: u64,
    /// Total bytes dropped across all peers.
    pub total_dropped_bytes: u64,
    /// Number of peers currently tracked.
    pub active_peers: usize,
    /// Total tokens still waiting in all queues.
    pub total_queued: usize,
}
/// Policy applied when the queue reaches `max_queue_depth`.
#[derive(Clone, Debug)]
pub enum DropPolicy {
    /// Drop the newest arriving entry (tail drop).
    Tail,
    /// Drop the oldest entry to make room (head drop).
    Head,
    /// Random Early Detection — probabilistic drop before queue is full.
    RED {
        /// Queue depth below which no drops occur.
        min_threshold: usize,
        /// Queue depth at or above which all new arrivals are dropped.
        max_threshold: usize,
    },
}
/// Weighted fair queuing per-class state.
#[derive(Debug)]
pub(super) struct WfqClass {
    pub(super) weight: u32,
    /// Accumulated deficit counter for deficit round-robin.
    pub(super) deficit: u32,
    pub(super) queue: VecDeque<QueueEntry>,
}
/// Production-quality traffic shaper with pluggable queuing disciplines.
pub struct TrafficShaper {
    pub(super) config: ShaperConfig,
    /// FIFO / generic fallback queue.
    pub(super) fifo_queue: VecDeque<QueueEntry>,
    /// Per-priority-band queues (index 0 = highest priority).
    pub(super) priority_bands: Vec<VecDeque<QueueEntry>>,
    /// WFQ per-class state keyed by class label.
    pub(super) wfq_classes: Vec<WfqClass>,
    /// DiffServ DSCP → class lookup (built from config).
    pub(super) dscp_table: HashMap<u8, TrafficClass>,
    /// Per-class FIFO queues for DiffServ priority ordering.
    pub(super) diffserv_queues: HashMap<String, VecDeque<QueueEntry>>,
    /// Token bucket: available tokens in bytes.
    pub(super) tokens: u64,
    /// Timestamp of last token replenishment (microseconds).
    pub(super) last_replenish_us: u64,
    /// Leaky bucket: bytes currently in the bucket.
    pub(super) leaky_level: u64,
    /// Timestamp of last leaky drain (microseconds).
    pub(super) last_drain_us: u64,
    /// PRNG state for RED.
    pub(super) rng_state: u64,
    /// Accumulated statistics.
    pub(super) stats: ShaperStats,
    /// Per-class dequeue counter (label → count).
    pub(super) class_counters: HashMap<String, u64>,
    /// Event ring-buffer.
    pub(super) events: VecDeque<ShaperEvent>,
    /// EMA smoothing factor (alpha) for queue depth and latency.
    pub(super) ema_alpha: f64,
}
impl TrafficShaper {
    /// Create a new shaper from the given config.
    pub fn new(config: ShaperConfig) -> Result<Self, ShaperError> {
        config.validate()?;
        let num_bands = if let QueuingDiscipline::PriorityQueue(bands) = &config.discipline {
            *bands as usize
        } else {
            0
        };
        let wfq_classes =
            if let QueuingDiscipline::WeightedFairQueuing { weights } = &config.discipline {
                weights
                    .iter()
                    .map(|(_cls, w)| WfqClass {
                        weight: *w,
                        deficit: 0,
                        queue: VecDeque::new(),
                    })
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
        let dscp_table = if let QueuingDiscipline::DiffServ { dscp_map } = &config.discipline {
            dscp_map.iter().cloned().collect()
        } else {
            HashMap::new()
        };
        let tokens = if let QueuingDiscipline::TokenBucket { burst_bytes, .. } = &config.discipline
        {
            *burst_bytes
        } else {
            config.burst_allowance_bytes
        };
        let shaper = Self {
            config,
            fifo_queue: VecDeque::new(),
            priority_bands: (0..num_bands).map(|_| VecDeque::new()).collect(),
            wfq_classes,
            dscp_table,
            diffserv_queues: HashMap::new(),
            tokens,
            last_replenish_us: 0,
            leaky_level: 0,
            last_drain_us: 0,
            rng_state: 0xDEAD_BEEF_CAFE_1234,
            stats: ShaperStats::default(),
            class_counters: HashMap::new(),
            events: VecDeque::new(),
            ema_alpha: 0.1,
        };
        Ok(shaper)
    }
    /// Total number of entries across all internal queues.
    pub(super) fn total_queued(&self) -> usize {
        match &self.config.discipline {
            QueuingDiscipline::Fifo
            | QueuingDiscipline::TokenBucket { .. }
            | QueuingDiscipline::LeakyBucket { .. } => self.fifo_queue.len(),
            QueuingDiscipline::PriorityQueue(_) => {
                self.priority_bands.iter().map(|b| b.len()).sum()
            }
            QueuingDiscipline::WeightedFairQueuing { .. } => {
                self.wfq_classes.iter().map(|c| c.queue.len()).sum()
            }
            QueuingDiscipline::DiffServ { .. } => {
                self.diffserv_queues.values().map(|q| q.len()).sum()
            }
        }
    }
    /// Map a `TrafficClass` to a priority band index (0 = highest).
    pub(super) fn class_to_band(&self, class: &TrafficClass, bands: u8) -> usize {
        let bands = bands as usize;
        let prio = class.priority() as usize;
        let band = bands
            .saturating_sub(1)
            .saturating_sub((prio.saturating_sub(1)) * bands / 5);
        band.min(bands.saturating_sub(1))
    }
    /// Determine WFQ class index for the given `TrafficClass`.
    pub(super) fn wfq_index(&self, class: &TrafficClass) -> Option<usize> {
        if let QueuingDiscipline::WeightedFairQueuing { weights } = &self.config.discipline {
            weights.iter().position(|(cls, _)| cls == class)
        } else {
            None
        }
    }
    /// Apply the configured drop policy.
    ///
    /// For Tail/Head: triggers only when `current_depth >= max_queue_depth`.
    /// For RED: triggers based on RED thresholds independently of `max_queue_depth`,
    /// but also hard-drops when `current_depth >= max_queue_depth`.
    ///
    /// Returns `Ok(())` if the entry can be enqueued (possibly after dropping head),
    /// or `Err(ShaperError::QueueFull)` if the entry itself must be dropped.
    pub(super) fn apply_drop_policy(
        &mut self,
        entry: &QueueEntry,
        current_depth: usize,
    ) -> Result<(), ShaperError> {
        let max = self.config.max_queue_depth;
        match &self.config.drop_policy.clone() {
            DropPolicy::Tail => {
                if current_depth >= max {
                    self.stats.dropped += 1;
                    self.events.push_back(ShaperEvent::PacketDropped {
                        id: entry.id,
                        reason: "tail drop: queue full".to_string(),
                    });
                    return Err(ShaperError::QueueFull(current_depth));
                }
                Ok(())
            }
            DropPolicy::Head => {
                if current_depth >= max {
                    let dropped_id = self.drop_head();
                    if let Some(id) = dropped_id {
                        self.stats.dropped += 1;
                        self.events.push_back(ShaperEvent::PacketDropped {
                            id,
                            reason: "head drop: evicted oldest".to_string(),
                        });
                    }
                }
                Ok(())
            }
            DropPolicy::RED {
                min_threshold,
                max_threshold,
            } => {
                let min_t = *min_threshold;
                let max_t = *max_threshold;
                if current_depth >= max_t || current_depth >= max {
                    self.stats.dropped += 1;
                    self.events.push_back(ShaperEvent::PacketDropped {
                        id: entry.id,
                        reason: "RED: above max threshold".to_string(),
                    });
                    return Err(ShaperError::QueueFull(current_depth));
                }
                if current_depth >= min_t {
                    let p = (current_depth - min_t) as f64 / (max_t - min_t) as f64;
                    let r = xorshift_f64(&mut self.rng_state);
                    if r < p {
                        self.stats.dropped += 1;
                        self.events.push_back(ShaperEvent::PacketDropped {
                            id: entry.id,
                            reason: format!("RED: probabilistic drop (p={p:.3})"),
                        });
                        return Err(ShaperError::QueueFull(current_depth));
                    }
                }
                Ok(())
            }
        }
    }
    /// Drop the oldest (head) entry across all internal queues; return its id.
    pub(super) fn drop_head(&mut self) -> Option<u64> {
        match &self.config.discipline {
            QueuingDiscipline::Fifo
            | QueuingDiscipline::TokenBucket { .. }
            | QueuingDiscipline::LeakyBucket { .. } => self.fifo_queue.pop_front().map(|e| e.id),
            QueuingDiscipline::PriorityQueue(_) => {
                for band in self.priority_bands.iter_mut().rev() {
                    if let Some(e) = band.pop_front() {
                        return Some(e.id);
                    }
                }
                None
            }
            QueuingDiscipline::WeightedFairQueuing { .. } => {
                let n = self.wfq_classes.len();
                if n == 0 {
                    return None;
                }
                let mut min_idx = None;
                let mut min_weight = u32::MAX;
                for (i, cls) in self.wfq_classes.iter().enumerate() {
                    if !cls.queue.is_empty() && cls.weight < min_weight {
                        min_weight = cls.weight;
                        min_idx = Some(i);
                    }
                }
                min_idx.and_then(|i| self.wfq_classes[i].queue.pop_front().map(|e| e.id))
            }
            QueuingDiscipline::DiffServ { .. } => {
                let all_classes = [
                    TrafficClass::Background,
                    TrafficClass::BulkData,
                    TrafficClass::Management,
                    TrafficClass::Interactive,
                    TrafficClass::RealTime,
                ];
                for cls in &all_classes {
                    let label = cls.label().to_string();
                    if let Some(q) = self.diffserv_queues.get_mut(&label) {
                        if let Some(e) = q.pop_front() {
                            return Some(e.id);
                        }
                    }
                }
                None
            }
        }
    }
    /// Update EMA for queue depth and latency.
    pub(super) fn update_ema(&mut self, depth: usize, latency_us: f64) {
        let a = self.ema_alpha;
        self.stats.avg_queue_depth = a * depth as f64 + (1.0 - a) * self.stats.avg_queue_depth;
        self.stats.avg_latency_us = a * latency_us + (1.0 - a) * self.stats.avg_latency_us;
    }
    /// Record a class-level dequeue.
    pub(super) fn record_class_dequeue(&mut self, class: &TrafficClass) {
        let label = class.label().to_string();
        *self.class_counters.entry(label).or_insert(0) += 1;
    }
    /// Refresh `stats.class_stats` from `class_counters`.
    pub(super) fn refresh_class_stats(&mut self) {
        self.stats.class_stats = self
            .class_counters
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        self.stats.class_stats.sort_by(|a, b| a.0.cmp(&b.0));
    }
    /// Try to drain leaky bucket: update water level by elapsed time.
    /// Returns `true` if there is now room to enqueue `bytes`.
    pub(super) fn leaky_try_enqueue(
        &mut self,
        bytes: usize,
        current_ts: u64,
        drain_rate_bps: u64,
        bucket_bytes: u64,
    ) -> bool {
        let elapsed_us = current_ts.saturating_sub(self.last_drain_us);
        let drained = (drain_rate_bps as u128 * elapsed_us as u128 / 8_000_000) as u64;
        self.leaky_level = self.leaky_level.saturating_sub(drained);
        self.last_drain_us = current_ts;
        let new_level = self.leaky_level + bytes as u64;
        if new_level <= bucket_bytes {
            self.leaky_level = new_level;
            true
        } else {
            false
        }
    }
    /// Token bucket: consume `bytes` if tokens allow; return true on success.
    pub(super) fn token_try_consume(
        &mut self,
        bytes: usize,
        current_ts: u64,
        rate_bps: u64,
        burst_bytes: u64,
    ) -> bool {
        self.tokens = tokens_available(
            self.tokens,
            rate_bps,
            current_ts.saturating_sub(self.last_replenish_us),
            burst_bytes,
        );
        self.last_replenish_us = current_ts;
        if self.tokens >= bytes as u64 {
            self.tokens -= bytes as u64;
            true
        } else {
            false
        }
    }
    /// Enqueue an entry.
    ///
    /// Returns the `PacketEnqueued` event on success, or a `ShaperError` if the
    /// entry is dropped or the rate limit is exceeded.
    pub fn enqueue(
        &mut self,
        entry: QueueEntry,
        current_ts: u64,
    ) -> Result<ShaperEvent, ShaperError> {
        let id = entry.id;
        let bytes = entry.data_bytes;
        if let QueuingDiscipline::TokenBucket {
            rate_bps,
            burst_bytes,
        } = self.config.discipline.clone()
        {
            if !self.token_try_consume(bytes, current_ts, rate_bps, burst_bytes) {
                self.stats.dropped += 1;
                self.events.push_back(ShaperEvent::RateLimitHit);
                return Err(ShaperError::RateLimitExceeded(rate_bps));
            }
        }
        if let QueuingDiscipline::LeakyBucket {
            drain_rate_bps,
            bucket_bytes,
        } = self.config.discipline.clone()
        {
            if !self.leaky_try_enqueue(bytes, current_ts, drain_rate_bps, bucket_bytes) {
                self.stats.dropped += 1;
                self.events.push_back(ShaperEvent::RateLimitHit);
                return Err(ShaperError::RateLimitExceeded(drain_rate_bps));
            }
        }
        let current_depth = self.total_queued();
        self.apply_drop_policy(&entry, current_depth)?;
        match &self.config.discipline.clone() {
            QueuingDiscipline::Fifo
            | QueuingDiscipline::TokenBucket { .. }
            | QueuingDiscipline::LeakyBucket { .. } => {
                self.fifo_queue.push_back(entry);
            }
            QueuingDiscipline::PriorityQueue(bands) => {
                let band = self.class_to_band(&entry.class, *bands);
                while self.priority_bands.len() <= band {
                    self.priority_bands.push(VecDeque::new());
                }
                self.priority_bands[band].push_back(entry);
            }
            QueuingDiscipline::WeightedFairQueuing { .. } => {
                let idx = self.wfq_index(&entry.class);
                if let Some(i) = idx {
                    self.wfq_classes[i].queue.push_back(entry);
                } else {
                    if !self.wfq_classes.is_empty() {
                        let last = self.wfq_classes.len() - 1;
                        self.wfq_classes[last].queue.push_back(entry);
                    } else {
                        self.fifo_queue.push_back(entry);
                    }
                }
            }
            QueuingDiscipline::DiffServ { dscp_map: _ } => {
                let label = entry.class.label().to_string();
                self.diffserv_queues
                    .entry(label)
                    .or_default()
                    .push_back(entry);
            }
        }
        self.stats.enqueued += 1;
        let new_depth = self.total_queued();
        self.update_ema(new_depth, 0.0);
        self.events.push_back(ShaperEvent::PacketEnqueued(id));
        Ok(ShaperEvent::PacketEnqueued(id))
    }
    /// Dequeue the next entry according to the discipline.
    pub fn dequeue(&mut self, current_ts: u64) -> Option<QueueEntry> {
        let entry = match &self.config.discipline.clone() {
            QueuingDiscipline::Fifo => self.fifo_queue.pop_front(),
            QueuingDiscipline::TokenBucket {
                rate_bps,
                burst_bytes,
            } => {
                let elapsed = current_ts.saturating_sub(self.last_replenish_us);
                self.tokens = tokens_available(self.tokens, *rate_bps, elapsed, *burst_bytes);
                self.last_replenish_us = current_ts;
                self.fifo_queue.pop_front()
            }
            QueuingDiscipline::LeakyBucket {
                drain_rate_bps,
                bucket_bytes: _,
            } => {
                let elapsed_us = current_ts.saturating_sub(self.last_drain_us);
                let drained = (*drain_rate_bps as u128 * elapsed_us as u128 / 8_000_000) as u64;
                self.leaky_level = self.leaky_level.saturating_sub(drained);
                self.last_drain_us = current_ts;
                self.fifo_queue.pop_front()
            }
            QueuingDiscipline::PriorityQueue(_) => {
                let mut result = None;
                for band in self.priority_bands.iter_mut() {
                    if let Some(e) = band.pop_front() {
                        result = Some(e);
                        break;
                    }
                }
                result
            }
            QueuingDiscipline::WeightedFairQueuing { weights: _ } => {
                let n = self.wfq_classes.len();
                if n == 0 {
                    return None;
                }
                for cls in self.wfq_classes.iter_mut() {
                    if !cls.queue.is_empty() {
                        cls.deficit = cls.deficit.saturating_add(cls.weight);
                    }
                }
                let mut best_idx = None;
                let mut best_deficit = 0u32;
                for (i, cls) in self.wfq_classes.iter().enumerate() {
                    if !cls.queue.is_empty() && cls.deficit > best_deficit {
                        best_deficit = cls.deficit;
                        best_idx = Some(i);
                    }
                }
                if let Some(i) = best_idx {
                    let entry = self.wfq_classes[i].queue.pop_front();
                    if let Some(ref e) = entry {
                        let cost = e.data_bytes as u32;
                        self.wfq_classes[i].deficit =
                            self.wfq_classes[i].deficit.saturating_sub(cost);
                    }
                    entry
                } else {
                    None
                }
            }
            QueuingDiscipline::DiffServ { .. } => {
                let priority_order = [
                    TrafficClass::RealTime,
                    TrafficClass::Interactive,
                    TrafficClass::Management,
                    TrafficClass::BulkData,
                    TrafficClass::Background,
                ];
                let mut result = None;
                for cls in &priority_order {
                    let label = cls.label().to_string();
                    if let Some(q) = self.diffserv_queues.get_mut(&label) {
                        if let Some(e) = q.pop_front() {
                            result = Some(e);
                            break;
                        }
                    }
                }
                result
            }
        };
        if let Some(ref e) = entry {
            let latency_us = current_ts.saturating_sub(e.enqueued_at) as f64;
            let depth = self.total_queued();
            self.update_ema(depth, latency_us);
            self.stats.dequeued += 1;
            self.stats.bytes_shaped += e.data_bytes as u64;
            self.record_class_dequeue(&e.class);
            self.refresh_class_stats();
            self.events.push_back(ShaperEvent::PacketDequeued(e.id));
        }
        entry
    }
    /// Peek at the next entry that would be dequeued, without removing it.
    pub fn peek(&self) -> Option<&QueueEntry> {
        match &self.config.discipline {
            QueuingDiscipline::Fifo
            | QueuingDiscipline::TokenBucket { .. }
            | QueuingDiscipline::LeakyBucket { .. } => self.fifo_queue.front(),
            QueuingDiscipline::PriorityQueue(_) => {
                for band in &self.priority_bands {
                    if let Some(e) = band.front() {
                        return Some(e);
                    }
                }
                None
            }
            QueuingDiscipline::WeightedFairQueuing { .. } => {
                let mut best_idx = None;
                let mut best_deficit = 0u32;
                for (i, cls) in self.wfq_classes.iter().enumerate() {
                    if !cls.queue.is_empty() && cls.deficit >= best_deficit {
                        best_deficit = cls.deficit;
                        best_idx = Some(i);
                    }
                }
                best_idx.and_then(|i| self.wfq_classes[i].queue.front())
            }
            QueuingDiscipline::DiffServ { .. } => {
                let priority_order = [
                    TrafficClass::RealTime,
                    TrafficClass::Interactive,
                    TrafficClass::Management,
                    TrafficClass::BulkData,
                    TrafficClass::Background,
                ];
                for cls in &priority_order {
                    let label = cls.label().to_string();
                    if let Some(q) = self.diffserv_queues.get(&label) {
                        if let Some(e) = q.front() {
                            return Some(e);
                        }
                    }
                }
                None
            }
        }
    }
    /// Force-drain up to `n` entries of the given class from the queue.
    pub fn drain_class(&mut self, class: TrafficClass, n: usize) -> Vec<QueueEntry> {
        let mut out = Vec::new();
        match &self.config.discipline.clone() {
            QueuingDiscipline::Fifo
            | QueuingDiscipline::TokenBucket { .. }
            | QueuingDiscipline::LeakyBucket { .. } => {
                let mut remaining = VecDeque::new();
                while let Some(e) = self.fifo_queue.pop_front() {
                    if e.class == class && out.len() < n {
                        out.push(e);
                    } else {
                        remaining.push_back(e);
                    }
                }
                self.fifo_queue = remaining;
            }
            QueuingDiscipline::PriorityQueue(bands) => {
                let target_band = self.class_to_band(&class, *bands);
                let mut remaining = VecDeque::new();
                while let Some(e) = self.priority_bands[target_band].pop_front() {
                    if e.class == class && out.len() < n {
                        out.push(e);
                    } else {
                        remaining.push_back(e);
                    }
                }
                self.priority_bands[target_band] = remaining;
            }
            QueuingDiscipline::WeightedFairQueuing { .. } => {
                if let Some(idx) = self.wfq_index(&class) {
                    let mut remaining = VecDeque::new();
                    while let Some(e) = self.wfq_classes[idx].queue.pop_front() {
                        if out.len() < n {
                            out.push(e);
                        } else {
                            remaining.push_back(e);
                        }
                    }
                    self.wfq_classes[idx].queue = remaining;
                }
            }
            QueuingDiscipline::DiffServ { .. } => {
                let label = class.label().to_string();
                if let Some(q) = self.diffserv_queues.get_mut(&label) {
                    let mut remaining = VecDeque::new();
                    while let Some(e) = q.pop_front() {
                        if out.len() < n {
                            out.push(e);
                        } else {
                            remaining.push_back(e);
                        }
                    }
                    *q = remaining;
                }
            }
        }
        out
    }
    /// Update the rate limit (applies to TokenBucket and LeakyBucket disciplines).
    pub fn update_rate(&mut self, new_rate_bps: u64) -> Result<(), ShaperError> {
        match &mut self.config.discipline {
            QueuingDiscipline::TokenBucket { rate_bps, .. } => {
                *rate_bps = new_rate_bps;
                Ok(())
            }
            QueuingDiscipline::LeakyBucket { drain_rate_bps, .. } => {
                *drain_rate_bps = new_rate_bps;
                Ok(())
            }
            _ => Err(ShaperError::InvalidConfig(
                "update_rate only applies to TokenBucket and LeakyBucket disciplines".to_string(),
            )),
        }
    }
    /// Replenish token bucket based on elapsed time; return new token count.
    /// For non-token-bucket disciplines, replenishes `config.burst_allowance_bytes`.
    pub fn replenish_tokens(&mut self, current_ts: u64) -> u64 {
        let elapsed = current_ts.saturating_sub(self.last_replenish_us);
        let (rate, burst) = match &self.config.discipline {
            QueuingDiscipline::TokenBucket {
                rate_bps,
                burst_bytes,
            } => (*rate_bps, *burst_bytes),
            _ => (
                self.config.rate_limit_bps,
                self.config.burst_allowance_bytes,
            ),
        };
        self.tokens = tokens_available(self.tokens, rate, elapsed, burst);
        self.last_replenish_us = current_ts;
        self.tokens
    }
    /// Current total number of entries in the queue.
    pub fn queue_depth(&self) -> usize {
        self.total_queued()
    }
    /// Snapshot of current statistics.
    pub fn stats(&mut self) -> ShaperStats {
        self.refresh_class_stats();
        self.stats.clone()
    }
    /// Drain and return all accumulated events.
    pub fn drain_events(&mut self) -> Vec<ShaperEvent> {
        self.events.drain(..).collect()
    }
    /// Resolve a DSCP value to a `TrafficClass` using the DiffServ map.
    /// Returns `TrafficClass::BulkData` for unknown DSCP values.
    pub fn resolve_dscp(&self, dscp: u8) -> TrafficClass {
        self.dscp_table
            .get(&dscp)
            .cloned()
            .unwrap_or(TrafficClass::BulkData)
    }
}
/// Errors produced by [`TrafficShaper`].
#[derive(Clone, Debug, PartialEq)]
pub enum ShaperError {
    /// Queue is full at the given depth.
    QueueFull(usize),
    /// Rate limit exceeded (value = current rate bps).
    RateLimitExceeded(u64),
    /// Configuration is invalid.
    InvalidConfig(String),
    /// Requested entry id not found.
    EntryNotFound(u64),
}
/// Per-peer token bucket with a priority queue of pending tokens.
#[derive(Debug)]
pub struct PeerTokenBucket {
    /// Peer identifier this bucket belongs to.
    pub peer_id: String,
    /// Available send-budget tokens (1 token == 1 byte).
    pub tokens: u64,
    /// Maximum token accumulation cap (burst limit).
    pub max_burst: u64,
    /// Bytes added per `tick` call.
    pub rate_bps: u64,
    /// Pending tokens sorted by class weight descending.
    pub queued: Vec<TrafficToken>,
    /// Cumulative bytes successfully sent via `dequeue`.
    pub sent_bytes: u64,
    /// Cumulative bytes dropped.
    pub dropped_bytes: u64,
}
impl PeerTokenBucket {
    /// Create a new bucket with the supplied limits.
    pub fn new(peer_id: String, max_burst: u64, rate_bps: u64) -> Self {
        Self {
            peer_id,
            tokens: 0,
            max_burst,
            rate_bps,
            queued: Vec::new(),
            sent_bytes: 0,
            dropped_bytes: 0,
        }
    }
    /// Add `tick_bytes` to the token pool, capping at `max_burst`.
    pub fn refill(&mut self, tick_bytes: u64) {
        self.tokens = self.tokens.saturating_add(tick_bytes).min(self.max_burst);
    }
    /// Take the highest-priority token if the bucket has enough budget.
    pub fn dequeue(&mut self) -> Option<TrafficToken> {
        if self.queued.is_empty() {
            return None;
        }
        let required = self.queued[0].bytes;
        if self.tokens < required {
            return None;
        }
        self.tokens -= required;
        let token = self.queued.remove(0);
        self.sent_bytes += token.bytes;
        Some(token)
    }
}
/// A single unit of data waiting in a shaper queue.
#[derive(Clone, Debug)]
pub struct QueueEntry {
    /// Unique entry identifier.
    pub id: u64,
    /// Payload size in bytes.
    pub data_bytes: usize,
    /// Traffic classification.
    pub class: TrafficClass,
    /// Timestamp when the entry was enqueued (microseconds, monotonic).
    pub enqueued_at: u64,
    /// Source address / identifier.
    pub source: String,
    /// Destination address / identifier.
    pub destination: String,
}
/// A pending outbound traffic unit for one peer.
#[derive(Clone, Debug)]
pub struct TrafficToken {
    /// Destination peer identifier.
    pub peer_id: String,
    /// Traffic classification.
    pub class: PeerTrafficClass,
    /// Payload size in bytes.
    pub bytes: u64,
    /// Monotonic counter set at enqueue time.
    pub queued_at: u64,
}
/// Queuing discipline that governs enqueue/dequeue order and rate control.
#[derive(Clone, Debug)]
pub enum QueuingDiscipline {
    /// Simple first-in-first-out queue.
    Fifo,
    /// Strict-priority queue with `bands` priority levels.
    PriorityQueue(u8),
    /// Weighted fair queuing — dequeues round-robin weighted by class.
    WeightedFairQueuing {
        /// (TrafficClass, weight) pairs; higher weight → more bandwidth share.
        weights: Vec<(TrafficClass, u32)>,
    },
    /// Token bucket rate control.
    TokenBucket {
        /// Committed information rate in bits per second.
        rate_bps: u64,
        /// Maximum burst size in bytes.
        burst_bytes: u64,
    },
    /// Leaky bucket rate control.
    LeakyBucket {
        /// Drain rate in bits per second.
        drain_rate_bps: u64,
        /// Bucket capacity in bytes.
        bucket_bytes: u64,
    },
    /// DiffServ — map DSCP value to traffic class, then use priority ordering.
    DiffServ {
        /// (dscp_value, TrafficClass) mapping.
        dscp_map: Vec<(u8, TrafficClass)>,
    },
}
/// Traffic classification for outbound messages (legacy peer shaper).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PeerTrafficClass {
    /// Proof requests and consensus messages — highest priority.
    Critical,
    /// Block sends and receives — medium priority.
    DataTransfer,
    /// GC announcements, metrics — lowest priority.
    LowBackground,
}
impl PeerTrafficClass {
    /// Scheduling weight used for priority ordering.
    pub fn weight(&self) -> u32 {
        match self {
            PeerTrafficClass::Critical => 8,
            PeerTrafficClass::DataTransfer => 4,
            PeerTrafficClass::LowBackground => 1,
        }
    }
}
/// Aggregate statistics for a [`TrafficShaper`].
#[derive(Clone, Debug, Default)]
pub struct ShaperStats {
    /// Total entries successfully enqueued.
    pub enqueued: u64,
    /// Total entries successfully dequeued.
    pub dequeued: u64,
    /// Total entries dropped (queue-full or rate-limit).
    pub dropped: u64,
    /// Total bytes that passed through dequeue successfully.
    pub bytes_shaped: u64,
    /// Exponential moving average of queue depth sampled at each enqueue/dequeue.
    pub avg_queue_depth: f64,
    /// Exponential moving average of queueing latency in microseconds.
    pub avg_latency_us: f64,
    /// Per-class dequeue counts: (class_label, count).
    pub class_stats: Vec<(String, u64)>,
}
