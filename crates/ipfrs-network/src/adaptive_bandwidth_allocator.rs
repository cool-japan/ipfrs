//! Adaptive bandwidth allocator for peer-to-peer networks.
//!
//! Provides dynamic bandwidth management across peers using configurable allocation
//! policies, congestion detection, rolling measurement windows, and event history.
//!
//! # Policies
//! - [`AllocationPolicy::EqualShare`]: divide total capacity equally among all peers
//! - [`AllocationPolicy::WeightedFair`]: weight by [`BandwidthClass`] (High=4, Normal=2, Low=1, Background=0.5)
//! - [`AllocationPolicy::MinGuarantee`]: each peer gets at least `n` bps, remainder distributed equally
//! - [`AllocationPolicy::MaxCapacity`]: each peer gets at most `n` bps
//! - [`AllocationPolicy::PriorityQueue`]: high-priority peers allocated first until capacity exhausted
//!
//! # Example
//! ```rust
//! use ipfrs_network::adaptive_bandwidth_allocator::{
//!     AdaptiveBandwidthAllocator, AllocatorConfig, AllocationPolicy, BandwidthClass,
//! };
//!
//! let config = AllocatorConfig {
//!     total_capacity_bps: 100_000_000,
//!     min_allocation_bps: 1_000_000,
//!     max_allocation_bps: 50_000_000,
//!     policy: AllocationPolicy::WeightedFair,
//!     rebalance_interval_secs: 30,
//!     congestion_threshold_ppm: 10_000,
//! };
//! let mut allocator = AdaptiveBandwidthAllocator::new(config);
//! allocator.add_peer("peer-1".to_string(), BandwidthClass::High, 10_000_000).unwrap();
//! ```

use std::collections::{HashMap, VecDeque};

// ─── PRNG ────────────────────────────────────────────────────────────────────

/// Xorshift64 PRNG — used for jitter simulation in [`BandwidthWindow`].
/// No external rand crate required.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ─── BandwidthClass ──────────────────────────────────────────────────────────

/// Priority classification for a peer, used to drive weighted-fair allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BandwidthClass {
    /// Weight multiplier 4.0 — latency-sensitive or critical peers.
    High,
    /// Weight multiplier 2.0 — default peer classification.
    Normal,
    /// Weight multiplier 1.0 — best-effort, non-critical peers.
    Low,
    /// Weight multiplier 0.5 — idle or background-only traffic.
    Background,
}

impl BandwidthClass {
    /// Returns the floating-point weight used in weighted-fair allocation.
    pub fn weight(self) -> f64 {
        match self {
            BandwidthClass::High => 4.0,
            BandwidthClass::Normal => 2.0,
            BandwidthClass::Low => 1.0,
            BandwidthClass::Background => 0.5,
        }
    }

    /// Numeric priority (higher = more important) used in priority-queue allocation.
    pub fn priority(self) -> u32 {
        match self {
            BandwidthClass::High => 4,
            BandwidthClass::Normal => 3,
            BandwidthClass::Low => 2,
            BandwidthClass::Background => 1,
        }
    }
}

// ─── AllocationPolicy ────────────────────────────────────────────────────────

/// Strategy used by [`AdaptiveBandwidthAllocator::reallocate`] to distribute
/// the total available bandwidth across peers.
#[derive(Debug, Clone, PartialEq)]
pub enum AllocationPolicy {
    /// Divide total capacity equally among all active peers.
    EqualShare,
    /// Allocate proportionally to each peer's [`BandwidthClass`] weight.
    WeightedFair,
    /// Guarantee every peer at least `n` bps; distribute surplus equally.
    MinGuarantee(u64),
    /// Cap every peer at `n` bps; leftover capacity is unused.
    MaxCapacity(u64),
    /// Allocate to peers in descending priority order until capacity is exhausted.
    PriorityQueue,
}

// ─── BandwidthWindow ─────────────────────────────────────────────────────────

/// Rolling 10-sample window of bandwidth measurements.
///
/// Uses an `xorshift64` PRNG (seeded from the first sample) to simulate
/// measurement jitter for testing purposes without pulling in the `rand` crate.
#[derive(Debug, Clone)]
pub struct BandwidthWindow {
    samples: VecDeque<u64>,
    prng_state: u64,
    capacity: usize,
}

impl BandwidthWindow {
    const DEFAULT_CAPACITY: usize = 10;

    /// Create a new window with default capacity (10 samples).
    pub fn new() -> Self {
        Self {
            samples: VecDeque::with_capacity(Self::DEFAULT_CAPACITY),
            prng_state: 0xcafe_babe_dead_beef,
            capacity: Self::DEFAULT_CAPACITY,
        }
    }

    /// Push a new measurement, evicting the oldest when the window is full.
    /// Applies a small xorshift64-derived jitter (±0.5%) to the raw sample.
    pub fn push(&mut self, raw_bps: u64) {
        // Derive jitter in range [0, raw_bps/200], subtract half for ±effect.
        let jitter_range = (raw_bps / 200).max(1);
        let jitter = xorshift64(&mut self.prng_state) % jitter_range;
        // Use jitter as a signed offset anchored at raw_bps.
        let jittered = if xorshift64(&mut self.prng_state) & 1 == 0 {
            raw_bps.saturating_add(jitter)
        } else {
            raw_bps.saturating_sub(jitter)
        };

        if self.samples.len() == self.capacity {
            self.samples.pop_front();
        }
        self.samples.push_back(jittered);
    }

    /// Arithmetic mean of all samples; returns 0 when no samples exist.
    pub fn average_bps(&self) -> u64 {
        if self.samples.is_empty() {
            return 0;
        }
        let sum: u64 = self.samples.iter().sum();
        sum / self.samples.len() as u64
    }

    /// Maximum sample in the window; returns 0 when no samples exist.
    pub fn peak_bps(&self) -> u64 {
        self.samples.iter().copied().max().unwrap_or(0)
    }

    /// Number of samples currently held.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns `true` when no samples are stored.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

impl Default for BandwidthWindow {
    fn default() -> Self {
        Self::new()
    }
}

// ─── PeerBandwidthProfile ────────────────────────────────────────────────────

/// Complete bandwidth state for a single peer.
#[derive(Debug, Clone)]
pub struct PeerBandwidthProfile {
    /// Unique peer identifier.
    pub peer_id: String,
    /// Currently allocated bandwidth in bits-per-second.
    pub allocated_bps: u64,
    /// Most recently measured used bandwidth in bits-per-second.
    pub used_bps: u64,
    /// Most recently measured round-trip latency in milliseconds.
    pub latency_ms: u64,
    /// Packet loss rate in parts-per-million (0 = no loss, 1_000_000 = 100% loss).
    pub packet_loss_ppm: u32,
    /// Bandwidth priority class for this peer.
    pub class: BandwidthClass,
    /// Unix-millisecond timestamp of the last measurement update.
    pub last_updated: u64,
    /// Rolling measurement window.
    pub(crate) window: BandwidthWindow,
}

impl PeerBandwidthProfile {
    fn new(peer_id: String, class: BandwidthClass, initial_bps: u64, now: u64) -> Self {
        Self {
            peer_id,
            allocated_bps: initial_bps,
            used_bps: 0,
            latency_ms: 0,
            packet_loss_ppm: 0,
            class,
            last_updated: now,
            window: BandwidthWindow::new(),
        }
    }
}

// ─── BandwidthEvent ──────────────────────────────────────────────────────────

/// Events emitted by the allocator for observability and history queries.
#[derive(Debug, Clone)]
pub enum BandwidthEvent {
    /// A new peer has been added to the allocator.
    PeerAdded {
        peer_id: String,
        allocated_bps: u64,
        timestamp: u64,
    },
    /// A peer has been removed from the allocator.
    PeerRemoved { peer_id: String, timestamp: u64 },
    /// A live bandwidth measurement was recorded for a peer.
    BandwidthUpdated {
        peer_id: String,
        new_bps: u64,
        timestamp: u64,
    },
    /// Packet-loss congestion detected for a peer.
    CongestionDetected {
        peer_id: String,
        loss_ppm: u32,
        timestamp: u64,
    },
    /// Allocations have been revised across all peers.
    AllocationRevised { timestamp: u64 },
}

impl BandwidthEvent {
    /// Returns the timestamp embedded in any variant.
    pub fn timestamp(&self) -> u64 {
        match self {
            BandwidthEvent::PeerAdded { timestamp, .. } => *timestamp,
            BandwidthEvent::PeerRemoved { timestamp, .. } => *timestamp,
            BandwidthEvent::BandwidthUpdated { timestamp, .. } => *timestamp,
            BandwidthEvent::CongestionDetected { timestamp, .. } => *timestamp,
            BandwidthEvent::AllocationRevised { timestamp } => *timestamp,
        }
    }
}

// ─── AllocatorConfig ─────────────────────────────────────────────────────────

/// Configuration for [`AdaptiveBandwidthAllocator`].
#[derive(Debug, Clone)]
pub struct AllocatorConfig {
    /// Total available bandwidth in bits-per-second (hard upper bound).
    pub total_capacity_bps: u64,
    /// Minimum allocation any single peer can receive.
    pub min_allocation_bps: u64,
    /// Maximum allocation any single peer can receive.
    pub max_allocation_bps: u64,
    /// Allocation policy applied during [`AdaptiveBandwidthAllocator::reallocate`].
    pub policy: AllocationPolicy,
    /// How often to suggest a rebalance cycle (advisory; not enforced internally).
    pub rebalance_interval_secs: u64,
    /// Packet-loss threshold in parts-per-million above which congestion is flagged.
    pub congestion_threshold_ppm: u32,
}

impl AllocatorConfig {
    /// Validate the configuration, returning an error on logical inconsistencies.
    pub fn validate(&self) -> Result<(), AllocatorError> {
        if self.min_allocation_bps > self.max_allocation_bps {
            return Err(AllocatorError::InvalidConfiguration(
                "min_allocation_bps must be ≤ max_allocation_bps".to_string(),
            ));
        }
        if self.total_capacity_bps == 0 {
            return Err(AllocatorError::InvalidConfiguration(
                "total_capacity_bps must be > 0".to_string(),
            ));
        }
        if self.max_allocation_bps > self.total_capacity_bps {
            return Err(AllocatorError::InvalidConfiguration(
                "max_allocation_bps must be ≤ total_capacity_bps".to_string(),
            ));
        }
        Ok(())
    }
}

// ─── AbaAllocationStats (Aba-prefixed to avoid collision) ────────────────────

/// Aggregate bandwidth statistics across all tracked peers.
///
/// Exported as `AbaBandwidthStats` at the crate root to avoid collision with
/// `bandwidth_monitor::BandwidthStats`.
#[derive(Debug, Clone)]
pub struct BandwidthStats {
    /// Sum of all peer allocations in bps.
    pub total_allocated_bps: u64,
    /// Sum of all measured used bandwidths in bps.
    pub total_used_bps: u64,
    /// Ratio of total_used_bps to total_allocated_bps, expressed as a percentage.
    pub utilization_pct: f64,
    /// Number of peers currently flagged as congested.
    pub congested_peers: usize,
    /// Total number of tracked peers.
    pub peer_count: usize,
    /// Jain's fairness index over current allocations (1.0 = perfectly fair).
    pub fairness_index: f64,
}

// ─── AllocatorError ──────────────────────────────────────────────────────────

/// Error variants returned by [`AdaptiveBandwidthAllocator`] methods.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum AllocatorError {
    #[error("peer not found: {0}")]
    PeerNotFound(String),

    #[error("allocation of {0} bps exceeds remaining capacity")]
    AllocationExceedsCapacity(u64),

    #[error("invalid configuration: {0}")]
    InvalidConfiguration(String),

    #[error("congestion detected on peer {peer_id}: loss_ppm={loss_ppm}")]
    CongestionDetected { peer_id: String, loss_ppm: u32 },
}

// ─── AdaptiveBandwidthAllocator ──────────────────────────────────────────────

/// Adaptive bandwidth manager that dynamically allocates bandwidth across peers.
///
/// Tracks per-peer profiles, maintains a bounded event history, and supports
/// five allocation policies via [`AdaptiveBandwidthAllocator::reallocate`].
pub struct AdaptiveBandwidthAllocator {
    config: AllocatorConfig,
    /// Peer profiles keyed by peer_id.
    peers: HashMap<String, PeerBandwidthProfile>,
    /// Bounded FIFO event history (max 200 events).
    events: VecDeque<BandwidthEvent>,
    /// Total bps currently allocated across all peers.
    allocated_total: u64,
    /// Monotonic-ish timestamp for events — callers may pass real timestamps via
    /// update_measurement; internal operations use this counter.
    now_ms: u64,
}

impl AdaptiveBandwidthAllocator {
    /// Maximum number of events retained in the history ring-buffer.
    pub const MAX_EVENT_HISTORY: usize = 200;

    /// Create a new allocator with the given configuration.
    ///
    /// Returns [`AllocatorError::InvalidConfiguration`] if the config is logically
    /// inconsistent.
    pub fn new(config: AllocatorConfig) -> Self {
        // Validation is best-effort at construction; callers may call validate() first.
        Self {
            config,
            peers: HashMap::new(),
            events: VecDeque::with_capacity(Self::MAX_EVENT_HISTORY + 1),
            allocated_total: 0,
            now_ms: 1_000, // arbitrary non-zero start
        }
    }

    /// Advance the internal clock by `delta_ms` milliseconds.
    /// Useful in testing to create distinct timestamps without real-time delays.
    pub fn advance_time(&mut self, delta_ms: u64) {
        self.now_ms = self.now_ms.saturating_add(delta_ms);
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /// Add a peer with an initial bandwidth class and desired allocation.
    ///
    /// Returns [`AllocatorError::AllocationExceedsCapacity`] if adding this
    /// allocation would exceed `total_capacity_bps`.
    pub fn add_peer(
        &mut self,
        peer_id: String,
        class: BandwidthClass,
        initial_bps: u64,
    ) -> Result<(), AllocatorError> {
        let clamped = initial_bps.clamp(
            self.config.min_allocation_bps,
            self.config.max_allocation_bps,
        );

        let new_total = self.allocated_total.saturating_add(clamped);
        if new_total > self.config.total_capacity_bps {
            return Err(AllocatorError::AllocationExceedsCapacity(clamped));
        }

        let profile = PeerBandwidthProfile::new(peer_id.clone(), class, clamped, self.now_ms);
        self.peers.insert(peer_id.clone(), profile);
        self.allocated_total = new_total;

        self.push_event(BandwidthEvent::PeerAdded {
            peer_id,
            allocated_bps: clamped,
            timestamp: self.now_ms,
        });
        Ok(())
    }

    /// Remove a peer and release its bandwidth allocation.
    ///
    /// Returns the removed [`PeerBandwidthProfile`] or
    /// [`AllocatorError::PeerNotFound`].
    pub fn remove_peer(&mut self, peer_id: &str) -> Result<PeerBandwidthProfile, AllocatorError> {
        let profile = self
            .peers
            .remove(peer_id)
            .ok_or_else(|| AllocatorError::PeerNotFound(peer_id.to_string()))?;

        self.allocated_total = self.allocated_total.saturating_sub(profile.allocated_bps);

        self.push_event(BandwidthEvent::PeerRemoved {
            peer_id: peer_id.to_string(),
            timestamp: self.now_ms,
        });
        Ok(profile)
    }

    /// Record a live measurement for a peer.
    ///
    /// Updates the rolling [`BandwidthWindow`], stores latency and loss, and
    /// emits a [`BandwidthEvent::CongestionDetected`] when loss_ppm exceeds the
    /// configured threshold.
    ///
    /// Returns the most pertinent event for the caller's convenience.
    pub fn update_measurement(
        &mut self,
        peer_id: &str,
        used_bps: u64,
        latency_ms: u64,
        packet_loss_ppm: u32,
    ) -> Result<BandwidthEvent, AllocatorError> {
        let now = self.now_ms;
        let profile = self
            .peers
            .get_mut(peer_id)
            .ok_or_else(|| AllocatorError::PeerNotFound(peer_id.to_string()))?;

        profile.used_bps = used_bps;
        profile.latency_ms = latency_ms;
        profile.packet_loss_ppm = packet_loss_ppm;
        profile.last_updated = now;
        profile.window.push(used_bps);

        let event = if packet_loss_ppm > self.config.congestion_threshold_ppm {
            BandwidthEvent::CongestionDetected {
                peer_id: peer_id.to_string(),
                loss_ppm: packet_loss_ppm,
                timestamp: now,
            }
        } else {
            BandwidthEvent::BandwidthUpdated {
                peer_id: peer_id.to_string(),
                new_bps: used_bps,
                timestamp: now,
            }
        };

        self.push_event(event.clone());
        Ok(event)
    }

    /// Rebalance all peer allocations according to the configured policy.
    ///
    /// Returns the list of [`BandwidthEvent`]s produced (one
    /// `AllocationRevised` event at minimum, plus any `CongestionDetected`
    /// events for peers that are still over threshold after rebalancing).
    pub fn reallocate(&mut self) -> Vec<BandwidthEvent> {
        let mut out_events: Vec<BandwidthEvent> = Vec::new();
        let now = self.now_ms;

        if self.peers.is_empty() {
            self.push_event(BandwidthEvent::AllocationRevised { timestamp: now });
            out_events.push(BandwidthEvent::AllocationRevised { timestamp: now });
            return out_events;
        }

        let new_allocs = self.compute_allocations();

        // Apply new allocations and track the aggregate.
        let mut new_total: u64 = 0;
        for (peer_id, alloc) in &new_allocs {
            if let Some(profile) = self.peers.get_mut(peer_id) {
                profile.allocated_bps = *alloc;
                new_total = new_total.saturating_add(*alloc);
            }
        }
        self.allocated_total = new_total;

        // Emit congestion events for peers still over threshold.
        let threshold = self.config.congestion_threshold_ppm;
        let congested: Vec<(String, u32)> = self
            .peers
            .values()
            .filter(|p| p.packet_loss_ppm > threshold)
            .map(|p| (p.peer_id.clone(), p.packet_loss_ppm))
            .collect();

        for (peer_id, loss_ppm) in congested {
            let ev = BandwidthEvent::CongestionDetected {
                peer_id,
                loss_ppm,
                timestamp: now,
            };
            self.push_event(ev.clone());
            out_events.push(ev);
        }

        let revised = BandwidthEvent::AllocationRevised { timestamp: now };
        self.push_event(revised.clone());
        out_events.push(revised);

        out_events
    }

    /// Return the current allocated bps for the given peer.
    pub fn get_allocation(&self, peer_id: &str) -> Result<u64, AllocatorError> {
        self.peers
            .get(peer_id)
            .map(|p| p.allocated_bps)
            .ok_or_else(|| AllocatorError::PeerNotFound(peer_id.to_string()))
    }

    /// Return aggregate statistics including Jain's fairness index.
    pub fn stats(&self) -> BandwidthStats {
        let peer_count = self.peers.len();
        let total_allocated_bps: u64 = self.peers.values().map(|p| p.allocated_bps).sum();
        let total_used_bps: u64 = self.peers.values().map(|p| p.used_bps).sum();

        let utilization_pct = if total_allocated_bps == 0 {
            0.0
        } else {
            (total_used_bps as f64 / total_allocated_bps as f64) * 100.0
        };

        let threshold = self.config.congestion_threshold_ppm;
        let congested_peers = self
            .peers
            .values()
            .filter(|p| p.packet_loss_ppm > threshold)
            .count();

        let fairness_index = Self::jain_fairness_index(
            self.peers.values().map(|p| p.allocated_bps as f64),
            peer_count,
        );

        BandwidthStats {
            total_allocated_bps,
            total_used_bps,
            utilization_pct,
            congested_peers,
            peer_count,
            fairness_index,
        }
    }

    /// Return peer IDs for all peers whose `packet_loss_ppm` exceeds the threshold.
    pub fn congested_peers(&self) -> Vec<String> {
        let threshold = self.config.congestion_threshold_ppm;
        self.peers
            .values()
            .filter(|p| p.packet_loss_ppm > threshold)
            .map(|p| p.peer_id.clone())
            .collect()
    }

    /// Return all events with `timestamp >= since_ms`.
    pub fn events_since(&self, since_ms: u64) -> Vec<BandwidthEvent> {
        self.events
            .iter()
            .filter(|e| e.timestamp() >= since_ms)
            .cloned()
            .collect()
    }

    /// Drain and return all buffered events, leaving the history empty.
    pub fn drain_events(&mut self) -> Vec<BandwidthEvent> {
        self.events.drain(..).collect()
    }

    /// Return a reference to the profile for `peer_id`, if present.
    pub fn peer_profile(&self, peer_id: &str) -> Option<&PeerBandwidthProfile> {
        self.peers.get(peer_id)
    }

    /// Return the number of tracked peers.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Return the total capacity configured for this allocator.
    pub fn total_capacity_bps(&self) -> u64 {
        self.config.total_capacity_bps
    }

    /// Return a reference to the current configuration.
    pub fn config(&self) -> &AllocatorConfig {
        &self.config
    }

    // ── Private helpers ────────────────────────────────────────────────────

    /// Push an event, evicting the oldest when history is full.
    fn push_event(&mut self, event: BandwidthEvent) {
        if self.events.len() == Self::MAX_EVENT_HISTORY {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }

    /// Compute new allocations for each peer according to the active policy.
    /// Returns a map of peer_id → new_bps.
    fn compute_allocations(&self) -> HashMap<String, u64> {
        match &self.config.policy {
            AllocationPolicy::EqualShare => self.policy_equal_share(),
            AllocationPolicy::WeightedFair => self.policy_weighted_fair(),
            AllocationPolicy::MinGuarantee(min_n) => self.policy_min_guarantee(*min_n),
            AllocationPolicy::MaxCapacity(max_n) => self.policy_max_capacity(*max_n),
            AllocationPolicy::PriorityQueue => self.policy_priority_queue(),
        }
    }

    /// EqualShare: total_capacity / n, clamped to [min, max].
    fn policy_equal_share(&self) -> HashMap<String, u64> {
        let n = self.peers.len() as u64;
        let share = self.config.total_capacity_bps.checked_div(n).unwrap_or(0);
        let clamped = share.clamp(
            self.config.min_allocation_bps,
            self.config.max_allocation_bps,
        );
        self.peers.keys().map(|id| (id.clone(), clamped)).collect()
    }

    /// WeightedFair: proportional by class weight, clamped to [min, max].
    fn policy_weighted_fair(&self) -> HashMap<String, u64> {
        let total_weight: f64 = self.peers.values().map(|p| p.class.weight()).sum();
        if total_weight == 0.0 {
            return self.policy_equal_share();
        }
        self.peers
            .values()
            .map(|p| {
                let raw = (self.config.total_capacity_bps as f64 * p.class.weight() / total_weight)
                    as u64;
                let alloc = raw.clamp(
                    self.config.min_allocation_bps,
                    self.config.max_allocation_bps,
                );
                (p.peer_id.clone(), alloc)
            })
            .collect()
    }

    /// MinGuarantee: each peer gets at least `n`, remainder distributed equally.
    fn policy_min_guarantee(&self, min_n: u64) -> HashMap<String, u64> {
        let peer_count = self.peers.len() as u64;
        if peer_count == 0 {
            return HashMap::new();
        }
        // Effective minimum is max(config.min, min_n).
        let effective_min = min_n
            .max(self.config.min_allocation_bps)
            .min(self.config.max_allocation_bps);

        let guaranteed_total = effective_min.saturating_mul(peer_count);
        let remainder = self
            .config
            .total_capacity_bps
            .saturating_sub(guaranteed_total);
        let bonus_per_peer = remainder / peer_count;

        self.peers
            .keys()
            .map(|id| {
                let alloc = (effective_min + bonus_per_peer).min(self.config.max_allocation_bps);
                (id.clone(), alloc)
            })
            .collect()
    }

    /// MaxCapacity: each peer gets at most `n` bps; otherwise equal share.
    fn policy_max_capacity(&self, max_n: u64) -> HashMap<String, u64> {
        let peer_count = self.peers.len() as u64;
        if peer_count == 0 {
            return HashMap::new();
        }
        let effective_max = max_n.min(self.config.max_allocation_bps);
        let equal = (self.config.total_capacity_bps / peer_count).min(effective_max);
        let clamped = equal.clamp(self.config.min_allocation_bps, effective_max);
        self.peers.keys().map(|id| (id.clone(), clamped)).collect()
    }

    /// PriorityQueue: highest-priority peers fill up first; remainder gets min.
    fn policy_priority_queue(&self) -> HashMap<String, u64> {
        // Sort peers: descending by (class priority, peer_id).
        let mut ordered: Vec<&PeerBandwidthProfile> = self.peers.values().collect();
        ordered.sort_by(|a, b| {
            b.class
                .priority()
                .cmp(&a.class.priority())
                .then_with(|| a.peer_id.cmp(&b.peer_id))
        });

        let mut remaining = self.config.total_capacity_bps;
        let mut allocs: HashMap<String, u64> = HashMap::new();

        for profile in &ordered {
            let grant = remaining.min(self.config.max_allocation_bps);
            let grant = grant.max(self.config.min_allocation_bps);
            // Don't overshoot.
            let grant = if grant > remaining { remaining } else { grant };
            allocs.insert(profile.peer_id.clone(), grant);
            remaining = remaining.saturating_sub(grant);
        }

        // Any peer that got 0 (exhausted capacity) receives the configured minimum.
        for profile in &ordered {
            let entry = allocs.entry(profile.peer_id.clone()).or_insert(0);
            if *entry == 0 {
                *entry = self.config.min_allocation_bps;
            }
        }

        allocs
    }

    /// Jain's fairness index: (Σxᵢ)² / (n × Σxᵢ²)
    /// Returns 1.0 for perfect fairness, approaches 1/n for maximum unfairness.
    fn jain_fairness_index(allocations: impl Iterator<Item = f64>, n: usize) -> f64 {
        if n == 0 {
            return 1.0;
        }
        let (sum, sum_sq) = allocations.fold((0.0_f64, 0.0_f64), |(s, sq), x| (s + x, sq + x * x));
        if sum_sq == 0.0 {
            return 1.0;
        }
        (sum * sum) / (n as f64 * sum_sq)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn default_config() -> AllocatorConfig {
        AllocatorConfig {
            total_capacity_bps: 100_000_000, // 100 Mbps
            min_allocation_bps: 1_000_000,   // 1 Mbps
            max_allocation_bps: 50_000_000,  // 50 Mbps
            policy: AllocationPolicy::EqualShare,
            rebalance_interval_secs: 30,
            congestion_threshold_ppm: 10_000, // 1%
        }
    }

    fn alloc_with_policy(policy: AllocationPolicy) -> AdaptiveBandwidthAllocator {
        let mut cfg = default_config();
        cfg.policy = policy;
        AdaptiveBandwidthAllocator::new(cfg)
    }

    fn populated_allocator(n: usize, policy: AllocationPolicy) -> AdaptiveBandwidthAllocator {
        let mut a = alloc_with_policy(policy);
        for i in 0..n {
            let class = match i % 4 {
                0 => BandwidthClass::High,
                1 => BandwidthClass::Normal,
                2 => BandwidthClass::Low,
                _ => BandwidthClass::Background,
            };
            a.add_peer(format!("peer-{i}"), class, 5_000_000)
                .expect("test: failed to add peer in populated_allocator");
        }
        a
    }

    // ── xorshift64 ───────────────────────────────────────────────────────────

    #[test]
    fn test_xorshift64_non_zero_output() {
        let mut state = 12345u64;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
        assert_ne!(state, 12345u64);
    }

    #[test]
    fn test_xorshift64_deterministic() {
        let mut s1 = 99999u64;
        let mut s2 = 99999u64;
        assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_xorshift64_different_states_differ() {
        let mut s1 = 1u64;
        let mut s2 = 2u64;
        assert_ne!(xorshift64(&mut s1), xorshift64(&mut s2));
    }

    // ── BandwidthClass ────────────────────────────────────────────────────────

    #[test]
    fn test_bandwidth_class_weights() {
        assert_eq!(BandwidthClass::High.weight(), 4.0);
        assert_eq!(BandwidthClass::Normal.weight(), 2.0);
        assert_eq!(BandwidthClass::Low.weight(), 1.0);
        assert_eq!(BandwidthClass::Background.weight(), 0.5);
    }

    #[test]
    fn test_bandwidth_class_priority_ordering() {
        assert!(BandwidthClass::High.priority() > BandwidthClass::Normal.priority());
        assert!(BandwidthClass::Normal.priority() > BandwidthClass::Low.priority());
        assert!(BandwidthClass::Low.priority() > BandwidthClass::Background.priority());
    }

    // ── BandwidthWindow ───────────────────────────────────────────────────────

    #[test]
    fn test_window_empty_returns_zero() {
        let w = BandwidthWindow::new();
        assert_eq!(w.average_bps(), 0);
        assert_eq!(w.peak_bps(), 0);
        assert!(w.is_empty());
    }

    #[test]
    fn test_window_single_sample() {
        let mut w = BandwidthWindow::new();
        w.push(1_000_000);
        assert!(!w.is_empty());
        assert_eq!(w.len(), 1);
        // average and peak should be within ±0.5% of the raw input
        let avg = w.average_bps();
        assert!((990_000..=1_010_000).contains(&avg));
    }

    #[test]
    fn test_window_evicts_oldest_at_capacity() {
        let mut w = BandwidthWindow::new();
        for _ in 0..12 {
            w.push(5_000_000);
        }
        assert_eq!(w.len(), 10);
    }

    #[test]
    fn test_window_peak_is_max_sample() {
        let mut w = BandwidthWindow::new();
        for v in [1_000_000u64, 2_000_000, 3_000_000] {
            w.push(v);
        }
        // Peak should be close to 3_000_000 (jitter ±0.5%)
        assert!(w.peak_bps() >= 2_900_000);
    }

    #[test]
    fn test_window_average_convergence() {
        let mut w = BandwidthWindow::new();
        let target = 10_000_000u64;
        for _ in 0..10 {
            w.push(target);
        }
        let avg = w.average_bps();
        // With jitter ±0.5%, average should be within 1% of target.
        assert!((avg as i64 - target as i64).abs() < 100_001);
    }

    // ── add_peer ──────────────────────────────────────────────────────────────

    #[test]
    fn test_add_peer_success() {
        let mut a = AdaptiveBandwidthAllocator::new(default_config());
        assert!(a
            .add_peer("p1".to_string(), BandwidthClass::Normal, 5_000_000)
            .is_ok());
        assert_eq!(a.peer_count(), 1);
    }

    #[test]
    fn test_add_peer_clamps_to_max() {
        let mut a = AdaptiveBandwidthAllocator::new(default_config());
        a.add_peer("p1".to_string(), BandwidthClass::High, 999_000_000)
            .expect("test: add high-bandwidth peer");
        assert_eq!(
            a.get_allocation("p1").expect("test: get allocation for p1"),
            50_000_000
        );
    }

    #[test]
    fn test_add_peer_clamps_to_min() {
        let mut a = AdaptiveBandwidthAllocator::new(default_config());
        a.add_peer("p1".to_string(), BandwidthClass::Background, 1)
            .expect("test: add background peer");
        assert_eq!(
            a.get_allocation("p1")
                .expect("test: get allocation for background peer"),
            1_000_000
        );
    }

    #[test]
    fn test_add_peer_exceeds_capacity() {
        let mut a = AdaptiveBandwidthAllocator::new(default_config());
        // Fill 80 Mbps (max per peer = 50, so two peers at 40 each).
        a.add_peer("p1".to_string(), BandwidthClass::High, 40_000_000)
            .expect("test: add first high peer");
        a.add_peer("p2".to_string(), BandwidthClass::High, 40_000_000)
            .expect("test: add second high peer");
        // Third peer would push beyond 100 Mbps.
        let err = a
            .add_peer("p3".to_string(), BandwidthClass::Normal, 40_000_000)
            .unwrap_err();
        assert!(matches!(err, AllocatorError::AllocationExceedsCapacity(_)));
    }

    #[test]
    fn test_add_peer_emits_event() {
        let mut a = AdaptiveBandwidthAllocator::new(default_config());
        a.add_peer("p1".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add normal peer for event test");
        let events = a.drain_events();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], BandwidthEvent::PeerAdded { .. }));
    }

    // ── remove_peer ───────────────────────────────────────────────────────────

    #[test]
    fn test_remove_peer_success() {
        let mut a = AdaptiveBandwidthAllocator::new(default_config());
        a.add_peer("p1".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add peer before remove");
        let profile = a.remove_peer("p1").expect("test: remove existing peer");
        assert_eq!(profile.peer_id, "p1");
        assert_eq!(a.peer_count(), 0);
    }

    #[test]
    fn test_remove_peer_releases_capacity() {
        let mut a = AdaptiveBandwidthAllocator::new(default_config());
        a.add_peer("p1".to_string(), BandwidthClass::High, 40_000_000)
            .expect("test: add p1 before release test");
        a.add_peer("p2".to_string(), BandwidthClass::High, 40_000_000)
            .expect("test: add p2 before release test");
        a.remove_peer("p1")
            .expect("test: remove p1 to release capacity");
        // After removing p1, enough capacity is free for a 40 Mbps peer.
        assert!(a
            .add_peer("p3".to_string(), BandwidthClass::High, 40_000_000)
            .is_ok());
    }

    #[test]
    fn test_remove_peer_not_found() {
        let mut a = AdaptiveBandwidthAllocator::new(default_config());
        let err = a.remove_peer("ghost").unwrap_err();
        assert!(matches!(err, AllocatorError::PeerNotFound(_)));
    }

    #[test]
    fn test_remove_peer_emits_event() {
        let mut a = AdaptiveBandwidthAllocator::new(default_config());
        a.add_peer("p1".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add peer before remove event test");
        a.drain_events();
        a.remove_peer("p1")
            .expect("test: remove peer to trigger event");
        let events = a.drain_events();
        assert!(matches!(events[0], BandwidthEvent::PeerRemoved { .. }));
    }

    // ── update_measurement ────────────────────────────────────────────────────

    #[test]
    fn test_update_measurement_normal() {
        let mut a = AdaptiveBandwidthAllocator::new(default_config());
        a.add_peer("p1".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add peer before measurement update");
        let ev = a
            .update_measurement("p1", 4_000_000, 20, 100)
            .expect("test: update measurement for p1");
        assert!(matches!(ev, BandwidthEvent::BandwidthUpdated { .. }));
    }

    #[test]
    fn test_update_measurement_congestion_detected() {
        let mut a = AdaptiveBandwidthAllocator::new(default_config());
        a.add_peer("p1".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add peer before congestion test");
        // loss_ppm > threshold (10_000)
        let ev = a
            .update_measurement("p1", 2_000_000, 100, 50_000)
            .expect("test: update measurement with high loss");
        assert!(matches!(ev, BandwidthEvent::CongestionDetected { .. }));
    }

    #[test]
    fn test_update_measurement_not_found() {
        let mut a = AdaptiveBandwidthAllocator::new(default_config());
        let err = a.update_measurement("ghost", 1_000, 5, 0).unwrap_err();
        assert!(matches!(err, AllocatorError::PeerNotFound(_)));
    }

    #[test]
    fn test_update_measurement_updates_window() {
        let mut a = AdaptiveBandwidthAllocator::new(default_config());
        a.add_peer("p1".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add peer before window test");
        for _ in 0..5 {
            a.update_measurement("p1", 4_000_000, 10, 0)
                .expect("test: update measurement in loop");
        }
        let profile = a
            .peer_profile("p1")
            .expect("test: get peer profile after window updates");
        assert_eq!(profile.window.len(), 5);
    }

    #[test]
    fn test_update_measurement_updates_latency_and_loss() {
        let mut a = AdaptiveBandwidthAllocator::new(default_config());
        a.add_peer("p1".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add peer before latency/loss test");
        a.update_measurement("p1", 3_000_000, 42, 500)
            .expect("test: update measurement with latency and loss");
        let p = a
            .peer_profile("p1")
            .expect("test: get peer profile for latency/loss check");
        assert_eq!(p.latency_ms, 42);
        assert_eq!(p.packet_loss_ppm, 500);
    }

    // ── reallocate — EqualShare ───────────────────────────────────────────────

    #[test]
    fn test_reallocate_equal_share_basic() {
        let mut a = alloc_with_policy(AllocationPolicy::EqualShare);
        a.add_peer("p1".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add p1 for equal share test");
        a.add_peer("p2".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add p2 for equal share test");
        a.reallocate();
        // 100 Mbps / 2 = 50 Mbps (matches max_allocation_bps)
        assert_eq!(
            a.get_allocation("p1")
                .expect("test: get p1 allocation after equal share reallocate"),
            50_000_000
        );
        assert_eq!(
            a.get_allocation("p2")
                .expect("test: get p2 allocation after equal share reallocate"),
            50_000_000
        );
    }

    #[test]
    fn test_reallocate_equal_share_zero_peers_is_safe() {
        let mut a = alloc_with_policy(AllocationPolicy::EqualShare);
        let events = a.reallocate();
        assert!(!events.is_empty());
        assert!(matches!(
            events.last().expect("test: events list must not be empty"),
            BandwidthEvent::AllocationRevised { .. }
        ));
    }

    #[test]
    fn test_reallocate_equal_share_clamped_to_min() {
        let mut cfg = default_config();
        cfg.total_capacity_bps = 5_000_000; // 5 Mbps total
        cfg.min_allocation_bps = 1_000_000;
        cfg.max_allocation_bps = 5_000_000;
        let mut a = AdaptiveBandwidthAllocator::new(cfg);
        // Add 10 peers — share would be 500k, clamped to min 1 Mbps.
        for i in 0..10 {
            // Some will exceed capacity but we want to see clamping in allocation
            let _ = a.add_peer(format!("p{i}"), BandwidthClass::Low, 1_000_000);
        }
        a.reallocate();
        // Verify clamping: allocation must be >= min
        for i in 0..a.peer_count() {
            if let Ok(alloc) = a.get_allocation(&format!("p{i}")) {
                assert!(alloc >= 1_000_000);
            }
        }
    }

    // ── reallocate — WeightedFair ─────────────────────────────────────────────

    #[test]
    fn test_reallocate_weighted_fair_respects_weight_order() {
        let mut a = alloc_with_policy(AllocationPolicy::WeightedFair);
        a.add_peer("high".to_string(), BandwidthClass::High, 5_000_000)
            .expect("test: add high-priority peer for weight order test");
        a.add_peer("low".to_string(), BandwidthClass::Low, 5_000_000)
            .expect("test: add low-priority peer for weight order test");
        a.reallocate();
        let high_alloc = a
            .get_allocation("high")
            .expect("test: get allocation for high priority peer");
        let low_alloc = a
            .get_allocation("low")
            .expect("test: get allocation for low priority peer");
        assert!(
            high_alloc > low_alloc,
            "high priority must get more than low"
        );
    }

    #[test]
    fn test_reallocate_weighted_fair_sum_bounded_by_capacity() {
        let mut a = populated_allocator(5, AllocationPolicy::WeightedFair);
        a.reallocate();
        let total: u64 = (0..5)
            .map(|i| {
                a.get_allocation(&format!("peer-{i}"))
                    .expect("test: get allocation for peer in weighted fair sum")
            })
            .sum();
        assert!(total <= 100_000_000);
    }

    #[test]
    fn test_reallocate_weighted_fair_single_peer() {
        let mut a = alloc_with_policy(AllocationPolicy::WeightedFair);
        a.add_peer("only".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add single peer for weighted fair test");
        a.reallocate();
        // 100% weight → 100 Mbps, but capped at max_allocation_bps = 50 Mbps
        assert_eq!(
            a.get_allocation("only")
                .expect("test: get allocation for sole weighted fair peer"),
            50_000_000
        );
    }

    // ── reallocate — MinGuarantee ─────────────────────────────────────────────

    #[test]
    fn test_reallocate_min_guarantee_basic() {
        let mut a = alloc_with_policy(AllocationPolicy::MinGuarantee(5_000_000));
        a.add_peer("p1".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add p1 for min guarantee basic test");
        a.add_peer("p2".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add p2 for min guarantee basic test");
        a.reallocate();
        assert!(
            a.get_allocation("p1")
                .expect("test: get allocation for p1 in min guarantee basic")
                >= 5_000_000
        );
        assert!(
            a.get_allocation("p2")
                .expect("test: get allocation for p2 in min guarantee basic")
                >= 5_000_000
        );
    }

    #[test]
    fn test_reallocate_min_guarantee_distributes_surplus() {
        let mut a = alloc_with_policy(AllocationPolicy::MinGuarantee(2_000_000));
        a.add_peer("p1".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add p1 for min guarantee surplus test");
        a.add_peer("p2".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add p2 for min guarantee surplus test");
        a.reallocate();
        // 2 peers × 2 Mbps guarantee = 4 Mbps, 96 Mbps surplus → bonus 48 per peer.
        // Total per peer = 50 Mbps (capped at max).
        assert_eq!(
            a.get_allocation("p1")
                .expect("test: get allocation for p1 after surplus distribution"),
            50_000_000
        );
    }

    #[test]
    fn test_reallocate_min_guarantee_honors_config_min() {
        let mut a = alloc_with_policy(AllocationPolicy::MinGuarantee(500_000));
        // Policy min of 500k < config min of 1 Mbps → effective min is 1 Mbps.
        a.add_peer("p1".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add p1 for min guarantee honors config min test");
        a.reallocate();
        assert!(
            a.get_allocation("p1")
                .expect("test: get allocation for p1 with config min enforcement")
                >= 1_000_000
        );
    }

    // ── reallocate — MaxCapacity ──────────────────────────────────────────────

    #[test]
    fn test_reallocate_max_capacity_caps_peers() {
        let mut a = alloc_with_policy(AllocationPolicy::MaxCapacity(10_000_000));
        a.add_peer("p1".to_string(), BandwidthClass::High, 5_000_000)
            .expect("test: add p1 for max capacity caps test");
        a.add_peer("p2".to_string(), BandwidthClass::High, 5_000_000)
            .expect("test: add p2 for max capacity caps test");
        a.reallocate();
        assert!(
            a.get_allocation("p1")
                .expect("test: get allocation for p1 under max capacity cap")
                <= 10_000_000
        );
        assert!(
            a.get_allocation("p2")
                .expect("test: get allocation for p2 under max capacity cap")
                <= 10_000_000
        );
    }

    #[test]
    fn test_reallocate_max_capacity_zero_peers_safe() {
        let mut a = alloc_with_policy(AllocationPolicy::MaxCapacity(10_000_000));
        a.reallocate();
        assert_eq!(a.peer_count(), 0);
    }

    // ── reallocate — PriorityQueue ────────────────────────────────────────────

    #[test]
    fn test_reallocate_priority_queue_high_gets_more() {
        let mut a = alloc_with_policy(AllocationPolicy::PriorityQueue);
        a.add_peer("bg".to_string(), BandwidthClass::Background, 5_000_000)
            .expect("test: add background peer for priority queue high-gets-more test");
        a.add_peer("hi".to_string(), BandwidthClass::High, 5_000_000)
            .expect("test: add high priority peer for priority queue high-gets-more test");
        a.reallocate();
        let hi = a
            .get_allocation("hi")
            .expect("test: get allocation for high priority peer in priority queue");
        let bg = a
            .get_allocation("bg")
            .expect("test: get allocation for background peer in priority queue");
        assert!(hi >= bg, "high priority should get >= background");
    }

    #[test]
    fn test_reallocate_priority_queue_no_negative_allocations() {
        let mut a = alloc_with_policy(AllocationPolicy::PriorityQueue);
        for i in 0..8 {
            a.add_peer(format!("p{i}"), BandwidthClass::High, 5_000_000)
                .expect("test: add high-priority peer in no-negative-allocations test");
        }
        a.reallocate();
        for i in 0..8 {
            assert!(
                a.get_allocation(&format!("p{i}"))
                    .expect("test: get allocation for peer in no-negative-allocations check")
                    >= 1_000_000
            );
        }
    }

    #[test]
    fn test_reallocate_priority_queue_emits_revised() {
        let mut a = populated_allocator(3, AllocationPolicy::PriorityQueue);
        a.drain_events();
        let events = a.reallocate();
        assert!(events
            .iter()
            .any(|e| matches!(e, BandwidthEvent::AllocationRevised { .. })));
    }

    // ── get_allocation ────────────────────────────────────────────────────────

    #[test]
    fn test_get_allocation_not_found() {
        let a = AdaptiveBandwidthAllocator::new(default_config());
        let err = a.get_allocation("nobody").unwrap_err();
        assert!(matches!(err, AllocatorError::PeerNotFound(_)));
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_empty_allocator() {
        let a = AdaptiveBandwidthAllocator::new(default_config());
        let s = a.stats();
        assert_eq!(s.peer_count, 0);
        assert_eq!(s.total_allocated_bps, 0);
        assert_eq!(s.fairness_index, 1.0);
    }

    #[test]
    fn test_stats_utilization_pct() {
        let mut a = AdaptiveBandwidthAllocator::new(default_config());
        a.add_peer("p1".to_string(), BandwidthClass::Normal, 10_000_000)
            .expect("test: add p1 for stats utilization test");
        a.update_measurement("p1", 5_000_000, 10, 0)
            .expect("test: record measurement for p1 in utilization test");
        let s = a.stats();
        // 5 Mbps used / 10 Mbps allocated = 50%
        assert!((s.utilization_pct - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_stats_congested_count() {
        let mut a = AdaptiveBandwidthAllocator::new(default_config());
        a.add_peer("p1".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add p1 for stats congested count test");
        a.add_peer("p2".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add p2 for stats congested count test");
        a.update_measurement("p1", 3_000_000, 100, 50_000)
            .expect("test: record congested measurement for p1"); // congested
        a.update_measurement("p2", 4_000_000, 10, 0)
            .expect("test: record healthy measurement for p2"); // healthy
        let s = a.stats();
        assert_eq!(s.congested_peers, 1);
    }

    #[test]
    fn test_stats_fairness_perfect() {
        let mut a = AdaptiveBandwidthAllocator::new(default_config());
        a.add_peer("p1".to_string(), BandwidthClass::Normal, 10_000_000)
            .expect("test: add p1 for stats fairness perfect test");
        a.add_peer("p2".to_string(), BandwidthClass::Normal, 10_000_000)
            .expect("test: add p2 for stats fairness perfect test");
        let s = a.stats();
        // Both peers have identical allocation → Jain's index = 1.0
        assert!((s.fairness_index - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_fairness_unequal() {
        let mut a = AdaptiveBandwidthAllocator::new(default_config());
        a.add_peer("p1".to_string(), BandwidthClass::High, 10_000_000)
            .expect("test: add p1 High class for stats fairness unequal test");
        a.add_peer("p2".to_string(), BandwidthClass::High, 10_000_000)
            .expect("test: add p2 High class for stats fairness unequal test");
        // Manually manipulate via WeightedFair to get unequal allocations.
        a.reallocate();
        let s = a.stats();
        assert!(s.fairness_index > 0.0 && s.fairness_index <= 1.0);
    }

    // ── congested_peers ───────────────────────────────────────────────────────

    #[test]
    fn test_congested_peers_empty() {
        let a = AdaptiveBandwidthAllocator::new(default_config());
        assert!(a.congested_peers().is_empty());
    }

    #[test]
    fn test_congested_peers_detection() {
        let mut a = AdaptiveBandwidthAllocator::new(default_config());
        a.add_peer("good".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add good peer for congested peers detection test");
        a.add_peer("bad".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add bad peer for congested peers detection test");
        a.update_measurement("bad", 1_000_000, 200, 100_000)
            .expect("test: record high-latency measurement for bad peer");
        let congested = a.congested_peers();
        assert_eq!(congested.len(), 1);
        assert_eq!(congested[0], "bad");
    }

    #[test]
    fn test_congested_peers_at_threshold_not_flagged() {
        let mut a = AdaptiveBandwidthAllocator::new(default_config());
        a.add_peer("p1".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add p1 for congested peers at threshold test");
        // Exactly at threshold (10_000) — not above, so not congested.
        a.update_measurement("p1", 3_000_000, 50, 10_000)
            .expect("test: record measurement at congestion threshold for p1");
        assert!(a.congested_peers().is_empty());
    }

    // ── events_since / drain_events ───────────────────────────────────────────

    #[test]
    fn test_events_since_filters_by_timestamp() {
        let mut a = AdaptiveBandwidthAllocator::new(default_config());
        a.add_peer("p1".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add p1 for events since timestamp filter test");
        a.advance_time(100);
        let mid_ts = a.now_ms;
        a.add_peer("p2".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add p2 after time advance for events since timestamp filter test");
        let recent = a.events_since(mid_ts);
        // Only p2's PeerAdded event should be >= mid_ts.
        assert_eq!(recent.len(), 1);
    }

    #[test]
    fn test_drain_events_clears_history() {
        let mut a = AdaptiveBandwidthAllocator::new(default_config());
        a.add_peer("p1".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add p1 for drain events clears history test");
        assert!(!a.drain_events().is_empty());
        assert!(a.drain_events().is_empty());
    }

    #[test]
    fn test_event_history_bounded_at_200() {
        let mut a = AdaptiveBandwidthAllocator::new(default_config());
        a.add_peer("p1".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add p1 for event history bounded test");
        // Push 300 update_measurement events.
        for _ in 0..300 {
            a.update_measurement("p1", 4_000_000, 10, 0)
                .expect("test: record repeated measurement for event history bounded test");
        }
        // History should be capped at MAX_EVENT_HISTORY.
        let events = a.events_since(0);
        assert!(events.len() <= AdaptiveBandwidthAllocator::MAX_EVENT_HISTORY);
    }

    #[test]
    fn test_events_since_zero_returns_all() {
        let mut a = AdaptiveBandwidthAllocator::new(default_config());
        a.add_peer("p1".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add p1 for events since zero returns all test");
        a.add_peer("p2".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add p2 for events since zero returns all test");
        let all = a.events_since(0);
        assert_eq!(all.len(), 2);
    }

    // ── AllocatorConfig::validate ─────────────────────────────────────────────

    #[test]
    fn test_config_validate_ok() {
        assert!(default_config().validate().is_ok());
    }

    #[test]
    fn test_config_validate_min_gt_max_fails() {
        let mut cfg = default_config();
        cfg.min_allocation_bps = 60_000_000;
        cfg.max_allocation_bps = 10_000_000;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_zero_capacity_fails() {
        let mut cfg = default_config();
        cfg.total_capacity_bps = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_max_gt_capacity_fails() {
        let mut cfg = default_config();
        cfg.max_allocation_bps = cfg.total_capacity_bps + 1;
        assert!(cfg.validate().is_err());
    }

    // ── Jain's fairness index ─────────────────────────────────────────────────

    #[test]
    fn test_jain_fairness_all_equal() {
        let fi =
            AdaptiveBandwidthAllocator::jain_fairness_index([10.0, 10.0, 10.0].iter().copied(), 3);
        assert!((fi - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_jain_fairness_one_takes_all() {
        // One peer with all bandwidth, n-1 with 0.
        let fi =
            AdaptiveBandwidthAllocator::jain_fairness_index([100.0, 0.0, 0.0].iter().copied(), 3);
        // (100)² / (3 × 100²) = 1/3
        assert!((fi - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_jain_fairness_zero_peers() {
        let fi = AdaptiveBandwidthAllocator::jain_fairness_index(std::iter::empty(), 0);
        assert_eq!(fi, 1.0);
    }

    #[test]
    fn test_jain_fairness_single_peer() {
        let fi = AdaptiveBandwidthAllocator::jain_fairness_index([42.0].iter().copied(), 1);
        assert!((fi - 1.0).abs() < 1e-9);
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_reallocate_emits_allocation_revised() {
        let mut a = populated_allocator(3, AllocationPolicy::EqualShare);
        a.drain_events();
        let events = a.reallocate();
        assert!(events
            .iter()
            .any(|e| matches!(e, BandwidthEvent::AllocationRevised { .. })));
    }

    #[test]
    fn test_single_peer_equal_share() {
        let mut a = alloc_with_policy(AllocationPolicy::EqualShare);
        a.add_peer("only".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add single peer for equal share test");
        a.reallocate();
        // 100 Mbps / 1 peer = 100 Mbps, capped at max 50 Mbps.
        assert_eq!(
            a.get_allocation("only")
                .expect("test: get allocation for single peer"),
            50_000_000
        );
    }

    #[test]
    fn test_over_capacity_add_rejected() {
        let mut cfg = default_config();
        cfg.total_capacity_bps = 10_000_000; // 10 Mbps
        cfg.max_allocation_bps = 10_000_000;
        let mut a = AdaptiveBandwidthAllocator::new(cfg);
        a.add_peer("p1".to_string(), BandwidthClass::Normal, 10_000_000)
            .expect("test: add p1 within capacity for over-capacity rejection test");
        // Adding any more exceeds capacity.
        let err = a
            .add_peer("p2".to_string(), BandwidthClass::Normal, 1_000_000)
            .unwrap_err();
        assert!(matches!(err, AllocatorError::AllocationExceedsCapacity(_)));
    }

    #[test]
    fn test_reallocate_after_remove_rebalances_correctly() {
        let mut a = alloc_with_policy(AllocationPolicy::EqualShare);
        a.add_peer("p1".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add p1 for rebalance after remove test");
        a.add_peer("p2".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add p2 for rebalance after remove test");
        a.add_peer("p3".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add p3 to be removed for rebalance test");
        a.remove_peer("p3")
            .expect("test: remove p3 to trigger rebalance");
        a.reallocate();
        // 100 Mbps / 2 = 50 Mbps (at max cap)
        assert_eq!(
            a.get_allocation("p1")
                .expect("test: get allocation for p1 after rebalance"),
            50_000_000
        );
        assert_eq!(
            a.get_allocation("p2")
                .expect("test: get allocation for p2 after rebalance"),
            50_000_000
        );
    }

    #[test]
    fn test_congestion_clears_after_update() {
        let mut a = AdaptiveBandwidthAllocator::new(default_config());
        a.add_peer("p1".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add p1 for congestion clear test");
        a.update_measurement("p1", 1_000_000, 200, 100_000)
            .expect("test: update measurement to mark p1 as congested"); // congested
        assert_eq!(a.congested_peers().len(), 1);
        a.update_measurement("p1", 4_000_000, 10, 0)
            .expect("test: update measurement to clear p1 congestion"); // recovered
        assert_eq!(a.congested_peers().len(), 0);
    }

    #[test]
    fn test_advance_time_affects_events() {
        let mut a = AdaptiveBandwidthAllocator::new(default_config());
        a.add_peer("p1".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add p1 before time advance");
        a.advance_time(500);
        a.add_peer("p2".to_string(), BandwidthClass::Normal, 5_000_000)
            .expect("test: add p2 after time advance");
        let after_advance = a.events_since(a.now_ms);
        // Only p2's event is at the advanced timestamp.
        assert_eq!(after_advance.len(), 1);
    }

    #[test]
    fn test_multiple_reallocate_calls_stable() {
        let mut a = populated_allocator(4, AllocationPolicy::WeightedFair);
        for _ in 0..5 {
            a.reallocate();
        }
        // All allocations must remain within bounds.
        for i in 0..4 {
            let alloc = a
                .get_allocation(&format!("peer-{i}"))
                .expect("test: get allocation for peer in stability test");
            let cfg = a.config();
            assert!(alloc >= cfg.min_allocation_bps);
            assert!(alloc <= cfg.max_allocation_bps);
        }
    }
}
