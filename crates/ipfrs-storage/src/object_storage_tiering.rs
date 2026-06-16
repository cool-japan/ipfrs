//! ObjectStorageTiering — hierarchical object storage tiering (hot/warm/cold).
//!
//! Provides a production-quality tiered object storage system that automatically
//! migrates objects between Hot, Warm, and Cold storage tiers based on configurable
//! policies (access frequency, recency, size thresholds, cost optimization).
//!
//! # Type Aliases
//!
//! Due to name collisions with existing crate types, all public types use the
//! `Ost` prefix:
//! - `OstStorageTier`  — Hot/Warm/Cold tier enum
//! - `OstTierPolicy`   — tiering policy variants
//! - `OstTierConfig`   — tier capacity and policy configuration
//! - `OstTierTransition` — record of an object moving between tiers
//!
//! # Example
//!
//! ```rust
//! use ipfrs_storage::object_storage_tiering::{
//!     ObjectStorageTiering, OstStorageTier, OstTierConfig, OstTierPolicy, TieredObject,
//! };
//!
//! let config = OstTierConfig {
//!     hot_capacity_bytes: 1_024 * 1_024,
//!     warm_capacity_bytes: 10 * 1_024 * 1_024,
//!     cold_capacity_bytes: u64::MAX,
//!     policy: OstTierPolicy::AccessFrequency(5),
//!     promotion_threshold: 3,
//!     demotion_interval_us: 60_000_000,
//! };
//!
//! let mut tiering = ObjectStorageTiering::new(config);
//!
//! let obj = TieredObject {
//!     id: "obj-1".to_string(),
//!     size_bytes: 512,
//!     current_tier: OstStorageTier::Hot,
//!     access_count: 0,
//!     last_accessed: 0,
//!     created_at: 0,
//!     tags: vec![],
//!     cost_per_hour: 0.001,
//! };
//!
//! let tier = tiering.store(obj).unwrap();
//! assert_eq!(tier, OstStorageTier::Hot);
//! ```

use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// OstStorageTier
// ---------------------------------------------------------------------------

/// Three-level object storage tier hierarchy.
///
/// `Ost` prefix used to avoid collision with `StorageTier` in `cold_storage`,
/// `lifecycle`, `tier_manager`, and `tier_migration_engine`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OstStorageTier {
    /// Hot tier — fastest access, highest cost, limited capacity.
    Hot,
    /// Warm tier — balanced access speed, medium cost.
    Warm,
    /// Cold tier — slowest access, cheapest, essentially unlimited capacity.
    Cold,
}

impl OstStorageTier {
    /// Returns the tier one level colder, or `None` if already at Cold.
    pub fn colder(self) -> Option<OstStorageTier> {
        match self {
            OstStorageTier::Hot => Some(OstStorageTier::Warm),
            OstStorageTier::Warm => Some(OstStorageTier::Cold),
            OstStorageTier::Cold => None,
        }
    }

    /// Returns the tier one level hotter, or `None` if already at Hot.
    pub fn hotter(self) -> Option<OstStorageTier> {
        match self {
            OstStorageTier::Cold => Some(OstStorageTier::Warm),
            OstStorageTier::Warm => Some(OstStorageTier::Hot),
            OstStorageTier::Hot => None,
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            OstStorageTier::Hot => "hot",
            OstStorageTier::Warm => "warm",
            OstStorageTier::Cold => "cold",
        }
    }

    /// Numeric rank: lower = hotter.
    pub fn rank(self) -> u8 {
        match self {
            OstStorageTier::Hot => 0,
            OstStorageTier::Warm => 1,
            OstStorageTier::Cold => 2,
        }
    }
}

impl std::fmt::Display for OstStorageTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// OstTierPolicy
// ---------------------------------------------------------------------------

/// Policy that drives automatic tier promotions and demotions.
///
/// `Ost` prefix avoids collision with `TierPolicy` in `cold_storage`,
/// `tier_manager`, and `tier_migration_engine`.
#[derive(Debug, Clone)]
pub enum OstTierPolicy {
    /// Promote to Hot when `access_count > n`; no automatic demotion.
    AccessFrequency(u32),
    /// Demote to Cold when not accessed for `n` microseconds.
    RecencyBased(u64),
    /// Keep each tier under its byte capacity; demote largest objects first.
    SizeThreshold {
        /// Maximum bytes allowed in the Hot tier.
        hot_max_bytes: u64,
        /// Maximum bytes allowed in the Warm tier.
        warm_max_bytes: u64,
    },
    /// Only explicit `promote`/`demote` calls move objects; no auto policy.
    ManualOnly,
    /// Demote low-access-count objects to minimise cost; multiplier on
    /// `cost_per_byte_per_hour`.
    CostOptimized(f64),
}

// ---------------------------------------------------------------------------
// OstTierConfig
// ---------------------------------------------------------------------------

/// Capacity and policy configuration for the tiering engine.
///
/// `Ost` prefix avoids collision with `TierConfig` in `tiering`.
#[derive(Debug, Clone)]
pub struct OstTierConfig {
    /// Maximum bytes that can live in the Hot tier.
    pub hot_capacity_bytes: u64,
    /// Maximum bytes that can live in the Warm tier.
    pub warm_capacity_bytes: u64,
    /// Maximum bytes for the Cold tier (`u64::MAX` = effectively unlimited).
    pub cold_capacity_bytes: u64,
    /// Policy that drives automatic promotions and demotions.
    pub policy: OstTierPolicy,
    /// Access-count threshold for Warm → Hot promotion.
    pub promotion_threshold: u32,
    /// How often (in microseconds) to check for cold-eligible objects.
    pub demotion_interval_us: u64,
}

impl Default for OstTierConfig {
    fn default() -> Self {
        Self {
            hot_capacity_bytes: 512 * 1_024 * 1_024,         // 512 MiB
            warm_capacity_bytes: 10 * 1_024 * 1_024 * 1_024, // 10 GiB
            cold_capacity_bytes: u64::MAX,
            policy: OstTierPolicy::AccessFrequency(10),
            promotion_threshold: 5,
            demotion_interval_us: 3_600_000_000, // 1 hour
        }
    }
}

// ---------------------------------------------------------------------------
// TieredObject
// ---------------------------------------------------------------------------

/// An object managed by the tiering engine.
#[derive(Debug, Clone)]
pub struct TieredObject {
    /// Unique identifier for this object.
    pub id: String,
    /// Size of the object in bytes.
    pub size_bytes: u64,
    /// Current storage tier.
    pub current_tier: OstStorageTier,
    /// Total number of times this object has been accessed.
    pub access_count: u64,
    /// Timestamp (microseconds since epoch) of the last access.
    pub last_accessed: u64,
    /// Timestamp (microseconds since epoch) when the object was created.
    pub created_at: u64,
    /// Arbitrary string tags for metadata.
    pub tags: Vec<String>,
    /// Estimated cost per hour for storing this object (in abstract units).
    pub cost_per_hour: f64,
}

// ---------------------------------------------------------------------------
// OstTierTransition
// ---------------------------------------------------------------------------

/// Record of an object moving between storage tiers.
///
/// `Ost` prefix avoids collision with `TierTransition` in `tier_manager`.
#[derive(Debug, Clone)]
pub struct OstTierTransition {
    /// ID of the object that was moved.
    pub object_id: String,
    /// Tier the object was moved *from*.
    pub from_tier: OstStorageTier,
    /// Tier the object was moved *to*.
    pub to_tier: OstStorageTier,
    /// Human-readable reason for the transition.
    pub reason: String,
    /// Timestamp (microseconds since epoch) of the transition.
    pub transitioned_at: u64,
    /// Number of bytes that were moved.
    pub bytes_moved: u64,
}

// ---------------------------------------------------------------------------
// TieringStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for the tiering engine.
#[derive(Debug, Clone, Default)]
pub struct TieringStats {
    /// Number of objects currently in the Hot tier.
    pub hot_objects: usize,
    /// Number of objects currently in the Warm tier.
    pub warm_objects: usize,
    /// Number of objects currently in the Cold tier.
    pub cold_objects: usize,
    /// Total bytes in the Hot tier.
    pub hot_bytes: u64,
    /// Total bytes in the Warm tier.
    pub warm_bytes: u64,
    /// Total bytes in the Cold tier.
    pub cold_bytes: u64,
    /// Total number of promotions performed (demotion → hotter tier).
    pub promotions: u64,
    /// Total number of demotions performed (hotter → colder tier).
    pub demotions: u64,
    /// Sum of `cost_per_hour` across all managed objects.
    pub total_cost_per_hour: f64,
}

// ---------------------------------------------------------------------------
// TieringError
// ---------------------------------------------------------------------------

/// Errors returned by the tiering engine.
#[derive(Debug, Clone, PartialEq)]
pub enum TieringError {
    /// No object with the given ID was found.
    ObjectNotFound(String),
    /// The target tier does not have enough capacity.
    TierFull {
        /// Which tier is full.
        tier: OstStorageTier,
        /// How many bytes were needed.
        needed: u64,
        /// How many bytes are currently available.
        available: u64,
    },
    /// The requested operation conflicts with the active policy.
    PolicyConflict(String),
    /// The supplied configuration is invalid.
    InvalidConfiguration(String),
}

impl std::fmt::Display for TieringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TieringError::ObjectNotFound(id) => write!(f, "object not found: {id}"),
            TieringError::TierFull {
                tier,
                needed,
                available,
            } => {
                write!(
                    f,
                    "tier {tier} full: needed {needed} bytes, {available} available"
                )
            }
            TieringError::PolicyConflict(msg) => write!(f, "policy conflict: {msg}"),
            TieringError::InvalidConfiguration(msg) => {
                write!(f, "invalid configuration: {msg}")
            }
        }
    }
}

impl std::error::Error for TieringError {}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Maximum number of transitions kept in the history ring buffer.
const MAX_HISTORY: usize = 500;

/// Returns the capacity of the given tier as configured.
fn tier_capacity(cfg: &OstTierConfig, tier: OstStorageTier) -> u64 {
    match tier {
        OstStorageTier::Hot => cfg.hot_capacity_bytes,
        OstStorageTier::Warm => cfg.warm_capacity_bytes,
        OstStorageTier::Cold => cfg.cold_capacity_bytes,
    }
}

// ---------------------------------------------------------------------------
// ObjectStorageTiering
// ---------------------------------------------------------------------------

/// Production-quality hierarchical object storage tiering engine.
///
/// Maintains a set of `TieredObject` instances distributed across Hot, Warm,
/// and Cold tiers. Automatic promotion/demotion is driven by the
/// `OstTierPolicy` supplied in `OstTierConfig`.
pub struct ObjectStorageTiering {
    config: OstTierConfig,
    /// Primary object store: id → object.
    objects: HashMap<String, TieredObject>,
    /// Per-tier byte usage counters.
    tier_bytes: [u64; 3],
    /// Ring buffer of the last `MAX_HISTORY` tier transitions.
    history: VecDeque<OstTierTransition>,
    /// Running counts for statistics.
    promotions: u64,
    demotions: u64,
    /// IDs of objects that need a promotion on the next `run_policy` call
    /// (populated by `retrieve` when access crosses the threshold).
    pending_promotions: Vec<String>,
}

impl ObjectStorageTiering {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Create a new tiering engine with the supplied configuration.
    pub fn new(config: OstTierConfig) -> Self {
        Self {
            config,
            objects: HashMap::new(),
            tier_bytes: [0u64; 3],
            history: VecDeque::with_capacity(MAX_HISTORY + 1),
            promotions: 0,
            demotions: 0,
            pending_promotions: Vec::new(),
        }
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// Bytes currently used by `tier`.
    fn used_bytes(&self, tier: OstStorageTier) -> u64 {
        self.tier_bytes[tier.rank() as usize]
    }

    /// Free bytes available in `tier`.
    fn available_bytes(&self, tier: OstStorageTier) -> u64 {
        let cap = tier_capacity(&self.config, tier);
        let used = self.used_bytes(tier);
        cap.saturating_sub(used)
    }

    /// Add `delta` bytes to the usage counter for `tier`.
    fn add_bytes(&mut self, tier: OstStorageTier, delta: u64) {
        self.tier_bytes[tier.rank() as usize] =
            self.tier_bytes[tier.rank() as usize].saturating_add(delta);
    }

    /// Subtract `delta` bytes from the usage counter for `tier`.
    fn sub_bytes(&mut self, tier: OstStorageTier, delta: u64) {
        self.tier_bytes[tier.rank() as usize] =
            self.tier_bytes[tier.rank() as usize].saturating_sub(delta);
    }

    /// Append a transition to the history ring buffer.
    fn push_history(&mut self, t: OstTierTransition) {
        if self.history.len() >= MAX_HISTORY {
            self.history.pop_front();
        }
        self.history.push_back(t);
    }

    /// Determine the preferred initial tier for a new object of `size_bytes`.
    /// Falls back to progressively colder tiers if a tier is full.
    fn initial_tier_for(&self, size_bytes: u64) -> Result<OstStorageTier, TieringError> {
        for tier in [
            OstStorageTier::Hot,
            OstStorageTier::Warm,
            OstStorageTier::Cold,
        ] {
            if self.available_bytes(tier) >= size_bytes {
                return Ok(tier);
            }
        }
        Err(TieringError::TierFull {
            tier: OstStorageTier::Cold,
            needed: size_bytes,
            available: self.available_bytes(OstStorageTier::Cold),
        })
    }

    /// Internal move: update bookkeeping and record the transition.
    /// Does NOT check capacity — caller must ensure space exists.
    fn internal_move(
        &mut self,
        id: &str,
        to_tier: OstStorageTier,
        reason: &str,
        current_ts: u64,
    ) -> OstTierTransition {
        // We need to get size and from_tier from the object.
        let (from_tier, size_bytes) = {
            let obj = self.objects.get(id).expect("caller verified existence");
            (obj.current_tier, obj.size_bytes)
        };
        // Update accounting.
        self.sub_bytes(from_tier, size_bytes);
        self.add_bytes(to_tier, size_bytes);
        // Update the object itself.
        if let Some(obj) = self.objects.get_mut(id) {
            obj.current_tier = to_tier;
        }
        let t = OstTierTransition {
            object_id: id.to_string(),
            from_tier,
            to_tier,
            reason: reason.to_string(),
            transitioned_at: current_ts,
            bytes_moved: size_bytes,
        };
        self.push_history(t.clone());
        t
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Store an object, placing it in the most appropriate tier based on
    /// policy and available capacity.
    ///
    /// New objects always attempt Hot first, then Warm, then Cold. The
    /// `current_tier` field of the supplied object is ignored; the actual
    /// assigned tier is returned.
    ///
    /// Returns `TieringError::TierFull` if no tier can accept the object.
    pub fn store(&mut self, mut object: TieredObject) -> Result<OstStorageTier, TieringError> {
        let tier = self.initial_tier_for(object.size_bytes)?;
        object.current_tier = tier;
        self.add_bytes(tier, object.size_bytes);
        self.objects.insert(object.id.clone(), object);
        Ok(tier)
    }

    /// Retrieve an object by ID, updating its `last_accessed` timestamp and
    /// incrementing `access_count`. If the access count crosses the configured
    /// promotion threshold the object is queued for promotion.
    ///
    /// Returns `TieringError::ObjectNotFound` if the ID is unknown.
    pub fn retrieve(&mut self, id: &str, current_ts: u64) -> Result<&TieredObject, TieringError> {
        let obj = self
            .objects
            .get_mut(id)
            .ok_or_else(|| TieringError::ObjectNotFound(id.to_string()))?;
        obj.last_accessed = current_ts;
        obj.access_count = obj.access_count.saturating_add(1);
        // Queue for promotion if threshold crossed and not already hot.
        let should_queue = obj.access_count > obj.access_count.saturating_sub(1)
            && obj.access_count == (self.config.promotion_threshold as u64).saturating_add(1)
            && obj.current_tier != OstStorageTier::Hot;
        if should_queue {
            self.pending_promotions.push(id.to_string());
        }
        Ok(self.objects.get(id).expect("just inserted/accessed"))
    }

    /// Promote an object to a hotter tier.
    ///
    /// Returns `TieringError::TierFull` if the target tier lacks capacity.
    /// Returns `TieringError::ObjectNotFound` if the ID is unknown.
    /// Returns `TieringError::PolicyConflict` if `to` is colder than or equal
    /// to the object's current tier (use `demote` instead).
    pub fn promote(
        &mut self,
        id: &str,
        to: OstStorageTier,
        current_ts: u64,
    ) -> Result<OstTierTransition, TieringError> {
        let (current_tier, size_bytes) = {
            let obj = self
                .objects
                .get(id)
                .ok_or_else(|| TieringError::ObjectNotFound(id.to_string()))?;
            (obj.current_tier, obj.size_bytes)
        };
        if to >= current_tier {
            return Err(TieringError::PolicyConflict(format!(
                "promote: target tier {to} is not hotter than current tier {current_tier}"
            )));
        }
        let avail = self.available_bytes(to);
        if avail < size_bytes {
            return Err(TieringError::TierFull {
                tier: to,
                needed: size_bytes,
                available: avail,
            });
        }
        self.promotions += 1;
        Ok(self.internal_move(id, to, "explicit promotion", current_ts))
    }

    /// Demote an object to a colder tier.
    ///
    /// Returns `TieringError::TierFull` if the target tier lacks capacity.
    /// Returns `TieringError::ObjectNotFound` if the ID is unknown.
    /// Returns `TieringError::PolicyConflict` if `to` is hotter than or equal
    /// to the object's current tier.
    pub fn demote(
        &mut self,
        id: &str,
        to: OstStorageTier,
        current_ts: u64,
    ) -> Result<OstTierTransition, TieringError> {
        let (current_tier, size_bytes) = {
            let obj = self
                .objects
                .get(id)
                .ok_or_else(|| TieringError::ObjectNotFound(id.to_string()))?;
            (obj.current_tier, obj.size_bytes)
        };
        if to <= current_tier {
            return Err(TieringError::PolicyConflict(format!(
                "demote: target tier {to} is not colder than current tier {current_tier}"
            )));
        }
        let avail = self.available_bytes(to);
        if avail < size_bytes {
            return Err(TieringError::TierFull {
                tier: to,
                needed: size_bytes,
                available: avail,
            });
        }
        self.demotions += 1;
        Ok(self.internal_move(id, to, "explicit demotion", current_ts))
    }

    /// Evaluate all objects against the active policy and perform automatic
    /// promotions and demotions. Returns all transitions that occurred.
    ///
    /// Policies applied:
    /// - `AccessFrequency(n)`: objects with `access_count > n` not in Hot → promote.
    /// - `RecencyBased(window_us)`: objects not accessed in `window_us` µs → Cold.
    /// - `SizeThreshold`: enforce per-tier byte caps by demoting largest objects first.
    /// - `CostOptimized(multiplier)`: demote objects whose cost-adjusted score
    ///   ranks them as low-value, keeping total cost under budget.
    /// - `ManualOnly`: no automatic transitions.
    pub fn run_policy(&mut self, current_ts: u64) -> Vec<OstTierTransition> {
        let mut transitions = Vec::new();

        match self.config.policy.clone() {
            OstTierPolicy::AccessFrequency(threshold) => {
                transitions.extend(self.run_access_frequency_policy(threshold, current_ts));
            }
            OstTierPolicy::RecencyBased(window_us) => {
                transitions.extend(self.run_recency_policy(window_us, current_ts));
            }
            OstTierPolicy::SizeThreshold {
                hot_max_bytes,
                warm_max_bytes,
            } => {
                transitions.extend(self.run_size_threshold_policy(
                    hot_max_bytes,
                    warm_max_bytes,
                    current_ts,
                ));
            }
            OstTierPolicy::ManualOnly => {}
            OstTierPolicy::CostOptimized(multiplier) => {
                transitions.extend(self.run_cost_optimized_policy(multiplier, current_ts));
            }
        }

        // Also drain pending promotions from retrieve().
        let pending: Vec<String> = std::mem::take(&mut self.pending_promotions);
        for id in pending {
            if let Some(obj) = self.objects.get(&id) {
                if obj.current_tier == OstStorageTier::Hot {
                    continue;
                }
                let target = OstStorageTier::Hot;
                let size = obj.size_bytes;
                if self.available_bytes(target) >= size {
                    self.promotions += 1;
                    let t =
                        self.internal_move(&id, target, "access-threshold promotion", current_ts);
                    transitions.push(t);
                }
            }
        }

        transitions
    }

    /// Demote objects from `tier` to make room for `needed_bytes`, using
    /// Least-Recently-Used ordering. Returns all transitions performed.
    ///
    /// Objects are demoted one tier at a time (Hot→Warm or Warm→Cold).
    pub fn evict_tier(
        &mut self,
        tier: OstStorageTier,
        needed_bytes: u64,
    ) -> Vec<OstTierTransition> {
        let current_ts = 0u64; // timestamp not provided; use 0 as sentinel
        self.evict_tier_ts(tier, needed_bytes, current_ts)
    }

    /// Like `evict_tier` but accepts an explicit timestamp.
    pub fn evict_tier_ts(
        &mut self,
        tier: OstStorageTier,
        needed_bytes: u64,
        current_ts: u64,
    ) -> Vec<OstTierTransition> {
        let target_tier = match tier.colder() {
            Some(t) => t,
            None => return vec![], // Cold has nowhere colder
        };

        // Collect candidates sorted by LRU (oldest last_accessed first).
        let mut candidates: Vec<(String, u64, u64)> = self
            .objects
            .values()
            .filter(|o| o.current_tier == tier)
            .map(|o| (o.id.clone(), o.last_accessed, o.size_bytes))
            .collect();
        candidates.sort_by_key(|(_, last_accessed, _)| *last_accessed);

        let mut freed = 0u64;
        let mut transitions = Vec::new();

        for (id, _last_accessed, size) in candidates {
            if freed >= needed_bytes {
                break;
            }
            let avail = self.available_bytes(target_tier);
            if avail < size {
                // Target tier also full — cascade demotion if possible.
                continue;
            }
            self.demotions += 1;
            let t = self.internal_move(&id, target_tier, "lru eviction", current_ts);
            freed += size;
            transitions.push(t);
        }

        transitions
    }

    /// Return references to all objects currently in `tier`.
    pub fn tier_objects(&self, tier: OstStorageTier) -> Vec<&TieredObject> {
        self.objects
            .values()
            .filter(|o| o.current_tier == tier)
            .collect()
    }

    /// Return a copy of the last up-to-500 tier transitions (oldest first).
    pub fn transition_history(&self) -> Vec<OstTierTransition> {
        self.history.iter().cloned().collect()
    }

    /// Return aggregate statistics for all tiers.
    pub fn stats(&self) -> TieringStats {
        let mut stats = TieringStats {
            promotions: self.promotions,
            demotions: self.demotions,
            ..Default::default()
        };
        for obj in self.objects.values() {
            stats.total_cost_per_hour += obj.cost_per_hour;
            match obj.current_tier {
                OstStorageTier::Hot => {
                    stats.hot_objects += 1;
                    stats.hot_bytes += obj.size_bytes;
                }
                OstStorageTier::Warm => {
                    stats.warm_objects += 1;
                    stats.warm_bytes += obj.size_bytes;
                }
                OstStorageTier::Cold => {
                    stats.cold_objects += 1;
                    stats.cold_bytes += obj.size_bytes;
                }
            }
        }
        stats
    }

    // ------------------------------------------------------------------
    // Policy runners (private)
    // ------------------------------------------------------------------

    fn run_access_frequency_policy(
        &mut self,
        threshold: u32,
        current_ts: u64,
    ) -> Vec<OstTierTransition> {
        let promote_ids: Vec<String> = self
            .objects
            .values()
            .filter(|o| o.access_count > threshold as u64 && o.current_tier != OstStorageTier::Hot)
            .map(|o| o.id.clone())
            .collect();

        let mut transitions = Vec::new();
        for id in promote_ids {
            let size = match self.objects.get(&id) {
                Some(o) => o.size_bytes,
                None => continue,
            };
            if self.available_bytes(OstStorageTier::Hot) >= size {
                self.promotions += 1;
                let t = self.internal_move(
                    &id,
                    OstStorageTier::Hot,
                    "access-frequency promotion",
                    current_ts,
                );
                transitions.push(t);
            }
        }
        transitions
    }

    fn run_recency_policy(&mut self, window_us: u64, current_ts: u64) -> Vec<OstTierTransition> {
        let cutoff = current_ts.saturating_sub(window_us);
        let demote_ids: Vec<String> = self
            .objects
            .values()
            .filter(|o| o.last_accessed < cutoff && o.current_tier != OstStorageTier::Cold)
            .map(|o| o.id.clone())
            .collect();

        let mut transitions = Vec::new();
        for id in demote_ids {
            let size = match self.objects.get(&id) {
                Some(o) => o.size_bytes,
                None => continue,
            };
            if self.available_bytes(OstStorageTier::Cold) >= size {
                self.demotions += 1;
                let t =
                    self.internal_move(&id, OstStorageTier::Cold, "recency demotion", current_ts);
                transitions.push(t);
            }
        }
        transitions
    }

    fn run_size_threshold_policy(
        &mut self,
        hot_max_bytes: u64,
        warm_max_bytes: u64,
        current_ts: u64,
    ) -> Vec<OstTierTransition> {
        let mut transitions = Vec::new();

        // Enforce Hot cap: demote largest objects first until within cap.
        while self.used_bytes(OstStorageTier::Hot) > hot_max_bytes {
            // Pick the largest hot object.
            let victim = self
                .objects
                .values()
                .filter(|o| o.current_tier == OstStorageTier::Hot)
                .max_by_key(|o| o.size_bytes)
                .map(|o| (o.id.clone(), o.size_bytes));
            match victim {
                None => break,
                Some((id, size)) => {
                    if self.available_bytes(OstStorageTier::Warm) >= size {
                        self.demotions += 1;
                        let t = self.internal_move(
                            &id,
                            OstStorageTier::Warm,
                            "size-threshold hot→warm",
                            current_ts,
                        );
                        transitions.push(t);
                    } else if self.available_bytes(OstStorageTier::Cold) >= size {
                        self.demotions += 1;
                        let t = self.internal_move(
                            &id,
                            OstStorageTier::Cold,
                            "size-threshold hot→cold",
                            current_ts,
                        );
                        transitions.push(t);
                    } else {
                        break; // nowhere to put it
                    }
                }
            }
        }

        // Enforce Warm cap.
        while self.used_bytes(OstStorageTier::Warm) > warm_max_bytes {
            let victim = self
                .objects
                .values()
                .filter(|o| o.current_tier == OstStorageTier::Warm)
                .max_by_key(|o| o.size_bytes)
                .map(|o| (o.id.clone(), o.size_bytes));
            match victim {
                None => break,
                Some((id, size)) => {
                    if self.available_bytes(OstStorageTier::Cold) >= size {
                        self.demotions += 1;
                        let t = self.internal_move(
                            &id,
                            OstStorageTier::Cold,
                            "size-threshold warm→cold",
                            current_ts,
                        );
                        transitions.push(t);
                    } else {
                        break;
                    }
                }
            }
        }

        transitions
    }

    fn run_cost_optimized_policy(
        &mut self,
        multiplier: f64,
        current_ts: u64,
    ) -> Vec<OstTierTransition> {
        // Score each non-Cold object: score = access_count / (cost_per_hour * multiplier + 1.0).
        // Low score → good demotion candidate.
        let mut scored: Vec<(String, f64, OstStorageTier, u64)> = self
            .objects
            .values()
            .filter(|o| o.current_tier != OstStorageTier::Cold)
            .map(|o| {
                let score = o.access_count as f64 / (o.cost_per_hour * multiplier + 1.0);
                (o.id.clone(), score, o.current_tier, o.size_bytes)
            })
            .collect();
        // Sort ascending — lowest score first.
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Demote bottom 25% (at least 1 if any candidates).
        let demote_count = (scored.len() as f64 * 0.25).ceil() as usize;
        let mut transitions = Vec::new();

        for (id, _score, current_tier, size) in scored.into_iter().take(demote_count) {
            let target = match current_tier.colder() {
                Some(t) => t,
                None => continue,
            };
            if self.available_bytes(target) >= size {
                self.demotions += 1;
                let t = self.internal_move(&id, target, "cost-optimized demotion", current_ts);
                transitions.push(t);
            }
        }

        transitions
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Tiny deterministic PRNG (xorshift64) — avoids the `rand` crate.
    // -----------------------------------------------------------------------

    fn xorshift64(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn make_object(id: &str, size: u64, access_count: u64, last_accessed: u64) -> TieredObject {
        TieredObject {
            id: id.to_string(),
            size_bytes: size,
            current_tier: OstStorageTier::Hot, // will be overridden by store()
            access_count,
            last_accessed,
            created_at: 0,
            tags: vec![],
            cost_per_hour: 0.001 * size as f64,
        }
    }

    fn default_config() -> OstTierConfig {
        OstTierConfig {
            hot_capacity_bytes: 10_000,
            warm_capacity_bytes: 100_000,
            cold_capacity_bytes: u64::MAX,
            policy: OstTierPolicy::ManualOnly,
            promotion_threshold: 5,
            demotion_interval_us: 1_000_000,
        }
    }

    fn tiering() -> ObjectStorageTiering {
        ObjectStorageTiering::new(default_config())
    }

    // -----------------------------------------------------------------------
    // store() tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_store_small_goes_to_hot() {
        let mut t = tiering();
        let obj = make_object("a", 100, 0, 0);
        let tier = t.store(obj).unwrap();
        assert_eq!(tier, OstStorageTier::Hot);
    }

    #[test]
    fn test_store_fills_hot_then_warm() {
        let mut t = tiering();
        // Fill hot tier completely.
        let obj1 = make_object("big", 10_000, 0, 0);
        assert_eq!(t.store(obj1).unwrap(), OstStorageTier::Hot);
        // Next object overflows hot, should go to warm.
        let obj2 = make_object("next", 100, 0, 0);
        assert_eq!(t.store(obj2).unwrap(), OstStorageTier::Warm);
    }

    #[test]
    fn test_store_fills_all_tiers_cold() {
        let mut cfg = default_config();
        cfg.cold_capacity_bytes = 1_000_000;
        let mut t = ObjectStorageTiering::new(cfg);
        let _ = t.store(make_object("h", 10_000, 0, 0)).unwrap();
        let _ = t.store(make_object("w", 100_000, 0, 0)).unwrap();
        let tier = t.store(make_object("c", 500_000, 0, 0)).unwrap();
        assert_eq!(tier, OstStorageTier::Cold);
    }

    #[test]
    fn test_store_all_full_returns_tier_full_error() {
        let mut cfg = default_config();
        cfg.cold_capacity_bytes = 5;
        let mut t = ObjectStorageTiering::new(cfg);
        let _ = t.store(make_object("h", 10_000, 0, 0)).unwrap();
        let _ = t.store(make_object("w", 100_000, 0, 0)).unwrap();
        let err = t.store(make_object("c", 100, 0, 0)).unwrap_err();
        assert!(matches!(
            err,
            TieringError::TierFull {
                tier: OstStorageTier::Cold,
                ..
            }
        ));
    }

    #[test]
    fn test_store_updates_tier_bytes() {
        let mut t = tiering();
        t.store(make_object("x", 1_000, 0, 0)).unwrap();
        assert_eq!(t.used_bytes(OstStorageTier::Hot), 1_000);
    }

    #[test]
    fn test_store_multiple_objects() {
        let mut t = tiering();
        let mut state = 42u64;
        for i in 0..20 {
            let size = (xorshift64(&mut state) % 400) + 10;
            let id = format!("obj-{i}");
            t.store(make_object(&id, size, 0, 0)).unwrap();
        }
        let s = t.stats();
        assert_eq!(s.hot_objects + s.warm_objects + s.cold_objects, 20);
    }

    // -----------------------------------------------------------------------
    // retrieve() tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_retrieve_found() {
        let mut t = tiering();
        t.store(make_object("r1", 100, 0, 0)).unwrap();
        let obj = t.retrieve("r1", 1000).unwrap();
        assert_eq!(obj.id, "r1");
    }

    #[test]
    fn test_retrieve_updates_access_count() {
        let mut t = tiering();
        t.store(make_object("r2", 100, 0, 0)).unwrap();
        t.retrieve("r2", 1000).unwrap();
        t.retrieve("r2", 2000).unwrap();
        let obj = t.retrieve("r2", 3000).unwrap();
        assert_eq!(obj.access_count, 3);
    }

    #[test]
    fn test_retrieve_updates_last_accessed() {
        let mut t = tiering();
        t.store(make_object("r3", 100, 0, 0)).unwrap();
        t.retrieve("r3", 9999).unwrap();
        let obj = t.retrieve("r3", 9999).unwrap();
        assert_eq!(obj.last_accessed, 9999);
    }

    #[test]
    fn test_retrieve_not_found_error() {
        let mut t = tiering();
        let err = t.retrieve("nope", 0).unwrap_err();
        assert!(matches!(err, TieringError::ObjectNotFound(_)));
    }

    #[test]
    fn test_retrieve_queues_pending_promotion() {
        let mut cfg = default_config();
        cfg.promotion_threshold = 2;
        let mut t = ObjectStorageTiering::new(cfg);
        // Store into warm (fill hot first).
        t.store(make_object("fill-hot", 10_000, 0, 0)).unwrap();
        t.store(make_object("warm-obj", 100, 0, 0)).unwrap();
        // Access warm-obj enough times to cross threshold.
        t.retrieve("warm-obj", 1).unwrap();
        t.retrieve("warm-obj", 2).unwrap();
        t.retrieve("warm-obj", 3).unwrap(); // crosses threshold (>2 = 3)
        assert!(!t.pending_promotions.is_empty());
    }

    // -----------------------------------------------------------------------
    // promote() tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_promote_warm_to_hot() {
        let mut cfg = default_config();
        cfg.hot_capacity_bytes = 10_000;
        let mut t = ObjectStorageTiering::new(cfg);
        // Fill hot completely.
        t.store(make_object("hot-fill", 10_000, 0, 0)).unwrap();
        // "w" overflows to warm.
        t.store(make_object("w", 100, 0, 0)).unwrap();
        assert_eq!(t.objects["w"].current_tier, OstStorageTier::Warm);
        // Evict hot-fill from hot to make room.
        let transitions = t.evict_tier_ts(OstStorageTier::Hot, 10_000, 100);
        assert!(!transitions.is_empty());
        // Now hot is empty; promote "w" from warm to hot.
        let r = t.promote("w", OstStorageTier::Hot, 200);
        assert!(r.is_ok());
        assert_eq!(t.objects["w"].current_tier, OstStorageTier::Hot);
    }

    #[test]
    fn test_promote_cold_to_warm() {
        // Use a config where warm has plenty of room after hot is filled.
        let mut cfg = default_config();
        cfg.hot_capacity_bytes = 10_000;
        cfg.warm_capacity_bytes = 200_000;
        let mut t = ObjectStorageTiering::new(cfg);
        // Fill hot entirely.
        t.store(make_object("hot-fill", 10_000, 0, 0)).unwrap();
        // Partially fill warm.
        t.store(make_object("warm-partial", 50_000, 0, 0)).unwrap();
        // Next object goes to cold (warm still has room but let's force cold by
        // filling warm too).
        t.store(make_object("warm-fill2", 150_000, 0, 0)).unwrap();
        // cold-obj should land in cold since warm is now full.
        let cold_tier = t.store(make_object("cold-obj", 500, 0, 0)).unwrap();
        assert_eq!(cold_tier, OstStorageTier::Cold);
        // Evict from warm to make room.
        t.evict_tier_ts(OstStorageTier::Warm, 600, 500);
        let r = t.promote("cold-obj", OstStorageTier::Warm, 1000);
        assert!(r.is_ok());
        let tr = r.unwrap();
        assert_eq!(tr.from_tier, OstStorageTier::Cold);
        assert_eq!(tr.to_tier, OstStorageTier::Warm);
    }

    #[test]
    fn test_promote_increments_promotion_counter() {
        let mut t = tiering();
        t.store(make_object("hot-fill", 10_000, 0, 0)).unwrap();
        t.store(make_object("warm-obj", 100, 0, 0)).unwrap();
        t.evict_tier_ts(OstStorageTier::Hot, 200, 50);
        t.promote("warm-obj", OstStorageTier::Hot, 100).unwrap();
        assert_eq!(t.stats().promotions, 1);
    }

    #[test]
    fn test_promote_rejects_same_or_lower_tier() {
        let mut t = tiering();
        t.store(make_object("obj", 100, 0, 0)).unwrap();
        let err = t.promote("obj", OstStorageTier::Warm, 0).unwrap_err();
        assert!(matches!(err, TieringError::PolicyConflict(_)));
    }

    #[test]
    fn test_promote_tier_full_error() {
        let mut t = tiering();
        // Fill hot.
        t.store(make_object("fill", 10_000, 0, 0)).unwrap();
        // Store in warm.
        t.store(make_object("w", 100, 0, 0)).unwrap();
        // Hot is full — promote should fail.
        let err = t.promote("w", OstStorageTier::Hot, 0).unwrap_err();
        assert!(matches!(
            err,
            TieringError::TierFull {
                tier: OstStorageTier::Hot,
                ..
            }
        ));
    }

    #[test]
    fn test_promote_not_found() {
        let mut t = tiering();
        let err = t.promote("missing", OstStorageTier::Hot, 0).unwrap_err();
        assert!(matches!(err, TieringError::ObjectNotFound(_)));
    }

    // -----------------------------------------------------------------------
    // demote() tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_demote_hot_to_warm() {
        let mut t = tiering();
        t.store(make_object("obj", 100, 0, 0)).unwrap();
        let tr = t.demote("obj", OstStorageTier::Warm, 10).unwrap();
        assert_eq!(tr.from_tier, OstStorageTier::Hot);
        assert_eq!(tr.to_tier, OstStorageTier::Warm);
    }

    #[test]
    fn test_demote_hot_to_cold() {
        let mut t = tiering();
        t.store(make_object("obj", 100, 0, 0)).unwrap();
        let tr = t.demote("obj", OstStorageTier::Cold, 10).unwrap();
        assert_eq!(tr.to_tier, OstStorageTier::Cold);
    }

    #[test]
    fn test_demote_increments_demotion_counter() {
        let mut t = tiering();
        t.store(make_object("obj", 100, 0, 0)).unwrap();
        t.demote("obj", OstStorageTier::Cold, 0).unwrap();
        assert_eq!(t.stats().demotions, 1);
    }

    #[test]
    fn test_demote_rejects_same_or_higher_tier() {
        let mut t = tiering();
        t.store(make_object("hot-fill", 10_000, 0, 0)).unwrap();
        t.store(make_object("warm-obj", 100, 0, 0)).unwrap();
        // warm-obj is in Warm; demoting to Hot should fail.
        let err = t.demote("warm-obj", OstStorageTier::Hot, 0).unwrap_err();
        assert!(matches!(err, TieringError::PolicyConflict(_)));
    }

    #[test]
    fn test_demote_not_found() {
        let mut t = tiering();
        let err = t.demote("ghost", OstStorageTier::Cold, 0).unwrap_err();
        assert!(matches!(err, TieringError::ObjectNotFound(_)));
    }

    #[test]
    fn test_demote_tier_full() {
        let mut cfg = default_config();
        cfg.warm_capacity_bytes = 50; // tiny warm
        let mut t = ObjectStorageTiering::new(cfg);
        t.store(make_object("obj", 100, 0, 0)).unwrap();
        let err = t.demote("obj", OstStorageTier::Warm, 0).unwrap_err();
        assert!(matches!(
            err,
            TieringError::TierFull {
                tier: OstStorageTier::Warm,
                ..
            }
        ));
    }

    // -----------------------------------------------------------------------
    // run_policy() — AccessFrequency
    // -----------------------------------------------------------------------

    #[test]
    fn test_run_policy_access_frequency_promotes() {
        let mut cfg = default_config();
        cfg.policy = OstTierPolicy::AccessFrequency(3);
        let mut t = ObjectStorageTiering::new(cfg);
        t.store(make_object("fill", 9_900, 0, 0)).unwrap();
        t.store(make_object("warm", 50, 0, 0)).unwrap();
        // Give "warm" many accesses (but don't promote yet).
        if let Some(obj) = t.objects.get_mut("warm") {
            obj.access_count = 10;
        }
        // Evict fill so hot has room.
        t.evict_tier_ts(OstStorageTier::Hot, 10_000, 100);
        let transitions = t.run_policy(200);
        let promoted = transitions
            .iter()
            .any(|tr| tr.object_id == "warm" && tr.to_tier == OstStorageTier::Hot);
        assert!(promoted, "expected warm object to be promoted to hot");
    }

    #[test]
    fn test_run_policy_access_frequency_no_promotion_if_already_hot() {
        let mut cfg = default_config();
        cfg.policy = OstTierPolicy::AccessFrequency(3);
        let mut t = ObjectStorageTiering::new(cfg);
        t.store(make_object("obj", 100, 10, 0)).unwrap();
        let transitions = t.run_policy(0);
        assert!(transitions.iter().all(|tr| tr.object_id != "obj"
            || tr.to_tier != OstStorageTier::Hot
            || tr.from_tier == OstStorageTier::Hot));
    }

    // -----------------------------------------------------------------------
    // run_policy() — RecencyBased
    // -----------------------------------------------------------------------

    #[test]
    fn test_run_policy_recency_demotes_stale() {
        let mut cfg = default_config();
        cfg.policy = OstTierPolicy::RecencyBased(1_000_000); // 1 second window
        let mut t = ObjectStorageTiering::new(cfg);
        // Object last accessed at ts=0, current ts=2_000_000 → stale.
        t.store(make_object("stale", 100, 0, 0)).unwrap();
        let transitions = t.run_policy(2_000_000);
        assert!(transitions
            .iter()
            .any(|tr| tr.object_id == "stale" && tr.to_tier == OstStorageTier::Cold));
    }

    #[test]
    fn test_run_policy_recency_keeps_fresh() {
        let mut cfg = default_config();
        cfg.policy = OstTierPolicy::RecencyBased(1_000_000);
        let mut t = ObjectStorageTiering::new(cfg);
        // Fresh: last_accessed = 1_500_000, current = 2_000_000, window = 1_000_000 → within window.
        let mut obj = make_object("fresh", 100, 0, 1_500_000);
        obj.last_accessed = 1_500_000;
        t.store(obj).unwrap();
        let transitions = t.run_policy(2_000_000);
        assert!(transitions
            .iter()
            .all(|tr| tr.object_id != "fresh" || tr.to_tier != OstStorageTier::Cold));
    }

    #[test]
    fn test_run_policy_recency_skips_already_cold() {
        let mut cfg = default_config();
        cfg.policy = OstTierPolicy::RecencyBased(100);
        let mut t = ObjectStorageTiering::new(cfg);
        t.store(make_object("hot-fill", 10_000, 0, 0)).unwrap();
        t.store(make_object("warm-fill", 100_000, 0, 0)).unwrap();
        t.store(make_object("cold-obj", 100, 0, 0)).unwrap();
        // cold-obj is already cold; run_policy should not produce a transition for it.
        let before_demotions = t.demotions;
        let transitions = t.run_policy(999_999_999);
        let cold_moved: Vec<_> = transitions
            .iter()
            .filter(|tr| tr.object_id == "cold-obj")
            .collect();
        assert!(cold_moved.is_empty() || t.demotions == before_demotions);
    }

    // -----------------------------------------------------------------------
    // run_policy() — SizeThreshold
    // -----------------------------------------------------------------------

    #[test]
    fn test_run_policy_size_threshold_demotes_largest() {
        let mut cfg = default_config();
        cfg.policy = OstTierPolicy::SizeThreshold {
            hot_max_bytes: 5_000,
            warm_max_bytes: 50_000,
        };
        let mut t = ObjectStorageTiering::new(cfg);
        // Put 3 objects in hot. hot_capacity=10_000 but threshold=5_000.
        t.store(make_object("small", 1_000, 0, 0)).unwrap();
        t.store(make_object("medium", 2_000, 0, 0)).unwrap();
        t.store(make_object("large", 3_000, 0, 0)).unwrap();
        // Hot used = 6_000 > 5_000 threshold.
        let transitions = t.run_policy(0);
        // Largest object should have been demoted.
        assert!(transitions.iter().any(|tr| tr.object_id == "large"));
    }

    #[test]
    fn test_run_policy_size_threshold_no_action_if_within_limits() {
        let mut cfg = default_config();
        cfg.policy = OstTierPolicy::SizeThreshold {
            hot_max_bytes: 10_000,
            warm_max_bytes: 100_000,
        };
        let mut t = ObjectStorageTiering::new(cfg);
        t.store(make_object("a", 100, 0, 0)).unwrap();
        let transitions = t.run_policy(0);
        assert!(transitions.is_empty());
    }

    // -----------------------------------------------------------------------
    // run_policy() — CostOptimized
    // -----------------------------------------------------------------------

    #[test]
    fn test_run_policy_cost_optimized_demotes_low_access() {
        let mut cfg = default_config();
        cfg.policy = OstTierPolicy::CostOptimized(1.0);
        let mut t = ObjectStorageTiering::new(cfg);
        // High-value object.
        let mut high = make_object("high", 500, 0, 0);
        high.access_count = 1000;
        t.store(high).unwrap();
        // Low-value object.
        let mut low = make_object("low", 500, 0, 0);
        low.access_count = 1;
        t.store(low).unwrap();
        let transitions = t.run_policy(0);
        // At least one demotion should have occurred.
        assert!(!transitions.is_empty());
    }

    #[test]
    fn test_run_policy_manual_only_no_transitions() {
        let mut cfg = default_config();
        cfg.policy = OstTierPolicy::ManualOnly;
        let mut t = ObjectStorageTiering::new(cfg);
        t.store(make_object("a", 100, 100, 0)).unwrap();
        let transitions = t.run_policy(999_999_999);
        assert!(transitions.is_empty());
    }

    // -----------------------------------------------------------------------
    // evict_tier() tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_evict_tier_lru_order() {
        let mut t = tiering();
        // Three hot objects with different last_accessed times.
        let mut o1 = make_object("oldest", 1_000, 0, 100);
        o1.last_accessed = 100;
        let mut o2 = make_object("middle", 1_000, 0, 200);
        o2.last_accessed = 200;
        let mut o3 = make_object("newest", 1_000, 0, 300);
        o3.last_accessed = 300;
        t.store(o1).unwrap();
        t.store(o2).unwrap();
        t.store(o3).unwrap();
        let transitions = t.evict_tier_ts(OstStorageTier::Hot, 1_001, 400);
        // Should evict "oldest" first.
        assert!(!transitions.is_empty());
        assert_eq!(transitions[0].object_id, "oldest");
    }

    #[test]
    fn test_evict_tier_stops_when_enough_freed() {
        let mut t = tiering();
        for i in 0..5 {
            t.store(make_object(&format!("o{i}"), 1_000, 0, i as u64))
                .unwrap();
        }
        let transitions = t.evict_tier_ts(OstStorageTier::Hot, 1_500, 100);
        // Should evict exactly 2 objects (2×1000 ≥ 1500).
        assert_eq!(transitions.len(), 2);
    }

    #[test]
    fn test_evict_cold_tier_no_op() {
        let mut t = tiering();
        t.store(make_object("fill-hot", 10_000, 0, 0)).unwrap();
        t.store(make_object("fill-warm", 100_000, 0, 0)).unwrap();
        t.store(make_object("c", 100, 0, 0)).unwrap();
        let transitions = t.evict_tier(OstStorageTier::Cold, 100);
        assert!(transitions.is_empty()); // Cold has nowhere colder.
    }

    #[test]
    fn test_evict_tier_increments_demotion_counter() {
        let mut t = tiering();
        t.store(make_object("a", 1_000, 0, 1)).unwrap();
        t.store(make_object("b", 1_000, 0, 2)).unwrap();
        t.evict_tier_ts(OstStorageTier::Hot, 1_000, 0);
        assert!(t.stats().demotions > 0);
    }

    // -----------------------------------------------------------------------
    // tier_objects() tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tier_objects_returns_correct_set() {
        let mut t = tiering(); // hot=10_000, warm=100_000
        t.store(make_object("h1", 100, 0, 0)).unwrap();
        t.store(make_object("h2", 100, 0, 0)).unwrap();
        // Fill the rest of hot so the next object must go to warm.
        t.store(make_object("fill", 9_800, 0, 0)).unwrap();
        // hot is now exactly full; this must land in warm.
        t.store(make_object("w1", 100, 0, 0)).unwrap();
        let hot = t.tier_objects(OstStorageTier::Hot);
        assert!(hot.len() >= 2);
        let warm = t.tier_objects(OstStorageTier::Warm);
        assert!(!warm.is_empty());
    }

    #[test]
    fn test_tier_objects_empty_for_unused_tier() {
        let t = tiering();
        assert!(t.tier_objects(OstStorageTier::Cold).is_empty());
    }

    // -----------------------------------------------------------------------
    // transition_history() tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_transition_history_records_demotions() {
        let mut t = tiering();
        t.store(make_object("obj", 100, 0, 0)).unwrap();
        t.demote("obj", OstStorageTier::Warm, 50).unwrap();
        let history = t.transition_history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].from_tier, OstStorageTier::Hot);
        assert_eq!(history[0].to_tier, OstStorageTier::Warm);
    }

    #[test]
    fn test_transition_history_capped_at_500() {
        let mut t = tiering();
        for i in 0u64..600 {
            let id = format!("obj-{i}");
            t.store(make_object(&id, 50, 0, 0)).unwrap();
            t.demote(&id, OstStorageTier::Warm, i).unwrap();
        }
        assert_eq!(t.transition_history().len(), 500);
    }

    #[test]
    fn test_transition_history_stores_reason() {
        let mut t = tiering();
        t.store(make_object("x", 100, 0, 0)).unwrap();
        t.demote("x", OstStorageTier::Cold, 1).unwrap();
        let history = t.transition_history();
        assert!(history[0].reason.contains("demotion"));
    }

    #[test]
    fn test_transition_history_stores_timestamp() {
        let mut t = tiering();
        t.store(make_object("x", 100, 0, 0)).unwrap();
        t.demote("x", OstStorageTier::Cold, 12345).unwrap();
        let history = t.transition_history();
        assert_eq!(history[0].transitioned_at, 12345);
    }

    #[test]
    fn test_transition_history_bytes_moved() {
        let mut t = tiering();
        t.store(make_object("big", 7_777, 0, 0)).unwrap();
        t.demote("big", OstStorageTier::Warm, 0).unwrap();
        let history = t.transition_history();
        assert_eq!(history[0].bytes_moved, 7_777);
    }

    // -----------------------------------------------------------------------
    // stats() tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_counts_per_tier() {
        let mut t = tiering();
        t.store(make_object("h1", 100, 0, 0)).unwrap();
        t.store(make_object("h2", 200, 0, 0)).unwrap();
        let s = t.stats();
        assert_eq!(s.hot_objects, 2);
        assert_eq!(s.hot_bytes, 300);
    }

    #[test]
    fn test_stats_total_cost() {
        let mut t = tiering();
        let mut obj = make_object("c", 100, 0, 0);
        obj.cost_per_hour = 1.5;
        t.store(obj).unwrap();
        let s = t.stats();
        assert!((s.total_cost_per_hour - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_stats_promotions_and_demotions() {
        let mut t = tiering(); // hot=10_000
                               // Fill hot completely.
        t.store(make_object("fill", 10_000, 0, 0)).unwrap();
        // "w" must land in warm.
        t.store(make_object("w", 100, 0, 0)).unwrap();
        // Evict "fill" from hot to warm, making room in hot.
        t.evict_tier_ts(OstStorageTier::Hot, 10_000, 0);
        // Now hot is empty; promote "w" from warm to hot.
        t.promote("w", OstStorageTier::Hot, 1).unwrap();
        // Demote "w" to cold.
        t.demote("w", OstStorageTier::Cold, 2).unwrap();
        let s = t.stats();
        assert!(s.promotions >= 1);
        assert!(s.demotions >= 2);
    }

    #[test]
    fn test_stats_empty_tiering() {
        let t = tiering();
        let s = t.stats();
        assert_eq!(s.hot_objects, 0);
        assert_eq!(s.warm_objects, 0);
        assert_eq!(s.cold_objects, 0);
        assert_eq!(s.hot_bytes, 0);
        assert_eq!(s.promotions, 0);
        assert_eq!(s.demotions, 0);
    }

    // -----------------------------------------------------------------------
    // Capacity enforcement tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_capacity_hot_never_exceeded() {
        let mut t = tiering(); // hot=10_000
        for i in 0..20 {
            let id = format!("cap-{i}");
            let _ = t.store(make_object(&id, 1_000, 0, 0));
        }
        assert!(t.used_bytes(OstStorageTier::Hot) <= 10_000);
    }

    #[test]
    fn test_capacity_warm_never_exceeded() {
        let mut t = tiering(); // warm=100_000
        for i in 0..200 {
            let id = format!("w-{i}");
            let _ = t.store(make_object(&id, 1_000, 0, 0));
        }
        assert!(t.used_bytes(OstStorageTier::Warm) <= 100_000);
    }

    #[test]
    fn test_available_bytes_decreases_after_store() {
        let mut t = tiering();
        let before = t.available_bytes(OstStorageTier::Hot);
        t.store(make_object("x", 500, 0, 0)).unwrap();
        assert_eq!(t.available_bytes(OstStorageTier::Hot), before - 500);
    }

    #[test]
    fn test_available_bytes_increases_after_demote() {
        let mut t = tiering();
        t.store(make_object("x", 500, 0, 0)).unwrap();
        let before_hot = t.available_bytes(OstStorageTier::Hot);
        t.demote("x", OstStorageTier::Warm, 0).unwrap();
        assert_eq!(t.available_bytes(OstStorageTier::Hot), before_hot + 500);
    }

    // -----------------------------------------------------------------------
    // Error-case tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tiering_error_display_object_not_found() {
        let e = TieringError::ObjectNotFound("abc".to_string());
        assert!(e.to_string().contains("abc"));
    }

    #[test]
    fn test_tiering_error_display_tier_full() {
        let e = TieringError::TierFull {
            tier: OstStorageTier::Hot,
            needed: 100,
            available: 50,
        };
        let s = e.to_string();
        assert!(s.contains("hot"));
        assert!(s.contains("100"));
    }

    #[test]
    fn test_tiering_error_display_policy_conflict() {
        let e = TieringError::PolicyConflict("test msg".to_string());
        assert!(e.to_string().contains("test msg"));
    }

    #[test]
    fn test_tiering_error_display_invalid_config() {
        let e = TieringError::InvalidConfiguration("bad".to_string());
        assert!(e.to_string().contains("bad"));
    }

    // -----------------------------------------------------------------------
    // OstStorageTier helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_tier_colder() {
        assert_eq!(OstStorageTier::Hot.colder(), Some(OstStorageTier::Warm));
        assert_eq!(OstStorageTier::Warm.colder(), Some(OstStorageTier::Cold));
        assert_eq!(OstStorageTier::Cold.colder(), None);
    }

    #[test]
    fn test_tier_hotter() {
        assert_eq!(OstStorageTier::Cold.hotter(), Some(OstStorageTier::Warm));
        assert_eq!(OstStorageTier::Warm.hotter(), Some(OstStorageTier::Hot));
        assert_eq!(OstStorageTier::Hot.hotter(), None);
    }

    #[test]
    fn test_tier_rank_ordering() {
        assert!(OstStorageTier::Hot.rank() < OstStorageTier::Warm.rank());
        assert!(OstStorageTier::Warm.rank() < OstStorageTier::Cold.rank());
    }

    #[test]
    fn test_tier_name() {
        assert_eq!(OstStorageTier::Hot.name(), "hot");
        assert_eq!(OstStorageTier::Warm.name(), "warm");
        assert_eq!(OstStorageTier::Cold.name(), "cold");
    }

    #[test]
    fn test_tier_display() {
        assert_eq!(OstStorageTier::Hot.to_string(), "hot");
    }

    // -----------------------------------------------------------------------
    // Integration / stress tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_round_trip_store_retrieve_demote_promote() {
        let mut cfg = default_config();
        cfg.hot_capacity_bytes = 100_000;
        cfg.warm_capacity_bytes = 1_000_000;
        let mut t = ObjectStorageTiering::new(cfg);
        let obj = make_object("rt", 1_000, 0, 0);
        let tier = t.store(obj).unwrap();
        assert_eq!(tier, OstStorageTier::Hot);
        t.retrieve("rt", 500).unwrap();
        t.demote("rt", OstStorageTier::Warm, 1000).unwrap();
        let obj_ref = t.retrieve("rt", 1500).unwrap();
        assert_eq!(obj_ref.current_tier, OstStorageTier::Warm);
        t.evict_tier_ts(OstStorageTier::Hot, 1, 2000); // make room
        t.promote("rt", OstStorageTier::Hot, 2500).unwrap();
        let obj_ref2 = t.retrieve("rt", 3000).unwrap();
        assert_eq!(obj_ref2.current_tier, OstStorageTier::Hot);
    }

    #[test]
    fn test_stress_many_objects_stats_consistent() {
        let mut cfg = default_config();
        cfg.hot_capacity_bytes = 50_000;
        cfg.warm_capacity_bytes = 500_000;
        let mut t = ObjectStorageTiering::new(cfg);
        let mut state = 12345u64;
        for i in 0..100 {
            let size = xorshift64(&mut state) % 1_000 + 100;
            let _ = t.store(make_object(&format!("s{i}"), size, 0, 0));
        }
        let s = t.stats();
        let total_objects = s.hot_objects + s.warm_objects + s.cold_objects;
        let total_bytes = s.hot_bytes + s.warm_bytes + s.cold_bytes;
        // Verify byte accounting is consistent.
        assert_eq!(
            t.used_bytes(OstStorageTier::Hot)
                + t.used_bytes(OstStorageTier::Warm)
                + t.used_bytes(OstStorageTier::Cold),
            total_bytes
        );
        assert_eq!(total_objects, t.objects.len());
    }

    #[test]
    fn test_default_config_sensible_capacities() {
        let cfg = OstTierConfig::default();
        assert!(cfg.hot_capacity_bytes > 0);
        assert!(cfg.warm_capacity_bytes > cfg.hot_capacity_bytes);
        assert_eq!(cfg.cold_capacity_bytes, u64::MAX);
    }
}
