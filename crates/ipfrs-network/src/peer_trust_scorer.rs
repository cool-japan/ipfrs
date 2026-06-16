//! Composite trust scoring system for network peers.
//!
//! [`PeerTrustScorer`] combines five orthogonal trust dimensions —
//! uptime, content validity, protocol compliance, response latency, and data
//! availability — into a single weighted composite score.  Scores are bounded
//! to `[min_score, max_score]`, subject to time-based decay, and mapped to a
//! human-readable [`TrustBand`].
//!
//! # Example
//!
//! ```rust
//! use ipfrs_network::peer_trust_scorer::{
//!     PeerTrustScorer, TrustConfig, TrustDimension, TrustEvent, TrustBand,
//! };
//!
//! let config = TrustConfig::default();
//! let mut scorer = PeerTrustScorer::new(config, 1000);
//!
//! let event = TrustEvent {
//!     peer_id: "peer-A".to_string(),
//!     dimension: TrustDimension::ContentValidity,
//!     delta: 0.1,
//!     timestamp: 0,
//!     description: "Valid content received".to_string(),
//! };
//! scorer.record_event(event);
//!
//! let profile = scorer.get_profile("peer-A").expect("profile must exist");
//! assert!(profile.composite_score >= 0.5);
//! ```

use std::collections::{HashMap, VecDeque};

// ─────────────────────────────────────────────────────────────────────────────
// Trust dimension
// ─────────────────────────────────────────────────────────────────────────────

/// The five orthogonal dimensions along which a peer is evaluated.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TrustDimension {
    /// Fraction of time the peer is reachable.
    Uptime,
    /// How reliably the peer sends well-formed, valid content.
    ContentValidity,
    /// Adherence to the IPFRS protocol specification.
    ProtocolCompliance,
    /// Response latency relative to network baseline.
    ResponseLatency,
    /// Probability that announced data is actually retrievable.
    DataAvailability,
}

impl TrustDimension {
    /// Return the canonical string key used as a `HashMap` key.
    pub fn as_key(&self) -> &'static str {
        match self {
            TrustDimension::Uptime => "Uptime",
            TrustDimension::ContentValidity => "ContentValidity",
            TrustDimension::ProtocolCompliance => "ProtocolCompliance",
            TrustDimension::ResponseLatency => "ResponseLatency",
            TrustDimension::DataAvailability => "DataAvailability",
        }
    }

    /// All dimension variants in a stable order.
    pub fn all() -> &'static [TrustDimension] {
        &[
            TrustDimension::Uptime,
            TrustDimension::ContentValidity,
            TrustDimension::ProtocolCompliance,
            TrustDimension::ResponseLatency,
            TrustDimension::DataAvailability,
        ]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Trust event
// ─────────────────────────────────────────────────────────────────────────────

/// A single observed behavioural signal for a peer.
///
/// `delta > 0` rewards good behaviour; `delta < 0` penalises bad behaviour.
#[derive(Debug, Clone)]
pub struct TrustEvent {
    /// Peer identifier (opaque string, not a `libp2p::PeerId` to keep the
    /// module dependency-free).
    pub peer_id: String,
    /// Which trust dimension this event affects.
    pub dimension: TrustDimension,
    /// Score change in `[min_score - max_score, max_score - min_score]`.
    pub delta: f64,
    /// Unix-epoch milliseconds at which the event was observed.
    pub timestamp: u64,
    /// Human-readable description for audit logs.
    pub description: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Trust band
// ─────────────────────────────────────────────────────────────────────────────

/// Coarse-grained label derived from the composite trust score.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrustBand {
    /// Score < 0.10 — peer should be rejected.
    Blocked = 0,
    /// Score in [0.10, 0.30) — proceed with extreme caution.
    Untrusted = 1,
    /// Score in [0.30, 0.60) — acceptable but unverified.
    Neutral = 2,
    /// Score in [0.60, 0.85) — reliably well-behaved.
    Trusted = 3,
    /// Score ≥ 0.85 — exemplary peer.
    HighlyTrusted = 4,
}

impl TrustBand {
    /// Derive the appropriate band from a composite score.
    pub fn from_score(score: f64) -> TrustBand {
        if score < 0.10 {
            TrustBand::Blocked
        } else if score < 0.30 {
            TrustBand::Untrusted
        } else if score < 0.60 {
            TrustBand::Neutral
        } else if score < 0.85 {
            TrustBand::Trusted
        } else {
            TrustBand::HighlyTrusted
        }
    }

    /// Numeric ordinal (same as the discriminant value).
    pub fn ordinal(&self) -> u8 {
        match self {
            TrustBand::Blocked => 0,
            TrustBand::Untrusted => 1,
            TrustBand::Neutral => 2,
            TrustBand::Trusted => 3,
            TrustBand::HighlyTrusted => 4,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`PeerTrustScorer`].
#[derive(Debug, Clone)]
pub struct TrustConfig {
    /// Per-dimension weight used when computing the composite score.
    /// Keys must match [`TrustDimension::as_key()`].
    pub dimension_weights: HashMap<String, f64>,
    /// Score decay per hour (multiplicative, applied to each dimension score).
    pub decay_rate: f64,
    /// Lower bound for any dimension score.
    pub min_score: f64,
    /// Upper bound for any dimension score.
    pub max_score: f64,
    /// Number of events returned per page by [`PeerTrustScorer::peer_events`].
    pub events_per_page: usize,
}

impl Default for TrustConfig {
    fn default() -> Self {
        let mut weights = HashMap::new();
        weights.insert(TrustDimension::Uptime.as_key().to_string(), 0.20);
        weights.insert(TrustDimension::ContentValidity.as_key().to_string(), 0.30);
        weights.insert(
            TrustDimension::ProtocolCompliance.as_key().to_string(),
            0.20,
        );
        weights.insert(TrustDimension::ResponseLatency.as_key().to_string(), 0.15);
        weights.insert(TrustDimension::DataAvailability.as_key().to_string(), 0.15);
        TrustConfig {
            dimension_weights: weights,
            decay_rate: 0.01,
            min_score: 0.0,
            max_score: 1.0,
            events_per_page: 50,
        }
    }
}

impl TrustConfig {
    /// Validate that all five dimensions are present and weights sum to ≈ 1.0.
    pub fn is_valid(&self) -> bool {
        let total: f64 = self.dimension_weights.values().sum();
        (total - 1.0).abs() < 1e-9
            && TrustDimension::all()
                .iter()
                .all(|d| self.dimension_weights.contains_key(d.as_key()))
    }

    /// Retrieve the weight for `dimension`, falling back to `0.0`.
    pub fn weight_for(&self, dimension: &TrustDimension) -> f64 {
        *self
            .dimension_weights
            .get(dimension.as_key())
            .unwrap_or(&0.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Peer trust profile
// ─────────────────────────────────────────────────────────────────────────────

/// The current trust state of a single peer.
#[derive(Debug, Clone)]
pub struct PeerTrustProfile {
    /// Opaque peer identifier.
    pub peer_id: String,
    /// Per-dimension scores, keyed by [`TrustDimension::as_key()`].
    pub scores: HashMap<String, f64>,
    /// Weighted composite of `scores`.
    pub composite_score: f64,
    /// Total number of events recorded for this peer.
    pub event_count: u64,
    /// Timestamp (ms) of the last update.
    pub last_updated: u64,
    /// Band derived from `composite_score`.
    pub band: TrustBand,
}

impl PeerTrustProfile {
    /// Construct a fresh profile with all dimensions initialised to `0.5`.
    fn new(peer_id: &str, now: u64) -> Self {
        let mut scores = HashMap::new();
        for d in TrustDimension::all() {
            scores.insert(d.as_key().to_string(), 0.5);
        }
        PeerTrustProfile {
            peer_id: peer_id.to_string(),
            scores,
            composite_score: 0.5,
            event_count: 0,
            last_updated: now,
            band: TrustBand::Neutral,
        }
    }

    /// Recompute `composite_score` and `band` from the current per-dimension
    /// scores using `config`'s weights.
    fn recompute_composite(&mut self, config: &TrustConfig) {
        let mut total_weight = 0.0_f64;
        let mut weighted_sum = 0.0_f64;
        for d in TrustDimension::all() {
            let weight = config.weight_for(d);
            let score = self.scores.get(d.as_key()).copied().unwrap_or(0.5);
            weighted_sum += weight * score;
            total_weight += weight;
        }
        self.composite_score = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.5
        };
        self.band = TrustBand::from_score(self.composite_score);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scorer statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics for the entire scorer.
#[derive(Debug, Clone)]
pub struct TrustScorerStats {
    /// Number of distinct peers tracked.
    pub total_peers: usize,
    /// Peers whose band is ≥ [`TrustBand::Trusted`].
    pub trusted_count: usize,
    /// Peers whose band is [`TrustBand::Blocked`].
    pub blocked_count: usize,
    /// Cumulative number of events ever recorded.
    pub total_events: u64,
    /// Mean composite score across all profiles (`0.5` when no peers).
    pub avg_composite_score: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// PeerTrustScorer
// ─────────────────────────────────────────────────────────────────────────────

/// Composite trust scoring engine for a set of peers.
///
/// Maintains one [`PeerTrustProfile`] per peer and a bounded event log.
/// All operations are synchronous and operate in O(1) or O(p) where p is the
/// number of distinct peers.
pub struct PeerTrustScorer {
    /// Scorer configuration (weights, decay, bounds, pagination).
    pub config: TrustConfig,
    /// One profile per peer identifier.
    profiles: HashMap<String, PeerTrustProfile>,
    /// Bounded ring-buffer of recent events (oldest evicted when full).
    event_log: VecDeque<TrustEvent>,
    /// Maximum number of events retained in `event_log`.
    max_events: usize,
    /// Monotonically increasing counter of all events ever recorded.
    total_events: u64,
}

impl PeerTrustScorer {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Create a new scorer with the given configuration and event-log capacity.
    ///
    /// `max_events` controls how many recent [`TrustEvent`]s are kept in
    /// memory; older events are evicted in FIFO order.
    pub fn new(config: TrustConfig, max_events: usize) -> Self {
        PeerTrustScorer {
            config,
            profiles: HashMap::new(),
            event_log: VecDeque::with_capacity(max_events.min(4096)),
            max_events: max_events.max(1),
            total_events: 0,
        }
    }

    // ── Profile access ────────────────────────────────────────────────────────

    /// Return the canonical string key for `dimension`.
    pub fn dimension_key(dimension: &TrustDimension) -> String {
        dimension.as_key().to_string()
    }

    /// Return a mutable reference to the profile for `peer_id`, creating one
    /// initialised to `0.5` on all dimensions if it does not yet exist.
    pub fn get_or_create_profile(&mut self, peer_id: &str, now: u64) -> &mut PeerTrustProfile {
        self.profiles
            .entry(peer_id.to_string())
            .or_insert_with(|| PeerTrustProfile::new(peer_id, now))
    }

    /// Return a shared reference to the profile for `peer_id`, or `None`.
    pub fn get_profile(&self, peer_id: &str) -> Option<&PeerTrustProfile> {
        self.profiles.get(peer_id)
    }

    // ── Event recording ───────────────────────────────────────────────────────

    /// Record a behavioural event for a peer.
    ///
    /// The event is:
    /// 1. Applied as a clamped score delta to the affected dimension.
    /// 2. Used to recompute the composite score and trust band.
    /// 3. Appended to the bounded event log (oldest entry evicted if full).
    pub fn record_event(&mut self, event: TrustEvent) {
        let peer_id = event.peer_id.clone();
        let timestamp = event.timestamp;
        let dimension_key = event.dimension.as_key().to_string();
        let delta = event.delta;
        let min = self.config.min_score;
        let max = self.config.max_score;

        // Get or create profile and mutate it.
        let profile = self.get_or_create_profile(&peer_id, timestamp);
        let current = *profile.scores.get(&dimension_key).unwrap_or(&0.5);
        let updated = (current + delta).clamp(min, max);
        profile.scores.insert(dimension_key, updated);
        profile.event_count += 1;
        profile.last_updated = timestamp;

        // Recompute composite — borrow config separately.
        let config = self.config.clone();
        let profile = self.profiles.get_mut(&peer_id).unwrap_or_else(|| {
            // This branch is unreachable because we just inserted the profile,
            // but the compiler cannot verify that through the borrow above.
            panic!("profile vanished unexpectedly")
        });
        profile.recompute_composite(&config);

        // Append event to bounded log.
        if self.event_log.len() >= self.max_events {
            self.event_log.pop_front();
        }
        self.event_log.push_back(event);
        self.total_events += 1;
    }

    // ── Decay ─────────────────────────────────────────────────────────────────

    /// Apply exponential decay to all profiles.
    ///
    /// For each profile the elapsed time since `last_updated` is converted to
    /// fractional hours and used to scale each dimension score:
    ///
    /// ```text
    /// score_new = score_old × max(0, 1 − decay_rate × elapsed_hours)
    /// ```
    ///
    /// After scaling, the composite and band are recomputed and `last_updated`
    /// is set to `now`.
    pub fn apply_decay(&mut self, now: u64) {
        let decay_rate = self.config.decay_rate;
        let min = self.config.min_score;
        let max = self.config.max_score;
        let config = self.config.clone();

        for profile in self.profiles.values_mut() {
            let elapsed_ms = now.saturating_sub(profile.last_updated);
            let elapsed_hours = elapsed_ms as f64 / 3_600_000.0;
            let factor = (1.0 - decay_rate * elapsed_hours).max(0.0);

            for score in profile.scores.values_mut() {
                *score = (*score * factor).clamp(min, max);
            }
            profile.recompute_composite(&config);
            profile.last_updated = now;
        }
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Return all profiles whose [`TrustBand`] is ≥ `min_band`, sorted by
    /// `composite_score` descending.
    pub fn trusted_peers(&self, min_band: TrustBand) -> Vec<&PeerTrustProfile> {
        let mut result: Vec<&PeerTrustProfile> = self
            .profiles
            .values()
            .filter(|p| p.band >= min_band)
            .collect();
        result.sort_by(|a, b| {
            b.composite_score
                .partial_cmp(&a.composite_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result
    }

    /// Return the peer IDs of all peers whose band is [`TrustBand::Blocked`].
    pub fn blocked_peers(&self) -> Vec<&str> {
        self.profiles
            .values()
            .filter(|p| p.band == TrustBand::Blocked)
            .map(|p| p.peer_id.as_str())
            .collect()
    }

    /// Return a paginated slice of events for `peer_id` from the event log.
    ///
    /// Page 0 is the oldest matching events; pages are `config.events_per_page`
    /// entries wide.
    pub fn peer_events(&self, peer_id: &str, page: usize) -> Vec<&TrustEvent> {
        let per_page = self.config.events_per_page;
        let skip = page * per_page;
        self.event_log
            .iter()
            .filter(|e| e.peer_id == peer_id)
            .skip(skip)
            .take(per_page)
            .collect()
    }

    // ── Administrative actions ────────────────────────────────────────────────

    /// Set all dimension scores for `peer_id` to `0.0` (effectively blocks the
    /// peer regardless of subsequent events until explicitly rehabilitated).
    pub fn ban_peer(&mut self, peer_id: &str, now: u64) {
        let config = self.config.clone();
        let profile = self.get_or_create_profile(peer_id, now);
        for score in profile.scores.values_mut() {
            *score = 0.0;
        }
        profile.composite_score = 0.0;
        profile.band = TrustBand::Blocked;
        profile.last_updated = now;
        // Trigger full recompute to be consistent with config weights.
        let profile = self
            .profiles
            .get_mut(peer_id)
            .expect("profile was just created");
        profile.recompute_composite(&config);
        // Composite might round above 0; force band after.
        profile.band = TrustBand::Blocked;
    }

    /// Reset all dimension scores for `peer_id` to `0.3` (Untrusted band),
    /// giving the peer a fresh but cautious start.
    pub fn rehabilitate_peer(&mut self, peer_id: &str, now: u64) {
        let config = self.config.clone();
        let profile = self.get_or_create_profile(peer_id, now);
        for score in profile.scores.values_mut() {
            *score = 0.3;
        }
        profile.last_updated = now;
        let profile = self
            .profiles
            .get_mut(peer_id)
            .expect("profile was just created");
        profile.recompute_composite(&config);
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    /// Compute aggregate statistics over all tracked profiles.
    pub fn scorer_stats(&self) -> TrustScorerStats {
        let total_peers = self.profiles.len();
        let mut trusted_count = 0_usize;
        let mut blocked_count = 0_usize;
        let mut score_sum = 0.0_f64;

        for profile in self.profiles.values() {
            if profile.band >= TrustBand::Trusted {
                trusted_count += 1;
            }
            if profile.band == TrustBand::Blocked {
                blocked_count += 1;
            }
            score_sum += profile.composite_score;
        }

        let avg_composite_score = if total_peers > 0 {
            score_sum / total_peers as f64
        } else {
            0.5
        };

        TrustScorerStats {
            total_peers,
            trusted_count,
            blocked_count,
            total_events: self.total_events,
            avg_composite_score,
        }
    }

    // ── Accessors (read-only) ─────────────────────────────────────────────────

    /// Number of distinct peers currently tracked.
    pub fn peer_count(&self) -> usize {
        self.profiles.len()
    }

    /// Number of events currently retained in the log.
    pub fn event_log_len(&self) -> usize {
        self.event_log.len()
    }

    /// Cumulative event count since creation.
    pub fn total_events(&self) -> u64 {
        self.total_events
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::peer_trust_scorer::{
        PeerTrustScorer, TrustBand, TrustConfig, TrustDimension, TrustEvent, TrustScorerStats,
    };
    use std::collections::HashMap;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn default_scorer() -> PeerTrustScorer {
        PeerTrustScorer::new(TrustConfig::default(), 200)
    }

    fn make_event(peer: &str, dim: TrustDimension, delta: f64, ts: u64) -> TrustEvent {
        TrustEvent {
            peer_id: peer.to_string(),
            dimension: dim,
            delta,
            timestamp: ts,
            description: "test event".to_string(),
        }
    }

    // ── TrustBand ─────────────────────────────────────────────────────────────

    #[test]
    fn band_from_score_blocked() {
        assert_eq!(TrustBand::from_score(0.0), TrustBand::Blocked);
        assert_eq!(TrustBand::from_score(0.09), TrustBand::Blocked);
    }

    #[test]
    fn band_from_score_untrusted() {
        assert_eq!(TrustBand::from_score(0.10), TrustBand::Untrusted);
        assert_eq!(TrustBand::from_score(0.29), TrustBand::Untrusted);
    }

    #[test]
    fn band_from_score_neutral() {
        assert_eq!(TrustBand::from_score(0.30), TrustBand::Neutral);
        assert_eq!(TrustBand::from_score(0.59), TrustBand::Neutral);
    }

    #[test]
    fn band_from_score_trusted() {
        assert_eq!(TrustBand::from_score(0.60), TrustBand::Trusted);
        assert_eq!(TrustBand::from_score(0.84), TrustBand::Trusted);
    }

    #[test]
    fn band_from_score_highly_trusted() {
        assert_eq!(TrustBand::from_score(0.85), TrustBand::HighlyTrusted);
        assert_eq!(TrustBand::from_score(1.0), TrustBand::HighlyTrusted);
    }

    #[test]
    fn band_ordinal_ordering() {
        assert!(TrustBand::Blocked < TrustBand::Untrusted);
        assert!(TrustBand::Untrusted < TrustBand::Neutral);
        assert!(TrustBand::Neutral < TrustBand::Trusted);
        assert!(TrustBand::Trusted < TrustBand::HighlyTrusted);
    }

    #[test]
    fn band_ordinal_values() {
        assert_eq!(TrustBand::Blocked.ordinal(), 0);
        assert_eq!(TrustBand::Untrusted.ordinal(), 1);
        assert_eq!(TrustBand::Neutral.ordinal(), 2);
        assert_eq!(TrustBand::Trusted.ordinal(), 3);
        assert_eq!(TrustBand::HighlyTrusted.ordinal(), 4);
    }

    // ── TrustConfig ───────────────────────────────────────────────────────────

    #[test]
    fn default_config_is_valid() {
        assert!(TrustConfig::default().is_valid());
    }

    #[test]
    fn default_config_weights_sum_to_one() {
        let total: f64 = TrustConfig::default().dimension_weights.values().sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn config_weight_for_returns_correct_values() {
        let cfg = TrustConfig::default();
        assert!((cfg.weight_for(&TrustDimension::ContentValidity) - 0.30).abs() < 1e-9);
        assert!((cfg.weight_for(&TrustDimension::Uptime) - 0.20).abs() < 1e-9);
    }

    #[test]
    fn config_weight_for_missing_dimension_returns_zero() {
        let cfg = TrustConfig {
            dimension_weights: HashMap::new(),
            ..TrustConfig::default()
        };
        assert_eq!(cfg.weight_for(&TrustDimension::Uptime), 0.0);
    }

    // ── Profile initialisation ────────────────────────────────────────────────

    #[test]
    fn new_profile_initialised_to_neutral() {
        let mut scorer = default_scorer();
        let profile = scorer.get_or_create_profile("peer-init", 0);
        // All dimensions should start at 0.5 → composite ≈ 0.5 → Neutral band.
        assert_eq!(profile.band, TrustBand::Neutral);
        assert!((profile.composite_score - 0.5).abs() < 1e-9);
    }

    #[test]
    fn new_profile_all_dimensions_present() {
        let mut scorer = default_scorer();
        let profile = scorer.get_or_create_profile("peer-dims", 0);
        for d in TrustDimension::all() {
            assert!(
                profile.scores.contains_key(d.as_key()),
                "missing dimension {:?}",
                d
            );
        }
    }

    #[test]
    fn get_profile_returns_none_for_unknown_peer() {
        let scorer = default_scorer();
        assert!(scorer.get_profile("ghost").is_none());
    }

    // ── Event recording ───────────────────────────────────────────────────────

    #[test]
    fn record_positive_event_increases_score() {
        let mut scorer = default_scorer();
        let before = scorer
            .get_or_create_profile("peer-pos", 0)
            .scores
            .get(TrustDimension::ContentValidity.as_key())
            .copied()
            .unwrap_or(0.5);
        scorer.record_event(make_event(
            "peer-pos",
            TrustDimension::ContentValidity,
            0.1,
            1,
        ));
        let after = scorer
            .get_profile("peer-pos")
            .expect("profile")
            .scores
            .get(TrustDimension::ContentValidity.as_key())
            .copied()
            .unwrap_or(0.0);
        assert!(after > before);
    }

    #[test]
    fn record_negative_event_decreases_score() {
        let mut scorer = default_scorer();
        scorer.record_event(make_event(
            "peer-neg",
            TrustDimension::ProtocolCompliance,
            -0.2,
            1,
        ));
        let score = scorer
            .get_profile("peer-neg")
            .expect("profile")
            .scores
            .get(TrustDimension::ProtocolCompliance.as_key())
            .copied()
            .unwrap_or(1.0);
        assert!(score < 0.5);
    }

    #[test]
    fn score_clamped_at_max() {
        let mut scorer = default_scorer();
        for _ in 0..20 {
            scorer.record_event(make_event("peer-max", TrustDimension::Uptime, 0.3, 1));
        }
        let score = scorer
            .get_profile("peer-max")
            .expect("profile")
            .scores
            .get(TrustDimension::Uptime.as_key())
            .copied()
            .unwrap_or(0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn score_clamped_at_min() {
        let mut scorer = default_scorer();
        for _ in 0..20 {
            scorer.record_event(make_event(
                "peer-min",
                TrustDimension::ResponseLatency,
                -0.3,
                1,
            ));
        }
        let score = scorer
            .get_profile("peer-min")
            .expect("profile")
            .scores
            .get(TrustDimension::ResponseLatency.as_key())
            .copied()
            .unwrap_or(1.0);
        assert!(score >= 0.0);
    }

    #[test]
    fn event_count_increments_per_event() {
        let mut scorer = default_scorer();
        for i in 0..5_u64 {
            scorer.record_event(make_event(
                "peer-cnt",
                TrustDimension::DataAvailability,
                0.01,
                i,
            ));
        }
        assert_eq!(
            scorer.get_profile("peer-cnt").expect("profile").event_count,
            5
        );
    }

    #[test]
    fn total_events_counts_across_peers() {
        let mut scorer = default_scorer();
        scorer.record_event(make_event("A", TrustDimension::Uptime, 0.1, 0));
        scorer.record_event(make_event("B", TrustDimension::Uptime, 0.1, 0));
        scorer.record_event(make_event("A", TrustDimension::Uptime, 0.1, 0));
        assert_eq!(scorer.total_events(), 3);
    }

    #[test]
    fn event_log_evicts_oldest_when_full() {
        let mut scorer = PeerTrustScorer::new(TrustConfig::default(), 3);
        for i in 0..5_u64 {
            scorer.record_event(make_event("peer-evict", TrustDimension::Uptime, 0.01, i));
        }
        // Log capped at 3.
        assert_eq!(scorer.event_log_len(), 3);
    }

    #[test]
    fn composite_score_is_weighted_average() {
        // Build a scorer where we can control scores manually.
        let mut scorer = default_scorer();
        // Force all dimensions to known values via events.
        let peer = "peer-weighted";
        // Start at 0.5; push ContentValidity (w=0.30) up by 0.1 → 0.6.
        scorer.record_event(make_event(peer, TrustDimension::ContentValidity, 0.1, 0));
        let profile = scorer.get_profile(peer).expect("profile");
        // Composite must be between 0.5 and 0.6 (only one dimension changed).
        assert!(profile.composite_score > 0.5);
        assert!(profile.composite_score <= 0.6);
    }

    #[test]
    fn band_updates_after_positive_events() {
        let mut scorer = default_scorer();
        let peer = "peer-band-up";
        // Push all dimensions to near-max.
        for d in TrustDimension::all() {
            for _ in 0..4 {
                scorer.record_event(make_event(peer, d.clone(), 0.1, 1));
            }
        }
        let profile = scorer.get_profile(peer).expect("profile");
        assert!(profile.band >= TrustBand::Trusted);
    }

    #[test]
    fn band_updates_after_negative_events() {
        let mut scorer = default_scorer();
        let peer = "peer-band-down";
        for d in TrustDimension::all() {
            for _ in 0..6 {
                scorer.record_event(make_event(peer, d.clone(), -0.1, 1));
            }
        }
        let profile = scorer.get_profile(peer).expect("profile");
        assert!(profile.band <= TrustBand::Untrusted);
    }

    // ── Decay ─────────────────────────────────────────────────────────────────

    #[test]
    fn apply_decay_reduces_scores() {
        let mut scorer = default_scorer();
        scorer.record_event(make_event("peer-decay", TrustDimension::Uptime, 0.2, 0));
        let before = scorer
            .get_profile("peer-decay")
            .expect("profile")
            .scores
            .get(TrustDimension::Uptime.as_key())
            .copied()
            .unwrap_or(0.5);
        // Advance by 1 hour = 3_600_000 ms.
        scorer.apply_decay(3_600_000);
        let after = scorer
            .get_profile("peer-decay")
            .expect("profile")
            .scores
            .get(TrustDimension::Uptime.as_key())
            .copied()
            .unwrap_or(1.0);
        assert!(after < before);
    }

    #[test]
    fn apply_decay_updates_last_updated() {
        let mut scorer = default_scorer();
        scorer.record_event(make_event("peer-decay2", TrustDimension::Uptime, 0.0, 100));
        scorer.apply_decay(9_000_000);
        assert_eq!(
            scorer
                .get_profile("peer-decay2")
                .expect("profile")
                .last_updated,
            9_000_000
        );
    }

    #[test]
    fn apply_decay_with_zero_elapsed_is_identity() {
        let mut scorer = default_scorer();
        let ts = 5_000;
        scorer.record_event(make_event("peer-nodecay", TrustDimension::Uptime, 0.1, ts));
        let before = scorer
            .get_profile("peer-nodecay")
            .expect("profile")
            .composite_score;
        scorer.apply_decay(ts); // 0 ms elapsed
        let after = scorer
            .get_profile("peer-nodecay")
            .expect("profile")
            .composite_score;
        assert!((before - after).abs() < 1e-12);
    }

    #[test]
    fn apply_decay_does_not_go_below_min() {
        let cfg = TrustConfig {
            decay_rate: 100.0, // extreme decay
            ..TrustConfig::default()
        };
        let mut scorer = PeerTrustScorer::new(cfg, 100);
        scorer.record_event(make_event("peer-floor", TrustDimension::Uptime, 0.0, 0));
        scorer.apply_decay(3_600_000);
        let score = scorer
            .get_profile("peer-floor")
            .expect("profile")
            .scores
            .get(TrustDimension::Uptime.as_key())
            .copied()
            .unwrap_or(1.0);
        assert!(score >= 0.0);
    }

    // ── trusted_peers ─────────────────────────────────────────────────────────

    #[test]
    fn trusted_peers_filters_by_min_band() {
        let mut scorer = default_scorer();
        // Create a blocked peer.
        scorer.ban_peer("peer-blocked", 0);
        // Create a neutral peer (default 0.5).
        scorer.get_or_create_profile("peer-neutral", 0);
        // Trusted peers must exclude Blocked.
        let trusted = scorer.trusted_peers(TrustBand::Neutral);
        let ids: Vec<&str> = trusted.iter().map(|p| p.peer_id.as_str()).collect();
        assert!(!ids.contains(&"peer-blocked"));
        assert!(ids.contains(&"peer-neutral"));
    }

    #[test]
    fn trusted_peers_sorted_descending_by_composite() {
        let mut scorer = default_scorer();
        scorer.record_event(make_event("peer-high", TrustDimension::Uptime, 0.3, 0));
        scorer.record_event(make_event("peer-low", TrustDimension::Uptime, -0.1, 0));
        let peers = scorer.trusted_peers(TrustBand::Blocked); // include all
        if peers.len() >= 2 {
            let first = peers[0].composite_score;
            let last = peers[peers.len() - 1].composite_score;
            assert!(first >= last);
        }
    }

    // ── blocked_peers ─────────────────────────────────────────────────────────

    #[test]
    fn blocked_peers_returns_only_blocked() {
        let mut scorer = default_scorer();
        scorer.ban_peer("peer-b1", 0);
        scorer.ban_peer("peer-b2", 0);
        scorer.get_or_create_profile("peer-ok", 0);
        let blocked = scorer.blocked_peers();
        assert!(blocked.contains(&"peer-b1"));
        assert!(blocked.contains(&"peer-b2"));
        assert!(!blocked.contains(&"peer-ok"));
    }

    #[test]
    fn blocked_peers_empty_when_none_banned() {
        let scorer = default_scorer();
        assert!(scorer.blocked_peers().is_empty());
    }

    // ── peer_events pagination ─────────────────────────────────────────────────

    #[test]
    fn peer_events_page_zero_returns_first_n() {
        let mut scorer = PeerTrustScorer::new(TrustConfig::default(), 1000);
        for i in 0..10_u64 {
            scorer.record_event(make_event("peer-pg", TrustDimension::Uptime, 0.01, i));
        }
        let page = scorer.peer_events("peer-pg", 0);
        // Default events_per_page = 50; all 10 fit on page 0.
        assert_eq!(page.len(), 10);
    }

    #[test]
    fn peer_events_page_one_skips_first_page() {
        let cfg = TrustConfig {
            events_per_page: 3,
            ..TrustConfig::default()
        };
        let mut scorer = PeerTrustScorer::new(cfg, 1000);
        for i in 0..7_u64 {
            scorer.record_event(make_event("peer-pages", TrustDimension::Uptime, 0.0, i));
        }
        let page1 = scorer.peer_events("peer-pages", 1);
        // Page 1 should have events 3-5 (indices 3..6).
        assert_eq!(page1.len(), 3);
    }

    #[test]
    fn peer_events_filters_by_peer_id() {
        let mut scorer = default_scorer();
        scorer.record_event(make_event("A", TrustDimension::Uptime, 0.01, 0));
        scorer.record_event(make_event("B", TrustDimension::Uptime, 0.01, 1));
        scorer.record_event(make_event("A", TrustDimension::Uptime, 0.01, 2));
        let events = scorer.peer_events("A", 0);
        assert!(events.iter().all(|e| e.peer_id == "A"));
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn peer_events_empty_page_beyond_end() {
        let mut scorer = default_scorer();
        scorer.record_event(make_event("peer-eof", TrustDimension::Uptime, 0.0, 0));
        let page99 = scorer.peer_events("peer-eof", 99);
        assert!(page99.is_empty());
    }

    // ── ban / rehabilitate ────────────────────────────────────────────────────

    #[test]
    fn ban_peer_sets_band_to_blocked() {
        let mut scorer = default_scorer();
        scorer.ban_peer("peer-ban", 0);
        assert_eq!(
            scorer.get_profile("peer-ban").expect("profile").band,
            TrustBand::Blocked
        );
    }

    #[test]
    fn ban_peer_sets_all_scores_to_zero() {
        let mut scorer = default_scorer();
        scorer.ban_peer("peer-zeroed", 0);
        let profile = scorer.get_profile("peer-zeroed").expect("profile");
        for score in profile.scores.values() {
            assert!(*score <= 0.0);
        }
    }

    #[test]
    fn ban_creates_profile_if_absent() {
        let mut scorer = default_scorer();
        scorer.ban_peer("brand-new", 42);
        assert!(scorer.get_profile("brand-new").is_some());
    }

    #[test]
    fn rehabilitate_peer_resets_scores_to_0_3() {
        let mut scorer = default_scorer();
        scorer.ban_peer("peer-rehab", 0);
        scorer.rehabilitate_peer("peer-rehab", 100);
        let profile = scorer.get_profile("peer-rehab").expect("profile");
        for score in profile.scores.values() {
            assert!((*score - 0.3).abs() < 1e-12);
        }
    }

    #[test]
    fn rehabilitate_updates_band_above_blocked() {
        let mut scorer = default_scorer();
        scorer.ban_peer("peer-reh2", 0);
        scorer.rehabilitate_peer("peer-reh2", 100);
        let band = &scorer.get_profile("peer-reh2").expect("profile").band;
        assert!(*band > TrustBand::Blocked);
    }

    #[test]
    fn rehabilitate_creates_profile_if_absent() {
        let mut scorer = default_scorer();
        scorer.rehabilitate_peer("ghost-rehab", 0);
        assert!(scorer.get_profile("ghost-rehab").is_some());
    }

    // ── scorer_stats ──────────────────────────────────────────────────────────

    #[test]
    fn scorer_stats_no_peers() {
        let scorer = default_scorer();
        let stats = scorer.scorer_stats();
        assert_eq!(stats.total_peers, 0);
        assert_eq!(stats.total_events, 0);
        assert!((stats.avg_composite_score - 0.5).abs() < 1e-9);
    }

    #[test]
    fn scorer_stats_counts_blocked_correctly() {
        let mut scorer = default_scorer();
        scorer.ban_peer("b1", 0);
        scorer.ban_peer("b2", 0);
        let stats = scorer.scorer_stats();
        assert_eq!(stats.blocked_count, 2);
    }

    #[test]
    fn scorer_stats_counts_trusted_correctly() {
        let mut scorer = default_scorer();
        // Push two peers into Trusted band.
        for peer in &["t1", "t2"] {
            for d in TrustDimension::all() {
                for _ in 0..3 {
                    scorer.record_event(make_event(peer, d.clone(), 0.1, 0));
                }
            }
        }
        let stats = scorer.scorer_stats();
        assert!(stats.trusted_count >= 2);
    }

    #[test]
    fn scorer_stats_total_events_matches_recorded() {
        let mut scorer = default_scorer();
        let n = 7_u64;
        for i in 0..n {
            scorer.record_event(make_event("ev-peer", TrustDimension::Uptime, 0.0, i));
        }
        assert_eq!(scorer.scorer_stats().total_events, n);
    }

    #[test]
    fn scorer_stats_avg_composite_is_mean() {
        let mut scorer = default_scorer();
        // Two peers: one with composite ≈ 0.5 (default), one with extra boost.
        scorer.get_or_create_profile("p-avg1", 0);
        scorer.record_event(make_event("p-avg2", TrustDimension::Uptime, 0.2, 0));
        let stats = scorer.scorer_stats();
        assert!(stats.avg_composite_score > 0.5); // boosted peer pulls average up
    }

    // ── dimension key helper ──────────────────────────────────────────────────

    #[test]
    fn dimension_key_returns_stable_strings() {
        assert_eq!(
            PeerTrustScorer::dimension_key(&TrustDimension::Uptime),
            "Uptime"
        );
        assert_eq!(
            PeerTrustScorer::dimension_key(&TrustDimension::ContentValidity),
            "ContentValidity"
        );
        assert_eq!(
            PeerTrustScorer::dimension_key(&TrustDimension::ProtocolCompliance),
            "ProtocolCompliance"
        );
        assert_eq!(
            PeerTrustScorer::dimension_key(&TrustDimension::ResponseLatency),
            "ResponseLatency"
        );
        assert_eq!(
            PeerTrustScorer::dimension_key(&TrustDimension::DataAvailability),
            "DataAvailability"
        );
    }

    // ── Multi-peer isolation ──────────────────────────────────────────────────

    #[test]
    fn events_for_different_peers_do_not_interfere() {
        let mut scorer = default_scorer();
        scorer.record_event(make_event("peer-iso-A", TrustDimension::Uptime, 0.3, 0));
        scorer.record_event(make_event("peer-iso-B", TrustDimension::Uptime, -0.3, 0));
        let a = scorer
            .get_profile("peer-iso-A")
            .expect("profile A")
            .composite_score;
        let b = scorer
            .get_profile("peer-iso-B")
            .expect("profile B")
            .composite_score;
        assert!(a > b);
    }

    #[test]
    fn peer_count_tracks_distinct_peers() {
        let mut scorer = default_scorer();
        // Three distinct peers; recording multiple events for the same peer
        // must not inflate the count.
        for _ in 0..5 {
            scorer.record_event(make_event("P1", TrustDimension::Uptime, 0.0, 0));
        }
        scorer.record_event(make_event("P2", TrustDimension::Uptime, 0.0, 0));
        scorer.record_event(make_event("P3", TrustDimension::Uptime, 0.0, 0));
        assert_eq!(scorer.peer_count(), 3);
    }

    // ── Edge / boundary cases ─────────────────────────────────────────────────

    #[test]
    fn max_events_of_one_retains_only_latest() {
        let mut scorer = PeerTrustScorer::new(TrustConfig::default(), 1);
        scorer.record_event(make_event("edge", TrustDimension::Uptime, 0.01, 0));
        scorer.record_event(make_event("edge", TrustDimension::Uptime, 0.01, 1));
        assert_eq!(scorer.event_log_len(), 1);
    }

    #[test]
    fn rehabilitate_then_ban_leaves_peer_blocked() {
        let mut scorer = default_scorer();
        scorer.rehabilitate_peer("cycle", 0);
        scorer.ban_peer("cycle", 100);
        assert_eq!(
            scorer.get_profile("cycle").expect("profile").band,
            TrustBand::Blocked
        );
    }

    #[test]
    fn trusted_peers_with_highly_trusted_min_band() {
        let mut scorer = default_scorer();
        // Only one peer achieves HighlyTrusted.
        for d in TrustDimension::all() {
            for _ in 0..5 {
                scorer.record_event(make_event("elite", d.clone(), 0.1, 0));
            }
        }
        scorer.get_or_create_profile("average", 0); // stays Neutral (0.5)
        let ht = scorer.trusted_peers(TrustBand::HighlyTrusted);
        assert!(ht.iter().all(|p| p.band == TrustBand::HighlyTrusted));
    }

    #[test]
    fn profile_implements_debug() {
        let mut scorer = default_scorer();
        let profile = scorer.get_or_create_profile("debug-peer", 0);
        let s = format!("{:?}", profile);
        assert!(s.contains("PeerTrustProfile"));
    }

    #[test]
    fn trust_scorer_stats_implements_debug() {
        let stats = TrustScorerStats {
            total_peers: 0,
            trusted_count: 0,
            blocked_count: 0,
            total_events: 0,
            avg_composite_score: 0.5,
        };
        let s = format!("{:?}", stats);
        assert!(s.contains("TrustScorerStats"));
    }

    #[test]
    fn trust_event_implements_clone() {
        let ev = make_event("clone-peer", TrustDimension::Uptime, 0.1, 42);
        let ev2 = ev.clone();
        assert_eq!(ev2.peer_id, ev.peer_id);
        assert_eq!(ev2.delta, ev.delta);
    }

    #[test]
    fn trust_config_implements_clone() {
        let cfg = TrustConfig::default();
        let cfg2 = cfg.clone();
        assert!((cfg2.decay_rate - cfg.decay_rate).abs() < 1e-12);
    }

    #[test]
    fn trust_dimension_all_has_five_variants() {
        assert_eq!(TrustDimension::all().len(), 5);
    }
}
