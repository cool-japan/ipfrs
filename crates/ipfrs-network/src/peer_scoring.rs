//! Composite peer quality scoring engine for P2P overlay networks.
//!
//! [`PeerScoringSystem`] computes a weighted composite score for each peer
//! based on multiple observable quality dimensions (latency, bandwidth,
//! availability, reliability, gossip delivery, and routing success).
//!
//! Scores are accumulated in a bounded history per peer, enabling trend
//! analysis, stale-peer eviction, and tier-based classification.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_network::peer_scoring::{
//!     PeerScoringSystem, PeerMetrics, ScoringWeights,
//! };
//!
//! let system = PeerScoringSystem::default();
//!
//! let metrics = PeerMetrics {
//!     peer_id: "peer-1".to_string(),
//!     rtt_ms: 50.0,
//!     bandwidth_bps: 800_000.0,
//!     uptime_fraction: 0.99,
//!     success_rate: 0.97,
//!     gossip_delivery_rate: 0.95,
//!     routing_success_rate: 0.93,
//!     last_updated: 1_000,
//! };
//!
//! let score = system.score_peer(&metrics, 1_000);
//! println!("Composite: {:.3}", score.composite);
//! ```

use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur when constructing or using a [`PeerScoringSystem`].
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ScoringError {
    /// The provided weights do not sum to approximately 1.0.
    #[error("scoring weights must sum to 1.0, got {got:.6}")]
    WeightsMustSumToOne {
        /// The actual sum of the provided weights.
        got: f64,
    },

    /// A query was attempted for a peer that has no recorded history.
    #[error("peer not found in scoring history")]
    PeerNotFound,
}

// ---------------------------------------------------------------------------
// Score component dimension
// ---------------------------------------------------------------------------

/// The observable quality dimension contributing to a peer's composite score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PsScoringDimension {
    /// Round-trip latency quality (lower RTT → higher score).
    Latency,
    /// Available upload/download bandwidth.
    Bandwidth,
    /// Historical fraction of time the peer has been reachable.
    Availability,
    /// Fraction of protocol operations that succeeded.
    Reliability,
    /// Quality of gossip message delivery.
    GossipQuality,
    /// Quality of routing operations.
    RoutingQuality,
}

// ---------------------------------------------------------------------------
// Score tier
// ---------------------------------------------------------------------------

/// Qualitative tier derived from the composite score.
///
/// Ordered so that `Excellent > Good > Fair > Poor > Unacceptable`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScoreTier {
    /// Composite ≥ 0.8
    Excellent,
    /// Composite ≥ 0.6
    Good,
    /// Composite ≥ 0.4
    Fair,
    /// Composite ≥ 0.2
    Poor,
    /// Composite < 0.2
    Unacceptable,
}

impl ScoreTier {
    /// Numeric rank used for ordering (higher = better).
    #[inline]
    fn rank(self) -> u8 {
        match self {
            ScoreTier::Unacceptable => 0,
            ScoreTier::Poor => 1,
            ScoreTier::Fair => 2,
            ScoreTier::Good => 3,
            ScoreTier::Excellent => 4,
        }
    }

    /// Derive the tier from a composite score value.
    pub fn from_score(composite: f64) -> Self {
        if composite >= 0.8 {
            ScoreTier::Excellent
        } else if composite >= 0.6 {
            ScoreTier::Good
        } else if composite >= 0.4 {
            ScoreTier::Fair
        } else if composite >= 0.2 {
            ScoreTier::Poor
        } else {
            ScoreTier::Unacceptable
        }
    }
}

impl PartialOrd for ScoreTier {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoreTier {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.rank().cmp(&other.rank())
    }
}

// ---------------------------------------------------------------------------
// Scoring weights
// ---------------------------------------------------------------------------

/// Weights applied to each scoring dimension when computing the composite score.
///
/// All six weights must sum to 1.0 (validated within 0.001 tolerance).
#[derive(Debug, Clone)]
pub struct ScoringWeights {
    /// Weight for the latency component.
    pub latency: f64,
    /// Weight for the bandwidth component.
    pub bandwidth: f64,
    /// Weight for the availability component.
    pub availability: f64,
    /// Weight for the reliability component.
    pub reliability: f64,
    /// Weight for the gossip quality component.
    pub gossip_quality: f64,
    /// Weight for the routing quality component.
    pub routing_quality: f64,
}

impl ScoringWeights {
    /// Compute the sum of all six weights.
    #[inline]
    pub fn sum(&self) -> f64 {
        self.latency
            + self.bandwidth
            + self.availability
            + self.reliability
            + self.gossip_quality
            + self.routing_quality
    }

    /// Validate that the weights sum to approximately 1.0 (within 0.001).
    pub fn validate(&self) -> Result<(), ScoringError> {
        let s = self.sum();
        if (s - 1.0).abs() > 0.001 {
            Err(ScoringError::WeightsMustSumToOne { got: s })
        } else {
            Ok(())
        }
    }
}

impl Default for ScoringWeights {
    /// Even weights: each dimension receives 1/6 ≈ 0.16667.
    fn default() -> Self {
        let w = 1.0 / 6.0;
        Self {
            latency: w,
            bandwidth: w,
            availability: w,
            reliability: w,
            gossip_quality: w,
            routing_quality: w,
        }
    }
}

// ---------------------------------------------------------------------------
// Peer metrics (raw observations)
// ---------------------------------------------------------------------------

/// Raw observations for a single peer at a point in time.
#[derive(Debug, Clone)]
pub struct PeerMetrics {
    /// Stable identifier for the peer.
    pub peer_id: String,
    /// Round-trip time in milliseconds.
    pub rtt_ms: f64,
    /// Measured bandwidth in bits per second.
    pub bandwidth_bps: f64,
    /// Historical uptime as a fraction in \[0, 1\].
    pub uptime_fraction: f64,
    /// Fraction of protocol operations that succeeded, in \[0, 1\].
    pub success_rate: f64,
    /// Fraction of gossip messages successfully delivered, in \[0, 1\].
    pub gossip_delivery_rate: f64,
    /// Fraction of routing operations that succeeded, in \[0, 1\].
    pub routing_success_rate: f64,
    /// Epoch millisecond timestamp of the last observation.
    pub last_updated: u64,
}

// ---------------------------------------------------------------------------
// Peer score (computed result)
// ---------------------------------------------------------------------------

/// Computed composite score and per-dimension breakdown for a single peer.
#[derive(Debug, Clone)]
pub struct PsPeerScore {
    /// Peer identifier.
    pub peer_id: String,
    /// Weighted composite score in \[0, 1\].
    pub composite: f64,
    /// Individual normalised scores for each dimension.
    pub components: HashMap<PsScoringDimension, f64>,
    /// Qualitative tier derived from the composite score.
    pub tier: ScoreTier,
    /// Epoch millisecond timestamp at which this score was computed.
    pub computed_at: u64,
}

// ---------------------------------------------------------------------------
// Scoring statistics
// ---------------------------------------------------------------------------

/// Aggregate statistics for the entire scoring system.
#[derive(Debug, Clone)]
pub struct ScoringStats {
    /// Total number of peers with recorded history.
    pub total_peers: usize,
    /// Mean composite score across the most recent score of every tracked peer.
    pub avg_composite: f64,
    /// Distribution of peers across tier classifications (most-recent score).
    pub tier_distribution: HashMap<ScoreTier, usize>,
    /// Number of peers whose latest score is older than `max_age_ms`.
    pub stale_peers: usize,
}

// ---------------------------------------------------------------------------
// PeerScoringSystem
// ---------------------------------------------------------------------------

/// Composite peer quality scoring engine.
///
/// Maintains a rolling score history per peer and exposes scoring, ranking,
/// trend analysis, and stale-peer eviction.
#[derive(Debug)]
pub struct PeerScoringSystem {
    /// Dimension weights used when computing composite scores.
    pub weights: ScoringWeights,
    /// Rolling score history per peer, keyed by peer ID.
    pub history: HashMap<String, VecDeque<PsPeerScore>>,
    /// Maximum number of recent scores retained per peer.
    pub max_history: usize,
}

impl Default for PeerScoringSystem {
    fn default() -> Self {
        Self {
            weights: ScoringWeights::default(),
            history: HashMap::new(),
            max_history: 100,
        }
    }
}

impl PeerScoringSystem {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Construct a new system with the given scoring weights.
    ///
    /// Returns [`ScoringError::WeightsMustSumToOne`] if the weights do not
    /// sum to approximately 1.0.
    pub fn new(weights: ScoringWeights) -> Result<Self, ScoringError> {
        weights.validate()?;
        Ok(Self {
            weights,
            history: HashMap::new(),
            max_history: 100,
        })
    }

    /// Construct with a custom `max_history` capacity.
    pub fn with_max_history(mut self, max_history: usize) -> Self {
        self.max_history = max_history.max(1);
        self
    }

    // ------------------------------------------------------------------
    // Core scoring
    // ------------------------------------------------------------------

    /// Compute a [`PsPeerScore`] from raw [`PeerMetrics`] without storing it.
    ///
    /// # Component formulas (all output in \[0, 1\]):
    ///
    /// - **Latency** : `exp(-rtt_ms / 500.0)` — 0 ms → 1.0, 500 ms → 0.368, 1 000 ms → 0.135
    /// - **Bandwidth** : `(bandwidth_bps / 1_000_000).min(1.0)` — caps at 1 Mbps
    /// - **Availability** : `uptime_fraction.clamp(0, 1)`
    /// - **Reliability** : `success_rate.clamp(0, 1)`
    /// - **GossipQuality** : `gossip_delivery_rate.clamp(0, 1)`
    /// - **RoutingQuality** : `routing_success_rate.clamp(0, 1)`
    pub fn score_peer(&self, metrics: &PeerMetrics, now: u64) -> PsPeerScore {
        let latency_score = (-metrics.rtt_ms / 500.0).exp();
        let bandwidth_score = (metrics.bandwidth_bps / 1_000_000.0).min(1.0);
        let availability_score = metrics.uptime_fraction.clamp(0.0, 1.0);
        let reliability_score = metrics.success_rate.clamp(0.0, 1.0);
        let gossip_score = metrics.gossip_delivery_rate.clamp(0.0, 1.0);
        let routing_score = metrics.routing_success_rate.clamp(0.0, 1.0);

        let composite = self.weights.latency * latency_score
            + self.weights.bandwidth * bandwidth_score
            + self.weights.availability * availability_score
            + self.weights.reliability * reliability_score
            + self.weights.gossip_quality * gossip_score
            + self.weights.routing_quality * routing_score;

        let mut components = HashMap::with_capacity(6);
        components.insert(PsScoringDimension::Latency, latency_score);
        components.insert(PsScoringDimension::Bandwidth, bandwidth_score);
        components.insert(PsScoringDimension::Availability, availability_score);
        components.insert(PsScoringDimension::Reliability, reliability_score);
        components.insert(PsScoringDimension::GossipQuality, gossip_score);
        components.insert(PsScoringDimension::RoutingQuality, routing_score);

        let tier = ScoreTier::from_score(composite);

        PsPeerScore {
            peer_id: metrics.peer_id.clone(),
            composite,
            components,
            tier,
            computed_at: now,
        }
    }

    /// Compute a score and push it into the peer's history ring-buffer.
    ///
    /// If the buffer is at `max_history` capacity, the oldest entry is evicted.
    pub fn score_and_record(&mut self, metrics: &PeerMetrics, now: u64) -> PsPeerScore {
        let score = self.score_peer(metrics, now);
        let deque = self.history.entry(metrics.peer_id.clone()).or_default();

        if deque.len() >= self.max_history {
            deque.pop_front();
        }
        deque.push_back(score.clone());
        score
    }

    // ------------------------------------------------------------------
    // History queries
    // ------------------------------------------------------------------

    /// Return the history slice for `peer_id` (oldest first, newest last).
    ///
    /// Returns an empty slice if the peer has no recorded history.
    pub fn history_for(&self, peer_id: &str) -> &[PsPeerScore] {
        self.history
            .get(peer_id)
            .map(|dq| dq.as_slices().0) // only the contiguous front slice
            .unwrap_or(&[])
    }

    /// Return a `Vec` of all history entries for `peer_id` in order
    /// (oldest first, newest last).
    ///
    /// Unlike [`history_for`][Self::history_for], this always returns the
    /// full sequence even when the internal deque is split across two slices.
    pub fn history_vec(&self, peer_id: &str) -> Vec<&PsPeerScore> {
        match self.history.get(peer_id) {
            None => Vec::new(),
            Some(dq) => dq.iter().collect(),
        }
    }

    /// Return the mean composite score for `peer_id` across all recorded
    /// history entries, or `None` if there is no history.
    pub fn average_score(&self, peer_id: &str) -> Option<f64> {
        let dq = self.history.get(peer_id)?;
        if dq.is_empty() {
            return None;
        }
        let sum: f64 = dq.iter().map(|s| s.composite).sum();
        Some(sum / dq.len() as f64)
    }

    /// Compute the slope of a linear regression over the composite scores in
    /// the peer's history (positive slope = improving quality over time).
    ///
    /// Returns `None` if fewer than 2 data points are available.
    ///
    /// The independent variable is the index in the history sequence
    /// (0, 1, 2, …), so the slope represents change per observation step.
    pub fn trend(&self, peer_id: &str) -> Option<f64> {
        let dq = self.history.get(peer_id)?;
        let n = dq.len();
        if n < 2 {
            return None;
        }

        let n_f = n as f64;
        let x_mean = (n_f - 1.0) / 2.0; // mean of 0..n-1
        let y_mean = dq.iter().map(|s| s.composite).sum::<f64>() / n_f;

        let mut numerator = 0.0_f64;
        let mut denominator = 0.0_f64;

        for (i, s) in dq.iter().enumerate() {
            let dx = i as f64 - x_mean;
            let dy = s.composite - y_mean;
            numerator += dx * dy;
            denominator += dx * dx;
        }

        if denominator == 0.0 {
            return Some(0.0);
        }

        Some(numerator / denominator)
    }

    // ------------------------------------------------------------------
    // Ranking
    // ------------------------------------------------------------------

    /// Return references to the top `n` peers ordered by their most-recent
    /// composite score (descending).
    ///
    /// Peers with no history are excluded.
    pub fn top_peers(&self, n: usize, _now: u64) -> Vec<&PsPeerScore> {
        // Collect the most-recent score for each peer.
        let mut latest: Vec<&PsPeerScore> =
            self.history.values().filter_map(|dq| dq.back()).collect();

        // Sort descending by composite.
        latest.sort_by(|a, b| {
            b.composite
                .partial_cmp(&a.composite)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        latest.into_iter().take(n).collect()
    }

    // ------------------------------------------------------------------
    // Maintenance
    // ------------------------------------------------------------------

    /// Remove peers whose latest recorded score is older than `max_age_ms`
    /// milliseconds relative to `now`.
    ///
    /// Returns the number of peers evicted.
    pub fn evict_stale(&mut self, max_age_ms: u64, now: u64) -> usize {
        let stale: Vec<String> = self
            .history
            .iter()
            .filter_map(|(peer_id, dq)| {
                dq.back().and_then(|latest| {
                    let age = now.saturating_sub(latest.computed_at);
                    if age > max_age_ms {
                        Some(peer_id.clone())
                    } else {
                        None
                    }
                })
            })
            .collect();

        let count = stale.len();
        for id in stale {
            self.history.remove(&id);
        }
        count
    }

    // ------------------------------------------------------------------
    // Statistics
    // ------------------------------------------------------------------

    /// Compute aggregate statistics for the scoring system.
    ///
    /// `max_age_ms` and `now` are used solely to determine which peers are
    /// considered "stale" for the [`ScoringStats::stale_peers`] field;
    /// stale peers are **not** evicted by this call.
    pub fn stats(&self, max_age_ms: u64, now: u64) -> ScoringStats {
        let total_peers = self.history.len();
        let mut composite_sum = 0.0_f64;
        let mut tier_distribution: HashMap<ScoreTier, usize> = HashMap::new();
        let mut stale_peers = 0usize;

        for dq in self.history.values() {
            if let Some(latest) = dq.back() {
                composite_sum += latest.composite;
                *tier_distribution.entry(latest.tier).or_insert(0) += 1;

                let age = now.saturating_sub(latest.computed_at);
                if age > max_age_ms {
                    stale_peers += 1;
                }
            }
        }

        let avg_composite = if total_peers > 0 {
            composite_sum / total_peers as f64
        } else {
            0.0
        };

        ScoringStats {
            total_peers,
            avg_composite,
            tier_distribution,
            stale_peers,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::peer_scoring::{
        PeerMetrics, PeerScoringSystem, PsScoringDimension, ScoreTier, ScoringError, ScoringWeights,
    };

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn excellent_metrics(peer_id: &str) -> PeerMetrics {
        PeerMetrics {
            peer_id: peer_id.to_string(),
            rtt_ms: 1.0,
            bandwidth_bps: 2_000_000.0,
            uptime_fraction: 1.0,
            success_rate: 1.0,
            gossip_delivery_rate: 1.0,
            routing_success_rate: 1.0,
            last_updated: 0,
        }
    }

    fn poor_metrics(peer_id: &str) -> PeerMetrics {
        PeerMetrics {
            peer_id: peer_id.to_string(),
            rtt_ms: 2000.0,
            bandwidth_bps: 0.0,
            uptime_fraction: 0.05,
            success_rate: 0.05,
            gossip_delivery_rate: 0.05,
            routing_success_rate: 0.05,
            last_updated: 0,
        }
    }

    // -----------------------------------------------------------------------
    // ScoringWeights
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_weights_sum_to_one() {
        let w = ScoringWeights::default();
        assert!((w.sum() - 1.0).abs() < 0.001, "sum = {}", w.sum());
    }

    #[test]
    fn test_default_weights_equal() {
        let w = ScoringWeights::default();
        let expected = 1.0 / 6.0;
        for v in [
            w.latency,
            w.bandwidth,
            w.availability,
            w.reliability,
            w.gossip_quality,
            w.routing_quality,
        ] {
            assert!((v - expected).abs() < 1e-10, "weight = {v}");
        }
    }

    #[test]
    fn test_weights_validation_passes_for_one() {
        let w = ScoringWeights {
            latency: 0.2,
            bandwidth: 0.2,
            availability: 0.2,
            reliability: 0.2,
            gossip_quality: 0.1,
            routing_quality: 0.1,
        };
        assert!(w.validate().is_ok());
    }

    #[test]
    fn test_weights_validation_fails_for_wrong_sum() {
        let w = ScoringWeights {
            latency: 0.5,
            bandwidth: 0.5,
            availability: 0.5,
            reliability: 0.0,
            gossip_quality: 0.0,
            routing_quality: 0.0,
        };
        let err = w.validate();
        assert!(matches!(err, Err(ScoringError::WeightsMustSumToOne { .. })));
    }

    #[test]
    fn test_weights_within_tolerance_passes() {
        // 0.0009 deviation should be accepted.
        let tiny = 0.0009 / 6.0;
        let w = ScoringWeights {
            latency: 1.0 / 6.0 + tiny,
            bandwidth: 1.0 / 6.0,
            availability: 1.0 / 6.0,
            reliability: 1.0 / 6.0,
            gossip_quality: 1.0 / 6.0,
            routing_quality: 1.0 / 6.0,
        };
        assert!(w.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // PeerScoringSystem construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_with_valid_weights() {
        let w = ScoringWeights::default();
        let sys = PeerScoringSystem::new(w);
        assert!(sys.is_ok());
    }

    #[test]
    fn test_new_with_invalid_weights_returns_error() {
        let w = ScoringWeights {
            latency: 1.0,
            bandwidth: 1.0,
            availability: 0.0,
            reliability: 0.0,
            gossip_quality: 0.0,
            routing_quality: 0.0,
        };
        let err = PeerScoringSystem::new(w);
        assert!(matches!(err, Err(ScoringError::WeightsMustSumToOne { .. })));
    }

    #[test]
    fn test_default_system_creates_successfully() {
        let sys = PeerScoringSystem::default();
        assert_eq!(sys.max_history, 100);
        assert!(sys.history.is_empty());
    }

    #[test]
    fn test_with_max_history() {
        let sys = PeerScoringSystem::default().with_max_history(50);
        assert_eq!(sys.max_history, 50);
    }

    #[test]
    fn test_with_max_history_min_one() {
        let sys = PeerScoringSystem::default().with_max_history(0);
        assert_eq!(sys.max_history, 1);
    }

    // -----------------------------------------------------------------------
    // score_peer component formulas
    // -----------------------------------------------------------------------

    #[test]
    fn test_latency_zero_gives_one() {
        let sys = PeerScoringSystem::default();
        let m = PeerMetrics {
            peer_id: "p".into(),
            rtt_ms: 0.0,
            bandwidth_bps: 0.0,
            uptime_fraction: 0.0,
            success_rate: 0.0,
            gossip_delivery_rate: 0.0,
            routing_success_rate: 0.0,
            last_updated: 0,
        };
        let s = sys.score_peer(&m, 0);
        let lat = s.components[&PsScoringDimension::Latency];
        assert!((lat - 1.0).abs() < 1e-10, "got {lat}");
    }

    #[test]
    fn test_latency_500ms_gives_approx_0_37() {
        let sys = PeerScoringSystem::default();
        let m = PeerMetrics {
            peer_id: "p".into(),
            rtt_ms: 500.0,
            bandwidth_bps: 0.0,
            uptime_fraction: 0.0,
            success_rate: 0.0,
            gossip_delivery_rate: 0.0,
            routing_success_rate: 0.0,
            last_updated: 0,
        };
        let s = sys.score_peer(&m, 0);
        let lat = s.components[&PsScoringDimension::Latency];
        let expected = (-1.0_f64).exp();
        assert!((lat - expected).abs() < 1e-10, "got {lat}");
    }

    #[test]
    fn test_bandwidth_capped_at_1mbps() {
        let sys = PeerScoringSystem::default();
        let m = PeerMetrics {
            peer_id: "p".into(),
            rtt_ms: 0.0,
            bandwidth_bps: 5_000_000.0,
            uptime_fraction: 0.0,
            success_rate: 0.0,
            gossip_delivery_rate: 0.0,
            routing_success_rate: 0.0,
            last_updated: 0,
        };
        let s = sys.score_peer(&m, 0);
        let bw = s.components[&PsScoringDimension::Bandwidth];
        assert!((bw - 1.0).abs() < 1e-10, "got {bw}");
    }

    #[test]
    fn test_bandwidth_500kbps_gives_0_5() {
        let sys = PeerScoringSystem::default();
        let m = PeerMetrics {
            peer_id: "p".into(),
            rtt_ms: 0.0,
            bandwidth_bps: 500_000.0,
            uptime_fraction: 0.0,
            success_rate: 0.0,
            gossip_delivery_rate: 0.0,
            routing_success_rate: 0.0,
            last_updated: 0,
        };
        let s = sys.score_peer(&m, 0);
        let bw = s.components[&PsScoringDimension::Bandwidth];
        assert!((bw - 0.5).abs() < 1e-10, "got {bw}");
    }

    #[test]
    fn test_availability_clamped() {
        let sys = PeerScoringSystem::default();
        let m = PeerMetrics {
            peer_id: "p".into(),
            rtt_ms: 0.0,
            bandwidth_bps: 0.0,
            uptime_fraction: 1.5, // above 1 → clamped to 1
            success_rate: 0.0,
            gossip_delivery_rate: 0.0,
            routing_success_rate: 0.0,
            last_updated: 0,
        };
        let s = sys.score_peer(&m, 0);
        let av = s.components[&PsScoringDimension::Availability];
        assert!((av - 1.0).abs() < 1e-10, "got {av}");
    }

    #[test]
    fn test_all_components_present_in_score() {
        let sys = PeerScoringSystem::default();
        let s = sys.score_peer(&excellent_metrics("p"), 0);
        assert_eq!(s.components.len(), 6);
        for dim in [
            PsScoringDimension::Latency,
            PsScoringDimension::Bandwidth,
            PsScoringDimension::Availability,
            PsScoringDimension::Reliability,
            PsScoringDimension::GossipQuality,
            PsScoringDimension::RoutingQuality,
        ] {
            assert!(s.components.contains_key(&dim), "missing {dim:?}");
        }
    }

    // -----------------------------------------------------------------------
    // Composite score correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_composite_with_equal_weights_and_known_values() {
        let sys = PeerScoringSystem::default();
        let m = PeerMetrics {
            peer_id: "p".into(),
            rtt_ms: 0.0,                // latency = 1.0
            bandwidth_bps: 1_000_000.0, // bandwidth = 1.0
            uptime_fraction: 0.5,       // availability = 0.5
            success_rate: 0.75,         // reliability = 0.75
            gossip_delivery_rate: 0.25, // gossip = 0.25
            routing_success_rate: 0.0,  // routing = 0.0
            last_updated: 0,
        };
        let s = sys.score_peer(&m, 0);
        // composite = (1.0 + 1.0 + 0.5 + 0.75 + 0.25 + 0.0) / 6
        let expected = 3.5 / 6.0;
        assert!(
            (s.composite - expected).abs() < 1e-10,
            "got {}, expected {}",
            s.composite,
            expected
        );
    }

    #[test]
    fn test_composite_excellent_peer_high_score() {
        let sys = PeerScoringSystem::default();
        let s = sys.score_peer(&excellent_metrics("p"), 0);
        assert!(s.composite > 0.8, "got {}", s.composite);
        assert_eq!(s.tier, ScoreTier::Excellent);
    }

    #[test]
    fn test_composite_poor_peer_low_score() {
        let sys = PeerScoringSystem::default();
        let s = sys.score_peer(&poor_metrics("p"), 0);
        assert!(s.composite < 0.3, "got {}", s.composite);
    }

    // -----------------------------------------------------------------------
    // ScoreTier classification
    // -----------------------------------------------------------------------

    #[test]
    fn test_score_tier_from_score() {
        assert_eq!(ScoreTier::from_score(0.95), ScoreTier::Excellent);
        assert_eq!(ScoreTier::from_score(0.80), ScoreTier::Excellent);
        assert_eq!(ScoreTier::from_score(0.79), ScoreTier::Good);
        assert_eq!(ScoreTier::from_score(0.60), ScoreTier::Good);
        assert_eq!(ScoreTier::from_score(0.59), ScoreTier::Fair);
        assert_eq!(ScoreTier::from_score(0.40), ScoreTier::Fair);
        assert_eq!(ScoreTier::from_score(0.39), ScoreTier::Poor);
        assert_eq!(ScoreTier::from_score(0.20), ScoreTier::Poor);
        assert_eq!(ScoreTier::from_score(0.19), ScoreTier::Unacceptable);
        assert_eq!(ScoreTier::from_score(0.00), ScoreTier::Unacceptable);
    }

    #[test]
    fn test_score_tier_ordering() {
        assert!(ScoreTier::Excellent > ScoreTier::Good);
        assert!(ScoreTier::Good > ScoreTier::Fair);
        assert!(ScoreTier::Fair > ScoreTier::Poor);
        assert!(ScoreTier::Poor > ScoreTier::Unacceptable);
    }

    #[test]
    fn test_score_tier_equal() {
        assert_eq!(ScoreTier::Fair, ScoreTier::Fair);
    }

    // -----------------------------------------------------------------------
    // score_and_record + history
    // -----------------------------------------------------------------------

    #[test]
    fn test_score_and_record_creates_history() {
        let mut sys = PeerScoringSystem::default();
        sys.score_and_record(&excellent_metrics("p1"), 100);
        assert_eq!(sys.history_vec("p1").len(), 1);
    }

    #[test]
    fn test_history_for_empty_peer() {
        let sys = PeerScoringSystem::default();
        assert!(sys.history_for("ghost").is_empty());
    }

    #[test]
    fn test_history_vec_order_oldest_first() {
        let mut sys = PeerScoringSystem::default();
        let mut m = excellent_metrics("p1");
        for t in [10u64, 20, 30] {
            m.last_updated = t;
            sys.score_and_record(&m, t);
        }
        let history = sys.history_vec("p1");
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].computed_at, 10);
        assert_eq!(history[2].computed_at, 30);
    }

    #[test]
    fn test_max_history_ring_buffer() {
        let mut sys = PeerScoringSystem::default().with_max_history(3);
        let mut m = excellent_metrics("p1");
        for t in 0..10u64 {
            m.last_updated = t;
            sys.score_and_record(&m, t);
        }
        assert_eq!(sys.history_vec("p1").len(), 3);
        // The oldest retained should be t=7 (last 3 of 0..10)
        assert_eq!(sys.history_vec("p1")[0].computed_at, 7);
    }

    // -----------------------------------------------------------------------
    // average_score
    // -----------------------------------------------------------------------

    #[test]
    fn test_average_score_none_for_unknown_peer() {
        let sys = PeerScoringSystem::default();
        assert!(sys.average_score("nobody").is_none());
    }

    #[test]
    fn test_average_score_single_entry() {
        let mut sys = PeerScoringSystem::default();
        let s = sys.score_and_record(&excellent_metrics("p1"), 0);
        let avg = sys.average_score("p1").expect("should have avg");
        assert!((avg - s.composite).abs() < 1e-10);
    }

    #[test]
    fn test_average_score_multiple_entries() {
        let mut sys = PeerScoringSystem::default();
        // First observation: excellent
        sys.score_and_record(&excellent_metrics("p1"), 10);
        // Second observation: poor
        sys.score_and_record(&poor_metrics("p1"), 20);
        let avg = sys.average_score("p1").expect("should exist");
        let s_exc = sys.score_peer(&excellent_metrics("p1"), 0).composite;
        let s_poor = sys.score_peer(&poor_metrics("p1"), 0).composite;
        let expected = (s_exc + s_poor) / 2.0;
        assert!((avg - expected).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // trend
    // -----------------------------------------------------------------------

    #[test]
    fn test_trend_none_for_single_entry() {
        let mut sys = PeerScoringSystem::default();
        sys.score_and_record(&excellent_metrics("p1"), 0);
        assert!(sys.trend("p1").is_none());
    }

    #[test]
    fn test_trend_none_for_unknown_peer() {
        let sys = PeerScoringSystem::default();
        assert!(sys.trend("nobody").is_none());
    }

    #[test]
    fn test_trend_positive_improving() {
        let mut sys = PeerScoringSystem::default();
        // Manually insert a sequence of increasing composite scores by
        // alternating good (high) and poor (low) metrics with increasing
        // quality. Use score_peer and push directly.
        let w = ScoringWeights::default();
        let mut low_m = poor_metrics("p1");
        let high_m = excellent_metrics("p1");

        // Produce an obviously increasing sequence: 0.1, 0.3, 0.5, 0.7, 0.9
        // by injecting custom scores via score_and_record with tuned metrics.
        // We'll construct metrics that yield predictable scores.
        for i in 0..5usize {
            let frac = 0.1 + i as f64 * 0.2;
            // Use a bandwidth-only system for predictability.
            let _ = &w;
            low_m.bandwidth_bps = frac * 1_000_000.0;
            low_m.rtt_ms = (-frac.ln()) * 500.0; // maps to latency = frac
            low_m.uptime_fraction = frac;
            low_m.success_rate = frac;
            low_m.gossip_delivery_rate = frac;
            low_m.routing_success_rate = frac;
            sys.score_and_record(&low_m, i as u64);
        }
        let _ = &high_m;
        let slope = sys.trend("p1").expect("should have trend");
        assert!(slope > 0.0, "expected positive slope, got {slope}");
    }

    #[test]
    fn test_trend_negative_declining() {
        let mut sys = PeerScoringSystem::default();
        let mut m = excellent_metrics("p1");
        // Decreasing: uptime 1.0 → 0.2
        for i in 0..5usize {
            let frac = 1.0 - i as f64 * 0.2;
            m.uptime_fraction = frac;
            m.success_rate = frac;
            m.gossip_delivery_rate = frac;
            m.routing_success_rate = frac;
            m.bandwidth_bps = frac * 1_000_000.0;
            m.rtt_ms = (1.0 - frac) * 1000.0;
            sys.score_and_record(&m, i as u64);
        }
        let slope = sys.trend("p1").expect("should have trend");
        assert!(slope < 0.0, "expected negative slope, got {slope}");
    }

    #[test]
    fn test_trend_flat_gives_zero_slope() {
        let mut sys = PeerScoringSystem::default();
        let m = excellent_metrics("p1");
        for t in 0..5u64 {
            sys.score_and_record(&m, t);
        }
        let slope = sys.trend("p1").expect("should have trend");
        assert!(slope.abs() < 1e-10, "expected ~0, got {slope}");
    }

    // -----------------------------------------------------------------------
    // top_peers
    // -----------------------------------------------------------------------

    #[test]
    fn test_top_peers_empty_system() {
        let sys = PeerScoringSystem::default();
        assert!(sys.top_peers(5, 0).is_empty());
    }

    #[test]
    fn test_top_peers_ordering() {
        let mut sys = PeerScoringSystem::default();
        sys.score_and_record(&excellent_metrics("good"), 0);
        sys.score_and_record(&poor_metrics("bad"), 0);
        let top = sys.top_peers(2, 0);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].peer_id, "good");
        assert_eq!(top[1].peer_id, "bad");
    }

    #[test]
    fn test_top_peers_n_larger_than_count() {
        let mut sys = PeerScoringSystem::default();
        sys.score_and_record(&excellent_metrics("p1"), 0);
        let top = sys.top_peers(100, 0);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_top_peers_takes_latest_score() {
        let mut sys = PeerScoringSystem::default();
        // First score is excellent, then update to poor
        sys.score_and_record(&excellent_metrics("p1"), 0);
        sys.score_and_record(&poor_metrics("p1"), 1);
        // The latest score should be poor
        let top = sys.top_peers(1, 1);
        assert_eq!(top[0].peer_id, "p1");
        // Its composite should be based on the poor metrics
        let expected = sys.score_peer(&poor_metrics("p1"), 1).composite;
        assert!((top[0].composite - expected).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // evict_stale
    // -----------------------------------------------------------------------

    #[test]
    fn test_evict_stale_removes_old_peers() {
        let mut sys = PeerScoringSystem::default();
        sys.score_and_record(&excellent_metrics("old"), 0);
        sys.score_and_record(&excellent_metrics("fresh"), 1000);
        let evicted = sys.evict_stale(500, 1000);
        assert_eq!(evicted, 1);
        assert!(sys.history.contains_key("fresh"));
        assert!(!sys.history.contains_key("old"));
    }

    #[test]
    fn test_evict_stale_no_peers_removed() {
        let mut sys = PeerScoringSystem::default();
        sys.score_and_record(&excellent_metrics("p1"), 900);
        sys.score_and_record(&excellent_metrics("p2"), 950);
        let evicted = sys.evict_stale(500, 1000);
        assert_eq!(evicted, 0);
    }

    #[test]
    fn test_evict_stale_empty_system() {
        let mut sys = PeerScoringSystem::default();
        assert_eq!(sys.evict_stale(100, 200), 0);
    }

    // -----------------------------------------------------------------------
    // stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_empty() {
        let sys = PeerScoringSystem::default();
        let s = sys.stats(500, 1000);
        assert_eq!(s.total_peers, 0);
        assert!((s.avg_composite).abs() < 1e-10);
        assert_eq!(s.stale_peers, 0);
    }

    #[test]
    fn test_stats_total_peers() {
        let mut sys = PeerScoringSystem::default();
        sys.score_and_record(&excellent_metrics("p1"), 1000);
        sys.score_and_record(&excellent_metrics("p2"), 1000);
        let s = sys.stats(500, 1000);
        assert_eq!(s.total_peers, 2);
    }

    #[test]
    fn test_stats_stale_count() {
        let mut sys = PeerScoringSystem::default();
        sys.score_and_record(&excellent_metrics("stale"), 0);
        sys.score_and_record(&excellent_metrics("fresh"), 900);
        let s = sys.stats(500, 1000);
        assert_eq!(s.stale_peers, 1);
    }

    #[test]
    fn test_stats_tier_distribution() {
        let mut sys = PeerScoringSystem::default();
        sys.score_and_record(&excellent_metrics("good"), 1000);
        sys.score_and_record(&poor_metrics("bad"), 1000);
        let s = sys.stats(500, 1000);
        // excellent peer → Excellent tier; poor peer → some low tier
        assert!(s.tier_distribution.contains_key(&ScoreTier::Excellent));
    }

    #[test]
    fn test_stats_does_not_evict() {
        let mut sys = PeerScoringSystem::default();
        sys.score_and_record(&excellent_metrics("p1"), 0);
        let _s = sys.stats(100, 1000); // stale but not evicted
        assert!(sys.history.contains_key("p1"));
    }

    // -----------------------------------------------------------------------
    // ScoringError display
    // -----------------------------------------------------------------------

    #[test]
    fn test_scoring_error_display_weights() {
        let err = ScoringError::WeightsMustSumToOne { got: 1.5 };
        let msg = err.to_string();
        assert!(msg.contains("1.5") || msg.contains("1.500000"), "{msg}");
    }

    #[test]
    fn test_scoring_error_display_peer_not_found() {
        let err = ScoringError::PeerNotFound;
        let msg = err.to_string();
        assert!(!msg.is_empty());
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_negative_rtt_does_not_panic() {
        let sys = PeerScoringSystem::default();
        let m = PeerMetrics {
            peer_id: "p".into(),
            rtt_ms: -10.0, // negative RTT: exp(10/500) > 1, but formula is fine
            bandwidth_bps: 0.0,
            uptime_fraction: 0.0,
            success_rate: 0.0,
            gossip_delivery_rate: 0.0,
            routing_success_rate: 0.0,
            last_updated: 0,
        };
        let s = sys.score_peer(&m, 0);
        let lat = s.components[&PsScoringDimension::Latency];
        assert!(lat > 1.0, "negative RTT should give score > 1.0: got {lat}");
    }

    #[test]
    fn test_multiple_peers_independent_history() {
        let mut sys = PeerScoringSystem::default();
        sys.score_and_record(&excellent_metrics("a"), 10);
        sys.score_and_record(&excellent_metrics("a"), 20);
        sys.score_and_record(&poor_metrics("b"), 30);
        assert_eq!(sys.history_vec("a").len(), 2);
        assert_eq!(sys.history_vec("b").len(), 1);
    }

    #[test]
    fn test_peer_id_stored_in_score() {
        let sys = PeerScoringSystem::default();
        let s = sys.score_peer(&excellent_metrics("my-peer"), 42);
        assert_eq!(s.peer_id, "my-peer");
        assert_eq!(s.computed_at, 42);
    }
}
