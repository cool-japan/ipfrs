//! Peer Latency Predictor
//!
//! Predicts future peer latency using exponentially-weighted moving average (EWMA)
//! and trend detection. Tracks per-peer RTT samples, computes jitter via EWMA of
//! squared deviations, and provides conservative predicted RTT estimates.
//!
//! # Features
//!
//! - EWMA-based latency smoothing with configurable alpha factor
//! - EWMA-based variance (jitter) tracking with configurable beta factor
//! - Trend detection (Improving / Stable / Degrading) from rolling EWMA history
//! - Stale peer eviction based on configurable timeout
//! - Best-peer ranking by predicted RTT
//!
//! # Example
//!
//! ```rust
//! use ipfrs_network::latency_predictor::{
//!     LatencySample, PeerLatencyPredictor, PredictorConfig,
//! };
//!
//! let config = PredictorConfig::default();
//! let mut predictor = PeerLatencyPredictor::new(config);
//!
//! predictor.record_sample(LatencySample {
//!     peer_id: "peer-a".to_string(),
//!     rtt_ms: 42.0,
//!     timestamp_secs: 1_000,
//! });
//!
//! let predicted = predictor.predict("peer-a");
//! assert!(predicted.is_some());
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single RTT measurement for a peer.
#[derive(Debug, Clone)]
pub struct LatencySample {
    /// Identifier of the remote peer.
    pub peer_id: String,
    /// Round-trip time in milliseconds.
    pub rtt_ms: f64,
    /// Unix timestamp (seconds) when the sample was taken.
    pub timestamp_secs: u64,
}

/// Direction of latency trend inferred from recent EWMA history.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrendDirection {
    /// Recent EWMAs are decreasing — connection is getting faster.
    Improving,
    /// Recent EWMAs are neither clearly improving nor degrading.
    Stable,
    /// Recent EWMAs are increasing — connection is getting slower.
    Degrading,
}

/// Per-peer latency tracking state.
#[derive(Debug, Clone)]
pub struct PeerLatencyState {
    /// Identifier of the tracked peer.
    pub peer_id: String,
    /// Current EWMA of RTT (milliseconds).
    pub ewma_ms: f64,
    /// EWMA of squared deviation from the EWMA (variance proxy for jitter).
    pub ewma_variance: f64,
    /// Number of samples recorded so far.
    pub sample_count: u64,
    /// Rolling window of the last 5 EWMA values (oldest → newest).
    pub recent_ewmas: Vec<f64>,
    /// Unix timestamp (seconds) of the most recent sample.
    pub last_updated_secs: u64,
}

impl PeerLatencyState {
    /// Returns the estimated jitter in milliseconds (sqrt of EWMA variance).
    pub fn jitter_ms(&self) -> f64 {
        self.ewma_variance.sqrt()
    }

    /// Returns a conservative predicted RTT: EWMA + 0.5 * jitter.
    pub fn predicted_rtt_ms(&self) -> f64 {
        self.ewma_ms + 0.5 * self.jitter_ms()
    }

    /// Infers the latency trend from the rolling EWMA window.
    ///
    /// Requires at least 2 values. Splits `recent_ewmas` into first-half and
    /// second-half, comparing their means:
    /// - second > first × 1.05  → `Degrading`
    /// - second < first × 0.95  → `Improving`
    /// - otherwise               → `Stable`
    pub fn trend(&self) -> TrendDirection {
        let n = self.recent_ewmas.len();
        if n < 2 {
            return TrendDirection::Stable;
        }

        let mid = n / 2;
        let first_half = &self.recent_ewmas[..mid];
        let second_half = &self.recent_ewmas[mid..];

        let first_avg = first_half.iter().copied().sum::<f64>() / first_half.len() as f64;
        let second_avg = second_half.iter().copied().sum::<f64>() / second_half.len() as f64;

        if second_avg > first_avg * 1.05 {
            TrendDirection::Degrading
        } else if second_avg < first_avg * 0.95 {
            TrendDirection::Improving
        } else {
            TrendDirection::Stable
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`PeerLatencyPredictor`].
#[derive(Debug, Clone)]
pub struct PredictorConfig {
    /// EWMA smoothing factor for the mean RTT (0 < alpha ≤ 1).
    ///
    /// Higher values give more weight to recent samples.
    pub alpha: f64,

    /// EWMA smoothing factor for the variance (0 < beta ≤ 1).
    pub beta: f64,

    /// Age threshold (seconds) after which a peer is considered stale and
    /// eligible for eviction via [`PeerLatencyPredictor::evict_stale`].
    pub stale_threshold_secs: u64,
}

impl Default for PredictorConfig {
    fn default() -> Self {
        Self {
            alpha: 0.2,
            beta: 0.1,
            stale_threshold_secs: 300,
        }
    }
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/// Aggregated statistics snapshot for a [`PeerLatencyPredictor`].
#[derive(Debug, Clone)]
pub struct PredictorStats {
    /// Number of peers currently tracked.
    pub active_peers: usize,
    /// Cumulative number of samples recorded across all peers.
    pub total_samples: u64,
}

impl PredictorStats {
    /// Mean predicted RTT across the provided peer states.
    ///
    /// Returns `0.0` if `states` is empty.
    pub fn avg_predicted_rtt(&self, states: &[PeerLatencyState]) -> f64 {
        if states.is_empty() {
            return 0.0;
        }
        let sum: f64 = states.iter().map(|s| s.predicted_rtt_ms()).sum();
        sum / states.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Predictor
// ---------------------------------------------------------------------------

/// Maximum number of recent EWMA values to retain per peer.
const RECENT_EWMA_WINDOW: usize = 5;

/// Tracks per-peer RTT samples and provides latency predictions.
#[derive(Debug)]
pub struct PeerLatencyPredictor {
    /// Per-peer state keyed by peer ID string.
    pub states: HashMap<String, PeerLatencyState>,
    /// Predictor configuration.
    pub config: PredictorConfig,
    /// Cumulative sample count across all peers.
    pub total_samples: u64,
}

impl PeerLatencyPredictor {
    /// Creates a new predictor with the given configuration.
    pub fn new(config: PredictorConfig) -> Self {
        Self {
            states: HashMap::new(),
            config,
            total_samples: 0,
        }
    }

    /// Records a latency sample for a peer, updating EWMA and variance.
    ///
    /// On the **first** sample for a peer the EWMA is initialised to the
    /// observed RTT and the variance to `0.0`. Subsequent samples update via:
    ///
    /// ```text
    /// ewma      = alpha * rtt + (1 - alpha) * ewma
    /// deviation = rtt - ewma          (post-update deviation)
    /// variance  = beta * deviation^2 + (1 - beta) * variance
    /// ```
    pub fn record_sample(&mut self, sample: LatencySample) {
        let alpha = self.config.alpha;
        let beta = self.config.beta;
        let rtt = sample.rtt_ms;
        let ts = sample.timestamp_secs;

        if !self.states.contains_key(&sample.peer_id) {
            // First observation: initialise directly.
            self.states.insert(
                sample.peer_id.clone(),
                PeerLatencyState {
                    peer_id: sample.peer_id,
                    ewma_ms: rtt,
                    ewma_variance: 0.0,
                    sample_count: 1,
                    recent_ewmas: vec![rtt],
                    last_updated_secs: ts,
                },
            );
            self.total_samples += 1;
            return;
        }

        // Subsequent observation: update EWMA and variance.
        let state = self
            .states
            .get_mut(&sample.peer_id)
            .expect("key confirmed above");

        state.ewma_ms = alpha * rtt + (1.0 - alpha) * state.ewma_ms;

        let deviation = rtt - state.ewma_ms;
        state.ewma_variance = beta * deviation * deviation + (1.0 - beta) * state.ewma_variance;

        // Maintain rolling window of recent EWMA values (max RECENT_EWMA_WINDOW).
        state.recent_ewmas.push(state.ewma_ms);
        if state.recent_ewmas.len() > RECENT_EWMA_WINDOW {
            state.recent_ewmas.remove(0);
        }

        state.sample_count += 1;
        state.last_updated_secs = ts;
        self.total_samples += 1;
    }

    /// Returns the predicted RTT for `peer_id`, or `None` if unknown.
    pub fn predict(&self, peer_id: &str) -> Option<f64> {
        self.states.get(peer_id).map(|s| s.predicted_rtt_ms())
    }

    /// Removes peers whose last sample is older than `now_secs - stale_threshold_secs`.
    ///
    /// Returns the number of peers evicted.
    pub fn evict_stale(&mut self, now_secs: u64) -> usize {
        let threshold = self.config.stale_threshold_secs;
        let before = self.states.len();
        self.states.retain(|_, state| {
            // Keep peers that are recent enough.
            now_secs.saturating_sub(state.last_updated_secs) < threshold
        });
        before - self.states.len()
    }

    /// Returns peer IDs of the `n` peers with the lowest predicted RTT, in
    /// ascending order (best first).
    ///
    /// If `n` exceeds the number of tracked peers, all are returned.
    pub fn best_peers(&self, n: usize) -> Vec<&str> {
        let mut entries: Vec<(&str, f64)> = self
            .states
            .iter()
            .map(|(id, state)| (id.as_str(), state.predicted_rtt_ms()))
            .collect();

        // Sort ascending by predicted RTT; use peer_id as tiebreaker for determinism.
        entries.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(b.0))
        });

        entries.into_iter().take(n).map(|(id, _)| id).collect()
    }

    /// Returns a snapshot of predictor statistics.
    pub fn stats(&self) -> PredictorStats {
        PredictorStats {
            active_peers: self.states.len(),
            total_samples: self.total_samples,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sample(peer_id: &str, rtt_ms: f64, timestamp_secs: u64) -> LatencySample {
        LatencySample {
            peer_id: peer_id.to_string(),
            rtt_ms,
            timestamp_secs,
        }
    }

    fn default_predictor() -> PeerLatencyPredictor {
        PeerLatencyPredictor::new(PredictorConfig::default())
    }

    // ------------------------------------------------------------------
    // 1. New peer is initialised correctly
    // ------------------------------------------------------------------

    #[test]
    fn test_new_peer_ewma_equals_first_rtt() {
        let mut p = default_predictor();
        p.record_sample(make_sample("peer-a", 50.0, 1000));
        let state = p.states.get("peer-a").expect("peer should exist");
        assert_eq!(state.ewma_ms, 50.0);
    }

    #[test]
    fn test_new_peer_variance_is_zero() {
        let mut p = default_predictor();
        p.record_sample(make_sample("peer-b", 30.0, 1000));
        let state = p.states.get("peer-b").expect("peer should exist");
        assert_eq!(state.ewma_variance, 0.0);
    }

    #[test]
    fn test_new_peer_sample_count_is_one() {
        let mut p = default_predictor();
        p.record_sample(make_sample("peer-c", 20.0, 1000));
        let state = p.states.get("peer-c").expect("peer should exist");
        assert_eq!(state.sample_count, 1);
    }

    #[test]
    fn test_new_peer_recent_ewmas_has_one_entry() {
        let mut p = default_predictor();
        p.record_sample(make_sample("peer-d", 20.0, 1000));
        let state = p.states.get("peer-d").expect("peer should exist");
        assert_eq!(state.recent_ewmas.len(), 1);
        assert_eq!(state.recent_ewmas[0], 20.0);
    }

    // ------------------------------------------------------------------
    // 2. EWMA moves toward recent samples
    // ------------------------------------------------------------------

    #[test]
    fn test_ewma_moves_toward_recent_higher_value() {
        let mut p = default_predictor();
        // Initialize at 50 ms
        p.record_sample(make_sample("peer-a", 50.0, 1000));
        // Feed a much higher sample — EWMA should increase
        p.record_sample(make_sample("peer-a", 200.0, 1001));
        let state = p.states.get("peer-a").expect("peer should exist");
        assert!(
            state.ewma_ms > 50.0,
            "EWMA ({}) should be above initial 50 ms",
            state.ewma_ms
        );
        assert!(
            state.ewma_ms < 200.0,
            "EWMA ({}) should be below raw sample 200 ms",
            state.ewma_ms
        );
    }

    #[test]
    fn test_ewma_moves_toward_recent_lower_value() {
        let mut p = default_predictor();
        p.record_sample(make_sample("peer-a", 200.0, 1000));
        p.record_sample(make_sample("peer-a", 20.0, 1001));
        let state = p.states.get("peer-a").expect("peer should exist");
        assert!(
            state.ewma_ms < 200.0,
            "EWMA ({}) should be below initial 200 ms",
            state.ewma_ms
        );
        assert!(
            state.ewma_ms > 20.0,
            "EWMA ({}) should be above raw sample 20 ms",
            state.ewma_ms
        );
    }

    #[test]
    fn test_ewma_converges_on_constant_input() {
        let mut p = default_predictor();
        p.record_sample(make_sample("peer-a", 100.0, 1000));
        for t in 1..=30_u64 {
            p.record_sample(make_sample("peer-a", 10.0, 1000 + t));
        }
        let state = p.states.get("peer-a").expect("peer should exist");
        // After many samples of 10 ms the EWMA should be close to 10.
        assert!(
            state.ewma_ms < 15.0,
            "EWMA ({}) should converge toward 10 ms",
            state.ewma_ms
        );
    }

    // ------------------------------------------------------------------
    // 3. jitter_ms = sqrt(ewma_variance)
    // ------------------------------------------------------------------

    #[test]
    fn test_jitter_ms_is_sqrt_of_variance() {
        let state = PeerLatencyState {
            peer_id: "x".to_string(),
            ewma_ms: 50.0,
            ewma_variance: 25.0,
            sample_count: 5,
            recent_ewmas: vec![50.0],
            last_updated_secs: 1000,
        };
        let jitter = state.jitter_ms();
        assert!(
            (jitter - 5.0).abs() < 1e-9,
            "jitter_ms should be sqrt(25) = 5.0, got {}",
            jitter
        );
    }

    #[test]
    fn test_jitter_zero_for_new_peer() {
        let mut p = default_predictor();
        p.record_sample(make_sample("peer-a", 50.0, 1000));
        let state = p.states.get("peer-a").expect("peer should exist");
        assert_eq!(state.jitter_ms(), 0.0);
    }

    // ------------------------------------------------------------------
    // 4. predicted_rtt_ms includes jitter
    // ------------------------------------------------------------------

    #[test]
    fn test_predicted_rtt_includes_half_jitter() {
        let state = PeerLatencyState {
            peer_id: "x".to_string(),
            ewma_ms: 40.0,
            ewma_variance: 16.0, // jitter = 4.0
            sample_count: 3,
            recent_ewmas: vec![40.0],
            last_updated_secs: 1000,
        };
        // expected = 40.0 + 0.5 * 4.0 = 42.0
        let predicted = state.predicted_rtt_ms();
        assert!(
            (predicted - 42.0).abs() < 1e-9,
            "predicted_rtt_ms should be 42.0, got {}",
            predicted
        );
    }

    #[test]
    fn test_predicted_rtt_equals_ewma_when_no_jitter() {
        let state = PeerLatencyState {
            peer_id: "x".to_string(),
            ewma_ms: 55.0,
            ewma_variance: 0.0,
            sample_count: 1,
            recent_ewmas: vec![55.0],
            last_updated_secs: 1000,
        };
        assert_eq!(state.predicted_rtt_ms(), 55.0);
    }

    // ------------------------------------------------------------------
    // 5. Trend detection
    // ------------------------------------------------------------------

    #[test]
    fn test_trend_stable_with_fewer_than_two_ewmas() {
        let state = PeerLatencyState {
            peer_id: "x".to_string(),
            ewma_ms: 50.0,
            ewma_variance: 0.0,
            sample_count: 1,
            recent_ewmas: vec![50.0],
            last_updated_secs: 1000,
        };
        assert_eq!(state.trend(), TrendDirection::Stable);
    }

    #[test]
    fn test_trend_degrading() {
        // recent_ewmas ascending — second half average >> first half
        let state = PeerLatencyState {
            peer_id: "x".to_string(),
            ewma_ms: 100.0,
            ewma_variance: 0.0,
            sample_count: 5,
            // first half: [10, 12] avg=11; second half: [80, 90, 100] avg=90
            recent_ewmas: vec![10.0, 12.0, 80.0, 90.0, 100.0],
            last_updated_secs: 1000,
        };
        assert_eq!(state.trend(), TrendDirection::Degrading);
    }

    #[test]
    fn test_trend_improving() {
        let state = PeerLatencyState {
            peer_id: "x".to_string(),
            ewma_ms: 10.0,
            ewma_variance: 0.0,
            sample_count: 5,
            // first half: [100, 90] avg=95; second half: [20, 15, 10] avg=15
            recent_ewmas: vec![100.0, 90.0, 20.0, 15.0, 10.0],
            last_updated_secs: 1000,
        };
        assert_eq!(state.trend(), TrendDirection::Improving);
    }

    #[test]
    fn test_trend_stable_flat() {
        let state = PeerLatencyState {
            peer_id: "x".to_string(),
            ewma_ms: 50.0,
            ewma_variance: 0.0,
            sample_count: 4,
            // first half: [50, 51] avg≈50.5; second half: [50, 50] avg=50 — within 5%
            recent_ewmas: vec![50.0, 51.0, 50.0, 50.0],
            last_updated_secs: 1000,
        };
        assert_eq!(state.trend(), TrendDirection::Stable);
    }

    // ------------------------------------------------------------------
    // 6. evict_stale
    // ------------------------------------------------------------------

    #[test]
    fn test_evict_stale_removes_old_peers() {
        let mut p = PeerLatencyPredictor::new(PredictorConfig {
            stale_threshold_secs: 60,
            ..Default::default()
        });
        // old-peer last updated at t=1000, fresh-peer at t=2000
        p.record_sample(make_sample("old-peer", 50.0, 1000));
        p.record_sample(make_sample("fresh-peer", 30.0, 2000));

        // now_secs = 2050 → old-peer age=1050 >= 60 → evicted
        //                  → fresh-peer age=50 < 60  → kept
        let evicted = p.evict_stale(2050);
        assert_eq!(evicted, 1, "one stale peer should be evicted");
        assert!(!p.states.contains_key("old-peer"));
        assert!(p.states.contains_key("fresh-peer"));
    }

    #[test]
    fn test_evict_stale_keeps_fresh_peers() {
        let mut p = PeerLatencyPredictor::new(PredictorConfig {
            stale_threshold_secs: 300,
            ..Default::default()
        });
        p.record_sample(make_sample("peer-a", 40.0, 1000));
        let evicted = p.evict_stale(1200); // age = 200 < 300
        assert_eq!(evicted, 0);
        assert!(p.states.contains_key("peer-a"));
    }

    #[test]
    fn test_evict_stale_returns_zero_when_nothing_to_evict() {
        let mut p = default_predictor();
        assert_eq!(p.evict_stale(99999), 0);
    }

    // ------------------------------------------------------------------
    // 7. best_peers
    // ------------------------------------------------------------------

    #[test]
    fn test_best_peers_sorted_ascending() {
        let mut p = default_predictor();
        p.record_sample(make_sample("slow", 200.0, 1000));
        p.record_sample(make_sample("fast", 10.0, 1000));
        p.record_sample(make_sample("medium", 50.0, 1000));

        let best = p.best_peers(3);
        assert_eq!(best[0], "fast");
        assert_eq!(best[1], "medium");
        assert_eq!(best[2], "slow");
    }

    #[test]
    fn test_best_peers_respects_n_limit() {
        let mut p = default_predictor();
        for i in 0..10_u64 {
            p.record_sample(make_sample(
                &format!("peer-{}", i),
                i as f64 * 10.0 + 5.0,
                1000,
            ));
        }
        let best = p.best_peers(3);
        assert_eq!(best.len(), 3);
    }

    #[test]
    fn test_best_peers_all_when_n_exceeds_peer_count() {
        let mut p = default_predictor();
        p.record_sample(make_sample("a", 10.0, 1000));
        p.record_sample(make_sample("b", 20.0, 1000));
        let best = p.best_peers(100);
        assert_eq!(best.len(), 2);
    }

    // ------------------------------------------------------------------
    // 8. stats
    // ------------------------------------------------------------------

    #[test]
    fn test_stats_active_peers() {
        let mut p = default_predictor();
        p.record_sample(make_sample("p1", 10.0, 1000));
        p.record_sample(make_sample("p2", 20.0, 1000));
        assert_eq!(p.stats().active_peers, 2);
    }

    #[test]
    fn test_stats_total_samples() {
        let mut p = default_predictor();
        p.record_sample(make_sample("p1", 10.0, 1000));
        p.record_sample(make_sample("p1", 15.0, 1001));
        p.record_sample(make_sample("p2", 20.0, 1002));
        assert_eq!(p.stats().total_samples, 3);
    }

    #[test]
    fn test_stats_avg_predicted_rtt() {
        let states = vec![
            PeerLatencyState {
                peer_id: "a".to_string(),
                ewma_ms: 40.0,
                ewma_variance: 0.0,
                sample_count: 1,
                recent_ewmas: vec![40.0],
                last_updated_secs: 1000,
            },
            PeerLatencyState {
                peer_id: "b".to_string(),
                ewma_ms: 60.0,
                ewma_variance: 0.0,
                sample_count: 1,
                recent_ewmas: vec![60.0],
                last_updated_secs: 1000,
            },
        ];
        let stats = PredictorStats {
            active_peers: 2,
            total_samples: 2,
        };
        let avg = stats.avg_predicted_rtt(&states);
        assert!(
            (avg - 50.0).abs() < 1e-9,
            "avg should be 50 ms, got {}",
            avg
        );
    }

    // ------------------------------------------------------------------
    // 9. predict returns None for unknown peer
    // ------------------------------------------------------------------

    #[test]
    fn test_predict_returns_none_for_unknown_peer() {
        let p = default_predictor();
        assert!(p.predict("no-such-peer").is_none());
    }

    #[test]
    fn test_predict_returns_some_for_known_peer() {
        let mut p = default_predictor();
        p.record_sample(make_sample("known", 100.0, 1000));
        assert!(p.predict("known").is_some());
    }

    // ------------------------------------------------------------------
    // 10. recent_ewmas window capped at 5
    // ------------------------------------------------------------------

    #[test]
    fn test_recent_ewmas_capped_at_five() {
        let mut p = default_predictor();
        for t in 0..20_u64 {
            p.record_sample(make_sample("peer-a", 50.0, 1000 + t));
        }
        let state = p.states.get("peer-a").expect("peer should exist");
        assert!(
            state.recent_ewmas.len() <= 5,
            "recent_ewmas should be capped at 5, got {}",
            state.recent_ewmas.len()
        );
    }
}
