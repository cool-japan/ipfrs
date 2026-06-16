//! Per-peer adaptive timeout estimation using the TCP-inspired SRTT/RTTVAR algorithm.
//!
//! This module dynamically adjusts per-peer operation timeouts based on observed RTT history,
//! ensuring timeouts are neither too aggressive nor too conservative. The algorithm is modelled
//! after RFC 6298 (TCP retransmission timer computation).
//!
//! # Algorithm
//!
//! On the **first** RTT sample for a peer:
//! ```text
//! SRTT   = R
//! RTTVAR = R / 2
//! ```
//!
//! On every **subsequent** sample:
//! ```text
//! RTTVAR = (1 - β) * RTTVAR + β * |SRTT - R|
//! SRTT   = (1 - α) * SRTT   + α * R
//! RTO    = SRTT + 4 * RTTVAR
//! ```
//!
//! where α = 1/8 and β = 1/4 by default (RFC 6298 recommendations).

use std::collections::HashMap;

/// A single RTT measurement for a specific peer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RttSample {
    /// Identifies the remote peer.
    pub peer_id: String,
    /// Round-trip time in microseconds.
    pub rtt_micros: u64,
    /// Logical clock tick at which the sample was taken (application-defined).
    pub sampled_at_tick: u64,
}

/// The current timeout estimate and smoothed RTT state for a single peer.
#[derive(Debug, Clone)]
pub struct TimeoutEstimate {
    /// Identifies the remote peer.
    pub peer_id: String,
    /// Exponentially-weighted moving average of the RTT (µs).
    pub srtt_micros: f64,
    /// Mean deviation of the RTT (µs), used as a variance proxy.
    pub rttvar_micros: f64,
    /// Recommended timeout in microseconds: `srtt + 4 * rttvar`, clamped to [`AdaptiveTimeoutConfig::min_timeout_micros`] .. [`AdaptiveTimeoutConfig::max_timeout_micros`].
    pub timeout_micros: u64,
    /// Number of RTT samples recorded for this peer.
    pub sample_count: u32,
}

/// Configuration for [`PeerAdaptiveTimeout`].
#[derive(Debug, Clone)]
pub struct AdaptiveTimeoutConfig {
    /// SRTT smoothing factor (α). RFC 6298 recommends 1/8.
    pub alpha: f64,
    /// RTTVAR smoothing factor (β). RFC 6298 recommends 1/4.
    pub beta: f64,
    /// Minimum allowed timeout in microseconds (floor). Default: 1 000 µs = 1 ms.
    pub min_timeout_micros: u64,
    /// Maximum allowed timeout in microseconds (ceiling). Default: 30 000 000 µs = 30 s.
    pub max_timeout_micros: u64,
}

impl Default for AdaptiveTimeoutConfig {
    fn default() -> Self {
        Self {
            alpha: 0.125,
            beta: 0.25,
            min_timeout_micros: 1_000,
            max_timeout_micros: 30_000_000,
        }
    }
}

/// Aggregate statistics across all tracked peers.
#[derive(Debug, Clone)]
pub struct TimeoutStats {
    /// Number of distinct peers currently tracked.
    pub total_peers: usize,
    /// Total RTT samples recorded across all peers.
    pub total_samples: u64,
    /// Mean recommended timeout across all tracked peers (µs). 0.0 when no peers are tracked.
    pub avg_timeout_micros: f64,
}

/// Per-peer adaptive timeout manager.
///
/// Maintains a [`TimeoutEstimate`] for every peer that has provided at least one RTT sample and
/// exposes methods to query the recommended timeout or remove stale peers.
#[derive(Debug)]
pub struct PeerAdaptiveTimeout {
    /// Map from peer_id to its current timeout estimate.
    pub estimates: HashMap<String, TimeoutEstimate>,
    /// Algorithm configuration.
    pub config: AdaptiveTimeoutConfig,
    /// Total RTT samples recorded across all peers (monotonically increasing).
    pub total_samples: u64,
}

impl PeerAdaptiveTimeout {
    /// Create a new [`PeerAdaptiveTimeout`] with the given configuration.
    pub fn new(config: AdaptiveTimeoutConfig) -> Self {
        Self {
            estimates: HashMap::new(),
            config,
            total_samples: 0,
        }
    }

    /// Record a new RTT sample and update the timeout estimate for the associated peer.
    ///
    /// # First sample
    ///
    /// ```text
    /// SRTT   = rtt
    /// RTTVAR = rtt / 2
    /// ```
    ///
    /// # Subsequent samples
    ///
    /// ```text
    /// RTTVAR = (1 - β) * RTTVAR + β * |SRTT - rtt|
    /// SRTT   = (1 - α) * SRTT   + α * rtt
    /// ```
    pub fn record_sample(&mut self, sample: RttSample) {
        let rtt = sample.rtt_micros as f64;
        let alpha = self.config.alpha;
        let beta = self.config.beta;
        let min_t = self.config.min_timeout_micros;
        let max_t = self.config.max_timeout_micros;

        let estimate = self
            .estimates
            .entry(sample.peer_id.clone())
            .or_insert_with(|| TimeoutEstimate {
                peer_id: sample.peer_id.clone(),
                srtt_micros: 0.0,
                rttvar_micros: 0.0,
                timeout_micros: 0,
                sample_count: 0,
            });

        if estimate.sample_count == 0 {
            // Initialise on the very first sample.
            estimate.srtt_micros = rtt;
            estimate.rttvar_micros = rtt / 2.0;
        } else {
            // RFC 6298 §2: update RTTVAR before SRTT so the deviation uses the *previous* SRTT.
            let diff = (estimate.srtt_micros - rtt).abs();
            estimate.rttvar_micros = (1.0 - beta) * estimate.rttvar_micros + beta * diff;
            estimate.srtt_micros = (1.0 - alpha) * estimate.srtt_micros + alpha * rtt;
        }

        let raw_timeout = estimate.srtt_micros + 4.0 * estimate.rttvar_micros;
        let clamped = raw_timeout.max(min_t as f64).min(max_t as f64) as u64;
        estimate.timeout_micros = clamped.clamp(min_t, max_t);

        estimate.sample_count += 1;
        self.total_samples += 1;
    }

    /// Returns the recommended timeout for `peer_id` if it has been tracked, or `None` otherwise.
    pub fn timeout_for(&self, peer_id: &str) -> Option<u64> {
        self.estimates.get(peer_id).map(|e| e.timeout_micros)
    }

    /// Returns the recommended timeout for `peer_id`, falling back to `max_timeout_micros` for
    /// unknown peers (conservative default).
    pub fn timeout_or_default(&self, peer_id: &str) -> u64 {
        self.timeout_for(peer_id)
            .unwrap_or(self.config.max_timeout_micros)
    }

    /// Remove all state for `peer_id`.
    ///
    /// Returns `true` if the peer was tracked and has been removed, `false` if it was unknown.
    pub fn remove_peer(&mut self, peer_id: &str) -> bool {
        self.estimates.remove(peer_id).is_some()
    }

    /// Compute aggregate statistics over all currently tracked peers.
    pub fn stats(&self) -> TimeoutStats {
        let total_peers = self.estimates.len();
        let avg_timeout_micros = if total_peers == 0 {
            0.0
        } else {
            let sum: f64 = self
                .estimates
                .values()
                .map(|e| e.timeout_micros as f64)
                .sum();
            sum / total_peers as f64
        };
        TimeoutStats {
            total_peers,
            total_samples: self.total_samples,
            avg_timeout_micros,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_pat() -> PeerAdaptiveTimeout {
        PeerAdaptiveTimeout::new(AdaptiveTimeoutConfig::default())
    }

    fn sample(peer: &str, rtt: u64) -> RttSample {
        RttSample {
            peer_id: peer.to_string(),
            rtt_micros: rtt,
            sampled_at_tick: 0,
        }
    }

    // ── 1. First sample initialises SRTT and RTTVAR correctly ──────────────

    #[test]
    fn test_first_sample_srtt() {
        let mut pat = default_pat();
        pat.record_sample(sample("peer-a", 10_000));
        let est = pat
            .estimates
            .get("peer-a")
            .expect("test: peer-a estimate should exist");
        assert!(
            (est.srtt_micros - 10_000.0).abs() < f64::EPSILON,
            "SRTT should equal first RTT"
        );
    }

    #[test]
    fn test_first_sample_rttvar() {
        let mut pat = default_pat();
        pat.record_sample(sample("peer-a", 10_000));
        let est = pat
            .estimates
            .get("peer-a")
            .expect("test: peer-a estimate should exist");
        assert!(
            (est.rttvar_micros - 5_000.0).abs() < f64::EPSILON,
            "RTTVAR should be RTT/2 on first sample"
        );
    }

    #[test]
    fn test_first_sample_timeout_formula() {
        let mut pat = default_pat();
        pat.record_sample(sample("peer-a", 10_000));
        let est = pat
            .estimates
            .get("peer-a")
            .expect("test: peer-a estimate should exist");
        // timeout = srtt + 4*rttvar = 10000 + 4*5000 = 30000
        assert_eq!(
            est.timeout_micros, 30_000,
            "First-sample timeout = srtt + 4*rttvar"
        );
    }

    // ── 2. Second sample SRTT via EWMA ─────────────────────────────────────

    #[test]
    fn test_second_sample_srtt_ewma() {
        let mut pat = default_pat();
        pat.record_sample(sample("peer-a", 8_000));
        pat.record_sample(sample("peer-a", 12_000));
        let est = pat
            .estimates
            .get("peer-a")
            .expect("test: peer-a estimate should exist");
        // SRTT = (1 - 0.125) * 8000 + 0.125 * 12000 = 7000 + 1500 = 8500
        let expected_srtt = (1.0 - 0.125_f64) * 8_000.0 + 0.125 * 12_000.0;
        assert!(
            (est.srtt_micros - expected_srtt).abs() < 1.0,
            "SRTT EWMA mismatch: got {}, expected {}",
            est.srtt_micros,
            expected_srtt
        );
    }

    // ── 3. Timeout = SRTT + 4*RTTVAR ──────────────────────────────────────

    #[test]
    fn test_timeout_formula_after_two_samples() {
        let mut pat = default_pat();
        pat.record_sample(sample("peer-a", 8_000));
        pat.record_sample(sample("peer-a", 12_000));
        let est = pat
            .estimates
            .get("peer-a")
            .expect("test: peer-a estimate should exist");
        let expected = (est.srtt_micros + 4.0 * est.rttvar_micros) as u64;
        assert_eq!(
            est.timeout_micros, expected,
            "timeout_micros should equal srtt + 4*rttvar"
        );
    }

    // ── 4. Min clamping ────────────────────────────────────────────────────

    #[test]
    fn test_min_timeout_clamping() {
        let config = AdaptiveTimeoutConfig {
            min_timeout_micros: 50_000,
            ..Default::default()
        };
        let mut pat = PeerAdaptiveTimeout::new(config);
        // Tiny RTT — raw timeout will be 1 µs + 4*0.5 µs = 3 µs < 50_000
        pat.record_sample(sample("peer-a", 1));
        let t = pat
            .timeout_for("peer-a")
            .expect("test: peer-a timeout should exist after sample");
        assert_eq!(t, 50_000, "Timeout should be clamped to min_timeout_micros");
    }

    // ── 5. Max clamping ────────────────────────────────────────────────────

    #[test]
    fn test_max_timeout_clamping() {
        let config = AdaptiveTimeoutConfig {
            max_timeout_micros: 5_000,
            ..Default::default()
        };
        let mut pat = PeerAdaptiveTimeout::new(config);
        // Huge RTT — raw timeout far exceeds 5_000
        pat.record_sample(sample("peer-a", 1_000_000));
        let t = pat
            .timeout_for("peer-a")
            .expect("test: peer-a timeout should exist after sample");
        assert_eq!(t, 5_000, "Timeout should be clamped to max_timeout_micros");
    }

    // ── 6. remove_peer returns true when peer existed ──────────────────────

    #[test]
    fn test_remove_peer_existing() {
        let mut pat = default_pat();
        pat.record_sample(sample("peer-a", 1_000));
        assert!(
            pat.remove_peer("peer-a"),
            "Should return true for known peer"
        );
        assert!(
            !pat.estimates.contains_key("peer-a"),
            "Peer should be gone after removal"
        );
    }

    // ── 7. remove_peer returns false for unknown peer ──────────────────────

    #[test]
    fn test_remove_peer_unknown() {
        let mut pat = default_pat();
        assert!(
            !pat.remove_peer("ghost"),
            "Should return false for unknown peer"
        );
    }

    // ── 8. timeout_or_default returns max for unknown peer ─────────────────

    #[test]
    fn test_timeout_or_default_unknown() {
        let pat = default_pat();
        assert_eq!(
            pat.timeout_or_default("nobody"),
            pat.config.max_timeout_micros,
            "Unknown peer should receive max timeout"
        );
    }

    // ── 9. timeout_for returns None for unknown peer ───────────────────────

    #[test]
    fn test_timeout_for_unknown() {
        let pat = default_pat();
        assert!(pat.timeout_for("nobody").is_none());
    }

    // ── 10. stats avg_timeout_micros ──────────────────────────────────────

    #[test]
    fn test_stats_avg_timeout() {
        let mut pat = default_pat();
        pat.record_sample(sample("peer-a", 10_000));
        pat.record_sample(sample("peer-b", 20_000));
        let stats = pat.stats();
        let ta = pat.estimates["peer-a"].timeout_micros as f64;
        let tb = pat.estimates["peer-b"].timeout_micros as f64;
        let expected_avg = (ta + tb) / 2.0;
        assert!(
            (stats.avg_timeout_micros - expected_avg).abs() < 1.0,
            "avg_timeout_micros mismatch"
        );
    }

    // ── 11. stats total_peers ─────────────────────────────────────────────

    #[test]
    fn test_stats_total_peers() {
        let mut pat = default_pat();
        pat.record_sample(sample("peer-a", 5_000));
        pat.record_sample(sample("peer-b", 5_000));
        assert_eq!(pat.stats().total_peers, 2);
    }

    // ── 12. stats total_samples ───────────────────────────────────────────

    #[test]
    fn test_stats_total_samples() {
        let mut pat = default_pat();
        pat.record_sample(sample("peer-a", 1_000));
        pat.record_sample(sample("peer-a", 2_000));
        pat.record_sample(sample("peer-b", 3_000));
        assert_eq!(pat.stats().total_samples, 3);
    }

    // ── 13. Multiple peers tracked independently ──────────────────────────

    #[test]
    fn test_multiple_peers_independent() {
        let mut pat = default_pat();
        pat.record_sample(sample("peer-a", 1_000));
        pat.record_sample(sample("peer-b", 100_000));
        let ta = pat
            .timeout_for("peer-a")
            .expect("test: peer-a timeout should exist");
        let tb = pat
            .timeout_for("peer-b")
            .expect("test: peer-b timeout should exist");
        assert!(
            ta < tb,
            "Peer with lower RTT should have lower timeout: {} vs {}",
            ta,
            tb
        );
    }

    // ── 14. sample_count increments correctly ────────────────────────────

    #[test]
    fn test_sample_count_increments() {
        let mut pat = default_pat();
        for i in 0..5u64 {
            pat.record_sample(sample("peer-a", 1_000 * (i + 1)));
        }
        assert_eq!(pat.estimates["peer-a"].sample_count, 5);
    }

    // ── 15. total_samples increments across peers ─────────────────────────

    #[test]
    fn test_total_samples_across_peers() {
        let mut pat = default_pat();
        pat.record_sample(sample("peer-a", 1_000));
        pat.record_sample(sample("peer-b", 2_000));
        pat.record_sample(sample("peer-c", 3_000));
        assert_eq!(pat.total_samples, 3);
    }

    // ── 16. Stable RTT converges timeout toward RTT ───────────────────────

    #[test]
    fn test_stable_rtt_converges() {
        let mut pat = default_pat();
        let stable_rtt = 20_000u64;
        // After many identical samples SRTT → rtt, RTTVAR → 0, timeout → rtt (floored at min)
        for _ in 0..200 {
            pat.record_sample(sample("peer-a", stable_rtt));
        }
        let est = pat
            .estimates
            .get("peer-a")
            .expect("test: peer-a estimate should exist after convergence");
        let srtt_err = (est.srtt_micros - stable_rtt as f64).abs();
        assert!(
            srtt_err < 1.0,
            "SRTT should converge to stable RTT; error={srtt_err}"
        );
        assert!(
            est.rttvar_micros < 10.0,
            "RTTVAR should converge to near-zero; got {}",
            est.rttvar_micros
        );
        assert!(
            est.timeout_micros < stable_rtt + 100,
            "Timeout should converge near RTT; got {}",
            est.timeout_micros
        );
    }

    // ── 17. Custom alpha/beta configuration ──────────────────────────────

    #[test]
    fn test_custom_alpha_beta() {
        let config = AdaptiveTimeoutConfig {
            alpha: 0.5,
            beta: 0.5,
            ..Default::default()
        };
        let mut pat = PeerAdaptiveTimeout::new(config);
        pat.record_sample(sample("peer-a", 10_000));
        pat.record_sample(sample("peer-a", 20_000));
        let est = pat
            .estimates
            .get("peer-a")
            .expect("test: peer-a estimate should exist");
        // SRTT = (1-0.5)*10000 + 0.5*20000 = 15000
        assert!((est.srtt_micros - 15_000.0).abs() < 1.0);
    }

    // ── 18. Remove peer then re-add resets state ──────────────────────────

    #[test]
    fn test_remove_then_readd_resets_state() {
        let mut pat = default_pat();
        for _ in 0..10 {
            pat.record_sample(sample("peer-a", 5_000));
        }
        pat.remove_peer("peer-a");
        pat.record_sample(sample("peer-a", 100_000));
        let est = pat
            .estimates
            .get("peer-a")
            .expect("test: peer-a estimate should exist after re-add");
        // Should be initialised fresh: srtt = 100_000, rttvar = 50_000
        assert_eq!(est.sample_count, 1);
        assert!((est.srtt_micros - 100_000.0).abs() < 1.0);
        assert!((est.rttvar_micros - 50_000.0).abs() < 1.0);
    }

    // ── 19. Empty tracker stats ───────────────────────────────────────────

    #[test]
    fn test_empty_stats() {
        let pat = default_pat();
        let stats = pat.stats();
        assert_eq!(stats.total_peers, 0);
        assert_eq!(stats.total_samples, 0);
        assert!((stats.avg_timeout_micros - 0.0).abs() < f64::EPSILON);
    }

    // ── 20. timeout_or_default for known peer returns peer timeout ─────────

    #[test]
    fn test_timeout_or_default_known() {
        let mut pat = default_pat();
        pat.record_sample(sample("peer-a", 5_000));
        let expected = pat
            .timeout_for("peer-a")
            .expect("test: peer-a timeout should exist");
        assert_eq!(pat.timeout_or_default("peer-a"), expected);
    }

    // ── 21. Monotonic total_samples ───────────────────────────────────────

    #[test]
    fn test_total_samples_monotone() {
        let mut pat = default_pat();
        for i in 0..10u64 {
            pat.record_sample(sample("peer-a", 1_000 * (i + 1)));
            assert_eq!(pat.total_samples, i + 1);
        }
    }

    // ── 22. High-RTT variance is captured in RTTVAR ───────────────────────

    #[test]
    fn test_high_variance_rttvar() {
        let mut pat = default_pat();
        pat.record_sample(sample("peer-a", 1_000));
        pat.record_sample(sample("peer-a", 100_000));
        let est = pat
            .estimates
            .get("peer-a")
            .expect("test: peer-a estimate should exist");
        // The deviation between 1_000 and 100_000 is large — RTTVAR should be substantial.
        assert!(
            est.rttvar_micros > 10_000.0,
            "RTTVAR should reflect high variance; got {}",
            est.rttvar_micros
        );
    }

    // ── 23. Minimum timeout respected even for tiny RTT ──────────────────

    #[test]
    fn test_min_floor_tiny_rtt() {
        let mut pat = default_pat(); // min = 1_000
        pat.record_sample(sample("peer-a", 1)); // rtt = 1 µs
        assert!(
            pat.timeout_for("peer-a")
                .expect("test: peer-a timeout should be present")
                >= 1_000,
            "Timeout must not fall below min"
        );
    }

    // ── 24. Successive samples do not regress below min ───────────────────

    #[test]
    fn test_timeout_never_below_min_over_many_samples() {
        let mut pat = default_pat();
        for i in 0..50u64 {
            pat.record_sample(sample("peer-a", i + 1));
        }
        let t = pat
            .timeout_for("peer-a")
            .expect("test: peer-a timeout should be present");
        assert!(t >= pat.config.min_timeout_micros);
    }

    // ── 25. Successive samples do not exceed max ──────────────────────────

    #[test]
    fn test_timeout_never_above_max_over_many_samples() {
        let config = AdaptiveTimeoutConfig {
            max_timeout_micros: 10_000,
            ..Default::default()
        };
        let mut pat = PeerAdaptiveTimeout::new(config.clone());
        for i in 0..50u64 {
            pat.record_sample(sample("peer-a", 1_000_000 * (i + 1)));
        }
        let t = pat
            .timeout_for("peer-a")
            .expect("test: peer-a timeout should exist");
        assert!(
            t <= config.max_timeout_micros,
            "Timeout must not exceed max; got {t}"
        );
    }

    // ── 26. Stats after remove_peer reflect the removal ──────────────────

    #[test]
    fn test_stats_after_remove() {
        let mut pat = default_pat();
        pat.record_sample(sample("peer-a", 5_000));
        pat.record_sample(sample("peer-b", 5_000));
        pat.remove_peer("peer-a");
        let stats = pat.stats();
        assert_eq!(stats.total_peers, 1, "Only one peer should remain");
        // total_samples should not change after removal
        assert_eq!(stats.total_samples, 2, "total_samples is cumulative");
    }

    // ── 27. RTTVAR update order: deviation uses previous SRTT ─────────────

    #[test]
    fn test_rttvar_update_uses_prev_srtt() {
        let mut pat = default_pat();
        pat.record_sample(sample("peer-a", 8_000));
        // After first: srtt = 8000, rttvar = 4000
        pat.record_sample(sample("peer-a", 12_000));
        let est = pat
            .estimates
            .get("peer-a")
            .expect("test: peer-a estimate should exist");
        // |srtt_prev - rtt| = |8000 - 12000| = 4000
        // rttvar = (1-0.25)*4000 + 0.25*4000 = 4000 exactly (no change when diff = rttvar)
        let expected_rttvar = (1.0 - 0.25_f64) * 4_000.0 + 0.25 * (8_000.0 - 12_000.0_f64).abs();
        assert!(
            (est.rttvar_micros - expected_rttvar).abs() < 1.0,
            "RTTVAR mismatch: {} vs {}",
            est.rttvar_micros,
            expected_rttvar
        );
    }
}
