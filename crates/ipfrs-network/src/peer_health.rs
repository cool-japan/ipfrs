//! Peer Health Monitor
//!
//! Tracks health scores for peers based on recent ping successes, message delivery
//! rates, and protocol compliance, with automatic degradation over time.
//!
//! # Examples
//!
//! ```rust
//! use ipfrs_network::peer_health::{PeerHealthMonitor, MonitorConfig, HealthSample};
//!
//! let config = MonitorConfig::default();
//! let mut monitor = PeerHealthMonitor::new(config);
//!
//! let sample = HealthSample {
//!     timestamp_secs: 1_000_000,
//!     ping_rtt_ms: Some(12.5),
//!     messages_delivered: 100,
//!     messages_failed: 0,
//! };
//! monitor.record_sample("peer-1", sample);
//!
//! if let Some(score) = monitor.score("peer-1") {
//!     println!("Health score: {:.2}", score.score);
//! }
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// HealthStatus
// ---------------------------------------------------------------------------

/// The health classification of a peer.
#[derive(Clone, Debug, PartialEq)]
pub enum HealthStatus {
    /// Peer is operating normally (score >= healthy_threshold).
    Healthy,
    /// Peer is experiencing issues (degraded_threshold <= score < healthy_threshold).
    Degraded { reason: String },
    /// Peer is not functioning correctly (score < degraded_threshold).
    Unhealthy { reason: String },
    /// No data has been collected yet.
    Unknown,
}

// ---------------------------------------------------------------------------
// HealthSample
// ---------------------------------------------------------------------------

/// A single observation recorded for a peer at a point in time.
#[derive(Clone, Debug)]
pub struct HealthSample {
    /// Unix timestamp (seconds) when this sample was recorded.
    pub timestamp_secs: u64,
    /// Round-trip time in milliseconds, or `None` if the ping failed.
    pub ping_rtt_ms: Option<f64>,
    /// Number of messages that were successfully delivered in this window.
    pub messages_delivered: u64,
    /// Number of messages that failed delivery in this window.
    pub messages_failed: u64,
}

impl HealthSample {
    /// Fraction of messages that were successfully delivered.
    ///
    /// Returns `delivered / (delivered + failed).max(1)`, so it is always in
    /// the range `[0.0, 1.0]`.
    pub fn delivery_rate(&self) -> f64 {
        let total = (self.messages_delivered + self.messages_failed).max(1);
        self.messages_delivered as f64 / total as f64
    }

    /// Returns `true` when the ping completed successfully (RTT is known).
    pub fn ping_ok(&self) -> bool {
        self.ping_rtt_ms.is_some()
    }
}

// ---------------------------------------------------------------------------
// HealthScore
// ---------------------------------------------------------------------------

/// Derived health information for a peer.
#[derive(Clone, Debug)]
pub struct HealthScore {
    /// Composite health score in `[0.0, 1.0]`; 0.0 = dead, 1.0 = perfect.
    pub score: f64,
    /// Human-readable classification of the current score.
    pub status: HealthStatus,
    /// Total number of samples that have been incorporated.
    pub sample_count: usize,
    /// Unix timestamp (seconds) of the most recent update.
    pub last_updated_secs: u64,
}

impl HealthScore {
    /// Returns `true` when the score can be acted upon: the status is known
    /// *and* at least 3 samples have been recorded.
    pub fn is_actionable(&self) -> bool {
        self.status != HealthStatus::Unknown && self.sample_count >= 3
    }
}

impl Default for HealthScore {
    fn default() -> Self {
        Self {
            score: 0.0,
            status: HealthStatus::Unknown,
            sample_count: 0,
            last_updated_secs: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// MonitorConfig
// ---------------------------------------------------------------------------

/// Configuration for [`PeerHealthMonitor`].
#[derive(Clone, Debug)]
pub struct MonitorConfig {
    /// Score at or above which a peer is considered `Healthy` (default 0.8).
    pub healthy_threshold: f64,
    /// Score at or above which a peer is considered `Degraded` (default 0.5).
    pub degraded_threshold: f64,
    /// Per-sample decay factor applied when the sample window is not fully
    /// saturated (default 0.95: `score *= 0.95^missing_samples`).
    pub decay_rate: f64,
    /// Fraction of the raw score contributed by ping success (default 0.4).
    pub ping_weight: f64,
    /// Fraction of the raw score contributed by delivery rate (default 0.6).
    pub delivery_weight: f64,
    /// Maximum number of most-recent samples to retain per peer (default 10).
    pub window_samples: usize,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            healthy_threshold: 0.8,
            degraded_threshold: 0.5,
            decay_rate: 0.95,
            ping_weight: 0.4,
            delivery_weight: 0.6,
            window_samples: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// PeerHealthMonitor
// ---------------------------------------------------------------------------

/// Tracks health scores for a set of peers.
///
/// For each peer the monitor maintains a sliding window of [`HealthSample`]s
/// and derives a composite [`HealthScore`] every time a new sample is
/// recorded.
pub struct PeerHealthMonitor {
    /// `peer_id -> (samples, score)`
    peers: HashMap<String, (Vec<HealthSample>, HealthScore)>,
    /// Configuration controlling thresholds, weights, and window size.
    pub config: MonitorConfig,
}

impl PeerHealthMonitor {
    /// Create a new monitor with the supplied configuration.
    pub fn new(config: MonitorConfig) -> Self {
        Self {
            peers: HashMap::new(),
            config,
        }
    }

    // -----------------------------------------------------------------------
    // Mutation
    // -----------------------------------------------------------------------

    /// Record a new health observation for `peer_id`.
    ///
    /// The sample is appended to the peer's sliding window (oldest entry
    /// discarded when the window is full) and the [`HealthScore`] is
    /// recomputed immediately.
    pub fn record_sample(&mut self, peer_id: &str, sample: HealthSample) {
        let timestamp = sample.timestamp_secs;
        let entry = self
            .peers
            .entry(peer_id.to_owned())
            .or_insert_with(|| (Vec::new(), HealthScore::default()));

        // Maintain sliding window.
        entry.0.push(sample);
        if entry.0.len() > self.config.window_samples {
            let excess = entry.0.len() - self.config.window_samples;
            entry.0.drain(..excess);
        }

        let samples = &entry.0;
        let n = samples.len();

        // --- ping component ---------------------------------------------------
        let ping_ok_count = samples.iter().filter(|s| s.ping_ok()).count();
        let ping_component = (ping_ok_count as f64 / n as f64) * self.config.ping_weight;

        // --- delivery component -----------------------------------------------
        let avg_delivery: f64 = samples.iter().map(|s| s.delivery_rate()).sum::<f64>() / n as f64;
        let delivery_component = avg_delivery * self.config.delivery_weight;

        // --- raw score --------------------------------------------------------
        let raw_score = ping_component + delivery_component;

        // --- decay ------------------------------------------------------------
        // Apply decay for each "missing" sample slot so that a small window is
        // penalised relative to a fully-saturated window.
        let missing = self.config.window_samples.saturating_sub(n);
        let decay_factor = self.config.decay_rate.powi(missing as i32);
        let decayed = (raw_score * decay_factor).clamp(0.0, 1.0);

        // --- status -----------------------------------------------------------
        let status = if decayed >= self.config.healthy_threshold {
            HealthStatus::Healthy
        } else if decayed >= self.config.degraded_threshold {
            HealthStatus::Degraded {
                reason: format!(
                    "score {:.3} below healthy threshold {:.3}",
                    decayed, self.config.healthy_threshold
                ),
            }
        } else {
            HealthStatus::Unhealthy {
                reason: format!(
                    "score {:.3} below degraded threshold {:.3}",
                    decayed, self.config.degraded_threshold
                ),
            }
        };

        // --- update -----------------------------------------------------------
        entry.1 = HealthScore {
            score: decayed,
            status,
            sample_count: n,
            last_updated_secs: timestamp,
        };
    }

    /// Remove all data for `peer_id`.  Returns `true` if the peer existed.
    pub fn remove_peer(&mut self, peer_id: &str) -> bool {
        self.peers.remove(peer_id).is_some()
    }

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /// Return the current [`HealthScore`] for `peer_id`, or `None` if unknown.
    pub fn score(&self, peer_id: &str) -> Option<&HealthScore> {
        self.peers.get(peer_id).map(|(_, score)| score)
    }

    /// Return the IDs of all peers currently classified as [`HealthStatus::Healthy`].
    pub fn healthy_peers(&self) -> Vec<&str> {
        self.peers
            .iter()
            .filter_map(|(id, (_, score))| {
                if score.status == HealthStatus::Healthy {
                    Some(id.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Return the IDs of all peers currently classified as [`HealthStatus::Unhealthy`].
    pub fn unhealthy_peers(&self) -> Vec<&str> {
        self.peers
            .iter()
            .filter_map(|(id, (_, score))| {
                if matches!(score.status, HealthStatus::Unhealthy { .. }) {
                    Some(id.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Return the IDs of all peers currently classified as [`HealthStatus::Degraded`].
    pub fn degraded_peers(&self) -> Vec<&str> {
        self.peers
            .iter()
            .filter_map(|(id, (_, score))| {
                if matches!(score.status, HealthStatus::Degraded { .. }) {
                    Some(id.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Return the top-`n` peers by score, highest first.
    ///
    /// Each entry is `(peer_id, score)`.
    pub fn top_peers(&self, n: usize) -> Vec<(&str, f64)> {
        let mut pairs: Vec<(&str, f64)> = self
            .peers
            .iter()
            .map(|(id, (_, score))| (id.as_str(), score.score))
            .collect();

        // Sort descending; use total_cmp for NaN-safety.
        pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
        pairs.truncate(n);
        pairs
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn good_sample(ts: u64) -> HealthSample {
        HealthSample {
            timestamp_secs: ts,
            ping_rtt_ms: Some(10.0),
            messages_delivered: 100,
            messages_failed: 0,
        }
    }

    fn bad_sample(ts: u64) -> HealthSample {
        HealthSample {
            timestamp_secs: ts,
            ping_rtt_ms: None,
            messages_delivered: 0,
            messages_failed: 100,
        }
    }

    fn mixed_sample(ts: u64) -> HealthSample {
        HealthSample {
            timestamp_secs: ts,
            ping_rtt_ms: Some(50.0),
            messages_delivered: 50,
            messages_failed: 50,
        }
    }

    // ------------------------------------------------------------------
    // 1. new() produces an empty monitor
    // ------------------------------------------------------------------
    #[test]
    fn test_new_empty() {
        let monitor = PeerHealthMonitor::new(MonitorConfig::default());
        assert!(monitor.peers.is_empty());
    }

    // ------------------------------------------------------------------
    // 2. record_sample creates a peer entry
    // ------------------------------------------------------------------
    #[test]
    fn test_record_sample_creates_entry() {
        let mut monitor = PeerHealthMonitor::new(MonitorConfig::default());
        monitor.record_sample("peer-A", good_sample(1000));
        assert!(monitor.score("peer-A").is_some());
    }

    // ------------------------------------------------------------------
    // 3. delivery_rate: all delivered → 1.0
    // ------------------------------------------------------------------
    #[test]
    fn test_delivery_rate_all_delivered() {
        let s = HealthSample {
            timestamp_secs: 0,
            ping_rtt_ms: None,
            messages_delivered: 42,
            messages_failed: 0,
        };
        assert!((s.delivery_rate() - 1.0).abs() < f64::EPSILON);
    }

    // ------------------------------------------------------------------
    // 4. delivery_rate: all failed → 0.0
    // ------------------------------------------------------------------
    #[test]
    fn test_delivery_rate_all_failed() {
        let s = HealthSample {
            timestamp_secs: 0,
            ping_rtt_ms: None,
            messages_delivered: 0,
            messages_failed: 99,
        };
        assert!(s.delivery_rate() < f64::EPSILON);
    }

    // ------------------------------------------------------------------
    // 5. ping_ok: Some vs None
    // ------------------------------------------------------------------
    #[test]
    fn test_ping_ok() {
        let ok = HealthSample {
            timestamp_secs: 0,
            ping_rtt_ms: Some(5.0),
            messages_delivered: 0,
            messages_failed: 0,
        };
        let fail = HealthSample {
            timestamp_secs: 0,
            ping_rtt_ms: None,
            messages_delivered: 0,
            messages_failed: 0,
        };
        assert!(ok.ping_ok());
        assert!(!fail.ping_ok());
    }

    // ------------------------------------------------------------------
    // 6. record_sample: healthy after enough good samples
    // ------------------------------------------------------------------
    #[test]
    fn test_record_sample_healthy() {
        let config = MonitorConfig {
            window_samples: 5,
            ..Default::default()
        };
        let mut monitor = PeerHealthMonitor::new(config);
        for i in 0..5u64 {
            monitor.record_sample("peer-X", good_sample(i));
        }
        let score = monitor.score("peer-X").expect("entry must exist");
        assert_eq!(score.status, HealthStatus::Healthy);
        assert!(score.score >= 0.8);
    }

    // ------------------------------------------------------------------
    // 7. record_sample: unhealthy after enough bad samples
    // ------------------------------------------------------------------
    #[test]
    fn test_record_sample_unhealthy() {
        let config = MonitorConfig {
            window_samples: 5,
            ..Default::default()
        };
        let mut monitor = PeerHealthMonitor::new(config);
        for i in 0..5u64 {
            monitor.record_sample("peer-Y", bad_sample(i));
        }
        let score = monitor.score("peer-Y").expect("entry must exist");
        assert!(matches!(score.status, HealthStatus::Unhealthy { .. }));
        assert!(score.score < 0.5);
    }

    // ------------------------------------------------------------------
    // 8. record_sample: degraded with mixed samples
    // ------------------------------------------------------------------
    #[test]
    fn test_record_sample_degraded() {
        let config = MonitorConfig {
            window_samples: 10,
            ..Default::default()
        };
        let mut monitor = PeerHealthMonitor::new(config);
        // 5 samples with 50% ping success and 50% delivery → raw ~0.5,
        // but decayed because only 5 of 10 slots filled.
        for i in 0..5u64 {
            monitor.record_sample("peer-Z", mixed_sample(i));
        }
        let score = monitor.score("peer-Z").expect("entry must exist");
        // Score should be somewhere in the degraded range or unhealthy due to decay.
        assert!(
            score.score < 0.8,
            "score should not be healthy: {}",
            score.score
        );
    }

    // ------------------------------------------------------------------
    // 9. score: None for unknown peer
    // ------------------------------------------------------------------
    #[test]
    fn test_score_none_unknown_peer() {
        let monitor = PeerHealthMonitor::new(MonitorConfig::default());
        assert!(monitor.score("no-such-peer").is_none());
    }

    // ------------------------------------------------------------------
    // 10. score: sample_count increments
    // ------------------------------------------------------------------
    #[test]
    fn test_sample_count_increments() {
        let mut monitor = PeerHealthMonitor::new(MonitorConfig::default());
        monitor.record_sample("peer-C", good_sample(1));
        assert_eq!(
            monitor
                .score("peer-C")
                .expect("test: peer-C score should exist")
                .sample_count,
            1
        );
        monitor.record_sample("peer-C", good_sample(2));
        assert_eq!(
            monitor
                .score("peer-C")
                .expect("test: peer-C score should exist")
                .sample_count,
            2
        );
        monitor.record_sample("peer-C", good_sample(3));
        assert_eq!(
            monitor
                .score("peer-C")
                .expect("test: peer-C score should exist")
                .sample_count,
            3
        );
    }

    // ------------------------------------------------------------------
    // 11. healthy_peers filtered correctly
    // ------------------------------------------------------------------
    #[test]
    fn test_healthy_peers_filtered() {
        let config = MonitorConfig {
            window_samples: 5,
            ..Default::default()
        };
        let mut monitor = PeerHealthMonitor::new(config);
        for i in 0..5u64 {
            monitor.record_sample("good-peer", good_sample(i));
            monitor.record_sample("bad-peer", bad_sample(i));
        }
        let healthy = monitor.healthy_peers();
        assert!(healthy.contains(&"good-peer"), "good-peer must be healthy");
        assert!(
            !healthy.contains(&"bad-peer"),
            "bad-peer must not be healthy"
        );
    }

    // ------------------------------------------------------------------
    // 12. unhealthy_peers filtered correctly
    // ------------------------------------------------------------------
    #[test]
    fn test_unhealthy_peers_filtered() {
        let config = MonitorConfig {
            window_samples: 5,
            ..Default::default()
        };
        let mut monitor = PeerHealthMonitor::new(config);
        for i in 0..5u64 {
            monitor.record_sample("good-peer", good_sample(i));
            monitor.record_sample("bad-peer", bad_sample(i));
        }
        let unhealthy = monitor.unhealthy_peers();
        assert!(
            unhealthy.contains(&"bad-peer"),
            "bad-peer must be unhealthy"
        );
        assert!(
            !unhealthy.contains(&"good-peer"),
            "good-peer must not be unhealthy"
        );
    }

    // ------------------------------------------------------------------
    // 13. degraded_peers filtered correctly
    // ------------------------------------------------------------------
    #[test]
    fn test_degraded_peers_filtered() {
        let config = MonitorConfig {
            healthy_threshold: 0.8,
            degraded_threshold: 0.5,
            window_samples: 5,
            decay_rate: 1.0, // no decay so score is purely from samples
            ping_weight: 0.4,
            delivery_weight: 0.6,
        };
        let mut monitor = PeerHealthMonitor::new(config);

        // Construct a sample that produces ~0.65 raw score (no decay):
        //   ping_ok=true  → ping_component = 0.4
        //   delivery_rate = 40/100 = 0.4 → delivery_component = 0.24
        //   raw ≈ 0.64  →  Degraded
        let degraded_sample = HealthSample {
            timestamp_secs: 1,
            ping_rtt_ms: Some(30.0),
            messages_delivered: 40,
            messages_failed: 60,
        };
        for i in 0..5u64 {
            let mut s = degraded_sample.clone();
            s.timestamp_secs = i;
            monitor.record_sample("mid-peer", s);
        }

        let degraded = monitor.degraded_peers();
        assert!(
            degraded.contains(&"mid-peer"),
            "mid-peer should be degraded, got status: {:?}",
            monitor.score("mid-peer").map(|s| &s.status)
        );
    }

    // ------------------------------------------------------------------
    // 14. remove_peer: true when peer existed, false otherwise
    // ------------------------------------------------------------------
    #[test]
    fn test_remove_peer() {
        let mut monitor = PeerHealthMonitor::new(MonitorConfig::default());
        monitor.record_sample("to-remove", good_sample(1));
        assert!(monitor.remove_peer("to-remove"));
        assert!(!monitor.remove_peer("to-remove"));
        assert!(!monitor.remove_peer("never-existed"));
    }

    // ------------------------------------------------------------------
    // 15. top_peers sorted descending
    // ------------------------------------------------------------------
    #[test]
    fn test_top_peers_sorted_descending() {
        let config = MonitorConfig {
            window_samples: 5,
            ..Default::default()
        };
        let mut monitor = PeerHealthMonitor::new(config);
        for i in 0..5u64 {
            monitor.record_sample("alpha", good_sample(i));
            monitor.record_sample("beta", bad_sample(i));
        }
        let top = monitor.top_peers(2);
        assert_eq!(top.len(), 2);
        // First entry must have a higher or equal score than the second.
        assert!(
            top[0].1 >= top[1].1,
            "expected descending order, got {:?}",
            top
        );
        assert_eq!(top[0].0, "alpha", "alpha should rank first");
    }

    // ------------------------------------------------------------------
    // 16. is_actionable: false when sample_count < 3
    // ------------------------------------------------------------------
    #[test]
    fn test_is_actionable_false_below_three_samples() {
        let config = MonitorConfig {
            window_samples: 5,
            ..Default::default()
        };
        let mut monitor = PeerHealthMonitor::new(config);
        monitor.record_sample("peer-act", good_sample(1));
        assert!(!monitor
            .score("peer-act")
            .expect("test: peer-act score should exist")
            .is_actionable());
        monitor.record_sample("peer-act", good_sample(2));
        assert!(!monitor
            .score("peer-act")
            .expect("test: peer-act score should exist")
            .is_actionable());
        monitor.record_sample("peer-act", good_sample(3));
        // Now sample_count == 3 and status should be non-Unknown.
        assert!(monitor
            .score("peer-act")
            .expect("test: peer-act score should exist")
            .is_actionable());
    }

    // ------------------------------------------------------------------
    // 17. window_samples cap: keeps only last N
    // ------------------------------------------------------------------
    #[test]
    fn test_window_samples_cap() {
        let config = MonitorConfig {
            window_samples: 3,
            ..Default::default()
        };
        let mut monitor = PeerHealthMonitor::new(config);
        // Insert 5 samples; the window should hold at most 3.
        for i in 0..5u64 {
            monitor.record_sample("peer-win", good_sample(i));
        }
        let score = monitor
            .score("peer-win")
            .expect("test: peer-win score should exist");
        assert_eq!(
            score.sample_count, 3,
            "sample_count should be capped at window_samples"
        );
    }
}
