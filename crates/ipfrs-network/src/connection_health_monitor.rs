//! Connection Health Monitor
//!
//! Tracks connection quality metrics per peer with health scoring, anomaly
//! detection, and alerting. Each peer maintains a sliding window of the last
//! 50 samples per metric type. A composite health score in `[0.0, 1.0]` is
//! computed from latency, bandwidth, packet loss, jitter, and error rate.
//! Threshold crossings trigger `HealthAlert` values that are returned from
//! `record()` and stored internally.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_network::connection_health_monitor::{
//!     AlertThresholds, ChmHealthMetric, ChmHealthSample, ConnectionHealthMonitor,
//! };
//!
//! let thresholds = AlertThresholds::default();
//! let mut monitor = ConnectionHealthMonitor::new(thresholds, 1_000);
//!
//! let sample = ChmHealthSample {
//!     peer_id: "peer-1".to_string(),
//!     metric: ChmHealthMetric::Latency(120.0),
//!     timestamp: 1_000_000,
//! };
//! let alerts = monitor.record(sample);
//! assert!(alerts.is_empty());
//!
//! let score = monitor.health_score("peer-1");
//! assert!(score.is_some());
//! ```

use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of samples retained per metric type for each peer.
const WINDOW_SIZE: usize = 50;

// ---------------------------------------------------------------------------
// HealthMetric
// ---------------------------------------------------------------------------

/// A single connection quality observation.
///
/// All values are non-negative. `PacketLoss` and `ErrorRate` are in `[0.0, 1.0]`.
#[derive(Clone, Debug, PartialEq)]
pub enum ChmHealthMetric {
    /// Round-trip latency in milliseconds.
    Latency(f64),
    /// Fraction of lost packets in `[0.0, 1.0]`.
    PacketLoss(f64),
    /// Available bandwidth in bits-per-second.
    Bandwidth(f64),
    /// Packet arrival jitter in milliseconds.
    Jitter(f64),
    /// Fraction of requests that resulted in an error in `[0.0, 1.0]`.
    ErrorRate(f64),
}

impl ChmHealthMetric {
    /// Return the inner `f64` value regardless of variant.
    pub fn value(&self) -> f64 {
        match self {
            Self::Latency(v)
            | Self::PacketLoss(v)
            | Self::Bandwidth(v)
            | Self::Jitter(v)
            | Self::ErrorRate(v) => *v,
        }
    }

    /// Short string tag used for categorisation.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Latency(_) => "latency",
            Self::PacketLoss(_) => "packet_loss",
            Self::Bandwidth(_) => "bandwidth",
            Self::Jitter(_) => "jitter",
            Self::ErrorRate(_) => "error_rate",
        }
    }
}

// ---------------------------------------------------------------------------
// HealthSample
// ---------------------------------------------------------------------------

/// A single observation recorded for a specific peer.
#[derive(Clone, Debug)]
pub struct ChmHealthSample {
    /// Identifier of the peer being observed.
    pub peer_id: String,
    /// The metric value captured in this observation.
    pub metric: ChmHealthMetric,
    /// Unix timestamp in milliseconds when the sample was captured.
    pub timestamp: u64,
}

// ---------------------------------------------------------------------------
// AlertSeverity
// ---------------------------------------------------------------------------

/// Severity level of a `ChmHealthAlert`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ChmAlertSeverity {
    /// Informational — within normal parameters.
    Info,
    /// Warning — 80 % of the critical threshold.
    Warning,
    /// Critical — at or beyond the threshold limit.
    Critical,
}

// ---------------------------------------------------------------------------
// HealthAlert
// ---------------------------------------------------------------------------

/// An alert generated when a metric crosses a threshold.
#[derive(Clone, Debug)]
pub struct ChmHealthAlert {
    /// Peer for which the alert was raised.
    pub peer_id: String,
    /// The metric that triggered the alert.
    pub metric: ChmHealthMetric,
    /// Severity classification.
    pub severity: ChmAlertSeverity,
    /// The observed value that crossed the threshold.
    pub value: f64,
    /// The threshold that was crossed.
    pub threshold: f64,
    /// Unix timestamp in milliseconds when the alert was generated.
    pub timestamp: u64,
}

// ---------------------------------------------------------------------------
// AlertThresholds
// ---------------------------------------------------------------------------

/// Per-metric thresholds used to generate `ChmHealthAlert` values.
///
/// Warning alerts fire at 80 % of the configured threshold; critical alerts
/// fire at 100 %.
#[derive(Clone, Debug)]
pub struct AlertThresholds {
    /// Maximum acceptable latency in milliseconds (default: 500 ms).
    pub max_latency_ms: f64,
    /// Maximum acceptable packet-loss fraction (default: 0.05).
    pub max_packet_loss: f64,
    /// Minimum acceptable bandwidth in bits-per-second (default: 100 000 bps).
    pub min_bandwidth_bps: f64,
    /// Maximum acceptable jitter in milliseconds (default: 100 ms).
    pub max_jitter_ms: f64,
    /// Maximum acceptable error-rate fraction (default: 0.10).
    pub max_error_rate: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_latency_ms: 500.0,
            max_packet_loss: 0.05,
            min_bandwidth_bps: 100_000.0,
            max_jitter_ms: 100.0,
            max_error_rate: 0.1,
        }
    }
}

impl AlertThresholds {
    /// Warning level — 80 % of the critical threshold.
    const WARNING_FACTOR: f64 = 0.8;

    /// Returns the warning threshold for latency.
    pub fn warning_latency_ms(&self) -> f64 {
        self.max_latency_ms * Self::WARNING_FACTOR
    }

    /// Returns the warning threshold for packet loss.
    pub fn warning_packet_loss(&self) -> f64 {
        self.max_packet_loss * Self::WARNING_FACTOR
    }

    /// Returns the warning threshold for bandwidth (inverted — lower is worse).
    pub fn warning_bandwidth_bps(&self) -> f64 {
        self.min_bandwidth_bps / Self::WARNING_FACTOR
    }

    /// Returns the warning threshold for jitter.
    pub fn warning_jitter_ms(&self) -> f64 {
        self.max_jitter_ms * Self::WARNING_FACTOR
    }

    /// Returns the warning threshold for error rate.
    pub fn warning_error_rate(&self) -> f64 {
        self.max_error_rate * Self::WARNING_FACTOR
    }
}

// ---------------------------------------------------------------------------
// ConnectionHealth
// ---------------------------------------------------------------------------

/// Per-peer health state maintained by the monitor.
#[derive(Clone, Debug)]
pub struct ConnectionHealth {
    /// Peer identifier.
    pub peer_id: String,
    /// Composite health score in `[0.0, 1.0]`.
    pub health_score: f64,
    /// Sliding window of recent samples (across all metric types).
    pub samples: VecDeque<ChmHealthSample>,
    /// Unix timestamp in milliseconds of the most recent update.
    pub last_updated: u64,
    /// Number of alerts that have been raised for this peer.
    pub alert_count: u32,
}

impl ConnectionHealth {
    fn new(peer_id: String) -> Self {
        Self {
            peer_id,
            health_score: 1.0,
            samples: VecDeque::new(),
            last_updated: 0,
            alert_count: 0,
        }
    }

    /// Collect the values for a particular metric kind from the sample window.
    fn values_for_kind(&self, kind: &str) -> Vec<f64> {
        self.samples
            .iter()
            .filter(|s| s.metric.kind() == kind)
            .map(|s| s.metric.value())
            .collect()
    }

    /// Arithmetic mean of a slice, or `None` if empty.
    fn mean(values: &[f64]) -> Option<f64> {
        if values.is_empty() {
            return None;
        }
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

// ---------------------------------------------------------------------------
// MonitorStats
// ---------------------------------------------------------------------------

/// Aggregate statistics snapshot for the whole monitor.
#[derive(Clone, Debug, Default)]
pub struct ChmMonitorStats {
    /// Total number of peers currently tracked.
    pub total_peers: usize,
    /// Number of peers with `health_score >= 0.8`.
    pub healthy_peers: usize,
    /// Number of peers with `0.5 <= health_score < 0.8`.
    pub degraded_peers: usize,
    /// Number of peers with `health_score < 0.5`.
    pub critical_peers: usize,
    /// Total number of alerts stored in the internal deque.
    pub total_alerts: usize,
    /// Average health score across all tracked peers (0.0 if none).
    pub avg_health_score: f64,
}

// ---------------------------------------------------------------------------
// ConnectionHealthMonitor
// ---------------------------------------------------------------------------

/// Tracks connection quality metrics per peer with health scoring, anomaly
/// detection, and alerting.
pub struct ConnectionHealthMonitor {
    /// Threshold configuration for generating alerts.
    pub thresholds: AlertThresholds,
    /// Per-peer health records.
    pub connections: HashMap<String, ConnectionHealth>,
    /// Ring buffer of recent alerts.
    pub alerts: VecDeque<ChmHealthAlert>,
    /// Maximum number of alerts to retain.
    pub max_alerts: usize,
}

impl ConnectionHealthMonitor {
    /// Create a new monitor with the given thresholds and alert capacity.
    pub fn new(thresholds: AlertThresholds, max_alerts: usize) -> Self {
        Self {
            thresholds,
            connections: HashMap::new(),
            alerts: VecDeque::new(),
            max_alerts: max_alerts.max(1),
        }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Record a new sample for a peer, recompute its health score, and return
    /// any new alerts that were triggered.
    pub fn record(&mut self, sample: ChmHealthSample) -> Vec<ChmHealthAlert> {
        let peer_id = sample.peer_id.clone();
        let value = sample.metric.value();
        let now = sample.timestamp;
        let metric_clone = sample.metric.clone();

        // Insert/update connection record.
        let conn = self
            .connections
            .entry(peer_id.clone())
            .or_insert_with(|| ConnectionHealth::new(peer_id.clone()));

        // Add sample; evict oldest sample of the same metric kind if window full.
        let kind = sample.metric.kind();
        let same_kind_count = conn
            .samples
            .iter()
            .filter(|s| s.metric.kind() == kind)
            .count();
        if same_kind_count >= WINDOW_SIZE {
            // Remove the oldest sample of this kind.
            if let Some(pos) = conn.samples.iter().position(|s| s.metric.kind() == kind) {
                conn.samples.remove(pos);
            }
        }
        conn.samples.push_back(sample);
        conn.last_updated = now;

        // Recompute health score.
        let new_score = Self::compute_health_score(conn);
        conn.health_score = new_score;

        // Check thresholds.
        let new_alerts = self.check_thresholds(&peer_id, &metric_clone, value, now);

        // Push alerts and update alert_count.
        if let Some(conn) = self.connections.get_mut(&peer_id) {
            conn.alert_count = conn.alert_count.saturating_add(new_alerts.len() as u32);
        }
        for alert in &new_alerts {
            self.alerts.push_back(alert.clone());
            while self.alerts.len() > self.max_alerts {
                self.alerts.pop_front();
            }
        }

        new_alerts
    }

    /// Return the current health score for a peer, if tracked.
    pub fn health_score(&self, peer_id: &str) -> Option<f64> {
        self.connections.get(peer_id).map(|c| c.health_score)
    }

    /// Compute the composite health score from a peer's sample window.
    ///
    /// Components and weights:
    /// - Latency   (0.30): `exp(-avg_latency / 1000.0)`
    /// - Bandwidth (0.20): `(avg_bandwidth / 1_000_000.0).min(1.0)`
    /// - Availability (0.20): `1.0 - avg_packet_loss`
    /// - Jitter    (0.15): `exp(-avg_jitter / 200.0)`
    /// - Reliability (0.15): `1.0 - avg_error_rate`
    ///
    /// Missing metrics default to 0.5.
    pub fn compute_health_score(conn: &ConnectionHealth) -> f64 {
        let latency_score = ConnectionHealth::mean(&conn.values_for_kind("latency"))
            .map(|v| (-v / 1_000.0_f64).exp())
            .unwrap_or(0.5);

        let bandwidth_score = ConnectionHealth::mean(&conn.values_for_kind("bandwidth"))
            .map(|v| (v / 1_000_000.0).min(1.0))
            .unwrap_or(0.5);

        let availability_score = ConnectionHealth::mean(&conn.values_for_kind("packet_loss"))
            .map(|v| 1.0 - v)
            .unwrap_or(0.5);

        let jitter_score = ConnectionHealth::mean(&conn.values_for_kind("jitter"))
            .map(|v| (-v / 200.0_f64).exp())
            .unwrap_or(0.5);

        let reliability_score = ConnectionHealth::mean(&conn.values_for_kind("error_rate"))
            .map(|v| 1.0 - v)
            .unwrap_or(0.5);

        let score = 0.30 * latency_score
            + 0.20 * bandwidth_score
            + 0.20 * availability_score
            + 0.15 * jitter_score
            + 0.15 * reliability_score;

        score.clamp(0.0, 1.0)
    }

    /// Check whether `value` crosses warning or critical thresholds for the
    /// given metric, and return any generated alerts.
    pub fn check_thresholds(
        &self,
        peer_id: &str,
        metric: &ChmHealthMetric,
        value: f64,
        now: u64,
    ) -> Vec<ChmHealthAlert> {
        let mut alerts = Vec::new();

        match metric {
            ChmHealthMetric::Latency(_) => {
                let critical = self.thresholds.max_latency_ms;
                let warning = self.thresholds.warning_latency_ms();
                if value >= critical {
                    alerts.push(self.make_alert(
                        peer_id,
                        metric.clone(),
                        ChmAlertSeverity::Critical,
                        value,
                        critical,
                        now,
                    ));
                } else if value >= warning {
                    alerts.push(self.make_alert(
                        peer_id,
                        metric.clone(),
                        ChmAlertSeverity::Warning,
                        value,
                        critical,
                        now,
                    ));
                }
            }
            ChmHealthMetric::PacketLoss(_) => {
                let critical = self.thresholds.max_packet_loss;
                let warning = self.thresholds.warning_packet_loss();
                if value >= critical {
                    alerts.push(self.make_alert(
                        peer_id,
                        metric.clone(),
                        ChmAlertSeverity::Critical,
                        value,
                        critical,
                        now,
                    ));
                } else if value >= warning {
                    alerts.push(self.make_alert(
                        peer_id,
                        metric.clone(),
                        ChmAlertSeverity::Warning,
                        value,
                        critical,
                        now,
                    ));
                }
            }
            ChmHealthMetric::Bandwidth(_) => {
                // Low bandwidth is bad — invert the comparison.
                let critical = self.thresholds.min_bandwidth_bps;
                let warning = self.thresholds.warning_bandwidth_bps();
                if value <= critical {
                    alerts.push(self.make_alert(
                        peer_id,
                        metric.clone(),
                        ChmAlertSeverity::Critical,
                        value,
                        critical,
                        now,
                    ));
                } else if value <= warning {
                    alerts.push(self.make_alert(
                        peer_id,
                        metric.clone(),
                        ChmAlertSeverity::Warning,
                        value,
                        critical,
                        now,
                    ));
                }
            }
            ChmHealthMetric::Jitter(_) => {
                let critical = self.thresholds.max_jitter_ms;
                let warning = self.thresholds.warning_jitter_ms();
                if value >= critical {
                    alerts.push(self.make_alert(
                        peer_id,
                        metric.clone(),
                        ChmAlertSeverity::Critical,
                        value,
                        critical,
                        now,
                    ));
                } else if value >= warning {
                    alerts.push(self.make_alert(
                        peer_id,
                        metric.clone(),
                        ChmAlertSeverity::Warning,
                        value,
                        critical,
                        now,
                    ));
                }
            }
            ChmHealthMetric::ErrorRate(_) => {
                let critical = self.thresholds.max_error_rate;
                let warning = self.thresholds.warning_error_rate();
                if value >= critical {
                    alerts.push(self.make_alert(
                        peer_id,
                        metric.clone(),
                        ChmAlertSeverity::Critical,
                        value,
                        critical,
                        now,
                    ));
                } else if value >= warning {
                    alerts.push(self.make_alert(
                        peer_id,
                        metric.clone(),
                        ChmAlertSeverity::Warning,
                        value,
                        critical,
                        now,
                    ));
                }
            }
        }

        alerts
    }

    /// Return the IDs of peers whose health score is below `min_score`,
    /// sorted lexicographically.
    pub fn peers_below_threshold(&self, min_score: f64) -> Vec<&str> {
        let mut result: Vec<&str> = self
            .connections
            .iter()
            .filter(|(_, c)| c.health_score < min_score)
            .map(|(id, _)| id.as_str())
            .collect();
        result.sort_unstable();
        result
    }

    /// Return all alerts with timestamp >= `since`.
    pub fn recent_alerts(&self, since: u64) -> Vec<&ChmHealthAlert> {
        self.alerts
            .iter()
            .filter(|a| a.timestamp >= since)
            .collect()
    }

    /// Remove all health data for a peer. Returns `true` if the peer existed.
    pub fn clear_peer(&mut self, peer_id: &str) -> bool {
        self.connections.remove(peer_id).is_some()
    }

    /// Evict peers that have not been updated within `max_age_ms` milliseconds.
    ///
    /// Returns the number of peers evicted.
    pub fn evict_stale(&mut self, max_age_ms: u64, now: u64) -> usize {
        let cutoff = now.saturating_sub(max_age_ms);
        let before = self.connections.len();
        self.connections
            .retain(|_, c| c.last_updated >= cutoff || c.last_updated == 0);
        before - self.connections.len()
    }

    /// Return the top `n` peers by health score in descending order.
    pub fn top_peers(&self, n: usize) -> Vec<(&str, f64)> {
        let mut peers: Vec<(&str, f64)> = self
            .connections
            .iter()
            .map(|(id, c)| (id.as_str(), c.health_score))
            .collect();
        // Sort descending by score, then ascending by peer_id for determinism.
        peers.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(b.0))
        });
        peers.truncate(n);
        peers
    }

    /// Compute aggregate statistics across all tracked peers.
    pub fn stats(&self) -> ChmMonitorStats {
        let total_peers = self.connections.len();
        let mut healthy_peers = 0usize;
        let mut degraded_peers = 0usize;
        let mut critical_peers = 0usize;
        let mut score_sum = 0.0f64;

        for conn in self.connections.values() {
            let s = conn.health_score;
            score_sum += s;
            if s >= 0.8 {
                healthy_peers += 1;
            } else if s >= 0.5 {
                degraded_peers += 1;
            } else {
                critical_peers += 1;
            }
        }

        let avg_health_score = if total_peers == 0 {
            0.0
        } else {
            score_sum / total_peers as f64
        };

        ChmMonitorStats {
            total_peers,
            healthy_peers,
            degraded_peers,
            critical_peers,
            total_alerts: self.alerts.len(),
            avg_health_score,
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn make_alert(
        &self,
        peer_id: &str,
        metric: ChmHealthMetric,
        severity: ChmAlertSeverity,
        value: f64,
        threshold: f64,
        timestamp: u64,
    ) -> ChmHealthAlert {
        ChmHealthAlert {
            peer_id: peer_id.to_string(),
            metric,
            severity,
            value,
            threshold,
            timestamp,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{
        AlertThresholds, ChmAlertSeverity, ChmHealthAlert, ChmHealthMetric, ChmHealthSample,
        ChmMonitorStats, ConnectionHealth, ConnectionHealthMonitor, WINDOW_SIZE,
    };

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn default_monitor() -> ConnectionHealthMonitor {
        ConnectionHealthMonitor::new(AlertThresholds::default(), 1_000)
    }

    fn latency_sample(peer_id: &str, ms: f64, ts: u64) -> ChmHealthSample {
        ChmHealthSample {
            peer_id: peer_id.to_string(),
            metric: ChmHealthMetric::Latency(ms),
            timestamp: ts,
        }
    }

    fn bandwidth_sample(peer_id: &str, bps: f64, ts: u64) -> ChmHealthSample {
        ChmHealthSample {
            peer_id: peer_id.to_string(),
            metric: ChmHealthMetric::Bandwidth(bps),
            timestamp: ts,
        }
    }

    fn packet_loss_sample(peer_id: &str, loss: f64, ts: u64) -> ChmHealthSample {
        ChmHealthSample {
            peer_id: peer_id.to_string(),
            metric: ChmHealthMetric::PacketLoss(loss),
            timestamp: ts,
        }
    }

    fn jitter_sample(peer_id: &str, ms: f64, ts: u64) -> ChmHealthSample {
        ChmHealthSample {
            peer_id: peer_id.to_string(),
            metric: ChmHealthMetric::Jitter(ms),
            timestamp: ts,
        }
    }

    fn error_rate_sample(peer_id: &str, rate: f64, ts: u64) -> ChmHealthSample {
        ChmHealthSample {
            peer_id: peer_id.to_string(),
            metric: ChmHealthMetric::ErrorRate(rate),
            timestamp: ts,
        }
    }

    // -----------------------------------------------------------------------
    // 1. Basic record / health_score
    // -----------------------------------------------------------------------

    #[test]
    fn test_record_new_peer_creates_entry() {
        let mut m = default_monitor();
        m.record(latency_sample("p1", 50.0, 1000));
        assert!(m.health_score("p1").is_some());
    }

    #[test]
    fn test_health_score_unknown_peer_returns_none() {
        let m = default_monitor();
        assert!(m.health_score("nobody").is_none());
    }

    #[test]
    fn test_record_returns_empty_alerts_for_good_latency() {
        let mut m = default_monitor();
        let alerts = m.record(latency_sample("p1", 50.0, 1000));
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_record_multiple_samples_updates_score() {
        let mut m = default_monitor();
        m.record(latency_sample("p1", 100.0, 1000));
        let score1 = m.health_score("p1").unwrap_or(0.0);
        m.record(latency_sample("p1", 200.0, 2000));
        let score2 = m.health_score("p1").unwrap_or(0.0);
        // Higher avg latency → lower latency score → lower composite.
        assert!(score2 < score1 || (score2 - score1).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 2. Health score formula
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_health_score_all_default_missing() {
        let conn = ConnectionHealth::new("p".to_string());
        let score = ConnectionHealthMonitor::compute_health_score(&conn);
        // All metrics missing → all default to 0.5 → composite = 0.5.
        assert!((score - 0.5).abs() < 1e-9, "expected 0.5 got {}", score);
    }

    #[test]
    fn test_compute_health_score_perfect_latency() {
        let mut conn = ConnectionHealth::new("p".to_string());
        // Very low latency → exp(0) ≈ 1.0
        conn.samples.push_back(ChmHealthSample {
            peer_id: "p".to_string(),
            metric: ChmHealthMetric::Latency(0.0),
            timestamp: 1,
        });
        let score = ConnectionHealthMonitor::compute_health_score(&conn);
        // latency_score=1.0, rest default 0.5
        let expected = 0.30 * 1.0 + 0.20 * 0.5 + 0.20 * 0.5 + 0.15 * 0.5 + 0.15 * 0.5;
        assert!((score - expected).abs() < 1e-9);
    }

    #[test]
    fn test_compute_health_score_zero_bandwidth() {
        let mut conn = ConnectionHealth::new("p".to_string());
        conn.samples.push_back(ChmHealthSample {
            peer_id: "p".to_string(),
            metric: ChmHealthMetric::Bandwidth(0.0),
            timestamp: 1,
        });
        let score = ConnectionHealthMonitor::compute_health_score(&conn);
        // bandwidth_score = 0.0; rest default 0.5
        let expected = 0.30 * 0.5 + 0.20 * 0.0 + 0.20 * 0.5 + 0.15 * 0.5 + 0.15 * 0.5;
        assert!((score - expected).abs() < 1e-9);
    }

    #[test]
    fn test_compute_health_score_full_packet_loss() {
        let mut conn = ConnectionHealth::new("p".to_string());
        conn.samples.push_back(ChmHealthSample {
            peer_id: "p".to_string(),
            metric: ChmHealthMetric::PacketLoss(1.0),
            timestamp: 1,
        });
        let score = ConnectionHealthMonitor::compute_health_score(&conn);
        // availability_score = 0.0; rest default 0.5
        let expected = 0.30 * 0.5 + 0.20 * 0.5 + 0.20 * 0.0 + 0.15 * 0.5 + 0.15 * 0.5;
        assert!((score - expected).abs() < 1e-9);
    }

    #[test]
    fn test_compute_health_score_all_perfect() {
        let mut conn = ConnectionHealth::new("p".to_string());
        conn.samples.push_back(ChmHealthSample {
            peer_id: "p".to_string(),
            metric: ChmHealthMetric::Latency(0.0),
            timestamp: 1,
        });
        conn.samples.push_back(ChmHealthSample {
            peer_id: "p".to_string(),
            metric: ChmHealthMetric::Bandwidth(1_000_000.0),
            timestamp: 2,
        });
        conn.samples.push_back(ChmHealthSample {
            peer_id: "p".to_string(),
            metric: ChmHealthMetric::PacketLoss(0.0),
            timestamp: 3,
        });
        conn.samples.push_back(ChmHealthSample {
            peer_id: "p".to_string(),
            metric: ChmHealthMetric::Jitter(0.0),
            timestamp: 4,
        });
        conn.samples.push_back(ChmHealthSample {
            peer_id: "p".to_string(),
            metric: ChmHealthMetric::ErrorRate(0.0),
            timestamp: 5,
        });
        let score = ConnectionHealthMonitor::compute_health_score(&conn);
        assert!((score - 1.0).abs() < 1e-9, "expected 1.0 got {score}");
    }

    #[test]
    fn test_health_score_clamped_to_one() {
        let mut conn = ConnectionHealth::new("p".to_string());
        // Latency of 0 gives exp(0)=1.0; bandwidth 10Mbps gives 1.0; rest 0 → score=1.0.
        conn.samples.push_back(ChmHealthSample {
            peer_id: "p".to_string(),
            metric: ChmHealthMetric::Latency(0.0),
            timestamp: 1,
        });
        conn.samples.push_back(ChmHealthSample {
            peer_id: "p".to_string(),
            metric: ChmHealthMetric::Bandwidth(10_000_000.0),
            timestamp: 2,
        });
        conn.samples.push_back(ChmHealthSample {
            peer_id: "p".to_string(),
            metric: ChmHealthMetric::PacketLoss(0.0),
            timestamp: 3,
        });
        conn.samples.push_back(ChmHealthSample {
            peer_id: "p".to_string(),
            metric: ChmHealthMetric::Jitter(0.0),
            timestamp: 4,
        });
        conn.samples.push_back(ChmHealthSample {
            peer_id: "p".to_string(),
            metric: ChmHealthMetric::ErrorRate(0.0),
            timestamp: 5,
        });
        let score = ConnectionHealthMonitor::compute_health_score(&conn);
        assert!(score <= 1.0);
    }

    // -----------------------------------------------------------------------
    // 3. Alert generation — latency
    // -----------------------------------------------------------------------

    #[test]
    fn test_latency_warning_alert() {
        let mut m = default_monitor();
        // 80 % of 500 ms = 400 ms
        let alerts = m.record(latency_sample("p1", 400.0, 1000));
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, ChmAlertSeverity::Warning);
    }

    #[test]
    fn test_latency_critical_alert() {
        let mut m = default_monitor();
        let alerts = m.record(latency_sample("p1", 500.0, 1000));
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, ChmAlertSeverity::Critical);
    }

    #[test]
    fn test_latency_above_critical_is_critical() {
        let mut m = default_monitor();
        let alerts = m.record(latency_sample("p1", 800.0, 1000));
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, ChmAlertSeverity::Critical);
    }

    #[test]
    fn test_latency_below_warning_no_alert() {
        let mut m = default_monitor();
        let alerts = m.record(latency_sample("p1", 100.0, 1000));
        assert!(alerts.is_empty());
    }

    // -----------------------------------------------------------------------
    // 4. Alert generation — packet loss
    // -----------------------------------------------------------------------

    #[test]
    fn test_packet_loss_warning_alert() {
        let mut m = default_monitor();
        // Warning threshold = 80% of 0.05 = 0.04 (computed as 0.05 * 0.8 ≈ 0.04000000000000001).
        // Use 0.045 to safely land above the floating-point result without reaching critical.
        let alerts = m.record(packet_loss_sample("p1", 0.045, 1000));
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, ChmAlertSeverity::Warning);
    }

    #[test]
    fn test_packet_loss_critical_alert() {
        let mut m = default_monitor();
        let alerts = m.record(packet_loss_sample("p1", 0.05, 1000));
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, ChmAlertSeverity::Critical);
    }

    // -----------------------------------------------------------------------
    // 5. Alert generation — bandwidth (inverted threshold)
    // -----------------------------------------------------------------------

    #[test]
    fn test_bandwidth_critical_alert_when_too_low() {
        let mut m = default_monitor();
        // Below min_bandwidth_bps (100_000)
        let alerts = m.record(bandwidth_sample("p1", 50_000.0, 1000));
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, ChmAlertSeverity::Critical);
    }

    #[test]
    fn test_bandwidth_warning_alert_between_thresholds() {
        let mut m = default_monitor();
        // warning = 100_000 / 0.8 = 125_000 bps; value between 100_000 and 125_000
        let alerts = m.record(bandwidth_sample("p1", 110_000.0, 1000));
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, ChmAlertSeverity::Warning);
    }

    #[test]
    fn test_bandwidth_no_alert_when_high() {
        let mut m = default_monitor();
        let alerts = m.record(bandwidth_sample("p1", 1_000_000.0, 1000));
        assert!(alerts.is_empty());
    }

    // -----------------------------------------------------------------------
    // 6. Alert generation — jitter
    // -----------------------------------------------------------------------

    #[test]
    fn test_jitter_warning_alert() {
        let mut m = default_monitor();
        // 80 % of 100 ms = 80 ms
        let alerts = m.record(jitter_sample("p1", 80.0, 1000));
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, ChmAlertSeverity::Warning);
    }

    #[test]
    fn test_jitter_critical_alert() {
        let mut m = default_monitor();
        let alerts = m.record(jitter_sample("p1", 100.0, 1000));
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, ChmAlertSeverity::Critical);
    }

    // -----------------------------------------------------------------------
    // 7. Alert generation — error rate
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_rate_warning_alert() {
        let mut m = default_monitor();
        // Warning threshold = 80% of 0.1 = 0.08 (computed as 0.1 * 0.8).
        // Use 0.09 to safely land above the floating-point result without reaching critical.
        let alerts = m.record(error_rate_sample("p1", 0.09, 1000));
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, ChmAlertSeverity::Warning);
    }

    #[test]
    fn test_error_rate_critical_alert() {
        let mut m = default_monitor();
        let alerts = m.record(error_rate_sample("p1", 0.1, 1000));
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, ChmAlertSeverity::Critical);
    }

    // -----------------------------------------------------------------------
    // 8. Alert storage and retrieval
    // -----------------------------------------------------------------------

    #[test]
    fn test_alerts_stored_internally() {
        let mut m = default_monitor();
        m.record(latency_sample("p1", 500.0, 1000));
        assert_eq!(m.alerts.len(), 1);
    }

    #[test]
    fn test_max_alerts_eviction() {
        let mut m = ConnectionHealthMonitor::new(AlertThresholds::default(), 3);
        for i in 0..10u64 {
            m.record(latency_sample("p1", 600.0, i * 1000));
        }
        assert_eq!(m.alerts.len(), 3);
    }

    #[test]
    fn test_recent_alerts_filter_by_timestamp() {
        let mut m = default_monitor();
        m.record(latency_sample("p1", 600.0, 1_000));
        m.record(latency_sample("p1", 600.0, 5_000));
        m.record(latency_sample("p1", 600.0, 10_000));
        let recent = m.recent_alerts(5_000);
        assert_eq!(recent.len(), 2);
    }

    #[test]
    fn test_recent_alerts_returns_all_when_since_zero() {
        let mut m = default_monitor();
        m.record(latency_sample("p1", 600.0, 1_000));
        m.record(latency_sample("p1", 600.0, 2_000));
        let recent = m.recent_alerts(0);
        assert_eq!(recent.len(), 2);
    }

    // -----------------------------------------------------------------------
    // 9. alert_count on ConnectionHealth
    // -----------------------------------------------------------------------

    #[test]
    fn test_alert_count_increments_per_peer() {
        let mut m = default_monitor();
        m.record(latency_sample("p1", 600.0, 1_000));
        m.record(latency_sample("p1", 600.0, 2_000));
        let conn = m.connections.get("p1").expect("peer exists");
        assert_eq!(conn.alert_count, 2);
    }

    // -----------------------------------------------------------------------
    // 10. peers_below_threshold
    // -----------------------------------------------------------------------

    #[test]
    fn test_peers_below_threshold_sorted() {
        let mut m = default_monitor();
        // Insert peers with varying latencies.
        m.record(latency_sample("b_peer", 900.0, 1000));
        m.record(latency_sample("a_peer", 950.0, 1000));
        m.record(latency_sample("c_peer", 10.0, 1000));
        // c_peer should have a good score; a_peer and b_peer poor scores.
        let below = m.peers_below_threshold(0.9);
        // At least a_peer and b_peer should be there; check sort.
        for i in 1..below.len() {
            assert!(below[i - 1] <= below[i]);
        }
    }

    #[test]
    fn test_peers_below_threshold_none_when_all_healthy() {
        let mut m = default_monitor();
        m.record(latency_sample("p1", 1.0, 1000));
        let below = m.peers_below_threshold(0.0);
        assert!(below.is_empty());
    }

    // -----------------------------------------------------------------------
    // 11. clear_peer
    // -----------------------------------------------------------------------

    #[test]
    fn test_clear_peer_removes_existing() {
        let mut m = default_monitor();
        m.record(latency_sample("p1", 50.0, 1000));
        assert!(m.clear_peer("p1"));
        assert!(m.health_score("p1").is_none());
    }

    #[test]
    fn test_clear_peer_returns_false_for_unknown() {
        let mut m = default_monitor();
        assert!(!m.clear_peer("ghost"));
    }

    // -----------------------------------------------------------------------
    // 12. evict_stale
    // -----------------------------------------------------------------------

    #[test]
    fn test_evict_stale_removes_old_peers() {
        let mut m = default_monitor();
        m.record(latency_sample("old", 50.0, 1_000));
        // "new" peer has ts=180_000 which is >= cutoff (200_000 - 50_000 = 150_000).
        m.record(latency_sample("new", 50.0, 180_000));
        let count = m.evict_stale(50_000, 200_000);
        assert_eq!(count, 1);
        assert!(m.health_score("old").is_none());
        assert!(m.health_score("new").is_some());
    }

    #[test]
    fn test_evict_stale_returns_zero_when_nothing_stale() {
        let mut m = default_monitor();
        m.record(latency_sample("p1", 50.0, 190_000));
        let count = m.evict_stale(50_000, 200_000);
        assert_eq!(count, 0);
    }

    // -----------------------------------------------------------------------
    // 13. top_peers
    // -----------------------------------------------------------------------

    #[test]
    fn test_top_peers_returns_n_best() {
        let mut m = default_monitor();
        m.record(bandwidth_sample("p1", 1_000_000.0, 1000));
        m.record(bandwidth_sample("p2", 500_000.0, 1000));
        m.record(bandwidth_sample("p3", 10_000.0, 1000));
        let top = m.top_peers(2);
        assert_eq!(top.len(), 2);
        // Descending order.
        assert!(top[0].1 >= top[1].1);
    }

    #[test]
    fn test_top_peers_handles_n_larger_than_count() {
        let mut m = default_monitor();
        m.record(latency_sample("p1", 50.0, 1000));
        let top = m.top_peers(100);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_top_peers_empty_monitor() {
        let m = default_monitor();
        assert!(m.top_peers(5).is_empty());
    }

    // -----------------------------------------------------------------------
    // 14. stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_default_empty() {
        let m = default_monitor();
        let s = m.stats();
        assert_eq!(s.total_peers, 0);
        assert_eq!(s.avg_health_score, 0.0);
    }

    #[test]
    fn test_stats_counts_healthy_degraded_critical() {
        let mut m = default_monitor();
        // Near-perfect peer → healthy.
        m.record(latency_sample("healthy", 1.0, 1));
        m.record(bandwidth_sample("healthy", 1_000_000.0, 2));
        m.record(packet_loss_sample("healthy", 0.0, 3));
        m.record(jitter_sample("healthy", 0.0, 4));
        m.record(error_rate_sample("healthy", 0.0, 5));

        // High latency peer → critical or degraded.
        m.record(latency_sample("bad", 10_000.0, 1));
        m.record(bandwidth_sample("bad", 1.0, 2));
        m.record(packet_loss_sample("bad", 0.9, 3));
        m.record(jitter_sample("bad", 5_000.0, 4));
        m.record(error_rate_sample("bad", 0.9, 5));

        let s = m.stats();
        assert_eq!(s.total_peers, 2);
        assert_eq!(s.healthy_peers + s.degraded_peers + s.critical_peers, 2);
    }

    #[test]
    fn test_stats_avg_health_score() {
        let mut m = default_monitor();
        // Only latency samples with identical value → scores equal.
        m.record(latency_sample("p1", 1.0, 1));
        m.record(latency_sample("p2", 1.0, 1));
        let s = m.stats();
        let score_p1 = m.health_score("p1").unwrap_or(0.0);
        let score_p2 = m.health_score("p2").unwrap_or(0.0);
        let expected_avg = (score_p1 + score_p2) / 2.0;
        assert!((s.avg_health_score - expected_avg).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 15. Window eviction
    // -----------------------------------------------------------------------

    #[test]
    fn test_window_evicts_oldest_same_metric() {
        let mut m = default_monitor();
        // Fill window with 50 latency samples.
        for i in 0..WINDOW_SIZE {
            m.record(latency_sample("p1", 10.0, i as u64 * 100));
        }
        let conn = m.connections.get("p1").expect("exists");
        let latency_count = conn
            .samples
            .iter()
            .filter(|s| s.metric.kind() == "latency")
            .count();
        assert_eq!(latency_count, WINDOW_SIZE);

        // Add one more latency sample — oldest should be evicted.
        m.record(latency_sample("p1", 10.0, WINDOW_SIZE as u64 * 100));
        let conn = m.connections.get("p1").expect("exists");
        let latency_count_after = conn
            .samples
            .iter()
            .filter(|s| s.metric.kind() == "latency")
            .count();
        assert_eq!(latency_count_after, WINDOW_SIZE);
    }

    #[test]
    fn test_window_different_metrics_independent() {
        let mut m = default_monitor();
        // Fill window with both latency and bandwidth.
        for i in 0..WINDOW_SIZE {
            m.record(latency_sample("p1", 10.0, i as u64 * 100));
            m.record(bandwidth_sample("p1", 1_000_000.0, i as u64 * 100 + 1));
        }
        let conn = m.connections.get("p1").expect("exists");
        let latency_count = conn
            .samples
            .iter()
            .filter(|s| s.metric.kind() == "latency")
            .count();
        let bandwidth_count = conn
            .samples
            .iter()
            .filter(|s| s.metric.kind() == "bandwidth")
            .count();
        assert_eq!(latency_count, WINDOW_SIZE);
        assert_eq!(bandwidth_count, WINDOW_SIZE);
    }

    // -----------------------------------------------------------------------
    // 16. HealthMetric helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_health_metric_value() {
        assert_eq!(ChmHealthMetric::Latency(42.0).value(), 42.0);
        assert_eq!(ChmHealthMetric::PacketLoss(0.03).value(), 0.03);
        assert_eq!(ChmHealthMetric::Bandwidth(1e6).value(), 1e6);
        assert_eq!(ChmHealthMetric::Jitter(15.0).value(), 15.0);
        assert_eq!(ChmHealthMetric::ErrorRate(0.07).value(), 0.07);
    }

    #[test]
    fn test_health_metric_kind() {
        assert_eq!(ChmHealthMetric::Latency(0.0).kind(), "latency");
        assert_eq!(ChmHealthMetric::PacketLoss(0.0).kind(), "packet_loss");
        assert_eq!(ChmHealthMetric::Bandwidth(0.0).kind(), "bandwidth");
        assert_eq!(ChmHealthMetric::Jitter(0.0).kind(), "jitter");
        assert_eq!(ChmHealthMetric::ErrorRate(0.0).kind(), "error_rate");
    }

    // -----------------------------------------------------------------------
    // 17. AlertThresholds defaults and warning levels
    // -----------------------------------------------------------------------

    #[test]
    fn test_alert_thresholds_defaults() {
        let t = AlertThresholds::default();
        assert_eq!(t.max_latency_ms, 500.0);
        assert_eq!(t.max_packet_loss, 0.05);
        assert_eq!(t.min_bandwidth_bps, 100_000.0);
        assert_eq!(t.max_jitter_ms, 100.0);
        assert_eq!(t.max_error_rate, 0.1);
    }

    #[test]
    fn test_alert_thresholds_warning_levels() {
        let t = AlertThresholds::default();
        assert!((t.warning_latency_ms() - 400.0).abs() < 1e-9);
        assert!((t.warning_packet_loss() - 0.04).abs() < 1e-9);
        assert!((t.warning_jitter_ms() - 80.0).abs() < 1e-9);
        assert!((t.warning_error_rate() - 0.08).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 18. check_thresholds directly
    // -----------------------------------------------------------------------

    #[test]
    fn test_check_thresholds_exact_boundary() {
        let m = default_monitor();
        // Exactly at warning boundary (400 ms).
        let alerts = m.check_thresholds("p1", &ChmHealthMetric::Latency(400.0), 400.0, 1000);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, ChmAlertSeverity::Warning);
    }

    #[test]
    fn test_check_thresholds_returns_threshold_value() {
        let m = default_monitor();
        let alerts = m.check_thresholds("p1", &ChmHealthMetric::Latency(600.0), 600.0, 1000);
        assert_eq!(alerts[0].threshold, 500.0);
        assert_eq!(alerts[0].value, 600.0);
    }

    // -----------------------------------------------------------------------
    // 19. Multi-peer isolation
    // -----------------------------------------------------------------------

    #[test]
    fn test_peers_do_not_interfere() {
        let mut m = default_monitor();
        m.record(latency_sample("p1", 10.0, 1000));
        m.record(latency_sample("p2", 900.0, 1000));
        let s1 = m.health_score("p1").unwrap_or(0.0);
        let s2 = m.health_score("p2").unwrap_or(0.0);
        assert!(s1 > s2);
    }

    // -----------------------------------------------------------------------
    // 20. last_updated is set correctly
    // -----------------------------------------------------------------------

    #[test]
    fn test_last_updated_reflects_sample_timestamp() {
        let mut m = default_monitor();
        m.record(latency_sample("p1", 50.0, 42_000));
        let conn = m.connections.get("p1").expect("exists");
        assert_eq!(conn.last_updated, 42_000);
    }

    // -----------------------------------------------------------------------
    // 21. Custom thresholds
    // -----------------------------------------------------------------------

    #[test]
    fn test_custom_thresholds_stricter_latency() {
        let thresholds = AlertThresholds {
            max_latency_ms: 100.0,
            ..Default::default()
        };
        let mut m = ConnectionHealthMonitor::new(thresholds, 100);
        // 80 ms >= 80 % of 100 ms → warning.
        let alerts = m.record(latency_sample("p1", 80.0, 1000));
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, ChmAlertSeverity::Warning);
    }

    #[test]
    fn test_custom_thresholds_relaxed_packet_loss() {
        let thresholds = AlertThresholds {
            max_packet_loss: 0.5,
            ..Default::default()
        };
        let mut m = ConnectionHealthMonitor::new(thresholds, 100);
        // 0.05 < 80 % of 0.5 = 0.4 → no alert.
        let alerts = m.record(packet_loss_sample("p1", 0.05, 1000));
        assert!(alerts.is_empty());
    }

    // -----------------------------------------------------------------------
    // 22. Bandwidth score capped at 1.0
    // -----------------------------------------------------------------------

    #[test]
    fn test_bandwidth_score_capped_at_one() {
        let mut conn = ConnectionHealth::new("p".to_string());
        // 100 Mbps >> 1 Mbps threshold → capped.
        conn.samples.push_back(ChmHealthSample {
            peer_id: "p".to_string(),
            metric: ChmHealthMetric::Bandwidth(100_000_000.0),
            timestamp: 1,
        });
        let score = ConnectionHealthMonitor::compute_health_score(&conn);
        assert!(score <= 1.0);
    }

    // -----------------------------------------------------------------------
    // 23. Mixed metric recording
    // -----------------------------------------------------------------------

    #[test]
    fn test_mixed_metrics_all_recorded() {
        let mut m = default_monitor();
        m.record(latency_sample("p1", 10.0, 1));
        m.record(bandwidth_sample("p1", 1_000_000.0, 2));
        m.record(packet_loss_sample("p1", 0.0, 3));
        m.record(jitter_sample("p1", 5.0, 4));
        m.record(error_rate_sample("p1", 0.0, 5));
        let conn = m.connections.get("p1").expect("exists");
        assert_eq!(conn.samples.len(), 5);
    }

    #[test]
    fn test_mixed_metrics_health_score_near_one() {
        let mut m = default_monitor();
        m.record(latency_sample("p1", 0.0, 1));
        m.record(bandwidth_sample("p1", 1_000_000.0, 2));
        m.record(packet_loss_sample("p1", 0.0, 3));
        m.record(jitter_sample("p1", 0.0, 4));
        m.record(error_rate_sample("p1", 0.0, 5));
        let score = m.health_score("p1").unwrap_or(0.0);
        assert!(score > 0.95, "score={score}");
    }

    // -----------------------------------------------------------------------
    // 24. ChmMonitorStats fields
    // -----------------------------------------------------------------------

    #[test]
    fn test_monitor_stats_fields_accessible() {
        let s = ChmMonitorStats::default();
        assert_eq!(s.total_peers, 0);
        assert_eq!(s.healthy_peers, 0);
        assert_eq!(s.degraded_peers, 0);
        assert_eq!(s.critical_peers, 0);
        assert_eq!(s.total_alerts, 0);
        assert_eq!(s.avg_health_score, 0.0);
    }

    // -----------------------------------------------------------------------
    // 25. ChmHealthAlert fields
    // -----------------------------------------------------------------------

    #[test]
    fn test_health_alert_fields() {
        let alert = ChmHealthAlert {
            peer_id: "test".to_string(),
            metric: ChmHealthMetric::Latency(600.0),
            severity: ChmAlertSeverity::Critical,
            value: 600.0,
            threshold: 500.0,
            timestamp: 99_000,
        };
        assert_eq!(alert.peer_id, "test");
        assert_eq!(alert.value, 600.0);
        assert_eq!(alert.threshold, 500.0);
        assert_eq!(alert.timestamp, 99_000);
    }
}
