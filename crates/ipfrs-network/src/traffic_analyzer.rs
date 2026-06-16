//! Network Traffic Analysis and Pattern Detection
//!
//! This module provides tools for analyzing network traffic patterns, detecting anomalies,
//! and gaining insights into network behavior.
//!
//! # Features
//!
//! - **Traffic Pattern Analysis**: Identify patterns in connection and query behavior
//! - **Anomaly Detection**: Detect unusual network activity
//! - **Bandwidth Analysis**: Analyze bandwidth usage patterns over time
//! - **Peer Behavior Profiling**: Profile peer connection and query patterns
//! - **Protocol Distribution**: Analyze protocol usage distribution
//! - **Time-Series Analysis**: Track metrics over time windows
//! - **Statistical Analysis**: Compute statistics and trends
//!
//! # Example
//!
//! ```rust
//! use ipfrs_network::traffic_analyzer::{TrafficAnalyzer, TrafficAnalyzerConfig, TrafficEvent};
//! use std::time::Duration;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = TrafficAnalyzerConfig::default();
//! let mut analyzer = TrafficAnalyzer::new(config);
//!
//! // Record traffic events
//! analyzer.record_connection("peer1".to_string(), 1024);
//! analyzer.record_query("peer1".to_string(), Duration::from_millis(50), true);
//!
//! // Analyze patterns
//! let analysis = analyzer.analyze()?;
//! println!("Total bandwidth: {} bytes", analysis.total_bandwidth);
//! println!("Anomalies detected: {}", analysis.anomalies.len());
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// Helper function for serde default
fn instant_now() -> Instant {
    Instant::now()
}

/// Configuration for traffic analyzer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficAnalyzerConfig {
    /// Window size for pattern analysis
    pub window_size: Duration,
    /// Number of historical windows to keep
    pub history_size: usize,
    /// Threshold for anomaly detection (standard deviations)
    pub anomaly_threshold: f64,
    /// Minimum samples needed for statistics
    pub min_samples: usize,
    /// Enable detailed peer profiling
    pub enable_peer_profiling: bool,
    /// Enable protocol distribution tracking
    pub enable_protocol_tracking: bool,
}

impl Default for TrafficAnalyzerConfig {
    fn default() -> Self {
        Self {
            window_size: Duration::from_secs(60),
            history_size: 100,
            anomaly_threshold: 3.0,
            min_samples: 10,
            enable_peer_profiling: true,
            enable_protocol_tracking: true,
        }
    }
}

impl TrafficAnalyzerConfig {
    /// Create configuration for short-term analysis (1 minute windows)
    pub fn short_term() -> Self {
        Self {
            window_size: Duration::from_secs(60),
            history_size: 60,
            ..Default::default()
        }
    }

    /// Create configuration for long-term analysis (1 hour windows)
    pub fn long_term() -> Self {
        Self {
            window_size: Duration::from_secs(3600),
            history_size: 24,
            ..Default::default()
        }
    }

    /// Create configuration for real-time monitoring (5 second windows)
    pub fn realtime() -> Self {
        Self {
            window_size: Duration::from_secs(5),
            history_size: 720, // 1 hour of 5-second windows
            anomaly_threshold: 2.5,
            min_samples: 5,
            ..Default::default()
        }
    }
}

/// Type of traffic event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrafficEvent {
    /// Connection established
    ConnectionEstablished {
        peer_id: String,
        #[serde(skip, default = "instant_now")]
        timestamp: Instant,
    },
    /// Connection closed
    ConnectionClosed {
        peer_id: String,
        duration: Duration,
        bytes_transferred: u64,
    },
    /// DHT query
    Query {
        peer_id: String,
        latency: Duration,
        success: bool,
    },
    /// Bandwidth sample
    BandwidthSample {
        bytes_sent: u64,
        bytes_received: u64,
        duration: Duration,
    },
    /// Protocol usage
    ProtocolUsage {
        protocol: String,
        message_count: u64,
    },
}

/// Traffic pattern detected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficPattern {
    /// Pattern type
    pub pattern_type: PatternType,
    /// Pattern description
    pub description: String,
    /// Confidence score (0.0-1.0)
    pub confidence: f64,
    /// Pattern start time
    #[serde(skip, default = "instant_now")]
    pub start_time: Instant,
    /// Pattern duration
    pub duration: Duration,
}

/// Type of traffic pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    /// Steady traffic
    Steady,
    /// Increasing traffic
    Increasing,
    /// Decreasing traffic
    Decreasing,
    /// Bursty traffic
    Bursty,
    /// Periodic traffic
    Periodic,
    /// Anomalous traffic
    Anomalous,
}

/// Detected anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficAnomaly {
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Description
    pub description: String,
    /// Severity (0.0-1.0, higher is more severe)
    pub severity: f64,
    /// Detection timestamp
    #[serde(skip, default = "instant_now")]
    pub timestamp: Instant,
    /// Affected peer (if applicable)
    pub peer_id: Option<String>,
}

/// Type of anomaly
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Bandwidth spike
    BandwidthSpike,
    /// Unusual connection pattern
    ConnectionAnomaly,
    /// Query failure spike
    QueryFailureSpike,
    /// Latency spike
    LatencySpike,
    /// Suspicious peer behavior
    SuspiciousPeer,
}

/// Peer behavior profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerProfile {
    /// Peer ID
    pub peer_id: String,
    /// Total connections
    pub total_connections: usize,
    /// Total queries
    pub total_queries: u64,
    /// Successful queries
    pub successful_queries: u64,
    /// Average latency
    pub average_latency: Duration,
    /// Total bytes transferred
    pub total_bytes: u64,
    /// First seen timestamp
    #[serde(skip, default = "instant_now")]
    pub first_seen: Instant,
    /// Last seen timestamp
    #[serde(skip, default = "instant_now")]
    pub last_seen: Instant,
    /// Behavior score (0.0-1.0, higher is better)
    pub behavior_score: f64,
}

/// Traffic analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficAnalysis {
    /// Analysis timestamp
    #[serde(skip, default = "instant_now")]
    pub timestamp: Instant,
    /// Total bandwidth (bytes)
    pub total_bandwidth: u64,
    /// Total connections
    pub total_connections: usize,
    /// Total queries
    pub total_queries: u64,
    /// Query success rate
    pub query_success_rate: f64,
    /// Average latency
    pub average_latency: Duration,
    /// Detected patterns
    pub patterns: Vec<TrafficPattern>,
    /// Detected anomalies
    pub anomalies: Vec<TrafficAnomaly>,
    /// Peer profiles
    pub peer_profiles: HashMap<String, PeerProfile>,
    /// Protocol distribution
    pub protocol_distribution: HashMap<String, u64>,
    /// Bandwidth trend (increasing/decreasing/steady)
    pub bandwidth_trend: TrendDirection,
    /// Connection trend
    pub connection_trend: TrendDirection,
}

/// Trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Increasing trend
    Increasing,
    /// Decreasing trend
    Decreasing,
    /// Steady trend
    Steady,
    /// Insufficient data
    Unknown,
}

/// Traffic analyzer
pub struct TrafficAnalyzer {
    config: TrafficAnalyzerConfig,
    events: Vec<TrafficEvent>,
    peer_data: HashMap<String, PeerData>,
    bandwidth_history: Vec<BandwidthSample>,
    connection_history: Vec<usize>,
    query_history: Vec<QuerySample>,
    start_time: Instant,
}

#[derive(Debug, Clone)]
struct PeerData {
    connections: usize,
    queries: u64,
    successful_queries: u64,
    latencies: Vec<Duration>,
    bytes_transferred: u64,
    first_seen: Instant,
    last_seen: Instant,
}

#[derive(Debug, Clone)]
struct BandwidthSample {
    timestamp: Instant,
    bytes: u64,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct QuerySample {
    timestamp: Instant,
    success_rate: f64,
    latency: Duration,
}

impl TrafficAnalyzer {
    /// Create a new traffic analyzer
    pub fn new(config: TrafficAnalyzerConfig) -> Self {
        Self {
            config,
            events: Vec::new(),
            peer_data: HashMap::new(),
            bandwidth_history: Vec::new(),
            connection_history: Vec::new(),
            query_history: Vec::new(),
            start_time: Instant::now(),
        }
    }

    /// Record a connection event
    pub fn record_connection(&mut self, peer_id: String, bytes: u64) {
        let timestamp = Instant::now();
        self.events.push(TrafficEvent::ConnectionEstablished {
            peer_id: peer_id.clone(),
            timestamp,
        });

        let data = self.peer_data.entry(peer_id).or_insert(PeerData {
            connections: 0,
            queries: 0,
            successful_queries: 0,
            latencies: Vec::new(),
            bytes_transferred: 0,
            first_seen: timestamp,
            last_seen: timestamp,
        });

        data.connections += 1;
        data.bytes_transferred += bytes;
        data.last_seen = timestamp;

        self.connection_history.push(self.peer_data.len());
    }

    /// Record a query event
    pub fn record_query(&mut self, peer_id: String, latency: Duration, success: bool) {
        self.events.push(TrafficEvent::Query {
            peer_id: peer_id.clone(),
            latency,
            success,
        });

        if let Some(data) = self.peer_data.get_mut(&peer_id) {
            data.queries += 1;
            if success {
                data.successful_queries += 1;
            }
            data.latencies.push(latency);
        }

        let success_rate = if success { 1.0 } else { 0.0 };
        self.query_history.push(QuerySample {
            timestamp: Instant::now(),
            success_rate,
            latency,
        });
    }

    /// Record bandwidth usage
    pub fn record_bandwidth(&mut self, bytes_sent: u64, bytes_received: u64) {
        let total = bytes_sent + bytes_received;
        self.bandwidth_history.push(BandwidthSample {
            timestamp: Instant::now(),
            bytes: total,
        });

        self.events.push(TrafficEvent::BandwidthSample {
            bytes_sent,
            bytes_received,
            duration: Duration::from_secs(1),
        });
    }

    /// Analyze traffic and return results
    pub fn analyze(&self) -> Result<TrafficAnalysis, TrafficAnalyzerError> {
        let timestamp = Instant::now();

        // Calculate total bandwidth
        let total_bandwidth: u64 = self.bandwidth_history.iter().map(|s| s.bytes).sum();

        // Calculate total connections and queries
        let total_connections = self.connection_history.last().copied().unwrap_or(0);
        let total_queries: u64 = self.peer_data.values().map(|p| p.queries).sum();
        let successful_queries: u64 = self.peer_data.values().map(|p| p.successful_queries).sum();

        let query_success_rate = if total_queries > 0 {
            (successful_queries as f64 / total_queries as f64) * 100.0
        } else {
            0.0
        };

        // Calculate average latency
        let all_latencies: Vec<Duration> = self
            .peer_data
            .values()
            .flat_map(|p| p.latencies.iter().copied())
            .collect();

        let average_latency = if !all_latencies.is_empty() {
            let sum: Duration = all_latencies.iter().sum();
            sum / all_latencies.len() as u32
        } else {
            Duration::ZERO
        };

        // Detect patterns
        let patterns = self.detect_patterns();

        // Detect anomalies
        let anomalies = self.detect_anomalies();

        // Build peer profiles
        let peer_profiles = self.build_peer_profiles();

        // Determine trends
        let bandwidth_trend = self.calculate_trend(
            &self
                .bandwidth_history
                .iter()
                .map(|s| s.bytes as f64)
                .collect::<Vec<_>>(),
        );
        let connection_trend = self.calculate_trend(
            &self
                .connection_history
                .iter()
                .map(|&c| c as f64)
                .collect::<Vec<_>>(),
        );

        Ok(TrafficAnalysis {
            timestamp,
            total_bandwidth,
            total_connections,
            total_queries,
            query_success_rate,
            average_latency,
            patterns,
            anomalies,
            peer_profiles,
            protocol_distribution: HashMap::new(),
            bandwidth_trend,
            connection_trend,
        })
    }

    /// Detect traffic patterns
    fn detect_patterns(&self) -> Vec<TrafficPattern> {
        let mut patterns = Vec::new();

        // Detect bursty pattern if bandwidth varies significantly
        if self.bandwidth_history.len() >= self.config.min_samples {
            let values: Vec<f64> = self
                .bandwidth_history
                .iter()
                .map(|s| s.bytes as f64)
                .collect();
            let (mean, stddev) = Self::calculate_statistics(&values);

            if stddev > mean * 0.5 {
                patterns.push(TrafficPattern {
                    pattern_type: PatternType::Bursty,
                    description: "Traffic shows bursty pattern with high variance".to_string(),
                    confidence: 0.8,
                    start_time: self.start_time,
                    duration: Instant::now().duration_since(self.start_time),
                });
            }
        }

        patterns
    }

    /// Detect anomalies
    fn detect_anomalies(&self) -> Vec<TrafficAnomaly> {
        let mut anomalies = Vec::new();

        // Detect bandwidth spikes
        if self.bandwidth_history.len() >= self.config.min_samples {
            let values: Vec<f64> = self
                .bandwidth_history
                .iter()
                .map(|s| s.bytes as f64)
                .collect();
            let (mean, stddev) = Self::calculate_statistics(&values);

            for sample in &self.bandwidth_history {
                let z_score = (sample.bytes as f64 - mean).abs() / stddev.max(1.0);
                if z_score > self.config.anomaly_threshold {
                    anomalies.push(TrafficAnomaly {
                        anomaly_type: AnomalyType::BandwidthSpike,
                        description: format!(
                            "Bandwidth spike detected: {} bytes ({:.1} σ)",
                            sample.bytes, z_score
                        ),
                        severity: (z_score / self.config.anomaly_threshold).min(1.0),
                        timestamp: sample.timestamp,
                        peer_id: None,
                    });
                }
            }
        }

        // Detect query failure spikes
        if self.query_history.len() >= self.config.min_samples {
            let success_rates: Vec<f64> =
                self.query_history.iter().map(|q| q.success_rate).collect();
            let recent_rate = success_rates.iter().rev().take(10).sum::<f64>() / 10.0;

            if recent_rate < 0.5 {
                anomalies.push(TrafficAnomaly {
                    anomaly_type: AnomalyType::QueryFailureSpike,
                    description: format!(
                        "High query failure rate: {:.1}%",
                        (1.0 - recent_rate) * 100.0
                    ),
                    severity: 1.0 - recent_rate,
                    timestamp: Instant::now(),
                    peer_id: None,
                });
            }
        }

        anomalies
    }

    /// Build peer profiles
    fn build_peer_profiles(&self) -> HashMap<String, PeerProfile> {
        self.peer_data
            .iter()
            .map(|(peer_id, data)| {
                let average_latency = if !data.latencies.is_empty() {
                    let sum: Duration = data.latencies.iter().sum();
                    sum / data.latencies.len() as u32
                } else {
                    Duration::ZERO
                };

                let behavior_score = if data.queries > 0 {
                    (data.successful_queries as f64 / data.queries as f64) * 100.0 / 100.0
                } else {
                    1.0
                };

                let profile = PeerProfile {
                    peer_id: peer_id.clone(),
                    total_connections: data.connections,
                    total_queries: data.queries,
                    successful_queries: data.successful_queries,
                    average_latency,
                    total_bytes: data.bytes_transferred,
                    first_seen: data.first_seen,
                    last_seen: data.last_seen,
                    behavior_score,
                };

                (peer_id.clone(), profile)
            })
            .collect()
    }

    /// Calculate trend direction
    fn calculate_trend(&self, values: &[f64]) -> TrendDirection {
        if values.len() < self.config.min_samples {
            return TrendDirection::Unknown;
        }

        // Simple linear regression to determine trend
        let n = values.len() as f64;
        let x_mean = (0..values.len()).map(|i| i as f64).sum::<f64>() / n;
        let y_mean = values.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &y) in values.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean) * (x - x_mean);
        }

        let slope = if denominator != 0.0 {
            numerator / denominator
        } else {
            0.0
        };

        if slope > y_mean * 0.01 {
            TrendDirection::Increasing
        } else if slope < -y_mean * 0.01 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Steady
        }
    }

    /// Calculate statistics (mean, standard deviation)
    fn calculate_statistics(values: &[f64]) -> (f64, f64) {
        if values.is_empty() {
            return (0.0, 0.0);
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let stddev = variance.sqrt();

        (mean, stddev)
    }

    /// Get current statistics
    pub fn get_stats(&self) -> TrafficAnalyzerStats {
        TrafficAnalyzerStats {
            total_events: self.events.len(),
            total_peers: self.peer_data.len(),
            bandwidth_samples: self.bandwidth_history.len(),
            query_samples: self.query_history.len(),
            uptime: Instant::now().duration_since(self.start_time),
        }
    }

    /// Clear all recorded data
    pub fn clear(&mut self) {
        self.events.clear();
        self.peer_data.clear();
        self.bandwidth_history.clear();
        self.connection_history.clear();
        self.query_history.clear();
        self.start_time = Instant::now();
    }
}

/// Traffic analyzer statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficAnalyzerStats {
    /// Total events recorded
    pub total_events: usize,
    /// Total peers tracked
    pub total_peers: usize,
    /// Bandwidth samples collected
    pub bandwidth_samples: usize,
    /// Query samples collected
    pub query_samples: usize,
    /// Analyzer uptime
    pub uptime: Duration,
}

/// Error types for traffic analyzer
#[derive(Debug, thiserror::Error)]
pub enum TrafficAnalyzerError {
    #[error("Insufficient data for analysis")]
    InsufficientData,

    #[error("Analysis failed: {0}")]
    AnalysisFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_presets() {
        let short = TrafficAnalyzerConfig::short_term();
        assert_eq!(short.window_size, Duration::from_secs(60));

        let long = TrafficAnalyzerConfig::long_term();
        assert_eq!(long.window_size, Duration::from_secs(3600));

        let realtime = TrafficAnalyzerConfig::realtime();
        assert_eq!(realtime.window_size, Duration::from_secs(5));
    }

    #[test]
    fn test_analyzer_creation() {
        let config = TrafficAnalyzerConfig::default();
        let analyzer = TrafficAnalyzer::new(config);
        let stats = analyzer.get_stats();
        assert_eq!(stats.total_events, 0);
        assert_eq!(stats.total_peers, 0);
    }

    #[test]
    fn test_record_connection() {
        let config = TrafficAnalyzerConfig::default();
        let mut analyzer = TrafficAnalyzer::new(config);

        analyzer.record_connection("peer1".to_string(), 1024);
        let stats = analyzer.get_stats();
        assert_eq!(stats.total_peers, 1);
    }

    #[test]
    fn test_record_query() {
        let config = TrafficAnalyzerConfig::default();
        let mut analyzer = TrafficAnalyzer::new(config);

        analyzer.record_connection("peer1".to_string(), 0);
        analyzer.record_query("peer1".to_string(), Duration::from_millis(50), true);

        let stats = analyzer.get_stats();
        assert_eq!(stats.query_samples, 1);
    }

    #[test]
    fn test_record_bandwidth() {
        let config = TrafficAnalyzerConfig::default();
        let mut analyzer = TrafficAnalyzer::new(config);

        analyzer.record_bandwidth(1000, 2000);
        let stats = analyzer.get_stats();
        assert_eq!(stats.bandwidth_samples, 1);
    }

    #[test]
    fn test_analyze() {
        let config = TrafficAnalyzerConfig::default();
        let mut analyzer = TrafficAnalyzer::new(config);

        analyzer.record_connection("peer1".to_string(), 1024);
        analyzer.record_query("peer1".to_string(), Duration::from_millis(50), true);
        analyzer.record_bandwidth(1000, 2000);

        let analysis = analyzer.analyze().expect("test: analyze should succeed");
        assert_eq!(analysis.total_connections, 1);
        assert_eq!(analysis.total_queries, 1);
        assert_eq!(analysis.query_success_rate, 100.0);
    }

    #[test]
    fn test_peer_profile() {
        let config = TrafficAnalyzerConfig::default();
        let mut analyzer = TrafficAnalyzer::new(config);

        analyzer.record_connection("peer1".to_string(), 1024);
        analyzer.record_query("peer1".to_string(), Duration::from_millis(50), true);

        let analysis = analyzer.analyze().expect("test: analyze should succeed");
        let profile = analysis
            .peer_profiles
            .get("peer1")
            .expect("test: peer1 profile should exist");
        assert_eq!(profile.total_connections, 1);
        assert_eq!(profile.total_queries, 1);
        assert_eq!(profile.behavior_score, 1.0);
    }

    #[test]
    fn test_clear() {
        let config = TrafficAnalyzerConfig::default();
        let mut analyzer = TrafficAnalyzer::new(config);

        analyzer.record_connection("peer1".to_string(), 1024);
        analyzer.clear();

        let stats = analyzer.get_stats();
        assert_eq!(stats.total_events, 0);
        assert_eq!(stats.total_peers, 0);
    }

    #[test]
    fn test_trend_calculation() {
        let config = TrafficAnalyzerConfig::default();
        let analyzer = TrafficAnalyzer::new(config);

        // Increasing trend
        let increasing = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(
            analyzer.calculate_trend(&increasing),
            TrendDirection::Increasing
        );

        // Decreasing trend
        let decreasing = vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(
            analyzer.calculate_trend(&decreasing),
            TrendDirection::Decreasing
        );

        // Steady trend
        let steady = vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0];
        assert_eq!(analyzer.calculate_trend(&steady), TrendDirection::Steady);
    }

    #[test]
    fn test_statistics_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, stddev) = TrafficAnalyzer::calculate_statistics(&values);
        assert_eq!(mean, 3.0);
        assert!((stddev - 1.414).abs() < 0.01);
    }
}
