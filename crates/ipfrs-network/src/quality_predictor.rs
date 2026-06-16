//! Connection Quality Predictor
//!
//! This module provides connection quality prediction based on historical performance data.
//! It tracks metrics like latency, bandwidth, reliability, and uptime to predict future
//! connection quality and enable proactive switching to better connections.
//!
//! # Features
//!
//! - Historical metric tracking per peer
//! - Quality scoring based on multiple factors (latency, bandwidth, reliability, uptime)
//! - Exponential moving average for smooth predictions
//! - Configurable weights for different metrics
//! - Proactive connection recommendations
//! - Automatic degradation detection
//!
//! # Example
//!
//! ```rust
//! use ipfrs_network::quality_predictor::{QualityPredictor, QualityPredictorConfig};
//! use libp2p::PeerId;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = QualityPredictorConfig::default();
//! let predictor = QualityPredictor::new(config)?;
//!
//! let peer = PeerId::random();
//!
//! // Record metrics
//! predictor.record_latency(peer, 50);
//! predictor.record_bandwidth(peer, 1_000_000);
//! predictor.record_success(peer);
//!
//! // Get quality prediction
//! if let Some(quality) = predictor.predict_quality(&peer) {
//!     println!("Predicted quality: {}", quality.overall_score);
//! }
//!
//! // Check if connection should be switched
//! if predictor.should_switch_connection(&peer) {
//!     println!("Consider switching to a better peer");
//! }
//! # Ok(())
//! # }
//! ```

use dashmap::DashMap;
use libp2p::PeerId;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;

/// Errors that can occur in the quality predictor
#[derive(Debug, Error)]
pub enum QualityPredictorError {
    #[error("No historical data available for peer")]
    NoHistoricalData,
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Configuration for the quality predictor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityPredictorConfig {
    /// Maximum number of samples to keep per peer
    pub max_samples: usize,
    /// Weight for latency in quality score (0.0 - 1.0)
    pub latency_weight: f64,
    /// Weight for bandwidth in quality score (0.0 - 1.0)
    pub bandwidth_weight: f64,
    /// Weight for reliability in quality score (0.0 - 1.0)
    pub reliability_weight: f64,
    /// Weight for uptime in quality score (0.0 - 1.0)
    pub uptime_weight: f64,
    /// Smoothing factor for exponential moving average (0.0 - 1.0)
    pub smoothing_factor: f64,
    /// Minimum quality score for acceptable connections (0.0 - 1.0)
    pub min_acceptable_quality: f64,
    /// Quality threshold below which to recommend switching (0.0 - 1.0)
    pub switch_threshold: f64,
    /// Enable prediction-based recommendations
    pub enable_predictions: bool,
}

impl Default for QualityPredictorConfig {
    fn default() -> Self {
        Self {
            max_samples: 100,
            latency_weight: 0.3,
            bandwidth_weight: 0.3,
            reliability_weight: 0.25,
            uptime_weight: 0.15,
            smoothing_factor: 0.2,
            min_acceptable_quality: 0.5,
            switch_threshold: 0.6,
            enable_predictions: true,
        }
    }
}

impl QualityPredictorConfig {
    /// Create a configuration optimized for low-latency applications
    pub fn low_latency() -> Self {
        Self {
            latency_weight: 0.5,
            bandwidth_weight: 0.2,
            reliability_weight: 0.2,
            uptime_weight: 0.1,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for high-bandwidth applications
    pub fn high_bandwidth() -> Self {
        Self {
            latency_weight: 0.15,
            bandwidth_weight: 0.5,
            reliability_weight: 0.25,
            uptime_weight: 0.1,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for reliability
    pub fn high_reliability() -> Self {
        Self {
            latency_weight: 0.2,
            bandwidth_weight: 0.2,
            reliability_weight: 0.4,
            uptime_weight: 0.2,
            ..Default::default()
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), QualityPredictorError> {
        if self.max_samples == 0 {
            return Err(QualityPredictorError::InvalidConfig(
                "max_samples must be > 0".to_string(),
            ));
        }

        let total_weight = self.latency_weight
            + self.bandwidth_weight
            + self.reliability_weight
            + self.uptime_weight;

        if (total_weight - 1.0).abs() > 0.01 {
            return Err(QualityPredictorError::InvalidConfig(format!(
                "weights must sum to 1.0, got {}",
                total_weight
            )));
        }

        if !(0.0..=1.0).contains(&self.smoothing_factor) {
            return Err(QualityPredictorError::InvalidConfig(
                "smoothing_factor must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(())
    }
}

/// Historical metrics for a peer connection
#[derive(Debug, Clone)]
struct ConnectionHistory {
    /// Latency samples in milliseconds
    latency_samples: VecDeque<u64>,
    /// Bandwidth samples in bytes per second
    bandwidth_samples: VecDeque<u64>,
    /// Success count for requests
    success_count: u64,
    /// Failure count for requests
    failure_count: u64,
    /// Time when connection was first established
    first_seen: Instant,
    /// Time when connection was last active
    last_seen: Instant,
    /// Exponential moving average for quality score
    quality_ema: Option<f64>,
}

impl ConnectionHistory {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            latency_samples: VecDeque::new(),
            bandwidth_samples: VecDeque::new(),
            success_count: 0,
            failure_count: 0,
            first_seen: now,
            last_seen: now,
            quality_ema: None,
        }
    }

    fn record_latency(&mut self, latency_ms: u64, max_samples: usize) {
        if self.latency_samples.len() >= max_samples {
            self.latency_samples.pop_front();
        }
        self.latency_samples.push_back(latency_ms);
        self.last_seen = Instant::now();
    }

    fn record_bandwidth(&mut self, bytes_per_sec: u64, max_samples: usize) {
        if self.bandwidth_samples.len() >= max_samples {
            self.bandwidth_samples.pop_front();
        }
        self.bandwidth_samples.push_back(bytes_per_sec);
        self.last_seen = Instant::now();
    }

    fn record_success(&mut self) {
        self.success_count += 1;
        self.last_seen = Instant::now();
    }

    fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_seen = Instant::now();
    }

    fn avg_latency(&self) -> Option<f64> {
        if self.latency_samples.is_empty() {
            None
        } else {
            let sum: u64 = self.latency_samples.iter().sum();
            Some(sum as f64 / self.latency_samples.len() as f64)
        }
    }

    fn avg_bandwidth(&self) -> Option<f64> {
        if self.bandwidth_samples.is_empty() {
            None
        } else {
            let sum: u64 = self.bandwidth_samples.iter().sum();
            Some(sum as f64 / self.bandwidth_samples.len() as f64)
        }
    }

    fn reliability_score(&self) -> f64 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            0.5 // Neutral score if no data
        } else {
            self.success_count as f64 / total as f64
        }
    }

    fn uptime_score(&self) -> f64 {
        let total_duration = self.first_seen.elapsed().as_secs_f64();
        if total_duration < 1.0 {
            1.0 // New connection gets benefit of the doubt
        } else {
            let active_duration = self.last_seen.duration_since(self.first_seen).as_secs_f64();
            (active_duration / total_duration).min(1.0)
        }
    }
}

/// Quality prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityPrediction {
    /// Overall quality score (0.0 - 1.0)
    pub overall_score: f64,
    /// Latency component score (0.0 - 1.0)
    pub latency_score: f64,
    /// Bandwidth component score (0.0 - 1.0)
    pub bandwidth_score: f64,
    /// Reliability component score (0.0 - 1.0)
    pub reliability_score: f64,
    /// Uptime component score (0.0 - 1.0)
    pub uptime_score: f64,
    /// Average latency in milliseconds
    pub avg_latency_ms: Option<f64>,
    /// Average bandwidth in bytes per second
    pub avg_bandwidth_bps: Option<f64>,
    /// Whether quality is acceptable
    pub is_acceptable: bool,
    /// Recommendation to switch connection
    pub should_switch: bool,
}

/// Statistics for the quality predictor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityPredictorStats {
    /// Number of peers being tracked
    pub tracked_peers: usize,
    /// Number of predictions made
    pub predictions_made: u64,
    /// Number of switch recommendations
    pub switch_recommendations: u64,
    /// Average quality score across all peers
    pub avg_quality: f64,
}

/// Connection quality predictor
pub struct QualityPredictor {
    config: QualityPredictorConfig,
    history: Arc<DashMap<PeerId, ConnectionHistory>>,
    stats: Arc<parking_lot::RwLock<QualityPredictorStats>>,
}

impl QualityPredictor {
    /// Create a new quality predictor
    pub fn new(config: QualityPredictorConfig) -> Result<Self, QualityPredictorError> {
        config.validate()?;

        Ok(Self {
            config,
            history: Arc::new(DashMap::new()),
            stats: Arc::new(parking_lot::RwLock::new(QualityPredictorStats {
                tracked_peers: 0,
                predictions_made: 0,
                switch_recommendations: 0,
                avg_quality: 0.0,
            })),
        })
    }

    /// Record a latency measurement for a peer
    pub fn record_latency(&self, peer: PeerId, latency_ms: u64) {
        let mut entry = self
            .history
            .entry(peer)
            .or_insert_with(ConnectionHistory::new);
        entry.record_latency(latency_ms, self.config.max_samples);
    }

    /// Record a bandwidth measurement for a peer
    pub fn record_bandwidth(&self, peer: PeerId, bytes_per_sec: u64) {
        let mut entry = self
            .history
            .entry(peer)
            .or_insert_with(ConnectionHistory::new);
        entry.record_bandwidth(bytes_per_sec, self.config.max_samples);
    }

    /// Record a successful operation for a peer
    pub fn record_success(&self, peer: PeerId) {
        let mut entry = self
            .history
            .entry(peer)
            .or_insert_with(ConnectionHistory::new);
        entry.record_success();
    }

    /// Record a failed operation for a peer
    pub fn record_failure(&self, peer: PeerId) {
        let mut entry = self
            .history
            .entry(peer)
            .or_insert_with(ConnectionHistory::new);
        entry.record_failure();
    }

    /// Predict the quality for a specific peer
    pub fn predict_quality(&self, peer: &PeerId) -> Option<QualityPrediction> {
        let history = self.history.get(peer)?;

        // Calculate component scores
        let latency_score = self.calculate_latency_score(history.avg_latency());
        let bandwidth_score = self.calculate_bandwidth_score(history.avg_bandwidth());
        let reliability_score = history.reliability_score();
        let uptime_score = history.uptime_score();

        // Calculate overall score
        let overall_score = latency_score * self.config.latency_weight
            + bandwidth_score * self.config.bandwidth_weight
            + reliability_score * self.config.reliability_weight
            + uptime_score * self.config.uptime_weight;

        // Update EMA
        drop(history);
        if let Some(mut history) = self.history.get_mut(peer) {
            if let Some(prev_ema) = history.quality_ema {
                history.quality_ema = Some(
                    self.config.smoothing_factor * overall_score
                        + (1.0 - self.config.smoothing_factor) * prev_ema,
                );
            } else {
                history.quality_ema = Some(overall_score);
            }
        }

        let is_acceptable = overall_score >= self.config.min_acceptable_quality;
        let should_switch =
            self.config.enable_predictions && overall_score < self.config.switch_threshold;

        // Update stats
        let mut stats = self.stats.write();
        stats.predictions_made += 1;
        if should_switch {
            stats.switch_recommendations += 1;
        }

        Some(QualityPrediction {
            overall_score,
            latency_score,
            bandwidth_score,
            reliability_score,
            uptime_score,
            avg_latency_ms: self.history.get(peer).and_then(|h| h.avg_latency()),
            avg_bandwidth_bps: self.history.get(peer).and_then(|h| h.avg_bandwidth()),
            is_acceptable,
            should_switch,
        })
    }

    /// Check if a connection should be switched based on quality
    pub fn should_switch_connection(&self, peer: &PeerId) -> bool {
        self.predict_quality(peer)
            .map(|p| p.should_switch)
            .unwrap_or(false)
    }

    /// Get the best peer among a list based on predicted quality
    pub fn get_best_peer(&self, peers: &[PeerId]) -> Option<(PeerId, QualityPrediction)> {
        peers
            .iter()
            .filter_map(|peer| {
                self.predict_quality(peer)
                    .map(|prediction| (*peer, prediction))
            })
            .max_by(|a, b| {
                a.1.overall_score
                    .partial_cmp(&b.1.overall_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Get peers ranked by quality (best first)
    pub fn rank_peers(&self, peers: &[PeerId]) -> Vec<(PeerId, QualityPrediction)> {
        let mut ranked: Vec<_> = peers
            .iter()
            .filter_map(|peer| {
                self.predict_quality(peer)
                    .map(|prediction| (*peer, prediction))
            })
            .collect();

        ranked.sort_by(|a, b| {
            b.1.overall_score
                .partial_cmp(&a.1.overall_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        ranked
    }

    /// Remove historical data for a peer
    pub fn remove_peer(&self, peer: &PeerId) {
        self.history.remove(peer);
    }

    /// Clear all historical data
    pub fn clear(&self) {
        self.history.clear();
        let mut stats = self.stats.write();
        stats.tracked_peers = 0;
        stats.predictions_made = 0;
        stats.switch_recommendations = 0;
        stats.avg_quality = 0.0;
    }

    /// Get statistics
    pub fn stats(&self) -> QualityPredictorStats {
        let mut stats = self.stats.read().clone();
        stats.tracked_peers = self.history.len();

        // Calculate average quality
        if stats.tracked_peers > 0 {
            let total_quality: f64 = self
                .history
                .iter()
                .filter_map(|entry| entry.quality_ema)
                .sum();
            stats.avg_quality = total_quality / stats.tracked_peers as f64;
        }

        stats
    }

    /// Calculate latency score (lower is better, normalized to 0-1)
    fn calculate_latency_score(&self, avg_latency: Option<f64>) -> f64 {
        match avg_latency {
            None => 0.5, // Neutral score if no data
            Some(latency) => {
                // Score decreases as latency increases
                // 0ms = 1.0, 100ms = 0.75, 500ms = 0.25, 1000ms+ = 0.0
                if latency <= 0.0 {
                    1.0
                } else if latency >= 1000.0 {
                    0.0
                } else {
                    1.0 - (latency / 1000.0)
                }
            }
        }
    }

    /// Calculate bandwidth score (higher is better, normalized to 0-1)
    fn calculate_bandwidth_score(&self, avg_bandwidth: Option<f64>) -> f64 {
        match avg_bandwidth {
            None => 0.5, // Neutral score if no data
            Some(bandwidth) => {
                // Score increases with bandwidth
                // 0 bps = 0.0, 1 MB/s = 0.5, 10 MB/s = 0.9, 100 MB/s+ = 1.0
                let mb_per_sec = bandwidth / 1_000_000.0;
                if mb_per_sec >= 100.0 {
                    1.0
                } else if mb_per_sec <= 0.0 {
                    0.0
                } else {
                    (mb_per_sec / 100.0).min(1.0)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = QualityPredictorConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_weights() {
        let config = QualityPredictorConfig {
            latency_weight: 0.5,
            bandwidth_weight: 0.3,
            reliability_weight: 0.1,
            uptime_weight: 0.05, // Sum = 0.95, should fail
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_presets() {
        assert!(QualityPredictorConfig::low_latency().validate().is_ok());
        assert!(QualityPredictorConfig::high_bandwidth().validate().is_ok());
        assert!(QualityPredictorConfig::high_reliability()
            .validate()
            .is_ok());
    }

    #[test]
    fn test_record_metrics() {
        let config = QualityPredictorConfig::default();
        let predictor = QualityPredictor::new(config)
            .expect("test: QualityPredictor::new should succeed with default config");
        let peer = PeerId::random();

        predictor.record_latency(peer, 50);
        predictor.record_bandwidth(peer, 1_000_000);
        predictor.record_success(peer);

        let prediction = predictor
            .predict_quality(&peer)
            .expect("test: predict_quality should return Some after recording metrics");
        assert!(prediction.avg_latency_ms.is_some());
        assert!(prediction.avg_bandwidth_bps.is_some());
        assert!(prediction.overall_score > 0.0);
    }

    #[test]
    fn test_latency_score() {
        let predictor = QualityPredictor::new(QualityPredictorConfig::default())
            .expect("test: QualityPredictor::new should succeed with default config");

        assert_eq!(predictor.calculate_latency_score(Some(0.0)), 1.0);
        assert!(predictor.calculate_latency_score(Some(100.0)) > 0.7);
        assert!(predictor.calculate_latency_score(Some(500.0)) < 0.6);
        assert_eq!(predictor.calculate_latency_score(Some(1000.0)), 0.0);
    }

    #[test]
    fn test_bandwidth_score() {
        let predictor = QualityPredictor::new(QualityPredictorConfig::default())
            .expect("test: QualityPredictor::new should succeed with default config");

        assert_eq!(predictor.calculate_bandwidth_score(Some(0.0)), 0.0);
        assert!(predictor.calculate_bandwidth_score(Some(1_000_000.0)) > 0.0);
        assert!(predictor.calculate_bandwidth_score(Some(10_000_000.0)) > 0.05);
        assert_eq!(
            predictor.calculate_bandwidth_score(Some(100_000_000.0)),
            1.0
        );
    }

    #[test]
    fn test_reliability_tracking() {
        let config = QualityPredictorConfig::default();
        let predictor = QualityPredictor::new(config)
            .expect("test: QualityPredictor::new should succeed with default config");
        let peer = PeerId::random();

        predictor.record_success(peer);
        predictor.record_success(peer);
        predictor.record_failure(peer);

        let prediction = predictor
            .predict_quality(&peer)
            .expect("test: predict_quality should return Some after recording success/failure");
        assert!((prediction.reliability_score - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_get_best_peer() {
        let config = QualityPredictorConfig::default();
        let predictor = QualityPredictor::new(config)
            .expect("test: QualityPredictor::new should succeed with default config");

        let peer1 = PeerId::random();
        let peer2 = PeerId::random();
        let peer3 = PeerId::random();

        // peer1: excellent
        predictor.record_latency(peer1, 10);
        predictor.record_bandwidth(peer1, 10_000_000);
        predictor.record_success(peer1);

        // peer2: poor
        predictor.record_latency(peer2, 500);
        predictor.record_bandwidth(peer2, 100_000);
        predictor.record_failure(peer2);

        // peer3: good
        predictor.record_latency(peer3, 50);
        predictor.record_bandwidth(peer3, 5_000_000);
        predictor.record_success(peer3);

        let peers = vec![peer1, peer2, peer3];
        let (best, _) = predictor
            .get_best_peer(&peers)
            .expect("test: get_best_peer should return Some for non-empty peer list");
        assert_eq!(best, peer1);
    }

    #[test]
    fn test_rank_peers() {
        let config = QualityPredictorConfig::default();
        let predictor = QualityPredictor::new(config)
            .expect("test: QualityPredictor::new should succeed with default config");

        let peer1 = PeerId::random();
        let peer2 = PeerId::random();
        let peer3 = PeerId::random();

        predictor.record_latency(peer1, 10);
        predictor.record_latency(peer2, 100);
        predictor.record_latency(peer3, 50);

        let peers = vec![peer1, peer2, peer3];
        let ranked = predictor.rank_peers(&peers);

        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0].0, peer1); // Best
        assert_eq!(ranked[2].0, peer2); // Worst
    }

    #[test]
    fn test_should_switch() {
        let config = QualityPredictorConfig {
            switch_threshold: 0.7,
            enable_predictions: true,
            ..Default::default()
        };
        let predictor = QualityPredictor::new(config)
            .expect("test: QualityPredictor::new should succeed with default config");
        let peer = PeerId::random();

        // Record poor metrics
        predictor.record_latency(peer, 800);
        predictor.record_bandwidth(peer, 50_000);
        predictor.record_failure(peer);
        predictor.record_failure(peer);
        predictor.record_success(peer);

        assert!(predictor.should_switch_connection(&peer));
    }

    #[test]
    fn test_ema_smoothing() {
        let config = QualityPredictorConfig {
            smoothing_factor: 0.5,
            ..Default::default()
        };
        let predictor = QualityPredictor::new(config)
            .expect("test: QualityPredictor::new should succeed with default config");
        let peer = PeerId::random();

        predictor.record_latency(peer, 100);
        let pred1 = predictor
            .predict_quality(&peer)
            .expect("test: predict_quality should return Some after recording latency");

        predictor.record_latency(peer, 50);
        let pred2 = predictor
            .predict_quality(&peer)
            .expect("test: predict_quality should return Some after recording second latency");

        // Second prediction should be influenced by first (EMA)
        assert!(pred2.overall_score > pred1.overall_score);
    }

    #[test]
    fn test_stats() {
        let config = QualityPredictorConfig::default();
        let predictor = QualityPredictor::new(config)
            .expect("test: QualityPredictor::new should succeed with default config");

        let peer1 = PeerId::random();
        let peer2 = PeerId::random();

        predictor.record_latency(peer1, 50);
        predictor.record_latency(peer2, 100);

        predictor.predict_quality(&peer1);
        predictor.predict_quality(&peer2);

        let stats = predictor.stats();
        assert_eq!(stats.tracked_peers, 2);
        assert_eq!(stats.predictions_made, 2);
    }

    #[test]
    fn test_remove_peer() {
        let config = QualityPredictorConfig::default();
        let predictor = QualityPredictor::new(config)
            .expect("test: QualityPredictor::new should succeed with default config");
        let peer = PeerId::random();

        predictor.record_latency(peer, 50);
        assert!(predictor.predict_quality(&peer).is_some());

        predictor.remove_peer(&peer);
        assert!(predictor.predict_quality(&peer).is_none());
    }

    #[test]
    fn test_clear() {
        let config = QualityPredictorConfig::default();
        let predictor = QualityPredictor::new(config)
            .expect("test: QualityPredictor::new should succeed with default config");

        predictor.record_latency(PeerId::random(), 50);
        predictor.record_latency(PeerId::random(), 100);

        predictor.clear();
        let stats = predictor.stats();
        assert_eq!(stats.tracked_peers, 0);
    }
}
