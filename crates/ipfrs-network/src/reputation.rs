//! Peer reputation system for tracking and scoring peer behavior
//!
//! This module provides a comprehensive reputation system that tracks peer behavior
//! over time and calculates reputation scores based on various factors including:
//! - Transfer success/failure rates
//! - Response times and latency
//! - Protocol violations
//! - Uptime and reliability
//! - Historical behavior patterns
//!
//! # Examples
//!
//! ```
//! use ipfrs_network::reputation::{ReputationManager, ReputationConfig};
//! use libp2p::PeerId;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = ReputationConfig::default();
//! let mut manager = ReputationManager::new(config.clone());
//!
//! // Track successful interaction
//! let peer_id = PeerId::random();
//! manager.record_successful_transfer(&peer_id, 1024);
//! manager.record_low_latency(&peer_id, 50);
//!
//! // Get reputation score
//! if let Some(score) = manager.get_reputation(&peer_id) {
//!     println!("Peer reputation: {:.2}", score.overall_score(&config));
//! }
//!
//! // Check if peer is trusted
//! assert!(manager.is_trusted(&peer_id));
//! # Ok(())
//! # }
//! ```

use dashmap::DashMap;
use libp2p::PeerId;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Configuration for the reputation system
#[derive(Debug, Clone)]
pub struct ReputationConfig {
    /// Minimum score to be considered trusted (0.0-1.0)
    pub trust_threshold: f64,

    /// Score below which a peer is considered bad (0.0-1.0)
    pub bad_peer_threshold: f64,

    /// Weight for transfer success rate (0.0-1.0)
    pub transfer_success_weight: f64,

    /// Weight for latency score (0.0-1.0)
    pub latency_weight: f64,

    /// Weight for protocol compliance (0.0-1.0)
    pub protocol_compliance_weight: f64,

    /// Weight for uptime score (0.0-1.0)
    pub uptime_weight: f64,

    /// Maximum number of peers to track
    pub max_tracked_peers: usize,

    /// How long to remember peer reputation after last interaction
    pub retention_period: Duration,

    /// Decay factor for old scores (0.0-1.0, higher = faster decay)
    pub score_decay_rate: f64,

    /// Exponential moving average alpha for score updates (0.0-1.0)
    pub ema_alpha: f64,
}

impl Default for ReputationConfig {
    fn default() -> Self {
        Self {
            trust_threshold: 0.7,
            bad_peer_threshold: 0.3,
            transfer_success_weight: 0.4,
            latency_weight: 0.2,
            protocol_compliance_weight: 0.2,
            uptime_weight: 0.2,
            max_tracked_peers: 10000,
            retention_period: Duration::from_secs(24 * 3600), // 24 hours
            score_decay_rate: 0.1,
            ema_alpha: 0.3,
        }
    }
}

impl ReputationConfig {
    /// Configuration for strict reputation requirements
    pub fn strict() -> Self {
        Self {
            trust_threshold: 0.85,
            bad_peer_threshold: 0.4,
            transfer_success_weight: 0.5,
            latency_weight: 0.2,
            protocol_compliance_weight: 0.2,
            uptime_weight: 0.1,
            ema_alpha: 0.4,
            ..Default::default()
        }
    }

    /// Configuration for lenient reputation requirements
    pub fn lenient() -> Self {
        Self {
            trust_threshold: 0.5,
            bad_peer_threshold: 0.2,
            transfer_success_weight: 0.3,
            latency_weight: 0.2,
            protocol_compliance_weight: 0.1,
            uptime_weight: 0.4,
            ema_alpha: 0.2,
            ..Default::default()
        }
    }

    /// Configuration optimized for performance-critical applications
    pub fn performance_focused() -> Self {
        Self {
            trust_threshold: 0.75,
            bad_peer_threshold: 0.35,
            transfer_success_weight: 0.3,
            latency_weight: 0.5,
            protocol_compliance_weight: 0.1,
            uptime_weight: 0.1,
            ema_alpha: 0.35,
            ..Default::default()
        }
    }
}

/// Reputation score for a peer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationScore {
    /// Transfer success rate (0.0-1.0)
    pub transfer_success_rate: f64,

    /// Latency score (0.0-1.0, higher is better)
    pub latency_score: f64,

    /// Protocol compliance score (0.0-1.0)
    pub protocol_compliance_score: f64,

    /// Uptime score (0.0-1.0)
    pub uptime_score: f64,

    /// Number of successful transfers
    pub successful_transfers: u64,

    /// Number of failed transfers
    pub failed_transfers: u64,

    /// Number of protocol violations
    pub protocol_violations: u64,

    /// Average latency in milliseconds
    pub average_latency_ms: u64,

    /// Total uptime duration (in seconds, serializable)
    #[serde(
        serialize_with = "serialize_duration",
        deserialize_with = "deserialize_duration"
    )]
    pub total_uptime: Duration,

    /// Last seen timestamp (skipped in serialization)
    #[serde(skip, default = "Instant::now")]
    pub last_seen: Instant,

    /// First seen timestamp (skipped in serialization)
    #[serde(skip, default = "Instant::now")]
    pub first_seen: Instant,
}

fn serialize_duration<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_u64(duration.as_secs())
}

fn deserialize_duration<'de, D>(deserializer: D) -> Result<Duration, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let secs = u64::deserialize(deserializer)?;
    Ok(Duration::from_secs(secs))
}

impl Default for ReputationScore {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            transfer_success_rate: 1.0, // Start with perfect score
            latency_score: 1.0,
            protocol_compliance_score: 1.0,
            uptime_score: 1.0,
            successful_transfers: 0,
            failed_transfers: 0,
            protocol_violations: 0,
            average_latency_ms: 0,
            total_uptime: Duration::from_secs(0),
            last_seen: now,
            first_seen: now,
        }
    }
}

impl ReputationScore {
    /// Calculate overall reputation score using weighted average
    pub fn overall_score(&self, config: &ReputationConfig) -> f64 {
        let transfer_score = self.transfer_success_rate * config.transfer_success_weight;
        let latency_score_weighted = self.latency_score * config.latency_weight;
        let protocol_score = self.protocol_compliance_score * config.protocol_compliance_weight;
        let uptime_score_weighted = self.uptime_score * config.uptime_weight;

        transfer_score + latency_score_weighted + protocol_score + uptime_score_weighted
    }

    /// Update transfer success rate with exponential moving average
    fn update_transfer_success_rate(&mut self, success: bool, alpha: f64) {
        let new_value = if success { 1.0 } else { 0.0 };
        self.transfer_success_rate = alpha * new_value + (1.0 - alpha) * self.transfer_success_rate;

        if success {
            self.successful_transfers = self.successful_transfers.saturating_add(1);
        } else {
            self.failed_transfers = self.failed_transfers.saturating_add(1);
        }
    }

    /// Update latency score based on new latency measurement
    fn update_latency_score(&mut self, latency_ms: u64, alpha: f64) {
        // Calculate EMA of latency
        let current_avg = self.average_latency_ms as f64;
        let new_avg = alpha * (latency_ms as f64) + (1.0 - alpha) * current_avg;
        self.average_latency_ms = new_avg as u64;

        // Convert latency to score (lower is better)
        // Score approaches 0 as latency approaches 1000ms, score = 1 at 0ms
        let normalized_latency = (latency_ms as f64).min(1000.0) / 1000.0;
        self.latency_score = 1.0 - normalized_latency;
    }

    /// Record a protocol violation
    fn record_protocol_violation(&mut self, alpha: f64) {
        self.protocol_violations = self.protocol_violations.saturating_add(1);

        // Decrease protocol compliance score
        self.protocol_compliance_score *= 1.0 - alpha;
    }

    /// Update uptime tracking
    fn update_uptime(&mut self, connected_duration: Duration) {
        self.total_uptime += connected_duration;

        // Calculate uptime score based on total time known vs time connected
        let known_duration = self.last_seen.duration_since(self.first_seen);
        if known_duration.as_secs() > 0 {
            self.uptime_score =
                (self.total_uptime.as_secs_f64() / known_duration.as_secs_f64()).min(1.0);
        }
    }

    /// Apply time-based decay to scores
    fn apply_decay(&mut self, decay_rate: f64) {
        let decay_factor = 1.0 - decay_rate;
        self.transfer_success_rate *= decay_factor;
        self.latency_score *= decay_factor;
        self.protocol_compliance_score *= decay_factor;
        self.uptime_score *= decay_factor;
    }
}

/// Reputation event types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReputationEvent {
    /// Successful data transfer
    SuccessfulTransfer,
    /// Failed data transfer
    FailedTransfer,
    /// Low latency response
    LowLatency,
    /// High latency response
    HighLatency,
    /// Protocol violation detected
    ProtocolViolation,
    /// Peer disconnected gracefully
    GracefulDisconnect,
    /// Peer disconnected unexpectedly
    UnexpectedDisconnect,
}

/// Reputation manager for tracking peer reputations
pub struct ReputationManager {
    config: ReputationConfig,
    reputations: DashMap<PeerId, ReputationScore>,
    stats: parking_lot::RwLock<ReputationStats>,
}

impl ReputationManager {
    /// Create a new reputation manager
    pub fn new(config: ReputationConfig) -> Self {
        Self {
            config,
            reputations: DashMap::new(),
            stats: parking_lot::RwLock::new(ReputationStats::default()),
        }
    }

    /// Get reputation score for a peer
    pub fn get_reputation(&self, peer_id: &PeerId) -> Option<ReputationScore> {
        self.reputations.get(peer_id).map(|entry| entry.clone())
    }

    /// Check if a peer is trusted based on their reputation
    pub fn is_trusted(&self, peer_id: &PeerId) -> bool {
        self.reputations
            .get(peer_id)
            .map(|score| score.overall_score(&self.config) >= self.config.trust_threshold)
            .unwrap_or(false)
    }

    /// Check if a peer has a bad reputation
    pub fn is_bad_peer(&self, peer_id: &PeerId) -> bool {
        self.reputations
            .get(peer_id)
            .map(|score| score.overall_score(&self.config) < self.config.bad_peer_threshold)
            .unwrap_or(false)
    }

    /// Get list of trusted peers
    pub fn get_trusted_peers(&self) -> Vec<PeerId> {
        self.reputations
            .iter()
            .filter(|entry| {
                entry.value().overall_score(&self.config) >= self.config.trust_threshold
            })
            .map(|entry| *entry.key())
            .collect()
    }

    /// Get list of bad peers
    pub fn get_bad_peers(&self) -> Vec<PeerId> {
        self.reputations
            .iter()
            .filter(|entry| {
                entry.value().overall_score(&self.config) < self.config.bad_peer_threshold
            })
            .map(|entry| *entry.key())
            .collect()
    }

    /// Record a successful transfer
    pub fn record_successful_transfer(&mut self, peer_id: &PeerId, _bytes: u64) {
        let mut score = self.reputations.entry(*peer_id).or_default();
        score.update_transfer_success_rate(true, self.config.ema_alpha);
        score.last_seen = Instant::now();

        let mut stats = self.stats.write();
        stats.successful_events += 1;
    }

    /// Record a failed transfer
    pub fn record_failed_transfer(&mut self, peer_id: &PeerId) {
        let mut score = self.reputations.entry(*peer_id).or_default();
        score.update_transfer_success_rate(false, self.config.ema_alpha);
        score.last_seen = Instant::now();

        let mut stats = self.stats.write();
        stats.failed_events += 1;
    }

    /// Record low latency response
    pub fn record_low_latency(&mut self, peer_id: &PeerId, latency_ms: u64) {
        let mut score = self.reputations.entry(*peer_id).or_default();
        score.update_latency_score(latency_ms, self.config.ema_alpha);
        score.last_seen = Instant::now();

        let mut stats = self.stats.write();
        stats.latency_updates += 1;
    }

    /// Record a protocol violation
    pub fn record_protocol_violation(&mut self, peer_id: &PeerId) {
        let mut score = self.reputations.entry(*peer_id).or_default();
        score.record_protocol_violation(self.config.ema_alpha);
        score.last_seen = Instant::now();

        let mut stats = self.stats.write();
        stats.protocol_violations += 1;
    }

    /// Update uptime for a peer
    pub fn update_uptime(&mut self, peer_id: &PeerId, duration: Duration) {
        let mut score = self.reputations.entry(*peer_id).or_default();
        score.update_uptime(duration);
        score.last_seen = Instant::now();

        let mut stats = self.stats.write();
        stats.uptime_updates += 1;
    }

    /// Record a reputation event
    pub fn record_event(&mut self, peer_id: &PeerId, event: ReputationEvent) {
        match event {
            ReputationEvent::SuccessfulTransfer => self.record_successful_transfer(peer_id, 0),
            ReputationEvent::FailedTransfer => self.record_failed_transfer(peer_id),
            ReputationEvent::LowLatency => self.record_low_latency(peer_id, 50),
            ReputationEvent::HighLatency => self.record_low_latency(peer_id, 500),
            ReputationEvent::ProtocolViolation => self.record_protocol_violation(peer_id),
            ReputationEvent::GracefulDisconnect => {
                // Maintain current score, just update last_seen
                if let Some(mut score) = self.reputations.get_mut(peer_id) {
                    score.last_seen = Instant::now();
                }
            }
            ReputationEvent::UnexpectedDisconnect => {
                // Penalize slightly for unexpected disconnect
                self.record_failed_transfer(peer_id);
            }
        }
    }

    /// Apply time-based decay to all reputation scores
    pub fn apply_decay(&mut self) {
        for mut entry in self.reputations.iter_mut() {
            entry.value_mut().apply_decay(self.config.score_decay_rate);
        }
    }

    /// Remove stale peer reputations based on retention period
    pub fn cleanup_stale(&mut self) -> usize {
        let retention = self.config.retention_period;
        let now = Instant::now();
        let mut removed = 0;

        self.reputations.retain(|_, score| {
            let should_keep = now.duration_since(score.last_seen) < retention;
            if !should_keep {
                removed += 1;
            }
            should_keep
        });

        if removed > 0 {
            let mut stats = self.stats.write();
            stats.peers_removed += removed as u64;
        }

        removed
    }

    /// Get the number of tracked peers
    pub fn tracked_peer_count(&self) -> usize {
        self.reputations.len()
    }

    /// Get reputation statistics
    pub fn stats(&self) -> ReputationStats {
        self.stats.read().clone()
    }

    /// Clear all reputation data
    pub fn clear(&mut self) {
        self.reputations.clear();
        *self.stats.write() = ReputationStats::default();
    }
}

/// Statistics about reputation tracking
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReputationStats {
    /// Number of successful events recorded
    pub successful_events: u64,

    /// Number of failed events recorded
    pub failed_events: u64,

    /// Number of protocol violations recorded
    pub protocol_violations: u64,

    /// Number of latency updates
    pub latency_updates: u64,

    /// Number of uptime updates
    pub uptime_updates: u64,

    /// Number of peers removed due to staleness
    pub peers_removed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reputation_config_presets() {
        let strict = ReputationConfig::strict();
        assert!(strict.trust_threshold > ReputationConfig::default().trust_threshold);

        let lenient = ReputationConfig::lenient();
        assert!(lenient.trust_threshold < ReputationConfig::default().trust_threshold);

        let perf = ReputationConfig::performance_focused();
        assert!(perf.latency_weight > ReputationConfig::default().latency_weight);
    }

    #[test]
    fn test_reputation_score_default() {
        let score = ReputationScore::default();
        assert_eq!(score.transfer_success_rate, 1.0);
        assert_eq!(score.successful_transfers, 0);
        assert_eq!(score.failed_transfers, 0);
    }

    #[test]
    fn test_reputation_score_overall() {
        let config = ReputationConfig::default();
        let score = ReputationScore::default();

        let overall = score.overall_score(&config);
        assert!(overall > 0.9); // Should be close to 1.0 with default values
    }

    #[test]
    fn test_successful_transfer_updates() {
        let config = ReputationConfig::default();
        let mut manager = ReputationManager::new(config);
        let peer = PeerId::random();

        manager.record_successful_transfer(&peer, 1024);

        let score = manager.get_reputation(&peer).unwrap();
        assert_eq!(score.successful_transfers, 1);
        assert_eq!(score.failed_transfers, 0);
    }

    #[test]
    fn test_failed_transfer_updates() {
        let config = ReputationConfig::default();
        let mut manager = ReputationManager::new(config);
        let peer = PeerId::random();

        manager.record_failed_transfer(&peer);

        let score = manager.get_reputation(&peer).unwrap();
        assert_eq!(score.failed_transfers, 1);
        assert!(score.transfer_success_rate < 1.0);
    }

    #[test]
    fn test_latency_scoring() {
        let config = ReputationConfig::default();
        let mut manager = ReputationManager::new(config);
        let peer = PeerId::random();

        // Record low latency
        manager.record_low_latency(&peer, 50);
        let score = manager.get_reputation(&peer).unwrap();
        assert!(score.latency_score > 0.9);

        // Record high latency
        manager.record_low_latency(&peer, 900);
        let score = manager.get_reputation(&peer).unwrap();
        assert!(score.latency_score < 0.5);
    }

    #[test]
    fn test_protocol_violation() {
        let config = ReputationConfig::default();
        let mut manager = ReputationManager::new(config);
        let peer = PeerId::random();

        manager.record_protocol_violation(&peer);

        let score = manager.get_reputation(&peer).unwrap();
        assert_eq!(score.protocol_violations, 1);
        assert!(score.protocol_compliance_score < 1.0);
    }

    #[test]
    fn test_is_trusted() {
        let config = ReputationConfig::default();
        let mut manager = ReputationManager::new(config);
        let peer = PeerId::random();

        // New peer starts trusted
        manager.record_successful_transfer(&peer, 1024);
        assert!(manager.is_trusted(&peer));

        // Many failures reduce trust
        for _ in 0..10 {
            manager.record_failed_transfer(&peer);
        }
        assert!(!manager.is_trusted(&peer));
    }

    #[test]
    fn test_is_bad_peer() {
        // Use a config with higher transfer success weight and faster EMA
        let config = ReputationConfig {
            transfer_success_weight: 0.9, // Focus on transfer success
            ema_alpha: 0.5,               // Faster updates
            latency_weight: 0.05,
            protocol_compliance_weight: 0.025,
            uptime_weight: 0.025,
            ..Default::default()
        };

        let mut manager = ReputationManager::new(config);
        let peer = PeerId::random();

        // Record many failures
        for _ in 0..20 {
            manager.record_failed_transfer(&peer);
        }

        assert!(manager.is_bad_peer(&peer));
    }

    #[test]
    fn test_get_trusted_peers() {
        let config = ReputationConfig::default();
        let mut manager = ReputationManager::new(config);

        let peer1 = PeerId::random();
        let peer2 = PeerId::random();

        manager.record_successful_transfer(&peer1, 1024);
        manager.record_successful_transfer(&peer2, 1024);

        let trusted = manager.get_trusted_peers();
        assert_eq!(trusted.len(), 2);
    }

    #[test]
    fn test_get_bad_peers() {
        // Use a config with higher transfer success weight and faster EMA
        let config = ReputationConfig {
            transfer_success_weight: 0.9,
            ema_alpha: 0.5,
            latency_weight: 0.05,
            protocol_compliance_weight: 0.025,
            uptime_weight: 0.025,
            ..Default::default()
        };

        let mut manager = ReputationManager::new(config);

        let peer = PeerId::random();

        for _ in 0..20 {
            manager.record_failed_transfer(&peer);
        }

        let bad_peers = manager.get_bad_peers();
        assert_eq!(bad_peers.len(), 1);
    }

    #[test]
    fn test_uptime_tracking() {
        let config = ReputationConfig::default();
        let mut manager = ReputationManager::new(config);
        let peer = PeerId::random();

        manager.update_uptime(&peer, Duration::from_secs(3600));

        let score = manager.get_reputation(&peer).unwrap();
        assert_eq!(score.total_uptime.as_secs(), 3600);
    }

    #[test]
    fn test_reputation_events() {
        let config = ReputationConfig::default();
        let mut manager = ReputationManager::new(config);
        let peer = PeerId::random();

        manager.record_event(&peer, ReputationEvent::SuccessfulTransfer);
        manager.record_event(&peer, ReputationEvent::LowLatency);
        manager.record_event(&peer, ReputationEvent::ProtocolViolation);

        let score = manager.get_reputation(&peer).unwrap();
        assert!(score.successful_transfers > 0);
        assert!(score.protocol_violations > 0);
    }

    #[test]
    fn test_stats_tracking() {
        let config = ReputationConfig::default();
        let mut manager = ReputationManager::new(config);
        let peer = PeerId::random();

        manager.record_successful_transfer(&peer, 1024);
        manager.record_failed_transfer(&peer);
        manager.record_protocol_violation(&peer);

        let stats = manager.stats();
        assert_eq!(stats.successful_events, 1);
        assert_eq!(stats.failed_events, 1);
        assert_eq!(stats.protocol_violations, 1);
    }

    #[test]
    fn test_tracked_peer_count() {
        let config = ReputationConfig::default();
        let mut manager = ReputationManager::new(config);

        for _ in 0..5 {
            let peer = PeerId::random();
            manager.record_successful_transfer(&peer, 1024);
        }

        assert_eq!(manager.tracked_peer_count(), 5);
    }

    #[test]
    fn test_clear() {
        let config = ReputationConfig::default();
        let mut manager = ReputationManager::new(config);
        let peer = PeerId::random();

        manager.record_successful_transfer(&peer, 1024);
        assert_eq!(manager.tracked_peer_count(), 1);

        manager.clear();
        assert_eq!(manager.tracked_peer_count(), 0);
    }
}
