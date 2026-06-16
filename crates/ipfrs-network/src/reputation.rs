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

// ============================================================================
// PeerReputationTracker — event-driven peer scoring system
// ============================================================================

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

/// Classification tier for a peer based on their reputation score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PeerTier {
    /// Score <= -50: peer is banned and should not be used.
    Banned,
    /// -50 < score <= -20: peer is untrusted and should be de-prioritized.
    Untrusted,
    /// -20 < score < 20: peer has neutral standing.
    Neutral,
    /// 20 <= score < 50: peer is good and should be preferred.
    Good,
    /// score >= 50: peer is fully trusted.
    Trusted,
}

/// An event that affects a peer's reputation score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PeerReputationEvent {
    /// Peer successfully delivered a block.
    BlockDelivered {
        /// Round-trip latency in milliseconds.
        latency_ms: u64,
    },
    /// Peer failed to deliver a requested block.
    BlockFailed,
    /// Peer sent a block whose CID or content did not match what was requested.
    InvalidBlock,
    /// Peer returned a valid DHT response.
    DhtResponseValid {
        /// Latency of the DHT response in milliseconds.
        latency_ms: u64,
    },
    /// Peer's DHT query returned no results.
    DhtResponseEmpty,
    /// Connection to the peer failed.
    ConnectionFailed,
    /// Peer violated the protocol (severe).
    ProtocolViolation,
}

impl PeerReputationEvent {
    /// Returns the signed score delta this event contributes.
    ///
    /// Deltas:
    /// - `BlockDelivered`: +1.0, +0.5 bonus when `latency_ms < 100`
    /// - `BlockFailed`: -1.0
    /// - `InvalidBlock`: -10.0
    /// - `DhtResponseValid`: +0.5, +0.25 bonus when `latency_ms < 200`
    /// - `DhtResponseEmpty`: -0.25
    /// - `ConnectionFailed`: -2.0
    /// - `ProtocolViolation`: -15.0
    pub fn score_delta(&self) -> f64 {
        match self {
            PeerReputationEvent::BlockDelivered { latency_ms } => {
                let bonus = if *latency_ms < 100 { 0.5 } else { 0.0 };
                1.0 + bonus
            }
            PeerReputationEvent::BlockFailed => -1.0,
            PeerReputationEvent::InvalidBlock => -10.0,
            PeerReputationEvent::DhtResponseValid { latency_ms } => {
                let bonus = if *latency_ms < 200 { 0.25 } else { 0.0 };
                0.5 + bonus
            }
            PeerReputationEvent::DhtResponseEmpty => -0.25,
            PeerReputationEvent::ConnectionFailed => -2.0,
            PeerReputationEvent::ProtocolViolation => -15.0,
        }
    }
}

/// Per-peer reputation state.
#[derive(Debug, Clone)]
pub struct PeerReputation {
    /// String identifier for the peer (e.g. libp2p PeerId string representation).
    pub peer_id: String,
    /// Current reputation score, clamped to `[-100.0, 100.0]`. Starts at 0.0.
    pub score: f64,
    /// Total number of events recorded for this peer.
    pub event_count: u64,
    /// Instant of the last recorded event, or `None` if no events yet.
    pub last_event_at: Option<Instant>,
    /// Instant at which this record was created.
    pub created_at: Instant,
}

impl PeerReputation {
    /// Creates a fresh reputation record for the given peer.
    pub fn new(peer_id: impl Into<String>) -> Self {
        Self {
            peer_id: peer_id.into(),
            score: 0.0,
            event_count: 0,
            last_event_at: None,
            created_at: Instant::now(),
        }
    }

    /// Applies the score delta for `event`, clamps the result, and increments the event counter.
    pub fn record_event(&mut self, event: PeerReputationEvent) {
        self.score = (self.score + event.score_delta()).clamp(-100.0, 100.0);
        self.event_count = self.event_count.saturating_add(1);
        self.last_event_at = Some(Instant::now());
    }

    /// Returns `true` when the peer's score has fallen to or below -50.0.
    pub fn is_banned(&self) -> bool {
        self.score <= -50.0
    }

    /// Returns `true` when the peer's score has reached or exceeded +50.0.
    pub fn is_trusted(&self) -> bool {
        self.score >= 50.0
    }

    /// Applies time-based exponential decay toward 0.
    ///
    /// The formula used is:
    /// ```text
    /// score *= 1.0 - 0.001 * elapsed.as_secs_f64().min(100.0)
    /// ```
    /// This approximates a decay of 0.1 % per second, capped at 10 % for any
    /// single call regardless of how much real time has elapsed.
    pub fn decay(&mut self, elapsed: Duration) {
        let factor = 1.0 - 0.001 * elapsed.as_secs_f64().min(100.0);
        self.score *= factor;
    }

    /// Returns the `PeerTier` that corresponds to the current score.
    pub fn tier(&self) -> PeerTier {
        if self.score <= -50.0 {
            PeerTier::Banned
        } else if self.score <= -20.0 {
            PeerTier::Untrusted
        } else if self.score < 20.0 {
            PeerTier::Neutral
        } else if self.score < 50.0 {
            PeerTier::Good
        } else {
            PeerTier::Trusted
        }
    }
}

/// Atomic snapshot of aggregate statistics collected by [`PeerReputationTracker`].
#[derive(Debug, Default)]
pub struct PeerReputationStats {
    /// Total number of events processed across all peers.
    pub total_events: AtomicU64,
    /// Number of times a peer's score dropped to or below the ban threshold.
    pub total_bans: AtomicU64,
    /// Number of times a previously-banned peer's score recovered above the ban threshold.
    pub total_unbans: AtomicU64,
}

impl PeerReputationStats {
    /// Creates a new zeroed stats instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a plain-data snapshot of the current atomic values.
    pub fn snapshot(&self) -> PeerReputationStatsSnapshot {
        PeerReputationStatsSnapshot {
            total_events: self.total_events.load(Ordering::Relaxed),
            total_bans: self.total_bans.load(Ordering::Relaxed),
            total_unbans: self.total_unbans.load(Ordering::Relaxed),
        }
    }
}

/// Non-atomic snapshot of [`PeerReputationStats`] values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerReputationStatsSnapshot {
    /// Total number of events processed across all peers.
    pub total_events: u64,
    /// Number of times a peer's score dropped to or below the ban threshold.
    pub total_bans: u64,
    /// Number of times a previously-banned peer's score recovered above the ban threshold.
    pub total_unbans: u64,
}

/// Thread-safe tracker that maintains per-peer reputation scores and exposes
/// aggregate statistics via `AtomicU64` counters.
///
/// # Thread Safety
///
/// All mutating operations acquire the internal `RwLock<HashMap>` write lock,
/// while read-only queries acquire only the read lock.  The aggregate stats are
/// updated via relaxed atomic stores so they never block readers.
pub struct PeerReputationTracker {
    peers: RwLock<HashMap<String, PeerReputation>>,
    stats: PeerReputationStats,
}

impl Default for PeerReputationTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl PeerReputationTracker {
    /// Creates a new, empty tracker.
    pub fn new() -> Self {
        Self {
            peers: RwLock::new(HashMap::new()),
            stats: PeerReputationStats::new(),
        }
    }

    /// Records an event for the given peer.
    ///
    /// If the peer is unknown a new record initialised to score 0 is created
    /// automatically.  The function also updates `total_bans` / `total_unbans`
    /// counters when the event crosses the ban threshold.
    pub fn record_event(&self, peer_id: &str, event: PeerReputationEvent) {
        let mut guard = self.peers.write().unwrap_or_else(|e| e.into_inner());

        let peer = guard
            .entry(peer_id.to_owned())
            .or_insert_with(|| PeerReputation::new(peer_id));

        let was_banned = peer.is_banned();
        peer.record_event(event);
        let now_banned = peer.is_banned();

        self.stats.total_events.fetch_add(1, Ordering::Relaxed);

        if !was_banned && now_banned {
            self.stats.total_bans.fetch_add(1, Ordering::Relaxed);
        } else if was_banned && !now_banned {
            self.stats.total_unbans.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Returns the current score for `peer_id`, or 0.0 if the peer is unknown.
    pub fn get_score(&self, peer_id: &str) -> f64 {
        let guard = self.peers.read().unwrap_or_else(|e| e.into_inner());
        guard.get(peer_id).map(|p| p.score).unwrap_or(0.0)
    }

    /// Returns the [`PeerTier`] for `peer_id`.
    ///
    /// Unknown peers are treated as having a score of 0 (`PeerTier::Neutral`).
    pub fn get_tier(&self, peer_id: &str) -> PeerTier {
        let guard = self.peers.read().unwrap_or_else(|e| e.into_inner());
        guard
            .get(peer_id)
            .map(|p| p.tier())
            .unwrap_or(PeerTier::Neutral)
    }

    /// Returns `true` when the peer is banned (score <= -50).
    pub fn is_banned(&self, peer_id: &str) -> bool {
        let guard = self.peers.read().unwrap_or_else(|e| e.into_inner());
        guard.get(peer_id).map(|p| p.is_banned()).unwrap_or(false)
    }

    /// Returns the peer IDs of all currently-banned peers.
    pub fn banned_peers(&self) -> Vec<String> {
        let guard = self.peers.read().unwrap_or_else(|e| e.into_inner());
        guard
            .values()
            .filter(|p| p.is_banned())
            .map(|p| p.peer_id.clone())
            .collect()
    }

    /// Returns the peer IDs of all currently-trusted peers (score >= 50).
    pub fn trusted_peers(&self) -> Vec<String> {
        let guard = self.peers.read().unwrap_or_else(|e| e.into_inner());
        guard
            .values()
            .filter(|p| p.is_trusted())
            .map(|p| p.peer_id.clone())
            .collect()
    }

    /// Returns the top `n` peers by score (descending), as `(peer_id, score)` pairs.
    pub fn top_peers(&self, n: usize) -> Vec<(String, f64)> {
        let guard = self.peers.read().unwrap_or_else(|e| e.into_inner());

        let mut entries: Vec<(String, f64)> = guard
            .values()
            .map(|p| (p.peer_id.clone(), p.score))
            .collect();

        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entries.truncate(n);
        entries
    }

    /// Applies time-based score decay to every tracked peer.
    pub fn apply_decay(&self, elapsed: Duration) {
        let mut guard = self.peers.write().unwrap_or_else(|e| e.into_inner());
        for peer in guard.values_mut() {
            peer.decay(elapsed);
        }
    }

    /// Removes the reputation record for `peer_id`.  Does nothing if the peer
    /// is unknown.
    pub fn remove_peer(&self, peer_id: &str) {
        let mut guard = self.peers.write().unwrap_or_else(|e| e.into_inner());
        guard.remove(peer_id);
    }

    /// Returns the total number of peers currently being tracked.
    pub fn peer_count(&self) -> usize {
        let guard = self.peers.read().unwrap_or_else(|e| e.into_inner());
        guard.len()
    }

    /// Returns a reference to the aggregate statistics.
    pub fn stats(&self) -> &PeerReputationStats {
        &self.stats
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tracker_tests {
    use super::{PeerReputation, PeerReputationEvent, PeerReputationTracker, PeerTier};
    use std::time::Duration;

    fn make_tracker() -> PeerReputationTracker {
        PeerReputationTracker::new()
    }

    // -------------------------------------------------------------------------
    // PeerReputationEvent::score_delta
    // -------------------------------------------------------------------------

    #[test]
    fn test_block_delivered_high_latency_delta() {
        let delta = PeerReputationEvent::BlockDelivered { latency_ms: 200 }.score_delta();
        assert!((delta - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_block_delivered_low_latency_bonus() {
        let delta = PeerReputationEvent::BlockDelivered { latency_ms: 50 }.score_delta();
        assert!((delta - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_block_failed_delta() {
        assert!((PeerReputationEvent::BlockFailed.score_delta() - (-1.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_invalid_block_delta() {
        assert!((PeerReputationEvent::InvalidBlock.score_delta() - (-10.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dht_valid_slow_delta() {
        let delta = PeerReputationEvent::DhtResponseValid { latency_ms: 500 }.score_delta();
        assert!((delta - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dht_valid_fast_bonus() {
        let delta = PeerReputationEvent::DhtResponseValid { latency_ms: 100 }.score_delta();
        assert!((delta - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dht_empty_delta() {
        assert!(
            (PeerReputationEvent::DhtResponseEmpty.score_delta() - (-0.25)).abs() < f64::EPSILON
        );
    }

    #[test]
    fn test_connection_failed_delta() {
        assert!(
            (PeerReputationEvent::ConnectionFailed.score_delta() - (-2.0)).abs() < f64::EPSILON
        );
    }

    #[test]
    fn test_protocol_violation_delta() {
        assert!(
            (PeerReputationEvent::ProtocolViolation.score_delta() - (-15.0)).abs() < f64::EPSILON
        );
    }

    // -------------------------------------------------------------------------
    // PeerReputation struct
    // -------------------------------------------------------------------------

    #[test]
    fn test_score_starts_at_zero() {
        let rep = PeerReputation::new("peer-A");
        assert_eq!(rep.score, 0.0);
        assert_eq!(rep.event_count, 0);
        assert!(rep.last_event_at.is_none());
    }

    #[test]
    fn test_score_clamped_at_positive_100() {
        let mut rep = PeerReputation::new("peer-B");
        // Apply many positive events: each BlockDelivered(low latency) = +1.5
        for _ in 0..200 {
            rep.record_event(PeerReputationEvent::BlockDelivered { latency_ms: 10 });
        }
        assert!(
            rep.score <= 100.0,
            "score should not exceed 100.0, got {}",
            rep.score
        );
        assert!((rep.score - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_score_clamped_at_negative_100() {
        let mut rep = PeerReputation::new("peer-C");
        // ProtocolViolation = -15 each; 7 times = -105 unclamped
        for _ in 0..8 {
            rep.record_event(PeerReputationEvent::ProtocolViolation);
        }
        assert!(rep.score >= -100.0, "score must not go below -100.0");
        assert!((rep.score - (-100.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_multiple_events_accumulate() {
        let mut rep = PeerReputation::new("peer-D");
        rep.record_event(PeerReputationEvent::BlockDelivered { latency_ms: 50 }); // +1.5
        rep.record_event(PeerReputationEvent::DhtResponseValid { latency_ms: 100 }); // +0.75
        rep.record_event(PeerReputationEvent::BlockFailed); // -1.0
                                                            // Expected: 1.5 + 0.75 - 1.0 = 1.25
        assert!((rep.score - 1.25).abs() < 1e-9);
        assert_eq!(rep.event_count, 3);
        assert!(rep.last_event_at.is_some());
    }

    #[test]
    fn test_is_banned_threshold() {
        let mut rep = PeerReputation::new("peer-E");
        // Drive score to exactly -50
        rep.score = -50.0;
        assert!(rep.is_banned());
        rep.score = -49.9;
        assert!(!rep.is_banned());
        rep.score = -100.0;
        assert!(rep.is_banned());
    }

    #[test]
    fn test_is_trusted_threshold() {
        let mut rep = PeerReputation::new("peer-F");
        rep.score = 50.0;
        assert!(rep.is_trusted());
        rep.score = 49.9;
        assert!(!rep.is_trusted());
        rep.score = 100.0;
        assert!(rep.is_trusted());
    }

    #[test]
    fn test_peer_tier_classification() {
        let mut rep = PeerReputation::new("peer-G");

        rep.score = -60.0;
        assert_eq!(rep.tier(), PeerTier::Banned);

        rep.score = -50.0;
        assert_eq!(rep.tier(), PeerTier::Banned);

        rep.score = -35.0;
        assert_eq!(rep.tier(), PeerTier::Untrusted);

        rep.score = -20.0;
        assert_eq!(rep.tier(), PeerTier::Untrusted);

        rep.score = -0.1;
        assert_eq!(rep.tier(), PeerTier::Neutral);

        rep.score = 0.0;
        assert_eq!(rep.tier(), PeerTier::Neutral);

        rep.score = 10.0;
        assert_eq!(rep.tier(), PeerTier::Neutral);

        rep.score = 20.0;
        assert_eq!(rep.tier(), PeerTier::Good);

        rep.score = 49.9;
        assert_eq!(rep.tier(), PeerTier::Good);

        rep.score = 50.0;
        assert_eq!(rep.tier(), PeerTier::Trusted);

        rep.score = 100.0;
        assert_eq!(rep.tier(), PeerTier::Trusted);
    }

    #[test]
    fn test_decay_reduces_positive_score() {
        let mut rep = PeerReputation::new("peer-H");
        rep.score = 50.0;
        let before = rep.score;
        rep.decay(Duration::from_secs(10));
        assert!(rep.score < before, "decay should reduce a positive score");
        assert!(
            rep.score > 0.0,
            "score should remain positive after small decay"
        );
    }

    #[test]
    fn test_decay_moves_negative_score_toward_zero() {
        let mut rep = PeerReputation::new("peer-I");
        rep.score = -50.0;
        let before = rep.score;
        rep.decay(Duration::from_secs(10));
        // score *= (1.0 - 0.001 * 10) = 0.99  →  -49.5
        assert!(
            rep.score > before,
            "decay should move a negative score toward 0"
        );
        assert!(rep.score < 0.0, "score should still be negative");
    }

    // -------------------------------------------------------------------------
    // PeerReputationTracker
    // -------------------------------------------------------------------------

    #[test]
    fn test_unknown_peer_returns_zero() {
        let tracker = make_tracker();
        assert_eq!(tracker.get_score("nobody"), 0.0);
        assert_eq!(tracker.get_tier("nobody"), PeerTier::Neutral);
        assert!(!tracker.is_banned("nobody"));
    }

    #[test]
    fn test_record_event_creates_peer() {
        let tracker = make_tracker();
        tracker.record_event(
            "alice",
            PeerReputationEvent::BlockDelivered { latency_ms: 50 },
        );
        assert_eq!(tracker.peer_count(), 1);
        assert!(tracker.get_score("alice") > 0.0);
    }

    #[test]
    fn test_banned_peers_list() {
        let tracker = make_tracker();
        // ban "bad-peer" with protocol violations
        for _ in 0..4 {
            tracker.record_event("bad-peer", PeerReputationEvent::ProtocolViolation);
        }
        // keep "good-peer" neutral
        tracker.record_event(
            "good-peer",
            PeerReputationEvent::BlockDelivered { latency_ms: 50 },
        );

        let banned = tracker.banned_peers();
        assert!(banned.contains(&"bad-peer".to_owned()));
        assert!(!banned.contains(&"good-peer".to_owned()));
    }

    #[test]
    fn test_trusted_peers_list() {
        let tracker = make_tracker();
        // give "star-peer" a score >= 50: need >= 34 BlockDelivered(high latency) at +1 each
        for _ in 0..55 {
            tracker.record_event(
                "star-peer",
                PeerReputationEvent::BlockDelivered { latency_ms: 200 },
            );
        }
        tracker.record_event("meh-peer", PeerReputationEvent::DhtResponseEmpty);

        let trusted = tracker.trusted_peers();
        assert!(trusted.contains(&"star-peer".to_owned()));
        assert!(!trusted.contains(&"meh-peer".to_owned()));
    }

    #[test]
    fn test_top_peers_ordering() {
        let tracker = make_tracker();

        for _ in 0..10 {
            tracker.record_event(
                "alpha",
                PeerReputationEvent::BlockDelivered { latency_ms: 50 },
            );
        }
        for _ in 0..5 {
            tracker.record_event(
                "beta",
                PeerReputationEvent::BlockDelivered { latency_ms: 50 },
            );
        }
        tracker.record_event("gamma", PeerReputationEvent::BlockFailed);

        let top = tracker.top_peers(3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, "alpha");
        assert!(top[0].1 >= top[1].1);
        assert!(top[1].1 >= top[2].1);
    }

    #[test]
    fn test_top_peers_fewer_than_n() {
        let tracker = make_tracker();
        tracker.record_event(
            "only-one",
            PeerReputationEvent::BlockDelivered { latency_ms: 200 },
        );
        let top = tracker.top_peers(10);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_apply_decay_all_peers() {
        let tracker = make_tracker();
        for _ in 0..30 {
            tracker.record_event(
                "peer-x",
                PeerReputationEvent::BlockDelivered { latency_ms: 200 },
            );
        }
        let score_before = tracker.get_score("peer-x");
        tracker.apply_decay(Duration::from_secs(60));
        let score_after = tracker.get_score("peer-x");
        assert!(
            score_after < score_before,
            "decay must reduce a positive score"
        );
    }

    #[test]
    fn test_remove_peer() {
        let tracker = make_tracker();
        tracker.record_event(
            "temp",
            PeerReputationEvent::BlockDelivered { latency_ms: 200 },
        );
        assert_eq!(tracker.peer_count(), 1);
        tracker.remove_peer("temp");
        assert_eq!(tracker.peer_count(), 0);
        // Removing unknown peer is a no-op
        tracker.remove_peer("nonexistent");
        assert_eq!(tracker.peer_count(), 0);
    }

    #[test]
    fn test_stats_total_events() {
        let tracker = make_tracker();
        tracker.record_event(
            "p1",
            PeerReputationEvent::BlockDelivered { latency_ms: 200 },
        );
        tracker.record_event("p1", PeerReputationEvent::BlockFailed);
        tracker.record_event("p2", PeerReputationEvent::DhtResponseEmpty);
        let snap = tracker.stats().snapshot();
        assert_eq!(snap.total_events, 3);
    }

    #[test]
    fn test_stats_ban_counter() {
        let tracker = make_tracker();
        // 4 × ProtocolViolation = -60 → ban triggered
        for _ in 0..4 {
            tracker.record_event("villain", PeerReputationEvent::ProtocolViolation);
        }
        let snap = tracker.stats().snapshot();
        assert_eq!(snap.total_bans, 1);
    }
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

        let score = manager
            .get_reputation(&peer)
            .expect("test: get_reputation should return Some after recording successful transfer");
        assert_eq!(score.successful_transfers, 1);
        assert_eq!(score.failed_transfers, 0);
    }

    #[test]
    fn test_failed_transfer_updates() {
        let config = ReputationConfig::default();
        let mut manager = ReputationManager::new(config);
        let peer = PeerId::random();

        manager.record_failed_transfer(&peer);

        let score = manager
            .get_reputation(&peer)
            .expect("test: get_reputation should return Some after recording failed transfer");
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
        let score = manager
            .get_reputation(&peer)
            .expect("test: get_reputation should return Some after recording low latency");
        assert!(score.latency_score > 0.9);

        // Record high latency
        manager.record_low_latency(&peer, 900);
        let score = manager
            .get_reputation(&peer)
            .expect("test: get_reputation should return Some after recording high latency");
        assert!(score.latency_score < 0.5);
    }

    #[test]
    fn test_protocol_violation() {
        let config = ReputationConfig::default();
        let mut manager = ReputationManager::new(config);
        let peer = PeerId::random();

        manager.record_protocol_violation(&peer);

        let score = manager
            .get_reputation(&peer)
            .expect("test: get_reputation should return Some after recording protocol violation");
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

        let score = manager
            .get_reputation(&peer)
            .expect("test: get_reputation should return Some after updating uptime");
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

        let score = manager
            .get_reputation(&peer)
            .expect("test: get_reputation should return Some after recording events");
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
