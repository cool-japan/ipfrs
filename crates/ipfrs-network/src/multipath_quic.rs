//! QUIC Multipath Support
//!
//! This module implements multipath QUIC connections, allowing a single connection
//! to use multiple network paths simultaneously for improved reliability, throughput,
//! and seamless network transitions.
//!
//! ## Features
//!
//! - **Multiple Paths**: Manage multiple network paths for a single connection
//! - **Path Quality Monitoring**: Track latency, bandwidth, and packet loss per path
//! - **Traffic Distribution**: Distribute traffic across paths based on quality and strategy
//! - **Path Migration**: Seamlessly migrate between paths when quality changes
//! - **Load Balancing**: Balance load across available paths for optimal throughput
//! - **Redundancy**: Send critical data on multiple paths for reliability
//!
//! ## Path Selection Strategies
//!
//! - **Round Robin**: Distribute traffic evenly across all paths
//! - **Quality Based**: Prefer paths with better quality metrics
//! - **Lowest Latency**: Always use the path with lowest latency
//! - **Highest Bandwidth**: Always use the path with highest bandwidth
//! - **Redundant**: Send data on all paths for maximum reliability
//!
//! ## Example
//!
//! ```rust,no_run
//! use ipfrs_network::multipath_quic::{MultipathQuicManager, MultipathConfig, PathSelectionStrategy};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create multipath manager with quality-based strategy
//! let config = MultipathConfig {
//!     max_paths: 4,
//!     strategy: PathSelectionStrategy::QualityBased,
//!     enable_redundancy: false,
//!     ..Default::default()
//! };
//!
//! let mut manager = MultipathQuicManager::new(config);
//!
//! // Paths will be automatically detected and managed
//! // Traffic will be distributed based on the configured strategy
//! # Ok(())
//! # }
//! ```

use dashmap::DashMap;
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tracing::{debug, info, warn};

/// Errors that can occur in multipath QUIC operations
#[derive(Debug, Error)]
pub enum MultipathError {
    #[error("No paths available")]
    NoPathsAvailable,

    #[error("Path not found: {0}")]
    PathNotFound(u64),

    #[error("Path quality too low: {0}")]
    PathQualityTooLow(f64),

    #[error("Maximum paths reached: {0}")]
    MaxPathsReached(usize),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Result type for multipath operations
pub type Result<T> = std::result::Result<T, MultipathError>;

/// Unique identifier for a network path
pub type PathId = u64;

/// Network path representing a single route to a peer
#[derive(Debug, Clone)]
pub struct NetworkPath {
    /// Unique identifier for this path
    pub id: PathId,

    /// Local socket address for this path
    pub local_addr: SocketAddr,

    /// Remote socket address for this path
    pub remote_addr: SocketAddr,

    /// Path state
    pub state: PathState,

    /// Quality metrics for this path
    pub quality: PathQuality,

    /// When this path was created
    pub created_at: Instant,

    /// When this path was last used
    pub last_used: Instant,

    /// Total bytes sent on this path
    pub bytes_sent: u64,

    /// Total bytes received on this path
    pub bytes_received: u64,
}

/// State of a network path
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathState {
    /// Path is being validated
    Validating,

    /// Path is active and ready to use
    Active,

    /// Path is standby (available but not preferred)
    Standby,

    /// Path is degraded (poor quality)
    Degraded,

    /// Path is failed and should not be used
    Failed,

    /// Path is being closed
    Closing,
}

/// Quality metrics for a network path
#[derive(Debug, Clone)]
pub struct PathQuality {
    /// Round-trip time (RTT) in milliseconds
    pub rtt_ms: f64,

    /// Bandwidth estimate in bytes/second
    pub bandwidth_bps: u64,

    /// Packet loss rate (0.0 - 1.0)
    pub loss_rate: f64,

    /// Jitter in milliseconds
    pub jitter_ms: f64,

    /// Overall quality score (0.0 - 1.0, higher is better)
    pub score: f64,

    /// Number of samples used for metrics
    pub sample_count: usize,

    /// Last time quality was updated
    pub last_updated: Instant,
}

impl Default for PathQuality {
    fn default() -> Self {
        Self {
            rtt_ms: 0.0,
            bandwidth_bps: 0,
            loss_rate: 0.0,
            jitter_ms: 0.0,
            score: 0.5, // Start with neutral score
            sample_count: 0,
            last_updated: Instant::now(),
        }
    }
}

impl PathQuality {
    /// Calculate overall quality score based on metrics
    pub fn calculate_score(&mut self) {
        // Weights for different metrics
        const RTT_WEIGHT: f64 = 0.3;
        const BANDWIDTH_WEIGHT: f64 = 0.3;
        const LOSS_WEIGHT: f64 = 0.3;
        const JITTER_WEIGHT: f64 = 0.1;

        // Normalize RTT (assume 0-500ms range, lower is better)
        let rtt_score = (1.0 - (self.rtt_ms / 500.0).clamp(0.0, 1.0)).max(0.0);

        // Normalize bandwidth (assume 0-100Mbps range, higher is better)
        let bandwidth_mbps = self.bandwidth_bps as f64 / 125_000.0;
        let bandwidth_score = (bandwidth_mbps / 100.0).clamp(0.0, 1.0);

        // Loss rate score (lower is better)
        let loss_score = (1.0 - self.loss_rate).max(0.0);

        // Jitter score (assume 0-50ms range, lower is better)
        let jitter_score = (1.0 - (self.jitter_ms / 50.0).clamp(0.0, 1.0)).max(0.0);

        // Calculate weighted score
        self.score = rtt_score * RTT_WEIGHT
            + bandwidth_score * BANDWIDTH_WEIGHT
            + loss_score * LOSS_WEIGHT
            + jitter_score * JITTER_WEIGHT;
    }

    /// Update quality metrics with exponential moving average
    pub fn update(&mut self, rtt_ms: f64, bandwidth_bps: u64, loss_rate: f64, jitter_ms: f64) {
        const ALPHA: f64 = 0.8; // Weight for new samples

        if self.sample_count == 0 {
            // First sample, use it directly
            self.rtt_ms = rtt_ms;
            self.bandwidth_bps = bandwidth_bps;
            self.loss_rate = loss_rate;
            self.jitter_ms = jitter_ms;
        } else {
            // Exponential moving average
            self.rtt_ms = self.rtt_ms * (1.0 - ALPHA) + rtt_ms * ALPHA;
            self.bandwidth_bps = ((self.bandwidth_bps as f64) * (1.0 - ALPHA)
                + (bandwidth_bps as f64) * ALPHA) as u64;
            self.loss_rate = self.loss_rate * (1.0 - ALPHA) + loss_rate * ALPHA;
            self.jitter_ms = self.jitter_ms * (1.0 - ALPHA) + jitter_ms * ALPHA;
        }

        self.sample_count += 1;
        self.last_updated = Instant::now();
        self.calculate_score();
    }
}

/// Strategy for selecting which path to use for traffic
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathSelectionStrategy {
    /// Distribute traffic evenly across all active paths
    RoundRobin,

    /// Prefer paths with better quality scores
    QualityBased,

    /// Always use the path with lowest latency
    LowestLatency,

    /// Always use the path with highest bandwidth
    HighestBandwidth,

    /// Send data on all paths for maximum reliability (redundant mode)
    Redundant,

    /// Distribute based on weighted round-robin using quality scores
    WeightedRoundRobin,
}

/// Configuration for multipath QUIC manager
#[derive(Debug, Clone)]
pub struct MultipathConfig {
    /// Maximum number of concurrent paths
    pub max_paths: usize,

    /// Path selection strategy
    pub strategy: PathSelectionStrategy,

    /// Enable redundant transmission for critical data
    pub enable_redundancy: bool,

    /// Minimum quality score to keep a path active (0.0 - 1.0)
    pub min_quality_threshold: f64,

    /// Interval for path quality monitoring
    pub quality_check_interval: Duration,

    /// Maximum age for a path without traffic before considering it stale
    pub path_idle_timeout: Duration,

    /// Enable automatic path migration based on quality
    pub enable_auto_migration: bool,

    /// Quality difference threshold to trigger migration (0.0 - 1.0)
    pub migration_quality_threshold: f64,
}

impl Default for MultipathConfig {
    fn default() -> Self {
        Self {
            max_paths: 4,
            strategy: PathSelectionStrategy::QualityBased,
            enable_redundancy: false,
            min_quality_threshold: 0.3,
            quality_check_interval: Duration::from_secs(5),
            path_idle_timeout: Duration::from_secs(60),
            enable_auto_migration: true,
            migration_quality_threshold: 0.2, // Migrate if 20% quality difference
        }
    }
}

impl MultipathConfig {
    /// Configuration optimized for low-latency applications
    pub fn low_latency() -> Self {
        Self {
            max_paths: 2,
            strategy: PathSelectionStrategy::LowestLatency,
            enable_redundancy: false,
            min_quality_threshold: 0.4,
            quality_check_interval: Duration::from_secs(1),
            path_idle_timeout: Duration::from_secs(30),
            enable_auto_migration: true,
            migration_quality_threshold: 0.15,
        }
    }

    /// Configuration optimized for high-bandwidth applications
    pub fn high_bandwidth() -> Self {
        Self {
            max_paths: 4,
            strategy: PathSelectionStrategy::HighestBandwidth,
            enable_redundancy: false,
            min_quality_threshold: 0.3,
            quality_check_interval: Duration::from_secs(5),
            path_idle_timeout: Duration::from_secs(60),
            enable_auto_migration: true,
            migration_quality_threshold: 0.25,
        }
    }

    /// Configuration optimized for high reliability
    pub fn high_reliability() -> Self {
        Self {
            max_paths: 4,
            strategy: PathSelectionStrategy::Redundant,
            enable_redundancy: true,
            min_quality_threshold: 0.2,
            quality_check_interval: Duration::from_secs(3),
            path_idle_timeout: Duration::from_secs(90),
            enable_auto_migration: false, // Don't migrate in redundant mode
            migration_quality_threshold: 0.3,
        }
    }

    /// Configuration optimized for mobile devices
    pub fn mobile() -> Self {
        Self {
            max_paths: 2, // WiFi + Cellular
            strategy: PathSelectionStrategy::QualityBased,
            enable_redundancy: false,
            min_quality_threshold: 0.25,
            quality_check_interval: Duration::from_secs(2),
            path_idle_timeout: Duration::from_secs(45),
            enable_auto_migration: true,
            migration_quality_threshold: 0.2,
        }
    }
}

/// Statistics for multipath operations
#[derive(Debug, Clone, Default)]
pub struct MultipathStats {
    /// Total number of paths created
    pub paths_created: usize,

    /// Number of currently active paths
    pub active_paths: usize,

    /// Total bytes sent across all paths
    pub total_bytes_sent: u64,

    /// Total bytes received across all paths
    pub total_bytes_received: u64,

    /// Number of path migrations performed
    pub migrations_count: usize,

    /// Number of path failures detected
    pub path_failures: usize,

    /// Average quality score across all active paths
    pub avg_quality_score: f64,

    /// Best path quality score
    pub best_path_quality: f64,

    /// Number of packets sent redundantly
    pub redundant_packets: usize,
}

/// Multipath QUIC connection manager
pub struct MultipathQuicManager {
    /// Configuration
    config: MultipathConfig,

    /// Active network paths
    paths: Arc<DashMap<PathId, NetworkPath>>,

    /// Next path ID
    next_path_id: Arc<RwLock<PathId>>,

    /// Round-robin index for path selection
    round_robin_index: Arc<RwLock<usize>>,

    /// Statistics
    stats: Arc<RwLock<MultipathStats>>,

    /// Quality monitoring history
    quality_history: Arc<RwLock<VecDeque<(Instant, f64)>>>,
}

impl MultipathQuicManager {
    /// Create a new multipath QUIC manager
    pub fn new(config: MultipathConfig) -> Self {
        info!(
            "Creating multipath QUIC manager with strategy: {:?}",
            config.strategy
        );

        Self {
            config,
            paths: Arc::new(DashMap::new()),
            next_path_id: Arc::new(RwLock::new(0)),
            round_robin_index: Arc::new(RwLock::new(0)),
            stats: Arc::new(RwLock::new(MultipathStats::default())),
            quality_history: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
        }
    }

    /// Add a new network path
    pub fn add_path(&self, local_addr: SocketAddr, remote_addr: SocketAddr) -> Result<PathId> {
        if self.paths.len() >= self.config.max_paths {
            warn!(
                "Maximum paths reached: {}, cannot add new path",
                self.config.max_paths
            );
            return Err(MultipathError::MaxPathsReached(self.config.max_paths));
        }

        let path_id = {
            let mut id = self.next_path_id.write();
            let current_id = *id;
            *id += 1;
            current_id
        };

        let path = NetworkPath {
            id: path_id,
            local_addr,
            remote_addr,
            state: PathState::Validating,
            quality: PathQuality::default(),
            created_at: Instant::now(),
            last_used: Instant::now(),
            bytes_sent: 0,
            bytes_received: 0,
        };

        self.paths.insert(path_id, path);

        let mut stats = self.stats.write();
        stats.paths_created += 1;
        stats.active_paths = self.paths.len();

        info!(
            "Added new path {}: {} -> {}",
            path_id, local_addr, remote_addr
        );

        Ok(path_id)
    }

    /// Remove a network path
    pub fn remove_path(&self, path_id: PathId) -> Result<()> {
        if self.paths.remove(&path_id).is_some() {
            let mut stats = self.stats.write();
            stats.active_paths = self.paths.len();

            info!("Removed path {}", path_id);
            Ok(())
        } else {
            Err(MultipathError::PathNotFound(path_id))
        }
    }

    /// Update path state
    pub fn update_path_state(&self, path_id: PathId, state: PathState) -> Result<()> {
        if let Some(mut path) = self.paths.get_mut(&path_id) {
            let old_state = path.state;
            path.state = state;

            debug!(
                "Path {} state changed: {:?} -> {:?}",
                path_id, old_state, state
            );

            if state == PathState::Failed {
                let mut stats = self.stats.write();
                stats.path_failures += 1;
            }

            Ok(())
        } else {
            Err(MultipathError::PathNotFound(path_id))
        }
    }

    /// Update path quality metrics
    pub fn update_path_quality(
        &self,
        path_id: PathId,
        rtt_ms: f64,
        bandwidth_bps: u64,
        loss_rate: f64,
        jitter_ms: f64,
    ) -> Result<()> {
        if let Some(mut path) = self.paths.get_mut(&path_id) {
            path.quality
                .update(rtt_ms, bandwidth_bps, loss_rate, jitter_ms);

            debug!(
                "Path {} quality updated: score={:.2}, rtt={:.1}ms, bandwidth={}Mbps, loss={:.2}%",
                path_id,
                path.quality.score,
                path.quality.rtt_ms,
                path.quality.bandwidth_bps / 125_000,
                path.quality.loss_rate * 100.0
            );

            // Check if quality is too low
            if path.quality.score < self.config.min_quality_threshold
                && path.state == PathState::Active
            {
                warn!(
                    "Path {} quality too low: {:.2}, marking as degraded",
                    path_id, path.quality.score
                );
                path.state = PathState::Degraded;
            }

            // Record quality in history
            let mut history = self.quality_history.write();
            history.push_back((Instant::now(), path.quality.score));
            if history.len() > 100 {
                history.pop_front();
            }

            Ok(())
        } else {
            Err(MultipathError::PathNotFound(path_id))
        }
    }

    /// Select the best path for sending data
    pub fn select_path(&self) -> Result<PathId> {
        let active_paths: Vec<_> = self
            .paths
            .iter()
            .filter(|entry| entry.value().state == PathState::Active)
            .collect();

        if active_paths.is_empty() {
            return Err(MultipathError::NoPathsAvailable);
        }

        match self.config.strategy {
            PathSelectionStrategy::RoundRobin => {
                let mut index = self.round_robin_index.write();
                let selected = &active_paths[*index % active_paths.len()];
                *index = (*index + 1) % active_paths.len();
                Ok(selected.value().id)
            }

            PathSelectionStrategy::QualityBased | PathSelectionStrategy::WeightedRoundRobin => {
                // Select path with highest quality score
                let best = active_paths
                    .iter()
                    .max_by(|a, b| {
                        a.value()
                            .quality
                            .score
                            .partial_cmp(&b.value().quality.score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .ok_or(MultipathError::NoPathsAvailable)?;

                Ok(best.value().id)
            }

            PathSelectionStrategy::LowestLatency => {
                // Select path with lowest RTT
                let best = active_paths
                    .iter()
                    .min_by(|a, b| {
                        a.value()
                            .quality
                            .rtt_ms
                            .partial_cmp(&b.value().quality.rtt_ms)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .ok_or(MultipathError::NoPathsAvailable)?;

                Ok(best.value().id)
            }

            PathSelectionStrategy::HighestBandwidth => {
                // Select path with highest bandwidth
                let best = active_paths
                    .iter()
                    .max_by(|a, b| {
                        a.value()
                            .quality
                            .bandwidth_bps
                            .cmp(&b.value().quality.bandwidth_bps)
                    })
                    .ok_or(MultipathError::NoPathsAvailable)?;

                Ok(best.value().id)
            }

            PathSelectionStrategy::Redundant => {
                // In redundant mode, return the first active path
                // (caller should send on all paths)
                Ok(active_paths[0].value().id)
            }
        }
    }

    /// Select all paths for redundant transmission
    pub fn select_all_paths(&self) -> Vec<PathId> {
        self.paths
            .iter()
            .filter(|entry| entry.value().state == PathState::Active)
            .map(|entry| entry.value().id)
            .collect()
    }

    /// Record data sent on a path
    pub fn record_sent(&self, path_id: PathId, bytes: u64) {
        if let Some(mut path) = self.paths.get_mut(&path_id) {
            path.bytes_sent += bytes;
            path.last_used = Instant::now();

            let mut stats = self.stats.write();
            stats.total_bytes_sent += bytes;
        }
    }

    /// Record data received on a path
    pub fn record_received(&self, path_id: PathId, bytes: u64) {
        if let Some(mut path) = self.paths.get_mut(&path_id) {
            path.bytes_received += bytes;
            path.last_used = Instant::now();

            let mut stats = self.stats.write();
            stats.total_bytes_received += bytes;
        }
    }

    /// Get path information
    pub fn get_path(&self, path_id: PathId) -> Option<NetworkPath> {
        self.paths.get(&path_id).map(|entry| entry.value().clone())
    }

    /// Get all active paths
    pub fn get_active_paths(&self) -> Vec<NetworkPath> {
        self.paths
            .iter()
            .filter(|entry| entry.value().state == PathState::Active)
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get all paths
    pub fn get_all_paths(&self) -> Vec<NetworkPath> {
        self.paths
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Check if automatic migration should be triggered
    pub fn should_migrate(&self, current_path_id: PathId) -> Option<PathId> {
        if !self.config.enable_auto_migration {
            return None;
        }

        let current_path = self.paths.get(&current_path_id)?;
        let current_quality = current_path.quality.score;

        // Find best alternative path
        let best_path = self
            .paths
            .iter()
            .filter(|entry| {
                entry.value().id != current_path_id && entry.value().state == PathState::Active
            })
            .max_by(|a, b| {
                a.value()
                    .quality
                    .score
                    .partial_cmp(&b.value().quality.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })?;

        let best_quality = best_path.quality.score;

        // Check if quality difference exceeds threshold
        if best_quality - current_quality >= self.config.migration_quality_threshold {
            info!(
                "Migration recommended: path {} (quality={:.2}) -> path {} (quality={:.2})",
                current_path_id,
                current_quality,
                best_path.value().id,
                best_quality
            );

            let mut stats = self.stats.write();
            stats.migrations_count += 1;

            return Some(best_path.value().id);
        }

        None
    }

    /// Cleanup stale paths
    pub fn cleanup_stale_paths(&self) {
        let now = Instant::now();
        let timeout = self.config.path_idle_timeout;

        let stale_paths: Vec<PathId> = self
            .paths
            .iter()
            .filter(|entry| {
                let path = entry.value();
                now.duration_since(path.last_used) > timeout && path.state != PathState::Active
            })
            .map(|entry| entry.value().id)
            .collect();

        for path_id in stale_paths {
            info!("Removing stale path {}", path_id);
            let _ = self.remove_path(path_id);
        }
    }

    /// Get statistics
    pub fn stats(&self) -> MultipathStats {
        let mut stats = self.stats.read().clone();

        // Calculate average quality score
        let active_paths: Vec<_> = self
            .paths
            .iter()
            .filter(|entry| entry.value().state == PathState::Active)
            .collect();

        if !active_paths.is_empty() {
            let total_quality: f64 = active_paths
                .iter()
                .map(|entry| entry.value().quality.score)
                .sum();
            stats.avg_quality_score = total_quality / active_paths.len() as f64;

            stats.best_path_quality = active_paths
                .iter()
                .map(|entry| entry.value().quality.score)
                .fold(0.0, f64::max);
        }

        stats.active_paths = active_paths.len();

        stats
    }

    /// Get configuration
    pub fn config(&self) -> &MultipathConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_quality_calculation() {
        let mut quality = PathQuality::default();

        // Update with good metrics
        quality.update(10.0, 10_000_000, 0.01, 2.0);

        assert!(quality.score > 0.7, "Good quality should have high score");
        assert_eq!(quality.sample_count, 1);
    }

    #[test]
    fn test_path_quality_ema() {
        let mut quality = PathQuality::default();

        // First update
        quality.update(100.0, 1_000_000, 0.1, 10.0);
        let first_rtt = quality.rtt_ms;

        // Second update with better RTT
        quality.update(50.0, 1_000_000, 0.1, 10.0);

        assert!(
            quality.rtt_ms < first_rtt,
            "RTT should decrease with better sample"
        );
        assert!(
            quality.rtt_ms > 50.0,
            "RTT should be smoothed with EMA, not exact"
        );
        assert_eq!(quality.sample_count, 2);
    }

    #[test]
    fn test_manager_creation() {
        let config = MultipathConfig::default();
        let manager = MultipathQuicManager::new(config);

        let stats = manager.stats();
        assert_eq!(stats.active_paths, 0);
        assert_eq!(stats.paths_created, 0);
    }

    #[test]
    fn test_add_path() {
        let config = MultipathConfig::default();
        let manager = MultipathQuicManager::new(config);

        let local = "127.0.0.1:8080"
            .parse()
            .expect("test: failed to parse local addr");
        let remote = "192.168.1.1:9090"
            .parse()
            .expect("test: failed to parse remote addr");

        let path_id = manager
            .add_path(local, remote)
            .expect("test: failed to add path");

        assert_eq!(path_id, 0);

        // Path starts in Validating state, so active_paths is 0
        let stats = manager.stats();
        assert_eq!(stats.active_paths, 0);
        assert_eq!(stats.paths_created, 1);

        // Activate the path
        manager
            .update_path_state(path_id, PathState::Active)
            .expect("test: failed to update path state to Active");

        // Now it should be counted as active
        let stats = manager.stats();
        assert_eq!(stats.active_paths, 1);
    }

    #[test]
    fn test_max_paths_limit() {
        let config = MultipathConfig {
            max_paths: 2,
            ..Default::default()
        };
        let manager = MultipathQuicManager::new(config);

        let local = "127.0.0.1:8080"
            .parse()
            .expect("test: failed to parse local addr");
        let remote = "192.168.1.1:9090"
            .parse()
            .expect("test: failed to parse remote addr");

        // Add 2 paths (should succeed)
        manager
            .add_path(local, remote)
            .expect("test: failed to add path 1");
        manager
            .add_path(local, remote)
            .expect("test: failed to add path 2");

        // Try to add 3rd path (should fail)
        let result = manager.add_path(local, remote);
        assert!(result.is_err());
        assert!(matches!(result, Err(MultipathError::MaxPathsReached(2))));
    }

    #[test]
    fn test_remove_path() {
        let config = MultipathConfig::default();
        let manager = MultipathQuicManager::new(config);

        let local = "127.0.0.1:8080"
            .parse()
            .expect("test: failed to parse local addr");
        let remote = "192.168.1.1:9090"
            .parse()
            .expect("test: failed to parse remote addr");

        let path_id = manager
            .add_path(local, remote)
            .expect("test: failed to add path");
        manager
            .remove_path(path_id)
            .expect("test: failed to remove path");

        let stats = manager.stats();
        assert_eq!(stats.active_paths, 0);
    }

    #[test]
    fn test_update_path_state() {
        let config = MultipathConfig::default();
        let manager = MultipathQuicManager::new(config);

        let local = "127.0.0.1:8080"
            .parse()
            .expect("test: failed to parse local addr");
        let remote = "192.168.1.1:9090"
            .parse()
            .expect("test: failed to parse remote addr");

        let path_id = manager
            .add_path(local, remote)
            .expect("test: failed to add path");

        manager
            .update_path_state(path_id, PathState::Active)
            .expect("test: failed to update path state to Active");

        let path = manager.get_path(path_id).expect("test: path should exist");
        assert_eq!(path.state, PathState::Active);
    }

    #[test]
    fn test_path_selection_round_robin() {
        let config = MultipathConfig {
            strategy: PathSelectionStrategy::RoundRobin,
            ..Default::default()
        };
        let manager = MultipathQuicManager::new(config);

        let local = "127.0.0.1:8080"
            .parse()
            .expect("test: failed to parse local addr");
        let remote = "192.168.1.1:9090"
            .parse()
            .expect("test: failed to parse remote addr");

        let path1 = manager
            .add_path(local, remote)
            .expect("test: failed to add path 1");
        let path2 = manager
            .add_path(local, remote)
            .expect("test: failed to add path 2");

        manager
            .update_path_state(path1, PathState::Active)
            .expect("test: failed to activate path1");
        manager
            .update_path_state(path2, PathState::Active)
            .expect("test: failed to activate path2");

        let selected1 = manager
            .select_path()
            .expect("test: failed to select path (first)");
        let selected2 = manager
            .select_path()
            .expect("test: failed to select path (second)");

        assert_ne!(selected1, selected2, "Round robin should alternate paths");
    }

    #[test]
    fn test_path_selection_quality_based() {
        let config = MultipathConfig {
            strategy: PathSelectionStrategy::QualityBased,
            ..Default::default()
        };
        let manager = MultipathQuicManager::new(config);

        let local = "127.0.0.1:8080"
            .parse()
            .expect("test: failed to parse local addr");
        let remote = "192.168.1.1:9090"
            .parse()
            .expect("test: failed to parse remote addr");

        let path1 = manager
            .add_path(local, remote)
            .expect("test: failed to add path 1");
        let path2 = manager
            .add_path(local, remote)
            .expect("test: failed to add path 2");

        manager
            .update_path_state(path1, PathState::Active)
            .expect("test: failed to activate path1");
        manager
            .update_path_state(path2, PathState::Active)
            .expect("test: failed to activate path2");

        // Give path2 better quality
        manager
            .update_path_quality(path1, 100.0, 1_000_000, 0.1, 10.0)
            .expect("test: failed to update path1 quality");
        manager
            .update_path_quality(path2, 10.0, 10_000_000, 0.01, 2.0)
            .expect("test: failed to update path2 quality");

        let selected = manager.select_path().expect("test: failed to select path");
        assert_eq!(selected, path2, "Should select higher quality path");
    }

    #[test]
    fn test_path_selection_lowest_latency() {
        let config = MultipathConfig {
            strategy: PathSelectionStrategy::LowestLatency,
            ..Default::default()
        };
        let manager = MultipathQuicManager::new(config);

        let local = "127.0.0.1:8080"
            .parse()
            .expect("test: failed to parse local addr");
        let remote = "192.168.1.1:9090"
            .parse()
            .expect("test: failed to parse remote addr");

        let path1 = manager
            .add_path(local, remote)
            .expect("test: failed to add path 1");
        let path2 = manager
            .add_path(local, remote)
            .expect("test: failed to add path 2");

        manager
            .update_path_state(path1, PathState::Active)
            .expect("test: failed to activate path1");
        manager
            .update_path_state(path2, PathState::Active)
            .expect("test: failed to activate path2");

        // path1 has lower latency
        manager
            .update_path_quality(path1, 10.0, 1_000_000, 0.1, 5.0)
            .expect("test: failed to update path1 quality");
        manager
            .update_path_quality(path2, 100.0, 10_000_000, 0.01, 2.0)
            .expect("test: failed to update path2 quality");

        let selected = manager.select_path().expect("test: failed to select path");
        assert_eq!(selected, path1, "Should select lowest latency path");
    }

    #[test]
    fn test_record_sent_received() {
        let config = MultipathConfig::default();
        let manager = MultipathQuicManager::new(config);

        let local = "127.0.0.1:8080"
            .parse()
            .expect("test: failed to parse local addr");
        let remote = "192.168.1.1:9090"
            .parse()
            .expect("test: failed to parse remote addr");

        let path_id = manager
            .add_path(local, remote)
            .expect("test: failed to add path");

        manager.record_sent(path_id, 1000);
        manager.record_received(path_id, 500);

        let path = manager
            .get_path(path_id)
            .expect("test: path should exist after recording");
        assert_eq!(path.bytes_sent, 1000);
        assert_eq!(path.bytes_received, 500);

        let stats = manager.stats();
        assert_eq!(stats.total_bytes_sent, 1000);
        assert_eq!(stats.total_bytes_received, 500);
    }

    #[test]
    fn test_auto_migration() {
        let config = MultipathConfig {
            enable_auto_migration: true,
            migration_quality_threshold: 0.2,
            ..Default::default()
        };
        let manager = MultipathQuicManager::new(config);

        let local = "127.0.0.1:8080"
            .parse()
            .expect("test: failed to parse local addr");
        let remote = "192.168.1.1:9090"
            .parse()
            .expect("test: failed to parse remote addr");

        let path1 = manager
            .add_path(local, remote)
            .expect("test: failed to add path 1");
        let path2 = manager
            .add_path(local, remote)
            .expect("test: failed to add path 2");

        manager
            .update_path_state(path1, PathState::Active)
            .expect("test: failed to activate path1");
        manager
            .update_path_state(path2, PathState::Active)
            .expect("test: failed to activate path2");

        // path1 has poor quality, path2 has good quality
        manager
            .update_path_quality(path1, 200.0, 500_000, 0.2, 20.0)
            .expect("test: failed to update path1 quality");
        manager
            .update_path_quality(path2, 10.0, 10_000_000, 0.01, 2.0)
            .expect("test: failed to update path2 quality");

        let migration = manager.should_migrate(path1);
        assert_eq!(migration, Some(path2), "Should recommend migration");

        let stats = manager.stats();
        assert_eq!(stats.migrations_count, 1);
    }

    #[test]
    fn test_config_presets() {
        let low_latency = MultipathConfig::low_latency();
        assert_eq!(low_latency.strategy, PathSelectionStrategy::LowestLatency);
        assert_eq!(low_latency.max_paths, 2);

        let high_bandwidth = MultipathConfig::high_bandwidth();
        assert_eq!(
            high_bandwidth.strategy,
            PathSelectionStrategy::HighestBandwidth
        );

        let high_reliability = MultipathConfig::high_reliability();
        assert_eq!(high_reliability.strategy, PathSelectionStrategy::Redundant);
        assert!(high_reliability.enable_redundancy);

        let mobile = MultipathConfig::mobile();
        assert_eq!(mobile.max_paths, 2);
        assert!(mobile.enable_auto_migration);
    }

    #[test]
    fn test_select_all_paths() {
        let config = MultipathConfig::default();
        let manager = MultipathQuicManager::new(config);

        let local = "127.0.0.1:8080"
            .parse()
            .expect("test: failed to parse local addr");
        let remote = "192.168.1.1:9090"
            .parse()
            .expect("test: failed to parse remote addr");

        let path1 = manager
            .add_path(local, remote)
            .expect("test: failed to add path 1");
        let path2 = manager
            .add_path(local, remote)
            .expect("test: failed to add path 2");

        manager
            .update_path_state(path1, PathState::Active)
            .expect("test: failed to activate path1");
        manager
            .update_path_state(path2, PathState::Active)
            .expect("test: failed to activate path2");

        let all_paths = manager.select_all_paths();
        assert_eq!(all_paths.len(), 2);
        assert!(all_paths.contains(&path1));
        assert!(all_paths.contains(&path2));
    }

    #[test]
    fn test_quality_threshold_degradation() {
        let config = MultipathConfig {
            min_quality_threshold: 0.5,
            ..Default::default()
        };
        let manager = MultipathQuicManager::new(config);

        let local = "127.0.0.1:8080"
            .parse()
            .expect("test: failed to parse local addr");
        let remote = "192.168.1.1:9090"
            .parse()
            .expect("test: failed to parse remote addr");

        let path_id = manager
            .add_path(local, remote)
            .expect("test: failed to add path");
        manager
            .update_path_state(path_id, PathState::Active)
            .expect("test: failed to activate path");

        // Update with poor quality
        manager
            .update_path_quality(path_id, 300.0, 100_000, 0.5, 50.0)
            .expect("test: failed to update path quality");

        let path = manager
            .get_path(path_id)
            .expect("test: path should exist after quality update");
        assert_eq!(
            path.state,
            PathState::Degraded,
            "Path should be marked degraded due to low quality"
        );
    }
}
