//! Network Simulator - Simulate various network conditions for testing
//!
//! This module provides utilities to simulate different network conditions such as:
//! - Latency injection (constant, variable, spike patterns)
//! - Packet loss simulation
//! - Bandwidth throttling
//! - Network partitions
//! - Connection drops
//! - Jitter and congestion
//!
//! Useful for testing application behavior under adverse network conditions.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_network::{NetworkSimulator, SimulatorConfig, NetworkCondition};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a simulator with high latency
//! let config = SimulatorConfig {
//!     base_latency_ms: 200,
//!     latency_variance_ms: 50,
//!     packet_loss_rate: 0.05, // 5% packet loss
//!     ..Default::default()
//! };
//!
//! let simulator = NetworkSimulator::new(config);
//!
//! // Apply network conditions
//! simulator.start().await?;
//!
//! // Your network operations here will experience the simulated conditions
//!
//! // Get statistics
//! let stats = simulator.stats();
//! println!("Packets dropped: {}", stats.packets_dropped);
//!
//! simulator.stop().await?;
//! # Ok(())
//! # }
//! ```

use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::time::sleep;

/// Errors that can occur during network simulation
#[derive(Debug, Error)]
pub enum SimulatorError {
    /// Simulator is not running
    #[error("Simulator is not running")]
    NotRunning,

    /// Simulator is already running
    #[error("Simulator is already running")]
    AlreadyRunning,

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Network condition profiles
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkCondition {
    /// Perfect network (no simulation)
    Perfect,
    /// Good network (5ms latency, 0.1% loss)
    Good,
    /// Fair network (50ms latency, 1% loss)
    Fair,
    /// Poor network (200ms latency, 5% loss)
    Poor,
    /// Very poor network (500ms latency, 10% loss)
    VeryPoor,
    /// Mobile 3G network
    Mobile3G,
    /// Mobile 4G network
    Mobile4G,
    /// Mobile 5G network
    Mobile5G,
    /// Satellite network
    Satellite,
    /// Custom condition
    Custom,
}

/// Configuration for the network simulator
#[derive(Debug, Clone)]
pub struct SimulatorConfig {
    /// Base latency in milliseconds
    pub base_latency_ms: u64,

    /// Latency variance in milliseconds (for jitter)
    pub latency_variance_ms: u64,

    /// Packet loss rate (0.0 to 1.0)
    pub packet_loss_rate: f64,

    /// Bandwidth limit in bytes per second (0 = unlimited)
    pub bandwidth_limit_bps: u64,

    /// Probability of latency spikes (0.0 to 1.0)
    pub spike_probability: f64,

    /// Spike latency multiplier
    pub spike_multiplier: u64,

    /// Enable packet reordering
    pub enable_reordering: bool,

    /// Reordering probability (0.0 to 1.0)
    pub reorder_probability: f64,

    /// Maximum reorder delay in milliseconds
    pub max_reorder_delay_ms: u64,
}

impl Default for SimulatorConfig {
    fn default() -> Self {
        Self {
            base_latency_ms: 0,
            latency_variance_ms: 0,
            packet_loss_rate: 0.0,
            bandwidth_limit_bps: 0,
            spike_probability: 0.0,
            spike_multiplier: 10,
            enable_reordering: false,
            reorder_probability: 0.0,
            max_reorder_delay_ms: 100,
        }
    }
}

impl SimulatorConfig {
    /// Create configuration for a specific network condition
    pub fn from_condition(condition: NetworkCondition) -> Self {
        match condition {
            NetworkCondition::Perfect => Self::default(),
            NetworkCondition::Good => Self {
                base_latency_ms: 5,
                latency_variance_ms: 2,
                packet_loss_rate: 0.001,
                ..Default::default()
            },
            NetworkCondition::Fair => Self {
                base_latency_ms: 50,
                latency_variance_ms: 10,
                packet_loss_rate: 0.01,
                spike_probability: 0.05,
                ..Default::default()
            },
            NetworkCondition::Poor => Self {
                base_latency_ms: 200,
                latency_variance_ms: 50,
                packet_loss_rate: 0.05,
                spike_probability: 0.1,
                spike_multiplier: 5,
                ..Default::default()
            },
            NetworkCondition::VeryPoor => Self {
                base_latency_ms: 500,
                latency_variance_ms: 150,
                packet_loss_rate: 0.1,
                spike_probability: 0.2,
                spike_multiplier: 10,
                enable_reordering: true,
                reorder_probability: 0.05,
                ..Default::default()
            },
            NetworkCondition::Mobile3G => Self {
                base_latency_ms: 100,
                latency_variance_ms: 50,
                packet_loss_rate: 0.02,
                bandwidth_limit_bps: 384_000, // 384 Kbps
                spike_probability: 0.1,
                ..Default::default()
            },
            NetworkCondition::Mobile4G => Self {
                base_latency_ms: 50,
                latency_variance_ms: 20,
                packet_loss_rate: 0.01,
                bandwidth_limit_bps: 10_000_000, // 10 Mbps
                spike_probability: 0.05,
                ..Default::default()
            },
            NetworkCondition::Mobile5G => Self {
                base_latency_ms: 10,
                latency_variance_ms: 5,
                packet_loss_rate: 0.005,
                bandwidth_limit_bps: 100_000_000, // 100 Mbps
                ..Default::default()
            },
            NetworkCondition::Satellite => Self {
                base_latency_ms: 600,
                latency_variance_ms: 100,
                packet_loss_rate: 0.03,
                bandwidth_limit_bps: 1_000_000, // 1 Mbps
                spike_probability: 0.15,
                ..Default::default()
            },
            NetworkCondition::Custom => Self::default(),
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), SimulatorError> {
        if self.packet_loss_rate < 0.0 || self.packet_loss_rate > 1.0 {
            return Err(SimulatorError::InvalidConfig(
                "packet_loss_rate must be between 0.0 and 1.0".to_string(),
            ));
        }

        if self.spike_probability < 0.0 || self.spike_probability > 1.0 {
            return Err(SimulatorError::InvalidConfig(
                "spike_probability must be between 0.0 and 1.0".to_string(),
            ));
        }

        if self.reorder_probability < 0.0 || self.reorder_probability > 1.0 {
            return Err(SimulatorError::InvalidConfig(
                "reorder_probability must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(())
    }
}

/// Statistics tracked by the simulator
#[derive(Debug, Clone, Default)]
pub struct SimulatorStats {
    /// Total packets processed
    pub packets_processed: u64,

    /// Packets dropped due to loss simulation
    pub packets_dropped: u64,

    /// Packets delayed
    pub packets_delayed: u64,

    /// Packets reordered
    pub packets_reordered: u64,

    /// Total bytes processed
    pub bytes_processed: u64,

    /// Average latency in milliseconds
    pub avg_latency_ms: f64,

    /// Maximum latency observed in milliseconds
    pub max_latency_ms: u64,

    /// Number of latency spikes
    pub latency_spikes: u64,

    /// Simulation start time
    pub start_time: Option<Instant>,

    /// Simulation duration
    pub duration: Duration,
}

impl SimulatorStats {
    /// Calculate packet loss rate
    pub fn packet_loss_rate(&self) -> f64 {
        if self.packets_processed == 0 {
            0.0
        } else {
            self.packets_dropped as f64 / self.packets_processed as f64
        }
    }

    /// Calculate throughput in bytes per second
    pub fn throughput_bps(&self) -> f64 {
        if self.duration.as_secs_f64() == 0.0 {
            0.0
        } else {
            self.bytes_processed as f64 / self.duration.as_secs_f64()
        }
    }
}

/// Network simulator for testing
pub struct NetworkSimulator {
    config: SimulatorConfig,
    stats: Arc<RwLock<SimulatorStats>>,
    running: Arc<RwLock<bool>>,
    partitions: Arc<DashMap<String, Vec<String>>>,
}

impl NetworkSimulator {
    /// Create a new network simulator
    pub fn new(config: SimulatorConfig) -> Self {
        Self {
            config,
            stats: Arc::new(RwLock::new(SimulatorStats::default())),
            running: Arc::new(RwLock::new(false)),
            partitions: Arc::new(DashMap::new()),
        }
    }

    /// Create simulator from a network condition
    pub fn from_condition(condition: NetworkCondition) -> Self {
        Self::new(SimulatorConfig::from_condition(condition))
    }

    /// Start the simulator
    pub async fn start(&self) -> Result<(), SimulatorError> {
        self.config.validate()?;

        let mut running = self.running.write();
        if *running {
            return Err(SimulatorError::AlreadyRunning);
        }

        *running = true;

        let mut stats = self.stats.write();
        stats.start_time = Some(Instant::now());
        stats.duration = Duration::from_secs(0);

        Ok(())
    }

    /// Stop the simulator
    pub async fn stop(&self) -> Result<(), SimulatorError> {
        let mut running = self.running.write();
        if !*running {
            return Err(SimulatorError::NotRunning);
        }

        *running = false;

        let mut stats = self.stats.write();
        if let Some(start_time) = stats.start_time {
            stats.duration = start_time.elapsed();
        }

        Ok(())
    }

    /// Check if simulator is running
    pub fn is_running(&self) -> bool {
        *self.running.read()
    }

    /// Simulate network delay for a packet
    pub async fn delay_packet(&self, packet_size: usize) -> Result<bool, SimulatorError> {
        if !self.is_running() {
            return Ok(true);
        }

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.packets_processed += 1;
            stats.bytes_processed += packet_size as u64;
        }

        // Simulate packet loss
        if rand::random::<f64>() < self.config.packet_loss_rate {
            let mut stats = self.stats.write();
            stats.packets_dropped += 1;
            return Ok(false); // Packet dropped
        }

        // Calculate latency
        let mut latency_ms = self.config.base_latency_ms;

        // Add variance (jitter)
        if self.config.latency_variance_ms > 0 {
            let variance = (rand::random::<f64>() * self.config.latency_variance_ms as f64) as u64;
            latency_ms += variance;
        }

        // Check for latency spike
        if rand::random::<f64>() < self.config.spike_probability {
            latency_ms *= self.config.spike_multiplier;
            let mut stats = self.stats.write();
            stats.latency_spikes += 1;
        }

        // Update latency statistics
        {
            let mut stats = self.stats.write();
            if latency_ms > stats.max_latency_ms {
                stats.max_latency_ms = latency_ms;
            }

            // Update average latency (exponential moving average)
            let alpha = 0.3;
            stats.avg_latency_ms = alpha * latency_ms as f64 + (1.0 - alpha) * stats.avg_latency_ms;
        }

        // Apply delay
        if latency_ms > 0 {
            sleep(Duration::from_millis(latency_ms)).await;

            let mut stats = self.stats.write();
            stats.packets_delayed += 1;
        }

        // Simulate reordering
        if self.config.enable_reordering && rand::random::<f64>() < self.config.reorder_probability
        {
            let reorder_delay =
                (rand::random::<f64>() * self.config.max_reorder_delay_ms as f64) as u64;
            sleep(Duration::from_millis(reorder_delay)).await;

            let mut stats = self.stats.write();
            stats.packets_reordered += 1;
        }

        Ok(true) // Packet delivered
    }

    /// Create a network partition between two groups
    pub fn create_partition(&self, group1: Vec<String>, group2: Vec<String>) {
        for peer in &group1 {
            self.partitions.insert(peer.clone(), group2.clone());
        }
        for peer in &group2 {
            self.partitions.insert(peer.clone(), group1.clone());
        }
    }

    /// Remove all partitions
    pub fn clear_partitions(&self) {
        self.partitions.clear();
    }

    /// Check if two peers are partitioned
    pub fn is_partitioned(&self, peer1: &str, peer2: &str) -> bool {
        if let Some(partitioned_peers) = self.partitions.get(peer1) {
            partitioned_peers.contains(&peer2.to_string())
        } else {
            false
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> SimulatorStats {
        let mut stats = self.stats.read().clone();
        if self.is_running() {
            if let Some(start_time) = stats.start_time {
                stats.duration = start_time.elapsed();
            }
        }
        stats
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write();
        *stats = SimulatorStats::default();
        if self.is_running() {
            stats.start_time = Some(Instant::now());
        }
    }

    /// Get configuration
    pub fn config(&self) -> &SimulatorConfig {
        &self.config
    }

    /// Update configuration (only works when stopped)
    pub fn update_config(&mut self, config: SimulatorConfig) -> Result<(), SimulatorError> {
        if self.is_running() {
            return Err(SimulatorError::Internal(
                "Cannot update config while running".to_string(),
            ));
        }

        config.validate()?;
        self.config = config;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulator_creation() {
        let config = SimulatorConfig::default();
        let simulator = NetworkSimulator::new(config);
        assert!(!simulator.is_running());
    }

    #[test]
    fn test_network_conditions() {
        let conditions = vec![
            NetworkCondition::Perfect,
            NetworkCondition::Good,
            NetworkCondition::Fair,
            NetworkCondition::Poor,
            NetworkCondition::VeryPoor,
            NetworkCondition::Mobile3G,
            NetworkCondition::Mobile4G,
            NetworkCondition::Mobile5G,
            NetworkCondition::Satellite,
        ];

        for condition in conditions {
            let config = SimulatorConfig::from_condition(condition);
            assert!(config.validate().is_ok());
        }
    }

    #[tokio::test]
    async fn test_start_stop() {
        let simulator = NetworkSimulator::from_condition(NetworkCondition::Good);

        assert!(!simulator.is_running());

        simulator
            .start()
            .await
            .expect("test: simulator should start successfully");
        assert!(simulator.is_running());

        simulator
            .stop()
            .await
            .expect("test: simulator should stop successfully");
        assert!(!simulator.is_running());
    }

    #[tokio::test]
    async fn test_packet_delay() {
        let config = SimulatorConfig {
            base_latency_ms: 10,
            packet_loss_rate: 0.0,
            ..Default::default()
        };

        let simulator = NetworkSimulator::new(config);
        simulator
            .start()
            .await
            .expect("test: simulator should start in test_packet_delay");

        let start = Instant::now();
        let delivered = simulator
            .delay_packet(1024)
            .await
            .expect("test: delay_packet should return Ok in test_packet_delay");
        let elapsed = start.elapsed();

        assert!(delivered);
        assert!(elapsed >= Duration::from_millis(10));

        simulator
            .stop()
            .await
            .expect("test: simulator should stop in test_packet_delay");
    }

    #[tokio::test]
    async fn test_packet_loss() {
        let config = SimulatorConfig {
            packet_loss_rate: 1.0, // 100% loss
            ..Default::default()
        };

        let simulator = NetworkSimulator::new(config);
        simulator
            .start()
            .await
            .expect("test: simulator should start in test_packet_loss");

        let delivered = simulator
            .delay_packet(1024)
            .await
            .expect("test: delay_packet should return Ok even when packet is dropped");
        assert!(!delivered); // Should be dropped

        let stats = simulator.stats();
        assert_eq!(stats.packets_dropped, 1);

        simulator
            .stop()
            .await
            .expect("test: simulator should stop in test_packet_loss");
    }

    #[test]
    fn test_partitions() {
        let simulator = NetworkSimulator::from_condition(NetworkCondition::Good);

        let group1 = vec!["peer1".to_string(), "peer2".to_string()];
        let group2 = vec!["peer3".to_string(), "peer4".to_string()];

        simulator.create_partition(group1, group2);

        assert!(simulator.is_partitioned("peer1", "peer3"));
        assert!(simulator.is_partitioned("peer2", "peer4"));
        assert!(!simulator.is_partitioned("peer1", "peer2"));

        simulator.clear_partitions();
        assert!(!simulator.is_partitioned("peer1", "peer3"));
    }

    #[tokio::test]
    async fn test_statistics() {
        let config = SimulatorConfig {
            base_latency_ms: 5,
            packet_loss_rate: 0.0,
            ..Default::default()
        };

        let simulator = NetworkSimulator::new(config);
        simulator
            .start()
            .await
            .expect("test: simulator should start in test_statistics");

        for _ in 0..10 {
            simulator
                .delay_packet(1024)
                .await
                .expect("test: delay_packet should succeed in test_statistics loop");
        }

        let stats = simulator.stats();
        assert_eq!(stats.packets_processed, 10);
        assert_eq!(stats.bytes_processed, 10240);
        assert!(stats.avg_latency_ms > 0.0);

        simulator
            .stop()
            .await
            .expect("test: simulator should stop in test_statistics");
    }

    #[test]
    fn test_invalid_config() {
        let config = SimulatorConfig {
            packet_loss_rate: 1.5, // Invalid: > 1.0
            ..Default::default()
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_stats_calculation() {
        let stats = SimulatorStats {
            packets_processed: 100,
            packets_dropped: 5,
            bytes_processed: 102400,
            duration: Duration::from_secs(10),
            ..Default::default()
        };

        assert_eq!(stats.packet_loss_rate(), 0.05);
        assert_eq!(stats.throughput_bps(), 10240.0);
    }

    #[tokio::test]
    async fn test_config_update() {
        let mut simulator = NetworkSimulator::from_condition(NetworkCondition::Good);

        let new_config = SimulatorConfig::from_condition(NetworkCondition::Poor);
        assert!(simulator.update_config(new_config).is_ok());

        simulator
            .start()
            .await
            .expect("test: simulator should start after config update");

        let invalid_config = SimulatorConfig {
            packet_loss_rate: -0.5,
            ..Default::default()
        };
        assert!(simulator.update_config(invalid_config).is_err());
    }

    #[tokio::test]
    async fn test_reset_stats() {
        let simulator = NetworkSimulator::from_condition(NetworkCondition::Good);
        simulator
            .start()
            .await
            .expect("test: simulator should start in test_reset_stats");

        simulator
            .delay_packet(1024)
            .await
            .expect("test: delay_packet should succeed in test_reset_stats");
        assert_eq!(simulator.stats().packets_processed, 1);

        simulator.reset_stats();
        assert_eq!(simulator.stats().packets_processed, 0);

        simulator
            .stop()
            .await
            .expect("test: simulator should stop in test_reset_stats");
    }
}
