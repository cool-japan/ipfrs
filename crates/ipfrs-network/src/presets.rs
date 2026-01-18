//! Configuration presets for common use cases
//!
//! This module provides pre-configured setups for common network scenarios,
//! combining multiple module configurations into ready-to-use presets.
//!
//! ## Available Presets
//!
//! - **Default**: Balanced configuration for general use
//! - **Low Memory**: Optimized for devices with limited RAM (< 128 MB)
//! - **IoT**: Internet of Things devices (128-512 MB RAM)
//! - **Mobile**: Mobile devices with battery constraints
//! - **High Performance**: Server/desktop with ample resources
//! - **Low Latency**: Gaming, real-time communications
//! - **High Throughput**: File transfers, video streaming
//! - **Privacy**: Maximum privacy with Tor integration
//! - **Development**: Development and testing
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_network::presets::NetworkPreset;
//!
//! // Get a mobile-optimized configuration
//! let preset = NetworkPreset::mobile();
//!
//! // Access individual configurations
//! let network_config = preset.network;
//! let quic_config = preset.quic;
//! let throttle_config = preset.throttle;
//! ```

use crate::{
    AdaptivePollingConfig, BackgroundModeConfig, ConnectionLimitsConfig, DhtConfig,
    GeoRouterConfig, MemoryMonitorConfig, MultipathConfig, NetworkConfig, OfflineQueueConfig,
    PeerSelectorConfig, PeerStoreConfig, QualityPredictorConfig, QueryBatcherConfig, QuicConfig,
    ThrottleConfig, TorConfig,
};
use std::time::Duration;

/// Complete network configuration preset
#[derive(Debug, Clone)]
pub struct NetworkPreset {
    /// Core network configuration
    pub network: NetworkConfig,
    /// QUIC transport configuration
    pub quic: QuicConfig,
    /// DHT configuration
    pub dht: DhtConfig,
    /// Peer store configuration
    pub peer_store: PeerStoreConfig,
    /// Connection limits
    pub connection_limits: ConnectionLimitsConfig,
    /// Bandwidth throttling
    pub throttle: Option<ThrottleConfig>,
    /// Adaptive polling
    pub adaptive_polling: Option<AdaptivePollingConfig>,
    /// Memory monitoring
    pub memory_monitor: Option<MemoryMonitorConfig>,
    /// Offline queue
    pub offline_queue: Option<OfflineQueueConfig>,
    /// Background mode
    pub background_mode: Option<BackgroundModeConfig>,
    /// Query batching
    pub query_batcher: Option<QueryBatcherConfig>,
    /// Geographic routing
    pub geo_router: Option<GeoRouterConfig>,
    /// Quality prediction
    pub quality_predictor: Option<QualityPredictorConfig>,
    /// Peer selection
    pub peer_selector: Option<PeerSelectorConfig>,
    /// Multipath QUIC
    pub multipath: Option<MultipathConfig>,
    /// Tor configuration
    pub tor: Option<TorConfig>,
    /// Preset description
    pub description: String,
}

impl NetworkPreset {
    /// Default preset - balanced configuration for general use
    pub fn default_preset() -> Self {
        Self {
            network: NetworkConfig::default(),
            quic: QuicConfig::default(),
            dht: DhtConfig::default(),
            peer_store: PeerStoreConfig::default(),
            connection_limits: ConnectionLimitsConfig::default(),
            throttle: None,
            adaptive_polling: None,
            memory_monitor: None,
            offline_queue: None,
            background_mode: None,
            query_batcher: None,
            geo_router: None,
            quality_predictor: None,
            peer_selector: None,
            multipath: None,
            tor: None,
            description: "Balanced configuration for general use".to_string(),
        }
    }

    /// Low memory preset - optimized for devices with < 128 MB RAM
    pub fn low_memory() -> Self {
        Self {
            network: NetworkConfig::low_memory(),
            quic: QuicConfig::mobile(), // Mobile QUIC is memory-efficient
            dht: DhtConfig::default(),
            peer_store: PeerStoreConfig::low_memory(),
            connection_limits: ConnectionLimitsConfig {
                max_connections: 16,
                max_inbound: 8,
                max_outbound: 8,
                reserved_slots: 2,
                idle_timeout: Duration::from_secs(180),
                min_score_threshold: 40,
            },
            throttle: Some(ThrottleConfig::low_power()),
            adaptive_polling: Some(AdaptivePollingConfig::low_power()),
            memory_monitor: Some(MemoryMonitorConfig::low_memory()),
            offline_queue: None,
            background_mode: None,
            query_batcher: Some(QueryBatcherConfig::low_power()),
            geo_router: None,
            quality_predictor: None,
            peer_selector: None,
            multipath: None,
            tor: None,
            description: "Optimized for devices with < 128 MB RAM (RPi Zero, embedded)".to_string(),
        }
    }

    /// IoT preset - Internet of Things devices (128-512 MB RAM)
    pub fn iot() -> Self {
        Self {
            network: NetworkConfig::iot(),
            quic: QuicConfig::mobile(),
            dht: DhtConfig::default(),
            peer_store: PeerStoreConfig::iot(),
            connection_limits: ConnectionLimitsConfig {
                max_connections: 32,
                max_inbound: 16,
                max_outbound: 16,
                reserved_slots: 4,
                idle_timeout: Duration::from_secs(240),
                min_score_threshold: 35,
            },
            throttle: Some(ThrottleConfig::iot()),
            adaptive_polling: Some(AdaptivePollingConfig::iot()),
            memory_monitor: Some(MemoryMonitorConfig::iot()),
            offline_queue: Some(OfflineQueueConfig::iot()),
            background_mode: None,
            query_batcher: Some(QueryBatcherConfig::low_power()),
            geo_router: None,
            quality_predictor: None,
            peer_selector: None,
            multipath: None,
            tor: None,
            description: "IoT devices with moderate resources (ESP32, RPi 3)".to_string(),
        }
    }

    /// Mobile preset - smartphones and tablets
    pub fn mobile() -> Self {
        Self {
            network: NetworkConfig::mobile(),
            quic: QuicConfig::mobile(),
            dht: DhtConfig::default(),
            peer_store: PeerStoreConfig::mobile(),
            connection_limits: ConnectionLimitsConfig {
                max_connections: 64,
                max_inbound: 32,
                max_outbound: 32,
                reserved_slots: 6,
                idle_timeout: Duration::from_secs(300),
                min_score_threshold: 30,
            },
            throttle: Some(ThrottleConfig::mobile()),
            adaptive_polling: Some(AdaptivePollingConfig::mobile()),
            memory_monitor: Some(MemoryMonitorConfig::mobile()),
            offline_queue: Some(OfflineQueueConfig::mobile()),
            background_mode: Some(BackgroundModeConfig::mobile()),
            query_batcher: Some(QueryBatcherConfig::mobile()),
            geo_router: Some(GeoRouterConfig::low_latency()),
            quality_predictor: Some(QualityPredictorConfig::low_latency()),
            peer_selector: Some(PeerSelectorConfig::mobile()),
            multipath: Some(MultipathConfig::mobile()),
            tor: None,
            description: "Mobile devices with battery optimization (iOS, Android)".to_string(),
        }
    }

    /// High performance preset - servers and desktops with ample resources
    pub fn high_performance() -> Self {
        Self {
            network: NetworkConfig::high_performance(),
            quic: QuicConfig::high_throughput(),
            dht: DhtConfig::default(),
            peer_store: PeerStoreConfig::server(),
            connection_limits: ConnectionLimitsConfig {
                max_connections: 1024,
                max_inbound: 512,
                max_outbound: 512,
                reserved_slots: 16,
                idle_timeout: Duration::from_secs(600),
                min_score_threshold: 20,
            },
            throttle: None, // No throttling for high performance
            adaptive_polling: Some(AdaptivePollingConfig::high_performance()),
            memory_monitor: None, // Not needed with ample resources
            offline_queue: None,
            background_mode: None,
            query_batcher: Some(QueryBatcherConfig::high_performance()),
            geo_router: Some(GeoRouterConfig::global()),
            quality_predictor: Some(QualityPredictorConfig::high_bandwidth()),
            peer_selector: Some(PeerSelectorConfig::high_bandwidth()),
            multipath: Some(MultipathConfig::high_bandwidth()),
            tor: None,
            description: "Servers and desktops with ample resources (> 2 GB RAM)".to_string(),
        }
    }

    /// Low latency preset - gaming, real-time communications
    pub fn low_latency() -> Self {
        Self {
            network: NetworkConfig::default(),
            quic: QuicConfig::low_latency(),
            dht: DhtConfig::default(),
            peer_store: PeerStoreConfig::default(),
            connection_limits: ConnectionLimitsConfig::default(),
            throttle: None,
            adaptive_polling: Some(AdaptivePollingConfig::high_performance()),
            memory_monitor: None,
            offline_queue: None,
            background_mode: None,
            query_batcher: None, // No batching for low latency
            geo_router: Some(GeoRouterConfig::low_latency()),
            quality_predictor: Some(QualityPredictorConfig::low_latency()),
            peer_selector: Some(PeerSelectorConfig::low_latency()),
            multipath: Some(MultipathConfig::low_latency()),
            tor: None,
            description: "Optimized for minimal latency (gaming, VoIP, real-time apps)".to_string(),
        }
    }

    /// High throughput preset - file transfers, video streaming
    pub fn high_throughput() -> Self {
        Self {
            network: NetworkConfig::high_performance(),
            quic: QuicConfig::high_throughput(),
            dht: DhtConfig::default(),
            peer_store: PeerStoreConfig::server(),
            connection_limits: ConnectionLimitsConfig {
                max_connections: 2048,
                max_inbound: 1024,
                max_outbound: 1024,
                reserved_slots: 32,
                idle_timeout: Duration::from_secs(900),
                min_score_threshold: 15,
            },
            throttle: None,
            adaptive_polling: Some(AdaptivePollingConfig::high_performance()),
            memory_monitor: None,
            offline_queue: None,
            background_mode: None,
            query_batcher: None,
            geo_router: Some(GeoRouterConfig::global()),
            quality_predictor: Some(QualityPredictorConfig::high_bandwidth()),
            peer_selector: Some(PeerSelectorConfig::high_bandwidth()),
            multipath: Some(MultipathConfig::high_bandwidth()),
            tor: None,
            description: "Optimized for maximum throughput (CDN, video streaming, bulk transfers)"
                .to_string(),
        }
    }

    /// Privacy preset - maximum privacy with Tor integration
    pub fn privacy() -> Self {
        Self {
            network: NetworkConfig::default(),
            quic: QuicConfig::default(),
            dht: DhtConfig::default(),
            peer_store: PeerStoreConfig::default(),
            connection_limits: ConnectionLimitsConfig::default(),
            throttle: None,
            adaptive_polling: None,
            memory_monitor: None,
            offline_queue: None,
            background_mode: None,
            query_batcher: None,
            geo_router: None,
            quality_predictor: None,
            peer_selector: None,
            multipath: None,
            tor: Some(TorConfig::high_privacy()),
            description: "Maximum privacy with Tor onion routing and stream isolation".to_string(),
        }
    }

    /// Development preset - convenient settings for testing and development
    pub fn development() -> Self {
        Self {
            network: NetworkConfig::default(),
            quic: QuicConfig::default(),
            dht: DhtConfig::default(),
            peer_store: PeerStoreConfig::default(),
            connection_limits: ConnectionLimitsConfig {
                max_connections: 50,
                max_inbound: 25,
                max_outbound: 25,
                reserved_slots: 2,
                idle_timeout: Duration::from_secs(300),
                min_score_threshold: 30,
            },
            throttle: None,
            adaptive_polling: None,
            memory_monitor: None,
            offline_queue: None,
            background_mode: None,
            query_batcher: None,
            geo_router: None,
            quality_predictor: None,
            peer_selector: None,
            multipath: None,
            tor: None,
            description: "Development and testing with moderate limits".to_string(),
        }
    }

    /// Get preset name
    pub fn name(&self) -> &str {
        if self.description.contains("general use") {
            "Default"
        } else if self.description.contains("< 128 MB") {
            "Low Memory"
        } else if self.description.contains("IoT") {
            "IoT"
        } else if self.description.contains("Mobile") {
            "Mobile"
        } else if self.description.contains("ample resources") {
            "High Performance"
        } else if self.description.contains("minimal latency") {
            "Low Latency"
        } else if self.description.contains("maximum throughput") {
            "High Throughput"
        } else if self.description.contains("privacy") {
            "Privacy"
        } else if self.description.contains("Development") {
            "Development"
        } else {
            "Custom"
        }
    }

    /// Check if throttling is enabled
    pub fn has_throttling(&self) -> bool {
        self.throttle.is_some()
    }

    /// Check if adaptive polling is enabled
    pub fn has_adaptive_polling(&self) -> bool {
        self.adaptive_polling.is_some()
    }

    /// Check if memory monitoring is enabled
    pub fn has_memory_monitoring(&self) -> bool {
        self.memory_monitor.is_some()
    }

    /// Check if offline queue is enabled
    pub fn has_offline_queue(&self) -> bool {
        self.offline_queue.is_some()
    }

    /// Check if Tor is enabled
    pub fn has_tor(&self) -> bool {
        self.tor.is_some()
    }

    /// Get a summary of enabled features
    pub fn features_summary(&self) -> Vec<String> {
        let mut features = Vec::new();

        if self.has_throttling() {
            features.push("Bandwidth Throttling".to_string());
        }
        if self.has_adaptive_polling() {
            features.push("Adaptive Polling".to_string());
        }
        if self.has_memory_monitoring() {
            features.push("Memory Monitoring".to_string());
        }
        if self.has_offline_queue() {
            features.push("Offline Queue".to_string());
        }
        if self.background_mode.is_some() {
            features.push("Background Mode".to_string());
        }
        if self.geo_router.is_some() {
            features.push("Geographic Routing".to_string());
        }
        if self.quality_predictor.is_some() {
            features.push("Connection Quality Prediction".to_string());
        }
        if self.multipath.is_some() {
            features.push("Multipath QUIC".to_string());
        }
        if self.has_tor() {
            features.push("Tor Privacy".to_string());
        }

        features
    }
}

impl Default for NetworkPreset {
    fn default() -> Self {
        Self::default_preset()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_preset() {
        let preset = NetworkPreset::default_preset();
        assert_eq!(preset.name(), "Default");
        assert_eq!(preset.description, "Balanced configuration for general use");
        assert!(!preset.has_throttling());
        assert!(!preset.has_memory_monitoring());
    }

    #[test]
    fn test_low_memory_preset() {
        let preset = NetworkPreset::low_memory();
        assert_eq!(preset.name(), "Low Memory");
        assert!(preset.has_throttling());
        assert!(preset.has_adaptive_polling());
        assert!(preset.has_memory_monitoring());
        assert!(preset.description.contains("< 128 MB"));
    }

    #[test]
    fn test_iot_preset() {
        let preset = NetworkPreset::iot();
        assert_eq!(preset.name(), "IoT");
        assert!(preset.has_throttling());
        assert!(preset.has_adaptive_polling());
        assert!(preset.has_memory_monitoring());
        assert!(preset.has_offline_queue());
    }

    #[test]
    fn test_mobile_preset() {
        let preset = NetworkPreset::mobile();
        assert_eq!(preset.name(), "Mobile");
        assert!(preset.has_throttling());
        assert!(preset.has_adaptive_polling());
        assert!(preset.has_offline_queue());
        assert!(preset.background_mode.is_some());
        assert!(preset.geo_router.is_some());
        assert!(preset.multipath.is_some());
    }

    #[test]
    fn test_high_performance_preset() {
        let preset = NetworkPreset::high_performance();
        assert_eq!(preset.name(), "High Performance");
        assert!(!preset.has_throttling()); // No throttling for performance
        assert!(!preset.has_memory_monitoring()); // Not needed with ample resources
        assert!(preset.geo_router.is_some());
        assert!(preset.multipath.is_some());
    }

    #[test]
    fn test_low_latency_preset() {
        let preset = NetworkPreset::low_latency();
        assert_eq!(preset.name(), "Low Latency");
        assert!(!preset.has_throttling());
        assert!(preset.geo_router.is_some());
        assert!(preset.quality_predictor.is_some());
        assert!(preset.multipath.is_some());
    }

    #[test]
    fn test_high_throughput_preset() {
        let preset = NetworkPreset::high_throughput();
        assert_eq!(preset.name(), "High Throughput");
        assert!(!preset.has_throttling());
        assert!(preset.geo_router.is_some());
        assert!(preset.multipath.is_some());
    }

    #[test]
    fn test_privacy_preset() {
        let preset = NetworkPreset::privacy();
        assert_eq!(preset.name(), "Privacy");
        assert!(preset.has_tor());
        assert!(preset.tor.is_some());
    }

    #[test]
    fn test_development_preset() {
        let preset = NetworkPreset::development();
        assert_eq!(preset.name(), "Development");
        assert!(!preset.has_throttling());
        assert!(!preset.has_tor());
    }

    #[test]
    fn test_features_summary() {
        let preset = NetworkPreset::mobile();
        let features = preset.features_summary();

        assert!(features.contains(&"Bandwidth Throttling".to_string()));
        assert!(features.contains(&"Adaptive Polling".to_string()));
        assert!(features.contains(&"Memory Monitoring".to_string()));
        assert!(features.contains(&"Offline Queue".to_string()));
        assert!(features.contains(&"Background Mode".to_string()));
        assert!(features.contains(&"Geographic Routing".to_string()));
        assert!(features.contains(&"Multipath QUIC".to_string()));
    }

    #[test]
    fn test_all_presets_have_descriptions() {
        let presets = vec![
            NetworkPreset::default_preset(),
            NetworkPreset::low_memory(),
            NetworkPreset::iot(),
            NetworkPreset::mobile(),
            NetworkPreset::high_performance(),
            NetworkPreset::low_latency(),
            NetworkPreset::high_throughput(),
            NetworkPreset::privacy(),
            NetworkPreset::development(),
        ];

        for preset in presets {
            assert!(!preset.description.is_empty());
            assert!(!preset.name().is_empty());
        }
    }

    #[test]
    fn test_preset_names_unique() {
        let presets = vec![
            NetworkPreset::default_preset(),
            NetworkPreset::low_memory(),
            NetworkPreset::iot(),
            NetworkPreset::mobile(),
            NetworkPreset::high_performance(),
            NetworkPreset::low_latency(),
            NetworkPreset::high_throughput(),
            NetworkPreset::privacy(),
            NetworkPreset::development(),
        ];

        let names: Vec<&str> = presets.iter().map(|p| p.name()).collect();
        let mut unique_names = names.clone();
        unique_names.sort();
        unique_names.dedup();

        assert_eq!(
            names.len(),
            unique_names.len(),
            "All preset names should be unique"
        );
    }
}
