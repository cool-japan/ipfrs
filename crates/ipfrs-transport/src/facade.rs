//! High-level facade for easy transport system setup
//!
//! This module provides a simplified interface for setting up a complete
//! IPFRS transport system with integrated monitoring, diagnostics, and auto-tuning.
//!
//! # Example
//!
//! ```
//! use ipfrs_transport::TransportFacade;
//! use std::time::Duration;
//!
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! // Create a complete transport system with default settings
//! let transport = TransportFacade::builder()
//!     .with_monitoring()
//!     .with_diagnostics()
//!     .with_auto_tuning()
//!     .build();
//!
//! // Get health status
//! let health = transport.overall_health();
//! println!("System health: {:?}", health);
//! # });
//! ```

use crate::{
    AutoTuner, AutoTunerConfig, ComponentHealth, ComponentType, ConcurrentPeerManager,
    ConcurrentWantList, DiagnosticConfig, DiagnosticEngine, DiagnosticReport, HealthMonitor,
    HealthMonitorConfig, NetworkMetrics, PeerScoringConfig, StatsCollector, WantListConfig,
};
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// Preset configurations for different use cases
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportPreset {
    /// Optimized for low-latency operations
    LowLatency,
    /// Optimized for high-throughput bulk transfers
    HighThroughput,
    /// Balanced configuration for general use
    Balanced,
    /// Optimized for edge devices with limited resources
    EdgeDevice,
    /// Optimized for federated learning workloads
    FederatedLearning,
}

/// Configuration for the transport facade
#[derive(Debug, Clone)]
pub struct TransportFacadeConfig {
    /// Want list configuration
    pub want_list_config: WantListConfig,
    /// Peer scoring configuration
    pub peer_scoring_config: PeerScoringConfig,
    /// Enable health monitoring
    pub enable_monitoring: bool,
    /// Health monitor configuration
    pub monitor_config: Option<HealthMonitorConfig>,
    /// Enable diagnostics
    pub enable_diagnostics: bool,
    /// Diagnostic configuration
    pub diagnostic_config: Option<DiagnosticConfig>,
    /// Enable auto-tuning
    pub enable_auto_tuning: bool,
    /// Auto-tuner configuration
    pub auto_tuner_config: Option<AutoTunerConfig>,
    /// Enable statistics collection
    pub enable_stats: bool,
    /// Maximum statistics history
    pub max_stats_history: usize,
}

impl Default for TransportFacadeConfig {
    fn default() -> Self {
        Self::from_preset(TransportPreset::Balanced)
    }
}

impl TransportFacadeConfig {
    /// Create configuration from a preset
    pub fn from_preset(preset: TransportPreset) -> Self {
        let (want_list_config, peer_scoring_config) = match preset {
            TransportPreset::LowLatency => (
                WantListConfig {
                    max_wants: 1000,
                    default_timeout: Duration::from_secs(30),
                    max_retries: 3,
                    base_retry_delay: Duration::from_millis(10),
                    max_retry_delay: Duration::from_secs(5),
                },
                PeerScoringConfig {
                    latency_weight: 0.6,
                    bandwidth_weight: 0.2,
                    reliability_weight: 0.2,
                    ewma_alpha: 0.3,
                    inactivity_decay: 0.05,
                    min_score: 0.1,
                    max_failures: 3,
                },
            ),
            TransportPreset::HighThroughput => (
                WantListConfig {
                    max_wants: 10000,
                    default_timeout: Duration::from_secs(120),
                    max_retries: 5,
                    base_retry_delay: Duration::from_millis(50),
                    max_retry_delay: Duration::from_secs(10),
                },
                PeerScoringConfig {
                    latency_weight: 0.2,
                    bandwidth_weight: 0.6,
                    reliability_weight: 0.2,
                    ewma_alpha: 0.2,
                    inactivity_decay: 0.01,
                    min_score: 0.05,
                    max_failures: 5,
                },
            ),
            TransportPreset::Balanced => (
                WantListConfig {
                    max_wants: 5000,
                    default_timeout: Duration::from_secs(60),
                    max_retries: 3,
                    base_retry_delay: Duration::from_millis(100),
                    max_retry_delay: Duration::from_secs(5),
                },
                PeerScoringConfig {
                    latency_weight: 0.4,
                    bandwidth_weight: 0.4,
                    reliability_weight: 0.2,
                    ewma_alpha: 0.2,
                    inactivity_decay: 0.01,
                    min_score: 0.1,
                    max_failures: 5,
                },
            ),
            TransportPreset::EdgeDevice => (
                WantListConfig {
                    max_wants: 500,
                    default_timeout: Duration::from_secs(90),
                    max_retries: 5,
                    base_retry_delay: Duration::from_millis(200),
                    max_retry_delay: Duration::from_secs(10),
                },
                PeerScoringConfig {
                    latency_weight: 0.3,
                    bandwidth_weight: 0.4,
                    reliability_weight: 0.3,
                    ewma_alpha: 0.1,
                    inactivity_decay: 0.005,
                    min_score: 0.2,
                    max_failures: 7,
                },
            ),
            TransportPreset::FederatedLearning => (
                WantListConfig {
                    max_wants: 2000,
                    default_timeout: Duration::from_secs(180),
                    max_retries: 7,
                    base_retry_delay: Duration::from_millis(500),
                    max_retry_delay: Duration::from_secs(30),
                },
                PeerScoringConfig {
                    latency_weight: 0.3,
                    bandwidth_weight: 0.3,
                    reliability_weight: 0.4,
                    ewma_alpha: 0.15,
                    inactivity_decay: 0.002,
                    min_score: 0.3,
                    max_failures: 10,
                },
            ),
        };

        Self {
            want_list_config,
            peer_scoring_config,
            enable_monitoring: true,
            monitor_config: Some(HealthMonitorConfig::default()),
            enable_diagnostics: true,
            diagnostic_config: Some(DiagnosticConfig::default()),
            enable_auto_tuning: true,
            auto_tuner_config: Some(AutoTunerConfig::default()),
            enable_stats: true,
            max_stats_history: 1000,
        }
    }
}

/// Builder for constructing a transport facade
pub struct TransportFacadeBuilder {
    config: TransportFacadeConfig,
}

impl TransportFacadeBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: TransportFacadeConfig::default(),
        }
    }

    /// Create a builder from a preset
    pub fn from_preset(preset: TransportPreset) -> Self {
        Self {
            config: TransportFacadeConfig::from_preset(preset),
        }
    }

    /// Configure the want list
    pub fn want_list_config(mut self, config: WantListConfig) -> Self {
        self.config.want_list_config = config;
        self
    }

    /// Configure peer scoring
    pub fn peer_scoring_config(mut self, config: PeerScoringConfig) -> Self {
        self.config.peer_scoring_config = config;
        self
    }

    /// Enable health monitoring with default configuration
    pub fn with_monitoring(mut self) -> Self {
        self.config.enable_monitoring = true;
        if self.config.monitor_config.is_none() {
            self.config.monitor_config = Some(HealthMonitorConfig::default());
        }
        self
    }

    /// Enable health monitoring with custom configuration
    pub fn with_monitoring_config(mut self, config: HealthMonitorConfig) -> Self {
        self.config.enable_monitoring = true;
        self.config.monitor_config = Some(config);
        self
    }

    /// Enable diagnostics with default configuration
    pub fn with_diagnostics(mut self) -> Self {
        self.config.enable_diagnostics = true;
        if self.config.diagnostic_config.is_none() {
            self.config.diagnostic_config = Some(DiagnosticConfig::default());
        }
        self
    }

    /// Enable diagnostics with custom configuration
    pub fn with_diagnostics_config(mut self, config: DiagnosticConfig) -> Self {
        self.config.enable_diagnostics = true;
        self.config.diagnostic_config = Some(config);
        self
    }

    /// Enable auto-tuning with default configuration
    pub fn with_auto_tuning(mut self) -> Self {
        self.config.enable_auto_tuning = true;
        if self.config.auto_tuner_config.is_none() {
            self.config.auto_tuner_config = Some(AutoTunerConfig::default());
        }
        self
    }

    /// Enable auto-tuning with custom configuration
    pub fn with_auto_tuning_config(mut self, config: AutoTunerConfig) -> Self {
        self.config.enable_auto_tuning = true;
        self.config.auto_tuner_config = Some(config);
        self
    }

    /// Enable statistics collection
    pub fn with_stats(mut self, max_history: usize) -> Self {
        self.config.enable_stats = true;
        self.config.max_stats_history = max_history;
        self
    }

    /// Disable health monitoring
    pub fn without_monitoring(mut self) -> Self {
        self.config.enable_monitoring = false;
        self
    }

    /// Disable diagnostics
    pub fn without_diagnostics(mut self) -> Self {
        self.config.enable_diagnostics = false;
        self
    }

    /// Disable auto-tuning
    pub fn without_auto_tuning(mut self) -> Self {
        self.config.enable_auto_tuning = false;
        self
    }

    /// Build the transport facade
    pub fn build(self) -> TransportFacade {
        TransportFacade::new(self.config)
    }
}

impl Default for TransportFacadeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// High-level facade for the transport system
///
/// Provides integrated access to all transport components with
/// built-in monitoring, diagnostics, and auto-tuning.
pub struct TransportFacade {
    /// Want list for managing block requests
    want_list: ConcurrentWantList,
    /// Peer manager for peer selection and scoring
    peer_manager: ConcurrentPeerManager,
    /// Health monitor (optional)
    health_monitor: Option<Arc<HealthMonitor>>,
    /// Diagnostic engine (optional)
    diagnostic_engine: Option<DiagnosticEngine>,
    /// Auto-tuner (optional)
    auto_tuner: Option<Arc<RwLock<AutoTuner>>>,
    /// Statistics collector (optional)
    stats_collector: Option<Arc<RwLock<StatsCollector>>>,
    /// Configuration
    config: TransportFacadeConfig,
}

impl TransportFacade {
    /// Create a new transport facade with the given configuration
    pub fn new(config: TransportFacadeConfig) -> Self {
        let want_list = ConcurrentWantList::new(config.want_list_config.clone());
        let peer_manager = ConcurrentPeerManager::new(config.peer_scoring_config.clone());

        let health_monitor = if config.enable_monitoring {
            let monitor = HealthMonitor::new(config.monitor_config.clone().unwrap_or_default());
            // Register components
            monitor.register_component(ComponentType::WantList, 100);
            monitor.register_component(ComponentType::PeerManager, 100);
            Some(Arc::new(monitor))
        } else {
            None
        };

        let diagnostic_engine = if config.enable_diagnostics {
            Some(DiagnosticEngine::with_config(
                config.diagnostic_config.clone().unwrap_or_default(),
            ))
        } else {
            None
        };

        let auto_tuner = if config.enable_auto_tuning {
            Some(Arc::new(RwLock::new(AutoTuner::with_config(
                config.auto_tuner_config.clone().unwrap_or_default(),
            ))))
        } else {
            None
        };

        let stats_collector = if config.enable_stats {
            Some(Arc::new(RwLock::new(StatsCollector::new(
                config.max_stats_history,
            ))))
        } else {
            None
        };

        Self {
            want_list,
            peer_manager,
            health_monitor,
            diagnostic_engine,
            auto_tuner,
            stats_collector,
            config,
        }
    }

    /// Create a builder for constructing a transport facade
    pub fn builder() -> TransportFacadeBuilder {
        TransportFacadeBuilder::new()
    }

    /// Get a reference to the want list
    pub fn want_list(&self) -> &ConcurrentWantList {
        &self.want_list
    }

    /// Get a reference to the peer manager
    pub fn peer_manager(&self) -> &ConcurrentPeerManager {
        &self.peer_manager
    }

    /// Get the overall system health
    pub fn overall_health(&self) -> ComponentHealth {
        self.health_monitor
            .as_ref()
            .map(|m| m.overall_health())
            .unwrap_or(ComponentHealth::Unknown)
    }

    /// Get health status for a specific component
    pub fn component_health(&self, component: ComponentType) -> ComponentHealth {
        self.health_monitor
            .as_ref()
            .map(|m| m.get_health(component))
            .unwrap_or(ComponentHealth::Unknown)
    }

    /// Run diagnostics and get a comprehensive report
    pub fn run_diagnostics(&self) -> Option<DiagnosticReport> {
        self.diagnostic_engine
            .as_ref()
            .map(|engine| engine.generate_report(&self.want_list, &self.peer_manager, &[]))
    }

    /// Update network metrics for auto-tuning
    pub fn update_network_metrics(&self, metrics: NetworkMetrics) -> bool {
        if let Some(tuner) = &self.auto_tuner {
            let mut tuner = tuner.write().unwrap_or_else(|e| e.into_inner());
            tuner.update_metrics(metrics)
        } else {
            false
        }
    }

    /// Get current auto-tuning recommendations
    pub fn get_tuning_recommendations(&self) -> Option<Vec<String>> {
        self.auto_tuner.as_ref().map(|tuner| {
            let tuner = tuner.read().unwrap_or_else(|e| e.into_inner());
            tuner.get_recommendations()
        })
    }

    /// Record aggregated statistics
    pub fn record_stats(&self) {
        if let Some(collector) = &self.stats_collector {
            let aggregated = crate::AggregatedStatsBuilder::new()
                .peer_stats(self.peer_manager.stats())
                .build();

            let mut collector = collector.write().unwrap_or_else(|e| e.into_inner());
            collector.record(aggregated);
        }
    }

    /// Get the latest statistics
    pub fn latest_stats(&self) -> Option<crate::AggregatedStats> {
        self.stats_collector.as_ref().and_then(|collector| {
            let collector = collector.read().unwrap_or_else(|e| e.into_inner());
            collector.latest().cloned()
        })
    }

    /// Get average throughput over history
    pub fn avg_throughput(&self) -> u64 {
        self.stats_collector
            .as_ref()
            .map(|collector| {
                let collector = collector.read().unwrap_or_else(|e| e.into_inner());
                collector.avg_throughput()
            })
            .unwrap_or(0)
    }

    /// Get health monitor reference (if enabled)
    pub fn health_monitor(&self) -> Option<Arc<HealthMonitor>> {
        self.health_monitor.clone()
    }

    /// Get auto-tuner reference (if enabled)
    pub fn auto_tuner(&self) -> Option<Arc<RwLock<AutoTuner>>> {
        self.auto_tuner.clone()
    }

    /// Get stats collector reference (if enabled)
    pub fn stats_collector(&self) -> Option<Arc<RwLock<StatsCollector>>> {
        self.stats_collector.clone()
    }

    /// Get configuration
    pub fn config(&self) -> &TransportFacadeConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_default() {
        let facade = TransportFacade::builder().build();
        // Default builder enables monitoring, so health should be Healthy
        assert!(matches!(
            facade.overall_health(),
            ComponentHealth::Healthy | ComponentHealth::Unknown
        ));
    }

    #[test]
    fn test_builder_with_monitoring() {
        let facade = TransportFacade::builder().with_monitoring().build();

        assert!(facade.health_monitor.is_some());
        assert_ne!(facade.overall_health(), ComponentHealth::Unknown);
    }

    #[test]
    fn test_builder_with_diagnostics() {
        let facade = TransportFacade::builder().with_diagnostics().build();

        assert!(facade.diagnostic_engine.is_some());
        let report = facade.run_diagnostics();
        assert!(report.is_some());
    }

    #[test]
    fn test_builder_with_auto_tuning() {
        let facade = TransportFacade::builder().with_auto_tuning().build();

        assert!(facade.auto_tuner.is_some());
        let recommendations = facade.get_tuning_recommendations();
        assert!(recommendations.is_some());
    }

    #[test]
    fn test_builder_with_stats() {
        let facade = TransportFacade::builder().with_stats(100).build();

        assert!(facade.stats_collector.is_some());
        facade.record_stats();
        assert!(facade.latest_stats().is_some());
    }

    #[test]
    fn test_builder_without_monitoring() {
        let facade = TransportFacade::builder().without_monitoring().build();

        assert!(facade.health_monitor.is_none());
        assert_eq!(facade.overall_health(), ComponentHealth::Unknown);
    }

    #[test]
    fn test_preset_low_latency() {
        let facade = TransportFacadeBuilder::from_preset(TransportPreset::LowLatency).build();

        assert_eq!(facade.config.want_list_config.max_wants, 1000);
        assert_eq!(facade.config.peer_scoring_config.latency_weight, 0.6);
    }

    #[test]
    fn test_preset_high_throughput() {
        let facade = TransportFacadeBuilder::from_preset(TransportPreset::HighThroughput).build();

        assert_eq!(facade.config.want_list_config.max_wants, 10000);
        assert_eq!(facade.config.peer_scoring_config.bandwidth_weight, 0.6);
    }

    #[test]
    fn test_preset_edge_device() {
        let facade = TransportFacadeBuilder::from_preset(TransportPreset::EdgeDevice).build();

        assert_eq!(facade.config.want_list_config.max_wants, 500);
        assert_eq!(facade.config.peer_scoring_config.max_failures, 7);
    }

    #[test]
    fn test_update_network_metrics() {
        let facade = TransportFacade::builder().with_auto_tuning().build();

        let metrics = NetworkMetrics {
            avg_latency: Duration::from_millis(100),
            latency_stddev: Duration::from_millis(10),
            avg_bandwidth: 1_000_000,
            packet_loss_rate: 0.01,
            success_rate: 0.95,
            active_peers: 5,
        };

        facade.update_network_metrics(metrics);
        let recommendations = facade.get_tuning_recommendations();
        assert!(recommendations.is_some());
        assert!(!recommendations
            .expect("test: tuning recommendations should be present after metrics update")
            .is_empty());
    }

    #[test]
    fn test_record_and_get_stats() {
        let facade = TransportFacade::builder().with_stats(100).build();

        facade.record_stats();
        facade.record_stats();

        let stats = facade.latest_stats();
        assert!(stats.is_some());

        let avg = facade.avg_throughput();
        assert_eq!(avg, 0); // No actual data transferred
    }

    #[test]
    fn test_all_features_enabled() {
        let facade = TransportFacade::builder()
            .with_monitoring()
            .with_diagnostics()
            .with_auto_tuning()
            .with_stats(100)
            .build();

        assert!(facade.health_monitor.is_some());
        assert!(facade.diagnostic_engine.is_some());
        assert!(facade.auto_tuner.is_some());
        assert!(facade.stats_collector.is_some());

        // Test all features work together
        facade.record_stats();
        let _health = facade.overall_health();
        let _report = facade.run_diagnostics();
        let _recommendations = facade.get_tuning_recommendations();
    }
}
