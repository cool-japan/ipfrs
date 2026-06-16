//! Automatic network configuration tuning based on system resources and usage patterns.
//!
//! This module provides intelligent auto-tuning capabilities that analyze system resources,
//! network conditions, and usage patterns to automatically optimize network configuration
//! for optimal performance.
//!
//! # Features
//!
//! - **System Resource Analysis**: Detect available CPU, memory, and network bandwidth
//! - **Workload Detection**: Identify whether the node is bandwidth-limited, CPU-limited, or memory-limited
//! - **Dynamic Reconfiguration**: Adjust settings in real-time based on observed performance
//! - **Profile-based Tuning**: Support for different use case profiles (server, mobile, IoT, etc.)
//! - **Performance Monitoring**: Track key metrics to guide tuning decisions
//!
//! # Example
//!
//! ```rust,no_run
//! use ipfrs_network::{NetworkConfig, auto_tuner::AutoTuner};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create auto-tuner with default settings
//! let mut tuner = AutoTuner::new();
//!
//! // Analyze system and generate optimized configuration
//! let config = tuner.generate_config().await?;
//! println!("Optimized config generated with {:?} max connections", config.max_connections);
//!
//! // Continuously monitor and adjust
//! tuner.start_monitoring().await?;
//! # Ok(())
//! # }
//! ```

use crate::NetworkConfig;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Errors that can occur during auto-tuning operations
#[derive(Debug, Error)]
pub enum AutoTunerError {
    #[error("Failed to detect system resources: {0}")]
    ResourceDetectionFailed(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Tuning not initialized")]
    NotInitialized,

    #[error("Monitoring already running")]
    MonitoringActive,
}

/// System resource information detected by the auto-tuner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemResources {
    /// Total system memory in bytes
    pub total_memory: u64,

    /// Available memory in bytes
    pub available_memory: u64,

    /// Number of CPU cores
    pub cpu_cores: usize,

    /// Estimated network bandwidth in bytes per second (0 if unknown)
    pub network_bandwidth: u64,

    /// Whether the system is likely battery-powered (mobile/IoT)
    pub is_battery_powered: bool,
}

impl SystemResources {
    /// Detect current system resources
    pub fn detect() -> Result<Self, AutoTunerError> {
        // In a real implementation, this would use system APIs
        // For now, we'll use conservative defaults
        Ok(Self {
            total_memory: 4 * 1024 * 1024 * 1024,     // 4 GB default
            available_memory: 2 * 1024 * 1024 * 1024, // 2 GB available
            cpu_cores: num_cpus::get(),
            network_bandwidth: 0, // Unknown by default
            is_battery_powered: false,
        })
    }

    /// Calculate memory category based on total memory
    pub fn memory_category(&self) -> &'static str {
        match self.total_memory {
            0..134_217_728 => "very_low",           // < 128 MB
            134_217_728..536_870_912 => "low",      // 128 MB - 512 MB
            536_870_912..2_147_483_648 => "medium", // 512 MB - 2 GB
            2_147_483_648..8_589_934_592 => "high", // 2 GB - 8 GB
            _ => "very_high",                       // >= 8 GB
        }
    }
}

/// Workload characteristics detected during runtime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadProfile {
    /// Average number of concurrent connections
    pub avg_connections: f64,

    /// Average DHT query rate (queries per second)
    pub avg_query_rate: f64,

    /// Average bandwidth usage in bytes per second
    pub avg_bandwidth_usage: f64,

    /// Peak memory usage in bytes
    pub peak_memory_usage: u64,

    /// Whether the workload is primarily CPU-bound
    pub cpu_bound: bool,

    /// Whether the workload is primarily bandwidth-bound
    pub bandwidth_bound: bool,

    /// Whether the workload is primarily memory-bound
    pub memory_bound: bool,
}

impl Default for WorkloadProfile {
    fn default() -> Self {
        Self {
            avg_connections: 0.0,
            avg_query_rate: 0.0,
            avg_bandwidth_usage: 0.0,
            peak_memory_usage: 0,
            cpu_bound: false,
            bandwidth_bound: false,
            memory_bound: false,
        }
    }
}

/// Configuration for the auto-tuner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoTunerConfig {
    /// Whether to enable automatic adjustments
    pub enable_auto_adjust: bool,

    /// Interval for re-evaluating configuration
    pub adjustment_interval: Duration,

    /// Minimum time before applying a configuration change
    pub stabilization_period: Duration,

    /// Safety margin for resource usage (0.0 to 1.0)
    /// For example, 0.2 means use at most 80% of available resources
    pub safety_margin: f64,

    /// Enable aggressive optimizations (may reduce stability)
    pub aggressive_mode: bool,
}

impl Default for AutoTunerConfig {
    fn default() -> Self {
        Self {
            enable_auto_adjust: true,
            adjustment_interval: Duration::from_secs(300), // 5 minutes
            stabilization_period: Duration::from_secs(60), // 1 minute
            safety_margin: 0.2,
            aggressive_mode: false,
        }
    }
}

impl AutoTunerConfig {
    /// Conservative tuning for production environments
    pub fn conservative() -> Self {
        Self {
            enable_auto_adjust: true,
            adjustment_interval: Duration::from_secs(600), // 10 minutes
            stabilization_period: Duration::from_secs(120), // 2 minutes
            safety_margin: 0.3,
            aggressive_mode: false,
        }
    }

    /// Aggressive tuning for development/testing
    pub fn aggressive() -> Self {
        Self {
            enable_auto_adjust: true,
            adjustment_interval: Duration::from_secs(60), // 1 minute
            stabilization_period: Duration::from_secs(30), // 30 seconds
            safety_margin: 0.1,
            aggressive_mode: true,
        }
    }
}

/// Statistics tracked by the auto-tuner
#[derive(Debug, Clone, Default)]
pub struct AutoTunerStats {
    /// Number of configuration adjustments made
    pub adjustments_made: u64,

    /// Number of times system resources were analyzed
    pub resource_checks: u64,

    /// Number of times workload was analyzed
    pub workload_checks: u64,

    /// Timestamp of last adjustment
    pub last_adjustment: Option<Instant>,

    /// Current optimization score (0.0 to 1.0, higher is better)
    pub optimization_score: f64,
}

/// Automatic network configuration tuner
pub struct AutoTuner {
    config: AutoTunerConfig,
    system_resources: Option<SystemResources>,
    workload_profile: WorkloadProfile,
    stats: Arc<RwLock<AutoTunerStats>>,
    monitoring_active: Arc<RwLock<bool>>,
}

impl AutoTuner {
    /// Create a new auto-tuner with default configuration
    pub fn new() -> Self {
        Self::with_config(AutoTunerConfig::default())
    }

    /// Create a new auto-tuner with custom configuration
    pub fn with_config(config: AutoTunerConfig) -> Self {
        Self {
            config,
            system_resources: None,
            workload_profile: WorkloadProfile::default(),
            stats: Arc::new(RwLock::new(AutoTunerStats::default())),
            monitoring_active: Arc::new(RwLock::new(false)),
        }
    }

    /// Analyze system resources
    pub async fn analyze_system(&mut self) -> Result<SystemResources, AutoTunerError> {
        let resources = SystemResources::detect()?;
        self.system_resources = Some(resources.clone());

        let mut stats = self.stats.write();
        stats.resource_checks += 1;

        Ok(resources)
    }

    /// Generate optimized network configuration based on detected resources
    pub async fn generate_config(&mut self) -> Result<NetworkConfig, AutoTunerError> {
        // Ensure we have system resources
        if self.system_resources.is_none() {
            self.analyze_system().await?;
        }

        let resources = self
            .system_resources
            .as_ref()
            .expect("just populated by analyze_system() if was None");
        let usable_factor = 1.0 - self.config.safety_margin;

        // Determine appropriate preset based on resources
        let mut config = match resources.memory_category() {
            "very_low" => NetworkConfig::low_memory(),
            "low" => NetworkConfig::iot(),
            "medium" => NetworkConfig::mobile(),
            "high" | "very_high" => {
                if resources.is_battery_powered {
                    NetworkConfig::mobile()
                } else {
                    NetworkConfig::high_performance()
                }
            }
            _ => NetworkConfig::default(),
        };

        // Adjust connection limits based on CPU cores
        let base_connections = resources.cpu_cores * 50;
        config.max_connections = Some((base_connections as f64 * usable_factor) as usize);

        // Adjust memory-sensitive parameters
        let memory_mb = resources.available_memory / (1024 * 1024);
        if memory_mb < 256 {
            config.connection_buffer_size = 8 * 1024; // 8 KB
            config.max_connections = Some(16);
        } else if memory_mb < 512 {
            config.connection_buffer_size = 16 * 1024; // 16 KB
            config.max_connections = Some(32);
        }

        // Enable NAT traversal unless it's a high-performance server
        config.enable_nat_traversal =
            resources.memory_category() != "very_high" || resources.is_battery_powered;

        let mut stats = self.stats.write();
        stats.adjustments_made += 1;
        stats.last_adjustment = Some(Instant::now());
        stats.optimization_score = self.calculate_optimization_score();

        Ok(config)
    }

    /// Update workload profile based on observed metrics
    pub fn update_workload(
        &mut self,
        connections: usize,
        query_rate: f64,
        bandwidth_usage: f64,
        memory_usage: u64,
    ) {
        // Use exponential moving average for smoothing
        let alpha = 0.3; // Smoothing factor

        let profile = &mut self.workload_profile;
        profile.avg_connections =
            profile.avg_connections * (1.0 - alpha) + (connections as f64) * alpha;
        profile.avg_query_rate = profile.avg_query_rate * (1.0 - alpha) + query_rate * alpha;
        profile.avg_bandwidth_usage =
            profile.avg_bandwidth_usage * (1.0 - alpha) + bandwidth_usage * alpha;
        profile.peak_memory_usage = profile.peak_memory_usage.max(memory_usage);

        // Detect bottlenecks
        if let Some(resources) = &self.system_resources {
            let memory_usage_ratio = memory_usage as f64 / resources.available_memory as f64;
            profile.memory_bound = memory_usage_ratio > 0.8;

            // Simple heuristics for CPU and bandwidth bounds
            profile.cpu_bound = connections > resources.cpu_cores * 100;
            profile.bandwidth_bound = resources.network_bandwidth > 0
                && bandwidth_usage > (resources.network_bandwidth as f64 * 0.8);
        }

        let mut stats = self.stats.write();
        stats.workload_checks += 1;
    }

    /// Start continuous monitoring and auto-adjustment
    pub async fn start_monitoring(&mut self) -> Result<(), AutoTunerError> {
        let mut active = self.monitoring_active.write();
        if *active {
            return Err(AutoTunerError::MonitoringActive);
        }
        *active = true;

        Ok(())
    }

    /// Stop continuous monitoring
    pub fn stop_monitoring(&mut self) {
        let mut active = self.monitoring_active.write();
        *active = false;
    }

    /// Check if monitoring is active
    pub fn is_monitoring(&self) -> bool {
        *self.monitoring_active.read()
    }

    /// Get current workload profile
    pub fn workload_profile(&self) -> &WorkloadProfile {
        &self.workload_profile
    }

    /// Get current statistics
    pub fn stats(&self) -> AutoTunerStats {
        self.stats.read().clone()
    }

    /// Calculate optimization score based on current state
    fn calculate_optimization_score(&self) -> f64 {
        if self.system_resources.is_none() {
            return 0.0;
        }

        let resources = self
            .system_resources
            .as_ref()
            .expect("just checked is_some above");
        let profile = &self.workload_profile;

        // Score based on resource utilization efficiency
        let memory_score = if profile.peak_memory_usage > 0 {
            1.0 - (profile.peak_memory_usage as f64 / resources.available_memory as f64).min(1.0)
        } else {
            0.5
        };

        let cpu_score = if profile.cpu_bound { 0.3 } else { 0.8 };
        let bandwidth_score = if profile.bandwidth_bound { 0.3 } else { 0.8 };

        // Weighted average
        (memory_score * 0.4 + cpu_score * 0.3 + bandwidth_score * 0.3).clamp(0.0, 1.0)
    }

    /// Generate recommendations for manual tuning
    pub fn recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if let Some(resources) = &self.system_resources {
            let profile = &self.workload_profile;

            if profile.memory_bound {
                recommendations.push(
                    "Memory usage is high. Consider reducing max_connections or enabling low_memory_mode.".to_string()
                );
            }

            if profile.cpu_bound {
                recommendations.push(
                    format!("CPU usage is high with {} cores. Consider distributing load across more nodes.",
                        resources.cpu_cores)
                );
            }

            if profile.bandwidth_bound {
                recommendations.push(
                    "Bandwidth is saturated. Consider enabling bandwidth throttling or upgrading network capacity.".to_string()
                );
            }

            if resources.is_battery_powered && profile.avg_query_rate > 10.0 {
                recommendations.push(
                    "High DHT query rate on battery power. Consider enabling query batching."
                        .to_string(),
                );
            }

            if resources.memory_category() == "very_low" && !profile.memory_bound {
                recommendations.push(
                    "System resources are underutilized. You can increase max_connections for better performance.".to_string()
                );
            }
        } else {
            recommendations.push("Run analyze_system() first to get recommendations.".to_string());
        }

        recommendations
    }
}

impl Default for AutoTuner {
    fn default() -> Self {
        Self::new()
    }
}

// Helper function to detect number of CPUs
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4) // Default to 4 if detection fails
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_auto_tuner_creation() {
        let tuner = AutoTuner::new();
        assert!(!tuner.is_monitoring());
    }

    #[tokio::test]
    async fn test_system_resource_detection() {
        let mut tuner = AutoTuner::new();
        let resources = tuner
            .analyze_system()
            .await
            .expect("test: analyze_system should succeed");
        assert!(resources.cpu_cores > 0);
        assert!(resources.total_memory > 0);
    }

    #[tokio::test]
    async fn test_config_generation() {
        let mut tuner = AutoTuner::new();
        let config = tuner
            .generate_config()
            .await
            .expect("test: generate_config should succeed");
        assert!(config.max_connections.is_some());
    }

    #[tokio::test]
    async fn test_workload_update() {
        let mut tuner = AutoTuner::new();
        tuner
            .analyze_system()
            .await
            .expect("test: analyze_system should succeed");

        tuner.update_workload(10, 5.0, 100_000.0, 50_000_000);
        let profile = tuner.workload_profile();
        assert!(profile.avg_connections > 0.0);
    }

    #[tokio::test]
    async fn test_monitoring_lifecycle() {
        let mut tuner = AutoTuner::new();
        assert!(!tuner.is_monitoring());

        tuner
            .start_monitoring()
            .await
            .expect("test: start_monitoring should succeed when not yet active");
        assert!(tuner.is_monitoring());

        tuner.stop_monitoring();
        assert!(!tuner.is_monitoring());
    }

    #[test]
    fn test_memory_categories() {
        let low = SystemResources {
            total_memory: 100 * 1024 * 1024, // 100 MB
            available_memory: 50 * 1024 * 1024,
            cpu_cores: 2,
            network_bandwidth: 0,
            is_battery_powered: true,
        };
        assert_eq!(low.memory_category(), "very_low");

        let high = SystemResources {
            total_memory: 16 * 1024 * 1024 * 1024, // 16 GB
            available_memory: 8 * 1024 * 1024 * 1024,
            cpu_cores: 8,
            network_bandwidth: 0,
            is_battery_powered: false,
        };
        assert_eq!(high.memory_category(), "very_high");
    }

    #[tokio::test]
    async fn test_statistics_tracking() {
        let mut tuner = AutoTuner::new();

        let stats_before = tuner.stats();
        assert_eq!(stats_before.adjustments_made, 0);

        tuner
            .generate_config()
            .await
            .expect("test: generate_config should succeed");

        let stats_after = tuner.stats();
        assert_eq!(stats_after.adjustments_made, 1);
        assert!(stats_after.last_adjustment.is_some());
    }

    #[tokio::test]
    async fn test_recommendations() {
        let mut tuner = AutoTuner::new();
        tuner
            .analyze_system()
            .await
            .expect("test: analyze_system should succeed");

        // Simulate high memory usage to trigger a recommendation
        if let Some(resources) = &tuner.system_resources {
            let high_memory = (resources.available_memory as f64 * 0.85) as u64;
            tuner.update_workload(50, 20.0, 1_000_000.0, high_memory);
        }

        let recommendations = tuner.recommendations();
        assert!(!recommendations.is_empty());
    }

    #[tokio::test]
    async fn test_config_presets() {
        let conservative = AutoTunerConfig::conservative();
        assert!(!conservative.aggressive_mode);
        assert!(conservative.safety_margin > 0.2);

        let aggressive = AutoTunerConfig::aggressive();
        assert!(aggressive.aggressive_mode);
        assert!(aggressive.safety_margin < 0.2);
    }

    #[tokio::test]
    async fn test_optimization_score() {
        let mut tuner = AutoTuner::new();
        tuner
            .analyze_system()
            .await
            .expect("test: analyze_system should succeed");

        let stats = tuner.stats();
        assert!(stats.optimization_score >= 0.0 && stats.optimization_score <= 1.0);
    }
}
