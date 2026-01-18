//! Automatic configuration tuning based on network conditions
//!
//! This module provides adaptive tuning of transport parameters
//! to optimize performance under varying network conditions.

use std::time::Duration;

/// Network conditions detected by the auto-tuner
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NetworkCondition {
    /// Excellent network (low latency, high bandwidth, low loss)
    Excellent,
    /// Good network (moderate latency, good bandwidth)
    Good,
    /// Fair network (higher latency or moderate loss)
    Fair,
    /// Poor network (high latency, low bandwidth, or high loss)
    Poor,
    /// Very poor network (extreme conditions)
    VeryPoor,
}

/// Tuning profile optimized for specific network conditions
#[derive(Debug, Clone)]
pub struct TuningProfile {
    /// Profile name
    pub name: String,
    /// Network condition this profile is for
    pub condition: NetworkCondition,
    /// Recommended max concurrent blocks
    pub max_concurrent_blocks: usize,
    /// Recommended want list timeout
    pub want_timeout: Duration,
    /// Recommended retry count
    pub max_retries: usize,
    /// Recommended base retry delay
    pub base_retry_delay: Duration,
    /// Recommended batch size
    pub batch_size: usize,
    /// Recommended QUIC initial window size
    pub quic_initial_window: usize,
    /// Enable/disable pipelining
    pub enable_pipelining: bool,
    /// Prefetch depth
    pub prefetch_depth: usize,
}

impl TuningProfile {
    /// Create a profile for excellent network conditions
    pub fn excellent() -> Self {
        Self {
            name: "Excellent".to_string(),
            condition: NetworkCondition::Excellent,
            max_concurrent_blocks: 1000,
            want_timeout: Duration::from_secs(30),
            max_retries: 2,
            base_retry_delay: Duration::from_millis(50),
            batch_size: 100,
            quic_initial_window: 10 * 1024 * 1024, // 10 MB
            enable_pipelining: true,
            prefetch_depth: 10,
        }
    }

    /// Create a profile for good network conditions
    pub fn good() -> Self {
        Self {
            name: "Good".to_string(),
            condition: NetworkCondition::Good,
            max_concurrent_blocks: 500,
            want_timeout: Duration::from_secs(60),
            max_retries: 3,
            base_retry_delay: Duration::from_millis(100),
            batch_size: 50,
            quic_initial_window: 5 * 1024 * 1024, // 5 MB
            enable_pipelining: true,
            prefetch_depth: 5,
        }
    }

    /// Create a profile for fair network conditions
    pub fn fair() -> Self {
        Self {
            name: "Fair".to_string(),
            condition: NetworkCondition::Fair,
            max_concurrent_blocks: 200,
            want_timeout: Duration::from_secs(90),
            max_retries: 4,
            base_retry_delay: Duration::from_millis(200),
            batch_size: 20,
            quic_initial_window: 2 * 1024 * 1024, // 2 MB
            enable_pipelining: false,
            prefetch_depth: 3,
        }
    }

    /// Create a profile for poor network conditions
    pub fn poor() -> Self {
        Self {
            name: "Poor".to_string(),
            condition: NetworkCondition::Poor,
            max_concurrent_blocks: 50,
            want_timeout: Duration::from_secs(120),
            max_retries: 5,
            base_retry_delay: Duration::from_millis(500),
            batch_size: 10,
            quic_initial_window: 1024 * 1024, // 1 MB
            enable_pipelining: false,
            prefetch_depth: 1,
        }
    }

    /// Create a profile for very poor network conditions
    pub fn very_poor() -> Self {
        Self {
            name: "VeryPoor".to_string(),
            condition: NetworkCondition::VeryPoor,
            max_concurrent_blocks: 20,
            want_timeout: Duration::from_secs(180),
            max_retries: 7,
            base_retry_delay: Duration::from_secs(1),
            batch_size: 5,
            quic_initial_window: 512 * 1024, // 512 KB
            enable_pipelining: false,
            prefetch_depth: 0,
        }
    }
}

/// Network metrics used for auto-tuning
#[derive(Debug, Clone, Copy)]
pub struct NetworkMetrics {
    /// Average latency (round-trip time)
    pub avg_latency: Duration,
    /// Latency standard deviation
    pub latency_stddev: Duration,
    /// Average bandwidth (bytes/sec)
    pub avg_bandwidth: u64,
    /// Packet loss rate (0.0 to 1.0)
    pub packet_loss_rate: f64,
    /// Request success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Number of active peers
    pub active_peers: usize,
}

impl NetworkMetrics {
    /// Create default metrics (unknown conditions)
    pub fn unknown() -> Self {
        Self {
            avg_latency: Duration::from_millis(100),
            latency_stddev: Duration::from_millis(10),
            avg_bandwidth: 1_000_000, // 1 MB/s
            packet_loss_rate: 0.0,
            success_rate: 1.0,
            active_peers: 5,
        }
    }
}

/// Automatic tuning engine
pub struct AutoTuner {
    /// Current network metrics
    current_metrics: NetworkMetrics,
    /// Current tuning profile
    current_profile: TuningProfile,
    /// Configuration
    config: AutoTunerConfig,
}

/// Auto-tuner configuration
#[derive(Debug, Clone)]
pub struct AutoTunerConfig {
    /// Enable automatic tuning
    pub enabled: bool,
    /// Minimum time between profile changes
    pub min_profile_change_interval: Duration,
    /// Latency threshold for "excellent" (ms)
    pub excellent_latency_ms: u64,
    /// Latency threshold for "good" (ms)
    pub good_latency_ms: u64,
    /// Latency threshold for "fair" (ms)
    pub fair_latency_ms: u64,
    /// Latency threshold for "poor" (ms)
    pub poor_latency_ms: u64,
    /// Bandwidth threshold for "excellent" (bytes/sec)
    pub excellent_bandwidth: u64,
    /// Bandwidth threshold for "good" (bytes/sec)
    pub good_bandwidth: u64,
    /// Loss rate threshold for degradation
    pub max_acceptable_loss: f64,
}

impl Default for AutoTunerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_profile_change_interval: Duration::from_secs(30),
            excellent_latency_ms: 20,
            good_latency_ms: 50,
            fair_latency_ms: 150,
            poor_latency_ms: 500,
            excellent_bandwidth: 10_000_000, // 10 MB/s
            good_bandwidth: 5_000_000,       // 5 MB/s
            max_acceptable_loss: 0.01,       // 1%
        }
    }
}

impl AutoTuner {
    /// Create a new auto-tuner with default configuration
    pub fn new() -> Self {
        Self::with_config(AutoTunerConfig::default())
    }

    /// Create a new auto-tuner with custom configuration
    pub fn with_config(config: AutoTunerConfig) -> Self {
        let metrics = NetworkMetrics::unknown();
        let profile = Self::select_profile(&config, &metrics);

        Self {
            current_metrics: metrics,
            current_profile: profile,
            config,
        }
    }

    /// Update network metrics and potentially adjust tuning
    pub fn update_metrics(&mut self, metrics: NetworkMetrics) -> bool {
        self.current_metrics = metrics;

        if !self.config.enabled {
            return false;
        }

        let new_profile = Self::select_profile(&self.config, &metrics);

        // Check if we should change profiles
        if new_profile.condition != self.current_profile.condition {
            self.current_profile = new_profile;
            true // Profile changed
        } else {
            false // No change
        }
    }

    /// Get the current tuning profile
    pub fn current_profile(&self) -> &TuningProfile {
        &self.current_profile
    }

    /// Get the current network condition
    pub fn current_condition(&self) -> NetworkCondition {
        self.current_profile.condition
    }

    /// Select appropriate profile based on metrics
    fn select_profile(config: &AutoTunerConfig, metrics: &NetworkMetrics) -> TuningProfile {
        let condition = Self::determine_condition(config, metrics);

        match condition {
            NetworkCondition::Excellent => TuningProfile::excellent(),
            NetworkCondition::Good => TuningProfile::good(),
            NetworkCondition::Fair => TuningProfile::fair(),
            NetworkCondition::Poor => TuningProfile::poor(),
            NetworkCondition::VeryPoor => TuningProfile::very_poor(),
        }
    }

    /// Determine network condition from metrics
    fn determine_condition(config: &AutoTunerConfig, metrics: &NetworkMetrics) -> NetworkCondition {
        let latency_ms = metrics.avg_latency.as_millis() as u64;

        // Check for very poor conditions first
        if metrics.packet_loss_rate > 0.1 || // > 10% loss
           metrics.success_rate < 0.7 ||     // < 70% success
           latency_ms > config.poor_latency_ms * 2
        {
            return NetworkCondition::VeryPoor;
        }

        // Check for poor conditions
        if metrics.packet_loss_rate > config.max_acceptable_loss * 5.0
            || metrics.success_rate < 0.85
            || latency_ms > config.poor_latency_ms
        {
            return NetworkCondition::Poor;
        }

        // Check for fair conditions
        if metrics.packet_loss_rate > config.max_acceptable_loss * 2.0
            || latency_ms > config.fair_latency_ms
            || metrics.avg_bandwidth < config.good_bandwidth / 2
        {
            return NetworkCondition::Fair;
        }

        // Check for good conditions
        if latency_ms > config.good_latency_ms || metrics.avg_bandwidth < config.excellent_bandwidth
        {
            return NetworkCondition::Good;
        }

        // Excellent conditions
        NetworkCondition::Excellent
    }

    /// Get tuning recommendations as a human-readable string
    pub fn get_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        let profile = &self.current_profile;

        recommendations.push(format!("Network condition: {:?}", self.current_condition()));
        recommendations.push(format!(
            "Max concurrent blocks: {}",
            profile.max_concurrent_blocks
        ));
        recommendations.push(format!("Want timeout: {:?}", profile.want_timeout));
        recommendations.push(format!("Max retries: {}", profile.max_retries));
        recommendations.push(format!("Batch size: {}", profile.batch_size));
        recommendations.push(format!(
            "Pipelining: {}",
            if profile.enable_pipelining {
                "enabled"
            } else {
                "disabled"
            }
        ));

        // Add specific recommendations based on metrics
        if self.current_metrics.packet_loss_rate > 0.05 {
            recommendations.push(format!(
                "High packet loss detected ({:.1}%). Consider reducing batch size or enabling error correction.",
                self.current_metrics.packet_loss_rate * 100.0
            ));
        }

        if self.current_metrics.avg_latency > Duration::from_millis(200) {
            recommendations.push(
                "High latency detected. Consider reducing concurrent requests and enabling adaptive timeouts.".to_string()
            );
        }

        if self.current_metrics.active_peers < 3 {
            recommendations.push(
                "Low peer count. Consider connecting to more peers for redundancy.".to_string(),
            );
        }

        recommendations
    }
}

impl Default for AutoTuner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tuning_profiles() {
        let excellent = TuningProfile::excellent();
        assert_eq!(excellent.condition, NetworkCondition::Excellent);
        assert!(excellent.max_concurrent_blocks > 500);

        let poor = TuningProfile::poor();
        assert_eq!(poor.condition, NetworkCondition::Poor);
        assert!(poor.max_concurrent_blocks < 100);
    }

    #[test]
    fn test_auto_tuner_creation() {
        let tuner = AutoTuner::new();
        assert!(tuner.config.enabled);
    }

    #[test]
    fn test_determine_excellent_condition() {
        let config = AutoTunerConfig::default();
        let metrics = NetworkMetrics {
            avg_latency: Duration::from_millis(15),
            latency_stddev: Duration::from_millis(2),
            avg_bandwidth: 15_000_000,
            packet_loss_rate: 0.001,
            success_rate: 0.99,
            active_peers: 10,
        };

        let condition = AutoTuner::determine_condition(&config, &metrics);
        assert_eq!(condition, NetworkCondition::Excellent);
    }

    #[test]
    fn test_determine_poor_condition() {
        let config = AutoTunerConfig::default();
        let metrics = NetworkMetrics {
            avg_latency: Duration::from_millis(600),
            latency_stddev: Duration::from_millis(100),
            avg_bandwidth: 500_000,
            packet_loss_rate: 0.08,
            success_rate: 0.80,
            active_peers: 2,
        };

        let condition = AutoTuner::determine_condition(&config, &metrics);
        assert_eq!(condition, NetworkCondition::Poor);
    }

    #[test]
    fn test_determine_very_poor_condition() {
        let config = AutoTunerConfig::default();
        let metrics = NetworkMetrics {
            avg_latency: Duration::from_secs(2),
            latency_stddev: Duration::from_millis(500),
            avg_bandwidth: 100_000,
            packet_loss_rate: 0.15,
            success_rate: 0.60,
            active_peers: 1,
        };

        let condition = AutoTuner::determine_condition(&config, &metrics);
        assert_eq!(condition, NetworkCondition::VeryPoor);
    }

    #[test]
    fn test_update_metrics() {
        let mut tuner = AutoTuner::new();

        // Start with good metrics
        let good_metrics = NetworkMetrics {
            avg_latency: Duration::from_millis(30),
            latency_stddev: Duration::from_millis(5),
            avg_bandwidth: 8_000_000,
            packet_loss_rate: 0.005,
            success_rate: 0.98,
            active_peers: 8,
        };

        tuner.update_metrics(good_metrics);
        // Might or might not change depending on initial state
        let initial_condition = tuner.current_condition();

        // Update to poor metrics
        let poor_metrics = NetworkMetrics {
            avg_latency: Duration::from_millis(600),
            latency_stddev: Duration::from_millis(100),
            avg_bandwidth: 500_000,
            packet_loss_rate: 0.08,
            success_rate: 0.80,
            active_peers: 2,
        };

        let changed = tuner.update_metrics(poor_metrics);
        if initial_condition != NetworkCondition::Poor
            && initial_condition != NetworkCondition::VeryPoor
        {
            assert!(changed);
        }
        assert_eq!(tuner.current_condition(), NetworkCondition::Poor);
    }

    #[test]
    fn test_profile_selection() {
        let config = AutoTunerConfig::default();

        let excellent_metrics = NetworkMetrics {
            avg_latency: Duration::from_millis(10),
            latency_stddev: Duration::from_millis(1),
            avg_bandwidth: 20_000_000,
            packet_loss_rate: 0.0,
            success_rate: 1.0,
            active_peers: 10,
        };

        let profile = AutoTuner::select_profile(&config, &excellent_metrics);
        assert_eq!(profile.condition, NetworkCondition::Excellent);
        assert!(profile.enable_pipelining);
    }

    #[test]
    fn test_recommendations() {
        let mut tuner = AutoTuner::new();

        let metrics = NetworkMetrics {
            avg_latency: Duration::from_millis(250),
            latency_stddev: Duration::from_millis(50),
            avg_bandwidth: 2_000_000,
            packet_loss_rate: 0.06,
            success_rate: 0.92,
            active_peers: 2,
        };

        tuner.update_metrics(metrics);
        let recommendations = tuner.get_recommendations();

        assert!(!recommendations.is_empty());
        assert!(recommendations
            .iter()
            .any(|r| r.contains("packet loss") || r.contains("latency") || r.contains("peer")));
    }

    #[test]
    fn test_disabled_tuner() {
        let config = AutoTunerConfig {
            enabled: false,
            ..Default::default()
        };

        let mut tuner = AutoTuner::with_config(config);
        let _initial_condition = tuner.current_condition();

        let metrics = NetworkMetrics {
            avg_latency: Duration::from_secs(5),
            latency_stddev: Duration::from_secs(1),
            avg_bandwidth: 100,
            packet_loss_rate: 0.5,
            success_rate: 0.1,
            active_peers: 1,
        };

        let changed = tuner.update_metrics(metrics);
        assert!(!changed); // Should not change when disabled
    }

    #[test]
    fn test_network_metrics_unknown() {
        let metrics = NetworkMetrics::unknown();
        assert!(metrics.avg_bandwidth > 0);
        assert!(metrics.success_rate > 0.0);
    }
}
