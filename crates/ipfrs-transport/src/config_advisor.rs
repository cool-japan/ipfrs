//! Configuration recommendation engine for transport layer optimization
//!
//! This module provides intelligent configuration recommendations based on
//! use case requirements, resource constraints, and network conditions.

use crate::{PeerScoringConfig, Priority, SessionConfig, WantListConfig};
use std::time::Duration;

/// Use case scenarios for configuration recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UseCase {
    /// Real-time streaming or interactive applications
    RealTime,
    /// Bulk data transfer or backup operations
    BulkTransfer,
    /// Machine learning model distribution
    MLDistribution,
    /// Content delivery and caching
    ContentDelivery,
    /// Edge computing or IoT devices
    EdgeComputing,
    /// Scientific computing and data analysis
    ScientificComputing,
    /// General purpose usage
    GeneralPurpose,
}

/// Resource constraint levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceLevel {
    /// Minimal resources (mobile, IoT)
    Minimal,
    /// Low resources (edge devices)
    Low,
    /// Moderate resources (standard servers)
    Moderate,
    /// High resources (data centers)
    High,
}

/// Network condition quality
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkQuality {
    /// Excellent connectivity (< 10ms latency, > 100 Mbps)
    Excellent,
    /// Good connectivity (10-50ms latency, 10-100 Mbps)
    Good,
    /// Fair connectivity (50-200ms latency, 1-10 Mbps)
    Fair,
    /// Poor connectivity (> 200ms latency, < 1 Mbps)
    Poor,
}

/// Configuration requirements for recommendation
#[derive(Debug, Clone)]
pub struct ConfigRequirements {
    /// Primary use case
    pub use_case: UseCase,
    /// Available resources
    pub resource_level: ResourceLevel,
    /// Expected network quality
    pub network_quality: NetworkQuality,
    /// Expected number of concurrent peers
    pub expected_peers: usize,
    /// Average block size in bytes
    pub avg_block_size: usize,
    /// Target latency requirement (if any)
    pub target_latency: Option<Duration>,
    /// Target throughput requirement in bytes/sec (if any)
    pub target_throughput: Option<u64>,
}

impl Default for ConfigRequirements {
    fn default() -> Self {
        Self {
            use_case: UseCase::GeneralPurpose,
            resource_level: ResourceLevel::Moderate,
            network_quality: NetworkQuality::Good,
            expected_peers: 10,
            avg_block_size: 256 * 1024, // 256 KB
            target_latency: None,
            target_throughput: None,
        }
    }
}

/// Recommended configuration bundle
#[derive(Debug, Clone)]
pub struct RecommendedConfig {
    /// Recommended want list configuration
    pub want_list: WantListConfig,
    /// Recommended peer scoring configuration
    pub peer_scoring: PeerScoringConfig,
    /// Recommended session configuration
    pub session: SessionConfig,
    /// Estimated memory usage in bytes
    pub estimated_memory: usize,
    /// Recommendation confidence (0.0 to 1.0)
    pub confidence: f64,
    /// Explanation of recommendations
    pub explanation: String,
}

/// Configuration advisor that recommends optimal settings
pub struct ConfigAdvisor;

impl ConfigAdvisor {
    /// Generate configuration recommendations based on requirements
    pub fn recommend(requirements: &ConfigRequirements) -> RecommendedConfig {
        let want_list = Self::recommend_want_list(requirements);
        let peer_scoring = Self::recommend_peer_scoring(requirements);
        let session = Self::recommend_session(requirements);

        let estimated_memory = Self::estimate_memory(requirements, &want_list);
        let confidence = Self::calculate_confidence(requirements);
        let explanation =
            Self::generate_explanation(requirements, &want_list, &peer_scoring, &session);

        RecommendedConfig {
            want_list,
            peer_scoring,
            session,
            estimated_memory,
            confidence,
            explanation,
        }
    }

    fn recommend_want_list(req: &ConfigRequirements) -> WantListConfig {
        let (max_wants, timeout, retries, base_delay, max_delay) =
            match (&req.use_case, &req.resource_level) {
                // Real-time with any resources
                (UseCase::RealTime, ResourceLevel::Minimal) => (200, 20, 2, 10, 5),
                (UseCase::RealTime, ResourceLevel::Low) => (500, 30, 2, 10, 10),
                (UseCase::RealTime, _) => (1000, 30, 3, 10, 15),

                // Bulk transfer
                (UseCase::BulkTransfer, ResourceLevel::Minimal) => (500, 60, 3, 50, 20),
                (UseCase::BulkTransfer, ResourceLevel::Low) => (2000, 90, 5, 50, 30),
                (UseCase::BulkTransfer, ResourceLevel::Moderate) => (10000, 120, 5, 50, 30),
                (UseCase::BulkTransfer, ResourceLevel::High) => (50000, 180, 10, 10, 30),

                // ML Distribution
                (UseCase::MLDistribution, ResourceLevel::Minimal) => (1000, 120, 3, 100, 30),
                (UseCase::MLDistribution, _) => (20000, 300, 8, 50, 60),

                // Content Delivery
                (UseCase::ContentDelivery, ResourceLevel::Minimal) => (500, 45, 3, 20, 15),
                (UseCase::ContentDelivery, _) => (5000, 90, 5, 20, 30),

                // Edge Computing
                (UseCase::EdgeComputing, _) => (500, 60, 2, 100, 30),

                // Scientific Computing
                (UseCase::ScientificComputing, ResourceLevel::High) => (50000, 600, 10, 10, 30),
                (UseCase::ScientificComputing, _) => (10000, 300, 8, 50, 60),

                // General Purpose
                (UseCase::GeneralPurpose, ResourceLevel::Minimal) => (500, 60, 2, 50, 20),
                (UseCase::GeneralPurpose, ResourceLevel::Low) => (1000, 60, 3, 50, 20),
                (UseCase::GeneralPurpose, ResourceLevel::Moderate) => (5000, 90, 5, 30, 30),
                (UseCase::GeneralPurpose, ResourceLevel::High) => (10000, 120, 5, 20, 30),
            };

        // Adjust for network quality
        let (timeout, retries) = match req.network_quality {
            NetworkQuality::Excellent => (timeout, retries),
            NetworkQuality::Good => (timeout + 30, retries),
            NetworkQuality::Fair => (timeout * 2, retries + 2),
            NetworkQuality::Poor => (timeout * 3, retries + 3),
        };

        WantListConfig {
            max_wants,
            default_timeout: Duration::from_secs(timeout),
            max_retries: retries,
            base_retry_delay: Duration::from_millis(base_delay),
            max_retry_delay: Duration::from_secs(max_delay),
        }
    }

    fn recommend_peer_scoring(req: &ConfigRequirements) -> PeerScoringConfig {
        let (lat_weight, bw_weight, rel_weight, alpha, decay, min_score, max_fail) =
            match req.use_case {
                UseCase::RealTime => (0.6, 0.2, 0.2, 0.35, 0.05, 0.2, 2),
                UseCase::BulkTransfer => (0.2, 0.6, 0.2, 0.2, 0.01, 0.05, 5),
                UseCase::MLDistribution => (0.3, 0.4, 0.3, 0.25, 0.02, 0.1, 4),
                UseCase::ContentDelivery => (0.4, 0.3, 0.3, 0.3, 0.03, 0.15, 3),
                UseCase::EdgeComputing => (0.4, 0.3, 0.3, 0.4, 0.1, 0.2, 2),
                UseCase::ScientificComputing => (0.25, 0.5, 0.25, 0.2, 0.01, 0.05, 6),
                UseCase::GeneralPurpose => (0.33, 0.34, 0.33, 0.25, 0.02, 0.1, 5),
            };

        PeerScoringConfig {
            latency_weight: lat_weight,
            bandwidth_weight: bw_weight,
            reliability_weight: rel_weight,
            ewma_alpha: alpha,
            inactivity_decay: decay,
            min_score,
            max_failures: max_fail,
        }
    }

    fn recommend_session(req: &ConfigRequirements) -> SessionConfig {
        let (timeout, priority, max_concurrent) = match req.use_case {
            UseCase::RealTime => (30, Priority::Urgent, 50),
            UseCase::BulkTransfer => (300, Priority::Normal, 500),
            UseCase::MLDistribution => (600, Priority::High, 1000),
            UseCase::ContentDelivery => (120, Priority::High, 200),
            UseCase::EdgeComputing => (60, Priority::Normal, 100),
            UseCase::ScientificComputing => (600, Priority::High, 1000),
            UseCase::GeneralPurpose => (120, Priority::Normal, 200),
        };

        // Adjust for resource level
        let max_concurrent = match req.resource_level {
            ResourceLevel::Minimal => max_concurrent / 4,
            ResourceLevel::Low => max_concurrent / 2,
            ResourceLevel::Moderate => max_concurrent,
            ResourceLevel::High => max_concurrent * 2,
        };

        SessionConfig {
            timeout: Duration::from_secs(timeout),
            default_priority: priority,
            max_concurrent_blocks: max_concurrent.max(10),
            progress_notifications: true,
        }
    }

    fn estimate_memory(req: &ConfigRequirements, want_list: &WantListConfig) -> usize {
        const BYTES_PER_WANT: usize = 100;
        const BYTES_PER_PEER: usize = 500;

        let want_list_mem = want_list.max_wants * BYTES_PER_WANT;
        let peer_mem = req.expected_peers * BYTES_PER_PEER;
        let overhead = 1024 * 1024; // 1 MB overhead

        want_list_mem + peer_mem + overhead
    }

    fn calculate_confidence(req: &ConfigRequirements) -> f64 {
        let mut confidence: f64 = 0.8;

        // Increase confidence for well-defined use cases
        if matches!(req.use_case, UseCase::RealTime | UseCase::BulkTransfer) {
            confidence += 0.1;
        }

        // Decrease confidence for extreme resource constraints
        if matches!(req.resource_level, ResourceLevel::Minimal) {
            confidence -= 0.1;
        }

        // Decrease confidence for poor network quality
        if matches!(req.network_quality, NetworkQuality::Poor) {
            confidence -= 0.15;
        }

        confidence.clamp(0.0, 1.0)
    }

    fn generate_explanation(
        req: &ConfigRequirements,
        want_list: &WantListConfig,
        peer_scoring: &PeerScoringConfig,
        session: &SessionConfig,
    ) -> String {
        let mut explanation = String::new();

        explanation.push_str(&format!(
            "Configuration optimized for {:?} use case.\n",
            req.use_case
        ));
        explanation.push_str(&format!(
            "Resource level: {:?}, Network quality: {:?}\n",
            req.resource_level, req.network_quality
        ));

        explanation.push_str("\nKey settings:\n");
        explanation.push_str(&format!(
            "- Want list: {} max entries, {}s timeout\n",
            want_list.max_wants,
            want_list.default_timeout.as_secs()
        ));
        explanation.push_str(&format!(
            "- Peer scoring: {:.0}% latency, {:.0}% bandwidth, {:.0}% reliability\n",
            peer_scoring.latency_weight * 100.0,
            peer_scoring.bandwidth_weight * 100.0,
            peer_scoring.reliability_weight * 100.0
        ));
        explanation.push_str(&format!(
            "- Session: {} concurrent blocks, {:?} priority\n",
            session.max_concurrent_blocks, session.default_priority
        ));

        explanation
    }
}

/// Performance profile for analyzing existing configurations
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    /// Estimated latency in milliseconds
    pub estimated_latency_ms: f64,
    /// Estimated throughput in bytes per second
    pub estimated_throughput_bps: u64,
    /// Memory efficiency score (0.0 to 1.0)
    pub memory_efficiency: f64,
    /// Network utilization score (0.0 to 1.0)
    pub network_utilization: f64,
    /// Overall efficiency score (0.0 to 1.0)
    pub overall_score: f64,
    /// Bottleneck analysis
    pub bottlenecks: Vec<String>,
}

impl ConfigAdvisor {
    /// Analyze the performance characteristics of a configuration
    pub fn analyze_performance(
        want_list: &WantListConfig,
        session: &SessionConfig,
        avg_block_size: usize,
        network_latency_ms: f64,
        bandwidth_bps: u64,
    ) -> PerformanceProfile {
        let mut bottlenecks = Vec::new();

        // Estimate latency
        let retry_overhead =
            want_list.base_retry_delay.as_secs_f64() * 1000.0 * want_list.max_retries as f64;
        let estimated_latency_ms = network_latency_ms + retry_overhead;

        // Estimate throughput
        let blocks_per_second =
            session.max_concurrent_blocks as f64 / (estimated_latency_ms / 1000.0);
        let estimated_throughput_bps = (blocks_per_second * avg_block_size as f64) as u64;

        // Check if bandwidth is the bottleneck
        if estimated_throughput_bps > bandwidth_bps {
            bottlenecks.push("Bandwidth is limiting throughput".to_string());
        }

        // Check if concurrency is too low
        if session.max_concurrent_blocks < 100 && bandwidth_bps > 100_000_000 {
            bottlenecks.push("Low concurrency limiting bandwidth utilization".to_string());
        }

        // Calculate memory efficiency
        let memory_per_block = 100; // Rough estimate
        let memory_usage = want_list.max_wants * memory_per_block;
        let memory_efficiency = if memory_usage > 10_000_000 {
            0.5 // Using more than 10 MB
        } else if memory_usage > 5_000_000 {
            0.7
        } else {
            1.0
        };

        // Calculate network utilization
        let actual_throughput = estimated_throughput_bps.min(bandwidth_bps);
        let network_utilization = (actual_throughput as f64 / bandwidth_bps as f64).min(1.0);

        // Overall score
        let overall_score = (memory_efficiency + network_utilization) / 2.0;

        PerformanceProfile {
            estimated_latency_ms,
            estimated_throughput_bps: actual_throughput,
            memory_efficiency,
            network_utilization,
            overall_score,
            bottlenecks,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_realtime_recommendation() {
        let req = ConfigRequirements {
            use_case: UseCase::RealTime,
            resource_level: ResourceLevel::Moderate,
            network_quality: NetworkQuality::Good,
            expected_peers: 10,
            avg_block_size: 64 * 1024,
            target_latency: Some(Duration::from_millis(100)),
            target_throughput: None,
        };

        let config = ConfigAdvisor::recommend(&req);

        // Real-time should have low timeout and high latency weight
        assert!(config.want_list.default_timeout.as_secs() <= 60);
        assert!(config.peer_scoring.latency_weight >= 0.5);
        assert_eq!(config.session.default_priority, Priority::Urgent);
    }

    #[test]
    fn test_bulk_transfer_recommendation() {
        let req = ConfigRequirements {
            use_case: UseCase::BulkTransfer,
            resource_level: ResourceLevel::High,
            network_quality: NetworkQuality::Excellent,
            expected_peers: 20,
            avg_block_size: 1024 * 1024,
            target_latency: None,
            target_throughput: Some(1_000_000_000),
        };

        let config = ConfigAdvisor::recommend(&req);

        // Bulk transfer should have high bandwidth weight and large want list
        assert!(config.peer_scoring.bandwidth_weight >= 0.5);
        assert!(config.want_list.max_wants >= 10000);
        assert!(config.session.max_concurrent_blocks >= 500);
    }

    #[test]
    fn test_edge_computing_recommendation() {
        let req = ConfigRequirements {
            use_case: UseCase::EdgeComputing,
            resource_level: ResourceLevel::Low,
            network_quality: NetworkQuality::Fair,
            expected_peers: 5,
            avg_block_size: 128 * 1024,
            target_latency: None,
            target_throughput: None,
        };

        let config = ConfigAdvisor::recommend(&req);

        // Edge should have low memory usage
        assert!(config.want_list.max_wants <= 1000);
        assert!(config.estimated_memory < 2_000_000); // Less than 2 MB (includes overhead)
    }

    #[test]
    fn test_network_quality_adjustment() {
        let req_good = ConfigRequirements {
            use_case: UseCase::GeneralPurpose,
            resource_level: ResourceLevel::Moderate,
            network_quality: NetworkQuality::Good,
            ..Default::default()
        };

        let req_poor = ConfigRequirements {
            network_quality: NetworkQuality::Poor,
            ..req_good.clone()
        };

        let config_good = ConfigAdvisor::recommend(&req_good);
        let config_poor = ConfigAdvisor::recommend(&req_poor);

        // Poor network should have longer timeout and more retries
        assert!(config_poor.want_list.default_timeout > config_good.want_list.default_timeout);
        assert!(config_poor.want_list.max_retries >= config_good.want_list.max_retries);
    }

    #[test]
    fn test_resource_level_adjustment() {
        let req_minimal = ConfigRequirements {
            use_case: UseCase::BulkTransfer,
            resource_level: ResourceLevel::Minimal,
            network_quality: NetworkQuality::Good,
            ..Default::default()
        };

        let req_high = ConfigRequirements {
            resource_level: ResourceLevel::High,
            ..req_minimal.clone()
        };

        let config_minimal = ConfigAdvisor::recommend(&req_minimal);
        let config_high = ConfigAdvisor::recommend(&req_high);

        // High resources should allow more concurrent operations
        assert!(
            config_high.session.max_concurrent_blocks
                > config_minimal.session.max_concurrent_blocks
        );
        assert!(config_high.want_list.max_wants > config_minimal.want_list.max_wants);
    }

    #[test]
    fn test_confidence_calculation() {
        let req_good = ConfigRequirements {
            use_case: UseCase::RealTime,
            resource_level: ResourceLevel::Moderate,
            network_quality: NetworkQuality::Good,
            ..Default::default()
        };

        let req_poor = ConfigRequirements {
            use_case: UseCase::GeneralPurpose,
            resource_level: ResourceLevel::Minimal,
            network_quality: NetworkQuality::Poor,
            ..Default::default()
        };

        let config_good = ConfigAdvisor::recommend(&req_good);
        let config_poor = ConfigAdvisor::recommend(&req_poor);

        // Good conditions should have higher confidence
        assert!(config_good.confidence > config_poor.confidence);
    }

    #[test]
    fn test_performance_analysis() {
        let want_list = WantListConfig {
            max_wants: 1000,
            default_timeout: Duration::from_secs(30),
            max_retries: 3,
            base_retry_delay: Duration::from_millis(10),
            max_retry_delay: Duration::from_secs(5),
        };

        let session = SessionConfig {
            timeout: Duration::from_secs(60),
            default_priority: Priority::Normal,
            max_concurrent_blocks: 100,
            progress_notifications: true,
        };

        let profile = ConfigAdvisor::analyze_performance(
            &want_list,
            &session,
            256 * 1024,  // 256 KB blocks
            50.0,        // 50ms latency
            100_000_000, // 100 Mbps
        );

        assert!(profile.estimated_latency_ms > 0.0);
        assert!(profile.estimated_throughput_bps > 0);
        assert!(profile.overall_score >= 0.0 && profile.overall_score <= 1.0);
    }

    #[test]
    fn test_bottleneck_detection() {
        let want_list = WantListConfig::default();
        let session = SessionConfig {
            timeout: Duration::from_secs(60),
            default_priority: Priority::Normal,
            max_concurrent_blocks: 10, // Very low
            progress_notifications: true,
        };

        let profile = ConfigAdvisor::analyze_performance(
            &want_list,
            &session,
            256 * 1024,
            10.0,
            1_000_000_000, // 1 Gbps available but low concurrency
        );

        // Should detect low concurrency as bottleneck
        assert!(!profile.bottlenecks.is_empty());
    }
}
