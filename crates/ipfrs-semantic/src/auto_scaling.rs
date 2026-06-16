//! Auto-scaling advisor for production deployments
//!
//! This module provides intelligent recommendations for scaling semantic search
//! systems based on observed metrics and workload patterns.
//!
//! # Features
//!
//! - **Load Analysis**: Analyze query load and resource utilization
//! - **Scaling Recommendations**: Suggest horizontal/vertical scaling
//! - **Cost Estimation**: Estimate infrastructure costs
//! - **Performance Prediction**: Predict performance under different configurations
//!
//! # Example
//!
//! ```rust
//! use ipfrs_semantic::auto_scaling::{AutoScalingAdvisor, WorkloadMetrics};
//! use std::time::Duration;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut advisor = AutoScalingAdvisor::new();
//!
//! // Record workload metrics
//! let metrics = WorkloadMetrics {
//!     queries_per_second: 1500.0,
//!     avg_latency: Duration::from_millis(10),
//!     p99_latency: Duration::from_millis(50),
//!     memory_usage_mb: 4096.0,
//!     cpu_utilization: 0.85,
//!     cache_hit_rate: 0.60,
//!     index_size: 10_000_000,
//! };
//!
//! // Get scaling recommendations
//! let recommendations = advisor.analyze(&metrics)?;
//! for rec in &recommendations.actions {
//!     println!("📊 {}: {}", rec.action_type, rec.description);
//!     println!("   Impact: {}", rec.expected_impact);
//! }
//! # Ok(())
//! # }
//! ```

use ipfrs_core::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Workload metrics for a semantic search system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadMetrics {
    /// Queries per second
    pub queries_per_second: f64,
    /// Average query latency
    pub avg_latency: Duration,
    /// P99 latency
    pub p99_latency: Duration,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// CPU utilization (0.0 to 1.0)
    pub cpu_utilization: f64,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,
    /// Total index size (number of vectors)
    pub index_size: usize,
}

/// Scaling action type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionType {
    /// Increase cache size
    IncreaseCache,
    /// Add more replicas
    ScaleHorizontally,
    /// Increase CPU/memory
    ScaleVertically,
    /// Optimize index parameters
    OptimizeParameters,
    /// Enable compression/quantization
    EnableCompression,
    /// Add warmup cache
    AddWarmupCache,
    /// No action needed
    NoAction,
}

impl std::fmt::Display for ActionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActionType::IncreaseCache => write!(f, "Increase Cache"),
            ActionType::ScaleHorizontally => write!(f, "Scale Horizontally"),
            ActionType::ScaleVertically => write!(f, "Scale Vertically"),
            ActionType::OptimizeParameters => write!(f, "Optimize Parameters"),
            ActionType::EnableCompression => write!(f, "Enable Compression"),
            ActionType::AddWarmupCache => write!(f, "Add Warmup Cache"),
            ActionType::NoAction => write!(f, "No Action"),
        }
    }
}

/// A specific scaling recommendation
#[derive(Debug, Clone)]
pub struct ScalingAction {
    /// Type of action
    pub action_type: ActionType,
    /// Priority (0.0 to 1.0, where 1.0 is highest)
    pub priority: f64,
    /// Description of the action
    pub description: String,
    /// Expected impact
    pub expected_impact: String,
    /// Estimated cost (relative units)
    pub cost_estimate: f64,
}

/// Scaling recommendations report
#[derive(Debug, Clone)]
pub struct ScalingRecommendations {
    /// Current system health score (0.0 to 1.0)
    pub health_score: f64,
    /// Predicted capacity before overload
    pub capacity_headroom: f64,
    /// Recommended actions
    pub actions: Vec<ScalingAction>,
    /// Cost-benefit analysis
    pub cost_benefit_ratio: f64,
}

/// Configuration for auto-scaling advisor
#[derive(Debug, Clone)]
pub struct AdvisorConfig {
    /// Target P99 latency threshold (ms)
    pub target_p99_latency_ms: u64,
    /// Target CPU utilization (0.0 to 1.0)
    pub target_cpu_utilization: f64,
    /// Minimum cache hit rate
    pub min_cache_hit_rate: f64,
    /// Target queries per second capacity
    pub target_qps_capacity: f64,
}

impl Default for AdvisorConfig {
    fn default() -> Self {
        Self {
            target_p99_latency_ms: 100,   // 100ms P99
            target_cpu_utilization: 0.70, // 70% CPU target
            min_cache_hit_rate: 0.75,     // 75% cache hit rate
            target_qps_capacity: 1000.0,  // 1000 QPS
        }
    }
}

/// Auto-scaling advisor
pub struct AutoScalingAdvisor {
    /// Configuration
    config: AdvisorConfig,
    /// Historical metrics
    history: Vec<WorkloadMetrics>,
}

impl AutoScalingAdvisor {
    /// Create a new advisor with default config
    pub fn new() -> Self {
        Self {
            config: AdvisorConfig::default(),
            history: Vec::new(),
        }
    }

    /// Create an advisor with custom config
    pub fn with_config(config: AdvisorConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
        }
    }

    /// Record workload metrics
    pub fn record(&mut self, metrics: WorkloadMetrics) {
        self.history.push(metrics);

        // Keep only last 1000 samples
        if self.history.len() > 1000 {
            self.history.remove(0);
        }
    }

    /// Analyze current workload and generate recommendations
    pub fn analyze(&self, current: &WorkloadMetrics) -> Result<ScalingRecommendations> {
        let mut actions = Vec::new();

        // Check P99 latency
        let p99_ms = current.p99_latency.as_millis() as u64;
        if p99_ms > self.config.target_p99_latency_ms {
            let latency_ratio = p99_ms as f64 / self.config.target_p99_latency_ms as f64;

            if latency_ratio > 2.0 {
                // Severe latency issues - need horizontal scaling
                actions.push(ScalingAction {
                    action_type: ActionType::ScaleHorizontally,
                    priority: 0.9,
                    description: format!(
                        "Add replicas to handle load. Current P99: {}ms, Target: {}ms",
                        p99_ms, self.config.target_p99_latency_ms
                    ),
                    expected_impact: format!(
                        "Reduce P99 latency by ~{}%",
                        ((latency_ratio - 1.0) * 50.0).min(70.0) as i32
                    ),
                    cost_estimate: latency_ratio * 10.0,
                });
            } else {
                // Moderate latency - optimize parameters
                actions.push(ScalingAction {
                    action_type: ActionType::OptimizeParameters,
                    priority: 0.6,
                    description: format!(
                        "Optimize HNSW parameters (reduce ef_search). Current P99: {}ms",
                        p99_ms
                    ),
                    expected_impact: "Reduce P99 latency by 20-30% with minimal accuracy loss"
                        .to_string(),
                    cost_estimate: 0.5,
                });
            }
        }

        // Check CPU utilization
        if current.cpu_utilization > 0.85 {
            actions.push(ScalingAction {
                action_type: ActionType::ScaleVertically,
                priority: 0.8,
                description: format!(
                    "Increase CPU resources. Current: {:.1}%, Saturated at >85%",
                    current.cpu_utilization * 100.0
                ),
                expected_impact: "Increase query throughput by 30-50%".to_string(),
                cost_estimate: current.cpu_utilization * 8.0,
            });
        }

        // Check cache hit rate
        if current.cache_hit_rate < self.config.min_cache_hit_rate {
            actions.push(ScalingAction {
                action_type: ActionType::IncreaseCache,
                priority: 0.7,
                description: format!(
                    "Increase cache size. Current hit rate: {:.1}%, Target: {:.1}%",
                    current.cache_hit_rate * 100.0,
                    self.config.min_cache_hit_rate * 100.0
                ),
                expected_impact: format!(
                    "Improve hit rate by {:.0}%, reduce latency by 15-25%",
                    (self.config.min_cache_hit_rate - current.cache_hit_rate) * 100.0
                ),
                cost_estimate: 3.0,
            });
        }

        // Check memory pressure for large indices
        if current.index_size > 5_000_000 && current.memory_usage_mb > 8192.0 {
            actions.push(ScalingAction {
                action_type: ActionType::EnableCompression,
                priority: 0.65,
                description: format!(
                    "Enable quantization for {} vectors using {}MB memory",
                    current.index_size, current.memory_usage_mb
                ),
                expected_impact: "Reduce memory by 4-8x with <5% accuracy loss".to_string(),
                cost_estimate: 1.0,
            });
        }

        // Sort actions by priority
        actions.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Calculate health score
        let health_score = self.calculate_health_score(current);

        // Calculate capacity headroom
        let capacity_headroom = self.calculate_capacity_headroom(current);

        // Calculate cost-benefit ratio
        let cost_benefit_ratio = if actions.is_empty() {
            0.0
        } else {
            let total_benefit: f64 = actions.iter().map(|a| a.priority).sum();
            let total_cost: f64 = actions.iter().map(|a| a.cost_estimate).sum();
            if total_cost > 0.0 {
                total_benefit / total_cost
            } else {
                0.0
            }
        };

        Ok(ScalingRecommendations {
            health_score,
            capacity_headroom,
            actions,
            cost_benefit_ratio,
        })
    }

    /// Calculate system health score (0.0 to 1.0)
    fn calculate_health_score(&self, metrics: &WorkloadMetrics) -> f64 {
        let mut score = 1.0;

        // Penalty for high latency
        let p99_ms = metrics.p99_latency.as_millis() as u64;
        if p99_ms > self.config.target_p99_latency_ms {
            let latency_penalty =
                (p99_ms as f64 / self.config.target_p99_latency_ms as f64 - 1.0) * 0.3;
            score -= latency_penalty.min(0.4);
        }

        // Penalty for high CPU
        if metrics.cpu_utilization > self.config.target_cpu_utilization {
            let cpu_penalty = (metrics.cpu_utilization - self.config.target_cpu_utilization) * 0.5;
            score -= cpu_penalty.min(0.3);
        }

        // Penalty for low cache hit rate
        if metrics.cache_hit_rate < self.config.min_cache_hit_rate {
            let cache_penalty = (self.config.min_cache_hit_rate - metrics.cache_hit_rate) * 0.3;
            score -= cache_penalty.min(0.2);
        }

        score.max(0.0)
    }

    /// Calculate capacity headroom (how much more load can be handled)
    fn calculate_capacity_headroom(&self, metrics: &WorkloadMetrics) -> f64 {
        // Estimate based on CPU utilization and current QPS
        let _cpu_headroom = (1.0 - metrics.cpu_utilization).max(0.0);
        let estimated_max_qps = metrics.queries_per_second / metrics.cpu_utilization;
        let additional_capacity = estimated_max_qps - metrics.queries_per_second;

        (additional_capacity / metrics.queries_per_second).clamp(0.0, 2.0)
    }

    /// Get historical trend analysis
    pub fn trend_analysis(&self) -> TrendReport {
        if self.history.len() < 2 {
            return TrendReport::default();
        }

        let recent = &self.history[self.history.len().saturating_sub(10)..];

        let avg_qps: f64 =
            recent.iter().map(|m| m.queries_per_second).sum::<f64>() / recent.len() as f64;
        let avg_cpu: f64 =
            recent.iter().map(|m| m.cpu_utilization).sum::<f64>() / recent.len() as f64;
        let avg_cache_hit: f64 =
            recent.iter().map(|m| m.cache_hit_rate).sum::<f64>() / recent.len() as f64;

        // Calculate trends
        let qps_trend = if recent.len() > 1 {
            (recent
                .last()
                .expect("recent.len() > 1 checked above")
                .queries_per_second
                - recent[0].queries_per_second)
                / recent[0].queries_per_second
        } else {
            0.0
        };

        TrendReport {
            avg_qps,
            avg_cpu_utilization: avg_cpu,
            avg_cache_hit_rate: avg_cache_hit,
            qps_trend_percent: qps_trend * 100.0,
            sample_count: recent.len(),
        }
    }
}

impl Default for AutoScalingAdvisor {
    fn default() -> Self {
        Self::new()
    }
}

/// Trend analysis report
#[derive(Debug, Clone, Default)]
pub struct TrendReport {
    /// Average QPS over recent period
    pub avg_qps: f64,
    /// Average CPU utilization
    pub avg_cpu_utilization: f64,
    /// Average cache hit rate
    pub avg_cache_hit_rate: f64,
    /// QPS trend (percent change)
    pub qps_trend_percent: f64,
    /// Number of samples analyzed
    pub sample_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advisor_creation() {
        let advisor = AutoScalingAdvisor::new();
        assert_eq!(advisor.history.len(), 0);
    }

    #[test]
    fn test_healthy_system() {
        let advisor = AutoScalingAdvisor::new();

        let metrics = WorkloadMetrics {
            queries_per_second: 500.0,
            avg_latency: Duration::from_millis(5),
            p99_latency: Duration::from_millis(20),
            memory_usage_mb: 2048.0,
            cpu_utilization: 0.50,
            cache_hit_rate: 0.85,
            index_size: 1_000_000,
        };

        let recommendations = advisor
            .analyze(&metrics)
            .expect("test: analyze should succeed for healthy metrics");
        assert!(recommendations.health_score > 0.8);
        assert!(recommendations.actions.is_empty() || recommendations.actions[0].priority < 0.5);
    }

    #[test]
    fn test_high_latency_detection() {
        let advisor = AutoScalingAdvisor::new();

        let metrics = WorkloadMetrics {
            queries_per_second: 1500.0,
            avg_latency: Duration::from_millis(50),
            p99_latency: Duration::from_millis(250), // Very high!
            memory_usage_mb: 4096.0,
            cpu_utilization: 0.85,
            cache_hit_rate: 0.60,
            index_size: 10_000_000,
        };

        let recommendations = advisor
            .analyze(&metrics)
            .expect("test: analyze should succeed for high latency metrics");
        assert!(recommendations.health_score < 0.7);
        assert!(!recommendations.actions.is_empty());
        assert!(recommendations
            .actions
            .iter()
            .any(|a| a.action_type == ActionType::ScaleHorizontally));
    }

    #[test]
    fn test_low_cache_hit_rate() {
        let advisor = AutoScalingAdvisor::new();

        let metrics = WorkloadMetrics {
            queries_per_second: 1000.0,
            avg_latency: Duration::from_millis(10),
            p99_latency: Duration::from_millis(50),
            memory_usage_mb: 2048.0,
            cpu_utilization: 0.60,
            cache_hit_rate: 0.40, // Very low!
            index_size: 5_000_000,
        };

        let recommendations = advisor
            .analyze(&metrics)
            .expect("test: analyze should succeed for low cache hit rate metrics");
        assert!(recommendations
            .actions
            .iter()
            .any(|a| a.action_type == ActionType::IncreaseCache));
    }

    #[test]
    fn test_high_cpu_utilization() {
        let advisor = AutoScalingAdvisor::new();

        let metrics = WorkloadMetrics {
            queries_per_second: 2000.0,
            avg_latency: Duration::from_millis(15),
            p99_latency: Duration::from_millis(60),
            memory_usage_mb: 4096.0,
            cpu_utilization: 0.92, // Very high!
            cache_hit_rate: 0.80,
            index_size: 8_000_000,
        };

        let recommendations = advisor
            .analyze(&metrics)
            .expect("test: analyze should succeed for high CPU metrics");
        assert!(recommendations
            .actions
            .iter()
            .any(|a| a.action_type == ActionType::ScaleVertically));
    }

    #[test]
    fn test_compression_recommendation() {
        let advisor = AutoScalingAdvisor::new();

        let metrics = WorkloadMetrics {
            queries_per_second: 1000.0,
            avg_latency: Duration::from_millis(10),
            p99_latency: Duration::from_millis(50),
            memory_usage_mb: 10000.0, // High memory usage
            cpu_utilization: 0.60,
            cache_hit_rate: 0.80,
            index_size: 10_000_000, // Large index
        };

        let recommendations = advisor
            .analyze(&metrics)
            .expect("test: analyze should succeed for high memory metrics");
        assert!(recommendations
            .actions
            .iter()
            .any(|a| a.action_type == ActionType::EnableCompression));
    }

    #[test]
    fn test_record_metrics() {
        let mut advisor = AutoScalingAdvisor::new();

        let metrics = WorkloadMetrics {
            queries_per_second: 1000.0,
            avg_latency: Duration::from_millis(10),
            p99_latency: Duration::from_millis(50),
            memory_usage_mb: 2048.0,
            cpu_utilization: 0.60,
            cache_hit_rate: 0.80,
            index_size: 5_000_000,
        };

        advisor.record(metrics.clone());
        advisor.record(metrics);

        assert_eq!(advisor.history.len(), 2);
    }

    #[test]
    fn test_capacity_headroom() {
        let advisor = AutoScalingAdvisor::new();

        let metrics = WorkloadMetrics {
            queries_per_second: 1000.0,
            avg_latency: Duration::from_millis(10),
            p99_latency: Duration::from_millis(50),
            memory_usage_mb: 2048.0,
            cpu_utilization: 0.50, // 50% CPU means 100% headroom
            cache_hit_rate: 0.80,
            index_size: 5_000_000,
        };

        let recommendations = advisor
            .analyze(&metrics)
            .expect("test: analyze should succeed for capacity headroom check");
        assert!(recommendations.capacity_headroom > 0.5);
    }

    #[test]
    fn test_trend_analysis() {
        let mut advisor = AutoScalingAdvisor::new();

        for i in 0..10 {
            let metrics = WorkloadMetrics {
                queries_per_second: 1000.0 + (i as f64 * 100.0),
                avg_latency: Duration::from_millis(10),
                p99_latency: Duration::from_millis(50),
                memory_usage_mb: 2048.0,
                cpu_utilization: 0.60,
                cache_hit_rate: 0.80,
                index_size: 5_000_000,
            };
            advisor.record(metrics);
        }

        let trend = advisor.trend_analysis();
        assert_eq!(trend.sample_count, 10);
        assert!(trend.qps_trend_percent > 0.0); // Increasing trend
    }

    #[test]
    fn test_custom_config() {
        let config = AdvisorConfig {
            target_p99_latency_ms: 50,
            target_cpu_utilization: 0.80,
            min_cache_hit_rate: 0.90,
            target_qps_capacity: 5000.0,
        };

        let advisor = AutoScalingAdvisor::with_config(config);

        let metrics = WorkloadMetrics {
            queries_per_second: 1000.0,
            avg_latency: Duration::from_millis(10),
            p99_latency: Duration::from_millis(75), // Over custom target
            memory_usage_mb: 2048.0,
            cpu_utilization: 0.70,
            cache_hit_rate: 0.85, // Below custom target
            index_size: 5_000_000,
        };

        let recommendations = advisor
            .analyze(&metrics)
            .expect("test: analyze should succeed with custom config");
        assert!(!recommendations.actions.is_empty());
    }

    #[test]
    fn test_action_priority_ordering() {
        let advisor = AutoScalingAdvisor::new();

        let metrics = WorkloadMetrics {
            queries_per_second: 2000.0,
            avg_latency: Duration::from_millis(50),
            p99_latency: Duration::from_millis(300), // Critical
            memory_usage_mb: 10000.0,
            cpu_utilization: 0.95, // Critical
            cache_hit_rate: 0.40,  // Poor
            index_size: 10_000_000,
        };

        let recommendations = advisor
            .analyze(&metrics)
            .expect("test: analyze should succeed for priority ordering check");

        // Actions should be sorted by priority
        for i in 1..recommendations.actions.len() {
            assert!(recommendations.actions[i - 1].priority >= recommendations.actions[i].priority);
        }
    }
}
