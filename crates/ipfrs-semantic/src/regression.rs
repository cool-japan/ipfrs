//! Performance regression detection and tracking
//!
//! This module provides tools for detecting performance regressions in semantic search
//! systems by comparing current performance against historical baselines.
//!
//! # Features
//!
//! - **Baseline Management**: Track performance baselines over time
//! - **Regression Detection**: Automatically detect performance degradation
//! - **Trend Analysis**: Identify performance trends
//! - **Alerting**: Flag significant regressions for investigation
//!
//! # Example
//!
//! ```rust
//! use ipfrs_semantic::regression::{RegressionDetector, PerformanceMetrics};
//! use std::time::Duration;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut detector = RegressionDetector::new();
//!
//! // Record baseline metrics
//! let baseline = PerformanceMetrics {
//!     avg_query_latency: Duration::from_micros(500),
//!     p99_latency: Duration::from_millis(2),
//!     throughput_qps: 5000.0,
//!     memory_mb: 512.0,
//!     index_size: 100000,
//! };
//! detector.set_baseline(baseline)?;
//!
//! // Test current performance
//! let current = PerformanceMetrics {
//!     avg_query_latency: Duration::from_micros(750), // 50% slower!
//!     p99_latency: Duration::from_millis(3),
//!     throughput_qps: 4000.0,
//!     memory_mb: 520.0,
//!     index_size: 100000,
//! };
//!
//! let report = detector.check_regression(&current)?;
//! if report.has_regression {
//!     println!("⚠️  Regression detected!");
//!     for issue in &report.issues {
//!         println!("  - {}: {:.1}% change", issue.metric, issue.percent_change);
//!     }
//! }
//! # Ok(())
//! # }
//! ```

use ipfrs_core::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Performance metrics for a specific test run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average query latency
    pub avg_query_latency: Duration,
    /// P99 latency
    pub p99_latency: Duration,
    /// Throughput in queries per second
    pub throughput_qps: f64,
    /// Memory usage in MB
    pub memory_mb: f64,
    /// Index size (number of entries)
    pub index_size: usize,
}

/// Regression issue detected
#[derive(Debug, Clone)]
pub struct RegressionIssue {
    /// Metric name that regressed
    pub metric: String,
    /// Baseline value
    pub baseline: f64,
    /// Current value
    pub current: f64,
    /// Percent change (negative = improvement, positive = regression)
    pub percent_change: f64,
    /// Severity (0.0 to 1.0, where 1.0 is most severe)
    pub severity: f64,
}

/// Regression detection report
#[derive(Debug, Clone)]
pub struct RegressionReport {
    /// Whether any regressions were detected
    pub has_regression: bool,
    /// List of detected issues
    pub issues: Vec<RegressionIssue>,
    /// Overall regression score (0.0 to 1.0)
    pub regression_score: f64,
}

/// Configuration for regression detection
#[derive(Debug, Clone)]
pub struct RegressionConfig {
    /// Threshold for latency regression (e.g., 0.2 = 20% slower)
    pub latency_threshold: f64,
    /// Threshold for throughput regression (e.g., 0.15 = 15% slower)
    pub throughput_threshold: f64,
    /// Threshold for memory regression (e.g., 0.25 = 25% more memory)
    pub memory_threshold: f64,
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            latency_threshold: 0.15,    // 15% slower
            throughput_threshold: 0.10, // 10% slower
            memory_threshold: 0.20,     // 20% more memory
        }
    }
}

/// Performance regression detector
pub struct RegressionDetector {
    /// Configuration
    config: RegressionConfig,
    /// Baseline metrics
    baseline: Option<PerformanceMetrics>,
    /// Historical metrics
    history: Vec<(std::time::SystemTime, PerformanceMetrics)>,
}

impl RegressionDetector {
    /// Create a new regression detector with default config
    pub fn new() -> Self {
        Self {
            config: RegressionConfig::default(),
            baseline: None,
            history: Vec::new(),
        }
    }

    /// Create a regression detector with custom config
    pub fn with_config(config: RegressionConfig) -> Self {
        Self {
            config,
            baseline: None,
            history: Vec::new(),
        }
    }

    /// Set the baseline metrics
    pub fn set_baseline(&mut self, metrics: PerformanceMetrics) -> Result<()> {
        self.baseline = Some(metrics);
        Ok(())
    }

    /// Record metrics in history
    pub fn record_metrics(&mut self, metrics: PerformanceMetrics) {
        let now = std::time::SystemTime::now();
        self.history.push((now, metrics));

        // Keep only last 100 entries
        if self.history.len() > 100 {
            self.history.remove(0);
        }
    }

    /// Check for regressions against baseline
    pub fn check_regression(&self, current: &PerformanceMetrics) -> Result<RegressionReport> {
        let baseline = self
            .baseline
            .as_ref()
            .ok_or_else(|| ipfrs_core::Error::InvalidInput("No baseline set".into()))?;

        let mut issues = Vec::new();

        // Check latency regression
        let latency_change = self.calculate_change(
            baseline.avg_query_latency.as_micros() as f64,
            current.avg_query_latency.as_micros() as f64,
        );
        if latency_change > self.config.latency_threshold {
            issues.push(RegressionIssue {
                metric: "avg_query_latency".to_string(),
                baseline: baseline.avg_query_latency.as_micros() as f64,
                current: current.avg_query_latency.as_micros() as f64,
                percent_change: latency_change * 100.0,
                severity: (latency_change / self.config.latency_threshold).min(1.0),
            });
        }

        // Check P99 latency regression
        let p99_change = self.calculate_change(
            baseline.p99_latency.as_micros() as f64,
            current.p99_latency.as_micros() as f64,
        );
        if p99_change > self.config.latency_threshold {
            issues.push(RegressionIssue {
                metric: "p99_latency".to_string(),
                baseline: baseline.p99_latency.as_micros() as f64,
                current: current.p99_latency.as_micros() as f64,
                percent_change: p99_change * 100.0,
                severity: (p99_change / self.config.latency_threshold).min(1.0),
            });
        }

        // Check throughput regression (negative change means worse)
        let throughput_change =
            self.calculate_change(baseline.throughput_qps, current.throughput_qps);
        if throughput_change < -self.config.throughput_threshold {
            issues.push(RegressionIssue {
                metric: "throughput_qps".to_string(),
                baseline: baseline.throughput_qps,
                current: current.throughput_qps,
                percent_change: throughput_change * 100.0,
                severity: (-throughput_change / self.config.throughput_threshold).min(1.0),
            });
        }

        // Check memory regression
        let memory_change = self.calculate_change(baseline.memory_mb, current.memory_mb);
        if memory_change > self.config.memory_threshold {
            issues.push(RegressionIssue {
                metric: "memory_mb".to_string(),
                baseline: baseline.memory_mb,
                current: current.memory_mb,
                percent_change: memory_change * 100.0,
                severity: (memory_change / self.config.memory_threshold).min(1.0),
            });
        }

        // Calculate overall regression score
        let regression_score = if issues.is_empty() {
            0.0
        } else {
            issues.iter().map(|i| i.severity).sum::<f64>() / issues.len() as f64
        };

        Ok(RegressionReport {
            has_regression: !issues.is_empty(),
            issues,
            regression_score,
        })
    }

    /// Calculate percent change (positive = increase, negative = decrease)
    fn calculate_change(&self, baseline: f64, current: f64) -> f64 {
        if baseline == 0.0 {
            return 0.0;
        }
        (current - baseline) / baseline
    }

    /// Get historical trend for a metric
    pub fn get_trend(&self, metric_name: &str) -> Vec<(std::time::SystemTime, f64)> {
        self.history
            .iter()
            .map(|(time, metrics)| {
                let value = match metric_name {
                    "avg_query_latency" => metrics.avg_query_latency.as_micros() as f64,
                    "p99_latency" => metrics.p99_latency.as_micros() as f64,
                    "throughput_qps" => metrics.throughput_qps,
                    "memory_mb" => metrics.memory_mb,
                    _ => 0.0,
                };
                (*time, value)
            })
            .collect()
    }

    /// Generate a summary report of all historical metrics
    pub fn summary(&self) -> HashMap<String, MetricSummary> {
        let mut summaries = HashMap::new();

        if self.history.is_empty() {
            return summaries;
        }

        // Collect metrics for each type
        let mut latencies = Vec::new();
        let mut p99_latencies = Vec::new();
        let mut throughputs = Vec::new();
        let mut memories = Vec::new();

        for (_, metrics) in &self.history {
            latencies.push(metrics.avg_query_latency.as_micros() as f64);
            p99_latencies.push(metrics.p99_latency.as_micros() as f64);
            throughputs.push(metrics.throughput_qps);
            memories.push(metrics.memory_mb);
        }

        summaries.insert(
            "avg_query_latency".to_string(),
            Self::compute_summary(&latencies),
        );
        summaries.insert(
            "p99_latency".to_string(),
            Self::compute_summary(&p99_latencies),
        );
        summaries.insert(
            "throughput_qps".to_string(),
            Self::compute_summary(&throughputs),
        );
        summaries.insert("memory_mb".to_string(), Self::compute_summary(&memories));

        summaries
    }

    /// Compute statistical summary for a metric
    fn compute_summary(values: &[f64]) -> MetricSummary {
        if values.is_empty() {
            return MetricSummary::default();
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted[0];
        let max = sorted[sorted.len() - 1];
        let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
        let median = sorted[sorted.len() / 2];

        MetricSummary {
            min,
            max,
            mean,
            median,
            count: values.len(),
        }
    }
}

impl Default for RegressionDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistical summary for a metric
#[derive(Debug, Clone, Default)]
pub struct MetricSummary {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    pub count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regression_detector_creation() {
        let detector = RegressionDetector::new();
        assert!(detector.baseline.is_none());
        assert_eq!(detector.history.len(), 0);
    }

    #[test]
    fn test_set_baseline() {
        let mut detector = RegressionDetector::new();
        let metrics = PerformanceMetrics {
            avg_query_latency: Duration::from_micros(500),
            p99_latency: Duration::from_millis(2),
            throughput_qps: 5000.0,
            memory_mb: 512.0,
            index_size: 100000,
        };

        detector
            .set_baseline(metrics)
            .expect("test: set_baseline should succeed with valid metrics");
        assert!(detector.baseline.is_some());
    }

    #[test]
    fn test_no_regression() {
        let mut detector = RegressionDetector::new();
        let baseline = PerformanceMetrics {
            avg_query_latency: Duration::from_micros(500),
            p99_latency: Duration::from_millis(2),
            throughput_qps: 5000.0,
            memory_mb: 512.0,
            index_size: 100000,
        };
        detector
            .set_baseline(baseline.clone())
            .expect("test: set_baseline should succeed with valid metrics");

        let report = detector
            .check_regression(&baseline)
            .expect("test: check_regression should succeed when baseline is set");
        assert!(!report.has_regression);
        assert_eq!(report.issues.len(), 0);
    }

    #[test]
    fn test_latency_regression() {
        let mut detector = RegressionDetector::new();
        let baseline = PerformanceMetrics {
            avg_query_latency: Duration::from_micros(500),
            p99_latency: Duration::from_millis(2),
            throughput_qps: 5000.0,
            memory_mb: 512.0,
            index_size: 100000,
        };
        detector
            .set_baseline(baseline)
            .expect("test: set_baseline should succeed with valid metrics");

        // 50% slower latency
        let current = PerformanceMetrics {
            avg_query_latency: Duration::from_micros(750),
            p99_latency: Duration::from_millis(2),
            throughput_qps: 5000.0,
            memory_mb: 512.0,
            index_size: 100000,
        };

        let report = detector
            .check_regression(&current)
            .expect("test: check_regression should succeed when baseline is set");
        assert!(report.has_regression);
        assert!(report
            .issues
            .iter()
            .any(|i| i.metric == "avg_query_latency"));
    }

    #[test]
    fn test_throughput_regression() {
        let mut detector = RegressionDetector::new();
        let baseline = PerformanceMetrics {
            avg_query_latency: Duration::from_micros(500),
            p99_latency: Duration::from_millis(2),
            throughput_qps: 5000.0,
            memory_mb: 512.0,
            index_size: 100000,
        };
        detector
            .set_baseline(baseline)
            .expect("test: set_baseline should succeed with valid metrics");

        // 20% lower throughput
        let current = PerformanceMetrics {
            avg_query_latency: Duration::from_micros(500),
            p99_latency: Duration::from_millis(2),
            throughput_qps: 4000.0,
            memory_mb: 512.0,
            index_size: 100000,
        };

        let report = detector
            .check_regression(&current)
            .expect("test: check_regression should succeed when baseline is set");
        assert!(report.has_regression);
        assert!(report.issues.iter().any(|i| i.metric == "throughput_qps"));
    }

    #[test]
    fn test_memory_regression() {
        let mut detector = RegressionDetector::new();
        let baseline = PerformanceMetrics {
            avg_query_latency: Duration::from_micros(500),
            p99_latency: Duration::from_millis(2),
            throughput_qps: 5000.0,
            memory_mb: 512.0,
            index_size: 100000,
        };
        detector
            .set_baseline(baseline)
            .expect("test: set_baseline should succeed with valid metrics");

        // 30% more memory
        let current = PerformanceMetrics {
            avg_query_latency: Duration::from_micros(500),
            p99_latency: Duration::from_millis(2),
            throughput_qps: 5000.0,
            memory_mb: 665.6, // +30%
            index_size: 100000,
        };

        let report = detector
            .check_regression(&current)
            .expect("test: check_regression should succeed when baseline is set");
        assert!(report.has_regression);
        assert!(report.issues.iter().any(|i| i.metric == "memory_mb"));
    }

    #[test]
    fn test_record_metrics() {
        let mut detector = RegressionDetector::new();
        let metrics = PerformanceMetrics {
            avg_query_latency: Duration::from_micros(500),
            p99_latency: Duration::from_millis(2),
            throughput_qps: 5000.0,
            memory_mb: 512.0,
            index_size: 100000,
        };

        detector.record_metrics(metrics.clone());
        detector.record_metrics(metrics);
        assert_eq!(detector.history.len(), 2);
    }

    #[test]
    fn test_summary() {
        let mut detector = RegressionDetector::new();

        for i in 0..10 {
            let metrics = PerformanceMetrics {
                avg_query_latency: Duration::from_micros(500 + i * 10),
                p99_latency: Duration::from_millis(2),
                throughput_qps: 5000.0,
                memory_mb: 512.0,
                index_size: 100000,
            };
            detector.record_metrics(metrics);
        }

        let summary = detector.summary();
        assert!(summary.contains_key("avg_query_latency"));
        assert_eq!(summary["avg_query_latency"].count, 10);
    }

    #[test]
    fn test_custom_thresholds() {
        let config = RegressionConfig {
            latency_threshold: 0.50, // 50% threshold
            throughput_threshold: 0.30,
            memory_threshold: 0.40,
        };

        let mut detector = RegressionDetector::with_config(config);
        let baseline = PerformanceMetrics {
            avg_query_latency: Duration::from_micros(500),
            p99_latency: Duration::from_millis(2),
            throughput_qps: 5000.0,
            memory_mb: 512.0,
            index_size: 100000,
        };
        detector
            .set_baseline(baseline)
            .expect("test: set_baseline should succeed with valid metrics");

        // 30% slower - should not trigger with 50% threshold
        let current = PerformanceMetrics {
            avg_query_latency: Duration::from_micros(650),
            p99_latency: Duration::from_millis(2),
            throughput_qps: 5000.0,
            memory_mb: 512.0,
            index_size: 100000,
        };

        let report = detector
            .check_regression(&current)
            .expect("test: check_regression should succeed when baseline is set");
        assert!(!report.has_regression);
    }
}
