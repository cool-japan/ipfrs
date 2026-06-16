//! Metrics time-series aggregator for historical tracking and analysis
//!
//! This module provides time-series aggregation of network metrics with:
//! - Configurable time windows (second, minute, hour, day)
//! - Statistical analysis (min, max, avg, percentiles)
//! - Historical data retention
//! - Trend analysis and forecasting
//! - Multiple aggregation strategies
//!
//! # Examples
//!
//! ```
//! use ipfrs_network::metrics_aggregator::{MetricsAggregator, AggregatorConfig, TimeWindow};
//! use std::time::Duration;
//!
//! let config = AggregatorConfig::default();
//! let mut aggregator = MetricsAggregator::new(config);
//!
//! // Record metrics
//! aggregator.record_bandwidth(1024);
//! aggregator.record_latency(50);
//! aggregator.record_connection_event();
//!
//! // Get statistics
//! let stats = aggregator.get_statistics(TimeWindow::Minute);
//! println!("Avg bandwidth: {:.2} B/s", stats.bandwidth.avg);
//! println!("P95 latency: {} ms", stats.latency.p95);
//! ```

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Time window for aggregation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeWindow {
    /// 1 second window
    Second,
    /// 1 minute window
    Minute,
    /// 1 hour window
    Hour,
    /// 1 day window
    Day,
}

impl TimeWindow {
    /// Get the duration for this time window
    pub fn duration(&self) -> Duration {
        match self {
            TimeWindow::Second => Duration::from_secs(1),
            TimeWindow::Minute => Duration::from_secs(60),
            TimeWindow::Hour => Duration::from_secs(3600),
            TimeWindow::Day => Duration::from_secs(86400),
        }
    }
}

/// Configuration for metrics aggregator
#[derive(Debug, Clone)]
pub struct AggregatorConfig {
    /// Maximum number of data points to retain per metric
    pub max_data_points: usize,

    /// Retention period for historical data
    pub retention_period: Duration,

    /// Enable percentile calculations (more CPU intensive)
    pub enable_percentiles: bool,

    /// Enable trend analysis
    pub enable_trends: bool,

    /// Sample rate for high-frequency metrics (1 = all, 10 = 1 in 10)
    pub sample_rate: usize,
}

impl Default for AggregatorConfig {
    fn default() -> Self {
        Self {
            max_data_points: 10000,
            retention_period: Duration::from_secs(3600), // 1 hour
            enable_percentiles: true,
            enable_trends: true,
            sample_rate: 1,
        }
    }
}

impl AggregatorConfig {
    /// Configuration for real-time monitoring (short retention, high detail)
    pub fn realtime() -> Self {
        Self {
            max_data_points: 1000,
            retention_period: Duration::from_secs(300), // 5 minutes
            enable_percentiles: true,
            enable_trends: false,
            sample_rate: 1,
        }
    }

    /// Configuration for long-term storage (extended retention, lower detail)
    pub fn longterm() -> Self {
        Self {
            max_data_points: 50000,
            retention_period: Duration::from_secs(86400 * 7), // 7 days
            enable_percentiles: false,
            enable_trends: true,
            sample_rate: 10, // Sample 1 in 10
        }
    }

    /// Configuration for high-frequency metrics (balanced)
    pub fn balanced() -> Self {
        Self {
            max_data_points: 5000,
            retention_period: Duration::from_secs(3600), // 1 hour
            enable_percentiles: true,
            enable_trends: true,
            sample_rate: 5,
        }
    }
}

/// A single data point with timestamp
#[derive(Debug, Clone, Copy)]
struct DataPoint {
    value: f64,
    timestamp: Instant,
}

/// Time series data for a metric
#[derive(Debug)]
struct TimeSeries {
    data: VecDeque<DataPoint>,
    sample_counter: usize,
}

impl TimeSeries {
    fn new(capacity: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(capacity),
            sample_counter: 0,
        }
    }

    fn add(&mut self, value: f64, max_points: usize, sample_rate: usize) {
        self.sample_counter += 1;
        if !self.sample_counter.is_multiple_of(sample_rate) {
            return;
        }

        let point = DataPoint {
            value,
            timestamp: Instant::now(),
        };

        self.data.push_back(point);

        // Remove oldest points if we exceed max
        while self.data.len() > max_points {
            self.data.pop_front();
        }
    }

    fn cleanup_old(&mut self, retention: Duration) {
        let now = Instant::now();
        while let Some(point) = self.data.front() {
            if now.duration_since(point.timestamp) > retention {
                self.data.pop_front();
            } else {
                break;
            }
        }
    }

    fn get_values_in_window(&self, window: Duration) -> Vec<f64> {
        let now = Instant::now();
        self.data
            .iter()
            .filter(|p| now.duration_since(p.timestamp) <= window)
            .map(|p| p.value)
            .collect()
    }
}

/// Statistics for a metric
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetricStatistics {
    /// Number of samples
    pub count: usize,

    /// Minimum value
    pub min: f64,

    /// Maximum value
    pub max: f64,

    /// Average value
    pub avg: f64,

    /// Standard deviation
    pub stddev: f64,

    /// 50th percentile (median)
    pub p50: f64,

    /// 95th percentile
    pub p95: f64,

    /// 99th percentile
    pub p99: f64,

    /// Current trend (positive = increasing, negative = decreasing)
    pub trend: f64,
}

/// Aggregated statistics for all metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregatedStatistics {
    /// Bandwidth statistics (bytes/sec)
    pub bandwidth: MetricStatistics,

    /// Latency statistics (milliseconds)
    pub latency: MetricStatistics,

    /// Connection event rate (events/sec)
    pub connection_rate: MetricStatistics,

    /// Query rate (queries/sec)
    pub query_rate: MetricStatistics,

    /// Error rate (errors/sec)
    pub error_rate: MetricStatistics,
}

/// Metrics aggregator for time-series data
pub struct MetricsAggregator {
    config: AggregatorConfig,
    bandwidth: RwLock<TimeSeries>,
    latency: RwLock<TimeSeries>,
    connections: RwLock<TimeSeries>,
    queries: RwLock<TimeSeries>,
    errors: RwLock<TimeSeries>,
}

impl MetricsAggregator {
    /// Create a new metrics aggregator
    pub fn new(config: AggregatorConfig) -> Self {
        let capacity = config.max_data_points;
        Self {
            config,
            bandwidth: RwLock::new(TimeSeries::new(capacity)),
            latency: RwLock::new(TimeSeries::new(capacity)),
            connections: RwLock::new(TimeSeries::new(capacity)),
            queries: RwLock::new(TimeSeries::new(capacity)),
            errors: RwLock::new(TimeSeries::new(capacity)),
        }
    }

    /// Record bandwidth measurement (bytes)
    pub fn record_bandwidth(&self, bytes: u64) {
        let mut series = self.bandwidth.write();
        series.add(
            bytes as f64,
            self.config.max_data_points,
            self.config.sample_rate,
        );
    }

    /// Record latency measurement (milliseconds)
    pub fn record_latency(&self, ms: u64) {
        let mut series = self.latency.write();
        series.add(
            ms as f64,
            self.config.max_data_points,
            self.config.sample_rate,
        );
    }

    /// Record connection event
    pub fn record_connection_event(&self) {
        let mut series = self.connections.write();
        series.add(1.0, self.config.max_data_points, self.config.sample_rate);
    }

    /// Record query event
    pub fn record_query_event(&self) {
        let mut series = self.queries.write();
        series.add(1.0, self.config.max_data_points, self.config.sample_rate);
    }

    /// Record error event
    pub fn record_error_event(&self) {
        let mut series = self.errors.write();
        series.add(1.0, self.config.max_data_points, self.config.sample_rate);
    }

    /// Get statistics for a time window
    pub fn get_statistics(&self, window: TimeWindow) -> AggregatedStatistics {
        let duration = window.duration();

        AggregatedStatistics {
            bandwidth: self.compute_statistics(&self.bandwidth, duration),
            latency: self.compute_statistics(&self.latency, duration),
            connection_rate: self.compute_statistics(&self.connections, duration),
            query_rate: self.compute_statistics(&self.queries, duration),
            error_rate: self.compute_statistics(&self.errors, duration),
        }
    }

    /// Compute statistics for a time series
    fn compute_statistics(
        &self,
        series: &RwLock<TimeSeries>,
        window: Duration,
    ) -> MetricStatistics {
        let data = series.read();
        let values = data.get_values_in_window(window);

        if values.is_empty() {
            return MetricStatistics::default();
        }

        let count = values.len();
        let sum: f64 = values.iter().sum();
        let avg = sum / count as f64;

        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        // Calculate standard deviation
        let variance: f64 = values.iter().map(|v| (v - avg).powi(2)).sum::<f64>() / count as f64;
        let stddev = variance.sqrt();

        // Calculate percentiles if enabled
        let (p50, p95, p99) = if self.config.enable_percentiles {
            let mut sorted = values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            (
                percentile(&sorted, 0.50),
                percentile(&sorted, 0.95),
                percentile(&sorted, 0.99),
            )
        } else {
            (avg, max, max)
        };

        // Calculate trend if enabled
        let trend = if self.config.enable_trends {
            calculate_trend(&values)
        } else {
            0.0
        };

        MetricStatistics {
            count,
            min,
            max,
            avg,
            stddev,
            p50,
            p95,
            p99,
            trend,
        }
    }

    /// Cleanup old data points
    pub fn cleanup(&self) {
        let retention = self.config.retention_period;
        self.bandwidth.write().cleanup_old(retention);
        self.latency.write().cleanup_old(retention);
        self.connections.write().cleanup_old(retention);
        self.queries.write().cleanup_old(retention);
        self.errors.write().cleanup_old(retention);
    }

    /// Get the number of data points currently stored
    pub fn data_point_count(&self) -> usize {
        self.bandwidth.read().data.len()
            + self.latency.read().data.len()
            + self.connections.read().data.len()
            + self.queries.read().data.len()
            + self.errors.read().data.len()
    }

    /// Clear all data
    pub fn clear(&self) {
        self.bandwidth.write().data.clear();
        self.latency.write().data.clear();
        self.connections.write().data.clear();
        self.queries.write().data.clear();
        self.errors.write().data.clear();
    }
}

/// Calculate percentile from sorted values
fn percentile(sorted_values: &[f64], p: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }

    let index = (p * (sorted_values.len() - 1) as f64) as usize;
    sorted_values[index]
}

/// Calculate trend using simple linear regression
fn calculate_trend(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let n = values.len() as f64;
    let x_mean = (n - 1.0) / 2.0;
    let y_mean = values.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (i, &y) in values.iter().enumerate() {
        let x = i as f64;
        numerator += (x - x_mean) * (y - y_mean);
        denominator += (x - x_mean).powi(2);
    }

    if denominator.abs() < 1e-10 {
        return 0.0;
    }

    numerator / denominator
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_presets() {
        let realtime = AggregatorConfig::realtime();
        assert_eq!(realtime.max_data_points, 1000);
        assert!(!realtime.enable_trends);

        let longterm = AggregatorConfig::longterm();
        assert_eq!(longterm.max_data_points, 50000);
        assert!(longterm.enable_trends);

        let balanced = AggregatorConfig::balanced();
        assert_eq!(balanced.sample_rate, 5);
    }

    #[test]
    fn test_time_window_duration() {
        assert_eq!(TimeWindow::Second.duration(), Duration::from_secs(1));
        assert_eq!(TimeWindow::Minute.duration(), Duration::from_secs(60));
        assert_eq!(TimeWindow::Hour.duration(), Duration::from_secs(3600));
        assert_eq!(TimeWindow::Day.duration(), Duration::from_secs(86400));
    }

    #[test]
    fn test_record_bandwidth() {
        let config = AggregatorConfig::default();
        let aggregator = MetricsAggregator::new(config);

        aggregator.record_bandwidth(1024);
        aggregator.record_bandwidth(2048);

        let stats = aggregator.get_statistics(TimeWindow::Minute);
        assert_eq!(stats.bandwidth.count, 2);
        assert_eq!(stats.bandwidth.min, 1024.0);
        assert_eq!(stats.bandwidth.max, 2048.0);
    }

    #[test]
    fn test_record_latency() {
        let config = AggregatorConfig::default();
        let aggregator = MetricsAggregator::new(config);

        aggregator.record_latency(50);
        aggregator.record_latency(100);
        aggregator.record_latency(75);

        let stats = aggregator.get_statistics(TimeWindow::Minute);
        assert_eq!(stats.latency.count, 3);
        assert_eq!(stats.latency.min, 50.0);
        assert_eq!(stats.latency.max, 100.0);
        assert_eq!(stats.latency.avg, 75.0);
    }

    #[test]
    fn test_connection_events() {
        let config = AggregatorConfig::default();
        let aggregator = MetricsAggregator::new(config);

        for _ in 0..5 {
            aggregator.record_connection_event();
        }

        let stats = aggregator.get_statistics(TimeWindow::Minute);
        assert_eq!(stats.connection_rate.count, 5);
    }

    #[test]
    fn test_query_events() {
        let config = AggregatorConfig::default();
        let aggregator = MetricsAggregator::new(config);

        for _ in 0..10 {
            aggregator.record_query_event();
        }

        let stats = aggregator.get_statistics(TimeWindow::Minute);
        assert_eq!(stats.query_rate.count, 10);
    }

    #[test]
    fn test_error_events() {
        let config = AggregatorConfig::default();
        let aggregator = MetricsAggregator::new(config);

        for _ in 0..3 {
            aggregator.record_error_event();
        }

        let stats = aggregator.get_statistics(TimeWindow::Minute);
        assert_eq!(stats.error_rate.count, 3);
    }

    #[test]
    fn test_percentile_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        assert_eq!(percentile(&values, 0.50), 5.0);
        assert_eq!(percentile(&values, 0.95), 9.0); // 95% of index 9 = 8.55 -> index 8 = 9.0
    }

    #[test]
    fn test_trend_calculation() {
        // Increasing trend
        let increasing = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let trend = calculate_trend(&increasing);
        assert!(trend > 0.0);

        // Decreasing trend
        let decreasing = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let trend = calculate_trend(&decreasing);
        assert!(trend < 0.0);

        // Flat trend
        let flat = vec![3.0, 3.0, 3.0, 3.0, 3.0];
        let trend = calculate_trend(&flat);
        assert!(trend.abs() < 0.01);
    }

    #[test]
    fn test_sample_rate() {
        let config = AggregatorConfig {
            sample_rate: 2, // Sample 1 in 2
            ..Default::default()
        };

        let aggregator = MetricsAggregator::new(config);

        for _ in 0..10 {
            aggregator.record_bandwidth(1024);
        }

        let stats = aggregator.get_statistics(TimeWindow::Minute);
        assert_eq!(stats.bandwidth.count, 5); // Half of 10
    }

    #[test]
    fn test_data_point_count() {
        let config = AggregatorConfig::default();
        let aggregator = MetricsAggregator::new(config);

        aggregator.record_bandwidth(1024);
        aggregator.record_latency(50);
        aggregator.record_connection_event();

        assert_eq!(aggregator.data_point_count(), 3);
    }

    #[test]
    fn test_clear() {
        let config = AggregatorConfig::default();
        let aggregator = MetricsAggregator::new(config);

        aggregator.record_bandwidth(1024);
        aggregator.record_latency(50);

        assert!(aggregator.data_point_count() > 0);

        aggregator.clear();
        assert_eq!(aggregator.data_point_count(), 0);
    }

    #[test]
    fn test_max_data_points() {
        let config = AggregatorConfig {
            max_data_points: 5,
            ..Default::default()
        };

        let aggregator = MetricsAggregator::new(config);

        for i in 0..10 {
            aggregator.record_bandwidth(i * 100);
        }

        // Should only keep the last 5 points
        let count = aggregator.bandwidth.read().data.len();
        assert_eq!(count, 5);
    }

    #[test]
    fn test_statistics_with_no_data() {
        let config = AggregatorConfig::default();
        let aggregator = MetricsAggregator::new(config);

        let stats = aggregator.get_statistics(TimeWindow::Minute);
        assert_eq!(stats.bandwidth.count, 0);
        assert_eq!(stats.bandwidth.avg, 0.0);
    }
}
