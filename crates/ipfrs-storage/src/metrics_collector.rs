//! Storage metrics collector with sliding window aggregation and alerting.
//!
//! Collects time-series storage metrics (throughput, latency, error rates)
//! with sliding window aggregation and configurable alert thresholds.

use std::collections::HashMap;

/// The kind of metric being recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MetricKind {
    /// Read throughput in bytes per second.
    ReadThroughput,
    /// Write throughput in bytes per second.
    WriteThroughput,
    /// Read latency in milliseconds.
    ReadLatencyMs,
    /// Write latency in milliseconds.
    WriteLatencyMs,
    /// Error rate as a fraction [0.0, 1.0].
    ErrorRate,
    /// Cache hit rate as a fraction [0.0, 1.0].
    CacheHitRate,
}

/// A single recorded metric sample.
#[derive(Clone, Debug)]
pub struct MetricSample {
    /// The kind of metric this sample represents.
    pub kind: MetricKind,
    /// The numeric value of the sample.
    pub value: f64,
    /// The logical tick at which this sample was recorded.
    pub tick: u64,
}

/// Aggregated statistics over the current sliding window for a metric kind.
#[derive(Clone, Debug)]
pub struct WindowStats {
    /// The kind of metric.
    pub kind: MetricKind,
    /// Minimum value in the window.
    pub min: f64,
    /// Maximum value in the window.
    pub max: f64,
    /// Arithmetic mean of values in the window.
    pub mean: f64,
    /// Number of samples in the window.
    pub count: usize,
}

impl WindowStats {
    /// Returns `max - min` (the range of values in the window).
    pub fn range(&self) -> f64 {
        self.max - self.min
    }
}

/// A threshold that triggers an alert when the window mean crosses a boundary.
#[derive(Clone, Debug)]
pub struct AlertThreshold {
    /// The metric kind this threshold applies to.
    pub kind: MetricKind,
    /// Alert if the window mean exceeds this value.
    pub max_value: Option<f64>,
    /// Alert if the window mean falls below this value.
    pub min_value: Option<f64>,
}

/// An alert emitted when a metric mean crosses a configured threshold.
#[derive(Clone, Debug)]
pub struct MetricAlert {
    /// The metric kind that triggered the alert.
    pub kind: MetricKind,
    /// Human-readable description of the alert.
    pub message: String,
    /// The current window mean that triggered the alert.
    pub current_value: f64,
    /// The threshold value that was crossed.
    pub threshold: f64,
}

/// Collects time-series storage metrics with sliding window aggregation.
///
/// Each `MetricKind` maintains an independent ring of up to `window_size`
/// samples. When the window is full the oldest sample is evicted before the
/// new one is inserted, keeping memory bounded. Configured
/// [`AlertThreshold`]s are evaluated lazily via [`check_alerts`].
///
/// # Example
///
/// ```
/// use ipfrs_storage::metrics_collector::{
///     MetricKind, StorageMetricsCollector, AlertThreshold,
/// };
///
/// let mut col = StorageMetricsCollector::new(50);
/// col.record(MetricKind::ReadLatencyMs, 12.5);
/// col.record(MetricKind::ReadLatencyMs, 15.0);
/// let stats = col.window_stats(MetricKind::ReadLatencyMs).unwrap();
/// assert_eq!(stats.count, 2);
/// ```
///
/// [`check_alerts`]: StorageMetricsCollector::check_alerts
pub struct StorageMetricsCollector {
    /// Per-kind sample windows (bounded to `window_size`).
    pub samples: HashMap<MetricKind, Vec<MetricSample>>,
    /// Maximum number of samples retained per metric kind.
    pub window_size: usize,
    /// Alert thresholds evaluated by `check_alerts`.
    pub thresholds: Vec<AlertThreshold>,
    /// Monotonically increasing logical clock, advanced by [`advance_tick`].
    ///
    /// [`advance_tick`]: StorageMetricsCollector::advance_tick
    pub current_tick: u64,
}

impl StorageMetricsCollector {
    /// Creates a new collector with the given sliding-window size.
    ///
    /// `window_size` controls how many samples are kept per `MetricKind`.
    /// A value of `0` is accepted but effectively means no samples are ever
    /// retained (every sample is immediately evicted).
    pub fn new(window_size: usize) -> Self {
        Self {
            samples: HashMap::new(),
            window_size,
            thresholds: Vec::new(),
            current_tick: 0,
        }
    }

    /// Records a sample for `kind` with the given `value` at `current_tick`.
    ///
    /// If the window for this kind already contains `window_size` samples the
    /// oldest (index 0) is removed before the new sample is appended.
    pub fn record(&mut self, kind: MetricKind, value: f64) {
        let tick = self.current_tick;
        let window = self.samples.entry(kind).or_default();
        if window.len() >= self.window_size {
            window.remove(0);
        }
        window.push(MetricSample { kind, value, tick });
    }

    /// Advances the logical clock by one tick.
    pub fn advance_tick(&mut self) {
        self.current_tick += 1;
    }

    /// Returns aggregated statistics for all samples of `kind` currently in
    /// the window, or `None` if no samples have been recorded for that kind.
    pub fn window_stats(&self, kind: MetricKind) -> Option<WindowStats> {
        let window = self.samples.get(&kind)?;
        if window.is_empty() {
            return None;
        }

        let count = window.len();
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut sum = 0.0_f64;

        for sample in window {
            if sample.value < min {
                min = sample.value;
            }
            if sample.value > max {
                max = sample.value;
            }
            sum += sample.value;
        }

        Some(WindowStats {
            kind,
            min,
            max,
            mean: sum / count as f64,
            count,
        })
    }

    /// Registers an alert threshold.
    pub fn add_threshold(&mut self, threshold: AlertThreshold) {
        self.thresholds.push(threshold);
    }

    /// Evaluates all registered thresholds against current window statistics
    /// and returns any triggered alerts.
    ///
    /// An alert is emitted when:
    /// - `threshold.max_value` is set and `mean > max_value`, or
    /// - `threshold.min_value` is set and `mean < min_value`.
    pub fn check_alerts(&self) -> Vec<MetricAlert> {
        let mut alerts = Vec::new();

        for threshold in &self.thresholds {
            let stats = match self.window_stats(threshold.kind) {
                Some(s) => s,
                None => continue,
            };

            if let Some(max) = threshold.max_value {
                if stats.mean > max {
                    alerts.push(MetricAlert {
                        kind: threshold.kind,
                        message: format!(
                            "{:?} mean {:.4} exceeds maximum threshold {:.4}",
                            threshold.kind, stats.mean, max
                        ),
                        current_value: stats.mean,
                        threshold: max,
                    });
                }
            }

            if let Some(min) = threshold.min_value {
                if stats.mean < min {
                    alerts.push(MetricAlert {
                        kind: threshold.kind,
                        message: format!(
                            "{:?} mean {:.4} is below minimum threshold {:.4}",
                            threshold.kind, stats.mean, min
                        ),
                        current_value: stats.mean,
                        threshold: min,
                    });
                }
            }
        }

        alerts
    }

    /// Returns window statistics for every metric kind that has at least one
    /// recorded sample.
    pub fn all_stats(&self) -> Vec<WindowStats> {
        let mut result = Vec::new();
        for &kind in self.samples.keys() {
            if let Some(stats) = self.window_stats(kind) {
                result.push(stats);
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // 1. Basic record: single sample is stored correctly
    // -----------------------------------------------------------------------
    #[test]
    fn test_record_single_sample() {
        let mut col = StorageMetricsCollector::new(100);
        col.record(MetricKind::ReadThroughput, 42.0);
        let window = col.samples.get(&MetricKind::ReadThroughput).unwrap();
        assert_eq!(window.len(), 1);
        assert!((window[0].value - 42.0).abs() < f64::EPSILON);
        assert_eq!(window[0].tick, 0);
    }

    // -----------------------------------------------------------------------
    // 2. Multiple records accumulate until window_size is reached
    // -----------------------------------------------------------------------
    #[test]
    fn test_record_multiple_samples_within_window() {
        let mut col = StorageMetricsCollector::new(5);
        for i in 0..5u64 {
            col.record(MetricKind::WriteThroughput, i as f64);
        }
        let window = col.samples.get(&MetricKind::WriteThroughput).unwrap();
        assert_eq!(window.len(), 5);
    }

    // -----------------------------------------------------------------------
    // 3. window_size eviction: oldest sample is removed
    // -----------------------------------------------------------------------
    #[test]
    fn test_window_size_eviction() {
        let mut col = StorageMetricsCollector::new(3);
        col.record(MetricKind::ReadLatencyMs, 1.0); // oldest
        col.record(MetricKind::ReadLatencyMs, 2.0);
        col.record(MetricKind::ReadLatencyMs, 3.0);
        col.record(MetricKind::ReadLatencyMs, 4.0); // should evict 1.0

        let window = col.samples.get(&MetricKind::ReadLatencyMs).unwrap();
        assert_eq!(window.len(), 3);
        assert!((window[0].value - 2.0).abs() < f64::EPSILON);
        assert!((window[2].value - 4.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // 4. window_stats – min
    // -----------------------------------------------------------------------
    #[test]
    fn test_window_stats_min() {
        let mut col = StorageMetricsCollector::new(100);
        for v in [10.0_f64, 3.0, 7.0, 1.0, 5.0] {
            col.record(MetricKind::WriteLatencyMs, v);
        }
        let stats = col.window_stats(MetricKind::WriteLatencyMs).unwrap();
        assert!((stats.min - 1.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // 5. window_stats – max
    // -----------------------------------------------------------------------
    #[test]
    fn test_window_stats_max() {
        let mut col = StorageMetricsCollector::new(100);
        for v in [10.0_f64, 3.0, 7.0, 1.0, 5.0] {
            col.record(MetricKind::WriteLatencyMs, v);
        }
        let stats = col.window_stats(MetricKind::WriteLatencyMs).unwrap();
        assert!((stats.max - 10.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // 6. window_stats – mean
    // -----------------------------------------------------------------------
    #[test]
    fn test_window_stats_mean() {
        let mut col = StorageMetricsCollector::new(100);
        for v in [2.0_f64, 4.0, 6.0] {
            col.record(MetricKind::ErrorRate, v);
        }
        let stats = col.window_stats(MetricKind::ErrorRate).unwrap();
        assert!((stats.mean - 4.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // 7. window_stats – count
    // -----------------------------------------------------------------------
    #[test]
    fn test_window_stats_count() {
        let mut col = StorageMetricsCollector::new(100);
        for _ in 0..7 {
            col.record(MetricKind::CacheHitRate, 0.9);
        }
        let stats = col.window_stats(MetricKind::CacheHitRate).unwrap();
        assert_eq!(stats.count, 7);
    }

    // -----------------------------------------------------------------------
    // 8. WindowStats::range returns max – min
    // -----------------------------------------------------------------------
    #[test]
    fn test_window_stats_range() {
        let mut col = StorageMetricsCollector::new(100);
        for v in [5.0_f64, 15.0, 10.0] {
            col.record(MetricKind::ReadThroughput, v);
        }
        let stats = col.window_stats(MetricKind::ReadThroughput).unwrap();
        assert!((stats.range() - 10.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // 9. add_threshold stores the threshold
    // -----------------------------------------------------------------------
    #[test]
    fn test_add_threshold_stored() {
        let mut col = StorageMetricsCollector::new(100);
        col.add_threshold(AlertThreshold {
            kind: MetricKind::ErrorRate,
            max_value: Some(0.05),
            min_value: None,
        });
        assert_eq!(col.thresholds.len(), 1);
        assert_eq!(col.thresholds[0].kind, MetricKind::ErrorRate);
    }

    // -----------------------------------------------------------------------
    // 10. check_alerts – max_value exceeded triggers alert
    // -----------------------------------------------------------------------
    #[test]
    fn test_check_alerts_max_exceeded() {
        let mut col = StorageMetricsCollector::new(100);
        col.record(MetricKind::ErrorRate, 0.10);
        col.record(MetricKind::ErrorRate, 0.12);
        col.add_threshold(AlertThreshold {
            kind: MetricKind::ErrorRate,
            max_value: Some(0.05),
            min_value: None,
        });
        let alerts = col.check_alerts();
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].kind, MetricKind::ErrorRate);
        assert!((alerts[0].threshold - 0.05).abs() < f64::EPSILON);
        assert!(alerts[0].current_value > 0.05);
    }

    // -----------------------------------------------------------------------
    // 11. check_alerts – min_value not met triggers alert
    // -----------------------------------------------------------------------
    #[test]
    fn test_check_alerts_min_not_met() {
        let mut col = StorageMetricsCollector::new(100);
        col.record(MetricKind::CacheHitRate, 0.50);
        col.record(MetricKind::CacheHitRate, 0.60);
        col.add_threshold(AlertThreshold {
            kind: MetricKind::CacheHitRate,
            max_value: None,
            min_value: Some(0.80),
        });
        let alerts = col.check_alerts();
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].kind, MetricKind::CacheHitRate);
        assert!((alerts[0].threshold - 0.80).abs() < f64::EPSILON);
        assert!(alerts[0].current_value < 0.80);
    }

    // -----------------------------------------------------------------------
    // 12. check_alerts – no alert when within threshold
    // -----------------------------------------------------------------------
    #[test]
    fn test_check_alerts_no_alert_within_threshold() {
        let mut col = StorageMetricsCollector::new(100);
        col.record(MetricKind::ErrorRate, 0.01);
        col.record(MetricKind::ErrorRate, 0.02);
        col.add_threshold(AlertThreshold {
            kind: MetricKind::ErrorRate,
            max_value: Some(0.05),
            min_value: Some(0.001),
        });
        let alerts = col.check_alerts();
        assert!(
            alerts.is_empty(),
            "expected no alerts, got {:?}",
            alerts.len()
        );
    }

    // -----------------------------------------------------------------------
    // 13. advance_tick increments current_tick
    // -----------------------------------------------------------------------
    #[test]
    fn test_advance_tick() {
        let mut col = StorageMetricsCollector::new(100);
        assert_eq!(col.current_tick, 0);
        col.advance_tick();
        assert_eq!(col.current_tick, 1);
        col.advance_tick();
        assert_eq!(col.current_tick, 2);
    }

    // -----------------------------------------------------------------------
    // 14. Tick is captured correctly in the sample
    // -----------------------------------------------------------------------
    #[test]
    fn test_tick_stored_in_sample() {
        let mut col = StorageMetricsCollector::new(100);
        col.record(MetricKind::ReadLatencyMs, 5.0);
        col.advance_tick();
        col.record(MetricKind::ReadLatencyMs, 10.0);

        let window = col.samples.get(&MetricKind::ReadLatencyMs).unwrap();
        assert_eq!(window[0].tick, 0);
        assert_eq!(window[1].tick, 1);
    }

    // -----------------------------------------------------------------------
    // 15. all_stats – returns stats for all kinds with data
    // -----------------------------------------------------------------------
    #[test]
    fn test_all_stats_count() {
        let mut col = StorageMetricsCollector::new(100);
        col.record(MetricKind::ReadThroughput, 1.0);
        col.record(MetricKind::WriteThroughput, 2.0);
        col.record(MetricKind::ReadLatencyMs, 3.0);
        let all = col.all_stats();
        assert_eq!(all.len(), 3);
    }

    // -----------------------------------------------------------------------
    // 16. empty kind returns None from window_stats
    // -----------------------------------------------------------------------
    #[test]
    fn test_empty_kind_returns_none() {
        let col = StorageMetricsCollector::new(100);
        assert!(col.window_stats(MetricKind::WriteLatencyMs).is_none());
    }

    // -----------------------------------------------------------------------
    // 17. Multiple kinds are independent
    // -----------------------------------------------------------------------
    #[test]
    fn test_multiple_kinds_independent() {
        let mut col = StorageMetricsCollector::new(3);
        // Fill read-throughput window completely
        col.record(MetricKind::ReadThroughput, 10.0);
        col.record(MetricKind::ReadThroughput, 20.0);
        col.record(MetricKind::ReadThroughput, 30.0);
        col.record(MetricKind::ReadThroughput, 40.0); // evicts 10.0

        // Write-throughput window untouched
        col.record(MetricKind::WriteThroughput, 100.0);

        let read_stats = col.window_stats(MetricKind::ReadThroughput).unwrap();
        let write_stats = col.window_stats(MetricKind::WriteThroughput).unwrap();

        assert_eq!(read_stats.count, 3);
        assert!((read_stats.min - 20.0).abs() < f64::EPSILON);

        assert_eq!(write_stats.count, 1);
        assert!((write_stats.mean - 100.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // 18. check_alerts – both max and min thresholds on the same kind
    // -----------------------------------------------------------------------
    #[test]
    fn test_check_alerts_both_bounds_violated() {
        // We set up two separate threshold structs to get two independent alerts.
        let mut col = StorageMetricsCollector::new(100);
        col.record(MetricKind::ReadLatencyMs, 200.0);
        // mean = 200 → exceeds max of 100 AND exceeds min check irrelevant, so
        // instead violate min with a different metric kind.
        col.record(MetricKind::CacheHitRate, 0.1);

        col.add_threshold(AlertThreshold {
            kind: MetricKind::ReadLatencyMs,
            max_value: Some(100.0),
            min_value: None,
        });
        col.add_threshold(AlertThreshold {
            kind: MetricKind::CacheHitRate,
            max_value: None,
            min_value: Some(0.5),
        });

        let alerts = col.check_alerts();
        assert_eq!(alerts.len(), 2);
    }

    // -----------------------------------------------------------------------
    // 19. window_stats after eviction reflects only retained samples
    // -----------------------------------------------------------------------
    #[test]
    fn test_window_stats_after_eviction() {
        let mut col = StorageMetricsCollector::new(2);
        col.record(MetricKind::ErrorRate, 100.0); // will be evicted
        col.record(MetricKind::ErrorRate, 200.0);
        col.record(MetricKind::ErrorRate, 300.0); // evicts 100

        let stats = col.window_stats(MetricKind::ErrorRate).unwrap();
        assert_eq!(stats.count, 2);
        assert!((stats.min - 200.0).abs() < f64::EPSILON);
        assert!((stats.max - 300.0).abs() < f64::EPSILON);
        assert!((stats.mean - 250.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // 20. all_stats – empty collector returns empty vec
    // -----------------------------------------------------------------------
    #[test]
    fn test_all_stats_empty() {
        let col = StorageMetricsCollector::new(100);
        assert!(col.all_stats().is_empty());
    }
}
