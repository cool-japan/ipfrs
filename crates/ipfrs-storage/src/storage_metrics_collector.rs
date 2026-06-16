//! Comprehensive metrics collection and aggregation system for storage operations.
//!
//! Tracks latencies, throughput, error rates, and capacity utilization with
//! time-series bucketing. Samples are grouped into fixed-duration [`TimeBucket`]s
//! and evicted once the bucket ring exceeds [`CollectorConfig::max_buckets`].
//!
//! # Example
//!
//! ```rust
//! use ipfrs_storage::storage_metrics_collector::{
//!     CollectorConfig, MetricKind, MetricSample, StorageMetricsCollector,
//! };
//! use std::collections::HashMap;
//!
//! let now = 1_700_000_000_000_u64;
//! let config = CollectorConfig::default();
//! let mut col = StorageMetricsCollector::new(config, now);
//!
//! col.record_latency(MetricKind::ReadLatency, 3.5, now);
//! col.record_latency(MetricKind::WriteLatency, 7.2, now);
//!
//! let stats = col.aggregated_stats(&MetricKind::ReadLatency, 120_000, now);
//! assert_eq!(stats.sample_count, 1);
//! ```

use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// MetricKind
// ---------------------------------------------------------------------------

/// The kind of metric being recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MetricKind {
    /// Latency of read operations in milliseconds.
    ReadLatency,
    /// Latency of write operations in milliseconds.
    WriteLatency,
    /// Latency of delete operations in milliseconds.
    DeleteLatency,
    /// Incremented each time a block is found in cache.
    CacheHit,
    /// Incremented each time a block is not found in cache.
    CacheMiss,
    /// Bytes transferred in read operations.
    ThroughputBytesRead,
    /// Bytes transferred in write operations.
    ThroughputBytesWritten,
    /// Count of storage errors.
    ErrorCount,
    /// Capacity utilised in bytes.
    CapacityUsed,
}

impl MetricKind {
    /// Returns the canonical string key for this variant.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ReadLatency => "ReadLatency",
            Self::WriteLatency => "WriteLatency",
            Self::DeleteLatency => "DeleteLatency",
            Self::CacheHit => "CacheHit",
            Self::CacheMiss => "CacheMiss",
            Self::ThroughputBytesRead => "ThroughputBytesRead",
            Self::ThroughputBytesWritten => "ThroughputBytesWritten",
            Self::ErrorCount => "ErrorCount",
            Self::CapacityUsed => "CapacityUsed",
        }
    }
}

// ---------------------------------------------------------------------------
// MetricSample
// ---------------------------------------------------------------------------

/// A single recorded metric observation.
#[derive(Clone, Debug)]
pub struct MetricSample {
    /// The category of metric.
    pub kind: MetricKind,
    /// The numeric value (unit depends on `kind`).
    pub value: f64,
    /// Wall-clock timestamp in milliseconds since Unix epoch.
    pub timestamp_ms: u64,
    /// Arbitrary key-value tags for filtering/grouping.
    pub tags: HashMap<String, String>,
}

impl MetricSample {
    /// Convenience constructor with no tags.
    pub fn new(kind: MetricKind, value: f64, timestamp_ms: u64) -> Self {
        Self {
            kind,
            value,
            timestamp_ms,
            tags: HashMap::new(),
        }
    }

    /// Convenience constructor with tags.
    pub fn with_tags(
        kind: MetricKind,
        value: f64,
        timestamp_ms: u64,
        tags: HashMap<String, String>,
    ) -> Self {
        Self {
            kind,
            value,
            timestamp_ms,
            tags,
        }
    }
}

// ---------------------------------------------------------------------------
// TimeBucket
// ---------------------------------------------------------------------------

/// A fixed-duration time window that accumulates samples.
///
/// Stores up to `max_samples` individual values for percentile computation,
/// while always tracking `count`, `sum`, `min`, and `max` regardless of the
/// sample cap.
#[derive(Clone, Debug)]
pub struct TimeBucket {
    /// Inclusive start of this bucket's time range (ms since epoch).
    pub start_ms: u64,
    /// Exclusive end of this bucket's time range (ms since epoch).
    pub end_ms: u64,
    /// Individual sample values retained for percentile calculation.
    /// May be capped at `max_samples_per_bucket`.
    pub samples: Vec<f64>,
    /// Total number of observations (including those beyond the sample cap).
    pub count: u64,
    /// Sum of all observed values.
    pub sum: f64,
    /// Minimum observed value.
    pub min: f64,
    /// Maximum observed value.
    pub max: f64,
}

impl TimeBucket {
    /// Create an empty bucket covering `[start_ms, start_ms + duration_ms)`.
    pub fn new(start_ms: u64, duration_ms: u64) -> Self {
        Self {
            start_ms,
            end_ms: start_ms.saturating_add(duration_ms),
            samples: Vec::new(),
            count: 0,
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Returns `true` when `timestamp_ms` falls inside `[start_ms, end_ms)`.
    #[inline]
    pub fn contains(&self, timestamp_ms: u64) -> bool {
        timestamp_ms >= self.start_ms && timestamp_ms < self.end_ms
    }

    /// Arithmetic mean of all observed values, or `0.0` when empty.
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    /// 95th-percentile value computed from stored samples.
    ///
    /// Returns `0.0` when no samples are stored.
    pub fn p95(&self) -> f64 {
        percentile(&self.samples, 95.0)
    }

    /// 99th-percentile value computed from stored samples.
    ///
    /// Returns `0.0` when no samples are stored.
    pub fn p99(&self) -> f64 {
        percentile(&self.samples, 99.0)
    }

    /// Add a value to this bucket, capped at `max_samples`.
    pub(crate) fn push(&mut self, value: f64, max_samples: usize) {
        self.count += 1;
        self.sum += value;
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
        if self.samples.len() < max_samples {
            self.samples.push(value);
        }
    }

    /// Minimum, replacing sentinel `f64::INFINITY` with `0.0` for empty buckets.
    #[allow(dead_code)]
    pub(crate) fn safe_min(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.min
        }
    }

    /// Maximum, replacing sentinel `f64::NEG_INFINITY` with `0.0` for empty buckets.
    #[allow(dead_code)]
    pub(crate) fn safe_max(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.max
        }
    }
}

/// Sort-based linear-interpolation percentile over a slice of samples.
///
/// Clones and sorts the samples internally; returns `0.0` when empty.
fn percentile(samples: &[f64], pct: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((pct / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

// ---------------------------------------------------------------------------
// MetricSeries
// ---------------------------------------------------------------------------

/// An ordered ring of [`TimeBucket`]s for a single [`MetricKind`].
#[derive(Clone, Debug)]
pub struct MetricSeries {
    /// The metric this series tracks.
    pub kind: MetricKind,
    /// Time-ordered ring of buckets; front = oldest, back = newest.
    pub buckets: VecDeque<TimeBucket>,
    /// Duration of each bucket in milliseconds.
    pub bucket_duration_ms: u64,
    /// Maximum number of buckets retained before evicting the oldest.
    pub max_buckets: usize,
}

impl MetricSeries {
    /// Create an empty series.
    pub fn new(kind: MetricKind, bucket_duration_ms: u64, max_buckets: usize) -> Self {
        Self {
            kind,
            buckets: VecDeque::new(),
            bucket_duration_ms,
            max_buckets,
        }
    }

    /// Return a mutable reference to the bucket that should receive a sample
    /// at `timestamp_ms`, creating a new bucket (and evicting the oldest if
    /// necessary) when the sample falls outside the last bucket.
    pub(crate) fn get_or_create_bucket(
        &mut self,
        timestamp_ms: u64,
        max_samples: usize,
    ) -> &mut TimeBucket {
        // Compute the canonical bucket start for this timestamp.
        let bucket_start = (timestamp_ms / self.bucket_duration_ms) * self.bucket_duration_ms;

        // Check if a bucket already exists at this start position.
        // We do a read-only scan first to avoid borrow conflicts.
        let existing_idx = self.buckets.iter().position(|b| b.start_ms == bucket_start);

        if let Some(idx) = existing_idx {
            return &mut self.buckets[idx];
        }

        // Need a new bucket — evict oldest if at capacity.
        if self.buckets.len() >= self.max_buckets {
            self.buckets.pop_front();
        }

        let mut new_bucket = TimeBucket::new(bucket_start, self.bucket_duration_ms);
        new_bucket.samples.reserve(max_samples.min(256));
        self.buckets.push_back(new_bucket);

        self.buckets.back_mut().expect("just pushed")
    }

    /// Return the most recently active bucket that contains `now`, if any.
    pub fn current_bucket(&self, now: u64) -> Option<&TimeBucket> {
        self.buckets.iter().rev().find(|b| b.contains(now))
    }

    /// Iterate over all buckets whose `[start_ms, end_ms)` overlaps
    /// `[now - window_ms, now]`.
    pub fn buckets_in_window(&self, window_ms: u64, now: u64) -> impl Iterator<Item = &TimeBucket> {
        let window_start = now.saturating_sub(window_ms);
        self.buckets
            .iter()
            .filter(move |b| b.end_ms > window_start && b.start_ms <= now)
    }

    /// The `start_ms` of the oldest bucket, or `0` when empty.
    pub fn oldest_start_ms(&self) -> u64 {
        self.buckets.front().map_or(0, |b| b.start_ms)
    }
}

// ---------------------------------------------------------------------------
// CollectorConfig
// ---------------------------------------------------------------------------

/// Configuration for [`StorageMetricsCollector`].
#[derive(Clone, Debug)]
pub struct CollectorConfig {
    /// Duration of each time bucket in milliseconds (default: 60 000 = 1 min).
    pub bucket_duration_ms: u64,
    /// Maximum number of buckets retained per series (default: 1 440 = 24 hrs).
    pub max_buckets: usize,
    /// Maximum individual samples stored per bucket for percentile calculation.
    pub max_samples_per_bucket: usize,
    /// When `true`, percentile fields in [`AggregatedStats`] are populated.
    pub enable_percentiles: bool,
}

impl Default for CollectorConfig {
    fn default() -> Self {
        Self {
            bucket_duration_ms: 60_000,
            max_buckets: 1_440,
            max_samples_per_bucket: 10_000,
            enable_percentiles: true,
        }
    }
}

// ---------------------------------------------------------------------------
// AggregatedStats
// ---------------------------------------------------------------------------

/// Aggregated statistics for a [`MetricKind`] over a time window.
#[derive(Clone, Debug)]
pub struct AggregatedStats {
    /// String name of the metric kind.
    pub kind: String,
    /// Total number of observations in the window.
    pub sample_count: u64,
    /// Arithmetic mean of all observations.
    pub mean: f64,
    /// Minimum observed value.
    pub min: f64,
    /// Maximum observed value.
    pub max: f64,
    /// 95th-percentile latency (only meaningful when `enable_percentiles` is true).
    pub p95: f64,
    /// 99th-percentile latency (only meaningful when `enable_percentiles` is true).
    pub p99: f64,
    /// Sum of all observations.
    pub sum: f64,
}

impl AggregatedStats {
    /// Returns an empty stats object with all zeroes.
    pub fn empty(kind: &MetricKind) -> Self {
        Self {
            kind: kind.as_str().to_owned(),
            sample_count: 0,
            mean: 0.0,
            min: 0.0,
            max: 0.0,
            p95: 0.0,
            p99: 0.0,
            sum: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// CollectorStats
// ---------------------------------------------------------------------------

/// Snapshot of the collector's own health metrics.
#[derive(Clone, Debug)]
pub struct CollectorStats {
    /// Number of active metric series.
    pub active_series: usize,
    /// Total samples recorded since construction.
    pub total_samples: u64,
    /// Elapsed time since construction in milliseconds.
    pub uptime_ms: u64,
    /// Timestamp of the oldest retained data, in ms since epoch.
    pub oldest_data_ms: u64,
}

// ---------------------------------------------------------------------------
// StorageMetricsCollector
// ---------------------------------------------------------------------------

/// Production-grade metrics collection and aggregation for storage operations.
///
/// Samples are bucketed by wall-clock time into fixed-duration [`TimeBucket`]s.
/// Old buckets are evicted once the series exceeds [`CollectorConfig::max_buckets`].
/// Aggregated statistics (mean, min, max, p95, p99) are computed on demand.
pub struct StorageMetricsCollector {
    /// Configuration controlling bucket sizes and retention.
    pub config: CollectorConfig,
    /// Per-kind metric time-series, keyed by `MetricKind::as_str()`.
    pub series: HashMap<String, MetricSeries>,
    /// Total samples ingested since construction.
    pub total_samples: u64,
    /// Wall-clock ms at construction time.
    pub started_at: u64,
}

impl StorageMetricsCollector {
    /// Create a new collector with the given configuration, anchored at `now`.
    pub fn new(config: CollectorConfig, now: u64) -> Self {
        Self {
            config,
            series: HashMap::new(),
            total_samples: 0,
            started_at: now,
        }
    }

    /// Return the canonical map key for a `MetricKind`.
    pub fn metric_key(kind: &MetricKind) -> String {
        kind.as_str().to_owned()
    }

    /// Record a [`MetricSample`].
    ///
    /// Looks up (or creates) the [`MetricSeries`] for `sample.kind`, then finds
    /// (or creates) the [`TimeBucket`] covering `sample.timestamp_ms`, and
    /// pushes the value into it. Evicts the oldest bucket when the series is
    /// at capacity.
    pub fn record(&mut self, sample: MetricSample) {
        let key = Self::metric_key(&sample.kind);
        let bucket_duration_ms = self.config.bucket_duration_ms;
        let max_buckets = self.config.max_buckets;
        let max_samples = self.config.max_samples_per_bucket;

        let series = self
            .series
            .entry(key)
            .or_insert_with(|| MetricSeries::new(sample.kind, bucket_duration_ms, max_buckets));

        let bucket = series.get_or_create_bucket(sample.timestamp_ms, max_samples);
        bucket.push(sample.value, max_samples);
        self.total_samples += 1;
    }

    /// Convenience: record a latency observation with no tags.
    ///
    /// `kind` must be one of `ReadLatency`, `WriteLatency`, or `DeleteLatency`
    /// (any `MetricKind` is accepted, but caller is responsible for semantics).
    pub fn record_latency(&mut self, kind: MetricKind, latency_ms: f64, now: u64) {
        self.record(MetricSample::new(kind, latency_ms, now));
    }

    /// Convenience: record a throughput observation.
    ///
    /// `bytes` is the number of bytes transferred; `is_write` selects between
    /// [`MetricKind::ThroughputBytesWritten`] and [`MetricKind::ThroughputBytesRead`].
    pub fn record_throughput(&mut self, bytes: u64, is_write: bool, now: u64) {
        let kind = if is_write {
            MetricKind::ThroughputBytesWritten
        } else {
            MetricKind::ThroughputBytesRead
        };
        self.record(MetricSample::new(kind, bytes as f64, now));
    }

    /// Return the most recent bucket for `kind` that contains `now`, if any.
    pub fn current_bucket(&self, kind: &MetricKind, now: u64) -> Option<&TimeBucket> {
        let key = Self::metric_key(kind);
        self.series.get(&key)?.current_bucket(now)
    }

    /// Aggregate all samples for `kind` within `[now - window_ms, now]`.
    ///
    /// Returns [`AggregatedStats::empty`] when no data is available.
    pub fn aggregated_stats(&self, kind: &MetricKind, window_ms: u64, now: u64) -> AggregatedStats {
        let key = Self::metric_key(kind);
        let series = match self.series.get(&key) {
            Some(s) => s,
            None => return AggregatedStats::empty(kind),
        };

        let mut total_count: u64 = 0;
        let mut total_sum: f64 = 0.0;
        let mut global_min = f64::INFINITY;
        let mut global_max = f64::NEG_INFINITY;
        let mut all_samples: Vec<f64> = Vec::new();

        for bucket in series.buckets_in_window(window_ms, now) {
            total_count += bucket.count;
            total_sum += bucket.sum;
            if bucket.min < global_min {
                global_min = bucket.min;
            }
            if bucket.max > global_max {
                global_max = bucket.max;
            }
            if self.config.enable_percentiles {
                all_samples.extend_from_slice(&bucket.samples);
            }
        }

        if total_count == 0 {
            return AggregatedStats::empty(kind);
        }

        let mean = total_sum / total_count as f64;
        let (p95, p99) = if self.config.enable_percentiles {
            (
                percentile(&all_samples, 95.0),
                percentile(&all_samples, 99.0),
            )
        } else {
            (0.0, 0.0)
        };

        AggregatedStats {
            kind: kind.as_str().to_owned(),
            sample_count: total_count,
            mean,
            min: if global_min.is_infinite() {
                0.0
            } else {
                global_min
            },
            max: if global_max.is_infinite() {
                0.0
            } else {
                global_max
            },
            p95,
            p99,
            sum: total_sum,
        }
    }

    /// Compute read and write throughput in bytes per second over `window_ms`.
    ///
    /// Returns `(read_bps, write_bps)`.
    pub fn throughput_bps(&self, window_ms: u64, now: u64) -> (f64, f64) {
        let read_stats = self.aggregated_stats(&MetricKind::ThroughputBytesRead, window_ms, now);
        let write_stats =
            self.aggregated_stats(&MetricKind::ThroughputBytesWritten, window_ms, now);

        let window_seconds = window_ms as f64 / 1_000.0;
        if window_seconds <= 0.0 {
            return (0.0, 0.0);
        }

        (
            read_stats.sum / window_seconds,
            write_stats.sum / window_seconds,
        )
    }

    /// Compute error rate as `error_count / total_ops` over `window_ms`.
    ///
    /// "total_ops" is the sum of reads + writes + errors observed in the window.
    /// Returns `0.0` when no operations have been recorded.
    pub fn error_rate(&self, window_ms: u64, now: u64) -> f64 {
        let errors = self.aggregated_stats(&MetricKind::ErrorCount, window_ms, now);
        let reads = self.aggregated_stats(&MetricKind::ReadLatency, window_ms, now);
        let writes = self.aggregated_stats(&MetricKind::WriteLatency, window_ms, now);

        let total_ops = errors.sample_count + reads.sample_count + writes.sample_count;
        if total_ops == 0 {
            0.0
        } else {
            errors.sample_count as f64 / total_ops as f64
        }
    }

    /// Compute cache hit rate as `hits / (hits + misses)` over `window_ms`.
    ///
    /// Returns `0.0` when no cache events have been recorded.
    pub fn cache_hit_rate(&self, window_ms: u64, now: u64) -> f64 {
        let hits = self.aggregated_stats(&MetricKind::CacheHit, window_ms, now);
        let misses = self.aggregated_stats(&MetricKind::CacheMiss, window_ms, now);

        let total = hits.sample_count + misses.sample_count;
        if total == 0 {
            0.0
        } else {
            hits.sample_count as f64 / total as f64
        }
    }

    /// Return a snapshot of the collector's own health.
    pub fn collector_stats(&self, now: u64) -> CollectorStats {
        let oldest_data_ms = self
            .series
            .values()
            .map(|s| s.oldest_start_ms())
            .filter(|&ms| ms > 0)
            .min()
            .unwrap_or(0);

        CollectorStats {
            active_series: self.series.len(),
            total_samples: self.total_samples,
            uptime_ms: now.saturating_sub(self.started_at),
            oldest_data_ms,
        }
    }

    /// Return the names of all series that currently hold data.
    pub fn active_series_names(&self) -> Vec<&str> {
        self.series.keys().map(String::as_str).collect()
    }

    /// Reset a specific series, discarding all buckets.
    pub fn reset_series(&mut self, kind: &MetricKind) {
        let key = Self::metric_key(kind);
        self.series.remove(&key);
    }

    /// Reset the entire collector, discarding all series and resetting counts.
    pub fn reset_all(&mut self, now: u64) {
        self.series.clear();
        self.total_samples = 0;
        self.started_at = now;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::storage_metrics_collector::{
        percentile, AggregatedStats, CollectorConfig, MetricKind, MetricSample, MetricSeries,
        StorageMetricsCollector, TimeBucket,
    };

    const NOW: u64 = 1_700_000_000_000; // arbitrary epoch ms

    fn default_collector() -> StorageMetricsCollector {
        StorageMetricsCollector::new(CollectorConfig::default(), NOW)
    }

    // -----------------------------------------------------------------------
    // 1. MetricKind::as_str round-trips
    // -----------------------------------------------------------------------
    #[test]
    fn test_metric_kind_as_str_all_variants() {
        assert_eq!(MetricKind::ReadLatency.as_str(), "ReadLatency");
        assert_eq!(MetricKind::WriteLatency.as_str(), "WriteLatency");
        assert_eq!(MetricKind::DeleteLatency.as_str(), "DeleteLatency");
        assert_eq!(MetricKind::CacheHit.as_str(), "CacheHit");
        assert_eq!(MetricKind::CacheMiss.as_str(), "CacheMiss");
        assert_eq!(
            MetricKind::ThroughputBytesRead.as_str(),
            "ThroughputBytesRead"
        );
        assert_eq!(
            MetricKind::ThroughputBytesWritten.as_str(),
            "ThroughputBytesWritten"
        );
        assert_eq!(MetricKind::ErrorCount.as_str(), "ErrorCount");
        assert_eq!(MetricKind::CapacityUsed.as_str(), "CapacityUsed");
    }

    // -----------------------------------------------------------------------
    // 2. metric_key matches as_str
    // -----------------------------------------------------------------------
    #[test]
    fn test_metric_key_matches_as_str() {
        for kind in [
            MetricKind::ReadLatency,
            MetricKind::WriteLatency,
            MetricKind::CacheHit,
        ] {
            assert_eq!(StorageMetricsCollector::metric_key(&kind), kind.as_str());
        }
    }

    // -----------------------------------------------------------------------
    // 3. MetricSample::new has empty tags
    // -----------------------------------------------------------------------
    #[test]
    fn test_metric_sample_new_no_tags() {
        let s = MetricSample::new(MetricKind::ReadLatency, 5.0, NOW);
        assert_eq!(s.kind, MetricKind::ReadLatency);
        assert_eq!(s.value, 5.0);
        assert_eq!(s.timestamp_ms, NOW);
        assert!(s.tags.is_empty());
    }

    // -----------------------------------------------------------------------
    // 4. MetricSample::with_tags preserves tags
    // -----------------------------------------------------------------------
    #[test]
    fn test_metric_sample_with_tags() {
        let mut tags = HashMap::new();
        tags.insert("region".to_owned(), "us-east-1".to_owned());
        let s = MetricSample::with_tags(MetricKind::WriteLatency, 10.0, NOW, tags);
        assert_eq!(s.tags.get("region").map(String::as_str), Some("us-east-1"));
    }

    // -----------------------------------------------------------------------
    // 5. TimeBucket::new initialises sentinels
    // -----------------------------------------------------------------------
    #[test]
    fn test_time_bucket_new() {
        let b = TimeBucket::new(1000, 500);
        assert_eq!(b.start_ms, 1000);
        assert_eq!(b.end_ms, 1500);
        assert_eq!(b.count, 0);
        assert_eq!(b.sum, 0.0);
        assert!(b.min.is_infinite() && b.min > 0.0);
        assert!(b.max.is_infinite() && b.max < 0.0);
    }

    // -----------------------------------------------------------------------
    // 6. TimeBucket::contains
    // -----------------------------------------------------------------------
    #[test]
    fn test_time_bucket_contains() {
        let b = TimeBucket::new(1000, 500);
        assert!(b.contains(1000));
        assert!(b.contains(1499));
        assert!(!b.contains(999));
        assert!(!b.contains(1500));
    }

    // -----------------------------------------------------------------------
    // 7. TimeBucket::push updates all stats
    // -----------------------------------------------------------------------
    #[test]
    fn test_time_bucket_push_stats() {
        let mut b = TimeBucket::new(0, 60_000);
        b.push(10.0, 100);
        b.push(20.0, 100);
        b.push(30.0, 100);
        assert_eq!(b.count, 3);
        assert!((b.sum - 60.0).abs() < 1e-9);
        assert!((b.min - 10.0).abs() < 1e-9);
        assert!((b.max - 30.0).abs() < 1e-9);
        assert!((b.mean() - 20.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 8. TimeBucket::mean on empty bucket returns 0
    // -----------------------------------------------------------------------
    #[test]
    fn test_time_bucket_mean_empty() {
        let b = TimeBucket::new(0, 60_000);
        assert_eq!(b.mean(), 0.0);
    }

    // -----------------------------------------------------------------------
    // 9. TimeBucket::p95 / p99 on empty returns 0
    // -----------------------------------------------------------------------
    #[test]
    fn test_time_bucket_percentiles_empty() {
        let b = TimeBucket::new(0, 60_000);
        assert_eq!(b.p95(), 0.0);
        assert_eq!(b.p99(), 0.0);
    }

    // -----------------------------------------------------------------------
    // 10. TimeBucket::p95 correctness
    // -----------------------------------------------------------------------
    #[test]
    fn test_time_bucket_p95_correctness() {
        let mut b = TimeBucket::new(0, 60_000);
        for i in 1..=100_u64 {
            b.push(i as f64, 200);
        }
        // p95 of [1..=100] should be 95
        let p95 = b.p95();
        assert!((p95 - 95.0).abs() < 1.0, "p95={p95}");
    }

    // -----------------------------------------------------------------------
    // 11. TimeBucket sample cap does not break min/max/sum/count
    // -----------------------------------------------------------------------
    #[test]
    fn test_time_bucket_sample_cap() {
        let mut b = TimeBucket::new(0, 60_000);
        for i in 0..200_u64 {
            b.push(i as f64, 10); // cap at 10 samples
        }
        assert_eq!(b.count, 200);
        assert_eq!(b.samples.len(), 10);
        assert!((b.min - 0.0).abs() < 1e-9);
        assert!((b.max - 199.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 12. percentile helper: single element
    // -----------------------------------------------------------------------
    #[test]
    fn test_percentile_single_element() {
        assert_eq!(percentile(&[42.0], 50.0), 42.0);
        assert_eq!(percentile(&[42.0], 99.0), 42.0);
    }

    // -----------------------------------------------------------------------
    // 13. percentile helper: empty slice
    // -----------------------------------------------------------------------
    #[test]
    fn test_percentile_empty_slice() {
        assert_eq!(percentile(&[], 95.0), 0.0);
    }

    // -----------------------------------------------------------------------
    // 14. CollectorConfig defaults
    // -----------------------------------------------------------------------
    #[test]
    fn test_collector_config_defaults() {
        let c = CollectorConfig::default();
        assert_eq!(c.bucket_duration_ms, 60_000);
        assert_eq!(c.max_buckets, 1_440);
        assert_eq!(c.max_samples_per_bucket, 10_000);
        assert!(c.enable_percentiles);
    }

    // -----------------------------------------------------------------------
    // 15. new collector starts empty
    // -----------------------------------------------------------------------
    #[test]
    fn test_new_collector_empty() {
        let col = default_collector();
        assert_eq!(col.total_samples, 0);
        assert!(col.series.is_empty());
        assert_eq!(col.started_at, NOW);
    }

    // -----------------------------------------------------------------------
    // 16. record creates a series and bucket
    // -----------------------------------------------------------------------
    #[test]
    fn test_record_creates_series() {
        let mut col = default_collector();
        col.record(MetricSample::new(MetricKind::ReadLatency, 5.0, NOW));
        assert_eq!(col.total_samples, 1);
        let key = StorageMetricsCollector::metric_key(&MetricKind::ReadLatency);
        assert!(col.series.contains_key(&key));
    }

    // -----------------------------------------------------------------------
    // 17. record_latency convenience
    // -----------------------------------------------------------------------
    #[test]
    fn test_record_latency_convenience() {
        let mut col = default_collector();
        col.record_latency(MetricKind::WriteLatency, 12.5, NOW);
        assert_eq!(col.total_samples, 1);
    }

    // -----------------------------------------------------------------------
    // 18. record_throughput selects correct kind
    // -----------------------------------------------------------------------
    #[test]
    fn test_record_throughput_read_vs_write() {
        let mut col = default_collector();
        col.record_throughput(1024, false, NOW);
        col.record_throughput(2048, true, NOW);

        let read_key = StorageMetricsCollector::metric_key(&MetricKind::ThroughputBytesRead);
        let write_key = StorageMetricsCollector::metric_key(&MetricKind::ThroughputBytesWritten);
        assert!(col.series.contains_key(&read_key));
        assert!(col.series.contains_key(&write_key));
    }

    // -----------------------------------------------------------------------
    // 19. aggregated_stats returns empty for missing kind
    // -----------------------------------------------------------------------
    #[test]
    fn test_aggregated_stats_missing_kind() {
        let col = default_collector();
        let stats = col.aggregated_stats(&MetricKind::ReadLatency, 60_000, NOW);
        assert_eq!(stats.sample_count, 0);
        assert_eq!(stats.mean, 0.0);
    }

    // -----------------------------------------------------------------------
    // 20. aggregated_stats basic accuracy
    // -----------------------------------------------------------------------
    #[test]
    fn test_aggregated_stats_accuracy() {
        let mut col = default_collector();
        col.record_latency(MetricKind::ReadLatency, 10.0, NOW);
        col.record_latency(MetricKind::ReadLatency, 20.0, NOW);
        col.record_latency(MetricKind::ReadLatency, 30.0, NOW);

        let stats = col.aggregated_stats(&MetricKind::ReadLatency, 120_000, NOW);
        assert_eq!(stats.sample_count, 3);
        assert!((stats.mean - 20.0).abs() < 1e-9, "mean={}", stats.mean);
        assert!((stats.min - 10.0).abs() < 1e-9);
        assert!((stats.max - 30.0).abs() < 1e-9);
        assert!((stats.sum - 60.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 21. aggregated_stats respects time window — older data excluded
    // -----------------------------------------------------------------------
    #[test]
    fn test_aggregated_stats_window_exclusion() {
        let mut col = default_collector();
        // old sample 2 hours ago
        let old_ts = NOW.saturating_sub(2 * 3_600_000);
        col.record_latency(MetricKind::ReadLatency, 999.0, old_ts);
        // fresh sample
        col.record_latency(MetricKind::ReadLatency, 5.0, NOW);

        // 1-minute window — should only see 5.0
        let stats = col.aggregated_stats(&MetricKind::ReadLatency, 60_000, NOW);
        assert_eq!(stats.sample_count, 1);
        assert!((stats.mean - 5.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 22. aggregated_stats includes p95/p99 when enable_percentiles=true
    // -----------------------------------------------------------------------
    #[test]
    fn test_aggregated_stats_percentiles_enabled() {
        let mut col = default_collector();
        for i in 1..=100 {
            col.record_latency(MetricKind::ReadLatency, i as f64, NOW);
        }
        let stats = col.aggregated_stats(&MetricKind::ReadLatency, 120_000, NOW);
        assert!(stats.p95 > 0.0, "p95 should be non-zero");
        assert!(stats.p99 >= stats.p95, "p99 >= p95");
    }

    // -----------------------------------------------------------------------
    // 23. aggregated_stats zeroes p95/p99 when enable_percentiles=false
    // -----------------------------------------------------------------------
    #[test]
    fn test_aggregated_stats_percentiles_disabled() {
        let config = CollectorConfig {
            enable_percentiles: false,
            ..Default::default()
        };
        let mut col = StorageMetricsCollector::new(config, NOW);
        col.record_latency(MetricKind::ReadLatency, 50.0, NOW);
        let stats = col.aggregated_stats(&MetricKind::ReadLatency, 120_000, NOW);
        assert_eq!(stats.p95, 0.0);
        assert_eq!(stats.p99, 0.0);
    }

    // -----------------------------------------------------------------------
    // 24. throughput_bps computes bytes / seconds
    // -----------------------------------------------------------------------
    #[test]
    fn test_throughput_bps() {
        let mut col = default_collector();
        // record 10 000 bytes of reads and 5 000 bytes of writes
        col.record_throughput(10_000, false, NOW);
        col.record_throughput(5_000, true, NOW);

        let window_ms = 10_000_u64; // 10 seconds
        let (read_bps, write_bps) = col.throughput_bps(window_ms, NOW);
        // read: 10_000 / 10 = 1_000
        assert!((read_bps - 1_000.0).abs() < 1.0, "read_bps={read_bps}");
        // write: 5_000 / 10 = 500
        assert!((write_bps - 500.0).abs() < 1.0, "write_bps={write_bps}");
    }

    // -----------------------------------------------------------------------
    // 25. throughput_bps on empty collector returns (0, 0)
    // -----------------------------------------------------------------------
    #[test]
    fn test_throughput_bps_empty() {
        let col = default_collector();
        let (r, w) = col.throughput_bps(60_000, NOW);
        assert_eq!(r, 0.0);
        assert_eq!(w, 0.0);
    }

    // -----------------------------------------------------------------------
    // 26. error_rate basic
    // -----------------------------------------------------------------------
    #[test]
    fn test_error_rate_basic() {
        let mut col = default_collector();
        // 1 error, 4 reads, 5 writes => 1 / 10 = 0.1
        col.record(MetricSample::new(MetricKind::ErrorCount, 1.0, NOW));
        for _ in 0..4 {
            col.record_latency(MetricKind::ReadLatency, 1.0, NOW);
        }
        for _ in 0..5 {
            col.record_latency(MetricKind::WriteLatency, 1.0, NOW);
        }
        let rate = col.error_rate(60_000, NOW);
        assert!((rate - 0.1).abs() < 1e-9, "rate={rate}");
    }

    // -----------------------------------------------------------------------
    // 27. error_rate returns 0 when no ops
    // -----------------------------------------------------------------------
    #[test]
    fn test_error_rate_no_ops() {
        let col = default_collector();
        assert_eq!(col.error_rate(60_000, NOW), 0.0);
    }

    // -----------------------------------------------------------------------
    // 28. cache_hit_rate basic
    // -----------------------------------------------------------------------
    #[test]
    fn test_cache_hit_rate_basic() {
        let mut col = default_collector();
        for _ in 0..3 {
            col.record(MetricSample::new(MetricKind::CacheHit, 1.0, NOW));
        }
        for _ in 0..1 {
            col.record(MetricSample::new(MetricKind::CacheMiss, 1.0, NOW));
        }
        let rate = col.cache_hit_rate(60_000, NOW);
        assert!((rate - 0.75).abs() < 1e-9, "rate={rate}");
    }

    // -----------------------------------------------------------------------
    // 29. cache_hit_rate returns 0 when no cache events
    // -----------------------------------------------------------------------
    #[test]
    fn test_cache_hit_rate_no_events() {
        let col = default_collector();
        assert_eq!(col.cache_hit_rate(60_000, NOW), 0.0);
    }

    // -----------------------------------------------------------------------
    // 30. current_bucket returns None for missing series
    // -----------------------------------------------------------------------
    #[test]
    fn test_current_bucket_missing_series() {
        let col = default_collector();
        assert!(col.current_bucket(&MetricKind::ReadLatency, NOW).is_none());
    }

    // -----------------------------------------------------------------------
    // 31. current_bucket returns bucket after recording
    // -----------------------------------------------------------------------
    #[test]
    fn test_current_bucket_present() {
        let mut col = default_collector();
        col.record_latency(MetricKind::ReadLatency, 5.0, NOW);
        let bucket = col.current_bucket(&MetricKind::ReadLatency, NOW);
        assert!(bucket.is_some());
        let b = bucket.expect("bucket must exist");
        assert_eq!(b.count, 1);
    }

    // -----------------------------------------------------------------------
    // 32. collector_stats uptime
    // -----------------------------------------------------------------------
    #[test]
    fn test_collector_stats_uptime() {
        let col = default_collector();
        let now2 = NOW + 5_000;
        let stats = col.collector_stats(now2);
        assert_eq!(stats.uptime_ms, 5_000);
        assert_eq!(stats.active_series, 0);
        assert_eq!(stats.total_samples, 0);
    }

    // -----------------------------------------------------------------------
    // 33. collector_stats active series count
    // -----------------------------------------------------------------------
    #[test]
    fn test_collector_stats_active_series() {
        let mut col = default_collector();
        col.record_latency(MetricKind::ReadLatency, 1.0, NOW);
        col.record_latency(MetricKind::WriteLatency, 2.0, NOW);
        let stats = col.collector_stats(NOW);
        assert_eq!(stats.active_series, 2);
        assert_eq!(stats.total_samples, 2);
    }

    // -----------------------------------------------------------------------
    // 34. bucket eviction when max_buckets exceeded
    // -----------------------------------------------------------------------
    #[test]
    fn test_bucket_eviction_on_overflow() {
        let config = CollectorConfig {
            bucket_duration_ms: 1_000,
            max_buckets: 3,
            max_samples_per_bucket: 100,
            enable_percentiles: true,
        };
        let mut col = StorageMetricsCollector::new(config, NOW);
        // Record samples in 5 distinct buckets (1 second apart)
        for i in 0..5_u64 {
            col.record_latency(MetricKind::ReadLatency, i as f64, NOW + i * 1_000);
        }
        let key = StorageMetricsCollector::metric_key(&MetricKind::ReadLatency);
        let series = col.series.get(&key).expect("series must exist");
        assert!(
            series.buckets.len() <= 3,
            "expected ≤3 buckets, got {}",
            series.buckets.len()
        );
    }

    // -----------------------------------------------------------------------
    // 35. multiple kinds are independent
    // -----------------------------------------------------------------------
    #[test]
    fn test_multiple_kinds_independent() {
        let mut col = default_collector();
        col.record_latency(MetricKind::ReadLatency, 10.0, NOW);
        col.record_latency(MetricKind::WriteLatency, 50.0, NOW);

        let read = col.aggregated_stats(&MetricKind::ReadLatency, 120_000, NOW);
        let write = col.aggregated_stats(&MetricKind::WriteLatency, 120_000, NOW);
        assert!((read.mean - 10.0).abs() < 1e-9);
        assert!((write.mean - 50.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 36. reset_series clears one series leaving others intact
    // -----------------------------------------------------------------------
    #[test]
    fn test_reset_series_selective() {
        let mut col = default_collector();
        col.record_latency(MetricKind::ReadLatency, 1.0, NOW);
        col.record_latency(MetricKind::WriteLatency, 2.0, NOW);
        col.reset_series(&MetricKind::ReadLatency);

        let read_key = StorageMetricsCollector::metric_key(&MetricKind::ReadLatency);
        let write_key = StorageMetricsCollector::metric_key(&MetricKind::WriteLatency);
        assert!(!col.series.contains_key(&read_key));
        assert!(col.series.contains_key(&write_key));
    }

    // -----------------------------------------------------------------------
    // 37. reset_all wipes everything
    // -----------------------------------------------------------------------
    #[test]
    fn test_reset_all() {
        let mut col = default_collector();
        col.record_latency(MetricKind::ReadLatency, 5.0, NOW);
        col.record_latency(MetricKind::WriteLatency, 7.0, NOW);
        let new_now = NOW + 10_000;
        col.reset_all(new_now);
        assert!(col.series.is_empty());
        assert_eq!(col.total_samples, 0);
        assert_eq!(col.started_at, new_now);
    }

    // -----------------------------------------------------------------------
    // 38. active_series_names returns correct keys
    // -----------------------------------------------------------------------
    #[test]
    fn test_active_series_names() {
        let mut col = default_collector();
        col.record_latency(MetricKind::CacheHit, 1.0, NOW);
        col.record_latency(MetricKind::CacheMiss, 1.0, NOW);
        let mut names = col.active_series_names();
        names.sort_unstable();
        assert_eq!(names, vec!["CacheHit", "CacheMiss"]);
    }

    // -----------------------------------------------------------------------
    // 39. MetricSeries oldest_start_ms
    // -----------------------------------------------------------------------
    #[test]
    fn test_metric_series_oldest_start_ms_empty() {
        let s = MetricSeries::new(MetricKind::ReadLatency, 60_000, 10);
        assert_eq!(s.oldest_start_ms(), 0);
    }

    // -----------------------------------------------------------------------
    // 40. MetricSeries oldest_start_ms after recording
    // -----------------------------------------------------------------------
    #[test]
    fn test_metric_series_oldest_start_ms_after_push() {
        let mut col = default_collector();
        // bucket 0: NOW ms
        col.record_latency(MetricKind::ReadLatency, 1.0, NOW);
        // bucket 1: NOW + 60s
        col.record_latency(MetricKind::ReadLatency, 2.0, NOW + 60_000);
        let key = StorageMetricsCollector::metric_key(&MetricKind::ReadLatency);
        let series = col.series.get(&key).expect("series");
        let oldest = series.oldest_start_ms();
        // The oldest bucket aligns to bucket_duration_ms boundary
        assert!(oldest <= NOW, "oldest={oldest}");
    }

    // -----------------------------------------------------------------------
    // 41. AggregatedStats::empty has correct kind string
    // -----------------------------------------------------------------------
    #[test]
    fn test_aggregated_stats_empty_kind_string() {
        let s = AggregatedStats::empty(&MetricKind::ErrorCount);
        assert_eq!(s.kind, "ErrorCount");
        assert_eq!(s.sample_count, 0);
    }

    // -----------------------------------------------------------------------
    // 42. TimeBucket safe_min / safe_max on empty
    // -----------------------------------------------------------------------
    #[test]
    fn test_time_bucket_safe_extremes_empty() {
        let b = TimeBucket::new(0, 1000);
        assert_eq!(b.safe_min(), 0.0);
        assert_eq!(b.safe_max(), 0.0);
    }

    // -----------------------------------------------------------------------
    // 43. Collector handles zero-window throughput gracefully
    // -----------------------------------------------------------------------
    #[test]
    fn test_throughput_bps_zero_window() {
        let mut col = default_collector();
        col.record_throughput(1_000, false, NOW);
        let (r, w) = col.throughput_bps(0, NOW);
        // window_seconds = 0 => return (0, 0)
        assert_eq!(r, 0.0);
        assert_eq!(w, 0.0);
    }

    // -----------------------------------------------------------------------
    // 44. Samples in future bucket not counted in narrow past window
    // -----------------------------------------------------------------------
    #[test]
    fn test_future_samples_excluded_from_past_window() {
        let mut col = default_collector();
        col.record_latency(MetricKind::ReadLatency, 99.0, NOW + 3_600_000);
        // window ending at NOW, 60 seconds back — future bucket must be excluded
        let stats = col.aggregated_stats(&MetricKind::ReadLatency, 60_000, NOW);
        assert_eq!(stats.sample_count, 0);
    }

    // -----------------------------------------------------------------------
    // 45. total_samples increments correctly for each record call
    // -----------------------------------------------------------------------
    #[test]
    fn test_total_samples_counter() {
        let mut col = default_collector();
        for i in 0..10 {
            col.record_latency(MetricKind::ReadLatency, i as f64, NOW);
        }
        assert_eq!(col.total_samples, 10);
    }

    // -----------------------------------------------------------------------
    // 46. collector_stats oldest_data_ms is non-zero after recording
    // -----------------------------------------------------------------------
    #[test]
    fn test_collector_stats_oldest_data_ms() {
        let mut col = default_collector();
        col.record_latency(MetricKind::ReadLatency, 1.0, NOW);
        let stats = col.collector_stats(NOW + 1000);
        assert!(stats.oldest_data_ms > 0, "oldest_data_ms should be > 0");
    }

    // -----------------------------------------------------------------------
    // 47. p99 is always >= p95
    // -----------------------------------------------------------------------
    #[test]
    fn test_p99_gte_p95_property() {
        let mut col = default_collector();
        for i in (1..=50).rev() {
            col.record_latency(MetricKind::WriteLatency, i as f64, NOW);
        }
        let stats = col.aggregated_stats(&MetricKind::WriteLatency, 120_000, NOW);
        assert!(
            stats.p99 >= stats.p95,
            "p99={} p95={}",
            stats.p99,
            stats.p95
        );
    }

    // -----------------------------------------------------------------------
    // 48. Multiple buckets across window are all summed
    // -----------------------------------------------------------------------
    #[test]
    fn test_multi_bucket_window_sum() {
        let config = CollectorConfig {
            bucket_duration_ms: 60_000,
            max_buckets: 1_440,
            max_samples_per_bucket: 1_000,
            enable_percentiles: true,
        };
        let mut col = StorageMetricsCollector::new(config, NOW);
        // 3 different buckets: NOW, NOW+1min, NOW+2min
        col.record_latency(MetricKind::ReadLatency, 10.0, NOW);
        col.record_latency(MetricKind::ReadLatency, 20.0, NOW + 60_000);
        col.record_latency(MetricKind::ReadLatency, 30.0, NOW + 120_000);
        let now3 = NOW + 180_000;
        let stats = col.aggregated_stats(&MetricKind::ReadLatency, 300_000, now3);
        assert_eq!(stats.sample_count, 3);
        assert!((stats.sum - 60.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 49. CapacityUsed metric recorded and retrieved
    // -----------------------------------------------------------------------
    #[test]
    fn test_capacity_used_metric() {
        let mut col = default_collector();
        col.record(MetricSample::new(
            MetricKind::CapacityUsed,
            10_000_000.0,
            NOW,
        ));
        let stats = col.aggregated_stats(&MetricKind::CapacityUsed, 120_000, NOW);
        assert_eq!(stats.sample_count, 1);
        assert!((stats.mean - 10_000_000.0).abs() < 1.0);
    }

    // -----------------------------------------------------------------------
    // 50. DeleteLatency metric kind works end-to-end
    // -----------------------------------------------------------------------
    #[test]
    fn test_delete_latency_end_to_end() {
        let mut col = default_collector();
        col.record_latency(MetricKind::DeleteLatency, 2.5, NOW);
        col.record_latency(MetricKind::DeleteLatency, 3.5, NOW);
        let stats = col.aggregated_stats(&MetricKind::DeleteLatency, 60_000, NOW);
        assert_eq!(stats.sample_count, 2);
        assert!((stats.mean - 3.0).abs() < 1e-9);
    }
}
