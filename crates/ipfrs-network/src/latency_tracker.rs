//! Per-peer latency histogram tracking with percentile reporting.
//!
//! This module provides fine-grained latency monitoring for each connected peer,
//! including:
//! - Recording RTT measurements per peer with ring-buffer sample management
//! - Percentile computation with linear interpolation (p50, p95, p99, arbitrary)
//! - Histogram generation with configurable bucket counts
//! - Fastest/slowest peer ranking by mean latency
//! - Global aggregate statistics across all tracked peers
//!
//! # Example
//!
//! ```rust
//! use ipfrs_network::latency_tracker::{PeerLatencyTracker};
//!
//! let mut tracker = PeerLatencyTracker::new(1000);
//!
//! tracker.record("peer-1", 1_500);
//! tracker.record("peer-1", 2_000);
//! tracker.record("peer-1", 1_800);
//!
//! if let Some(p99) = tracker.percentile("peer-1", 0.99) {
//!     println!("P99 = {} us", p99);
//! }
//!
//! if let Some(m) = tracker.mean("peer-1") {
//!     println!("Mean = {:.1} us", m);
//! }
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// LatencyBucket
// ---------------------------------------------------------------------------

/// A single bucket in a latency histogram.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatencyBucket {
    /// Lower bound of the bucket (inclusive), in microseconds.
    pub lower_bound_us: u64,
    /// Upper bound of the bucket (exclusive for all but the last bucket), in microseconds.
    pub upper_bound_us: u64,
    /// Number of samples falling into this bucket.
    pub count: u64,
}

// ---------------------------------------------------------------------------
// PeerLatency
// ---------------------------------------------------------------------------

/// Per-peer latency state with ring-buffer sample storage.
#[derive(Debug, Clone)]
pub struct PeerLatency {
    /// Stable identifier for the remote peer.
    pub peer_id: String,
    /// Raw latency samples in microseconds (ring buffer).
    pub samples: Vec<u64>,
    /// Maximum number of samples retained (ring buffer capacity).
    pub max_samples: usize,
    /// Minimum observed latency in microseconds.
    pub min_us: u64,
    /// Maximum observed latency in microseconds.
    pub max_us: u64,
    /// Cumulative sum of all recorded latencies in microseconds.
    pub sum_us: u64,
    /// Total number of latency samples ever recorded (not just in the buffer).
    pub count: u64,
    /// Write position for ring buffer (next index to overwrite).
    write_pos: usize,
    /// Whether the ring buffer has wrapped around at least once.
    wrapped: bool,
}

impl PeerLatency {
    /// Create a new, empty peer latency record.
    fn new(peer_id: impl Into<String>, max_samples: usize) -> Self {
        Self {
            peer_id: peer_id.into(),
            samples: Vec::with_capacity(max_samples.min(1024)),
            max_samples,
            min_us: u64::MAX,
            max_us: 0,
            sum_us: 0,
            count: 0,
            write_pos: 0,
            wrapped: false,
        }
    }

    /// Add a latency sample, maintaining the ring buffer invariant.
    fn add(&mut self, latency_us: u64) {
        if self.max_samples == 0 {
            // Degenerate case: no storage allowed, still track stats.
            self.count += 1;
            self.sum_us = self.sum_us.saturating_add(latency_us);
            if latency_us < self.min_us {
                self.min_us = latency_us;
            }
            if latency_us > self.max_us {
                self.max_us = latency_us;
            }
            return;
        }

        if self.samples.len() < self.max_samples {
            // Still filling the buffer.
            self.samples.push(latency_us);
        } else {
            // Overwrite oldest entry.
            self.samples[self.write_pos] = latency_us;
            self.wrapped = true;
        }
        self.write_pos = (self.write_pos + 1) % self.max_samples;

        self.count += 1;
        self.sum_us = self.sum_us.saturating_add(latency_us);
        if latency_us < self.min_us {
            self.min_us = latency_us;
        }
        if latency_us > self.max_us {
            self.max_us = latency_us;
        }
    }

    /// Return a sorted copy of the current samples.
    fn sorted_samples(&self) -> Vec<u64> {
        let mut sorted = self.samples.clone();
        sorted.sort_unstable();
        sorted
    }
}

// ---------------------------------------------------------------------------
// LatencyTrackerStats
// ---------------------------------------------------------------------------

/// Aggregate statistics produced by [`PeerLatencyTracker::stats`].
#[derive(Debug, Clone, PartialEq)]
pub struct LatencyTrackerStats {
    /// Number of distinct peers currently being tracked.
    pub tracked_peers: usize,
    /// Total number of samples recorded across all peers.
    pub global_samples: u64,
    /// Global mean latency in microseconds, or `None` if no samples.
    pub global_mean_us: Option<f64>,
}

// ---------------------------------------------------------------------------
// PeerLatencyTracker
// ---------------------------------------------------------------------------

/// Tracks per-peer RTT measurements with histogram, percentile, and ranking support.
///
/// Each peer maintains a fixed-size ring buffer of raw latency samples.
/// Statistics (min, max, sum, count) are tracked cumulatively so that
/// mean computation remains accurate even after samples are evicted.
#[derive(Debug)]
pub struct PeerLatencyTracker {
    /// Per-peer latency state.
    peers: HashMap<String, PeerLatency>,
    /// Maximum number of samples to retain per peer.
    max_samples_per_peer: usize,
    /// Global cumulative sample count.
    global_count: u64,
    /// Global cumulative sum of all latency values.
    global_sum_us: u64,
}

impl PeerLatencyTracker {
    /// Create a new tracker with the specified ring buffer size per peer.
    pub fn new(max_samples_per_peer: usize) -> Self {
        Self {
            peers: HashMap::new(),
            max_samples_per_peer,
            global_count: 0,
            global_sum_us: 0,
        }
    }

    /// Record a latency sample for the given peer.
    ///
    /// A [`PeerLatency`] entry is automatically created if one does not already
    /// exist. The ring buffer evicts the oldest sample when full.
    pub fn record(&mut self, peer_id: &str, latency_us: u64) {
        let max = self.max_samples_per_peer;
        let entry = self
            .peers
            .entry(peer_id.to_string())
            .or_insert_with(|| PeerLatency::new(peer_id, max));
        entry.add(latency_us);

        self.global_count += 1;
        self.global_sum_us = self.global_sum_us.saturating_add(latency_us);
    }

    /// Compute a percentile value for a peer using linear interpolation.
    ///
    /// `p` must be in `[0.0, 1.0]` (e.g., 0.99 for p99).
    /// Returns `None` if the peer is unknown or has no samples.
    pub fn percentile(&self, peer_id: &str, p: f64) -> Option<u64> {
        let entry = self.peers.get(peer_id)?;
        if entry.samples.is_empty() {
            return None;
        }

        let p = p.clamp(0.0, 1.0);
        let sorted = entry.sorted_samples();
        let n = sorted.len();

        if n == 1 {
            return Some(sorted[0]);
        }

        // Linear interpolation using the "C = 0" variant (R-7 in R terminology).
        let rank = p * (n - 1) as f64;
        let lower_idx = rank.floor() as usize;
        let upper_idx = rank.ceil() as usize;

        if lower_idx == upper_idx {
            return Some(sorted[lower_idx]);
        }

        let frac = rank - lower_idx as f64;
        let lower_val = sorted[lower_idx] as f64;
        let upper_val = sorted[upper_idx] as f64;
        let interpolated = lower_val + frac * (upper_val - lower_val);

        Some(interpolated.round() as u64)
    }

    /// Compute the mean latency for a peer.
    ///
    /// Uses cumulative sum/count for accuracy, not just buffered samples.
    /// Returns `None` if the peer is unknown or has no samples.
    pub fn mean(&self, peer_id: &str) -> Option<f64> {
        let entry = self.peers.get(peer_id)?;
        if entry.count == 0 {
            return None;
        }
        Some(entry.sum_us as f64 / entry.count as f64)
    }

    /// Compute the median (p50) latency for a peer.
    ///
    /// Returns `None` if the peer is unknown or has no samples.
    pub fn median(&self, peer_id: &str) -> Option<u64> {
        self.percentile(peer_id, 0.5)
    }

    /// Generate a histogram of latency samples for a peer.
    ///
    /// The range `[min, max]` is divided into `bucket_count` equal-width buckets.
    /// Returns `None` if the peer is unknown, has no samples, or `bucket_count` is 0.
    pub fn histogram(&self, peer_id: &str, bucket_count: usize) -> Option<Vec<LatencyBucket>> {
        if bucket_count == 0 {
            return None;
        }

        let entry = self.peers.get(peer_id)?;
        if entry.samples.is_empty() {
            return None;
        }

        let sorted = entry.sorted_samples();
        let min_val = sorted[0];
        let max_val = sorted[sorted.len() - 1];

        // When all samples are equal, put everything in one bucket.
        if min_val == max_val {
            let mut buckets = Vec::with_capacity(bucket_count);
            for i in 0..bucket_count {
                let count = if i == 0 { sorted.len() as u64 } else { 0 };
                buckets.push(LatencyBucket {
                    lower_bound_us: min_val,
                    upper_bound_us: max_val,
                    count,
                });
            }
            return Some(buckets);
        }

        let range = max_val - min_val;
        let bucket_width = range as f64 / bucket_count as f64;
        let mut buckets = Vec::with_capacity(bucket_count);

        for i in 0..bucket_count {
            let lower = min_val as f64 + i as f64 * bucket_width;
            let upper = if i == bucket_count - 1 {
                max_val as f64 + 1.0 // inclusive last bucket
            } else {
                min_val as f64 + (i + 1) as f64 * bucket_width
            };

            let count = sorted
                .iter()
                .filter(|&&v| {
                    let vf = v as f64;
                    if i == bucket_count - 1 {
                        vf >= lower && vf <= max_val as f64
                    } else {
                        vf >= lower && vf < upper
                    }
                })
                .count() as u64;

            buckets.push(LatencyBucket {
                lower_bound_us: lower.floor() as u64,
                upper_bound_us: if i == bucket_count - 1 {
                    max_val
                } else {
                    upper.ceil() as u64
                },
                count,
            });
        }

        Some(buckets)
    }

    /// Return the `n` peers with the lowest mean latency.
    ///
    /// Results are sorted ascending by mean latency.
    pub fn fastest_peers(&self, n: usize) -> Vec<(String, f64)> {
        let mut means: Vec<(String, f64)> = self
            .peers
            .iter()
            .filter(|(_, v)| v.count > 0)
            .map(|(k, v)| (k.clone(), v.sum_us as f64 / v.count as f64))
            .collect();

        means.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        means.truncate(n);
        means
    }

    /// Return the `n` peers with the highest mean latency.
    ///
    /// Results are sorted descending by mean latency.
    pub fn slowest_peers(&self, n: usize) -> Vec<(String, f64)> {
        let mut means: Vec<(String, f64)> = self
            .peers
            .iter()
            .filter(|(_, v)| v.count > 0)
            .map(|(k, v)| (k.clone(), v.sum_us as f64 / v.count as f64))
            .collect();

        means.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        means.truncate(n);
        means
    }

    /// Remove all tracking state for `peer_id`.
    ///
    /// Returns `true` if the peer existed and was removed.
    pub fn remove_peer(&mut self, peer_id: &str) -> bool {
        self.peers.remove(peer_id).is_some()
    }

    /// Return the number of peers currently being tracked.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Compute the global mean latency across all peers.
    ///
    /// Returns `None` when no samples have been recorded.
    pub fn global_mean(&self) -> Option<f64> {
        if self.global_count == 0 {
            return None;
        }
        Some(self.global_sum_us as f64 / self.global_count as f64)
    }

    /// Return aggregate statistics for the tracker.
    pub fn stats(&self) -> LatencyTrackerStats {
        LatencyTrackerStats {
            tracked_peers: self.peers.len(),
            global_samples: self.global_count,
            global_mean_us: self.global_mean(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tracker() -> PeerLatencyTracker {
        PeerLatencyTracker::new(1000)
    }

    // -------------------------------------------------------------------
    // Basic record tests
    // -------------------------------------------------------------------

    #[test]
    fn record_creates_peer_entry() {
        let mut t = make_tracker();
        t.record("p1", 200);
        assert_eq!(t.peer_count(), 1);
    }

    #[test]
    fn record_multiple_samples_same_peer() {
        let mut t = make_tracker();
        t.record("p1", 100);
        t.record("p1", 200);
        t.record("p1", 300);
        let entry = t.peers.get("p1").expect("peer should exist");
        assert_eq!(entry.samples.len(), 3);
        assert_eq!(entry.count, 3);
        assert_eq!(entry.sum_us, 600);
    }

    #[test]
    fn record_multiple_peers() {
        let mut t = make_tracker();
        t.record("p1", 100);
        t.record("p2", 200);
        t.record("p3", 300);
        assert_eq!(t.peer_count(), 3);
    }

    #[test]
    fn record_updates_min_max() {
        let mut t = make_tracker();
        t.record("p1", 500);
        t.record("p1", 100);
        t.record("p1", 900);
        let entry = t.peers.get("p1").expect("peer should exist");
        assert_eq!(entry.min_us, 100);
        assert_eq!(entry.max_us, 900);
    }

    // -------------------------------------------------------------------
    // Ring buffer eviction
    // -------------------------------------------------------------------

    #[test]
    fn ring_buffer_eviction() {
        let mut t = PeerLatencyTracker::new(3);
        t.record("p1", 10);
        t.record("p1", 20);
        t.record("p1", 30);
        t.record("p1", 40); // evicts 10

        let entry = t.peers.get("p1").expect("peer should exist");
        assert_eq!(entry.samples.len(), 3);
        assert_eq!(entry.count, 4);
        // Ring buffer overwrites index 0 with 40.
        let mut sorted = entry.samples.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![20, 30, 40]);
    }

    #[test]
    fn ring_buffer_size_one() {
        let mut t = PeerLatencyTracker::new(1);
        t.record("p1", 100);
        t.record("p1", 200);
        let entry = t.peers.get("p1").expect("peer should exist");
        assert_eq!(entry.samples.len(), 1);
        assert_eq!(entry.samples[0], 200);
        assert_eq!(entry.count, 2);
    }

    #[test]
    fn ring_buffer_full_cycle() {
        let mut t = PeerLatencyTracker::new(5);
        for i in 1..=10u64 {
            t.record("p1", i * 100);
        }
        let entry = t.peers.get("p1").expect("peer should exist");
        assert_eq!(entry.samples.len(), 5);
        assert_eq!(entry.count, 10);
        // Last 5 values: 600, 700, 800, 900, 1000
        let mut sorted = entry.samples.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![600, 700, 800, 900, 1000]);
    }

    // -------------------------------------------------------------------
    // Percentile tests
    // -------------------------------------------------------------------

    #[test]
    fn percentile_none_for_unknown_peer() {
        let t = make_tracker();
        assert!(t.percentile("nobody", 0.5).is_none());
    }

    #[test]
    fn percentile_single_sample() {
        let mut t = make_tracker();
        t.record("p1", 500);
        assert_eq!(t.percentile("p1", 0.0), Some(500));
        assert_eq!(t.percentile("p1", 0.5), Some(500));
        assert_eq!(t.percentile("p1", 1.0), Some(500));
    }

    #[test]
    fn percentile_p50_even_count() {
        let mut t = make_tracker();
        // 10 samples: 100, 200, ..., 1000
        for i in 1..=10u64 {
            t.record("p1", i * 100);
        }
        // p50: rank = 0.5 * 9 = 4.5 -> interpolate between sorted[4]=500 and sorted[5]=600
        // result = 500 + 0.5 * 100 = 550
        let p50 = t.percentile("p1", 0.5).expect("should have data");
        assert_eq!(p50, 550);
    }

    #[test]
    fn percentile_p95() {
        let mut t = make_tracker();
        for i in 1..=100u64 {
            t.record("p1", i);
        }
        // p95: rank = 0.95 * 99 = 94.05
        // sorted[94]=95, sorted[95]=96
        // result = 95 + 0.05 * 1 = 95.05 -> rounds to 95
        let p95 = t.percentile("p1", 0.95).expect("should have data");
        assert_eq!(p95, 95);
    }

    #[test]
    fn percentile_p99() {
        let mut t = make_tracker();
        for i in 1..=100u64 {
            t.record("p1", i);
        }
        // p99: rank = 0.99 * 99 = 98.01
        // sorted[98]=99, sorted[99]=100
        // result = 99 + 0.01 * 1 = 99.01 -> rounds to 99
        let p99 = t.percentile("p1", 0.99).expect("should have data");
        assert_eq!(p99, 99);
    }

    #[test]
    fn percentile_clamped() {
        let mut t = make_tracker();
        t.record("p1", 100);
        t.record("p1", 200);
        // p < 0 should clamp to 0 -> returns min
        assert_eq!(t.percentile("p1", -1.0), Some(100));
        // p > 1 should clamp to 1 -> returns max
        assert_eq!(t.percentile("p1", 2.0), Some(200));
    }

    #[test]
    fn percentile_p0_returns_min() {
        let mut t = make_tracker();
        t.record("p1", 100);
        t.record("p1", 200);
        t.record("p1", 300);
        assert_eq!(t.percentile("p1", 0.0), Some(100));
    }

    #[test]
    fn percentile_p100_returns_max() {
        let mut t = make_tracker();
        t.record("p1", 100);
        t.record("p1", 200);
        t.record("p1", 300);
        assert_eq!(t.percentile("p1", 1.0), Some(300));
    }

    // -------------------------------------------------------------------
    // Mean / Median tests
    // -------------------------------------------------------------------

    #[test]
    fn mean_none_for_unknown_peer() {
        let t = make_tracker();
        assert!(t.mean("nobody").is_none());
    }

    #[test]
    fn mean_single_sample() {
        let mut t = make_tracker();
        t.record("p1", 500);
        assert!((t.mean("p1").expect("should have data") - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mean_multiple_samples() {
        let mut t = make_tracker();
        t.record("p1", 100);
        t.record("p1", 200);
        t.record("p1", 300);
        let m = t.mean("p1").expect("should have data");
        assert!((m - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn median_none_for_unknown_peer() {
        let t = make_tracker();
        assert!(t.median("nobody").is_none());
    }

    #[test]
    fn median_returns_p50() {
        let mut t = make_tracker();
        t.record("p1", 100);
        t.record("p1", 200);
        t.record("p1", 300);
        // p50: rank = 0.5 * 2 = 1.0 -> sorted[1] = 200
        assert_eq!(t.median("p1"), Some(200));
    }

    // -------------------------------------------------------------------
    // Histogram tests
    // -------------------------------------------------------------------

    #[test]
    fn histogram_none_for_unknown_peer() {
        let t = make_tracker();
        assert!(t.histogram("nobody", 5).is_none());
    }

    #[test]
    fn histogram_none_for_zero_buckets() {
        let mut t = make_tracker();
        t.record("p1", 100);
        assert!(t.histogram("p1", 0).is_none());
    }

    #[test]
    fn histogram_single_bucket() {
        let mut t = make_tracker();
        t.record("p1", 100);
        t.record("p1", 200);
        t.record("p1", 300);
        let hist = t.histogram("p1", 1).expect("should have histogram");
        assert_eq!(hist.len(), 1);
        assert_eq!(hist[0].count, 3);
    }

    #[test]
    fn histogram_bucket_count_matches() {
        let mut t = make_tracker();
        for i in 1..=100u64 {
            t.record("p1", i);
        }
        let hist = t.histogram("p1", 10).expect("should have histogram");
        assert_eq!(hist.len(), 10);
        // All samples should be accounted for.
        let total: u64 = hist.iter().map(|b| b.count).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn histogram_all_same_values() {
        let mut t = make_tracker();
        for _ in 0..5 {
            t.record("p1", 42);
        }
        let hist = t.histogram("p1", 3).expect("should have histogram");
        assert_eq!(hist.len(), 3);
        // All samples in first bucket.
        assert_eq!(hist[0].count, 5);
        assert_eq!(hist[1].count, 0);
        assert_eq!(hist[2].count, 0);
    }

    // -------------------------------------------------------------------
    // Fastest / Slowest peers
    // -------------------------------------------------------------------

    #[test]
    fn fastest_peers_ordering() {
        let mut t = make_tracker();
        t.record("fast", 100);
        t.record("medium", 500);
        t.record("slow", 1000);
        let fastest = t.fastest_peers(3);
        assert_eq!(fastest.len(), 3);
        assert_eq!(fastest[0].0, "fast");
        assert_eq!(fastest[1].0, "medium");
        assert_eq!(fastest[2].0, "slow");
    }

    #[test]
    fn fastest_peers_truncates() {
        let mut t = make_tracker();
        t.record("a", 100);
        t.record("b", 200);
        t.record("c", 300);
        let fastest = t.fastest_peers(2);
        assert_eq!(fastest.len(), 2);
    }

    #[test]
    fn fastest_peers_empty() {
        let t = make_tracker();
        let fastest = t.fastest_peers(5);
        assert!(fastest.is_empty());
    }

    #[test]
    fn slowest_peers_ordering() {
        let mut t = make_tracker();
        t.record("fast", 100);
        t.record("medium", 500);
        t.record("slow", 1000);
        let slowest = t.slowest_peers(3);
        assert_eq!(slowest.len(), 3);
        assert_eq!(slowest[0].0, "slow");
        assert_eq!(slowest[1].0, "medium");
        assert_eq!(slowest[2].0, "fast");
    }

    #[test]
    fn slowest_peers_truncates() {
        let mut t = make_tracker();
        t.record("a", 100);
        t.record("b", 200);
        t.record("c", 300);
        let slowest = t.slowest_peers(1);
        assert_eq!(slowest.len(), 1);
        assert_eq!(slowest[0].0, "c");
    }

    // -------------------------------------------------------------------
    // Remove peer
    // -------------------------------------------------------------------

    #[test]
    fn remove_peer_returns_true_when_present() {
        let mut t = make_tracker();
        t.record("p1", 100);
        assert!(t.remove_peer("p1"));
        assert_eq!(t.peer_count(), 0);
    }

    #[test]
    fn remove_peer_returns_false_when_absent() {
        let mut t = make_tracker();
        assert!(!t.remove_peer("nobody"));
    }

    #[test]
    fn remove_peer_then_reinsert() {
        let mut t = make_tracker();
        t.record("p1", 100);
        t.remove_peer("p1");
        t.record("p1", 500);
        let entry = t.peers.get("p1").expect("peer should exist");
        assert_eq!(entry.count, 1);
        assert_eq!(entry.samples, vec![500]);
    }

    // -------------------------------------------------------------------
    // Global stats
    // -------------------------------------------------------------------

    #[test]
    fn global_mean_none_when_empty() {
        let t = make_tracker();
        assert!(t.global_mean().is_none());
    }

    #[test]
    fn global_mean_single_peer() {
        let mut t = make_tracker();
        t.record("p1", 100);
        t.record("p1", 200);
        assert!((t.global_mean().expect("should exist") - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn global_mean_multiple_peers() {
        let mut t = make_tracker();
        t.record("p1", 100);
        t.record("p2", 300);
        // global mean = (100 + 300) / 2 = 200
        assert!((t.global_mean().expect("should exist") - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_empty_tracker() {
        let t = make_tracker();
        let s = t.stats();
        assert_eq!(s.tracked_peers, 0);
        assert_eq!(s.global_samples, 0);
        assert!(s.global_mean_us.is_none());
    }

    #[test]
    fn stats_populated_tracker() {
        let mut t = make_tracker();
        t.record("p1", 100);
        t.record("p1", 200);
        t.record("p2", 300);
        let s = t.stats();
        assert_eq!(s.tracked_peers, 2);
        assert_eq!(s.global_samples, 3);
        let mean = s.global_mean_us.expect("should have mean");
        assert!((mean - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_after_remove_peer() {
        let mut t = make_tracker();
        t.record("p1", 100);
        t.record("p2", 200);
        t.remove_peer("p1");
        let s = t.stats();
        assert_eq!(s.tracked_peers, 1);
        // global_count is cumulative (not decremented on remove)
        assert_eq!(s.global_samples, 2);
    }

    // -------------------------------------------------------------------
    // Edge cases
    // -------------------------------------------------------------------

    #[test]
    fn single_sample_edge_case() {
        let mut t = make_tracker();
        t.record("p1", 42);
        assert_eq!(t.percentile("p1", 0.0), Some(42));
        assert_eq!(t.percentile("p1", 0.5), Some(42));
        assert_eq!(t.percentile("p1", 1.0), Some(42));
        assert_eq!(t.median("p1"), Some(42));
        assert!((t.mean("p1").expect("mean") - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn empty_peer_returns_none() {
        let t = make_tracker();
        assert!(t.percentile("ghost", 0.5).is_none());
        assert!(t.mean("ghost").is_none());
        assert!(t.median("ghost").is_none());
        assert!(t.histogram("ghost", 5).is_none());
    }

    #[test]
    fn peer_count_after_operations() {
        let mut t = make_tracker();
        assert_eq!(t.peer_count(), 0);
        t.record("p1", 100);
        assert_eq!(t.peer_count(), 1);
        t.record("p2", 200);
        assert_eq!(t.peer_count(), 2);
        t.remove_peer("p1");
        assert_eq!(t.peer_count(), 1);
    }

    #[test]
    fn large_sample_count() {
        let mut t = PeerLatencyTracker::new(100);
        for i in 0..500u64 {
            t.record("p1", i + 1);
        }
        let entry = t.peers.get("p1").expect("peer should exist");
        assert_eq!(entry.samples.len(), 100);
        assert_eq!(entry.count, 500);
        // Mean is cumulative: (1+2+...+500)/500 = 250.5
        let m = t.mean("p1").expect("mean");
        assert!((m - 250.5).abs() < f64::EPSILON);
    }

    #[test]
    fn histogram_two_samples_two_buckets() {
        let mut t = make_tracker();
        t.record("p1", 0);
        t.record("p1", 100);
        let hist = t.histogram("p1", 2).expect("should have histogram");
        assert_eq!(hist.len(), 2);
        let total: u64 = hist.iter().map(|b| b.count).sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn fastest_slowest_with_varied_means() {
        let mut t = make_tracker();
        // Give each peer multiple samples with different means.
        t.record("alpha", 100);
        t.record("alpha", 200); // mean 150
        t.record("beta", 400);
        t.record("beta", 600); // mean 500
        t.record("gamma", 50);
        t.record("gamma", 50); // mean 50

        let fastest = t.fastest_peers(2);
        assert_eq!(fastest[0].0, "gamma");
        assert_eq!(fastest[1].0, "alpha");

        let slowest = t.slowest_peers(2);
        assert_eq!(slowest[0].0, "beta");
        assert_eq!(slowest[1].0, "alpha");
    }
}
