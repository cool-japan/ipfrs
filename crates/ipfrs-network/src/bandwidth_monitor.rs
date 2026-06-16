//! Per-peer and aggregate bandwidth usage tracking with sliding window rate calculation.
//!
//! This module provides fine-grained bandwidth monitoring for each connected peer,
//! including:
//! - Recording inbound and outbound bytes per peer with timestamps
//! - Sliding window rate calculation (bytes/sec) over a configurable window
//! - Peak rate detection based on inter-sample intervals
//! - Aggregate statistics across all peers
//! - Atomic global counters for lock-free stats snapshots
//! - Idle peer eviction to bound memory usage
//!
//! # Example
//!
//! ```rust
//! use ipfrs_network::bandwidth_monitor::{BandwidthMonitor, Direction};
//! use std::time::Duration;
//!
//! let monitor = BandwidthMonitor::with_window(Duration::from_secs(10));
//!
//! monitor.record("peer-1", 1024, Direction::Inbound);
//! monitor.record("peer-1", 512, Direction::Outbound);
//!
//! let inbound_rate = monitor.rate_for_peer("peer-1", Direction::Inbound);
//! println!("peer-1 inbound rate: {:.1} B/s", inbound_rate);
//!
//! let top = monitor.top_receivers(5);
//! println!("top receiver: {:?}", top.first());
//! ```

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Direction
// ---------------------------------------------------------------------------

/// Traffic direction for a bandwidth sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Direction {
    /// Bytes received from a remote peer.
    Inbound,
    /// Bytes sent to a remote peer.
    Outbound,
}

// ---------------------------------------------------------------------------
// BandwidthSample
// ---------------------------------------------------------------------------

/// A single bandwidth observation recorded at a specific point in time.
#[derive(Debug, Clone)]
pub struct BandwidthSample {
    /// Number of bytes transferred in this sample.
    pub bytes: u64,
    /// Wall-clock instant when the transfer was observed.
    pub timestamp: Instant,
    /// Whether the bytes were received or sent.
    pub direction: Direction,
}

impl BandwidthSample {
    /// Create a new sample stamped at `now`.
    pub fn new(bytes: u64, direction: Direction, now: Instant) -> Self {
        Self {
            bytes,
            timestamp: now,
            direction,
        }
    }
}

// ---------------------------------------------------------------------------
// PeerBandwidth
// ---------------------------------------------------------------------------

/// Per-peer bandwidth state keeping a sliding window of recent samples.
#[derive(Debug)]
pub struct PeerBandwidth {
    /// Stable identifier for this peer (e.g. libp2p PeerId string).
    pub peer_id: String,
    /// Inbound samples within the current retention window.
    pub inbound_samples: VecDeque<BandwidthSample>,
    /// Outbound samples within the current retention window.
    pub outbound_samples: VecDeque<BandwidthSample>,
    /// Running total of all inbound bytes ever recorded (never decremented).
    pub total_inbound_bytes: u64,
    /// Running total of all outbound bytes ever recorded (never decremented).
    pub total_outbound_bytes: u64,
}

impl PeerBandwidth {
    /// Create a new, empty `PeerBandwidth` for the given peer.
    pub fn new(peer_id: impl Into<String>) -> Self {
        Self {
            peer_id: peer_id.into(),
            inbound_samples: VecDeque::new(),
            outbound_samples: VecDeque::new(),
            total_inbound_bytes: 0,
            total_outbound_bytes: 0,
        }
    }

    /// Record a transfer of `bytes` in `direction` at time `now`.
    ///
    /// Updates both the sliding-window sample queue and the all-time totals.
    pub fn record(&mut self, bytes: u64, direction: Direction, now: Instant) {
        let sample = BandwidthSample::new(bytes, direction, now);
        match direction {
            Direction::Inbound => {
                self.inbound_samples.push_back(sample);
                self.total_inbound_bytes = self.total_inbound_bytes.saturating_add(bytes);
            }
            Direction::Outbound => {
                self.outbound_samples.push_back(sample);
                self.total_outbound_bytes = self.total_outbound_bytes.saturating_add(bytes);
            }
        }
    }

    /// Remove samples older than `window` before `now` from both queues.
    pub fn evict_old(&mut self, now: Instant, window: Duration) {
        let cutoff = now.checked_sub(window).unwrap_or(now);
        while let Some(front) = self.inbound_samples.front() {
            if front.timestamp <= cutoff {
                self.inbound_samples.pop_front();
            } else {
                break;
            }
        }
        while let Some(front) = self.outbound_samples.front() {
            if front.timestamp <= cutoff {
                self.outbound_samples.pop_front();
            } else {
                break;
            }
        }
    }

    /// Compute the average rate (bytes/sec) for `direction` within the last
    /// `window` seconds ending at `now`.
    ///
    /// Returns `0.0` when there are no samples in the window or the window
    /// duration is zero.
    pub fn rate_bps(&self, direction: Direction, now: Instant, window: Duration) -> f64 {
        let window_secs = window.as_secs_f64();
        if window_secs <= 0.0 {
            return 0.0;
        }
        let cutoff = now.checked_sub(window).unwrap_or(now);
        let queue = match direction {
            Direction::Inbound => &self.inbound_samples,
            Direction::Outbound => &self.outbound_samples,
        };
        let total_bytes: u64 = queue
            .iter()
            .filter(|s| s.timestamp > cutoff)
            .map(|s| s.bytes)
            .sum();
        total_bytes as f64 / window_secs
    }

    /// Compute the peak instantaneous rate (bytes/sec) for `direction`.
    ///
    /// The peak is defined as the maximum single-sample byte count divided by
    /// the elapsed time since the previous sample in the queue.  The very first
    /// sample is excluded because there is no prior reference point.
    ///
    /// Returns `0.0` when fewer than two samples exist.
    pub fn peak_rate_bps(&self, direction: Direction) -> f64 {
        let queue = match direction {
            Direction::Inbound => &self.inbound_samples,
            Direction::Outbound => &self.outbound_samples,
        };
        if queue.len() < 2 {
            return 0.0;
        }
        let mut peak: f64 = 0.0;
        let samples: Vec<&BandwidthSample> = queue.iter().collect();
        for i in 1..samples.len() {
            let elapsed = samples[i]
                .timestamp
                .duration_since(samples[i - 1].timestamp)
                .as_secs_f64();
            if elapsed > 0.0 {
                let rate = samples[i].bytes as f64 / elapsed;
                if rate > peak {
                    peak = rate;
                }
            }
        }
        peak
    }

    /// Return the timestamp of the most recent sample in either direction,
    /// or `None` if no samples have been recorded yet.
    pub fn last_activity(&self) -> Option<Instant> {
        let inbound_ts = self.inbound_samples.back().map(|s| s.timestamp);
        let outbound_ts = self.outbound_samples.back().map(|s| s.timestamp);
        match (inbound_ts, outbound_ts) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// BandwidthStats
// ---------------------------------------------------------------------------

/// Atomic global counters for lock-free stats collection.
#[derive(Debug, Default)]
pub struct BandwidthStats {
    /// Total inbound bytes recorded across all peers since monitor creation.
    pub total_inbound_bytes: AtomicU64,
    /// Total outbound bytes recorded across all peers since monitor creation.
    pub total_outbound_bytes: AtomicU64,
    /// Total number of individual samples recorded (both directions).
    pub total_samples: AtomicU64,
}

/// A point-in-time snapshot of [`BandwidthStats`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthStatsSnapshot {
    /// Total inbound bytes at snapshot time.
    pub total_inbound_bytes: u64,
    /// Total outbound bytes at snapshot time.
    pub total_outbound_bytes: u64,
    /// Total sample count at snapshot time.
    pub total_samples: u64,
}

impl BandwidthStats {
    /// Create zeroed stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Take a consistent snapshot of all counters.
    ///
    /// Note: each field is read atomically but the three reads are not
    /// collectively atomic; the snapshot may reflect slightly different
    /// instants for each counter under heavy concurrent load.
    pub fn snapshot(&self) -> BandwidthStatsSnapshot {
        BandwidthStatsSnapshot {
            total_inbound_bytes: self.total_inbound_bytes.load(Ordering::Relaxed),
            total_outbound_bytes: self.total_outbound_bytes.load(Ordering::Relaxed),
            total_samples: self.total_samples.load(Ordering::Relaxed),
        }
    }
}

// ---------------------------------------------------------------------------
// BandwidthMonitor
// ---------------------------------------------------------------------------

/// Monitor that tracks per-peer and aggregate bandwidth usage.
///
/// `BandwidthMonitor` maintains a map of [`PeerBandwidth`] entries protected
/// by a [`RwLock`].  All mutating operations take a write lock; read-only
/// queries take a read lock.  Global byte totals are accumulated in
/// [`BandwidthStats`] using lock-free atomics, so callers can take cheap
/// snapshots without acquiring the peer map lock.
///
/// The sliding window used for rate calculations is configurable at
/// construction time (defaults to 10 seconds).  Samples older than the
/// window are lazily evicted on each call to [`record`](BandwidthMonitor::record).
pub struct BandwidthMonitor {
    /// Per-peer bandwidth state.
    peers: RwLock<HashMap<String, PeerBandwidth>>,
    /// Duration of the sliding window for rate calculations.
    window: Duration,
    /// Global atomic counters.
    stats: Arc<BandwidthStats>,
}

impl BandwidthMonitor {
    /// Create a new monitor with the default 10-second sliding window.
    pub fn new() -> Self {
        Self::with_window(Duration::from_secs(10))
    }

    /// Create a new monitor with a custom sliding `window` duration.
    pub fn with_window(window: Duration) -> Self {
        Self {
            peers: RwLock::new(HashMap::new()),
            window,
            stats: Arc::new(BandwidthStats::new()),
        }
    }

    /// Return the configured sliding window duration.
    pub fn window(&self) -> Duration {
        self.window
    }

    /// Return a shared reference to the global atomic stats.
    pub fn stats(&self) -> &Arc<BandwidthStats> {
        &self.stats
    }

    /// Record a transfer of `bytes` in `direction` for `peer_id`.
    ///
    /// This method:
    /// 1. Obtains a write lock on the peer map.
    /// 2. Creates a [`PeerBandwidth`] entry if one does not exist.
    /// 3. Appends a new sample stamped with the current instant.
    /// 4. Evicts samples older than the sliding window from that peer.
    /// 5. Increments the global atomic counters.
    pub fn record(&self, peer_id: &str, bytes: u64, direction: Direction) {
        let now = Instant::now();
        {
            let mut peers = self.peers.write();
            let entry = peers
                .entry(peer_id.to_owned())
                .or_insert_with(|| PeerBandwidth::new(peer_id));
            entry.record(bytes, direction, now);
            entry.evict_old(now, self.window);
        }
        // Update global atomics outside the write lock.
        match direction {
            Direction::Inbound => {
                self.stats
                    .total_inbound_bytes
                    .fetch_add(bytes, Ordering::Relaxed);
            }
            Direction::Outbound => {
                self.stats
                    .total_outbound_bytes
                    .fetch_add(bytes, Ordering::Relaxed);
            }
        }
        self.stats.total_samples.fetch_add(1, Ordering::Relaxed);
    }

    /// Return the current rate (bytes/sec) for `peer_id` in `direction`.
    ///
    /// Returns `0.0` if the peer is unknown.
    pub fn rate_for_peer(&self, peer_id: &str, direction: Direction) -> f64 {
        let now = Instant::now();
        let peers = self.peers.read();
        peers
            .get(peer_id)
            .map(|p| p.rate_bps(direction, now, self.window))
            .unwrap_or(0.0)
    }

    /// Return the top `n` peers by **outbound** rate, sorted descending.
    ///
    /// Each element is `(peer_id, rate_bps)`.  If there are fewer than `n`
    /// peers, all known peers are returned.
    pub fn top_senders(&self, n: usize) -> Vec<(String, f64)> {
        let now = Instant::now();
        let peers = self.peers.read();
        let mut rates: Vec<(String, f64)> = peers
            .values()
            .map(|p| {
                (
                    p.peer_id.clone(),
                    p.rate_bps(Direction::Outbound, now, self.window),
                )
            })
            .collect();
        rates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        rates.truncate(n);
        rates
    }

    /// Return the top `n` peers by **inbound** rate, sorted descending.
    ///
    /// Each element is `(peer_id, rate_bps)`.
    pub fn top_receivers(&self, n: usize) -> Vec<(String, f64)> {
        let now = Instant::now();
        let peers = self.peers.read();
        let mut rates: Vec<(String, f64)> = peers
            .values()
            .map(|p| {
                (
                    p.peer_id.clone(),
                    p.rate_bps(Direction::Inbound, now, self.window),
                )
            })
            .collect();
        rates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        rates.truncate(n);
        rates
    }

    /// Compute the aggregate rate (bytes/sec) across all peers for `direction`.
    pub fn total_rate_bps(&self, direction: Direction) -> f64 {
        let now = Instant::now();
        let peers = self.peers.read();
        peers
            .values()
            .map(|p| p.rate_bps(direction, now, self.window))
            .sum()
    }

    /// Remove peers whose most recent sample is older than `max_idle`.
    ///
    /// A peer is considered idle when:
    /// - It has no samples at all, **or**
    /// - Its most recent sample (in either direction) was recorded more than
    ///   `max_idle` ago.
    pub fn evict_idle_peers(&self, max_idle: Duration) {
        let now = Instant::now();
        let mut peers = self.peers.write();
        peers.retain(|_, peer| {
            match peer.last_activity() {
                None => false, // No samples at all — remove immediately.
                Some(ts) => now.duration_since(ts) <= max_idle,
            }
        });
    }

    /// Return the current number of tracked peers.
    pub fn peer_count(&self) -> usize {
        self.peers.read().len()
    }

    /// Take a snapshot of the global atomic stats without acquiring the peer
    /// map lock.
    pub fn stats_snapshot(&self) -> BandwidthStatsSnapshot {
        self.stats.snapshot()
    }
}

impl Default for BandwidthMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    // -----------------------------------------------------------------------
    // Helper: fabricate a PeerBandwidth with injected instants
    // -----------------------------------------------------------------------

    fn make_peer_with_samples(
        peer_id: &str,
        direction: Direction,
        samples: &[(u64, Instant)],
    ) -> PeerBandwidth {
        let mut peer = PeerBandwidth::new(peer_id);
        for (bytes, ts) in samples {
            let sample = BandwidthSample {
                bytes: *bytes,
                timestamp: *ts,
                direction,
            };
            match direction {
                Direction::Inbound => {
                    peer.inbound_samples.push_back(sample);
                    peer.total_inbound_bytes = peer.total_inbound_bytes.saturating_add(*bytes);
                }
                Direction::Outbound => {
                    peer.outbound_samples.push_back(sample);
                    peer.total_outbound_bytes = peer.total_outbound_bytes.saturating_add(*bytes);
                }
            }
        }
        peer
    }

    // -----------------------------------------------------------------------
    // 1. record() increments per-peer totals
    // -----------------------------------------------------------------------
    #[test]
    fn test_record_increments_totals() {
        let monitor = BandwidthMonitor::new();
        monitor.record("peer-a", 100, Direction::Inbound);
        monitor.record("peer-a", 200, Direction::Inbound);
        monitor.record("peer-a", 50, Direction::Outbound);

        let peers = monitor.peers.read();
        let peer = peers.get("peer-a").expect("peer-a should exist");
        assert_eq!(peer.total_inbound_bytes, 300);
        assert_eq!(peer.total_outbound_bytes, 50);
    }

    // -----------------------------------------------------------------------
    // 2. Global stats counters accumulate across peers
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_accumulation() {
        let monitor = BandwidthMonitor::new();
        monitor.record("p1", 1000, Direction::Inbound);
        monitor.record("p2", 2000, Direction::Inbound);
        monitor.record("p1", 500, Direction::Outbound);

        let snap = monitor.stats_snapshot();
        assert_eq!(snap.total_inbound_bytes, 3000);
        assert_eq!(snap.total_outbound_bytes, 500);
        assert_eq!(snap.total_samples, 3);
    }

    // -----------------------------------------------------------------------
    // 3. rate_bps() — known window with time-shifted samples
    // -----------------------------------------------------------------------
    #[test]
    fn test_rate_bps_correct_over_known_window() {
        // 10-second window; inject two samples 5 s apart; together they are
        // both within the 10-second window.  Rate = (100 + 200) / 10 = 30 B/s.
        let window = Duration::from_secs(10);
        let now = Instant::now();
        let t0 = now - Duration::from_secs(8);
        let t1 = now - Duration::from_secs(3);

        let peer = make_peer_with_samples("x", Direction::Inbound, &[(100, t0), (200, t1)]);

        let rate = peer.rate_bps(Direction::Inbound, now, window);
        assert!((rate - 30.0).abs() < 1e-9, "expected 30 B/s, got {rate}");
    }

    // -----------------------------------------------------------------------
    // 4. Old samples are evicted by evict_old()
    // -----------------------------------------------------------------------
    #[test]
    fn test_evict_old_removes_stale_samples() {
        let window = Duration::from_secs(10);
        let now = Instant::now();
        let old = now - Duration::from_secs(15);
        let recent = now - Duration::from_secs(2);

        let mut peer =
            make_peer_with_samples("y", Direction::Inbound, &[(999, old), (100, recent)]);

        assert_eq!(peer.inbound_samples.len(), 2);
        peer.evict_old(now, window);
        assert_eq!(
            peer.inbound_samples.len(),
            1,
            "stale sample should be evicted"
        );
        assert_eq!(peer.inbound_samples.front().map(|s| s.bytes), Some(100));
    }

    // -----------------------------------------------------------------------
    // 5. rate_bps() returns 0 when no samples are in the window
    // -----------------------------------------------------------------------
    #[test]
    fn test_rate_bps_zero_when_no_samples_in_window() {
        let window = Duration::from_secs(5);
        let now = Instant::now();
        let old = now - Duration::from_secs(20);

        let peer = make_peer_with_samples("z", Direction::Outbound, &[(500, old)]);
        let rate = peer.rate_bps(Direction::Outbound, now, window);
        assert_eq!(rate, 0.0);
    }

    // -----------------------------------------------------------------------
    // 6. peak_rate_bps() returns max inter-sample rate
    // -----------------------------------------------------------------------
    #[test]
    fn test_peak_rate_bps() {
        let now = Instant::now();
        // Two samples 1 second apart: 1000 bytes → 1000 B/s
        // Two samples 0.5 s apart:  2000 bytes → 4000 B/s  ← peak
        let t0 = now - Duration::from_millis(1500);
        let t1 = now - Duration::from_millis(500);
        let t2 = now;

        let peer = make_peer_with_samples(
            "peak",
            Direction::Outbound,
            &[(1000, t0), (1000, t1), (2000, t2)],
        );

        let peak = peer.peak_rate_bps(Direction::Outbound);
        // t1 → t2 gap is 500 ms; 2000 / 0.5 = 4000
        assert!(peak >= 3999.0, "expected peak >= 4000 B/s, got {peak}");
    }

    // -----------------------------------------------------------------------
    // 7. top_senders() is sorted descending by outbound rate
    // -----------------------------------------------------------------------
    #[test]
    fn test_top_senders_sorted_descending() {
        let monitor = BandwidthMonitor::with_window(Duration::from_secs(30));
        // Inject samples in a single slice so Instant::now() is used:
        // use small sleeps to ensure distinct instants don't matter — we just
        // need relative ordering within the window, so record directly.
        monitor.record("peer-low", 100, Direction::Outbound);
        monitor.record("peer-mid", 1000, Direction::Outbound);
        monitor.record("peer-high", 5000, Direction::Outbound);

        let top = monitor.top_senders(3);
        assert_eq!(top.len(), 3);
        // Rates are equal over the same window duration so ordering is by
        // absolute bytes — highest first.
        assert!(
            top[0].1 >= top[1].1,
            "first element should have rate >= second"
        );
        assert!(
            top[1].1 >= top[2].1,
            "second element should have rate >= third"
        );
        assert_eq!(top[0].0, "peer-high");
    }

    // -----------------------------------------------------------------------
    // 8. top_receivers() is sorted descending by inbound rate
    // -----------------------------------------------------------------------
    #[test]
    fn test_top_receivers_sorted_descending() {
        let monitor = BandwidthMonitor::with_window(Duration::from_secs(30));
        monitor.record("recv-a", 200, Direction::Inbound);
        monitor.record("recv-b", 800, Direction::Inbound);
        monitor.record("recv-c", 50, Direction::Inbound);

        let top = monitor.top_receivers(3);
        assert_eq!(top.len(), 3);
        assert!(top[0].1 >= top[1].1);
        assert!(top[1].1 >= top[2].1);
        assert_eq!(top[0].0, "recv-b");
    }

    // -----------------------------------------------------------------------
    // 9. top_senders() returns at most n entries
    // -----------------------------------------------------------------------
    #[test]
    fn test_top_senders_truncates_to_n() {
        let monitor = BandwidthMonitor::new();
        for i in 0..10_u64 {
            monitor.record(&format!("p{i}"), i * 100, Direction::Outbound);
        }
        let top = monitor.top_senders(3);
        assert_eq!(top.len(), 3);
    }

    // -----------------------------------------------------------------------
    // 10. total_rate_bps() sums all peer rates
    // -----------------------------------------------------------------------
    #[test]
    fn test_total_rate_bps_sums_all_peers() {
        let window = Duration::from_secs(10);
        let monitor = BandwidthMonitor::with_window(window);

        // All samples recorded at approximately "now", so rate per peer ≈ bytes / 10
        monitor.record("p1", 1000, Direction::Inbound);
        monitor.record("p2", 2000, Direction::Inbound);
        monitor.record("p3", 500, Direction::Inbound);

        let total = monitor.total_rate_bps(Direction::Inbound);
        // Expected: (1000 + 2000 + 500) / 10 = 350 B/s
        assert!(
            (total - 350.0).abs() < 1.0,
            "expected ~350 B/s, got {total}"
        );
    }

    // -----------------------------------------------------------------------
    // 11. evict_idle_peers() removes peers with no recent activity
    // -----------------------------------------------------------------------
    #[test]
    fn test_evict_idle_peers_removes_inactive() {
        let monitor = BandwidthMonitor::new();
        // Record for two peers; then sleep briefly so we can set a very short
        // max_idle below the sleep duration.
        monitor.record("active-peer", 100, Direction::Inbound);
        monitor.record("idle-peer", 100, Direction::Inbound);

        // Wait long enough so "idle-peer" becomes stale for a 1-ms window.
        thread::sleep(Duration::from_millis(5));

        // Record fresh activity for "active-peer" only.
        monitor.record("active-peer", 100, Direction::Inbound);

        // max_idle of 3 ms — "idle-peer" last recorded > 5 ms ago.
        monitor.evict_idle_peers(Duration::from_millis(3));

        let count = monitor.peer_count();
        assert_eq!(count, 1, "only active-peer should remain, got {count}");

        let peers = monitor.peers.read();
        assert!(peers.contains_key("active-peer"));
        assert!(!peers.contains_key("idle-peer"));
    }

    // -----------------------------------------------------------------------
    // 12. peer_count() returns correct count
    // -----------------------------------------------------------------------
    #[test]
    fn test_peer_count_correct() {
        let monitor = BandwidthMonitor::new();
        assert_eq!(monitor.peer_count(), 0);

        monitor.record("a", 1, Direction::Inbound);
        assert_eq!(monitor.peer_count(), 1);

        monitor.record("b", 1, Direction::Outbound);
        assert_eq!(monitor.peer_count(), 2);

        monitor.record("a", 1, Direction::Outbound); // existing peer
        assert_eq!(monitor.peer_count(), 2);
    }

    // -----------------------------------------------------------------------
    // 13. evict_idle_peers() keeps peers with recent activity
    // -----------------------------------------------------------------------
    #[test]
    fn test_evict_idle_peers_retains_active() {
        let monitor = BandwidthMonitor::new();
        monitor.record("fresh", 500, Direction::Inbound);
        // Use a generous max_idle so no peer is evicted.
        monitor.evict_idle_peers(Duration::from_secs(60));
        assert_eq!(monitor.peer_count(), 1);
    }

    // -----------------------------------------------------------------------
    // 14. Direction enum: Inbound and Outbound are distinct
    // -----------------------------------------------------------------------
    #[test]
    fn test_direction_inbound_outbound_independent() {
        let monitor = BandwidthMonitor::new();
        monitor.record("peer-x", 1000, Direction::Inbound);
        monitor.record("peer-x", 500, Direction::Outbound);

        let in_rate = monitor.rate_for_peer("peer-x", Direction::Inbound);
        let out_rate = monitor.rate_for_peer("peer-x", Direction::Outbound);

        // inbound rate should be higher than outbound rate
        assert!(
            in_rate > out_rate,
            "inbound ({in_rate}) should > outbound ({out_rate})"
        );

        // Verify totals are tracked separately
        let peers = monitor.peers.read();
        let p = peers.get("peer-x").expect("peer-x should exist");
        assert_eq!(p.total_inbound_bytes, 1000);
        assert_eq!(p.total_outbound_bytes, 500);
    }

    // -----------------------------------------------------------------------
    // 15. BandwidthStats::snapshot() returns consistent values
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_snapshot_values() {
        let monitor = BandwidthMonitor::new();
        monitor.record("s1", 4096, Direction::Inbound);
        monitor.record("s1", 1024, Direction::Outbound);
        monitor.record("s2", 8192, Direction::Inbound);

        let snap = monitor.stats_snapshot();
        assert_eq!(snap.total_inbound_bytes, 4096 + 8192);
        assert_eq!(snap.total_outbound_bytes, 1024);
        assert_eq!(snap.total_samples, 3);
    }

    // -----------------------------------------------------------------------
    // 16. Unknown peer rate_for_peer() returns 0.0
    // -----------------------------------------------------------------------
    #[test]
    fn test_rate_for_unknown_peer() {
        let monitor = BandwidthMonitor::new();
        assert_eq!(monitor.rate_for_peer("nobody", Direction::Inbound), 0.0);
        assert_eq!(monitor.rate_for_peer("nobody", Direction::Outbound), 0.0);
    }
}

// ---------------------------------------------------------------------------
// PeerBandwidthMonitor — tick-based sliding window bandwidth tracking
// ---------------------------------------------------------------------------

/// A single tick-based bandwidth measurement for a peer.
#[derive(Debug, Clone)]
pub struct TickBandwidthSample {
    /// The logical tick at which this sample was recorded.
    pub tick: u64,
    /// Number of bytes sent during this tick.
    pub bytes_sent: u64,
    /// Number of bytes received during this tick.
    pub bytes_received: u64,
}

/// Sliding-window bandwidth state for a single peer, indexed by logical tick.
#[derive(Debug, Clone)]
pub struct PeerBandwidthWindow {
    /// Stable identifier for this peer.
    pub peer_id: String,
    /// Ordered samples (oldest first); bounded by `window_size`.
    pub samples: Vec<TickBandwidthSample>,
    /// Maximum number of samples to retain.
    pub window_size: usize,
    /// Cumulative bytes sent across all samples ever recorded (not windowed).
    pub total_sent: u64,
    /// Cumulative bytes received across all samples ever recorded (not windowed).
    pub total_received: u64,
}

impl PeerBandwidthWindow {
    /// Create a new, empty window for the given peer with the given capacity.
    pub fn new(peer_id: impl Into<String>, window_size: usize) -> Self {
        Self {
            peer_id: peer_id.into(),
            samples: Vec::new(),
            window_size,
            total_sent: 0,
            total_received: 0,
        }
    }

    /// Append a new sample, evicting the oldest when over capacity.
    ///
    /// Updates cumulative totals before any eviction.
    pub fn add_sample(&mut self, tick: u64, bytes_sent: u64, bytes_received: u64) {
        self.total_sent = self.total_sent.saturating_add(bytes_sent);
        self.total_received = self.total_received.saturating_add(bytes_received);
        self.samples.push(TickBandwidthSample {
            tick,
            bytes_sent,
            bytes_received,
        });
        if self.samples.len() > self.window_size {
            self.samples.remove(0);
        }
    }

    /// Mean `bytes_sent` across all samples currently in the window.
    ///
    /// Returns `0.0` when the window is empty.
    pub fn avg_send_rate(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let total: u64 = self.samples.iter().map(|s| s.bytes_sent).sum();
        total as f64 / self.samples.len() as f64
    }

    /// Mean `bytes_received` across all samples currently in the window.
    ///
    /// Returns `0.0` when the window is empty.
    pub fn avg_recv_rate(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let total: u64 = self.samples.iter().map(|s| s.bytes_received).sum();
        total as f64 / self.samples.len() as f64
    }

    /// Maximum `bytes_sent` of any sample in the current window.
    ///
    /// Returns `0` when the window is empty.
    pub fn peak_send(&self) -> u64 {
        self.samples.iter().map(|s| s.bytes_sent).max().unwrap_or(0)
    }

    /// Maximum `bytes_received` of any sample in the current window.
    ///
    /// Returns `0` when the window is empty.
    pub fn peak_recv(&self) -> u64 {
        self.samples
            .iter()
            .map(|s| s.bytes_received)
            .max()
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// BandwidthAnomaly
// ---------------------------------------------------------------------------

/// An anomaly detected by [`PeerBandwidthMonitor`].
#[derive(Debug, Clone, PartialEq)]
pub enum BandwidthAnomaly {
    /// A single send sample is more than `spike_multiplier` times the prior
    /// rolling average.
    SendSpike {
        /// The peer that triggered the spike.
        peer_id: String,
        /// The raw bytes_sent value that caused the spike.
        sample_bytes: u64,
        /// The rolling average that was exceeded.
        avg_bytes: f64,
    },
    /// A single receive sample is more than `spike_multiplier` times the prior
    /// rolling average.
    RecvSpike {
        /// The peer that triggered the spike.
        peer_id: String,
        /// The raw bytes_received value that caused the spike.
        sample_bytes: u64,
        /// The rolling average that was exceeded.
        avg_bytes: f64,
    },
    /// A peer has produced no samples for at least `idle_threshold_ticks` ticks.
    Idle {
        /// The idle peer.
        peer_id: String,
        /// How many ticks have elapsed since the last recorded sample.
        ticks_since_last: u64,
    },
}

// ---------------------------------------------------------------------------
// MonitorConfig
// ---------------------------------------------------------------------------

/// Configuration for [`PeerBandwidthMonitor`].
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Number of ticks to keep in each peer's sliding window.
    pub window_size: usize,
    /// Threshold multiplier: a sample is a spike when `sample > multiplier * avg`.
    pub spike_multiplier: f64,
    /// Ticks without a sample before a peer is considered idle.
    pub idle_threshold_ticks: u64,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            window_size: 60,
            spike_multiplier: 3.0,
            idle_threshold_ticks: 120,
        }
    }
}

// ---------------------------------------------------------------------------
// BandwidthMonitorStats
// ---------------------------------------------------------------------------

/// Aggregate statistics maintained by [`PeerBandwidthMonitor`].
#[derive(Debug, Clone, Default)]
pub struct BandwidthMonitorStats {
    /// Number of distinct peers currently tracked.
    pub total_peers: usize,
    /// Total number of samples recorded across all peers.
    pub total_samples_recorded: u64,
    /// Total number of anomalies detected (spikes + idles).
    pub total_anomalies_detected: u64,
    /// Aggregate bytes sent across all peers (cumulative).
    pub aggregate_sent_bytes: u64,
    /// Aggregate bytes received across all peers (cumulative).
    pub aggregate_received_bytes: u64,
}

// ---------------------------------------------------------------------------
// PeerBandwidthMonitor
// ---------------------------------------------------------------------------

/// Tracks per-peer and aggregate bandwidth over a sliding tick window.
///
/// Call [`record`](PeerBandwidthMonitor::record) on each tick per peer.
/// Spike anomalies are appended to `pending_anomalies` and can be retrieved
/// with [`drain_anomalies`](PeerBandwidthMonitor::drain_anomalies).
/// Idle anomalies are computed on demand via
/// [`check_idle`](PeerBandwidthMonitor::check_idle).
pub struct PeerBandwidthMonitor {
    /// Per-peer sliding windows, keyed by peer_id.
    pub windows: HashMap<String, PeerBandwidthWindow>,
    /// Monitor configuration.
    pub config: MonitorConfig,
    /// Aggregate statistics.
    pub stats: BandwidthMonitorStats,
    /// Anomalies detected during `record` calls; cleared by `drain_anomalies`.
    pub pending_anomalies: Vec<BandwidthAnomaly>,
}

impl PeerBandwidthMonitor {
    /// Create a new monitor with the given configuration.
    pub fn new(config: MonitorConfig) -> Self {
        Self {
            windows: HashMap::new(),
            config,
            stats: BandwidthMonitorStats::default(),
            pending_anomalies: Vec::new(),
        }
    }

    /// Record a bandwidth sample for `peer_id` at logical `tick`.
    ///
    /// Automatically creates a window entry for new peers.  After recording,
    /// checks for send/receive spikes and appends any detected
    /// [`BandwidthAnomaly`] to `pending_anomalies`.
    pub fn record(&mut self, peer_id: &str, tick: u64, bytes_sent: u64, bytes_recv: u64) {
        let window_size = self.config.window_size;
        let window = self
            .windows
            .entry(peer_id.to_owned())
            .or_insert_with(|| PeerBandwidthWindow::new(peer_id, window_size));

        // Compute prior averages before the new sample is added (needs >= 1
        // existing sample so we have a meaningful baseline).
        let (prior_avg_send, prior_avg_recv, has_prior) = if !window.samples.is_empty() {
            (window.avg_send_rate(), window.avg_recv_rate(), true)
        } else {
            (0.0, 0.0, false)
        };

        window.add_sample(tick, bytes_sent, bytes_recv);

        // Update aggregate stats.
        self.stats.total_samples_recorded = self.stats.total_samples_recorded.saturating_add(1);
        self.stats.aggregate_sent_bytes =
            self.stats.aggregate_sent_bytes.saturating_add(bytes_sent);
        self.stats.aggregate_received_bytes = self
            .stats
            .aggregate_received_bytes
            .saturating_add(bytes_recv);
        self.stats.total_peers = self.windows.len();

        // Spike detection requires at least one prior sample.
        if !has_prior {
            return;
        }

        let multiplier = self.config.spike_multiplier;

        if prior_avg_send > 0.0 && bytes_sent as f64 > multiplier * prior_avg_send {
            self.pending_anomalies.push(BandwidthAnomaly::SendSpike {
                peer_id: peer_id.to_owned(),
                sample_bytes: bytes_sent,
                avg_bytes: prior_avg_send,
            });
            self.stats.total_anomalies_detected =
                self.stats.total_anomalies_detected.saturating_add(1);
        }

        if prior_avg_recv > 0.0 && bytes_recv as f64 > multiplier * prior_avg_recv {
            self.pending_anomalies.push(BandwidthAnomaly::RecvSpike {
                peer_id: peer_id.to_owned(),
                sample_bytes: bytes_recv,
                avg_bytes: prior_avg_recv,
            });
            self.stats.total_anomalies_detected =
                self.stats.total_anomalies_detected.saturating_add(1);
        }
    }

    /// Take all pending anomalies, leaving an empty list.
    pub fn drain_anomalies(&mut self) -> Vec<BandwidthAnomaly> {
        std::mem::take(&mut self.pending_anomalies)
    }

    /// Scan all tracked peers for idle conditions at `current_tick`.
    ///
    /// A peer is idle when it has no samples in its window **or** its most
    /// recent sample tick is more than `idle_threshold_ticks` behind
    /// `current_tick`.
    ///
    /// Unlike spike detection, idle anomalies are **returned directly** and
    /// are NOT added to `pending_anomalies`.
    pub fn check_idle(&mut self, current_tick: u64) -> Vec<BandwidthAnomaly> {
        let threshold = self.config.idle_threshold_ticks;
        let mut anomalies = Vec::new();

        for (peer_id, window) in &self.windows {
            let is_idle = match window.samples.last() {
                None => true,
                Some(last) => {
                    // saturating_sub avoids underflow if current_tick < last.tick
                    let elapsed = current_tick.saturating_sub(last.tick);
                    elapsed >= threshold
                }
            };

            if is_idle {
                let ticks_since_last = match window.samples.last() {
                    None => current_tick,
                    Some(last) => current_tick.saturating_sub(last.tick),
                };
                anomalies.push(BandwidthAnomaly::Idle {
                    peer_id: peer_id.clone(),
                    ticks_since_last,
                });
                self.stats.total_anomalies_detected =
                    self.stats.total_anomalies_detected.saturating_add(1);
            }
        }

        anomalies
    }

    /// Return a shared reference to the sliding window for `peer_id`, if any.
    pub fn peer_window(&self, peer_id: &str) -> Option<&PeerBandwidthWindow> {
        self.windows.get(peer_id)
    }

    /// Return a shared reference to the aggregate statistics.
    pub fn stats(&self) -> &BandwidthMonitorStats {
        &self.stats
    }
}

// ---------------------------------------------------------------------------
// PeerBandwidthMonitor tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod peer_monitor_tests {
    use super::{BandwidthAnomaly, MonitorConfig, PeerBandwidthMonitor, PeerBandwidthWindow};

    // -----------------------------------------------------------------------
    // Helper: build a fresh monitor with default config
    // -----------------------------------------------------------------------
    fn default_monitor() -> PeerBandwidthMonitor {
        PeerBandwidthMonitor::new(MonitorConfig::default())
    }

    // -----------------------------------------------------------------------
    // 1. add_sample — basic insertion
    // -----------------------------------------------------------------------
    #[test]
    fn test_add_sample_basic() {
        let mut w = PeerBandwidthWindow::new("p1", 5);
        w.add_sample(1, 100, 200);
        assert_eq!(w.samples.len(), 1);
        assert_eq!(w.samples[0].tick, 1);
        assert_eq!(w.samples[0].bytes_sent, 100);
        assert_eq!(w.samples[0].bytes_received, 200);
    }

    // -----------------------------------------------------------------------
    // 2. add_sample — sliding window eviction (oldest first)
    // -----------------------------------------------------------------------
    #[test]
    fn test_add_sample_evicts_oldest() {
        let mut w = PeerBandwidthWindow::new("p1", 3);
        w.add_sample(1, 10, 10);
        w.add_sample(2, 20, 20);
        w.add_sample(3, 30, 30);
        // Window is full; adding one more must evict tick=1
        w.add_sample(4, 40, 40);
        assert_eq!(w.samples.len(), 3);
        assert_eq!(
            w.samples[0].tick, 2,
            "oldest (tick=1) should have been evicted"
        );
        assert_eq!(w.samples[2].tick, 4);
    }

    // -----------------------------------------------------------------------
    // 3. add_sample — cumulative totals are not affected by eviction
    // -----------------------------------------------------------------------
    #[test]
    fn test_add_sample_totals_accumulate_beyond_window() {
        let mut w = PeerBandwidthWindow::new("p1", 2);
        w.add_sample(1, 100, 50);
        w.add_sample(2, 200, 100);
        w.add_sample(3, 300, 150); // evicts tick=1
        assert_eq!(w.total_sent, 600);
        assert_eq!(w.total_received, 300);
        // Only the last 2 samples are in the window
        assert_eq!(w.samples.len(), 2);
    }

    // -----------------------------------------------------------------------
    // 4. avg_send_rate — empty window
    // -----------------------------------------------------------------------
    #[test]
    fn test_avg_send_rate_empty() {
        let w = PeerBandwidthWindow::new("p1", 10);
        assert_eq!(w.avg_send_rate(), 0.0);
    }

    // -----------------------------------------------------------------------
    // 5. avg_send_rate — correct mean
    // -----------------------------------------------------------------------
    #[test]
    fn test_avg_send_rate_correct() {
        let mut w = PeerBandwidthWindow::new("p1", 10);
        w.add_sample(1, 100, 0);
        w.add_sample(2, 200, 0);
        w.add_sample(3, 300, 0);
        // mean = (100+200+300)/3 = 200
        assert!((w.avg_send_rate() - 200.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 6. avg_recv_rate — empty window
    // -----------------------------------------------------------------------
    #[test]
    fn test_avg_recv_rate_empty() {
        let w = PeerBandwidthWindow::new("p1", 10);
        assert_eq!(w.avg_recv_rate(), 0.0);
    }

    // -----------------------------------------------------------------------
    // 7. avg_recv_rate — correct mean
    // -----------------------------------------------------------------------
    #[test]
    fn test_avg_recv_rate_correct() {
        let mut w = PeerBandwidthWindow::new("p1", 10);
        w.add_sample(1, 0, 400);
        w.add_sample(2, 0, 600);
        // mean = 500
        assert!((w.avg_recv_rate() - 500.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 8. peak_send — empty window
    // -----------------------------------------------------------------------
    #[test]
    fn test_peak_send_empty() {
        let w = PeerBandwidthWindow::new("p1", 10);
        assert_eq!(w.peak_send(), 0);
    }

    // -----------------------------------------------------------------------
    // 9. peak_send — correct maximum
    // -----------------------------------------------------------------------
    #[test]
    fn test_peak_send_correct() {
        let mut w = PeerBandwidthWindow::new("p1", 10);
        w.add_sample(1, 50, 0);
        w.add_sample(2, 999, 0);
        w.add_sample(3, 100, 0);
        assert_eq!(w.peak_send(), 999);
    }

    // -----------------------------------------------------------------------
    // 10. peak_recv — empty window
    // -----------------------------------------------------------------------
    #[test]
    fn test_peak_recv_empty() {
        let w = PeerBandwidthWindow::new("p1", 10);
        assert_eq!(w.peak_recv(), 0);
    }

    // -----------------------------------------------------------------------
    // 11. peak_recv — correct maximum
    // -----------------------------------------------------------------------
    #[test]
    fn test_peak_recv_correct() {
        let mut w = PeerBandwidthWindow::new("p1", 10);
        w.add_sample(1, 0, 1000);
        w.add_sample(2, 0, 500);
        w.add_sample(3, 0, 2000);
        assert_eq!(w.peak_recv(), 2000);
    }

    // -----------------------------------------------------------------------
    // 12. record() auto-creates window for new peer
    // -----------------------------------------------------------------------
    #[test]
    fn test_record_creates_window() {
        let mut monitor = default_monitor();
        assert!(monitor.peer_window("peer-a").is_none());
        monitor.record("peer-a", 1, 100, 200);
        assert!(monitor.peer_window("peer-a").is_some());
    }

    // -----------------------------------------------------------------------
    // 13. record() updates aggregate stats
    // -----------------------------------------------------------------------
    #[test]
    fn test_record_updates_aggregate_stats() {
        let mut monitor = default_monitor();
        monitor.record("p1", 1, 100, 50);
        monitor.record("p2", 1, 200, 75);

        let s = monitor.stats();
        assert_eq!(s.total_peers, 2);
        assert_eq!(s.total_samples_recorded, 2);
        assert_eq!(s.aggregate_sent_bytes, 300);
        assert_eq!(s.aggregate_received_bytes, 125);
    }

    // -----------------------------------------------------------------------
    // 14. No spike on first sample (only 1 sample — no prior avg)
    // -----------------------------------------------------------------------
    #[test]
    fn test_no_spike_on_first_sample() {
        let mut monitor = default_monitor();
        // Enormous values that would definitely trigger a spike if checked
        monitor.record("p1", 1, u64::MAX / 2, u64::MAX / 2);
        assert!(
            monitor.pending_anomalies.is_empty(),
            "no spike should be detected on the very first sample"
        );
    }

    // -----------------------------------------------------------------------
    // 15. SendSpike detected when sample > 3x prior avg
    // -----------------------------------------------------------------------
    #[test]
    fn test_send_spike_detected() {
        let config = MonitorConfig {
            window_size: 60,
            spike_multiplier: 3.0,
            idle_threshold_ticks: 120,
        };
        let mut monitor = PeerBandwidthMonitor::new(config);
        // Establish baseline of 100 bytes/tick
        monitor.record("peer", 1, 100, 0);
        monitor.record("peer", 2, 100, 0);
        monitor.record("peer", 3, 100, 0);
        // Clear any incidental anomalies from setup
        let _ = monitor.drain_anomalies();

        // Now send 400 bytes — prior avg is 100, 400 > 3 * 100 → spike
        monitor.record("peer", 4, 400, 0);
        let anomalies = monitor.drain_anomalies();
        assert_eq!(anomalies.len(), 1);
        match &anomalies[0] {
            BandwidthAnomaly::SendSpike {
                peer_id,
                sample_bytes,
                ..
            } => {
                assert_eq!(peer_id, "peer");
                assert_eq!(*sample_bytes, 400);
            }
            other => panic!("expected SendSpike, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // 16. RecvSpike detected when sample > 3x prior avg
    // -----------------------------------------------------------------------
    #[test]
    fn test_recv_spike_detected() {
        let mut monitor = default_monitor();
        monitor.record("peer", 1, 0, 100);
        monitor.record("peer", 2, 0, 100);
        monitor.record("peer", 3, 0, 100);
        let _ = monitor.drain_anomalies();

        // 400 bytes recv > 3 * 100
        monitor.record("peer", 4, 0, 400);
        let anomalies = monitor.drain_anomalies();
        assert_eq!(anomalies.len(), 1);
        match &anomalies[0] {
            BandwidthAnomaly::RecvSpike {
                peer_id,
                sample_bytes,
                ..
            } => {
                assert_eq!(peer_id, "peer");
                assert_eq!(*sample_bytes, 400);
            }
            other => panic!("expected RecvSpike, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // 17. Exactly 3x multiplier does NOT trigger a spike (> not >=)
    // -----------------------------------------------------------------------
    #[test]
    fn test_spike_boundary_exactly_3x_no_spike() {
        let config = MonitorConfig {
            window_size: 60,
            spike_multiplier: 3.0,
            idle_threshold_ticks: 120,
        };
        let mut monitor = PeerBandwidthMonitor::new(config);
        // Baseline: avg_send = 100
        monitor.record("peer", 1, 100, 0);
        monitor.record("peer", 2, 100, 0);
        let _ = monitor.drain_anomalies();

        // Exactly 3 * avg = 300 → NOT a spike (must be strictly greater)
        monitor.record("peer", 3, 300, 0);
        let anomalies = monitor.drain_anomalies();
        assert!(
            anomalies.is_empty(),
            "exactly 3x average should NOT trigger a spike, got {:?}",
            anomalies
        );
    }

    // -----------------------------------------------------------------------
    // 18. One tick above 3x multiplier triggers a spike
    // -----------------------------------------------------------------------
    #[test]
    fn test_spike_boundary_above_3x_triggers_spike() {
        let config = MonitorConfig {
            window_size: 60,
            spike_multiplier: 3.0,
            idle_threshold_ticks: 120,
        };
        let mut monitor = PeerBandwidthMonitor::new(config);
        monitor.record("peer", 1, 100, 0);
        monitor.record("peer", 2, 100, 0);
        let _ = monitor.drain_anomalies();

        // 301 > 3 * 100 → spike
        monitor.record("peer", 3, 301, 0);
        let anomalies = monitor.drain_anomalies();
        assert_eq!(anomalies.len(), 1, "301 > 300 should trigger a spike");
    }

    // -----------------------------------------------------------------------
    // 19. drain_anomalies clears the pending list
    // -----------------------------------------------------------------------
    #[test]
    fn test_drain_anomalies_clears_list() {
        let mut monitor = default_monitor();
        monitor.record("peer", 1, 100, 0);
        monitor.record("peer", 2, 100, 0);
        monitor.record("peer", 3, 100, 0);
        // Trigger a spike
        monitor.record("peer", 4, 999, 0);

        let first = monitor.drain_anomalies();
        assert!(!first.is_empty(), "should have anomalies after spike");

        let second = monitor.drain_anomalies();
        assert!(second.is_empty(), "drain_anomalies should clear the list");
    }

    // -----------------------------------------------------------------------
    // 20. check_idle — detects idle peer with no samples
    // -----------------------------------------------------------------------
    #[test]
    fn test_check_idle_no_samples() {
        let config = MonitorConfig {
            window_size: 5,
            spike_multiplier: 3.0,
            idle_threshold_ticks: 10,
        };
        let mut monitor = PeerBandwidthMonitor::new(config);
        // Create a window with no samples by recording then draining via a
        // fresh monitor with a tiny window that evicts everything — simpler:
        // just insert an entry manually.
        monitor.windows.insert(
            "idle-peer".to_owned(),
            PeerBandwidthWindow::new("idle-peer", 5),
        );

        let anomalies = monitor.check_idle(50);
        assert_eq!(anomalies.len(), 1);
        match &anomalies[0] {
            BandwidthAnomaly::Idle {
                peer_id,
                ticks_since_last,
            } => {
                assert_eq!(peer_id, "idle-peer");
                assert_eq!(*ticks_since_last, 50); // current_tick when no samples
            }
            other => panic!("expected Idle, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // 21. check_idle — detects peer whose last sample is old enough
    // -----------------------------------------------------------------------
    #[test]
    fn test_check_idle_stale_last_sample() {
        let config = MonitorConfig {
            window_size: 60,
            spike_multiplier: 3.0,
            idle_threshold_ticks: 10,
        };
        let mut monitor = PeerBandwidthMonitor::new(config);
        monitor.record("peer", 5, 100, 100); // last tick = 5

        // current_tick = 16; elapsed = 11 >= threshold 10 → idle
        let anomalies = monitor.check_idle(16);
        assert_eq!(anomalies.len(), 1);
        match &anomalies[0] {
            BandwidthAnomaly::Idle {
                peer_id,
                ticks_since_last,
            } => {
                assert_eq!(peer_id, "peer");
                assert_eq!(*ticks_since_last, 11);
            }
            other => panic!("expected Idle, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // 22. check_idle — does NOT flag a recently-active peer
    // -----------------------------------------------------------------------
    #[test]
    fn test_check_idle_active_peer_not_flagged() {
        let config = MonitorConfig {
            window_size: 60,
            spike_multiplier: 3.0,
            idle_threshold_ticks: 10,
        };
        let mut monitor = PeerBandwidthMonitor::new(config);
        monitor.record("peer", 95, 100, 100); // last tick = 95

        // current_tick = 100; elapsed = 5 < threshold 10 → not idle
        let anomalies = monitor.check_idle(100);
        assert!(
            anomalies.is_empty(),
            "active peer should not be flagged as idle"
        );
    }

    // -----------------------------------------------------------------------
    // 23. check_idle — exactly at threshold IS idle (>=)
    // -----------------------------------------------------------------------
    #[test]
    fn test_check_idle_exactly_at_threshold() {
        let config = MonitorConfig {
            window_size: 60,
            spike_multiplier: 3.0,
            idle_threshold_ticks: 10,
        };
        let mut monitor = PeerBandwidthMonitor::new(config);
        monitor.record("peer", 90, 100, 100); // last tick = 90

        // elapsed = 10 = threshold → idle
        let anomalies = monitor.check_idle(100);
        assert_eq!(anomalies.len(), 1, "elapsed == threshold should be idle");
    }

    // -----------------------------------------------------------------------
    // 24. check_idle increments stats.total_anomalies_detected
    // -----------------------------------------------------------------------
    #[test]
    fn test_check_idle_increments_anomaly_count() {
        let config = MonitorConfig {
            window_size: 60,
            spike_multiplier: 3.0,
            idle_threshold_ticks: 5,
        };
        let mut monitor = PeerBandwidthMonitor::new(config);
        monitor
            .windows
            .insert("a".to_owned(), PeerBandwidthWindow::new("a", 60));
        monitor
            .windows
            .insert("b".to_owned(), PeerBandwidthWindow::new("b", 60));

        let before = monitor.stats().total_anomalies_detected;
        let anomalies = monitor.check_idle(100);
        let after = monitor.stats().total_anomalies_detected;

        assert_eq!(anomalies.len(), 2);
        assert_eq!(after - before, 2);
    }

    // -----------------------------------------------------------------------
    // 25. check_idle does NOT add to pending_anomalies
    // -----------------------------------------------------------------------
    #[test]
    fn test_check_idle_does_not_populate_pending() {
        let config = MonitorConfig {
            window_size: 60,
            spike_multiplier: 3.0,
            idle_threshold_ticks: 5,
        };
        let mut monitor = PeerBandwidthMonitor::new(config);
        monitor
            .windows
            .insert("idle".to_owned(), PeerBandwidthWindow::new("idle", 60));

        let _ = monitor.check_idle(100);
        assert!(
            monitor.pending_anomalies.is_empty(),
            "check_idle must not populate pending_anomalies"
        );
    }

    // -----------------------------------------------------------------------
    // 26. Both send and recv spikes can fire in the same record() call
    // -----------------------------------------------------------------------
    #[test]
    fn test_both_send_and_recv_spike_same_record() {
        let mut monitor = default_monitor();
        monitor.record("peer", 1, 100, 100);
        monitor.record("peer", 2, 100, 100);
        monitor.record("peer", 3, 100, 100);
        let _ = monitor.drain_anomalies();

        // 500 bytes send AND recv — both > 3 * 100
        monitor.record("peer", 4, 500, 500);
        let anomalies = monitor.drain_anomalies();
        assert_eq!(
            anomalies.len(),
            2,
            "should detect both a SendSpike and RecvSpike"
        );
        let has_send = anomalies
            .iter()
            .any(|a| matches!(a, BandwidthAnomaly::SendSpike { .. }));
        let has_recv = anomalies
            .iter()
            .any(|a| matches!(a, BandwidthAnomaly::RecvSpike { .. }));
        assert!(has_send, "expected SendSpike in anomalies");
        assert!(has_recv, "expected RecvSpike in anomalies");
    }

    // -----------------------------------------------------------------------
    // 27. peer_window() returns None for unknown peer
    // -----------------------------------------------------------------------
    #[test]
    fn test_peer_window_unknown_peer() {
        let monitor = default_monitor();
        assert!(monitor.peer_window("nobody").is_none());
    }

    // -----------------------------------------------------------------------
    // 28. stats() reflects the latest snapshot
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_reflects_current_state() {
        let mut monitor = default_monitor();
        monitor.record("a", 1, 1000, 2000);
        monitor.record("b", 1, 3000, 4000);

        let s = monitor.stats();
        assert_eq!(s.total_peers, 2);
        assert_eq!(s.aggregate_sent_bytes, 4000);
        assert_eq!(s.aggregate_received_bytes, 6000);
        assert_eq!(s.total_samples_recorded, 2);
    }

    // -----------------------------------------------------------------------
    // 29. Spike anomaly carries the correct avg_bytes field
    // -----------------------------------------------------------------------
    #[test]
    fn test_spike_carries_correct_avg_bytes() {
        let mut monitor = default_monitor();
        // avg_send after 2 samples of 100 = 100.0
        monitor.record("peer", 1, 100, 0);
        monitor.record("peer", 2, 100, 0);
        let _ = monitor.drain_anomalies();

        monitor.record("peer", 3, 400, 0);
        let anomalies = monitor.drain_anomalies();
        match &anomalies[0] {
            BandwidthAnomaly::SendSpike { avg_bytes, .. } => {
                assert!(
                    (avg_bytes - 100.0).abs() < 1e-6,
                    "avg_bytes should be 100.0, got {}",
                    avg_bytes
                );
            }
            other => panic!("expected SendSpike, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // 30. MonitorConfig::default() has expected field values
    // -----------------------------------------------------------------------
    #[test]
    fn test_monitor_config_defaults() {
        let cfg = MonitorConfig::default();
        assert_eq!(cfg.window_size, 60);
        assert!((cfg.spike_multiplier - 3.0).abs() < 1e-9);
        assert_eq!(cfg.idle_threshold_ticks, 120);
    }
}
