//! Adaptive Kademlia lookup scheduler that tunes alpha (parallelism) based on observed latency.
//!
//! ## Overview
//!
//! Kademlia's `alpha` parameter controls how many concurrent RPCs are issued per lookup
//! iteration.  The standard value is 3, but a static setting is suboptimal:
//!
//! * A **congested** network benefits from *lower* alpha so that fewer in-flight
//!   requests compete for the same scarce bandwidth.
//! * A **fast** network with low latency can sustain *higher* alpha, reducing total
//!   lookup latency through more parallelism.
//!
//! This module provides:
//!
//! - [`AdaptiveLookupScheduler`] — adjusts alpha in `[1, 8]` based on a rolling
//!   p90 latency window.
//! - [`PeerLatencyTracker`] — per-peer latency tracking with median-based ranking.
//! - [`LookupSchedulerStats`] — snapshot statistics (no atomics, clone-friendly).

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum parallelism degree for Kademlia lookups.
pub const ALPHA_MIN: u32 = 1;
/// Maximum parallelism degree for Kademlia lookups.
pub const ALPHA_MAX: u32 = 8;
/// Default parallelism degree (standard Kademlia value).
pub const ALPHA_DEFAULT: u32 = 3;

/// Sliding-window capacity for the global latency window.
const WINDOW_CAPACITY: usize = 64;
/// Sliding-window capacity per peer in [`PeerLatencyTracker`].
const PEER_WINDOW_CAPACITY: usize = 32;

/// If p90 exceeds this threshold, alpha is decremented.
const HIGH_LATENCY_THRESHOLD_MS: u64 = 500;
/// If p90 is below this threshold, alpha is incremented.
const LOW_LATENCY_THRESHOLD_MS: u64 = 100;

// ---------------------------------------------------------------------------
// LookupSchedulerStats
// ---------------------------------------------------------------------------

/// Point-in-time snapshot of [`AdaptiveLookupScheduler`] state.
///
/// All fields are plain values — no atomics — so the struct is cheaply cloneable
/// and serialisable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LookupSchedulerStats {
    /// Current Kademlia alpha (parallelism degree).
    pub current_alpha: u32,
    /// Number of samples currently in the sliding window.
    pub window_size: usize,
    /// 50th-percentile latency in milliseconds (0 when window is empty).
    pub p50_ms: u64,
    /// 90th-percentile latency in milliseconds (0 when window is empty).
    pub p90_ms: u64,
    /// 99th-percentile latency in milliseconds (0 when window is empty).
    pub p99_ms: u64,
}

// ---------------------------------------------------------------------------
// AdaptiveLookupScheduler
// ---------------------------------------------------------------------------

/// Maintains a rolling latency window and adaptively tunes Kademlia `alpha`.
///
/// # Thread Safety
///
/// The scheduler is designed for concurrent use.  `alpha` uses an `AtomicU32`
/// for lock-free reads while `window` is protected by a `Mutex`.
pub struct AdaptiveLookupScheduler {
    /// Current parallelism degree, always in `[ALPHA_MIN, ALPHA_MAX]`.
    alpha: AtomicU32,
    /// Sliding window of the last [`WINDOW_CAPACITY`] lookup durations.
    window: Mutex<VecDeque<Duration>>,
}

impl std::fmt::Debug for AdaptiveLookupScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptiveLookupScheduler")
            .field("alpha", &self.alpha.load(Ordering::Relaxed))
            .finish()
    }
}

impl Default for AdaptiveLookupScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveLookupScheduler {
    /// Create a new scheduler with `alpha = ALPHA_DEFAULT` and an empty window.
    pub fn new() -> Self {
        Self {
            alpha: AtomicU32::new(ALPHA_DEFAULT),
            window: Mutex::new(VecDeque::with_capacity(WINDOW_CAPACITY)),
        }
    }

    /// Record a single lookup latency sample.
    ///
    /// The sample is appended to the sliding window; the oldest sample is evicted
    /// when the window is full.  After updating the window, `Self::adjust` is
    /// called to possibly update alpha.
    pub fn record_latency(&self, d: Duration) {
        let mut win = self
            .window
            .lock()
            .expect("AdaptiveLookupScheduler window mutex poisoned");
        if win.len() == WINDOW_CAPACITY {
            win.pop_front();
        }
        win.push_back(d);
        // Re-borrow as slice for the p90 computation inside adjust.
        // We compute the percentile inline here to avoid a second lock acquisition.
        let p90 = percentile_duration(&win, 90);
        drop(win);
        self.adjust_with_p90(p90);
    }

    /// Return the current alpha value.
    pub fn current_alpha(&self) -> u32 {
        self.alpha.load(Ordering::Relaxed)
    }

    /// Return a statistics snapshot.
    pub fn stats(&self) -> LookupSchedulerStats {
        let win = self
            .window
            .lock()
            .expect("AdaptiveLookupScheduler window mutex poisoned");
        let window_size = win.len();
        let (p50_ms, p90_ms, p99_ms) = if window_size == 0 {
            (0, 0, 0)
        } else {
            (
                percentile_duration(&win, 50).as_millis() as u64,
                percentile_duration(&win, 90).as_millis() as u64,
                percentile_duration(&win, 99).as_millis() as u64,
            )
        };
        LookupSchedulerStats {
            current_alpha: self.alpha.load(Ordering::Relaxed),
            window_size,
            p50_ms,
            p90_ms,
            p99_ms,
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Core adjustment logic: given a pre-computed p90, nudge alpha up or down.
    fn adjust_with_p90(&self, p90: Duration) {
        let p90_ms = p90.as_millis() as u64;
        // Use a compare-exchange loop so concurrent callers converge correctly.
        let current = self.alpha.load(Ordering::Relaxed);
        if p90_ms > HIGH_LATENCY_THRESHOLD_MS && current > ALPHA_MIN {
            // High latency → reduce parallelism to ease congestion.
            let _ = self.alpha.compare_exchange(
                current,
                current - 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
        } else if p90_ms < LOW_LATENCY_THRESHOLD_MS && current < ALPHA_MAX {
            // Low latency → increase parallelism to exploit spare capacity.
            let _ = self.alpha.compare_exchange(
                current,
                current + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
        }
        // Clamp defensively (should never be needed but protects invariants).
        self.alpha
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.clamp(ALPHA_MIN, ALPHA_MAX))
            })
            .ok();
    }
}

// ---------------------------------------------------------------------------
// PeerLatencyTracker
// ---------------------------------------------------------------------------

/// Per-peer latency record stored in [`PeerLatencyTracker`].
struct PeerRecord {
    /// Rolling window of recent latency samples.
    samples: VecDeque<Duration>,
    /// Timestamp of the most recent sample (for staleness pruning).
    last_seen: Instant,
}

impl PeerRecord {
    fn new() -> Self {
        Self {
            samples: VecDeque::with_capacity(PEER_WINDOW_CAPACITY),
            last_seen: Instant::now(),
        }
    }

    fn push(&mut self, d: Duration) {
        if self.samples.len() == PEER_WINDOW_CAPACITY {
            self.samples.pop_front();
        }
        self.samples.push_back(d);
        self.last_seen = Instant::now();
    }

    /// Return the p90 latency for this peer, or `None` if no samples.
    fn p90(&self) -> Option<Duration> {
        if self.samples.is_empty() {
            None
        } else {
            Some(percentile_duration(&self.samples, 90))
        }
    }

    /// Return the median (p50) latency for this peer, or `None` if no samples.
    fn median(&self) -> Option<Duration> {
        if self.samples.is_empty() {
            None
        } else {
            Some(percentile_duration(&self.samples, 50))
        }
    }
}

/// Tracks per-peer latency with sliding windows and enables peer ranking.
///
/// This is complementary to [`AdaptiveLookupScheduler`]: the scheduler decides
/// *how many* concurrent requests to send, while `PeerLatencyTracker` decides
/// *which peers* to prefer.
pub struct PeerLatencyTracker {
    /// Keyed by peer-ID string representation.
    records: HashMap<String, PeerRecord>,
}

impl std::fmt::Debug for PeerLatencyTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PeerLatencyTracker")
            .field("peer_count", &self.records.len())
            .finish()
    }
}

impl Default for PeerLatencyTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl PeerLatencyTracker {
    /// Create an empty tracker.
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
        }
    }

    /// Record a latency sample for `peer_id`.
    pub fn record(&mut self, peer_id: &str, latency: Duration) {
        self.records
            .entry(peer_id.to_owned())
            .or_insert_with(PeerRecord::new)
            .push(latency);
    }

    /// Return the p90 latency for a specific peer, or `None` if unknown.
    pub fn p90_for_peer(&self, peer_id: &str) -> Option<Duration> {
        self.records.get(peer_id)?.p90()
    }

    /// Return the `n` peer IDs with the lowest median latency.
    ///
    /// Peers with no recorded samples are excluded.  If fewer than `n` peers
    /// have samples, all qualifying peers are returned.
    pub fn fastest_peers(&self, n: usize) -> Vec<String> {
        let mut ranked: Vec<(&String, Duration)> = self
            .records
            .iter()
            .filter_map(|(id, rec)| rec.median().map(|m| (id, m)))
            .collect();

        // Sort ascending by median latency (fastest first).
        ranked.sort_by_key(|&(_, d)| d);

        ranked
            .into_iter()
            .take(n)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Remove peers whose most recent sample is older than `max_age`.
    pub fn prune_stale(&mut self, max_age: Duration) {
        let now = Instant::now();
        self.records
            .retain(|_, rec| now.duration_since(rec.last_seen) <= max_age);
    }

    /// Number of peers currently tracked.
    pub fn peer_count(&self) -> usize {
        self.records.len()
    }
}

// ---------------------------------------------------------------------------
// Internal statistics helpers
// ---------------------------------------------------------------------------

/// Compute the Nth percentile of a `VecDeque<Duration>`.
///
/// Uses the "nearest rank" method.  The deque must be non-empty.
fn percentile_duration(samples: &VecDeque<Duration>, pct: u8) -> Duration {
    debug_assert!(
        !samples.is_empty(),
        "percentile_duration called on empty window"
    );
    debug_assert!(pct <= 100, "percentile must be in [0, 100]");

    let mut sorted: Vec<Duration> = samples.iter().copied().collect();
    sorted.sort_unstable();

    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }

    // Nearest-rank formula: index = ceil(pct / 100 * n) - 1, clamped.
    let index = if pct == 0 {
        0
    } else {
        let raw = (pct as usize * n).div_ceil(100);
        raw.saturating_sub(1).min(n - 1)
    };
    sorted[index]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a scheduler whose window already contains `n` copies of `d`.
    fn scheduler_with_uniform_latency(d: Duration, n: usize) -> AdaptiveLookupScheduler {
        let s = AdaptiveLookupScheduler::new();
        for _ in 0..n {
            s.record_latency(d);
        }
        s
    }

    // -----------------------------------------------------------------------
    // AdaptiveLookupScheduler — alpha adjustment
    // -----------------------------------------------------------------------

    /// If every recorded latency is above the high threshold the alpha should
    /// have been decremented from the default (3) toward the minimum (1).
    #[test]
    fn test_alpha_decrements_on_high_p90() {
        // 600 ms is well above HIGH_LATENCY_THRESHOLD_MS (500 ms).
        let s = scheduler_with_uniform_latency(Duration::from_millis(600), 20);
        assert!(
            s.current_alpha() < ALPHA_DEFAULT,
            "alpha should have decreased, got {}",
            s.current_alpha()
        );
    }

    /// Repeated high-latency samples should drive alpha down to ALPHA_MIN.
    #[test]
    fn test_alpha_reaches_minimum_under_sustained_high_latency() {
        let s = AdaptiveLookupScheduler::new();
        // Feed enough high-latency samples to exhaust all decrement opportunities.
        for _ in 0..20 {
            s.record_latency(Duration::from_millis(800));
        }
        assert_eq!(
            s.current_alpha(),
            ALPHA_MIN,
            "alpha should have saturated at ALPHA_MIN"
        );
    }

    /// Low-latency samples should increment alpha from the default.
    #[test]
    fn test_alpha_increments_on_low_p90() {
        // 50 ms is below LOW_LATENCY_THRESHOLD_MS (100 ms).
        let s = scheduler_with_uniform_latency(Duration::from_millis(50), 20);
        assert!(
            s.current_alpha() > ALPHA_DEFAULT,
            "alpha should have increased, got {}",
            s.current_alpha()
        );
    }

    /// Repeated low-latency samples should drive alpha up to ALPHA_MAX.
    #[test]
    fn test_alpha_reaches_maximum_under_sustained_low_latency() {
        let s = AdaptiveLookupScheduler::new();
        for _ in 0..20 {
            s.record_latency(Duration::from_millis(10));
        }
        assert_eq!(
            s.current_alpha(),
            ALPHA_MAX,
            "alpha should have saturated at ALPHA_MAX"
        );
    }

    /// Alpha must never fall below ALPHA_MIN regardless of input.
    #[test]
    fn test_alpha_never_below_minimum() {
        let s = AdaptiveLookupScheduler::new();
        for _ in 0..100 {
            s.record_latency(Duration::from_secs(10));
        }
        assert!(s.current_alpha() >= ALPHA_MIN);
    }

    /// Alpha must never exceed ALPHA_MAX regardless of input.
    #[test]
    fn test_alpha_never_above_maximum() {
        let s = AdaptiveLookupScheduler::new();
        for _ in 0..100 {
            s.record_latency(Duration::from_micros(1));
        }
        assert!(s.current_alpha() <= ALPHA_MAX);
    }

    /// Mid-range latency (between the two thresholds) should leave alpha stable.
    #[test]
    fn test_alpha_stable_on_mid_range_latency() {
        // 250 ms is between 100 ms and 500 ms.
        let s = scheduler_with_uniform_latency(Duration::from_millis(250), 30);
        assert_eq!(
            s.current_alpha(),
            ALPHA_DEFAULT,
            "alpha should remain at default for mid-range latency"
        );
    }

    /// The sliding window is capped at WINDOW_CAPACITY; old samples are evicted.
    #[test]
    fn test_window_capacity_is_bounded() {
        let s = AdaptiveLookupScheduler::new();
        // Insert more than WINDOW_CAPACITY samples.
        for _ in 0..WINDOW_CAPACITY + 10 {
            s.record_latency(Duration::from_millis(200));
        }
        let stats = s.stats();
        assert_eq!(
            stats.window_size, WINDOW_CAPACITY,
            "window should be capped at WINDOW_CAPACITY"
        );
    }

    /// stats() on an empty scheduler should return sensible zeros.
    #[test]
    fn test_stats_on_empty_scheduler() {
        let s = AdaptiveLookupScheduler::new();
        let stats = s.stats();
        assert_eq!(stats.current_alpha, ALPHA_DEFAULT);
        assert_eq!(stats.window_size, 0);
        assert_eq!(stats.p50_ms, 0);
        assert_eq!(stats.p90_ms, 0);
        assert_eq!(stats.p99_ms, 0);
    }

    /// stats() should reflect the recorded latencies.
    #[test]
    fn test_stats_reflect_recorded_latencies() {
        let s = AdaptiveLookupScheduler::new();
        // Insert 10 samples: 100 ms, 200 ms, …, 1000 ms.
        for i in 1..=10u64 {
            s.record_latency(Duration::from_millis(i * 100));
        }
        let stats = s.stats();
        assert_eq!(stats.window_size, 10);
        // p50 of {100,200,…,1000} → 500 ms (nearest rank index 4 of 10).
        assert!(
            stats.p50_ms >= 400 && stats.p50_ms <= 600,
            "unexpected p50: {}",
            stats.p50_ms
        );
        // p90 → should be around 900 ms.
        assert!(
            stats.p90_ms >= 800 && stats.p90_ms <= 1000,
            "unexpected p90: {}",
            stats.p90_ms
        );
    }

    // -----------------------------------------------------------------------
    // PeerLatencyTracker
    // -----------------------------------------------------------------------

    /// Record latency for a single peer and verify p90 retrieval.
    #[test]
    fn test_peer_p90_single_peer() {
        let mut tracker = PeerLatencyTracker::new();
        for i in 1..=10u64 {
            tracker.record("peer-A", Duration::from_millis(i * 100));
        }
        let p90 = tracker
            .p90_for_peer("peer-A")
            .expect("peer-A should be tracked");
        assert!(
            p90.as_millis() >= 800 && p90.as_millis() <= 1000,
            "unexpected p90: {}",
            p90.as_millis()
        );
    }

    /// Unknown peer returns None.
    #[test]
    fn test_peer_p90_unknown_peer_returns_none() {
        let tracker = PeerLatencyTracker::new();
        assert!(tracker.p90_for_peer("ghost-peer").is_none());
    }

    /// `fastest_peers` returns peers ordered by ascending median latency.
    #[test]
    fn test_fastest_peers_ordering() {
        let mut tracker = PeerLatencyTracker::new();

        // peer-A: consistently fast (~10 ms).
        for _ in 0..5 {
            tracker.record("peer-A", Duration::from_millis(10));
        }
        // peer-B: medium (~200 ms).
        for _ in 0..5 {
            tracker.record("peer-B", Duration::from_millis(200));
        }
        // peer-C: slow (~800 ms).
        for _ in 0..5 {
            tracker.record("peer-C", Duration::from_millis(800));
        }

        let fastest = tracker.fastest_peers(3);
        assert_eq!(fastest.len(), 3);
        assert_eq!(fastest[0], "peer-A", "peer-A should be fastest");
        assert_eq!(fastest[2], "peer-C", "peer-C should be slowest");
    }

    /// `fastest_peers(n)` with n > tracked peers returns all tracked peers.
    #[test]
    fn test_fastest_peers_fewer_than_requested() {
        let mut tracker = PeerLatencyTracker::new();
        tracker.record("only-peer", Duration::from_millis(50));
        let result = tracker.fastest_peers(10);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "only-peer");
    }

    /// `prune_stale` with a zero-duration max_age should remove all records.
    #[test]
    fn test_prune_stale_removes_all_with_zero_max_age() {
        let mut tracker = PeerLatencyTracker::new();
        tracker.record("peer-X", Duration::from_millis(100));
        tracker.record("peer-Y", Duration::from_millis(200));

        // Sleep briefly so that last_seen is in the past, then prune with 0 max_age.
        // We use Duration::ZERO which means any record older than "now" is pruned.
        // Since Instant::now() at push time is strictly before now at prune time,
        // a max_age of Duration::ZERO should remove all entries.
        // (On very fast machines the delta could be 0 ns; use a tiny positive guard.)
        std::thread::sleep(Duration::from_millis(1));
        tracker.prune_stale(Duration::ZERO);
        assert_eq!(tracker.peer_count(), 0, "all peers should have been pruned");
    }

    /// `prune_stale` with a generous max_age retains recent records.
    #[test]
    fn test_prune_stale_retains_recent_entries() {
        let mut tracker = PeerLatencyTracker::new();
        tracker.record("peer-A", Duration::from_millis(50));
        tracker.record("peer-B", Duration::from_millis(100));

        // Prune with 1-hour max_age — nothing should be removed.
        tracker.prune_stale(Duration::from_secs(3600));
        assert_eq!(
            tracker.peer_count(),
            2,
            "no peers should have been pruned within max_age"
        );
    }

    /// Per-peer window is capped at PEER_WINDOW_CAPACITY.
    #[test]
    fn test_peer_window_capacity_is_bounded() {
        let mut tracker = PeerLatencyTracker::new();
        // Insert twice the capacity.
        for i in 0..(PEER_WINDOW_CAPACITY * 2) as u64 {
            tracker.record("peer-A", Duration::from_millis(i));
        }
        // If p90 is reachable the window wasn't corrupted; also verify by
        // ensuring the oldest samples (very small durations) were evicted.
        let p90 = tracker.p90_for_peer("peer-A").expect("should exist");
        // The last PEER_WINDOW_CAPACITY values start at index PEER_WINDOW_CAPACITY
        // (i.e., 32 ms, 33 ms, …, 63 ms).  p90 should be ≥ the majority of those.
        assert!(
            p90.as_millis() >= 50,
            "old (small) samples should have been evicted; p90={}ms",
            p90.as_millis()
        );
    }
}
