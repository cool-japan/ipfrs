//! Adaptive Kademlia refresh under peer churn.
//!
//! This module implements a `ChurnResilienceManager` that tracks peer
//! join/leave events in a rolling time window and adjusts Kademlia routing
//! table refresh intervals based on observed churn rate.  High churn →
//! shorter refresh interval; stable network → longer interval.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A peer join/leave event.
#[derive(Debug, Clone)]
pub struct PeerChurnEvent {
    /// Peer identifier (e.g. PeerId string).
    pub peer_id: String,
    /// Kind of churn event.
    pub event_type: ChurnEventType,
    /// Wall-clock timestamp in milliseconds since epoch.
    pub timestamp_ms: u64,
}

/// Discriminator for churn event direction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChurnEventType {
    /// Peer became reachable / connected.
    Joined,
    /// Peer disconnected gracefully.
    Left,
    /// Peer missed keep-alives and was evicted.
    TimedOut,
}

/// Rolling-window churn metrics snapshot.
#[derive(Debug, Clone)]
pub struct ChurnMetrics {
    /// Observation window length in milliseconds.
    pub window_ms: u64,
    /// Peers that joined within the window.
    pub joins: usize,
    /// Peers that left within the window.
    pub leaves: usize,
    /// `(joins + leaves) / window_seconds`
    pub churn_rate: f64,
    /// `1.0 / (1.0 + churn_rate)` — 1.0 = perfectly stable, 0.0 = max churn.
    pub stability_score: f64,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for adaptive refresh scheduling.
#[derive(Debug, Clone)]
pub struct AdaptiveRefreshConfig {
    /// Base refresh interval when stable.  Default: 10 minutes.
    pub base_interval_ms: u64,
    /// Minimum refresh interval under extreme churn.  Default: 1 minute.
    pub min_interval_ms: u64,
    /// Maximum refresh interval when very stable.  Default: 30 minutes.
    pub max_interval_ms: u64,
    /// Churn-rate threshold above which we start shrinking the interval.
    /// Default: 2.0 events per second.
    pub high_churn_threshold: f64,
    /// Rolling window size for churn measurement in milliseconds.
    /// Default: 60 seconds.
    pub window_ms: u64,
    /// Maximum events to retain in rolling window (memory guard).
    pub max_events: usize,
}

impl Default for AdaptiveRefreshConfig {
    fn default() -> Self {
        Self {
            base_interval_ms: 10 * 60 * 1_000, // 10 min
            min_interval_ms: 60 * 1_000,       // 1 min
            max_interval_ms: 30 * 60 * 1_000,  // 30 min
            high_churn_threshold: 2.0,
            window_ms: 60 * 1_000, // 60 s
            max_events: 10_000,
        }
    }
}

// ---------------------------------------------------------------------------
// PeerChurnTracker
// ---------------------------------------------------------------------------

/// Tracks peer join/leave events and computes adaptive refresh intervals.
pub struct PeerChurnTracker {
    events: RwLock<VecDeque<PeerChurnEvent>>,
    config: AdaptiveRefreshConfig,
    total_joins: AtomicU64,
    total_leaves: AtomicU64,
}

impl PeerChurnTracker {
    /// Create a new tracker with the given configuration.
    pub fn new(config: AdaptiveRefreshConfig) -> Self {
        Self {
            events: RwLock::new(VecDeque::new()),
            config,
            total_joins: AtomicU64::new(0),
            total_leaves: AtomicU64::new(0),
        }
    }

    /// Record a peer join/leave event.
    pub fn record_event(&self, event: PeerChurnEvent) {
        match event.event_type {
            ChurnEventType::Joined => {
                self.total_joins.fetch_add(1, Ordering::Relaxed);
            }
            ChurnEventType::Left | ChurnEventType::TimedOut => {
                self.total_leaves.fetch_add(1, Ordering::Relaxed);
            }
        }

        let mut guard = self.events.write();
        guard.push_back(event);

        // Enforce hard cap so memory stays bounded even without periodic eviction.
        while guard.len() > self.config.max_events {
            guard.pop_front();
        }
    }

    /// Evict events older than `now_ms - window_ms`.
    ///
    /// Returns the number of events evicted.
    pub fn evict_old_events(&self, now_ms: u64) -> usize {
        let cutoff = now_ms.saturating_sub(self.config.window_ms);
        let mut guard = self.events.write();
        let before = guard.len();
        while let Some(front) = guard.front() {
            if front.timestamp_ms < cutoff {
                guard.pop_front();
            } else {
                break;
            }
        }
        before - guard.len()
    }

    /// Compute churn metrics for the current rolling window.
    pub fn metrics(&self, now_ms: u64) -> ChurnMetrics {
        let cutoff = now_ms.saturating_sub(self.config.window_ms);
        let guard = self.events.read();

        let mut joins = 0usize;
        let mut leaves = 0usize;

        for ev in guard.iter() {
            if ev.timestamp_ms >= cutoff {
                match ev.event_type {
                    ChurnEventType::Joined => joins += 1,
                    ChurnEventType::Left | ChurnEventType::TimedOut => leaves += 1,
                }
            }
        }

        let window_secs = self.config.window_ms as f64 / 1_000.0;
        let total_events = (joins + leaves) as f64;
        let churn_rate = if window_secs > 0.0 {
            total_events / window_secs
        } else {
            0.0
        };
        let stability_score = 1.0 / (1.0 + churn_rate);

        ChurnMetrics {
            window_ms: self.config.window_ms,
            joins,
            leaves,
            churn_rate,
            stability_score,
        }
    }

    /// Compute recommended refresh interval based on current churn.
    ///
    /// Algorithm:
    /// 1. If `churn_rate >= high_churn_threshold`:
    ///    linearly interpolate from `base_interval_ms` down to `min_interval_ms`
    ///    as `churn_rate` goes from threshold to `2 * threshold`.  Clamped to
    ///    `min_interval_ms`.
    /// 2. If `churn_rate < 0.1` (very stable):
    ///    linearly interpolate from `base_interval_ms` up to `max_interval_ms`
    ///    based on how far below 0.1 the rate is.
    /// 3. Otherwise: `base_interval_ms`.
    pub fn recommended_interval(&self, now_ms: u64) -> Duration {
        let m = self.metrics(now_ms);
        let rate = m.churn_rate;
        let cfg = &self.config;

        let interval_ms = if rate >= cfg.high_churn_threshold {
            // Linear scale from base→min as rate goes threshold→2*threshold.
            let t = ((rate - cfg.high_churn_threshold) / cfg.high_churn_threshold).clamp(0.0, 1.0);
            let base = cfg.base_interval_ms as f64;
            let min = cfg.min_interval_ms as f64;
            let v = base + t * (min - base);
            (v as u64).max(cfg.min_interval_ms)
        } else if rate < 0.1 {
            // Linear scale from base→max as rate goes from 0.1 down to 0.
            let t = (1.0 - rate / 0.1).clamp(0.0, 1.0);
            let base = cfg.base_interval_ms as f64;
            let max = cfg.max_interval_ms as f64;
            let v = base + t * (max - base);
            (v as u64).min(cfg.max_interval_ms)
        } else {
            cfg.base_interval_ms
        };

        Duration::from_millis(interval_ms)
    }

    /// Number of events currently inside the rolling window.
    pub fn window_event_count(&self, now_ms: u64) -> usize {
        let cutoff = now_ms.saturating_sub(self.config.window_ms);
        let guard = self.events.read();
        guard.iter().filter(|e| e.timestamp_ms >= cutoff).count()
    }

    /// Total join events observed since tracker creation.
    pub fn total_joins(&self) -> u64 {
        self.total_joins.load(Ordering::Relaxed)
    }

    /// Total leave/timeout events observed since tracker creation.
    pub fn total_leaves(&self) -> u64 {
        self.total_leaves.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// AdaptiveRefreshScheduler
// ---------------------------------------------------------------------------

/// Coordinates Kademlia routing-table refresh using adaptive intervals.
pub struct AdaptiveRefreshScheduler {
    tracker: Arc<PeerChurnTracker>,
    last_refresh_ms: AtomicU64,
    refresh_count: AtomicU64,
}

impl AdaptiveRefreshScheduler {
    /// Sentinel value stored in `last_refresh_ms` when no refresh has occurred.
    const NEVER_REFRESHED: u64 = u64::MAX;

    /// Create a new scheduler wrapped in an `Arc`.
    pub fn new(config: AdaptiveRefreshConfig) -> Arc<Self> {
        Arc::new(Self {
            tracker: Arc::new(PeerChurnTracker::new(config)),
            last_refresh_ms: AtomicU64::new(Self::NEVER_REFRESHED),
            refresh_count: AtomicU64::new(0),
        })
    }

    /// Record a peer event (delegates to the underlying tracker).
    pub fn record_peer_event(&self, event: PeerChurnEvent) {
        self.tracker.record_event(event);
    }

    /// Returns `true` when the current adaptive interval has elapsed since the
    /// last recorded refresh.
    pub fn is_refresh_due(&self, now_ms: u64) -> bool {
        let last = self.last_refresh_ms.load(Ordering::Relaxed);
        if last == Self::NEVER_REFRESHED {
            // Never refreshed — immediately due.
            return true;
        }
        let elapsed_ms = now_ms.saturating_sub(last);
        let interval_ms = self.tracker.recommended_interval(now_ms).as_millis() as u64;
        elapsed_ms >= interval_ms
    }

    /// Record that a refresh was performed at `now_ms`.
    pub fn mark_refreshed(&self, now_ms: u64) {
        self.last_refresh_ms.store(now_ms, Ordering::Relaxed);
        self.refresh_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Current recommended refresh interval.
    pub fn current_interval(&self, now_ms: u64) -> Duration {
        self.tracker.recommended_interval(now_ms)
    }

    /// Duration remaining until the next scheduled refresh.
    ///
    /// Returns `Duration::ZERO` if a refresh is already due.
    pub fn time_until_refresh(&self, now_ms: u64) -> Duration {
        let last = self.last_refresh_ms.load(Ordering::Relaxed);
        if last == Self::NEVER_REFRESHED {
            return Duration::ZERO;
        }
        let interval_ms = self.tracker.recommended_interval(now_ms).as_millis() as u64;
        let elapsed_ms = now_ms.saturating_sub(last);
        if elapsed_ms >= interval_ms {
            Duration::ZERO
        } else {
            Duration::from_millis(interval_ms - elapsed_ms)
        }
    }

    /// Total number of refreshes performed.
    pub fn refresh_count(&self) -> u64 {
        self.refresh_count.load(Ordering::Relaxed)
    }

    /// Current churn metrics (delegates to tracker).
    pub fn churn_metrics(&self, now_ms: u64) -> ChurnMetrics {
        self.tracker.metrics(now_ms)
    }
}

// ---------------------------------------------------------------------------
// ChurnResilienceManager
// ---------------------------------------------------------------------------

/// Full manager combining the churn tracker and refresh scheduler.
pub struct ChurnResilienceManager {
    /// The underlying adaptive refresh scheduler.
    pub scheduler: Arc<AdaptiveRefreshScheduler>,
}

impl ChurnResilienceManager {
    /// Create a new manager wrapped in an `Arc`.
    pub fn new(config: AdaptiveRefreshConfig) -> Arc<Self> {
        Arc::new(Self {
            scheduler: AdaptiveRefreshScheduler::new(config),
        })
    }

    /// Record that a peer joined at `now_ms`.
    pub fn peer_joined(&self, peer_id: impl Into<String>, now_ms: u64) {
        self.scheduler.record_peer_event(PeerChurnEvent {
            peer_id: peer_id.into(),
            event_type: ChurnEventType::Joined,
            timestamp_ms: now_ms,
        });
    }

    /// Record that a peer left gracefully at `now_ms`.
    pub fn peer_left(&self, peer_id: impl Into<String>, now_ms: u64) {
        self.scheduler.record_peer_event(PeerChurnEvent {
            peer_id: peer_id.into(),
            event_type: ChurnEventType::Left,
            timestamp_ms: now_ms,
        });
    }

    /// Record that a peer timed out at `now_ms`.
    pub fn peer_timed_out(&self, peer_id: impl Into<String>, now_ms: u64) {
        self.scheduler.record_peer_event(PeerChurnEvent {
            peer_id: peer_id.into(),
            event_type: ChurnEventType::TimedOut,
            timestamp_ms: now_ms,
        });
    }

    /// Returns `true` if a routing-table refresh is now due.
    pub fn should_refresh(&self, now_ms: u64) -> bool {
        self.scheduler.is_refresh_due(now_ms)
    }

    /// Mark that a refresh was performed at `now_ms`.
    pub fn mark_refreshed(&self, now_ms: u64) {
        self.scheduler.mark_refreshed(now_ms);
    }

    /// Current churn metrics.
    pub fn metrics(&self, now_ms: u64) -> ChurnMetrics {
        self.scheduler.churn_metrics(now_ms)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> AdaptiveRefreshConfig {
        AdaptiveRefreshConfig::default()
    }

    /// Helper: millisecond timestamp.
    fn ts(secs: u64) -> u64 {
        secs * 1_000
    }

    // ------------------------------------------------------------------
    // PeerChurnTracker tests
    // ------------------------------------------------------------------

    #[test]
    fn test_record_event_increments_totals() {
        let tracker = PeerChurnTracker::new(default_config());
        tracker.record_event(PeerChurnEvent {
            peer_id: "p1".into(),
            event_type: ChurnEventType::Joined,
            timestamp_ms: ts(10),
        });
        tracker.record_event(PeerChurnEvent {
            peer_id: "p2".into(),
            event_type: ChurnEventType::Left,
            timestamp_ms: ts(11),
        });
        tracker.record_event(PeerChurnEvent {
            peer_id: "p3".into(),
            event_type: ChurnEventType::TimedOut,
            timestamp_ms: ts(12),
        });
        assert_eq!(tracker.total_joins(), 1);
        assert_eq!(tracker.total_leaves(), 2);
    }

    #[test]
    fn test_evict_old_events() {
        let cfg = AdaptiveRefreshConfig {
            window_ms: 60_000,
            ..default_config()
        };
        let tracker = PeerChurnTracker::new(cfg);
        // Two events at t=0 (well outside window), one at t=120s (inside window at t=180s).
        tracker.record_event(PeerChurnEvent {
            peer_id: "a".into(),
            event_type: ChurnEventType::Joined,
            timestamp_ms: 0,
        });
        tracker.record_event(PeerChurnEvent {
            peer_id: "b".into(),
            event_type: ChurnEventType::Left,
            timestamp_ms: 0,
        });
        tracker.record_event(PeerChurnEvent {
            peer_id: "c".into(),
            event_type: ChurnEventType::Joined,
            timestamp_ms: ts(120),
        });

        // At now=180s the cutoff is 120s, so "a" and "b" (ts=0) are evicted.
        let evicted = tracker.evict_old_events(ts(180));
        assert_eq!(evicted, 2);
        assert_eq!(tracker.window_event_count(ts(180)), 1);
    }

    #[test]
    fn test_metrics_empty_window() {
        let tracker = PeerChurnTracker::new(default_config());
        let m = tracker.metrics(ts(100));
        assert_eq!(m.joins, 0);
        assert_eq!(m.leaves, 0);
        assert!((m.churn_rate - 0.0).abs() < f64::EPSILON);
        // stability = 1 / (1 + 0) = 1.0
        assert!((m.stability_score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_metrics_with_events() {
        let cfg = AdaptiveRefreshConfig {
            window_ms: 60_000,
            ..default_config()
        };
        let tracker = PeerChurnTracker::new(cfg);

        // 10 joins and 5 leaves, all within the window.
        for i in 0..10u64 {
            tracker.record_event(PeerChurnEvent {
                peer_id: format!("join-{i}"),
                event_type: ChurnEventType::Joined,
                timestamp_ms: ts(50) + i,
            });
        }
        for i in 0..5u64 {
            tracker.record_event(PeerChurnEvent {
                peer_id: format!("left-{i}"),
                event_type: ChurnEventType::Left,
                timestamp_ms: ts(55) + i,
            });
        }

        let m = tracker.metrics(ts(100));
        assert_eq!(m.joins, 10);
        assert_eq!(m.leaves, 5);
        // churn_rate = 15 / 60 = 0.25
        assert!((m.churn_rate - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_stability_score_stable() {
        let tracker = PeerChurnTracker::new(default_config());
        // Only 1 event in the last 60 s → very low churn.
        tracker.record_event(PeerChurnEvent {
            peer_id: "x".into(),
            event_type: ChurnEventType::Joined,
            timestamp_ms: ts(99),
        });
        let m = tracker.metrics(ts(100));
        // score should be close to 1.0
        assert!(
            m.stability_score > 0.9,
            "expected score > 0.9, got {}",
            m.stability_score
        );
    }

    #[test]
    fn test_stability_score_high_churn() {
        let cfg = AdaptiveRefreshConfig {
            window_ms: 60_000,
            ..default_config()
        };
        let tracker = PeerChurnTracker::new(cfg);
        // Add many events → churn_rate should be high.
        for i in 0..300u64 {
            tracker.record_event(PeerChurnEvent {
                peer_id: format!("p{i}"),
                event_type: if i % 2 == 0 {
                    ChurnEventType::Joined
                } else {
                    ChurnEventType::Left
                },
                timestamp_ms: ts(50) + i,
            });
        }
        let m = tracker.metrics(ts(100));
        // stability = 1/(1+5) = 0.167 → definitely < 0.5
        assert!(
            m.stability_score < 0.5,
            "expected score < 0.5, got {}",
            m.stability_score
        );
    }

    #[test]
    fn test_recommended_interval_stable() {
        let cfg = AdaptiveRefreshConfig {
            window_ms: 60_000,
            base_interval_ms: 600_000,
            max_interval_ms: 1_800_000,
            ..default_config()
        };
        let tracker = PeerChurnTracker::new(cfg.clone());
        // No events → churn_rate = 0 → interval should be max.
        let interval = tracker.recommended_interval(ts(100));
        assert!(
            interval.as_millis() as u64 >= cfg.base_interval_ms,
            "expected >= base, got {:?}",
            interval
        );
    }

    #[test]
    fn test_recommended_interval_high_churn() {
        let cfg = AdaptiveRefreshConfig {
            window_ms: 10_000, // 10 s window for easy math
            high_churn_threshold: 2.0,
            base_interval_ms: 600_000,
            min_interval_ms: 60_000,
            max_interval_ms: 1_800_000,
            max_events: 10_000,
        };
        let tracker = PeerChurnTracker::new(cfg.clone());

        // Inject events that produce churn_rate >> threshold.
        // 400 events in 10 s → rate = 40/s (well above 2*threshold=4).
        for i in 0..400u64 {
            tracker.record_event(PeerChurnEvent {
                peer_id: format!("p{i}"),
                event_type: ChurnEventType::Left,
                timestamp_ms: ts(90) + i * 25, // spread within 10 s
            });
        }

        let interval = tracker.recommended_interval(ts(100));
        assert_eq!(
            interval,
            Duration::from_millis(cfg.min_interval_ms),
            "expected min interval under extreme churn, got {:?}",
            interval
        );
    }

    // ------------------------------------------------------------------
    // AdaptiveRefreshScheduler tests
    // ------------------------------------------------------------------

    #[test]
    fn test_scheduler_refresh_due() {
        let cfg = AdaptiveRefreshConfig {
            base_interval_ms: 1_000, // 1 s base
            min_interval_ms: 500,
            max_interval_ms: 5_000,
            ..default_config()
        };
        let scheduler = AdaptiveRefreshScheduler::new(cfg);
        // Mark refresh at t=0.
        scheduler.mark_refreshed(0);
        // Before interval elapses.
        assert!(!scheduler.is_refresh_due(500));
        // After interval elapses.
        assert!(scheduler.is_refresh_due(10_000));
    }

    #[test]
    fn test_scheduler_not_due_after_refresh() {
        let cfg = AdaptiveRefreshConfig {
            base_interval_ms: 60_000,
            min_interval_ms: 10_000,
            max_interval_ms: 120_000,
            ..default_config()
        };
        let scheduler = AdaptiveRefreshScheduler::new(cfg);
        scheduler.mark_refreshed(ts(100));
        // Immediately after refresh, should not be due.
        assert!(!scheduler.is_refresh_due(ts(100) + 1));
    }

    #[test]
    fn test_scheduler_refresh_count() {
        let scheduler = AdaptiveRefreshScheduler::new(default_config());
        assert_eq!(scheduler.refresh_count(), 0);
        scheduler.mark_refreshed(ts(1));
        scheduler.mark_refreshed(ts(20));
        scheduler.mark_refreshed(ts(50));
        assert_eq!(scheduler.refresh_count(), 3);
    }

    // ------------------------------------------------------------------
    // ChurnResilienceManager tests
    // ------------------------------------------------------------------

    #[test]
    fn test_churn_manager_peer_joined_left() {
        let manager = ChurnResilienceManager::new(default_config());
        manager.peer_joined("peer-1", ts(10));
        manager.peer_left("peer-2", ts(11));
        manager.peer_timed_out("peer-3", ts(12));

        // Totals are maintained by the tracker.
        let tracker = &manager.scheduler.tracker;
        assert_eq!(tracker.total_joins(), 1);
        assert_eq!(tracker.total_leaves(), 2);
    }

    #[test]
    fn test_churn_manager_should_refresh() {
        let cfg = AdaptiveRefreshConfig {
            base_interval_ms: 1_000,
            min_interval_ms: 200,
            max_interval_ms: 5_000,
            ..default_config()
        };
        let manager = ChurnResilienceManager::new(cfg);

        // First call (never refreshed) → should refresh immediately.
        assert!(manager.should_refresh(ts(0)));

        // Mark refresh at t=0.
        manager.mark_refreshed(ts(0));

        // Should not be due immediately after.
        assert!(!manager.should_refresh(ts(0) + 100));

        // Should be due after enough time has elapsed.
        assert!(manager.should_refresh(ts(0) + 10_000));
    }

    // ------------------------------------------------------------------
    // Default config test
    // ------------------------------------------------------------------

    #[test]
    fn test_default_config_values() {
        let cfg = AdaptiveRefreshConfig::default();
        assert_eq!(cfg.base_interval_ms, 10 * 60 * 1_000);
        assert_eq!(cfg.min_interval_ms, 60 * 1_000);
        assert_eq!(cfg.max_interval_ms, 30 * 60 * 1_000);
        assert!((cfg.high_churn_threshold - 2.0).abs() < f64::EPSILON);
        assert_eq!(cfg.window_ms, 60 * 1_000);
        assert_eq!(cfg.max_events, 10_000);
    }
}
