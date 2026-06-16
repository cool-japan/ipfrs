//! Sled compaction scheduling with lock-free atomics.
//!
//! [`CompactionScheduler`] tracks write activity and decides when to trigger a
//! Sled WAL flush / compaction.  All state is held in atomics so the scheduler
//! is cheaply shareable across threads without any mutex.
//!
//! # Decision logic
//!
//! A compaction is recommended when **any** of the following hold and no
//! compaction is already in progress:
//!
//! * The store has been idle for at least [`CompactionConfig::idle_threshold`]
//!   **and** at least [`CompactionConfig::min_interval`] has elapsed since the
//!   last compaction.
//! * Bytes written since the last compaction exceed
//!   [`CompactionConfig::max_bytes_since_compact`].

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// CompactionConfig
// ---------------------------------------------------------------------------

/// Configuration for automatic Sled compaction scheduling.
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Minimum idle duration before triggering compaction.
    ///
    /// Default: 5 minutes.
    pub idle_threshold: Duration,

    /// Minimum interval between consecutive compactions.
    ///
    /// Default: 30 minutes.
    pub min_interval: Duration,

    /// Maximum bytes written since last compaction before forcing one regardless
    /// of the idle/interval constraints.
    ///
    /// Default: 100 MiB.
    pub max_bytes_since_compact: u64,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            idle_threshold: Duration::from_secs(5 * 60),
            min_interval: Duration::from_secs(30 * 60),
            max_bytes_since_compact: 100 * 1024 * 1024,
        }
    }
}

// ---------------------------------------------------------------------------
// CompactionScheduler
// ---------------------------------------------------------------------------

/// Lock-free compaction scheduler for Sled block stores.
///
/// All mutable state is held in atomics so `Arc<CompactionScheduler>` is
/// safely shared across threads and async tasks without a mutex.
pub struct CompactionScheduler {
    config: CompactionConfig,
    /// Unix-epoch milliseconds of the last put/delete operation.
    last_operation_ms: AtomicU64,
    /// Unix-epoch milliseconds of the last successful compaction.
    last_compaction_ms: AtomicU64,
    /// Bytes written to the store since the last compaction.
    bytes_since_compaction: AtomicU64,
    /// Total compaction count since creation.
    compaction_count: AtomicU64,
    /// Guard flag: true while a compaction is in-flight.
    is_compacting: AtomicBool,
}

impl std::fmt::Debug for CompactionScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompactionScheduler")
            .field("config", &self.config)
            .field(
                "last_operation_ms",
                &self.last_operation_ms.load(Ordering::Relaxed),
            )
            .field(
                "last_compaction_ms",
                &self.last_compaction_ms.load(Ordering::Relaxed),
            )
            .field(
                "bytes_since_compaction",
                &self.bytes_since_compaction.load(Ordering::Relaxed),
            )
            .field(
                "compaction_count",
                &self.compaction_count.load(Ordering::Relaxed),
            )
            .field("is_compacting", &self.is_compacting.load(Ordering::Relaxed))
            .finish()
    }
}

/// Returns the current Unix time in milliseconds.
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_millis() as u64
}

impl CompactionScheduler {
    /// Create a new scheduler wrapped in an `Arc`.
    pub fn new(config: CompactionConfig) -> Arc<Self> {
        let now = now_ms();
        Arc::new(Self {
            config,
            // Treat creation time as the epoch for both the last operation and
            // the last compaction so that an idle store does not fire immediately.
            last_operation_ms: AtomicU64::new(now),
            last_compaction_ms: AtomicU64::new(now),
            bytes_since_compaction: AtomicU64::new(0),
            compaction_count: AtomicU64::new(0),
            is_compacting: AtomicBool::new(false),
        })
    }

    /// Record a write of `bytes` bytes.
    ///
    /// Updates the "last operation" timestamp and the byte counter that feeds
    /// the bytes-threshold trigger.
    pub fn record_write(&self, bytes: usize) {
        self.last_operation_ms.store(now_ms(), Ordering::Relaxed);
        self.bytes_since_compaction
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }

    /// Returns `true` when the scheduler recommends a compaction be triggered.
    ///
    /// Specifically it returns `true` when **all** of the following hold:
    ///
    /// * No compaction is already in-flight (`!is_compacting`).
    /// * The bytes threshold is exceeded **or** (the store has been idle long
    ///   enough **and** the minimum inter-compaction interval has been met).
    pub fn should_compact(&self) -> bool {
        if self.is_compacting.load(Ordering::Acquire) {
            return false;
        }

        let bytes = self.bytes_since_compaction.load(Ordering::Relaxed);
        if bytes >= self.config.max_bytes_since_compact {
            return true;
        }

        self.idle_duration() >= self.config.idle_threshold
            && self.time_since_last_compaction() >= self.config.min_interval
    }

    /// Attempt to claim the compaction lock.
    ///
    /// Uses a compare-and-swap to atomically transition `is_compacting` from
    /// `false` to `true`.  Returns `true` when the caller won the race (they
    /// should proceed with the compaction); returns `false` if another goroutine
    /// is already compacting.
    pub fn mark_compaction_started(&self) -> bool {
        self.is_compacting
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Release the compaction lock and update post-compaction bookkeeping.
    ///
    /// Resets the bytes counter, updates the last-compaction timestamp, and
    /// increments the compaction count before clearing the in-flight flag.
    pub fn mark_compaction_done(&self) {
        self.bytes_since_compaction.store(0, Ordering::Relaxed);
        self.last_compaction_ms.store(now_ms(), Ordering::Relaxed);
        self.compaction_count.fetch_add(1, Ordering::Relaxed);
        self.is_compacting.store(false, Ordering::Release);
    }

    /// Total number of completed compactions.
    pub fn compaction_count(&self) -> u64 {
        self.compaction_count.load(Ordering::Relaxed)
    }

    /// Bytes written to the store since the last compaction completed.
    pub fn bytes_since_last_compaction(&self) -> u64 {
        self.bytes_since_compaction.load(Ordering::Relaxed)
    }

    /// Duration since the last recorded write operation.
    pub fn idle_duration(&self) -> Duration {
        let last_ms = self.last_operation_ms.load(Ordering::Relaxed);
        let now = now_ms();
        Duration::from_millis(now.saturating_sub(last_ms))
    }

    /// Duration since the last compaction completed.
    pub fn time_since_last_compaction(&self) -> Duration {
        let last_ms = self.last_compaction_ms.load(Ordering::Relaxed);
        let now = now_ms();
        Duration::from_millis(now.saturating_sub(last_ms))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_scheduler_with_config(
        idle_threshold_secs: u64,
        min_interval_secs: u64,
        max_bytes: u64,
    ) -> Arc<CompactionScheduler> {
        CompactionScheduler::new(CompactionConfig {
            idle_threshold: Duration::from_secs(idle_threshold_secs),
            min_interval: Duration::from_secs(min_interval_secs),
            max_bytes_since_compact: max_bytes,
        })
    }

    #[test]
    fn test_default_config() {
        let cfg = CompactionConfig::default();
        assert_eq!(cfg.idle_threshold, Duration::from_secs(5 * 60));
        assert_eq!(cfg.min_interval, Duration::from_secs(30 * 60));
        assert_eq!(cfg.max_bytes_since_compact, 100 * 1024 * 1024);
    }

    #[test]
    fn test_record_write_updates_bytes() {
        let sched = make_scheduler_with_config(300, 1800, 100 * 1024 * 1024);
        assert_eq!(sched.bytes_since_last_compaction(), 0);

        sched.record_write(1024);
        assert_eq!(sched.bytes_since_last_compaction(), 1024);

        sched.record_write(512);
        assert_eq!(sched.bytes_since_last_compaction(), 1536);
    }

    #[test]
    fn test_should_compact_by_bytes() {
        // Set max_bytes very low so that a single write crosses the threshold.
        let sched = make_scheduler_with_config(300, 1800, 100);

        // Write 101 bytes — exceeds the 100-byte threshold.
        sched.record_write(101);
        assert!(
            sched.should_compact(),
            "should compact once bytes threshold is exceeded"
        );
    }

    #[test]
    fn test_should_compact_needs_min_interval() {
        // Zero idle threshold but very long min_interval.
        let sched = make_scheduler_with_config(0, 86400, 100 * 1024 * 1024);

        // Even though idle_threshold is 0, min_interval (24 h) has not elapsed.
        assert!(
            !sched.should_compact(),
            "should NOT compact when min_interval has not elapsed"
        );
    }

    #[test]
    fn test_mark_compaction_lifecycle() {
        let sched = make_scheduler_with_config(300, 1800, 100);

        // Record writes to cross the bytes threshold.
        sched.record_write(200);
        assert!(sched.should_compact());

        // Claim the lock.
        let won = sched.mark_compaction_started();
        assert!(won, "first caller must win the CAS");

        // While in-flight, should_compact must return false.
        assert!(
            !sched.should_compact(),
            "in-flight guard must block re-entry"
        );

        // Finish the compaction.
        sched.mark_compaction_done();
        assert_eq!(
            sched.bytes_since_last_compaction(),
            0,
            "bytes counter must be reset after compaction"
        );
        assert_eq!(sched.compaction_count(), 1);
        // is_compacting must be false again.
        assert!(!sched.is_compacting.load(Ordering::Relaxed));
    }

    #[test]
    fn test_concurrent_compaction_prevention() {
        let sched = make_scheduler_with_config(300, 1800, 100 * 1024 * 1024);

        // First caller wins.
        let first = sched.mark_compaction_started();
        assert!(first, "first caller must win");

        // Second caller must lose.
        let second = sched.mark_compaction_started();
        assert!(
            !second,
            "second caller must be rejected while compaction is in-flight"
        );

        // Clean up.
        sched.mark_compaction_done();
    }

    #[test]
    fn test_compaction_count_increments() {
        let sched = make_scheduler_with_config(300, 1800, 100 * 1024 * 1024);

        assert_eq!(sched.compaction_count(), 0);

        for expected in 1..=5u64 {
            let won = sched.mark_compaction_started();
            assert!(won);
            sched.mark_compaction_done();
            assert_eq!(sched.compaction_count(), expected);
        }
    }
}
