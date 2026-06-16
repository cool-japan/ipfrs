//! Per-session block exchange configuration and metrics.
//!
//! This module provides fine-grained, per-session configuration for block
//! exchange operations (timeouts, concurrency, retry policy) alongside
//! lock-free atomic metrics collection and a registry for tracking multiple
//! concurrent sessions.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

// ─── SessionConfig ────────────────────────────────────────────────────────────

/// Per-session transfer configuration for block exchange operations.
///
/// Controls timeouts, concurrency limits, and retry behaviour for a single
/// block-exchange session.  Three convenience constructors cover the most
/// common scenarios; individual fields can always be overridden afterwards.
///
/// Note: this type lives in the `session_config` module and is distinct from
/// [`crate::session::SessionConfig`], which governs higher-level session
/// orchestration (priorities, concurrent blocks, etc.).
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Session-level timeout.  Default: 30 s.
    pub timeout: Duration,
    /// Maximum total bytes to transfer in this session.  Default: 1 GiB.
    pub max_bytes: u64,
    /// Maximum concurrent block requests.  Default: 16.
    pub max_concurrent_requests: usize,
    /// Per-request timeout.  Default: 10 s.
    pub request_timeout: Duration,
    /// Maximum retry attempts per block.  Default: 3.
    pub max_retries: usize,
    /// Exponential backoff base duration.  Default: 200 ms.
    pub backoff_base: Duration,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            max_bytes: 1024 * 1024 * 1024,
            max_concurrent_requests: 16,
            request_timeout: Duration::from_secs(10),
            max_retries: 3,
            backoff_base: Duration::from_millis(200),
        }
    }
}

impl SessionConfig {
    /// Create a high-throughput config (longer timeouts, more concurrency).
    pub fn high_throughput() -> Self {
        Self {
            timeout: Duration::from_secs(120),
            max_bytes: 10 * 1024 * 1024 * 1024,
            max_concurrent_requests: 64,
            request_timeout: Duration::from_secs(30),
            max_retries: 5,
            backoff_base: Duration::from_millis(100),
        }
    }

    /// Create a low-latency config (short timeouts, fail fast).
    pub fn low_latency() -> Self {
        Self {
            timeout: Duration::from_secs(5),
            max_bytes: 100 * 1024 * 1024,
            max_concurrent_requests: 8,
            request_timeout: Duration::from_secs(2),
            max_retries: 1,
            backoff_base: Duration::from_millis(50),
        }
    }

    /// Compute backoff duration for retry `n` using exponential backoff.
    ///
    /// `backoff = base × 2^min(retry, 6)` (capped at 32× base to prevent
    /// runaway growth).
    pub fn backoff_for_retry(&self, retry: usize) -> Duration {
        let exp = retry.min(6) as u32;
        self.backoff_base.saturating_mul(2u32.pow(exp))
    }
}

// ─── SessionMetricsSnapshot ───────────────────────────────────────────────────

/// Immutable metrics snapshot for a completed or in-progress session.
#[derive(Debug, Clone)]
pub struct SessionMetricsSnapshot {
    /// Unique identifier for this session.
    pub session_id: String,
    /// Total bytes successfully transferred.
    pub bytes_transferred: u64,
    /// Number of block requests issued.
    pub blocks_requested: u64,
    /// Number of blocks successfully received.
    pub blocks_received: u64,
    /// Number of blocks that ultimately failed.
    pub blocks_failed: u64,
    /// Total number of retried block requests.
    pub total_retries: u64,
    /// Wall-clock elapsed time in milliseconds.
    pub elapsed_ms: u64,
    /// Throughput in bytes per second (`bytes_transferred / (elapsed_ms / 1000)`).
    pub throughput_bps: f64,
}

// ─── SessionMetrics ───────────────────────────────────────────────────────────

/// Live per-session metrics backed by lock-free atomics.
///
/// All `record_*` methods use `Relaxed` ordering; the `snapshot` method uses
/// `Acquire` to observe a consistent view.
pub struct SessionMetrics {
    /// Unique identifier for this session.
    pub session_id: String,
    bytes_transferred: AtomicU64,
    blocks_requested: AtomicU64,
    blocks_received: AtomicU64,
    blocks_failed: AtomicU64,
    total_retries: AtomicU64,
    started_at_ms: u64,
}

impl SessionMetrics {
    /// Create a new metrics instance wrapped in an `Arc`.
    ///
    /// `started_at_ms` is the session start time as Unix milliseconds.
    pub fn new(session_id: impl Into<String>, started_at_ms: u64) -> Arc<Self> {
        Arc::new(Self {
            session_id: session_id.into(),
            bytes_transferred: AtomicU64::new(0),
            blocks_requested: AtomicU64::new(0),
            blocks_received: AtomicU64::new(0),
            blocks_failed: AtomicU64::new(0),
            total_retries: AtomicU64::new(0),
            started_at_ms,
        })
    }

    /// Accumulate `bytes` into the total bytes-transferred counter.
    pub fn record_bytes(&self, bytes: u64) {
        self.bytes_transferred.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Increment the blocks-requested counter by one.
    pub fn record_block_requested(&self) {
        self.blocks_requested.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the blocks-received counter by one.
    pub fn record_block_received(&self) {
        self.blocks_received.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the blocks-failed counter by one.
    pub fn record_block_failed(&self) {
        self.blocks_failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the total-retries counter by one.
    pub fn record_retry(&self) {
        self.total_retries.fetch_add(1, Ordering::Relaxed);
    }

    /// Produce a consistent snapshot observed at `now_ms` (Unix milliseconds).
    pub fn snapshot(&self, now_ms: u64) -> SessionMetricsSnapshot {
        let bytes_transferred = self.bytes_transferred.load(Ordering::Acquire);
        let elapsed_ms = now_ms.saturating_sub(self.started_at_ms);
        let elapsed_secs = elapsed_ms as f64 / 1_000.0;
        let throughput_bps = if elapsed_secs > 0.0 {
            bytes_transferred as f64 / elapsed_secs
        } else {
            0.0
        };

        SessionMetricsSnapshot {
            session_id: self.session_id.clone(),
            bytes_transferred,
            blocks_requested: self.blocks_requested.load(Ordering::Acquire),
            blocks_received: self.blocks_received.load(Ordering::Acquire),
            blocks_failed: self.blocks_failed.load(Ordering::Acquire),
            total_retries: self.total_retries.load(Ordering::Acquire),
            elapsed_ms,
            throughput_bps,
        }
    }

    /// Return the current bytes-transferred value.
    pub fn bytes_transferred(&self) -> u64 {
        self.bytes_transferred.load(Ordering::Relaxed)
    }

    /// Return the current blocks-received value.
    pub fn blocks_received(&self) -> u64 {
        self.blocks_received.load(Ordering::Relaxed)
    }
}

impl std::fmt::Debug for SessionMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionMetrics")
            .field("session_id", &self.session_id)
            .field(
                "bytes_transferred",
                &self.bytes_transferred.load(Ordering::Relaxed),
            )
            .field(
                "blocks_requested",
                &self.blocks_requested.load(Ordering::Relaxed),
            )
            .field(
                "blocks_received",
                &self.blocks_received.load(Ordering::Relaxed),
            )
            .field("blocks_failed", &self.blocks_failed.load(Ordering::Relaxed))
            .field("total_retries", &self.total_retries.load(Ordering::Relaxed))
            .field("started_at_ms", &self.started_at_ms)
            .finish()
    }
}

// ─── SessionMetricsStore ──────────────────────────────────────────────────────

/// Registry of per-session metrics.
///
/// Active sessions are stored by ID.  When a session is completed via
/// [`complete_session`](SessionMetricsStore::complete_session) its final
/// snapshot is appended to the `completed` ring buffer (capped at
/// `max_sessions` entries, oldest evicted first).
pub struct SessionMetricsStore {
    sessions: RwLock<HashMap<String, Arc<SessionMetrics>>>,
    max_sessions: usize,
    completed: RwLock<Vec<SessionMetricsSnapshot>>,
}

impl SessionMetricsStore {
    /// Create a new store wrapped in an `Arc`.
    ///
    /// `max_sessions` caps the number of retained completed-session snapshots.
    pub fn new(max_sessions: usize) -> Arc<Self> {
        Arc::new(Self {
            sessions: RwLock::new(HashMap::new()),
            max_sessions,
            completed: RwLock::new(Vec::new()),
        })
    }

    /// Create and register a new active session, returning its metrics handle.
    ///
    /// If a session with the same ID already exists it is replaced.
    pub fn create_session(
        &self,
        session_id: impl Into<String>,
        started_at_ms: u64,
    ) -> Arc<SessionMetrics> {
        let id: String = session_id.into();
        let metrics = SessionMetrics::new(id.clone(), started_at_ms);
        self.sessions.write().insert(id, Arc::clone(&metrics));
        metrics
    }

    /// Look up an active session by ID.
    pub fn get_session(&self, session_id: &str) -> Option<Arc<SessionMetrics>> {
        self.sessions.read().get(session_id).cloned()
    }

    /// Mark a session as complete: snapshot it and move it to the completed
    /// ring buffer.
    ///
    /// Returns `true` if the session was found and moved, `false` otherwise.
    pub fn complete_session(&self, session_id: &str, now_ms: u64) -> bool {
        let removed = self.sessions.write().remove(session_id);
        match removed {
            None => false,
            Some(metrics) => {
                let snap = metrics.snapshot(now_ms);
                let mut completed = self.completed.write();
                completed.push(snap);
                // Evict oldest entries when the ring buffer is full.
                while completed.len() > self.max_sessions {
                    completed.remove(0);
                }
                true
            }
        }
    }

    /// Return the number of currently active (not yet completed) sessions.
    pub fn active_count(&self) -> usize {
        self.sessions.read().len()
    }

    /// Clone and return all completed session snapshots.
    pub fn completed_snapshots(&self) -> Vec<SessionMetricsSnapshot> {
        self.completed.read().clone()
    }

    /// Sum `bytes_transferred` across all currently active sessions.
    pub fn total_bytes_transferred(&self) -> u64 {
        self.sessions
            .read()
            .values()
            .map(|m| m.bytes_transferred())
            .sum()
    }
}

impl std::fmt::Debug for SessionMetricsStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionMetricsStore")
            .field("active_count", &self.active_count())
            .field("max_sessions", &self.max_sessions)
            .field("completed_count", &self.completed.read().len())
            .finish()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── BlockExchangeConfig ──────────────────────────────────────────────────

    #[test]
    fn test_default_config_values() {
        let cfg = SessionConfig::default();
        assert_eq!(cfg.timeout, Duration::from_secs(30));
        assert_eq!(cfg.max_bytes, 1024 * 1024 * 1024);
        assert_eq!(cfg.max_concurrent_requests, 16);
        assert_eq!(cfg.request_timeout, Duration::from_secs(10));
        assert_eq!(cfg.max_retries, 3);
        assert_eq!(cfg.backoff_base, Duration::from_millis(200));
    }

    #[test]
    fn test_high_throughput_config() {
        let cfg = SessionConfig::high_throughput();
        assert_eq!(cfg.timeout, Duration::from_secs(120));
        assert_eq!(cfg.max_bytes, 10 * 1024 * 1024 * 1024);
        assert_eq!(cfg.max_concurrent_requests, 64);
        assert_eq!(cfg.request_timeout, Duration::from_secs(30));
        assert_eq!(cfg.max_retries, 5);
        assert_eq!(cfg.backoff_base, Duration::from_millis(100));
    }

    #[test]
    fn test_low_latency_config() {
        let cfg = SessionConfig::low_latency();
        assert_eq!(cfg.timeout, Duration::from_secs(5));
        assert_eq!(cfg.max_bytes, 100 * 1024 * 1024);
        assert_eq!(cfg.max_concurrent_requests, 8);
        assert_eq!(cfg.request_timeout, Duration::from_secs(2));
        assert_eq!(cfg.max_retries, 1);
        assert_eq!(cfg.backoff_base, Duration::from_millis(50));
    }

    #[test]
    fn test_backoff_exponential() {
        let cfg = SessionConfig::default(); // base = 200 ms
        assert_eq!(cfg.backoff_for_retry(0), Duration::from_millis(200)); // 200 * 2^0
        assert_eq!(cfg.backoff_for_retry(1), Duration::from_millis(400)); // 200 * 2^1
        assert_eq!(cfg.backoff_for_retry(2), Duration::from_millis(800)); // 200 * 2^2
        assert_eq!(cfg.backoff_for_retry(3), Duration::from_millis(1_600));
        assert_eq!(cfg.backoff_for_retry(4), Duration::from_millis(3_200));
        assert_eq!(cfg.backoff_for_retry(5), Duration::from_millis(6_400));
        // retry 6 and above are all capped at 2^6 = 64× base = 12 800 ms
        assert_eq!(cfg.backoff_for_retry(6), Duration::from_millis(12_800));
        assert_eq!(cfg.backoff_for_retry(7), Duration::from_millis(12_800));
        assert_eq!(cfg.backoff_for_retry(100), Duration::from_millis(12_800));
    }

    // ── SessionMetrics ───────────────────────────────────────────────────────

    #[test]
    fn test_session_metrics_record_bytes() {
        let m = SessionMetrics::new("sess-bytes", 0);
        assert_eq!(m.bytes_transferred(), 0);
        m.record_bytes(512);
        m.record_bytes(1024);
        assert_eq!(m.bytes_transferred(), 1_536);
    }

    #[test]
    fn test_session_metrics_record_blocks() {
        let m = SessionMetrics::new("sess-blocks", 0);
        m.record_block_requested();
        m.record_block_requested();
        m.record_block_requested();
        m.record_block_received();
        m.record_block_received();
        m.record_block_failed();
        m.record_retry();
        m.record_retry();

        let snap = m.snapshot(1_000);
        assert_eq!(snap.blocks_requested, 3);
        assert_eq!(snap.blocks_received, 2);
        assert_eq!(snap.blocks_failed, 1);
        assert_eq!(snap.total_retries, 2);
        assert_eq!(m.blocks_received(), 2);
    }

    #[test]
    fn test_session_metrics_snapshot_throughput() {
        let m = SessionMetrics::new("sess-tput", 0);
        // Transfer 1 MiB in a simulated 2 000 ms window.
        let mib: u64 = 1024 * 1024;
        m.record_bytes(mib);
        let snap = m.snapshot(2_000);
        assert_eq!(snap.elapsed_ms, 2_000);
        assert_eq!(snap.bytes_transferred, mib);
        // throughput = 1 048 576 bytes / 2 s = 524 288 bps
        let expected = mib as f64 / 2.0;
        let diff = (snap.throughput_bps - expected).abs();
        assert!(
            diff < 1.0,
            "throughput mismatch: got {}, expected {}",
            snap.throughput_bps,
            expected
        );
    }

    // ── SessionMetricsStore ──────────────────────────────────────────────────

    #[test]
    fn test_session_metrics_store_create_and_get() {
        let store = SessionMetricsStore::new(10);
        assert_eq!(store.active_count(), 0);

        let m = store.create_session("alpha", 1_000);
        m.record_bytes(256);
        assert_eq!(store.active_count(), 1);

        let fetched = store.get_session("alpha").expect("session should exist");
        assert_eq!(fetched.bytes_transferred(), 256);

        assert!(store.get_session("missing").is_none());
    }

    #[test]
    fn test_session_metrics_store_complete() {
        let store = SessionMetricsStore::new(10);
        store.create_session("beta", 0).record_bytes(4_096);

        assert_eq!(store.active_count(), 1);
        assert!(store.completed_snapshots().is_empty());

        let found = store.complete_session("beta", 1_000);
        assert!(found, "complete_session should return true");
        assert_eq!(store.active_count(), 0);

        let snaps = store.completed_snapshots();
        assert_eq!(snaps.len(), 1);
        assert_eq!(snaps[0].session_id, "beta");
        assert_eq!(snaps[0].bytes_transferred, 4_096);

        // Completing an unknown session returns false.
        assert!(!store.complete_session("ghost", 2_000));
    }

    #[test]
    fn test_session_metrics_store_total_bytes() {
        let store = SessionMetricsStore::new(10);
        store.create_session("s1", 0).record_bytes(100);
        store.create_session("s2", 0).record_bytes(200);
        store.create_session("s3", 0).record_bytes(300);

        assert_eq!(store.total_bytes_transferred(), 600);

        // Completing s2 removes it from active; total_bytes only covers active.
        store.complete_session("s2", 1_000);
        assert_eq!(store.total_bytes_transferred(), 400);
    }

    #[test]
    fn test_max_sessions_cap() {
        // max_sessions = 3: completing more sessions evicts the oldest snapshots.
        let store = SessionMetricsStore::new(3);

        for i in 0..6u64 {
            let id = format!("sess-{}", i);
            store.create_session(&id, i * 100).record_bytes(i * 10);
            store.complete_session(&id, (i + 1) * 100);
        }

        // Active sessions: all completed.
        assert_eq!(store.active_count(), 0);

        // Completed ring buffer must not exceed max_sessions (3).
        let snaps = store.completed_snapshots();
        assert_eq!(snaps.len(), 3);

        // The three most recent sessions (3, 4, 5) should be retained.
        let ids: Vec<&str> = snaps.iter().map(|s| s.session_id.as_str()).collect();
        assert!(ids.contains(&"sess-3"));
        assert!(ids.contains(&"sess-4"));
        assert!(ids.contains(&"sess-5"));

        // Creating new sessions beyond any limit always succeeds.
        let extra = store.create_session("extra-1", 9_000);
        extra.record_bytes(999);
        assert_eq!(store.active_count(), 1);
        assert_eq!(extra.bytes_transferred(), 999);
    }
}
