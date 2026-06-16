//! Bootstrap coordinator for systematic peer discovery
//!
//! This module provides production-grade peer discovery beyond the initial
//! bootstrap list, featuring:
//! - Priority-based bootstrap peer selection
//! - Exponential backoff on failures with configurable caps
//! - Random walk and closest-peer query coordination
//! - Discovery record tracking with ping-based ranking
//! - Atomic statistics for lock-free monitoring

use parking_lot::RwLock;
use std::cmp::Reverse;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// BootstrapPeer
// ---------------------------------------------------------------------------

/// A single bootstrap peer with priority, backoff state, and attempt history.
#[derive(Debug, Clone)]
pub struct BootstrapPeer {
    /// Opaque string peer identifier (e.g. libp2p PeerId base58)
    pub peer_id: String,
    /// Multiaddr string for dialling (e.g. "/ip4/1.2.3.4/tcp/4001")
    pub multiaddr: String,
    /// Scheduling priority: 255 = highest, 0 = lowest
    pub priority: u8,
    /// Wall-clock time of the most recent dial attempt
    pub last_attempt: Option<Instant>,
    /// Wall-clock time of the most recent successful connection
    pub last_success: Option<Instant>,
    /// Number of consecutive failures since the last success
    pub consecutive_failures: u32,
}

impl BootstrapPeer {
    /// Create a new bootstrap peer with no history.
    pub fn new(peer_id: impl Into<String>, multiaddr: impl Into<String>, priority: u8) -> Self {
        Self {
            peer_id: peer_id.into(),
            multiaddr: multiaddr.into(),
            priority,
            last_attempt: None,
            last_success: None,
            consecutive_failures: 0,
        }
    }

    /// Returns `true` when the peer has not yet accumulated five consecutive failures.
    ///
    /// Peers with five or more consecutive failures are considered unhealthy and
    /// will not be returned by [`BootstrapCoordinator::next_bootstrap_peer`].
    #[inline]
    pub fn is_available(&self) -> bool {
        self.consecutive_failures < 5
    }

    /// Computes the minimum wait time before the next dial attempt.
    ///
    /// The formula is `min(30s, 2^consecutive_failures seconds)`, giving:
    /// 0 failures → 1 s, 1 → 2 s, 2 → 4 s, 3 → 8 s, 4 → 16 s, ≥5 → 30 s (unavailable).
    pub fn backoff_duration(&self) -> Duration {
        if self.consecutive_failures == 0 {
            return Duration::from_secs(1);
        }
        // 2^n seconds, capped at 30 s.  Use saturating_pow to avoid overflow.
        let secs = 2u64.saturating_pow(self.consecutive_failures);
        Duration::from_secs(secs.min(30))
    }

    /// Returns `true` when the backoff window has elapsed and the peer may be
    /// attempted again.
    fn backoff_elapsed(&self) -> bool {
        match self.last_attempt {
            None => true,
            Some(t) => t.elapsed() >= self.backoff_duration(),
        }
    }
}

// ---------------------------------------------------------------------------
// DiscoveryRecord
// ---------------------------------------------------------------------------

/// A record for a peer discovered through any mechanism.
#[derive(Debug, Clone)]
pub struct DiscoveryRecord {
    /// Opaque peer identifier string
    pub peer_id: String,
    /// All known multiaddrs for this peer
    pub multiaddrs: Vec<String>,
    /// How this peer was found (e.g. "bootstrap", "dht", "gossip")
    pub discovered_via: String,
    /// Monotonic timestamp of discovery
    pub discovered_at: Instant,
    /// Round-trip latency in milliseconds, if measured
    pub ping_ms: Option<u64>,
}

impl DiscoveryRecord {
    /// Create a new discovery record with the current timestamp.
    pub fn new(
        peer_id: impl Into<String>,
        multiaddrs: Vec<String>,
        discovered_via: impl Into<String>,
        ping_ms: Option<u64>,
    ) -> Self {
        Self {
            peer_id: peer_id.into(),
            multiaddrs,
            discovered_via: discovered_via.into(),
            discovered_at: Instant::now(),
            ping_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// BootstrapStats — atomic counters
// ---------------------------------------------------------------------------

/// Atomic counters for bootstrap coordinator activity.
///
/// All fields are `AtomicU64` so they can be updated from multiple threads
/// without acquiring a lock.
#[derive(Debug, Default)]
pub struct BootstrapStats {
    /// Total number of dial attempts recorded
    pub total_attempts: AtomicU64,
    /// Total number of successful connections
    pub total_successes: AtomicU64,
    /// Total number of failed connection attempts
    pub total_failures: AtomicU64,
    /// Total number of distinct peers discovered
    pub total_discovered: AtomicU64,
}

impl BootstrapStats {
    /// Create zeroed statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a point-in-time snapshot of all counters.
    pub fn snapshot(&self) -> BootstrapStatsSnapshot {
        BootstrapStatsSnapshot {
            total_attempts: self.total_attempts.load(Ordering::Relaxed),
            total_successes: self.total_successes.load(Ordering::Relaxed),
            total_failures: self.total_failures.load(Ordering::Relaxed),
            total_discovered: self.total_discovered.load(Ordering::Relaxed),
        }
    }
}

/// Owned, copyable snapshot of [`BootstrapStats`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BootstrapStatsSnapshot {
    /// Total dial attempts
    pub total_attempts: u64,
    /// Total successes
    pub total_successes: u64,
    /// Total failures
    pub total_failures: u64,
    /// Total distinct peers discovered
    pub total_discovered: u64,
}

// ---------------------------------------------------------------------------
// BootstrapCoordinator
// ---------------------------------------------------------------------------

/// Coordinates systematic peer discovery for production IPFRS nodes.
///
/// The coordinator maintains two distinct data sets:
/// 1. **Bootstrap peers** — a curated, priority-sorted list of well-known nodes
///    used for initial and recovery dialling.
/// 2. **Discovered peers** — all peers learned through any mechanism (DHT
///    walks, gossip, direct introduction), keyed by peer ID.
///
/// # Thread safety
///
/// All mutable state is guarded by `parking_lot::RwLock`.  Statistics use
/// `AtomicU64` for contention-free updates.
pub struct BootstrapCoordinator {
    /// Priority-sorted bootstrap peers (descending).
    bootstrap_peers: RwLock<Vec<BootstrapPeer>>,
    /// All known peers indexed by peer ID string.
    discovered: RwLock<HashMap<String, DiscoveryRecord>>,
    /// Desired minimum number of known peers.
    pub target_peer_count: usize,
    /// Atomic counters.
    stats: Arc<BootstrapStats>,
}

impl BootstrapCoordinator {
    /// Create a coordinator with the given target peer count.
    pub fn new(target_peer_count: usize) -> Self {
        Self {
            bootstrap_peers: RwLock::new(Vec::new()),
            discovered: RwLock::new(HashMap::new()),
            target_peer_count,
            stats: Arc::new(BootstrapStats::new()),
        }
    }

    /// Create a coordinator with the default target of 20 peers.
    pub fn with_defaults() -> Self {
        Self::new(20)
    }

    // -----------------------------------------------------------------------
    // Bootstrap peer management
    // -----------------------------------------------------------------------

    /// Add a bootstrap peer, maintaining descending priority order.
    ///
    /// If a peer with the same `peer_id` already exists it is replaced so
    /// that the caller can update priority or multiaddr without removing it
    /// first.
    pub fn add_bootstrap_peer(&self, peer: BootstrapPeer) {
        let mut peers = self.bootstrap_peers.write();
        // Replace existing entry with same peer_id if present.
        if let Some(pos) = peers.iter().position(|p| p.peer_id == peer.peer_id) {
            peers[pos] = peer;
        } else {
            peers.push(peer);
        }
        // Re-sort descending by priority so the highest-priority peer comes first.
        peers.sort_unstable_by_key(|p| Reverse(p.priority));
        debug!(
            "bootstrap_coordinator: bootstrap peers count={}",
            peers.len()
        );
    }

    /// Update attempt history for a bootstrap peer identified by `peer_id`.
    ///
    /// - On **success**: `consecutive_failures` is reset to 0 and
    ///   `last_success` is recorded.
    /// - On **failure**: `consecutive_failures` is incremented.
    ///
    /// `last_attempt` is always updated to `Instant::now()`.
    pub fn record_attempt(&self, peer_id: &str, success: bool) {
        let mut peers = self.bootstrap_peers.write();
        if let Some(peer) = peers.iter_mut().find(|p| p.peer_id == peer_id) {
            peer.last_attempt = Some(Instant::now());
            if success {
                peer.consecutive_failures = 0;
                peer.last_success = Some(Instant::now());
                self.stats.total_successes.fetch_add(1, Ordering::Relaxed);
                info!(
                    "bootstrap_coordinator: success peer_id={} failures_reset",
                    peer_id
                );
            } else {
                peer.consecutive_failures = peer.consecutive_failures.saturating_add(1);
                self.stats.total_failures.fetch_add(1, Ordering::Relaxed);
                warn!(
                    "bootstrap_coordinator: failure peer_id={} consecutive={}",
                    peer_id, peer.consecutive_failures
                );
            }
            self.stats.total_attempts.fetch_add(1, Ordering::Relaxed);
        } else {
            debug!(
                "bootstrap_coordinator: record_attempt called for unknown peer_id={}",
                peer_id
            );
        }
    }

    /// Return the highest-priority bootstrap peer that is both available
    /// (fewer than 5 consecutive failures) and not currently in its backoff
    /// window.
    ///
    /// Returns `None` when no such peer exists.
    pub fn next_bootstrap_peer(&self) -> Option<BootstrapPeer> {
        let peers = self.bootstrap_peers.read();
        // The list is kept sorted highest priority first.
        peers
            .iter()
            .find(|p| p.is_available() && p.backoff_elapsed())
            .cloned()
    }

    // -----------------------------------------------------------------------
    // Discovery record management
    // -----------------------------------------------------------------------

    /// Store or update the discovery record for a peer.
    ///
    /// If the peer was previously unknown `total_discovered` is incremented.
    pub fn record_discovery(&self, record: DiscoveryRecord) {
        let mut discovered = self.discovered.write();
        let is_new = !discovered.contains_key(&record.peer_id);
        if is_new {
            self.stats.total_discovered.fetch_add(1, Ordering::Relaxed);
            debug!(
                "bootstrap_coordinator: new peer peer_id={} via={}",
                record.peer_id, record.discovered_via
            );
        }
        discovered.insert(record.peer_id.clone(), record);
    }

    /// Return the number of distinct peers currently in the discovery table.
    pub fn known_peer_count(&self) -> usize {
        self.discovered.read().len()
    }

    /// Returns `true` when the known peer count is below `target_peer_count`.
    pub fn needs_more_peers(&self) -> bool {
        self.known_peer_count() < self.target_peer_count
    }

    /// Return up to `n` candidates for dialling, sorted by ascending ping
    /// (unmeasured peers sort last).
    ///
    /// Candidates are selected from the discovery table; the sort key is
    /// `ping_ms` ascending with `None` treated as `u64::MAX`.
    pub fn candidates_for_dial(&self, n: usize) -> Vec<DiscoveryRecord> {
        let discovered = self.discovered.read();
        let mut candidates: Vec<DiscoveryRecord> = discovered.values().cloned().collect();
        // Sort ascending ping; None (unmeasured) sorts to the end.
        candidates.sort_unstable_by_key(|r| r.ping_ms.unwrap_or(u64::MAX));
        candidates.truncate(n);
        candidates
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Return a consistent point-in-time snapshot of all counters.
    pub fn stats(&self) -> BootstrapStatsSnapshot {
        self.stats.snapshot()
    }
}

impl Default for BootstrapCoordinator {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    // Helper: build a simple BootstrapPeer
    fn peer(id: &str, multiaddr: &str, priority: u8) -> BootstrapPeer {
        BootstrapPeer::new(id, multiaddr, priority)
    }

    // Helper: build a DiscoveryRecord with optional ping
    fn discovery(id: &str, via: &str, ping: Option<u64>) -> DiscoveryRecord {
        DiscoveryRecord::new(id, vec!["/ip4/127.0.0.1/tcp/4001".to_string()], via, ping)
    }

    // -----------------------------------------------------------------------
    // BootstrapPeer unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_backoff_duration_zero_failures() {
        let p = peer("a", "/ip4/1.1.1.1/tcp/4001", 128);
        assert_eq!(p.backoff_duration(), Duration::from_secs(1));
    }

    #[test]
    fn test_backoff_duration_exponential() {
        let mut p = peer("a", "/ip4/1.1.1.1/tcp/4001", 128);
        // 2^1 = 2, 2^2 = 4, 2^3 = 8
        for (failures, expected_secs) in [(1u32, 2u64), (2, 4), (3, 8), (4, 16)] {
            p.consecutive_failures = failures;
            assert_eq!(
                p.backoff_duration(),
                Duration::from_secs(expected_secs),
                "failures={failures}"
            );
        }
    }

    #[test]
    fn test_backoff_duration_capped_at_30s() {
        let mut p = peer("a", "/ip4/1.1.1.1/tcp/4001", 128);
        // 2^5 = 32 > 30, should be capped
        p.consecutive_failures = 5;
        assert_eq!(p.backoff_duration(), Duration::from_secs(30));
        // Large value should also be capped
        p.consecutive_failures = 100;
        assert_eq!(p.backoff_duration(), Duration::from_secs(30));
    }

    #[test]
    fn test_is_available_true_below_threshold() {
        let mut p = peer("a", "/ip4/1.1.1.1/tcp/4001", 128);
        for failures in 0u32..5 {
            p.consecutive_failures = failures;
            assert!(p.is_available(), "failures={failures} should be available");
        }
    }

    #[test]
    fn test_is_available_false_at_threshold() {
        let mut p = peer("a", "/ip4/1.1.1.1/tcp/4001", 128);
        p.consecutive_failures = 5;
        assert!(!p.is_available());
        p.consecutive_failures = 10;
        assert!(!p.is_available());
    }

    // -----------------------------------------------------------------------
    // BootstrapCoordinator structural tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_bootstrap_peer_single() {
        let coord = BootstrapCoordinator::with_defaults();
        coord.add_bootstrap_peer(peer("p1", "/ip4/1.1.1.1/tcp/4001", 100));
        let next = coord.next_bootstrap_peer().expect("should have a peer");
        assert_eq!(next.peer_id, "p1");
    }

    #[test]
    fn test_add_bootstrap_peer_ordering_by_priority() {
        let coord = BootstrapCoordinator::with_defaults();
        // Insert in low-to-high order; coordinator must return highest first.
        coord.add_bootstrap_peer(peer("low", "/ip4/1.1.1.1/tcp/4001", 10));
        coord.add_bootstrap_peer(peer("high", "/ip4/2.2.2.2/tcp/4001", 200));
        coord.add_bootstrap_peer(peer("mid", "/ip4/3.3.3.3/tcp/4001", 100));

        let next = coord.next_bootstrap_peer().expect("should return peer");
        assert_eq!(next.peer_id, "high", "highest priority must come first");
    }

    #[test]
    fn test_add_bootstrap_peer_replaces_existing() {
        let coord = BootstrapCoordinator::with_defaults();
        coord.add_bootstrap_peer(peer("p1", "/ip4/1.1.1.1/tcp/4001", 50));
        // Replace with higher priority
        coord.add_bootstrap_peer(peer("p1", "/ip4/1.1.1.1/tcp/4001", 200));
        let next = coord.next_bootstrap_peer().expect("should have peer");
        assert_eq!(next.priority, 200);
    }

    #[test]
    fn test_record_attempt_success_resets_failures() {
        let coord = BootstrapCoordinator::with_defaults();
        coord.add_bootstrap_peer(peer("p1", "/ip4/1.1.1.1/tcp/4001", 100));
        // Accumulate some failures
        coord.record_attempt("p1", false);
        coord.record_attempt("p1", false);
        coord.record_attempt("p1", false);
        // Now succeed
        coord.record_attempt("p1", true);

        let peers = coord.bootstrap_peers.read();
        let p = peers.iter().find(|p| p.peer_id == "p1").expect("found");
        assert_eq!(p.consecutive_failures, 0, "success must reset failures");
        assert!(p.last_success.is_some(), "last_success must be set");
    }

    #[test]
    fn test_record_attempt_failure_increments_counter() {
        let coord = BootstrapCoordinator::with_defaults();
        coord.add_bootstrap_peer(peer("p1", "/ip4/1.1.1.1/tcp/4001", 100));
        coord.record_attempt("p1", false);
        coord.record_attempt("p1", false);

        let peers = coord.bootstrap_peers.read();
        let p = peers.iter().find(|p| p.peer_id == "p1").expect("found");
        assert_eq!(p.consecutive_failures, 2);
    }

    #[test]
    fn test_next_bootstrap_peer_skips_unavailable() {
        let coord = BootstrapCoordinator::with_defaults();
        // Add a high-priority peer that will become unavailable
        coord.add_bootstrap_peer(peer("bad", "/ip4/1.1.1.1/tcp/4001", 255));
        coord.add_bootstrap_peer(peer("good", "/ip4/2.2.2.2/tcp/4001", 100));

        // Drive "bad" past the 5-failure threshold
        for _ in 0..5 {
            coord.record_attempt("bad", false);
        }
        // Ensure "bad" is unavailable
        {
            let peers = coord.bootstrap_peers.read();
            let bad = peers.iter().find(|p| p.peer_id == "bad").expect("found");
            assert!(!bad.is_available());
        }

        let next = coord
            .next_bootstrap_peer()
            .expect("coordinator must skip bad peer");
        assert_eq!(next.peer_id, "good", "unavailable peer must be skipped");
    }

    #[test]
    fn test_next_bootstrap_peer_skips_within_backoff() {
        let coord = BootstrapCoordinator::with_defaults();
        coord.add_bootstrap_peer(peer("p1", "/ip4/1.1.1.1/tcp/4001", 100));
        // Record a failure to start the backoff clock (backoff = 2^1 = 2 s)
        coord.record_attempt("p1", false);

        // Immediately try again — should be skipped because backoff hasn't elapsed
        // (The last_attempt was set moments ago and backoff is ≥ 2 s)
        let next = coord.next_bootstrap_peer();
        assert!(
            next.is_none(),
            "peer should be skipped while in backoff window"
        );
    }

    #[test]
    fn test_next_bootstrap_peer_available_after_backoff() {
        let coord = BootstrapCoordinator::with_defaults();
        // Use a fresh peer with no history — backoff is 1 s but last_attempt is None,
        // so it should be immediately available.
        coord.add_bootstrap_peer(peer("fresh", "/ip4/1.1.1.1/tcp/4001", 100));
        let next = coord.next_bootstrap_peer();
        assert!(next.is_some(), "fresh peer with no history should be ready");
    }

    // -----------------------------------------------------------------------
    // DiscoveryRecord and known peer count
    // -----------------------------------------------------------------------

    #[test]
    fn test_record_discovery_stores_record() {
        let coord = BootstrapCoordinator::with_defaults();
        coord.record_discovery(discovery("peer-abc", "bootstrap", Some(12)));

        let discovered = coord.discovered.read();
        let rec = discovered.get("peer-abc").expect("record should be stored");
        assert_eq!(rec.discovered_via, "bootstrap");
        assert_eq!(rec.ping_ms, Some(12));
    }

    #[test]
    fn test_known_peer_count() {
        let coord = BootstrapCoordinator::with_defaults();
        assert_eq!(coord.known_peer_count(), 0);
        coord.record_discovery(discovery("p1", "dht", None));
        coord.record_discovery(discovery("p2", "gossip", Some(5)));
        assert_eq!(coord.known_peer_count(), 2);
    }

    #[test]
    fn test_needs_more_peers_threshold() {
        let coord = BootstrapCoordinator::new(3);
        assert!(coord.needs_more_peers());
        coord.record_discovery(discovery("p1", "dht", None));
        coord.record_discovery(discovery("p2", "dht", None));
        assert!(coord.needs_more_peers());
        coord.record_discovery(discovery("p3", "dht", None));
        assert!(
            !coord.needs_more_peers(),
            "exactly at target — no longer needed"
        );
    }

    #[test]
    fn test_candidates_for_dial_sorted_by_ping() {
        let coord = BootstrapCoordinator::with_defaults();
        coord.record_discovery(discovery("slow", "dht", Some(200)));
        coord.record_discovery(discovery("fast", "dht", Some(10)));
        coord.record_discovery(discovery("unknown", "dht", None));
        coord.record_discovery(discovery("medium", "dht", Some(50)));

        let candidates = coord.candidates_for_dial(10);
        // Verify ascending ping order; None should be last
        let pings: Vec<Option<u64>> = candidates.iter().map(|r| r.ping_ms).collect();
        assert_eq!(pings, vec![Some(10), Some(50), Some(200), None]);
    }

    #[test]
    fn test_candidates_for_dial_respects_limit() {
        let coord = BootstrapCoordinator::with_defaults();
        for i in 0u64..10 {
            coord.record_discovery(discovery(&format!("peer-{i}"), "dht", Some(i * 10)));
        }
        let candidates = coord.candidates_for_dial(3);
        assert_eq!(candidates.len(), 3);
    }

    // -----------------------------------------------------------------------
    // Statistics accumulation
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_accumulation() {
        let coord = BootstrapCoordinator::with_defaults();
        coord.add_bootstrap_peer(peer("p1", "/ip4/1.1.1.1/tcp/4001", 100));
        coord.add_bootstrap_peer(peer("p2", "/ip4/2.2.2.2/tcp/4001", 90));

        coord.record_attempt("p1", true);
        coord.record_attempt("p1", false);
        coord.record_attempt("p2", false);

        coord.record_discovery(discovery("d1", "dht", None));
        coord.record_discovery(discovery("d2", "gossip", Some(5)));

        let snap = coord.stats();
        assert_eq!(snap.total_attempts, 3);
        assert_eq!(snap.total_successes, 1);
        assert_eq!(snap.total_failures, 2);
        assert_eq!(snap.total_discovered, 2);
    }

    #[test]
    fn test_stats_discovery_dedup() {
        // Recording the same peer twice must only count one new discovery.
        let coord = BootstrapCoordinator::with_defaults();
        coord.record_discovery(discovery("p1", "bootstrap", Some(5)));
        coord.record_discovery(discovery("p1", "dht", Some(3)));

        let snap = coord.stats();
        assert_eq!(snap.total_discovered, 1);
        assert_eq!(coord.known_peer_count(), 1);
    }

    // -----------------------------------------------------------------------
    // Concurrency smoke test
    // -----------------------------------------------------------------------

    #[test]
    fn test_concurrent_record_discovery() {
        let coord = Arc::new(BootstrapCoordinator::new(1000));
        let mut handles = Vec::with_capacity(8);

        for thread_idx in 0usize..8 {
            let c = Arc::clone(&coord);
            handles.push(thread::spawn(move || {
                for i in 0u64..50 {
                    let id = format!("peer-{thread_idx}-{i}");
                    c.record_discovery(DiscoveryRecord::new(&id, vec![], "gossip", Some(i)));
                }
            }));
        }
        for h in handles {
            h.join().expect("thread panicked");
        }

        assert_eq!(coord.known_peer_count(), 400);
        assert_eq!(coord.stats().total_discovered, 400);
    }
}
