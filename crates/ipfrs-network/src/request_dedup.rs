//! Request deduplication for concurrent DHT/Bitswap lookups.
//!
//! When multiple callers request the same CID simultaneously, only one
//! DHT/Bitswap request should be issued. All additional callers coalesce
//! onto the first ("leader") request and share its result once resolved.
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_network::request_dedup::{RequestDeduplicator, AcquireResult, ResolveResult};
//! use std::time::Duration;
//!
//! let dedup = RequestDeduplicator::new(64, Duration::from_secs(30));
//!
//! let cid = "QmExample";
//! match dedup.try_acquire(cid) {
//!     AcquireResult::Leader => {
//!         // Issue the DHT/Bitswap request, then call resolve()
//!         dedup.resolve(cid, ResolveResult::NotFound);
//!     }
//!     AcquireResult::Waiter(handle) => {
//!         // Another request is in flight; handle tracks the waiter slot
//!         let _ = handle;
//!     }
//! }
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

// ─── ResolveResult ────────────────────────────────────────────────────────────

/// The outcome of a DHT/Bitswap lookup operation.
#[derive(Debug, Clone)]
pub enum ResolveResult {
    /// One or more providers were found (peer IDs / multiaddrs).
    Providers(Vec<String>),
    /// The content was not found in the network.
    NotFound,
    /// The lookup failed with an error message.
    Error(String),
}

// ─── WaiterInfo ──────────────────────────────────────────────────────────────

/// Internal record of a single waiter within a flight.
///
/// The `position` and `registered_at` values are handed directly to the public
/// [`WaiterHandle`] and not re-read from this private struct, so we keep only
/// the fields that are actually used for internal book-keeping.
#[derive(Debug, Clone)]
struct WaiterInfo {}

// ─── FlightRecord ────────────────────────────────────────────────────────────

/// Tracks a single in-flight request and all coalescers waiting for its result.
///
/// The CID key is already present as the [`HashMap`] key, so it is not
/// duplicated here.
#[derive(Debug)]
struct FlightRecord {
    started_at: Instant,
    waiters: Vec<WaiterInfo>,
    result: Option<ResolveResult>,
}

impl FlightRecord {
    fn new() -> Self {
        Self {
            started_at: Instant::now(),
            waiters: Vec::new(),
            result: None,
        }
    }

    /// Returns `true` if the flight is older than `timeout`.
    fn is_timed_out(&self, now: Instant, timeout: Duration) -> bool {
        now.duration_since(self.started_at) >= timeout
    }
}

// ─── WaiterHandle ────────────────────────────────────────────────────────────

/// A lightweight handle returned to a caller that coalesced onto an existing
/// in-flight request.
#[derive(Debug, Clone)]
pub struct WaiterHandle {
    /// The CID that this handle is waiting for.
    pub cid: String,
    /// The waiter's slot index within the flight record.
    pub position: usize,
    /// When this waiter registered.
    pub registered_at: Instant,
}

// ─── AcquireResult ───────────────────────────────────────────────────────────

/// Returned by [`RequestDeduplicator::try_acquire`].
#[derive(Debug)]
pub enum AcquireResult {
    /// This caller should issue the DHT/Bitswap request.
    Leader,
    /// Another request for the same CID is already in flight; coalesce onto it.
    Waiter(WaiterHandle),
}

// ─── DedupStats ──────────────────────────────────────────────────────────────

/// Atomic performance counters for [`RequestDeduplicator`].
#[derive(Debug, Default)]
pub struct DedupStats {
    /// Number of callers that became leaders (issued an outbound request).
    pub total_leaders: AtomicU64,
    /// Number of callers that became waiters (coalesced onto an existing request).
    pub total_waiters: AtomicU64,
    /// Number of flight records that were resolved.
    pub total_resolved: AtomicU64,
    /// Number of flight records pruned by [`RequestDeduplicator::timeout_expired_flights`].
    pub total_timeouts: AtomicU64,
    /// Number of waiters that ultimately received a coalesced result.
    pub total_coalesced: AtomicU64,
}

impl DedupStats {
    /// Returns a point-in-time snapshot of all counters.
    pub fn snapshot(&self) -> DedupStatsSnapshot {
        DedupStatsSnapshot {
            total_leaders: self.total_leaders.load(Ordering::Relaxed),
            total_waiters: self.total_waiters.load(Ordering::Relaxed),
            total_resolved: self.total_resolved.load(Ordering::Relaxed),
            total_timeouts: self.total_timeouts.load(Ordering::Relaxed),
            total_coalesced: self.total_coalesced.load(Ordering::Relaxed),
        }
    }
}

/// A point-in-time view of [`DedupStats`] counters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DedupStatsSnapshot {
    /// Leaders issued
    pub total_leaders: u64,
    /// Waiters registered
    pub total_waiters: u64,
    /// Flights resolved
    pub total_resolved: u64,
    /// Flights pruned due to timeout
    pub total_timeouts: u64,
    /// Waiters that received a coalesced result
    pub total_coalesced: u64,
}

// ─── RequestDeduplicator ─────────────────────────────────────────────────────

/// Deduplicates concurrent requests for the same CID.
///
/// `try_acquire` returns `Leader` to the first caller; all subsequent callers
/// for the same CID before `resolve` is called receive a `Waiter` handle.
/// Once the leader calls `resolve`, the flight record is marked complete and
/// removed, so the next caller starts a fresh leader/waiter cycle.
///
/// If the waiter queue reaches `max_waiters_per_key`, additional callers are
/// promoted to `Leader` so they can independently issue their own request.
#[derive(Debug)]
pub struct RequestDeduplicator {
    in_flight: Mutex<HashMap<String, FlightRecord>>,
    /// Maximum number of waiters allowed per CID before overflow callers become leaders.
    pub max_waiters_per_key: usize,
    /// Duration after which unresolved flight records are considered stale.
    pub request_timeout: Duration,
    /// Atomic counters.
    pub stats: DedupStats,
}

impl Default for RequestDeduplicator {
    fn default() -> Self {
        Self::new(64, Duration::from_secs(30))
    }
}

impl RequestDeduplicator {
    /// Creates a new deduplicator with the given waiter cap and request timeout.
    ///
    /// # Arguments
    ///
    /// * `max_waiters_per_key` – once this many waiters exist for a CID, additional
    ///   callers are promoted to `Leader` instead of waiting.
    /// * `request_timeout` – flights older than this are pruned by
    ///   [`timeout_expired_flights`](RequestDeduplicator::timeout_expired_flights).
    pub fn new(max_waiters_per_key: usize, request_timeout: Duration) -> Self {
        Self {
            in_flight: Mutex::new(HashMap::new()),
            max_waiters_per_key,
            request_timeout,
            stats: DedupStats::default(),
        }
    }

    /// Attempt to acquire the right to issue the outbound request for `cid`.
    ///
    /// Returns `Leader` if no in-flight record exists (or the waiter queue is
    /// full), or `Waiter(handle)` if the caller should coalesce.
    pub fn try_acquire(&self, cid: &str) -> AcquireResult {
        let mut map = self
            .in_flight
            .lock()
            .expect("RequestDeduplicator lock poisoned");

        match map.get_mut(cid) {
            Some(record) if record.waiters.len() < self.max_waiters_per_key => {
                // Coalesce: add this caller as a waiter.
                let position = record.waiters.len();
                let registered_at = Instant::now();
                record.waiters.push(WaiterInfo {});
                self.stats.total_waiters.fetch_add(1, Ordering::Relaxed);
                AcquireResult::Waiter(WaiterHandle {
                    cid: cid.to_owned(),
                    position,
                    registered_at,
                })
            }
            Some(_) => {
                // Waiter queue is full — promote to leader so this caller can
                // issue its own independent request.
                self.stats.total_leaders.fetch_add(1, Ordering::Relaxed);
                AcquireResult::Leader
            }
            None => {
                // No in-flight record — create one and crown this caller leader.
                map.insert(cid.to_owned(), FlightRecord::new());
                self.stats.total_leaders.fetch_add(1, Ordering::Relaxed);
                AcquireResult::Leader
            }
        }
    }

    /// Mark the in-flight record for `cid` as resolved with `result`.
    ///
    /// The flight record is removed from `in_flight`; the number of waiters
    /// that were coalesced is added to [`DedupStats::total_coalesced`].
    /// Returns `true` if an active flight was found and resolved, `false` if
    /// no matching record existed.
    pub fn resolve(&self, cid: &str, result: ResolveResult) -> bool {
        let mut map = self
            .in_flight
            .lock()
            .expect("RequestDeduplicator lock poisoned");

        if let Some(mut record) = map.remove(cid) {
            record.result = Some(result);
            let waiter_count = record.waiters.len() as u64;
            self.stats.total_resolved.fetch_add(1, Ordering::Relaxed);
            self.stats
                .total_coalesced
                .fetch_add(waiter_count, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Prune all flight records that have exceeded `request_timeout`.
    ///
    /// Returns the number of records removed.
    pub fn timeout_expired_flights(&self) -> usize {
        let now = Instant::now();
        let timeout = self.request_timeout;

        let mut map = self
            .in_flight
            .lock()
            .expect("RequestDeduplicator lock poisoned");

        let before = map.len();
        map.retain(|_, record| !record.is_timed_out(now, timeout));
        let pruned = before - map.len();

        if pruned > 0 {
            self.stats
                .total_timeouts
                .fetch_add(pruned as u64, Ordering::Relaxed);
        }
        pruned
    }

    /// Returns the number of currently in-flight records.
    pub fn in_flight_count(&self) -> usize {
        self.in_flight
            .lock()
            .expect("RequestDeduplicator lock poisoned")
            .len()
    }

    /// Returns the current waiter count for a specific CID, or `None` if no
    /// in-flight record exists for that CID.
    pub fn waiter_count_for(&self, cid: &str) -> Option<usize> {
        self.in_flight
            .lock()
            .expect("RequestDeduplicator lock poisoned")
            .get(cid)
            .map(|r| r.waiters.len())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    const CID_A: &str = "QmTestCidA111111111111111111111111111111111111111";
    const CID_B: &str = "QmTestCidB222222222222222222222222222222222222222";

    fn dedup_default() -> RequestDeduplicator {
        RequestDeduplicator::new(64, Duration::from_secs(30))
    }

    // ── 1. First caller gets Leader ───────────────────────────────────────────
    #[test]
    fn first_caller_is_leader() {
        let dedup = dedup_default();
        let result = dedup.try_acquire(CID_A);
        assert!(
            matches!(result, AcquireResult::Leader),
            "expected Leader for first caller"
        );
    }

    // ── 2. Second caller for same CID gets Waiter ────────────────────────────
    #[test]
    fn second_caller_same_cid_is_waiter() {
        let dedup = dedup_default();
        let _ = dedup.try_acquire(CID_A); // leader
        let result = dedup.try_acquire(CID_A);
        assert!(
            matches!(result, AcquireResult::Waiter(_)),
            "expected Waiter for second caller with same CID"
        );
    }

    // ── 3. Third caller for a DIFFERENT CID gets Leader ──────────────────────
    #[test]
    fn different_cid_is_new_leader() {
        let dedup = dedup_default();
        let _ = dedup.try_acquire(CID_A);
        let _ = dedup.try_acquire(CID_A);
        let result = dedup.try_acquire(CID_B);
        assert!(
            matches!(result, AcquireResult::Leader),
            "expected Leader for a different CID"
        );
    }

    // ── 4. WaiterHandle carries correct metadata ──────────────────────────────
    #[test]
    fn waiter_handle_metadata() {
        let dedup = dedup_default();
        let _ = dedup.try_acquire(CID_A); // leader; position 0 in waiters

        let handle = match dedup.try_acquire(CID_A) {
            AcquireResult::Waiter(h) => h,
            AcquireResult::Leader => panic!("expected Waiter"),
        };
        assert_eq!(handle.cid, CID_A);
        assert_eq!(handle.position, 0, "first waiter is at position 0");
    }

    // ── 5. resolve() marks the flight complete and returns true ───────────────
    #[test]
    fn resolve_clears_flight() {
        let dedup = dedup_default();
        let _ = dedup.try_acquire(CID_A);
        assert_eq!(dedup.in_flight_count(), 1);

        let ok = dedup.resolve(CID_A, ResolveResult::NotFound);
        assert!(ok, "resolve should return true for active flight");
        assert_eq!(dedup.in_flight_count(), 0, "flight should be removed");
    }

    // ── 6. resolve() returns false for unknown CID ────────────────────────────
    #[test]
    fn resolve_unknown_cid_returns_false() {
        let dedup = dedup_default();
        let ok = dedup.resolve("QmNonExistent", ResolveResult::NotFound);
        assert!(!ok, "resolve on unknown CID should return false");
    }

    // ── 7. After resolve, next caller becomes Leader again ───────────────────
    #[test]
    fn after_resolve_next_caller_is_leader() {
        let dedup = dedup_default();
        let _ = dedup.try_acquire(CID_A);
        dedup.resolve(CID_A, ResolveResult::NotFound);

        let result = dedup.try_acquire(CID_A);
        assert!(
            matches!(result, AcquireResult::Leader),
            "post-resolve caller should become Leader again"
        );
    }

    // ── 8. max_waiters_per_key: overflow callers get Leader ───────────────────
    #[test]
    fn overflow_waiters_become_leaders() {
        let max = 3usize;
        let dedup = RequestDeduplicator::new(max, Duration::from_secs(30));

        // First caller → leader
        assert!(matches!(dedup.try_acquire(CID_A), AcquireResult::Leader));

        // Fill the waiter queue to the cap
        for _ in 0..max {
            assert!(matches!(dedup.try_acquire(CID_A), AcquireResult::Waiter(_)));
        }

        // One more → must become Leader because the queue is full
        let overflow = dedup.try_acquire(CID_A);
        assert!(
            matches!(overflow, AcquireResult::Leader),
            "overflow caller should be promoted to Leader"
        );
    }

    // ── 9. timeout_expired_flights() prunes old records ──────────────────────
    #[test]
    fn timeout_prunes_expired_flights() {
        // Use a 0-second timeout so the record is immediately stale
        let dedup = RequestDeduplicator::new(64, Duration::from_secs(0));
        let _ = dedup.try_acquire(CID_A);
        assert_eq!(dedup.in_flight_count(), 1);

        let pruned = dedup.timeout_expired_flights();
        assert_eq!(pruned, 1, "one expired flight should have been pruned");
        assert_eq!(dedup.in_flight_count(), 0);
    }

    // ── 10. timeout_expired_flights() leaves fresh records intact ─────────────
    #[test]
    fn timeout_leaves_fresh_flights() {
        let dedup = RequestDeduplicator::new(64, Duration::from_secs(9999));
        let _ = dedup.try_acquire(CID_A);
        let pruned = dedup.timeout_expired_flights();
        assert_eq!(pruned, 0, "fresh flight should not be pruned");
        assert_eq!(dedup.in_flight_count(), 1);
    }

    // ── 11. Stats: leaders / waiters counters ─────────────────────────────────
    #[test]
    fn stats_leaders_and_waiters() {
        let dedup = dedup_default();
        let _ = dedup.try_acquire(CID_A); // leader 1
        let _ = dedup.try_acquire(CID_B); // leader 2
        let _ = dedup.try_acquire(CID_A); // waiter 1
        let _ = dedup.try_acquire(CID_A); // waiter 2

        let snap = dedup.stats.snapshot();
        assert_eq!(snap.total_leaders, 2, "two distinct leader acquires");
        assert_eq!(snap.total_waiters, 2, "two coalesced waiters");
    }

    // ── 12. Stats: resolved and coalesced counters ────────────────────────────
    #[test]
    fn stats_resolved_and_coalesced() {
        let dedup = dedup_default();
        let _ = dedup.try_acquire(CID_A); // leader
        let _ = dedup.try_acquire(CID_A); // waiter 1
        let _ = dedup.try_acquire(CID_A); // waiter 2

        dedup.resolve(
            CID_A,
            ResolveResult::Providers(vec!["peer1".to_owned(), "peer2".to_owned()]),
        );

        let snap = dedup.stats.snapshot();
        assert_eq!(snap.total_resolved, 1);
        assert_eq!(snap.total_coalesced, 2, "both waiters coalesced");
    }

    // ── 13. Stats: timeout counter ────────────────────────────────────────────
    #[test]
    fn stats_timeout_counter() {
        let dedup = RequestDeduplicator::new(64, Duration::from_secs(0));
        let _ = dedup.try_acquire(CID_A);
        let _ = dedup.try_acquire(CID_B);
        let pruned = dedup.timeout_expired_flights();
        assert_eq!(pruned, 2);
        let snap = dedup.stats.snapshot();
        assert_eq!(snap.total_timeouts, 2);
    }

    // ── 14. Multiple independent CIDs, independent flights ───────────────────
    #[test]
    fn multiple_cids_independent_flights() {
        let dedup = dedup_default();
        let _ = dedup.try_acquire(CID_A);
        let _ = dedup.try_acquire(CID_B);
        assert_eq!(dedup.in_flight_count(), 2);

        dedup.resolve(CID_A, ResolveResult::NotFound);
        assert_eq!(dedup.in_flight_count(), 1);

        dedup.resolve(CID_B, ResolveResult::Error("timeout".to_owned()));
        assert_eq!(dedup.in_flight_count(), 0);
    }

    // ── 15. waiter_count_for helper ───────────────────────────────────────────
    #[test]
    fn waiter_count_for_helper() {
        let dedup = dedup_default();
        assert_eq!(dedup.waiter_count_for(CID_A), None);

        let _ = dedup.try_acquire(CID_A);
        assert_eq!(dedup.waiter_count_for(CID_A), Some(0));

        let _ = dedup.try_acquire(CID_A);
        let _ = dedup.try_acquire(CID_A);
        assert_eq!(dedup.waiter_count_for(CID_A), Some(2));
    }
}
