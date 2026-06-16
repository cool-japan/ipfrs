//! DHT-aware content routing table with geographic/latency affinity scoring.
//!
//! This module implements a `ContentRoutingTable` that maps CID strings to
//! ordered lists of provider [`RoutingEntry`] records.  Entries are ranked by
//! an *affinity score* that combines observed latency, recent activity and
//! remaining TTL so that the best provider is always returned first.
//!
//! ## Design
//!
//! * **Lock granularity** – a single `RwLock<HashMap<…>>` wrapping all
//!   entries keeps the implementation simple while still allowing concurrent
//!   reads.  Writes (add / remove / evict) are short-lived.
//! * **Capacity enforcement** – when the per-CID provider list reaches
//!   `max_providers_per_cid`, a new entry is rejected with
//!   [`RoutingError::CapacityExceeded`].
//! * **Affinity scoring** – see [`RoutingEntry::affinity_score`] for the
//!   exact formula.
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_network::routing_table::{ContentRoutingTable, RoutingEntry};
//! use std::time::{Duration, Instant};
//!
//! let table = ContentRoutingTable::new("local-peer".to_string(), 20);
//! let entry = RoutingEntry::new("peer-1".to_string(), vec!["/ip4/1.2.3.4/tcp/4001".to_string()]);
//! table.add_provider("QmFoo", entry).unwrap();
//! assert_eq!(table.provider_count(), 1);
//! ```

use parking_lot::RwLock;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use thiserror::Error;

// ────────────────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────────────────

/// Default maximum number of providers recorded per CID.
pub const DEFAULT_MAX_PROVIDERS: usize = 20;

/// Default TTL for a routing entry (24 hours).
pub const DEFAULT_ENTRY_TTL: Duration = Duration::from_secs(86_400);

/// Threshold below which an entry is considered "recently seen" (5 minutes).
const RECENT_SEEN_THRESHOLD: Duration = Duration::from_secs(300);

/// Fraction of TTL remaining below which an entry is penalised.
const TTL_NEARLY_EXPIRED_FRACTION: f64 = 0.10;

// ────────────────────────────────────────────────────────────────────────────
// Error type
// ────────────────────────────────────────────────────────────────────────────

/// Errors produced by [`ContentRoutingTable`] operations.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum RoutingError {
    /// The per-CID provider list is full.
    #[error("capacity exceeded for CID '{cid}': max {max} providers")]
    CapacityExceeded {
        /// The CID whose bucket is full.
        cid: String,
        /// The configured limit.
        max: usize,
    },

    /// The same peer is already registered as a provider for this CID.
    #[error("duplicate provider '{peer_id}' for CID '{cid}'")]
    DuplicateProvider {
        /// The CID for which duplication was detected.
        cid: String,
        /// The peer that was already registered.
        peer_id: String,
    },
}

// ────────────────────────────────────────────────────────────────────────────
// RoutingEntry
// ────────────────────────────────────────────────────────────────────────────

/// A single provider record held inside the routing table.
#[derive(Debug, Clone)]
pub struct RoutingEntry {
    /// Libp2p peer identifier (string form).
    pub peer_id: String,
    /// Known multiaddrs for this peer.
    pub multiaddrs: Vec<String>,
    /// Observed round-trip latency in milliseconds, if measured.
    pub latency_ms: Option<u64>,
    /// Monotonic timestamp of the last time this entry was refreshed.
    pub last_seen: Instant,
    /// How long this entry remains valid after creation / refresh.
    pub ttl: Duration,
}

impl RoutingEntry {
    /// Create a new entry with default TTL and no latency measurement.
    pub fn new(peer_id: String, multiaddrs: Vec<String>) -> Self {
        Self {
            peer_id,
            multiaddrs,
            latency_ms: None,
            last_seen: Instant::now(),
            ttl: DEFAULT_ENTRY_TTL,
        }
    }

    /// Create an entry with a custom TTL.
    pub fn with_ttl(peer_id: String, multiaddrs: Vec<String>, ttl: Duration) -> Self {
        Self {
            peer_id,
            multiaddrs,
            latency_ms: None,
            last_seen: Instant::now(),
            ttl,
        }
    }

    /// Create an entry with observed latency and a custom TTL.
    pub fn with_latency(
        peer_id: String,
        multiaddrs: Vec<String>,
        latency_ms: u64,
        ttl: Duration,
    ) -> Self {
        Self {
            peer_id,
            multiaddrs,
            latency_ms: Some(latency_ms),
            last_seen: Instant::now(),
            ttl,
        }
    }

    /// Compute a floating-point affinity score used for ranking.
    ///
    /// Formula:
    /// * Base score: **1.0**
    /// * Subtract `0.001 × latency_ms` if latency is known.
    /// * Subtract **0.5** if less than 10 % of the TTL remains.
    /// * Add **0.2** if the entry was seen within the last 5 minutes.
    ///
    /// Higher scores rank first.
    pub fn affinity_score(&self) -> f64 {
        let mut score = 1.0_f64;

        // Penalise for observed latency.
        if let Some(lat) = self.latency_ms {
            score -= 0.001 * (lat as f64);
        }

        // Penalise for near-expiry.
        let elapsed = self.last_seen.elapsed();
        let ttl_secs = self.ttl.as_secs_f64();
        if ttl_secs > 0.0 {
            let remaining_fraction = 1.0 - (elapsed.as_secs_f64() / ttl_secs);
            if remaining_fraction < TTL_NEARLY_EXPIRED_FRACTION {
                score -= 0.5;
            }
        }

        // Bonus for recency.
        if elapsed < RECENT_SEEN_THRESHOLD {
            score += 0.2;
        }

        score
    }

    /// Return `true` when the entry's TTL has elapsed relative to `now`.
    pub fn is_expired(&self, now: Instant) -> bool {
        // Duration since the entry was last refreshed
        let age = now.saturating_duration_since(self.last_seen);
        age >= self.ttl
    }
}

// ────────────────────────────────────────────────────────────────────────────
// RoutingTableStats
// ────────────────────────────────────────────────────────────────────────────

/// Summary statistics for a [`ContentRoutingTable`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutingTableStats {
    /// Number of distinct CIDs tracked.
    pub cid_count: usize,
    /// Total number of provider entries across all CIDs.
    pub provider_count: usize,
    /// Cumulative count of entries removed by [`ContentRoutingTable::evict_expired`].
    pub expired_evicted: u64,
}

// ────────────────────────────────────────────────────────────────────────────
// ContentRoutingTable
// ────────────────────────────────────────────────────────────────────────────

/// DHT-aware content routing table with per-CID provider lists and affinity
/// scoring.
pub struct ContentRoutingTable {
    /// CID string → ordered provider list (unsorted at rest; sorted on read).
    entries: RwLock<HashMap<String, Vec<RoutingEntry>>>,
    /// Maximum number of providers stored per CID.
    max_providers_per_cid: usize,
    /// Peer ID of the local node (reserved for future self-announcement logic).
    pub local_peer_id: String,
    /// Monotonically increasing count of expired entries evicted.
    expired_evicted: RwLock<u64>,
}

impl ContentRoutingTable {
    /// Create a new routing table.
    ///
    /// # Arguments
    ///
    /// * `local_peer_id` – string peer ID of the local node.
    /// * `max_providers_per_cid` – bucket capacity per CID.
    pub fn new(local_peer_id: String, max_providers_per_cid: usize) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            max_providers_per_cid,
            local_peer_id,
            expired_evicted: RwLock::new(0),
        }
    }

    /// Create a new routing table with [`DEFAULT_MAX_PROVIDERS`].
    pub fn with_defaults(local_peer_id: String) -> Self {
        Self::new(local_peer_id, DEFAULT_MAX_PROVIDERS)
    }

    // ── Mutations ────────────────────────────────────────────────────────────

    /// Register `entry` as a provider for `cid`.
    ///
    /// # Errors
    ///
    /// * [`RoutingError::DuplicateProvider`] – the peer is already listed.
    /// * [`RoutingError::CapacityExceeded`] – the bucket is full.
    pub fn add_provider(&self, cid: &str, entry: RoutingEntry) -> Result<(), RoutingError> {
        let mut map = self.entries.write();
        let bucket = map.entry(cid.to_string()).or_default();

        // Duplicate check.
        if bucket.iter().any(|e| e.peer_id == entry.peer_id) {
            return Err(RoutingError::DuplicateProvider {
                cid: cid.to_string(),
                peer_id: entry.peer_id,
            });
        }

        // Capacity check.
        if bucket.len() >= self.max_providers_per_cid {
            return Err(RoutingError::CapacityExceeded {
                cid: cid.to_string(),
                max: self.max_providers_per_cid,
            });
        }

        bucket.push(entry);
        Ok(())
    }

    /// Remove the provider identified by `peer_id` from the bucket for `cid`.
    ///
    /// This is a no-op if neither the CID nor the peer is found.
    pub fn remove_provider(&self, cid: &str, peer_id: &str) {
        let mut map = self.entries.write();
        if let Some(bucket) = map.get_mut(cid) {
            bucket.retain(|e| e.peer_id != peer_id);
            if bucket.is_empty() {
                map.remove(cid);
            }
        }
    }

    /// Evict all entries whose TTL has elapsed relative to `now`.
    ///
    /// Empty buckets are removed, and the internal eviction counter is updated.
    pub fn evict_expired(&self, now: Instant) {
        let mut map = self.entries.write();
        let mut total_evicted: u64 = 0;

        map.retain(|_cid, bucket| {
            let before = bucket.len();
            bucket.retain(|e| !e.is_expired(now));
            total_evicted += (before - bucket.len()) as u64;
            !bucket.is_empty()
        });

        if total_evicted > 0 {
            let mut counter = self.expired_evicted.write();
            *counter = counter.saturating_add(total_evicted);
        }
    }

    // ── Queries ──────────────────────────────────────────────────────────────

    /// Return all providers for `cid` sorted by affinity score (highest first).
    pub fn get_providers(&self, cid: &str) -> Vec<RoutingEntry> {
        let map = self.entries.read();
        let Some(bucket) = map.get(cid) else {
            return Vec::new();
        };
        let mut sorted = bucket.clone();
        sorted.sort_by(|a, b| {
            b.affinity_score()
                .partial_cmp(&a.affinity_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    /// Return the single provider with the highest affinity score, if any.
    pub fn best_provider(&self, cid: &str) -> Option<RoutingEntry> {
        let map = self.entries.read();
        let bucket = map.get(cid)?;
        bucket
            .iter()
            .max_by(|a, b| {
                a.affinity_score()
                    .partial_cmp(&b.affinity_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
    }

    /// Return the number of distinct CIDs tracked.
    pub fn cid_count(&self) -> usize {
        self.entries.read().len()
    }

    /// Return the total number of provider entries across all CIDs.
    pub fn provider_count(&self) -> usize {
        self.entries.read().values().map(Vec::len).sum()
    }

    /// Return a snapshot of routing table statistics.
    pub fn stats(&self) -> RoutingTableStats {
        let map = self.entries.read();
        let cid_count = map.len();
        let provider_count: usize = map.values().map(Vec::len).sum();
        let expired_evicted = *self.expired_evicted.read();
        RoutingTableStats {
            cid_count,
            provider_count,
            expired_evicted,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn make_table() -> ContentRoutingTable {
        ContentRoutingTable::new("local-peer".to_string(), 5)
    }

    fn make_entry(peer: &str) -> RoutingEntry {
        RoutingEntry::new(
            peer.to_string(),
            vec![format!("/ip4/1.2.3.4/tcp/400{peer}")],
        )
    }

    fn make_entry_with_latency(peer: &str, latency_ms: u64) -> RoutingEntry {
        RoutingEntry::with_latency(
            peer.to_string(),
            vec![format!("/ip4/1.2.3.4/tcp/4001")],
            latency_ms,
            DEFAULT_ENTRY_TTL,
        )
    }

    // ── 1. Add and retrieve providers ────────────────────────────────────────

    #[test]
    fn test_add_and_retrieve_providers() {
        let table = make_table();
        let cid = "QmTest1";

        table
            .add_provider(cid, make_entry("peer-a"))
            .expect("test: add peer-a provider");
        table
            .add_provider(cid, make_entry("peer-b"))
            .expect("test: add peer-b provider");

        let providers = table.get_providers(cid);
        assert_eq!(providers.len(), 2);

        let ids: Vec<&str> = providers.iter().map(|e| e.peer_id.as_str()).collect();
        assert!(ids.contains(&"peer-a"));
        assert!(ids.contains(&"peer-b"));
    }

    // ── 2. Empty result for unknown CID ──────────────────────────────────────

    #[test]
    fn test_get_providers_unknown_cid() {
        let table = make_table();
        assert!(table.get_providers("QmUnknown").is_empty());
    }

    // ── 3. Duplicate provider rejection ──────────────────────────────────────

    #[test]
    fn test_duplicate_provider_rejected() {
        let table = make_table();
        let cid = "QmDup";

        table
            .add_provider(cid, make_entry("peer-x"))
            .expect("test: add peer-x provider first time");
        let err = table.add_provider(cid, make_entry("peer-x")).unwrap_err();

        match err {
            RoutingError::DuplicateProvider { cid: c, peer_id: p } => {
                assert_eq!(c, cid);
                assert_eq!(p, "peer-x");
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    // ── 4. Capacity limit enforcement ────────────────────────────────────────

    #[test]
    fn test_capacity_limit_enforced() {
        let table = ContentRoutingTable::new("local".to_string(), 3);
        let cid = "QmCap";

        for i in 0..3 {
            table
                .add_provider(cid, make_entry(&format!("peer-{i}")))
                .expect("test: add provider within capacity");
        }

        let err = table
            .add_provider(cid, make_entry("peer-overflow"))
            .unwrap_err();

        match err {
            RoutingError::CapacityExceeded { cid: c, max } => {
                assert_eq!(c, cid);
                assert_eq!(max, 3);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    // ── 5. Affinity score ordering (lower latency ranks higher) ──────────────

    #[test]
    fn test_affinity_ordering_by_latency() {
        let table = make_table();
        let cid = "QmLatency";

        // peer-fast has 10 ms; peer-slow has 500 ms
        table
            .add_provider(cid, make_entry_with_latency("peer-slow", 500))
            .expect("test: add peer-slow provider");
        table
            .add_provider(cid, make_entry_with_latency("peer-fast", 10))
            .expect("test: add peer-fast provider");

        let providers = table.get_providers(cid);
        assert_eq!(providers[0].peer_id, "peer-fast");
        assert_eq!(providers[1].peer_id, "peer-slow");
    }

    // ── 6. Affinity score values ──────────────────────────────────────────────

    #[test]
    fn test_affinity_score_values() {
        let entry_no_lat = RoutingEntry::new("p".to_string(), vec![]);
        // No latency, recently seen → 1.0 + 0.2 = 1.2
        assert!((entry_no_lat.affinity_score() - 1.2).abs() < 1e-9);

        let entry_lat = make_entry_with_latency("p2", 100);
        // latency 100 ms → −0.1, recently seen → +0.2  ⇒ 1.1
        assert!((entry_lat.affinity_score() - 1.1).abs() < 1e-9);
    }

    // ── 7. evict_expired removes stale entries ───────────────────────────────

    #[test]
    fn test_evict_expired_removes_stale() {
        let table = make_table();
        let cid = "QmEvict";

        // Add a fresh entry with a very short TTL.
        let short_ttl = Duration::from_millis(1);
        let stale = RoutingEntry::with_ttl("stale-peer".to_string(), vec![], short_ttl);
        let fresh = make_entry("fresh-peer");

        table
            .add_provider(cid, stale)
            .expect("test: add stale provider");
        table
            .add_provider(cid, fresh)
            .expect("test: add fresh provider");

        // Advance time beyond the short TTL.
        std::thread::sleep(Duration::from_millis(5));
        let now = Instant::now();

        table.evict_expired(now);

        let providers = table.get_providers(cid);
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].peer_id, "fresh-peer");
    }

    // ── 8. evict_expired removes empty CID buckets ───────────────────────────

    #[test]
    fn test_evict_expired_removes_empty_cid() {
        let table = make_table();
        let cid = "QmFullEvict";

        let short_ttl = Duration::from_millis(1);
        let e = RoutingEntry::with_ttl("only-peer".to_string(), vec![], short_ttl);
        table
            .add_provider(cid, e)
            .expect("test: add only-peer provider");

        std::thread::sleep(Duration::from_millis(5));
        table.evict_expired(Instant::now());

        assert_eq!(table.cid_count(), 0);
        assert_eq!(table.provider_count(), 0);
    }

    // ── 9. Stats eviction counter increments ─────────────────────────────────

    #[test]
    fn test_stats_expired_evicted_counter() {
        let table = make_table();
        let cid = "QmStats";

        let short_ttl = Duration::from_millis(1);
        for i in 0..3 {
            let e = RoutingEntry::with_ttl(format!("peer-{i}"), vec![], short_ttl);
            table
                .add_provider(cid, e)
                .expect("test: add short-ttl provider for stats");
        }

        std::thread::sleep(Duration::from_millis(5));
        table.evict_expired(Instant::now());

        let stats = table.stats();
        assert_eq!(stats.expired_evicted, 3);
        assert_eq!(stats.cid_count, 0);
        assert_eq!(stats.provider_count, 0);
    }

    // ── 10. best_provider returns highest affinity ───────────────────────────

    #[test]
    fn test_best_provider_highest_affinity() {
        let table = make_table();
        let cid = "QmBest";

        table
            .add_provider(cid, make_entry_with_latency("slow", 800))
            .expect("test: add slow provider");
        table
            .add_provider(cid, make_entry_with_latency("fast", 5))
            .expect("test: add fast provider");
        table
            .add_provider(cid, make_entry_with_latency("medium", 200))
            .expect("test: add medium provider");

        let best = table
            .best_provider(cid)
            .expect("should have a best provider");
        assert_eq!(best.peer_id, "fast");
    }

    // ── 11. best_provider returns None for unknown CID ───────────────────────

    #[test]
    fn test_best_provider_unknown_cid() {
        let table = make_table();
        assert!(table.best_provider("QmNotHere").is_none());
    }

    // ── 12. remove_provider works ────────────────────────────────────────────

    #[test]
    fn test_remove_provider() {
        let table = make_table();
        let cid = "QmRemove";

        table
            .add_provider(cid, make_entry("keep"))
            .expect("test: add keep provider");
        table
            .add_provider(cid, make_entry("gone"))
            .expect("test: add gone provider");

        table.remove_provider(cid, "gone");

        let providers = table.get_providers(cid);
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].peer_id, "keep");
    }

    // ── 13. remove_provider on absent peer is a no-op ────────────────────────

    #[test]
    fn test_remove_provider_absent_noop() {
        let table = make_table();
        let cid = "QmNoop";
        table
            .add_provider(cid, make_entry("only"))
            .expect("test: add only provider");
        // Should not panic or error.
        table.remove_provider(cid, "nonexistent");
        assert_eq!(table.provider_count(), 1);
    }

    // ── 14. remove_provider removes CID bucket when last entry is gone ────────

    #[test]
    fn test_remove_provider_cleans_up_empty_bucket() {
        let table = make_table();
        let cid = "QmCleanup";
        table
            .add_provider(cid, make_entry("last"))
            .expect("test: add last provider");
        table.remove_provider(cid, "last");
        assert_eq!(table.cid_count(), 0);
    }

    // ── 15. cid_count and provider_count correctness ─────────────────────────

    #[test]
    fn test_counts_correctness() {
        let table = make_table();

        table
            .add_provider("Qm1", make_entry("a"))
            .expect("test: add provider a to Qm1");
        table
            .add_provider("Qm1", make_entry("b"))
            .expect("test: add provider b to Qm1");
        table
            .add_provider("Qm2", make_entry("c"))
            .expect("test: add provider c to Qm2");

        assert_eq!(table.cid_count(), 2);
        assert_eq!(table.provider_count(), 3);

        let stats = table.stats();
        assert_eq!(stats.cid_count, 2);
        assert_eq!(stats.provider_count, 3);
        assert_eq!(stats.expired_evicted, 0);
    }

    // ── 16. is_expired ───────────────────────────────────────────────────────

    #[test]
    fn test_is_expired() {
        let short = Duration::from_millis(1);
        let entry = RoutingEntry::with_ttl("p".to_string(), vec![], short);

        // Not expired yet (created just now, check at the same instant).
        let just_now = Instant::now();
        // TTL is 1 ms so it is borderline; give it a small margin.
        assert!(!entry.is_expired(
            just_now
                .checked_sub(Duration::from_nanos(1))
                .unwrap_or(just_now)
        ));

        std::thread::sleep(Duration::from_millis(5));
        assert!(entry.is_expired(Instant::now()));
    }

    // ── 17. Evict across multiple CIDs ───────────────────────────────────────

    #[test]
    fn test_evict_across_multiple_cids() {
        let table = make_table();
        let short = Duration::from_millis(1);

        for cid in &["QmA", "QmB", "QmC"] {
            let e = RoutingEntry::with_ttl(format!("peer-{cid}"), vec![], short);
            table
                .add_provider(cid, e)
                .expect("test: add short-ttl provider across multi-cid");
        }
        // Also add one long-lived entry.
        table
            .add_provider("QmD", make_entry("long-lived"))
            .expect("test: add long-lived provider");

        std::thread::sleep(Duration::from_millis(5));
        table.evict_expired(Instant::now());

        assert_eq!(table.cid_count(), 1);
        assert_eq!(table.provider_count(), 1);
        let stats = table.stats();
        assert_eq!(stats.expired_evicted, 3);
    }
}
