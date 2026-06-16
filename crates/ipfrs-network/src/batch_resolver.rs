//! Batch CID resolver and prefetch scheduler for DHT performance optimization
//!
//! This module provides:
//! - [`BatchCidResolver`]: batches multiple CID lookups to reduce DHT overhead
//! - [`PrefetchScheduler`]: predicts and schedules prefetch candidates based on access patterns

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ─── CachedResult ────────────────────────────────────────────────────────────

/// A resolved entry stored in the cache.
#[derive(Debug, Clone)]
pub struct CachedResult {
    /// Provider multiaddrs / peer IDs for this CID
    pub providers: Vec<String>,
    /// When the result was stored
    pub resolved_at: Instant,
    /// How long the entry is considered fresh
    pub ttl: Duration,
}

impl CachedResult {
    /// Returns `true` when the entry has expired relative to `now`.
    #[inline]
    pub fn is_expired(&self, now: Instant) -> bool {
        now.duration_since(self.resolved_at) >= self.ttl
    }
}

// ─── PendingLookup ────────────────────────────────────────────────────────────

/// A CID that has been queued but not yet resolved.
#[derive(Debug, Clone)]
pub struct PendingLookup {
    /// The CID string to look up
    pub cid: String,
    /// When the lookup was enqueued
    pub queued_at: Instant,
}

// ─── LookupHandle ────────────────────────────────────────────────────────────

/// A lightweight handle returned to the caller when they queue a CID lookup.
///
/// This is a plain value type — callers can use it to track which CID they
/// queued and when, without needing an async channel.
#[derive(Debug, Clone)]
pub struct LookupHandle {
    /// The CID string that was queued
    pub cid: String,
    /// When it was enqueued
    pub queued_at: Instant,
}

// ─── BatchResolverStats ───────────────────────────────────────────────────────

/// Atomic counters for [`BatchCidResolver`] activity.
#[derive(Debug, Default)]
pub struct BatchResolverStats {
    /// Total CIDs added via [`BatchCidResolver::queue_lookup`]
    pub total_queued: AtomicU64,
    /// Total CIDs for which results have been recorded
    pub total_resolved: AtomicU64,
    /// Total cache hits returned by [`BatchCidResolver::get_cached`]
    pub total_cache_hits: AtomicU64,
    /// Total calls to [`BatchCidResolver::drain_batch`] that returned ≥1 item
    pub total_batches_drained: AtomicU64,
    /// Total cache entries removed by [`BatchCidResolver::evict_expired`]
    pub total_evictions: AtomicU64,
}

/// A point-in-time snapshot of [`BatchResolverStats`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchResolverStatsSnapshot {
    /// Total CIDs queued
    pub total_queued: u64,
    /// Total CIDs resolved
    pub total_resolved: u64,
    /// Total cache hits
    pub total_cache_hits: u64,
    /// Total non-empty drain operations
    pub total_batches_drained: u64,
    /// Total cache entries evicted
    pub total_evictions: u64,
}

impl BatchResolverStats {
    /// Take an instantaneous snapshot of all counters.
    pub fn snapshot(&self) -> BatchResolverStatsSnapshot {
        BatchResolverStatsSnapshot {
            total_queued: self.total_queued.load(Ordering::Relaxed),
            total_resolved: self.total_resolved.load(Ordering::Relaxed),
            total_cache_hits: self.total_cache_hits.load(Ordering::Relaxed),
            total_batches_drained: self.total_batches_drained.load(Ordering::Relaxed),
            total_evictions: self.total_evictions.load(Ordering::Relaxed),
        }
    }
}

// ─── BatchCidResolver ─────────────────────────────────────────────────────────

/// Batches CID provider lookups and caches results with TTL-based expiry.
///
/// # Thread safety
///
/// All internal state is guarded by [`Mutex`]; the struct itself can be shared
/// behind an [`Arc`] without additional synchronization.
pub struct BatchCidResolver {
    /// CIDs waiting to be resolved, in insertion order
    pending: Mutex<Vec<PendingLookup>>,
    /// Resolved CID → provider list, keyed by CID string
    cache: Mutex<HashMap<String, CachedResult>>,
    /// Time-to-live applied to newly stored cache entries
    pub cache_ttl: Duration,
    /// Maximum number of pending items consumed per `drain_batch` call
    pub max_batch_size: usize,
    /// Operational statistics
    pub stats: Arc<BatchResolverStats>,
}

impl BatchCidResolver {
    /// Create a new resolver with default TTL (5 min) and batch size (32).
    pub fn new() -> Self {
        Self::with_config(Duration::from_secs(300), 32)
    }

    /// Create a resolver with custom TTL and batch size.
    pub fn with_config(cache_ttl: Duration, max_batch_size: usize) -> Self {
        Self {
            pending: Mutex::new(Vec::new()),
            cache: Mutex::new(HashMap::new()),
            cache_ttl,
            max_batch_size,
            stats: Arc::new(BatchResolverStats::default()),
        }
    }

    /// Queue a CID for resolution.
    ///
    /// Returns a [`LookupHandle`] the caller can use to track the request.
    /// If the CID is already in the cache the item is still queued; callers
    /// should call `get_cached` first if they want to avoid duplicate work.
    pub fn queue_lookup(&self, cid: &str) -> LookupHandle {
        let now = Instant::now();
        let lookup = PendingLookup {
            cid: cid.to_owned(),
            queued_at: now,
        };
        {
            let mut guard = self.pending.lock().unwrap_or_else(|e| e.into_inner());
            guard.push(lookup);
        }
        self.stats.total_queued.fetch_add(1, Ordering::Relaxed);
        LookupHandle {
            cid: cid.to_owned(),
            queued_at: now,
        }
    }

    /// Atomically drain up to `max_batch_size` pending lookups.
    ///
    /// Returns the drained items in FIFO order.  Increments
    /// `total_batches_drained` only when at least one item is returned.
    pub fn drain_batch(&self) -> Vec<PendingLookup> {
        let mut guard = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_empty() {
            return Vec::new();
        }
        let n = guard.len().min(self.max_batch_size);
        // Drain from the front to preserve FIFO order
        let drained: Vec<PendingLookup> = guard.drain(..n).collect();
        drop(guard);
        if !drained.is_empty() {
            self.stats
                .total_batches_drained
                .fetch_add(1, Ordering::Relaxed);
        }
        drained
    }

    /// Store a resolved provider list for `cid`.
    ///
    /// Any existing entry (including unexpired ones) is overwritten.
    pub fn record_result(&self, cid: &str, providers: Vec<String>) {
        let entry = CachedResult {
            providers,
            resolved_at: Instant::now(),
            ttl: self.cache_ttl,
        };
        {
            let mut guard = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            guard.insert(cid.to_owned(), entry);
        }
        self.stats.total_resolved.fetch_add(1, Ordering::Relaxed);
    }

    /// Return the cached provider list for `cid` if it has not expired.
    pub fn get_cached(&self, cid: &str) -> Option<Vec<String>> {
        let now = Instant::now();
        let guard = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        let entry = guard.get(cid)?;
        if entry.is_expired(now) {
            return None;
        }
        let providers = entry.providers.clone();
        drop(guard);
        self.stats.total_cache_hits.fetch_add(1, Ordering::Relaxed);
        Some(providers)
    }

    /// Remove all cache entries that have exceeded their TTL.
    pub fn evict_expired(&self) {
        let now = Instant::now();
        let mut guard = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        let before = guard.len();
        guard.retain(|_, v| !v.is_expired(now));
        let after = guard.len();
        let evicted = (before - after) as u64;
        drop(guard);
        if evicted > 0 {
            self.stats
                .total_evictions
                .fetch_add(evicted, Ordering::Relaxed);
        }
    }

    /// Return the number of CIDs currently in the pending queue.
    pub fn pending_count(&self) -> usize {
        let guard = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        guard.len()
    }

    /// Return the number of entries currently held in the cache
    /// (expired entries are counted until `evict_expired` is called).
    pub fn cache_size(&self) -> usize {
        let guard = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        guard.len()
    }
}

impl Default for BatchCidResolver {
    fn default() -> Self {
        Self::new()
    }
}

// ─── PrefetchScheduler ────────────────────────────────────────────────────────

/// Tracks CID access patterns and suggests prefetch candidates.
///
/// A sliding window of the last 256 accesses is maintained.  Within each
/// access event the scheduler updates co-access counts for up to the
/// 4 most-recent *other* CIDs (i.e., pairs within the last 5 accesses).
pub struct PrefetchScheduler {
    /// Rolling log of (cid, timestamp) — newest at the back
    access_log: Mutex<VecDeque<(String, Instant)>>,
    /// co_access[a][b] = number of times b was accessed within 5 steps of a
    co_access: Mutex<HashMap<String, HashMap<String, u32>>>,
}

impl PrefetchScheduler {
    /// Create a new scheduler.
    pub fn new() -> Self {
        Self {
            access_log: Mutex::new(VecDeque::new()),
            co_access: Mutex::new(HashMap::new()),
        }
    }

    /// Record that `cid` was accessed.
    ///
    /// Appends to the rolling log (trimmed to 256 entries) and updates
    /// co-access counts for all pairs formed by the current access and the
    /// preceding 4 entries (window of 5).
    pub fn record_access(&self, cid: &str) {
        let now = Instant::now();
        let mut log = self.access_log.lock().unwrap_or_else(|e| e.into_inner());

        // Collect the (up to 4) CIDs that form a window with the new one
        let window_size = 4.min(log.len());
        let recent: Vec<String> = log
            .iter()
            .rev()
            .take(window_size)
            .map(|(c, _)| c.clone())
            .collect();

        // Append the new entry
        log.push_back((cid.to_owned(), now));
        // Trim to rolling window of 256
        while log.len() > 256 {
            log.pop_front();
        }
        drop(log);

        // Update co-access counts
        if !recent.is_empty() {
            let mut co = self.co_access.lock().unwrap_or_else(|e| e.into_inner());
            for peer in &recent {
                // cid co-accessed with peer
                co.entry(cid.to_owned())
                    .or_default()
                    .entry(peer.clone())
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
                // peer co-accessed with cid
                co.entry(peer.clone())
                    .or_default()
                    .entry(cid.to_owned())
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
            }
        }
    }

    /// Return the top `top_n` CIDs most frequently co-accessed with `cid`,
    /// sorted by co-access count descending.
    pub fn prefetch_candidates(&self, cid: &str, top_n: usize) -> Vec<String> {
        if top_n == 0 {
            return Vec::new();
        }
        let co = self.co_access.lock().unwrap_or_else(|e| e.into_inner());
        let peers = match co.get(cid) {
            Some(map) => map,
            None => return Vec::new(),
        };
        let mut sorted: Vec<(String, u32)> = peers.iter().map(|(k, &v)| (k.clone(), v)).collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        sorted.into_iter().take(top_n).map(|(k, _)| k).collect()
    }

    /// Return the total number of entries in the access log.
    pub fn access_count(&self) -> usize {
        let log = self.access_log.lock().unwrap_or_else(|e| e.into_inner());
        log.len()
    }

    /// Remove log entries older than `max_age`.
    ///
    /// Note: co-access counts are *not* adjusted — they represent historical
    /// signal that remains valid even after pruning old log entries.
    pub fn prune_log(&self, max_age: Duration) {
        let now = Instant::now();
        let mut log = self.access_log.lock().unwrap_or_else(|e| e.into_inner());
        log.retain(|(_, ts)| now.duration_since(*ts) < max_age);
    }
}

impl Default for PrefetchScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    // Helper: build a resolver with a very short TTL for expiry tests
    fn short_ttl_resolver() -> BatchCidResolver {
        BatchCidResolver::with_config(Duration::from_millis(50), 32)
    }

    // ── BatchCidResolver tests ────────────────────────────────────────────────

    #[test]
    fn test_queue_and_pending_count() {
        let r = BatchCidResolver::new();
        assert_eq!(r.pending_count(), 0);
        r.queue_lookup("QmA");
        r.queue_lookup("QmB");
        assert_eq!(r.pending_count(), 2);
        assert_eq!(r.stats.snapshot().total_queued, 2);
    }

    #[test]
    fn test_drain_batch_respects_max_batch_size() {
        let r = BatchCidResolver::with_config(Duration::from_secs(300), 3);
        for i in 0..7u32 {
            r.queue_lookup(&format!("Qm{}", i));
        }
        let batch = r.drain_batch();
        assert_eq!(batch.len(), 3, "first drain should return max_batch_size");
        assert_eq!(r.pending_count(), 4, "remaining items still queued");
    }

    #[test]
    fn test_drain_batch_empty_returns_empty_vec() {
        let r = BatchCidResolver::new();
        let batch = r.drain_batch();
        assert!(batch.is_empty());
        // Empty drain must NOT increment total_batches_drained
        assert_eq!(r.stats.snapshot().total_batches_drained, 0);
    }

    #[test]
    fn test_drain_batch_increments_stat_on_nonempty() {
        let r = BatchCidResolver::new();
        r.queue_lookup("QmX");
        let _ = r.drain_batch();
        assert_eq!(r.stats.snapshot().total_batches_drained, 1);
    }

    #[test]
    fn test_drain_batch_preserves_fifo_order() {
        let r = BatchCidResolver::with_config(Duration::from_secs(300), 10);
        let cids = ["QmA", "QmB", "QmC", "QmD"];
        for c in &cids {
            r.queue_lookup(c);
        }
        let batch = r.drain_batch();
        let drained_cids: Vec<&str> = batch.iter().map(|p| p.cid.as_str()).collect();
        assert_eq!(drained_cids, cids);
    }

    #[test]
    fn test_cache_hit_returns_providers() {
        let r = BatchCidResolver::new();
        let providers = vec!["peer1".to_string(), "peer2".to_string()];
        r.record_result("QmFoo", providers.clone());
        let cached = r.get_cached("QmFoo");
        assert_eq!(cached, Some(providers));
        assert_eq!(r.stats.snapshot().total_cache_hits, 1);
    }

    #[test]
    fn test_cache_miss_returns_none() {
        let r = BatchCidResolver::new();
        assert!(r.get_cached("QmMissing").is_none());
        assert_eq!(r.stats.snapshot().total_cache_hits, 0);
    }

    #[test]
    fn test_expired_cache_entries_evicted() {
        let r = short_ttl_resolver();
        r.record_result("QmExpire", vec!["peer".to_string()]);
        assert_eq!(r.cache_size(), 1);

        // Wait for TTL to expire
        thread::sleep(Duration::from_millis(60));

        // get_cached should not return expired entry
        assert!(r.get_cached("QmExpire").is_none());

        // evict_expired should remove it
        r.evict_expired();
        assert_eq!(r.cache_size(), 0);
        assert_eq!(r.stats.snapshot().total_evictions, 1);
    }

    #[test]
    fn test_evict_expired_only_removes_stale_entries() {
        let r = short_ttl_resolver();
        r.record_result("QmOld", vec!["p1".to_string()]);
        thread::sleep(Duration::from_millis(60));
        // Add a fresh entry AFTER the old one expired
        r.record_result("QmNew", vec!["p2".to_string()]);
        r.evict_expired();
        assert_eq!(r.cache_size(), 1);
        assert!(r.get_cached("QmNew").is_some());
    }

    #[test]
    fn test_stats_accumulate_correctly() {
        let r = BatchCidResolver::new();
        r.queue_lookup("Qm1");
        r.queue_lookup("Qm2");
        r.drain_batch();
        r.record_result("Qm1", vec!["peer".to_string()]);
        r.get_cached("Qm1");
        let snap = r.stats.snapshot();
        assert_eq!(snap.total_queued, 2);
        assert_eq!(snap.total_resolved, 1);
        assert_eq!(snap.total_cache_hits, 1);
        assert_eq!(snap.total_batches_drained, 1);
    }

    // ── PrefetchScheduler tests ───────────────────────────────────────────────

    #[test]
    fn test_prefetch_records_co_access_patterns() {
        let s = PrefetchScheduler::new();
        s.record_access("A");
        s.record_access("B");
        // B should appear as a candidate for A (and vice versa)
        let candidates = s.prefetch_candidates("A", 5);
        assert!(candidates.contains(&"B".to_string()));
    }

    #[test]
    fn test_prefetch_candidates_sorted_by_frequency() {
        let s = PrefetchScheduler::new();
        // Build co-access between BASE→C many times, and BASE→B only once.
        // We avoid self-co-access by always interleaving with a neutral "NOISE" CID.
        //
        // Pattern (6 repetitions): NOISE, BASE, C   → each gives co(BASE,C) += 1
        for _ in 0..6 {
            s.record_access("NOISE");
            s.record_access("BASE");
            s.record_access("C");
        }
        // One occurrence of BASE near B
        s.record_access("NOISE2");
        s.record_access("BASE");
        s.record_access("B");

        let candidates = s.prefetch_candidates("BASE", 3);
        // Filter out NOISE entries — we only care about C vs B ranking
        let filtered: Vec<&str> = candidates
            .iter()
            .map(|s| s.as_str())
            .filter(|&c| c == "C" || c == "B")
            .collect();
        assert!(
            !filtered.is_empty(),
            "expected at least C or B in candidates"
        );
        // C should rank higher than B (more co-accesses)
        assert_eq!(filtered[0], "C", "C should be the top co-access candidate");
    }

    #[test]
    fn test_prefetch_candidates_empty_for_unknown_cid() {
        let s = PrefetchScheduler::new();
        s.record_access("X");
        let candidates = s.prefetch_candidates("UNKNOWN", 5);
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_prefetch_candidates_top_n_respected() {
        let s = PrefetchScheduler::new();
        // Create co-access relationships with many peers
        let base = "BASE";
        for i in 0..10u32 {
            s.record_access(base);
            s.record_access(&format!("PEER{}", i));
        }
        let candidates = s.prefetch_candidates(base, 3);
        assert!(candidates.len() <= 3);
    }

    #[test]
    fn test_prefetch_access_count() {
        let s = PrefetchScheduler::new();
        assert_eq!(s.access_count(), 0);
        s.record_access("A");
        s.record_access("B");
        s.record_access("C");
        assert_eq!(s.access_count(), 3);
    }

    #[test]
    fn test_prune_log_removes_old_entries() {
        let s = PrefetchScheduler::new();
        s.record_access("A");
        s.record_access("B");
        thread::sleep(Duration::from_millis(30));
        // Prune anything older than 20 ms — both entries should be removed
        s.prune_log(Duration::from_millis(20));
        assert_eq!(s.access_count(), 0);
    }

    #[test]
    fn test_prune_log_keeps_recent_entries() {
        let s = PrefetchScheduler::new();
        s.record_access("A");
        // Prune anything older than 1 second — entry is fresh, should survive
        s.prune_log(Duration::from_secs(1));
        assert_eq!(s.access_count(), 1);
    }

    #[test]
    fn test_access_log_capped_at_256() {
        let s = PrefetchScheduler::new();
        for i in 0..300u32 {
            s.record_access(&format!("Qm{}", i));
        }
        assert_eq!(s.access_count(), 256);
    }

    #[test]
    fn test_lookup_handle_fields() {
        let r = BatchCidResolver::new();
        let handle = r.queue_lookup("QmHandle");
        assert_eq!(handle.cid, "QmHandle");
        // queued_at should be very recent
        assert!(handle.queued_at.elapsed() < Duration::from_secs(1));
    }
}
