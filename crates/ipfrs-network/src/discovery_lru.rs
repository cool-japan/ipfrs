//! LRU Peer Discovery Cache
//!
//! Provides a tick-based LRU cache for recently discovered peer addresses with TTL expiry.
//! This complements the existing `discovery_cache` module by adding LRU eviction semantics
//! and tick-based expiration suitable for high-throughput peer discovery scenarios.

/// A single cached peer entry.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Peer identifier (libp2p PeerId as string).
    pub peer_id: String,
    /// Known multiaddrs for this peer.
    pub addresses: Vec<String>,
    /// Tick at which this entry was last inserted or refreshed.
    pub discovered_tick: u64,
    /// Number of ticks before this entry expires.
    pub ttl_ticks: u64,
    /// Discovery source label, e.g. "dht", "mdns", "pex".
    pub source: String,
    /// Number of times this entry has been accessed via `get`.
    pub access_count: u64,
}

impl CacheEntry {
    /// Returns `true` if the entry has expired at the given tick.
    fn is_expired(&self, current_tick: u64) -> bool {
        current_tick >= self.discovered_tick.saturating_add(self.ttl_ticks)
    }
}

/// Configuration for [`LruPeerDiscoveryCache`].
#[derive(Debug, Clone)]
pub struct DiscoveryCacheConfig {
    /// Maximum number of entries the cache may hold.
    pub max_entries: usize,
    /// Default TTL (in ticks) applied when no explicit TTL is supplied.
    pub default_ttl_ticks: u64,
}

impl Default for DiscoveryCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 500,
            default_ttl_ticks: 300,
        }
    }
}

/// Snapshot of cache statistics.
#[derive(Debug, Clone)]
pub struct LruDiscoveryCacheStats {
    /// Number of entries currently in the cache.
    pub entry_count: usize,
    /// Total cache hits since creation.
    pub hits: u64,
    /// Total cache misses since creation.
    pub misses: u64,
    /// Hit rate as `hits / (hits + misses)`, or 0.0 if no lookups.
    pub hit_rate: f64,
}

/// LRU cache for recently discovered peer addresses with tick-based TTL.
///
/// Entries are stored in a `Vec` ordered by last access time — the *front*
/// (index 0) is the **least recently used** and the *back* is the most recently
/// used. When capacity is exceeded, the LRU entry (front) is evicted.
#[derive(Debug)]
pub struct LruPeerDiscoveryCache {
    config: DiscoveryCacheConfig,
    /// Ordered by last access: index 0 = LRU, last index = MRU.
    entries: Vec<CacheEntry>,
    current_tick: u64,
    hits: u64,
    misses: u64,
}

impl LruPeerDiscoveryCache {
    /// Create a new cache with the given configuration.
    pub fn new(config: DiscoveryCacheConfig) -> Self {
        Self {
            config,
            entries: Vec::new(),
            current_tick: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Insert or update a peer entry.
    ///
    /// If the peer already exists its addresses, source, and TTL are updated and
    /// the entry is moved to the back (MRU position). If the cache is at capacity,
    /// the LRU entry (front) is evicted first.
    pub fn put(&mut self, peer_id: &str, addresses: Vec<String>, source: &str, ttl: Option<u64>) {
        let ttl_ticks = ttl.unwrap_or(self.config.default_ttl_ticks);

        // Check if already present — update in place then move to back.
        if let Some(pos) = self.entries.iter().position(|e| e.peer_id == peer_id) {
            let mut entry = self.entries.remove(pos);
            entry.addresses = addresses;
            entry.source = source.to_string();
            entry.discovered_tick = self.current_tick;
            entry.ttl_ticks = ttl_ticks;
            self.entries.push(entry);
            return;
        }

        // Evict LRU if at capacity.
        if self.entries.len() >= self.config.max_entries {
            self.entries.remove(0);
        }

        self.entries.push(CacheEntry {
            peer_id: peer_id.to_string(),
            addresses,
            discovered_tick: self.current_tick,
            ttl_ticks,
            source: source.to_string(),
            access_count: 0,
        });
    }

    /// Retrieve a peer entry, moving it to the MRU position.
    ///
    /// Returns `None` if the peer is not present or has expired (expired entries
    /// are removed on access). Increments `access_count` and updates hit/miss
    /// counters.
    pub fn get(&mut self, peer_id: &str) -> Option<&CacheEntry> {
        if let Some(pos) = self.entries.iter().position(|e| e.peer_id == peer_id) {
            if self.entries[pos].is_expired(self.current_tick) {
                self.entries.remove(pos);
                self.misses += 1;
                return None;
            }

            // Move to back (MRU).
            let mut entry = self.entries.remove(pos);
            entry.access_count += 1;
            self.entries.push(entry);
            self.hits += 1;

            // Return reference to the entry we just pushed (last element).
            self.entries.last()
        } else {
            self.misses += 1;
            None
        }
    }

    /// Remove a peer entry. Returns `true` if the entry existed.
    pub fn remove(&mut self, peer_id: &str) -> bool {
        if let Some(pos) = self.entries.iter().position(|e| e.peer_id == peer_id) {
            self.entries.remove(pos);
            true
        } else {
            false
        }
    }

    /// Check whether a peer is in the cache **without** updating LRU order or
    /// access counters.
    pub fn contains(&self, peer_id: &str) -> bool {
        self.entries.iter().any(|e| e.peer_id == peer_id)
    }

    /// Advance the internal tick counter by one and remove all expired entries.
    pub fn tick_cleanup(&mut self) {
        self.current_tick += 1;
        self.entries.retain(|e| !e.is_expired(self.current_tick));
    }

    /// Number of entries currently in the cache.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Hit rate as `hits / (hits + misses)`, or 0.0 when no lookups have been
    /// performed.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }

    /// Return references to all entries that match the given discovery source.
    pub fn entries_by_source(&self, source: &str) -> Vec<&CacheEntry> {
        self.entries.iter().filter(|e| e.source == source).collect()
    }

    /// Snapshot of current cache statistics.
    pub fn stats(&self) -> LruDiscoveryCacheStats {
        LruDiscoveryCacheStats {
            entry_count: self.entries.len(),
            hits: self.hits,
            misses: self.misses,
            hit_rate: self.hit_rate(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cache() -> LruPeerDiscoveryCache {
        LruPeerDiscoveryCache::new(DiscoveryCacheConfig::default())
    }

    fn small_cache(cap: usize) -> LruPeerDiscoveryCache {
        LruPeerDiscoveryCache::new(DiscoveryCacheConfig {
            max_entries: cap,
            default_ttl_ticks: 10,
        })
    }

    // -- basic put/get roundtrip --

    #[test]
    fn put_get_roundtrip() {
        let mut c = default_cache();
        c.put("peer1", vec!["addr1".into()], "dht", None);
        let e = c.get("peer1").expect("should exist");
        assert_eq!(e.peer_id, "peer1");
        assert_eq!(e.addresses, vec!["addr1".to_string()]);
        assert_eq!(e.source, "dht");
    }

    #[test]
    fn get_nonexistent_returns_none() {
        let mut c = default_cache();
        assert!(c.get("missing").is_none());
    }

    #[test]
    fn put_updates_existing_entry() {
        let mut c = default_cache();
        c.put("peer1", vec!["a1".into()], "dht", None);
        c.put("peer1", vec!["a2".into(), "a3".into()], "mdns", Some(50));
        let e = c.get("peer1").expect("should exist");
        assert_eq!(e.addresses, vec!["a2".to_string(), "a3".to_string()]);
        assert_eq!(e.source, "mdns");
        assert_eq!(e.ttl_ticks, 50);
    }

    // -- LRU eviction order --

    #[test]
    fn lru_eviction_order() {
        let mut c = small_cache(3);
        c.put("p1", vec![], "dht", None);
        c.put("p2", vec![], "dht", None);
        c.put("p3", vec![], "dht", None);

        // Cache full. Insert p4 — should evict p1 (LRU).
        c.put("p4", vec![], "dht", None);
        assert!(!c.contains("p1"));
        assert!(c.contains("p2"));
        assert!(c.contains("p3"));
        assert!(c.contains("p4"));
    }

    #[test]
    fn access_promotes_entry_preventing_eviction() {
        let mut c = small_cache(3);
        c.put("p1", vec![], "dht", None);
        c.put("p2", vec![], "dht", None);
        c.put("p3", vec![], "dht", None);

        // Access p1 — moves it to MRU. LRU is now p2.
        let _ = c.get("p1");

        c.put("p4", vec![], "dht", None);
        assert!(c.contains("p1"), "p1 should still be present after access");
        assert!(!c.contains("p2"), "p2 should have been evicted as LRU");
    }

    #[test]
    fn put_existing_moves_to_mru() {
        let mut c = small_cache(3);
        c.put("p1", vec![], "dht", None);
        c.put("p2", vec![], "dht", None);
        c.put("p3", vec![], "dht", None);

        // Re-put p1 — moves to MRU. LRU becomes p2.
        c.put("p1", vec!["new".into()], "dht", None);

        c.put("p4", vec![], "dht", None);
        assert!(c.contains("p1"));
        assert!(!c.contains("p2"));
    }

    // -- TTL expiry --

    #[test]
    fn ttl_expiry_on_get() {
        let mut c = small_cache(10);
        c.put("p1", vec![], "dht", Some(3));

        // Advance 3 ticks — entry should expire.
        for _ in 0..3 {
            c.tick_cleanup();
        }
        assert!(c.get("p1").is_none());
    }

    #[test]
    fn ttl_not_expired_within_window() {
        let mut c = small_cache(10);
        c.put("p1", vec![], "dht", Some(5));
        c.tick_cleanup(); // tick 1
        c.tick_cleanup(); // tick 2
        assert!(c.get("p1").is_some());
    }

    #[test]
    fn tick_cleanup_removes_expired() {
        let mut c = small_cache(10);
        c.put("p1", vec![], "dht", Some(2));
        c.put("p2", vec![], "dht", Some(100));

        c.tick_cleanup(); // tick 1
        c.tick_cleanup(); // tick 2 — p1 expires
        assert_eq!(c.entry_count(), 1);
        assert!(c.contains("p2"));
        assert!(!c.contains("p1"));
    }

    // -- access_count --

    #[test]
    fn access_count_increments() {
        let mut c = default_cache();
        c.put("p1", vec![], "dht", None);
        for _ in 0..5 {
            let _ = c.get("p1");
        }
        let e = c.get("p1").expect("should exist");
        assert_eq!(e.access_count, 6); // 5 + 1 from last get
    }

    #[test]
    fn access_count_starts_at_zero() {
        let mut c = default_cache();
        c.put("p1", vec![], "dht", None);
        // Peek via contains (no access_count bump).
        assert!(c.contains("p1"));
        let e = c.get("p1").expect("should exist");
        assert_eq!(e.access_count, 1); // first get
    }

    // -- hit / miss tracking --

    #[test]
    fn hit_miss_tracking() {
        let mut c = default_cache();
        c.put("p1", vec![], "dht", None);
        let _ = c.get("p1"); // hit
        let _ = c.get("p1"); // hit
        let _ = c.get("missing"); // miss
        let s = c.stats();
        assert_eq!(s.hits, 2);
        assert_eq!(s.misses, 1);
    }

    #[test]
    fn hit_rate_calculation() {
        let mut c = default_cache();
        c.put("p1", vec![], "dht", None);
        let _ = c.get("p1"); // hit
        let _ = c.get("missing"); // miss
        let rate = c.hit_rate();
        assert!((rate - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn hit_rate_empty_is_zero() {
        let c = default_cache();
        assert!((c.hit_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn expired_entry_counts_as_miss() {
        let mut c = small_cache(10);
        c.put("p1", vec![], "dht", Some(1));
        c.tick_cleanup(); // tick 1 — expires
        let _ = c.get("p1"); // miss (expired)
        assert_eq!(c.stats().misses, 1);
        assert_eq!(c.stats().hits, 0);
    }

    // -- contains --

    #[test]
    fn contains_does_not_update_order() {
        let mut c = small_cache(3);
        c.put("p1", vec![], "dht", None);
        c.put("p2", vec![], "dht", None);
        c.put("p3", vec![], "dht", None);

        // contains on p1 should NOT promote it.
        assert!(c.contains("p1"));

        // Insert p4 — LRU is still p1.
        c.put("p4", vec![], "dht", None);
        assert!(!c.contains("p1"), "p1 should have been evicted");
    }

    #[test]
    fn contains_returns_false_for_missing() {
        let c = default_cache();
        assert!(!c.contains("nope"));
    }

    // -- entries_by_source --

    #[test]
    fn entries_by_source_filtering() {
        let mut c = default_cache();
        c.put("p1", vec![], "dht", None);
        c.put("p2", vec![], "mdns", None);
        c.put("p3", vec![], "dht", None);
        c.put("p4", vec![], "pex", None);

        let dht_entries = c.entries_by_source("dht");
        assert_eq!(dht_entries.len(), 2);
        assert!(dht_entries.iter().all(|e| e.source == "dht"));
    }

    #[test]
    fn entries_by_source_empty_result() {
        let mut c = default_cache();
        c.put("p1", vec![], "dht", None);
        assert!(c.entries_by_source("relay").is_empty());
    }

    // -- remove --

    #[test]
    fn remove_existing() {
        let mut c = default_cache();
        c.put("p1", vec![], "dht", None);
        assert!(c.remove("p1"));
        assert!(!c.contains("p1"));
        assert_eq!(c.entry_count(), 0);
    }

    #[test]
    fn remove_nonexistent() {
        let mut c = default_cache();
        assert!(!c.remove("ghost"));
    }

    // -- empty cache --

    #[test]
    fn empty_cache_entry_count() {
        let c = default_cache();
        assert_eq!(c.entry_count(), 0);
    }

    #[test]
    fn empty_cache_stats() {
        let c = default_cache();
        let s = c.stats();
        assert_eq!(s.entry_count, 0);
        assert_eq!(s.hits, 0);
        assert_eq!(s.misses, 0);
        assert!((s.hit_rate - 0.0).abs() < f64::EPSILON);
    }

    // -- capacity enforcement --

    #[test]
    fn capacity_enforcement() {
        let mut c = small_cache(5);
        for i in 0..10 {
            c.put(&format!("p{i}"), vec![], "dht", None);
        }
        assert_eq!(c.entry_count(), 5);
        // Only the last 5 should remain.
        for i in 0..5 {
            assert!(!c.contains(&format!("p{i}")));
        }
        for i in 5..10 {
            assert!(c.contains(&format!("p{i}")));
        }
    }

    #[test]
    fn capacity_one() {
        let mut c = small_cache(1);
        c.put("p1", vec![], "dht", None);
        c.put("p2", vec![], "dht", None);
        assert_eq!(c.entry_count(), 1);
        assert!(c.contains("p2"));
        assert!(!c.contains("p1"));
    }

    // -- default config --

    #[test]
    fn default_config_values() {
        let cfg = DiscoveryCacheConfig::default();
        assert_eq!(cfg.max_entries, 500);
        assert_eq!(cfg.default_ttl_ticks, 300);
    }

    // -- stats snapshot --

    #[test]
    fn stats_reflects_current_state() {
        let mut c = small_cache(10);
        c.put("p1", vec![], "dht", None);
        c.put("p2", vec![], "dht", None);
        let _ = c.get("p1"); // hit
        let _ = c.get("nope"); // miss
        let s = c.stats();
        assert_eq!(s.entry_count, 2);
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 1);
        assert!((s.hit_rate - 0.5).abs() < f64::EPSILON);
    }

    // -- additional edge cases --

    #[test]
    fn tick_cleanup_on_empty_cache() {
        let mut c = default_cache();
        c.tick_cleanup();
        assert_eq!(c.entry_count(), 0);
    }

    #[test]
    fn multiple_tick_cleanups() {
        let mut c = small_cache(10);
        c.put("p1", vec![], "dht", Some(3));
        c.put("p2", vec![], "dht", Some(5));

        for _ in 0..3 {
            c.tick_cleanup();
        }
        assert!(!c.contains("p1"));
        assert!(c.contains("p2"));

        for _ in 0..2 {
            c.tick_cleanup();
        }
        assert!(!c.contains("p2"));
        assert_eq!(c.entry_count(), 0);
    }

    #[test]
    fn put_after_ttl_refresh_extends_lifetime() {
        let mut c = small_cache(10);
        c.put("p1", vec![], "dht", Some(3));
        c.tick_cleanup(); // tick 1
        c.tick_cleanup(); // tick 2

        // Re-put refreshes discovered_tick to current_tick (2).
        c.put("p1", vec!["new_addr".into()], "dht", Some(3));

        c.tick_cleanup(); // tick 3 — would have expired without refresh
        c.tick_cleanup(); // tick 4
        assert!(c.get("p1").is_some(), "re-put should have extended TTL");
    }

    #[test]
    fn hit_rate_all_hits() {
        let mut c = default_cache();
        c.put("p1", vec![], "dht", None);
        for _ in 0..10 {
            let _ = c.get("p1");
        }
        assert!((c.hit_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn hit_rate_all_misses() {
        let mut c = default_cache();
        for _ in 0..5 {
            let _ = c.get("ghost");
        }
        assert!((c.hit_rate() - 0.0).abs() < f64::EPSILON);
    }
}
