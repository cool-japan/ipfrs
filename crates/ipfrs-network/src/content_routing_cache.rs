//! Content Routing Cache — multi-tier DHT routing cache.
//!
//! Stores provider records, routing hints, and negative cache entries to reduce
//! DHT lookup latency. Organises data into three independent tiers with
//! configurable per-tier TTLs, per-CID provider limits, and global capacity
//! caps with oldest-CID eviction.

use std::collections::HashMap;

// ── Constants ──────────────────────────────────────────────────────────────

/// Default provider cache capacity (number of distinct CIDs).
pub const DEFAULT_MAX_PROVIDERS: usize = 10_000;
/// Default routing-hint cache capacity (number of distinct CIDs).
pub const DEFAULT_MAX_HINTS: usize = 5_000;
/// Default negative cache capacity (number of distinct CIDs).
pub const DEFAULT_MAX_NEGATIVE: usize = 2_000;
/// Default provider record TTL: 1 hour in milliseconds.
pub const DEFAULT_PROVIDER_TTL_MS: u64 = 3_600_000;
/// Default routing-hint TTL: 5 minutes in milliseconds.
pub const DEFAULT_HINT_TTL_MS: u64 = 300_000;
/// Default negative-entry TTL: 1 minute in milliseconds.
pub const DEFAULT_NEGATIVE_TTL_MS: u64 = 60_000;
/// Maximum provider records stored per CID.
const MAX_PROVIDERS_PER_CID: usize = 20;

// ── ProviderRecord ─────────────────────────────────────────────────────────

/// A single provider record: which peer provides a given CID.
#[derive(Debug, Clone, PartialEq)]
pub struct CrcProviderRecord {
    /// Content identifier (CID string).
    pub cid: String,
    /// Peer identifier (peer-id string).
    pub peer_id: String,
    /// Known multiaddresses for this peer.
    pub multiaddrs: Vec<String>,
    /// Millisecond timestamp when this record was last seen.
    pub last_provided: u64,
    /// Time-to-live in milliseconds.
    pub ttl_ms: u64,
}

impl CrcProviderRecord {
    /// Returns `true` when `now - last_provided > ttl_ms`.
    #[inline]
    pub fn is_expired(&self, now: u64) -> bool {
        now.saturating_sub(self.last_provided) > self.ttl_ms
    }
}

// ── RoutingHint ────────────────────────────────────────────────────────────

/// Cached routing hint: the nearest peers known for a CID.
#[derive(Debug, Clone, PartialEq)]
pub struct RoutingHint {
    /// Content identifier (CID string).
    pub cid: String,
    /// Peer IDs of the nearest known peers.
    pub nearest_peers: Vec<String>,
    /// Confidence score in [0.0, 1.0].
    pub confidence: f64,
    /// Millisecond timestamp when this hint was cached.
    pub cached_at: u64,
    /// Time-to-live in milliseconds.
    pub ttl_ms: u64,
}

impl RoutingHint {
    /// Returns `true` when `now - cached_at > ttl_ms`.
    #[inline]
    pub fn is_expired(&self, now: u64) -> bool {
        now.saturating_sub(self.cached_at) > self.ttl_ms
    }
}

// ── NegativeCacheEntry ─────────────────────────────────────────────────────

/// Negative-cache entry: CID was looked up and not found.
#[derive(Debug, Clone, PartialEq)]
pub struct NegativeCacheEntry {
    /// Content identifier (CID string).
    pub cid: String,
    /// Millisecond timestamp when the not-found result was recorded.
    pub not_found_at: u64,
    /// Time-to-live in milliseconds.
    pub ttl_ms: u64,
}

impl NegativeCacheEntry {
    /// Returns `true` when `now - not_found_at > ttl_ms`.
    #[inline]
    pub fn is_expired(&self, now: u64) -> bool {
        now.saturating_sub(self.not_found_at) > self.ttl_ms
    }
}

// ── CacheConfig ────────────────────────────────────────────────────────────

/// Configuration for [`ContentRoutingCache`].
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of CIDs in the provider tier.
    pub max_providers: usize,
    /// Maximum number of CIDs in the routing-hint tier.
    pub max_hints: usize,
    /// Maximum number of CIDs in the negative-cache tier.
    pub max_negative: usize,
    /// Default provider record TTL in milliseconds.
    pub default_provider_ttl_ms: u64,
    /// Default routing-hint TTL in milliseconds.
    pub default_hint_ttl_ms: u64,
    /// Default negative-cache TTL in milliseconds.
    pub default_negative_ttl_ms: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_providers: DEFAULT_MAX_PROVIDERS,
            max_hints: DEFAULT_MAX_HINTS,
            max_negative: DEFAULT_MAX_NEGATIVE,
            default_provider_ttl_ms: DEFAULT_PROVIDER_TTL_MS,
            default_hint_ttl_ms: DEFAULT_HINT_TTL_MS,
            default_negative_ttl_ms: DEFAULT_NEGATIVE_TTL_MS,
        }
    }
}

// ── CacheStats ─────────────────────────────────────────────────────────────

/// Snapshot of cache statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheStats {
    /// Number of distinct CIDs in the provider tier.
    pub provider_cids: usize,
    /// Total individual provider records stored.
    pub total_providers: usize,
    /// Number of routing hints cached.
    pub hints_cached: usize,
    /// Number of negative-cache entries.
    pub negative_cached: usize,
    /// Total provider lookups since creation.
    pub total_provider_lookups: u64,
    /// Total routing-hint lookups since creation.
    pub total_hint_lookups: u64,
    /// Fraction of lookups that were cache hits, in [0.0, 1.0].
    pub hit_rate: f64,
}

// ── Internal eviction helper ───────────────────────────────────────────────

/// Remove the entry with the smallest `key_fn` value from `map`.
/// Tie-breaks deterministically by `HashMap` iteration order.
fn evict_oldest_by<V, F>(map: &mut HashMap<String, V>, key_fn: F)
where
    F: Fn(&V) -> u64,
{
    if map.is_empty() {
        return;
    }
    let oldest = map
        .iter()
        .map(|(k, v)| (k.clone(), key_fn(v)))
        .min_by_key(|(_, ts)| *ts)
        .map(|(k, _)| k);
    if let Some(k) = oldest {
        map.remove(&k);
    }
}

// ── ContentRoutingCache ────────────────────────────────────────────────────

/// Multi-tier cache for DHT content routing.
///
/// Three independent tiers:
/// 1. **Provider tier** – stores [`CrcProviderRecord`]s (multiple per CID).
/// 2. **Hint tier** – stores one [`RoutingHint`] per CID.
/// 3. **Negative tier** – stores one [`NegativeCacheEntry`] per CID.
#[derive(Debug)]
pub struct ContentRoutingCache {
    /// Cache configuration.
    pub config: CacheConfig,
    /// Provider tier: CID → list of provider records.
    providers: HashMap<String, Vec<CrcProviderRecord>>,
    /// Hint tier: CID → routing hint.
    hints: HashMap<String, RoutingHint>,
    /// Negative tier: CID → negative cache entry.
    negative: HashMap<String, NegativeCacheEntry>,
    /// Cumulative provider-tier lookup count.
    pub total_provider_lookups: u64,
    /// Cumulative hint-tier lookup count.
    pub total_hint_lookups: u64,
    /// Cumulative cache hits (provider + hint tiers combined).
    pub cache_hits: u64,
    /// Cumulative cache misses (provider + hint tiers combined).
    pub cache_misses: u64,
}

impl ContentRoutingCache {
    // ── Construction ───────────────────────────────────────────────────────

    /// Create a new cache with the given configuration.
    pub fn new(config: CacheConfig) -> Self {
        Self {
            providers: HashMap::new(),
            hints: HashMap::new(),
            negative: HashMap::new(),
            total_provider_lookups: 0,
            total_hint_lookups: 0,
            cache_hits: 0,
            cache_misses: 0,
            config,
        }
    }

    // ── Provider tier ──────────────────────────────────────────────────────

    /// Insert a provider record.
    ///
    /// If the per-CID limit (`MAX_PROVIDERS_PER_CID`) would be exceeded the
    /// oldest record (smallest `last_provided`) is evicted first.  If the
    /// global provider-CID limit would be exceeded the CID whose most-recent
    /// provider record is oldest is removed entirely.
    pub fn add_provider(&mut self, record: CrcProviderRecord) {
        // Global CID-level eviction: if map is at capacity AND this CID is new.
        if !self.providers.contains_key(&record.cid)
            && self.providers.len() >= self.config.max_providers
        {
            // Evict the CID with the oldest most-recent provider.
            evict_oldest_by(&mut self.providers, |records| {
                records.iter().map(|r| r.last_provided).max().unwrap_or(0)
            });
        }

        let list = self.providers.entry(record.cid.clone()).or_default();

        // Per-CID limit: evict the oldest record.
        if list.len() >= MAX_PROVIDERS_PER_CID {
            if let Some(pos) = list
                .iter()
                .enumerate()
                .min_by_key(|(_, r)| r.last_provided)
                .map(|(i, _)| i)
            {
                list.remove(pos);
            }
        }

        list.push(record);
    }

    /// Return all non-expired provider records for `cid`.
    ///
    /// Expired records are removed in place.  Updates lookup and hit/miss
    /// counters.
    pub fn get_providers(&mut self, cid: &str, now: u64) -> Vec<&CrcProviderRecord> {
        self.total_provider_lookups += 1;

        // Sweep expired records for this CID only.
        if let Some(list) = self.providers.get_mut(cid) {
            list.retain(|r| !r.is_expired(now));
            if list.is_empty() {
                self.providers.remove(cid);
            }
        }

        match self.providers.get(cid) {
            Some(list) if !list.is_empty() => {
                self.cache_hits += 1;
                list.iter().collect()
            }
            _ => {
                self.cache_misses += 1;
                Vec::new()
            }
        }
    }

    /// Remove a specific provider record identified by `(cid, peer_id)`.
    ///
    /// Returns `true` if a record was removed.
    pub fn remove_provider(&mut self, cid: &str, peer_id: &str) -> bool {
        if let Some(list) = self.providers.get_mut(cid) {
            let before = list.len();
            list.retain(|r| r.peer_id != peer_id);
            let removed = list.len() < before;
            if list.is_empty() {
                self.providers.remove(cid);
            }
            return removed;
        }
        false
    }

    // ── Hint tier ──────────────────────────────────────────────────────────

    /// Insert a routing hint.
    ///
    /// If the hint-tier capacity would be exceeded the oldest hint (smallest
    /// `cached_at`) is evicted first.
    pub fn add_hint(&mut self, hint: RoutingHint) {
        if !self.hints.contains_key(&hint.cid) && self.hints.len() >= self.config.max_hints {
            evict_oldest_by(&mut self.hints, |h| h.cached_at);
        }
        self.hints.insert(hint.cid.clone(), hint);
    }

    /// Return the routing hint for `cid` if it exists and has not expired.
    ///
    /// If the hint exists but is expired it is removed and `None` is returned.
    /// Updates lookup and hit/miss counters.
    pub fn get_hint(&mut self, cid: &str, now: u64) -> Option<&RoutingHint> {
        self.total_hint_lookups += 1;

        // Check expiry first.
        if let Some(h) = self.hints.get(cid) {
            if h.is_expired(now) {
                self.hints.remove(cid);
                self.cache_misses += 1;
                return None;
            }
        }

        match self.hints.get(cid) {
            Some(_) => {
                self.cache_hits += 1;
                self.hints.get(cid)
            }
            None => {
                self.cache_misses += 1;
                None
            }
        }
    }

    /// Remove the hint for `cid`.  Returns `true` if a hint was present.
    pub fn remove_hint(&mut self, cid: &str) -> bool {
        self.hints.remove(cid).is_some()
    }

    // ── Negative tier ─────────────────────────────────────────────────────

    /// Insert a negative cache entry.
    ///
    /// If the negative-tier capacity would be exceeded the oldest entry
    /// (smallest `not_found_at`) is evicted first.
    pub fn add_negative(&mut self, entry: NegativeCacheEntry) {
        if !self.negative.contains_key(&entry.cid)
            && self.negative.len() >= self.config.max_negative
        {
            evict_oldest_by(&mut self.negative, |e| e.not_found_at);
        }
        self.negative.insert(entry.cid.clone(), entry);
    }

    /// Returns `true` if there is a live (non-expired) negative entry for `cid`.
    ///
    /// If the entry is expired it is removed.
    pub fn is_negative(&mut self, cid: &str, now: u64) -> bool {
        match self.negative.get(cid) {
            Some(e) if e.is_expired(now) => {
                self.negative.remove(cid);
                false
            }
            Some(_) => true,
            None => false,
        }
    }

    /// Remove the negative entry for `cid`.  Returns `true` if one existed.
    pub fn remove_negative(&mut self, cid: &str) -> bool {
        self.negative.remove(cid).is_some()
    }

    // ── Cross-tier sweeps ─────────────────────────────────────────────────

    /// Sweep all three tiers and remove expired entries.
    ///
    /// Returns the total number of items removed.
    pub fn evict_expired(&mut self, now: u64) -> usize {
        let mut removed = 0_usize;

        // Provider tier: remove individual records, then empty CID buckets.
        let mut empty_cids: Vec<String> = Vec::new();
        for (cid, list) in self.providers.iter_mut() {
            let before = list.len();
            list.retain(|r| !r.is_expired(now));
            removed += before - list.len();
            if list.is_empty() {
                empty_cids.push(cid.clone());
            }
        }
        for cid in empty_cids {
            self.providers.remove(&cid);
        }

        // Hint tier.
        let expired_hints: Vec<String> = self
            .hints
            .iter()
            .filter(|(_, h)| h.is_expired(now))
            .map(|(k, _)| k.clone())
            .collect();
        removed += expired_hints.len();
        for k in expired_hints {
            self.hints.remove(&k);
        }

        // Negative tier.
        let expired_neg: Vec<String> = self
            .negative
            .iter()
            .filter(|(_, e)| e.is_expired(now))
            .map(|(k, _)| k.clone())
            .collect();
        removed += expired_neg.len();
        for k in expired_neg {
            self.negative.remove(&k);
        }

        removed
    }

    // ── Query helpers ─────────────────────────────────────────────────────

    /// Returns `true` if the provider tier holds at least one record for `cid`.
    ///
    /// Note: this does **not** check expiry; call [`evict_expired`] or
    /// [`get_providers`] to prune first if freshness matters.
    ///
    /// [`evict_expired`]: ContentRoutingCache::evict_expired
    /// [`get_providers`]: ContentRoutingCache::get_providers
    pub fn has_content(&self, cid: &str) -> bool {
        self.providers
            .get(cid)
            .map(|v| !v.is_empty())
            .unwrap_or(false)
    }

    // ── Statistics ────────────────────────────────────────────────────────

    /// Return a statistics snapshot.
    pub fn cache_stats(&self) -> CacheStats {
        let provider_cids = self.providers.len();
        let total_providers: usize = self.providers.values().map(|v| v.len()).sum();
        let hints_cached = self.hints.len();
        let negative_cached = self.negative.len();

        let total_lookups = self.total_provider_lookups + self.total_hint_lookups;
        let hit_rate = if total_lookups == 0 {
            0.0_f64
        } else {
            self.cache_hits as f64 / total_lookups as f64
        };

        CacheStats {
            provider_cids,
            total_providers,
            hints_cached,
            negative_cached,
            total_provider_lookups: self.total_provider_lookups,
            total_hint_lookups: self.total_hint_lookups,
            hit_rate,
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        CacheConfig, CacheStats, ContentRoutingCache, CrcProviderRecord, NegativeCacheEntry,
        RoutingHint, DEFAULT_HINT_TTL_MS, DEFAULT_NEGATIVE_TTL_MS, DEFAULT_PROVIDER_TTL_MS,
        MAX_PROVIDERS_PER_CID,
    };

    // ── Helpers ───────────────────────────────────────────────────────────

    fn make_record(cid: &str, peer: &str, last_provided: u64, ttl_ms: u64) -> CrcProviderRecord {
        CrcProviderRecord {
            cid: cid.to_string(),
            peer_id: peer.to_string(),
            multiaddrs: vec![format!("/ip4/127.0.0.1/tcp/4001/{peer}")],
            last_provided,
            ttl_ms,
        }
    }

    fn make_hint(cid: &str, peers: &[&str], cached_at: u64, ttl_ms: u64) -> RoutingHint {
        RoutingHint {
            cid: cid.to_string(),
            nearest_peers: peers.iter().map(|s| s.to_string()).collect(),
            confidence: 0.9,
            cached_at,
            ttl_ms,
        }
    }

    fn make_negative(cid: &str, not_found_at: u64, ttl_ms: u64) -> NegativeCacheEntry {
        NegativeCacheEntry {
            cid: cid.to_string(),
            not_found_at,
            ttl_ms,
        }
    }

    fn default_cache() -> ContentRoutingCache {
        ContentRoutingCache::new(CacheConfig::default())
    }

    // ── CrcProviderRecord ─────────────────────────────────────────────────

    #[test]
    fn test_provider_record_not_expired_when_within_ttl() {
        let r = make_record("cid1", "peer1", 1000, 500);
        assert!(!r.is_expired(1400));
    }

    #[test]
    fn test_provider_record_expired_beyond_ttl() {
        let r = make_record("cid1", "peer1", 1000, 500);
        assert!(r.is_expired(1501));
    }

    #[test]
    fn test_provider_record_expired_exactly_at_boundary() {
        // is_expired uses >, so at exactly ttl_ms it is NOT expired.
        let r = make_record("cid1", "peer1", 0, 100);
        assert!(!r.is_expired(100));
    }

    #[test]
    fn test_provider_record_saturating_sub_prevents_underflow() {
        let r = make_record("cid1", "peer1", 5000, 100);
        // now < last_provided → saturating_sub returns 0, not expired.
        assert!(!r.is_expired(100));
    }

    // ── RoutingHint ────────────────────────────────────────────────────────

    #[test]
    fn test_routing_hint_not_expired() {
        let h = make_hint("cid1", &["p1"], 1000, 500);
        assert!(!h.is_expired(1499));
    }

    #[test]
    fn test_routing_hint_expired() {
        let h = make_hint("cid1", &["p1"], 1000, 500);
        assert!(h.is_expired(1501));
    }

    // ── NegativeCacheEntry ─────────────────────────────────────────────────

    #[test]
    fn test_negative_entry_not_expired() {
        let e = make_negative("cid1", 0, 60_000);
        assert!(!e.is_expired(59_999));
    }

    #[test]
    fn test_negative_entry_expired() {
        let e = make_negative("cid1", 0, 60_000);
        assert!(e.is_expired(60_001));
    }

    // ── add_provider / get_providers ───────────────────────────────────────

    #[test]
    fn test_add_and_get_provider_basic() {
        let mut c = default_cache();
        c.add_provider(make_record("cid1", "peer1", 0, DEFAULT_PROVIDER_TTL_MS));
        let got = c.get_providers("cid1", 0);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].peer_id, "peer1");
    }

    #[test]
    fn test_get_providers_empty_for_unknown_cid() {
        let mut c = default_cache();
        let got = c.get_providers("unknown", 0);
        assert!(got.is_empty());
    }

    #[test]
    fn test_get_providers_removes_expired_records() {
        let mut c = default_cache();
        c.add_provider(make_record("cid1", "peer1", 0, 100));
        // Not expired at t=50.
        assert_eq!(c.get_providers("cid1", 50).len(), 1);
        // Expired at t=200.
        let got = c.get_providers("cid1", 200);
        assert!(got.is_empty());
        // The CID should have been removed from the map entirely.
        assert!(!c.has_content("cid1"));
    }

    #[test]
    fn test_get_providers_updates_counters() {
        let mut c = default_cache();
        c.add_provider(make_record("cid1", "peer1", 0, DEFAULT_PROVIDER_TTL_MS));
        // hit
        c.get_providers("cid1", 0);
        // miss
        c.get_providers("unknown", 0);
        assert_eq!(c.total_provider_lookups, 2);
        assert_eq!(c.cache_hits, 1);
        assert_eq!(c.cache_misses, 1);
    }

    #[test]
    fn test_per_cid_provider_limit_evicts_oldest() {
        let mut c = default_cache();
        // Fill up to the limit with timestamps 0..MAX-1.
        for i in 0..MAX_PROVIDERS_PER_CID {
            c.add_provider(make_record(
                "cid1",
                &format!("peer{i}"),
                i as u64,
                DEFAULT_PROVIDER_TTL_MS,
            ));
        }
        // The 21st record triggers eviction of the oldest (peer0, ts=0).
        c.add_provider(make_record(
            "cid1",
            "peerNew",
            MAX_PROVIDERS_PER_CID as u64,
            DEFAULT_PROVIDER_TTL_MS,
        ));
        let providers = c.providers.get("cid1").expect("cid1 should exist");
        assert_eq!(providers.len(), MAX_PROVIDERS_PER_CID);
        // peer0 must be gone.
        assert!(!providers.iter().any(|r| r.peer_id == "peer0"));
        // peerNew must be present.
        assert!(providers.iter().any(|r| r.peer_id == "peerNew"));
    }

    #[test]
    fn test_global_provider_cid_limit_evicts_oldest_cid() {
        let config = CacheConfig {
            max_providers: 3,
            ..CacheConfig::default()
        };
        let mut c = ContentRoutingCache::new(config);
        c.add_provider(make_record("cid1", "peer1", 100, DEFAULT_PROVIDER_TTL_MS));
        c.add_provider(make_record("cid2", "peer2", 200, DEFAULT_PROVIDER_TTL_MS));
        c.add_provider(make_record("cid3", "peer3", 300, DEFAULT_PROVIDER_TTL_MS));
        // Fourth CID: cid1 (ts=100) is oldest and should be evicted.
        c.add_provider(make_record("cid4", "peer4", 400, DEFAULT_PROVIDER_TTL_MS));
        assert!(
            !c.providers.contains_key("cid1"),
            "cid1 should have been evicted"
        );
        assert!(c.providers.contains_key("cid4"));
    }

    // ── remove_provider ────────────────────────────────────────────────────

    #[test]
    fn test_remove_provider_returns_true_when_present() {
        let mut c = default_cache();
        c.add_provider(make_record("cid1", "peer1", 0, DEFAULT_PROVIDER_TTL_MS));
        assert!(c.remove_provider("cid1", "peer1"));
        assert!(!c.has_content("cid1"));
    }

    #[test]
    fn test_remove_provider_returns_false_when_absent() {
        let mut c = default_cache();
        assert!(!c.remove_provider("cid1", "peer1"));
    }

    #[test]
    fn test_remove_provider_leaves_other_peers() {
        let mut c = default_cache();
        c.add_provider(make_record("cid1", "peer1", 0, DEFAULT_PROVIDER_TTL_MS));
        c.add_provider(make_record("cid1", "peer2", 0, DEFAULT_PROVIDER_TTL_MS));
        c.remove_provider("cid1", "peer1");
        let got = c.get_providers("cid1", 0);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].peer_id, "peer2");
    }

    // ── add_hint / get_hint ────────────────────────────────────────────────

    #[test]
    fn test_add_and_get_hint_basic() {
        let mut c = default_cache();
        c.add_hint(make_hint("cid1", &["p1", "p2"], 0, DEFAULT_HINT_TTL_MS));
        let h = c.get_hint("cid1", 0).expect("hint should be present");
        assert_eq!(h.nearest_peers.len(), 2);
    }

    #[test]
    fn test_get_hint_returns_none_when_expired() {
        let mut c = default_cache();
        c.add_hint(make_hint("cid1", &["p1"], 0, 100));
        assert!(c.get_hint("cid1", 200).is_none());
        // Expired hint must have been removed.
        assert!(!c.hints.contains_key("cid1"));
    }

    #[test]
    fn test_get_hint_updates_counters() {
        let mut c = default_cache();
        c.add_hint(make_hint("cid1", &["p1"], 0, DEFAULT_HINT_TTL_MS));
        c.get_hint("cid1", 0); // hit
        c.get_hint("missing", 0); // miss
        assert_eq!(c.total_hint_lookups, 2);
        assert_eq!(c.cache_hits, 1);
        assert_eq!(c.cache_misses, 1);
    }

    #[test]
    fn test_hint_global_limit_evicts_oldest() {
        let config = CacheConfig {
            max_hints: 2,
            ..CacheConfig::default()
        };
        let mut c = ContentRoutingCache::new(config);
        c.add_hint(make_hint("cid1", &["p1"], 100, DEFAULT_HINT_TTL_MS));
        c.add_hint(make_hint("cid2", &["p2"], 200, DEFAULT_HINT_TTL_MS));
        // Third hint: cid1 (ts=100) should be evicted.
        c.add_hint(make_hint("cid3", &["p3"], 300, DEFAULT_HINT_TTL_MS));
        assert!(!c.hints.contains_key("cid1"), "cid1 hint should be evicted");
        assert!(c.hints.contains_key("cid3"));
    }

    // ── remove_hint ───────────────────────────────────────────────────────

    #[test]
    fn test_remove_hint_returns_true() {
        let mut c = default_cache();
        c.add_hint(make_hint("cid1", &["p1"], 0, DEFAULT_HINT_TTL_MS));
        assert!(c.remove_hint("cid1"));
        assert!(c.get_hint("cid1", 0).is_none());
    }

    #[test]
    fn test_remove_hint_returns_false_when_absent() {
        let mut c = default_cache();
        assert!(!c.remove_hint("cid1"));
    }

    // ── add_negative / is_negative ─────────────────────────────────────────

    #[test]
    fn test_is_negative_true_when_present_and_fresh() {
        let mut c = default_cache();
        c.add_negative(make_negative("cid1", 0, DEFAULT_NEGATIVE_TTL_MS));
        assert!(c.is_negative("cid1", 0));
    }

    #[test]
    fn test_is_negative_false_when_absent() {
        let mut c = default_cache();
        assert!(!c.is_negative("cid1", 0));
    }

    #[test]
    fn test_is_negative_false_and_removes_when_expired() {
        let mut c = default_cache();
        c.add_negative(make_negative("cid1", 0, 100));
        assert!(!c.is_negative("cid1", 200));
        assert!(!c.negative.contains_key("cid1"));
    }

    #[test]
    fn test_negative_global_limit_evicts_oldest() {
        let config = CacheConfig {
            max_negative: 2,
            ..CacheConfig::default()
        };
        let mut c = ContentRoutingCache::new(config);
        c.add_negative(make_negative("cid1", 100, DEFAULT_NEGATIVE_TTL_MS));
        c.add_negative(make_negative("cid2", 200, DEFAULT_NEGATIVE_TTL_MS));
        // Third entry: cid1 (ts=100) should be evicted.
        c.add_negative(make_negative("cid3", 300, DEFAULT_NEGATIVE_TTL_MS));
        assert!(
            !c.negative.contains_key("cid1"),
            "cid1 negative should be evicted"
        );
        assert!(c.negative.contains_key("cid3"));
    }

    // ── remove_negative ────────────────────────────────────────────────────

    #[test]
    fn test_remove_negative_returns_true() {
        let mut c = default_cache();
        c.add_negative(make_negative("cid1", 0, DEFAULT_NEGATIVE_TTL_MS));
        assert!(c.remove_negative("cid1"));
        assert!(!c.is_negative("cid1", 0));
    }

    #[test]
    fn test_remove_negative_returns_false_when_absent() {
        let mut c = default_cache();
        assert!(!c.remove_negative("cid1"));
    }

    // ── evict_expired ─────────────────────────────────────────────────────

    #[test]
    fn test_evict_expired_removes_all_tiers() {
        let mut c = default_cache();
        // Provider: cid1 expires, cid2 fresh.
        c.add_provider(make_record("cid1", "peer1", 0, 100));
        c.add_provider(make_record("cid2", "peer2", 0, DEFAULT_PROVIDER_TTL_MS));
        // Hint: cid3 expires, cid4 fresh.
        c.add_hint(make_hint("cid3", &["p3"], 0, 100));
        c.add_hint(make_hint("cid4", &["p4"], 0, DEFAULT_HINT_TTL_MS));
        // Negative: cid5 expires, cid6 fresh.
        c.add_negative(make_negative("cid5", 0, 100));
        c.add_negative(make_negative("cid6", 0, DEFAULT_NEGATIVE_TTL_MS));

        let removed = c.evict_expired(200);
        // 1 provider record + 1 hint + 1 negative = 3.
        assert_eq!(removed, 3);
        assert!(!c.providers.contains_key("cid1"));
        assert!(c.providers.contains_key("cid2"));
        assert!(!c.hints.contains_key("cid3"));
        assert!(c.hints.contains_key("cid4"));
        assert!(!c.negative.contains_key("cid5"));
        assert!(c.negative.contains_key("cid6"));
    }

    #[test]
    fn test_evict_expired_returns_zero_when_nothing_expired() {
        let mut c = default_cache();
        c.add_provider(make_record("cid1", "peer1", 0, DEFAULT_PROVIDER_TTL_MS));
        assert_eq!(c.evict_expired(0), 0);
    }

    #[test]
    fn test_evict_expired_multiple_providers_per_cid() {
        let mut c = default_cache();
        // Two providers for cid1: one expired, one fresh.
        c.add_provider(make_record("cid1", "peer1", 0, 50));
        c.add_provider(make_record("cid1", "peer2", 0, DEFAULT_PROVIDER_TTL_MS));
        let removed = c.evict_expired(100);
        assert_eq!(removed, 1);
        assert!(c.has_content("cid1"));
        let providers = c.providers.get("cid1").expect("cid1 must remain");
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].peer_id, "peer2");
    }

    // ── has_content ────────────────────────────────────────────────────────

    #[test]
    fn test_has_content_true_when_provider_present() {
        let mut c = default_cache();
        c.add_provider(make_record("cid1", "peer1", 0, DEFAULT_PROVIDER_TTL_MS));
        assert!(c.has_content("cid1"));
    }

    #[test]
    fn test_has_content_false_for_unknown_cid() {
        let c = default_cache();
        assert!(!c.has_content("unknown"));
    }

    // ── cache_stats ────────────────────────────────────────────────────────

    #[test]
    fn test_cache_stats_initial_state() {
        let c = default_cache();
        let s = c.cache_stats();
        assert_eq!(s.provider_cids, 0);
        assert_eq!(s.total_providers, 0);
        assert_eq!(s.hints_cached, 0);
        assert_eq!(s.negative_cached, 0);
        assert_eq!(s.hit_rate, 0.0);
    }

    #[test]
    fn test_cache_stats_counts_correctly() {
        let mut c = default_cache();
        c.add_provider(make_record("cid1", "peer1", 0, DEFAULT_PROVIDER_TTL_MS));
        c.add_provider(make_record("cid1", "peer2", 0, DEFAULT_PROVIDER_TTL_MS));
        c.add_provider(make_record("cid2", "peer3", 0, DEFAULT_PROVIDER_TTL_MS));
        c.add_hint(make_hint("cid3", &["p1"], 0, DEFAULT_HINT_TTL_MS));
        c.add_negative(make_negative("cid4", 0, DEFAULT_NEGATIVE_TTL_MS));

        let s = c.cache_stats();
        assert_eq!(s.provider_cids, 2);
        assert_eq!(s.total_providers, 3);
        assert_eq!(s.hints_cached, 1);
        assert_eq!(s.negative_cached, 1);
    }

    #[test]
    fn test_cache_stats_hit_rate_all_hits() {
        let mut c = default_cache();
        c.add_provider(make_record("cid1", "peer1", 0, DEFAULT_PROVIDER_TTL_MS));
        c.get_providers("cid1", 0);
        let s = c.cache_stats();
        assert!((s.hit_rate - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cache_stats_hit_rate_mixed() {
        let mut c = default_cache();
        c.add_provider(make_record("cid1", "peer1", 0, DEFAULT_PROVIDER_TTL_MS));
        c.get_providers("cid1", 0); // hit
        c.get_providers("miss1", 0); // miss
        c.get_providers("miss2", 0); // miss
        let s = c.cache_stats();
        // 1 hit out of 3 lookups.
        let expected = 1.0 / 3.0;
        assert!((s.hit_rate - expected).abs() < 1e-10);
    }

    // ── CacheConfig default ────────────────────────────────────────────────

    #[test]
    fn test_cache_config_defaults() {
        let cfg = CacheConfig::default();
        assert_eq!(cfg.max_providers, 10_000);
        assert_eq!(cfg.max_hints, 5_000);
        assert_eq!(cfg.max_negative, 2_000);
        assert_eq!(cfg.default_provider_ttl_ms, DEFAULT_PROVIDER_TTL_MS);
        assert_eq!(cfg.default_hint_ttl_ms, DEFAULT_HINT_TTL_MS);
        assert_eq!(cfg.default_negative_ttl_ms, DEFAULT_NEGATIVE_TTL_MS);
    }

    // ── CacheStats PartialEq ────────────────────────────────────────────────

    #[test]
    fn test_cache_stats_equality() {
        let s1 = CacheStats {
            provider_cids: 1,
            total_providers: 2,
            hints_cached: 3,
            negative_cached: 4,
            total_provider_lookups: 5,
            total_hint_lookups: 6,
            hit_rate: 0.5,
        };
        let s2 = s1.clone();
        assert_eq!(s1, s2);
    }

    // ── Hint overwrite ─────────────────────────────────────────────────────

    #[test]
    fn test_add_hint_overwrites_existing() {
        let mut c = default_cache();
        c.add_hint(make_hint("cid1", &["p1"], 0, DEFAULT_HINT_TTL_MS));
        c.add_hint(make_hint("cid1", &["p2", "p3"], 1000, DEFAULT_HINT_TTL_MS));
        let h = c.get_hint("cid1", 1000).expect("hint present");
        assert_eq!(h.nearest_peers.len(), 2);
        assert_eq!(c.hints.len(), 1, "overwrite should not increase count");
    }

    // ── Negative overwrite ─────────────────────────────────────────────────

    #[test]
    fn test_add_negative_overwrites_existing() {
        let mut c = default_cache();
        c.add_negative(make_negative("cid1", 0, DEFAULT_NEGATIVE_TTL_MS));
        c.add_negative(make_negative("cid1", 500, DEFAULT_NEGATIVE_TTL_MS));
        assert_eq!(c.negative.len(), 1);
        let entry = c.negative.get("cid1").expect("entry exists");
        assert_eq!(entry.not_found_at, 500);
    }

    // ── Multiple providers same CID ────────────────────────────────────────

    #[test]
    fn test_multiple_providers_same_cid_returned() {
        let mut c = default_cache();
        for i in 0_u64..5 {
            c.add_provider(make_record(
                "cid1",
                &format!("peer{i}"),
                i,
                DEFAULT_PROVIDER_TTL_MS,
            ));
        }
        let got = c.get_providers("cid1", 0);
        assert_eq!(got.len(), 5);
    }

    // ── Mixed expiry in single CID bucket ─────────────────────────────────

    #[test]
    fn test_partial_expiry_within_cid_bucket() {
        let mut c = default_cache();
        c.add_provider(make_record("cid1", "peer1", 0, 50));
        c.add_provider(make_record("cid1", "peer2", 0, 200));
        c.add_provider(make_record("cid1", "peer3", 0, 300));
        // At t=100: peer1 expired, peer2 and peer3 still live.
        let got = c.get_providers("cid1", 100);
        assert_eq!(got.len(), 2);
        let ids: Vec<&str> = got.iter().map(|r| r.peer_id.as_str()).collect();
        assert!(ids.contains(&"peer2"));
        assert!(ids.contains(&"peer3"));
    }
}
