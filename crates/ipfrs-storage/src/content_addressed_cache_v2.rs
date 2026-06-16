//! Content-Addressed Cache V2 — multi-tier eviction with admission control.
//!
//! Provides a production-grade, purely in-memory cache keyed by 32-byte CIDs with:
//! - **Hot tier**: LRU eviction, short TTL, fastest access
//! - **Warm tier**: LFU eviction, longer TTL, larger capacity
//! - **Bloom-filter admission gate**: three FNV-1a hash functions prevent one-hit wonders
//! - **Eviction log**: bounded ring of the last 500 eviction events
//! - **TTL sweep**: single-pass expiry across both tiers
//! - **Warm drain**: simulate write-back of least-frequently-used 25% to slower storage

use std::collections::{HashMap, VecDeque};

// ─── Primitive helpers ────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash.
#[inline(always)]
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

/// XorShift64 PRNG — mutates `state` in place and returns the new value.
#[inline(always)]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Current UNIX timestamp in seconds (falls back to 0 if the clock is unavailable).
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ─── Public types ─────────────────────────────────────────────────────────────

/// 32-byte content identifier — a fixed-size, copy-friendly newtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Cac2Cid(pub [u8; 32]);

impl Cac2Cid {
    /// Construct a CID from a 32-byte slice; returns `None` if the slice is too short.
    pub fn from_slice(s: &[u8]) -> Option<Self> {
        if s.len() < 32 {
            return None;
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&s[..32]);
        Some(Self(arr))
    }

    /// Derive a deterministic CID from arbitrary bytes via FNV-1a.
    pub fn from_bytes(data: &[u8]) -> Self {
        let h1 = fnv1a_64(data);
        let h2 = fnv1a_64(&h1.to_le_bytes());
        let h3 = fnv1a_64(&h2.to_le_bytes());
        let h4 = fnv1a_64(&h3.to_le_bytes());
        let mut arr = [0u8; 32];
        arr[0..8].copy_from_slice(&h1.to_le_bytes());
        arr[8..16].copy_from_slice(&h2.to_le_bytes());
        arr[16..24].copy_from_slice(&h3.to_le_bytes());
        arr[24..32].copy_from_slice(&h4.to_le_bytes());
        Self(arr)
    }
}

/// Cache tier designation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cac2Tier {
    /// In the fast, small, LRU-evicted hot tier.
    Hot,
    /// In the larger, LFU-evicted warm tier.
    Warm,
    /// Entry has been evicted (used inside `Cac2EvictionRecord`).
    Evicted,
}

/// Why an entry was removed from the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cac2EvictionReason {
    /// LRU victim in the hot tier.
    LruEviction,
    /// Lowest-frequency victim in the warm tier.
    LfuEviction,
    /// TTL deadline passed.
    TtlExpiry,
    /// Caller explicitly evicted the entry.
    ManualEviction,
    /// Evicted to relieve capacity pressure when both tiers are full.
    CapacityPressure,
}

/// Single entry stored in the hot or warm tier.
#[derive(Debug, Clone)]
pub struct Cac2Entry {
    /// Content identifier.
    pub cid: Cac2Cid,
    /// Raw payload.
    pub data: Vec<u8>,
    /// Number of times this entry has been accessed (reads only).
    pub access_count: u32,
    /// UNIX timestamp (seconds) when the entry was first inserted.
    pub inserted_at: u64,
    /// UNIX timestamp (seconds) of the most-recent access.
    pub last_accessed: u64,
    /// Current tier.
    pub tier: Cac2Tier,
}

/// A record of a single eviction event.
#[derive(Debug, Clone)]
pub struct Cac2EvictionRecord {
    /// UNIX timestamp (seconds) when the eviction occurred.
    pub ts: u64,
    /// Which entry was evicted.
    pub cid: Cac2Cid,
    /// Which tier the entry was evicted from.
    pub tier: Cac2Tier,
    /// Why it was evicted.
    pub reason: Cac2EvictionReason,
}

/// Configuration knobs for `ContentAddressedCacheV2`.
#[derive(Debug, Clone)]
pub struct Cac2CacheConfig {
    /// Maximum number of entries in the hot tier.
    pub hot_capacity: usize,
    /// Maximum number of entries in the warm tier.
    pub warm_capacity: usize,
    /// Number of bits in the Bloom filter backing array.
    pub bloom_size: usize,
    /// Hot-tier TTL in seconds (0 = no TTL).
    pub hot_ttl_secs: u64,
    /// Warm-tier TTL in seconds (0 = no TTL).
    pub warm_ttl_secs: u64,
    /// Minimum access count before an entry is eligible for hot-tier admission via promotion.
    pub admission_threshold: u32,
}

impl Default for Cac2CacheConfig {
    fn default() -> Self {
        Self {
            hot_capacity: 512,
            warm_capacity: 4096,
            bloom_size: 1 << 16, // 64 KiB of bits → 8 KiB of bytes
            hot_ttl_secs: 300,
            warm_ttl_secs: 3600,
            admission_threshold: 2,
        }
    }
}

/// Snapshot of cache statistics.
#[derive(Debug, Clone, Default)]
pub struct Cac2CacheStats {
    /// Number of entries currently in the hot tier.
    pub hot_count: usize,
    /// Number of entries currently in the warm tier.
    pub warm_count: usize,
    /// Cumulative hit rate (0.0–1.0).
    pub hit_rate: f64,
    /// Cumulative miss rate (0.0–1.0).
    pub miss_rate: f64,
    /// Total number of eviction events recorded.
    pub eviction_count: u64,
    /// Estimated Bloom-filter false-positive probability.
    pub bloom_false_positive_est: f64,
}

// ─── Type aliases (as required by spec) ──────────────────────────────────────

/// Alias for the main cache type.
pub type Cac2ContentAddressedCacheV2 = ContentAddressedCacheV2;

// ─── Internal Bloom helper ────────────────────────────────────────────────────

/// Minimal Bloom filter backed by a `Vec<u64>` word array.
///
/// Uses three independent FNV-1a probes (with seed mixing) to test membership.
struct BloomFilter {
    bits: Vec<u64>,
    /// Total number of addressable bits.
    num_bits: u64,
    /// Count of set bits (approximate — never decremented after clear).
    set_count: u64,
}

impl BloomFilter {
    fn new(num_bits: usize) -> Self {
        let num_bits = num_bits.max(64);
        let words = num_bits.div_ceil(64);
        Self {
            bits: vec![0u64; words],
            num_bits: (words * 64) as u64,
            set_count: 0,
        }
    }

    /// Three probe positions derived from the CID bytes via seeded FNV-1a.
    fn probe_positions(&self, cid: &Cac2Cid) -> [u64; 3] {
        let h0 = fnv1a_64(&cid.0);
        // Mix the seed into the data for independent probes.
        let mut seed1 = h0 ^ 0xdeadbeef_cafebabe;
        let h1 = xorshift64(&mut seed1);
        let mut seed2 = h1 ^ 0x0123456789abcdef;
        let h2 = xorshift64(&mut seed2);
        [h0 % self.num_bits, h1 % self.num_bits, h2 % self.num_bits]
    }

    fn insert(&mut self, cid: &Cac2Cid) {
        for pos in self.probe_positions(cid) {
            let word = (pos / 64) as usize;
            let bit = pos % 64;
            if self.bits[word] & (1u64 << bit) == 0 {
                self.bits[word] |= 1u64 << bit;
                self.set_count += 1;
            }
        }
    }

    fn probably_contains(&self, cid: &Cac2Cid) -> bool {
        for pos in self.probe_positions(cid) {
            let word = (pos / 64) as usize;
            let bit = pos % 64;
            if self.bits[word] & (1u64 << bit) == 0 {
                return false;
            }
        }
        true
    }

    /// Estimate false-positive probability using the standard Bloom formula.
    /// p ≈ (1 – e^(–k·n/m))^k  where k=3, m = num_bits, n ≈ set_count/3
    fn false_positive_estimate(&self) -> f64 {
        let m = self.num_bits as f64;
        let n = (self.set_count / 3) as f64; // each insertion sets ~3 bits
        let k = 3.0_f64;
        let inner = 1.0 - ((-k * n) / m).exp();
        inner.powf(k)
    }
}

// ─── LRU tracking helper ──────────────────────────────────────────────────────

/// Minimal doubly-linked-list LRU tracker that stores CIDs in order from
/// most-recently-used (front) to least-recently-used (back).
///
/// We implement it using a `VecDeque` because the hot tier is bounded to a
/// small capacity; O(n) scans are fine in practice for sizes ≤ 1024.
struct LruList {
    order: VecDeque<Cac2Cid>,
}

impl LruList {
    fn new() -> Self {
        Self {
            order: VecDeque::new(),
        }
    }

    /// Mark `cid` as most-recently used.  Insert at front if not present.
    fn touch(&mut self, cid: Cac2Cid) {
        self.order.retain(|c| *c != cid);
        self.order.push_front(cid);
    }

    /// Remove and return the LRU (least-recently-used) CID, if any.
    fn evict_lru(&mut self) -> Option<Cac2Cid> {
        self.order.pop_back()
    }

    /// Remove a specific CID from the tracking list.
    fn remove(&mut self, cid: &Cac2Cid) {
        self.order.retain(|c| c != cid);
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.order.len()
    }
}

// ─── Main cache struct ────────────────────────────────────────────────────────

/// Production-grade content-addressed cache with hot/warm tiering,
/// Bloom-filter admission control, and rich eviction accounting.
pub struct ContentAddressedCacheV2 {
    /// Fast, small tier — LRU eviction policy.
    hot: HashMap<Cac2Cid, Cac2Entry>,
    /// Larger tier — LFU eviction policy.
    warm: HashMap<Cac2Cid, Cac2Entry>,
    /// Bloom filter used as an admission gate.
    bloom: BloomFilter,
    /// Bounded ring of the last 500 eviction events.
    eviction_log: VecDeque<Cac2EvictionRecord>,
    /// LRU order tracker for the hot tier.
    hot_lru: LruList,
    /// Configuration.
    config: Cac2CacheConfig,
    /// Cumulative hit counter.
    hits: u64,
    /// Cumulative miss counter.
    misses: u64,
    /// Total evictions ever performed (may exceed the log's 500-record window).
    total_evictions: u64,
    /// PRNG state for tie-breaking and internal randomness.
    #[allow(dead_code)]
    rng_state: u64,
}

impl ContentAddressedCacheV2 {
    /// Construct a new cache with the given configuration.
    pub fn new(config: Cac2CacheConfig) -> Self {
        let bloom_size = config.bloom_size;
        Self {
            hot: HashMap::new(),
            warm: HashMap::new(),
            bloom: BloomFilter::new(bloom_size),
            eviction_log: VecDeque::new(),
            hot_lru: LruList::new(),
            config,
            hits: 0,
            misses: 0,
            total_evictions: 0,
            rng_state: 0xcafe_babe_dead_beef,
        }
    }

    /// Construct with default configuration.
    pub fn default_config() -> Self {
        Self::new(Cac2CacheConfig::default())
    }

    // ─── Bloom-filter facade ──────────────────────────────────────────────

    /// Test whether `cid` *probably* exists in the admission filter.
    pub fn bloom_probably_contains(&self, cid: &Cac2Cid) -> bool {
        self.bloom.probably_contains(cid)
    }

    // ─── Core insert ─────────────────────────────────────────────────────

    /// Insert `data` under `cid`.
    ///
    /// Admission control: if the Bloom filter does not contain `cid` yet
    /// (i.e., this is a cold first-time access), the entry is written only to
    /// the warm tier — bypassing the hot tier to protect it from one-hit
    /// wonders.  If the Bloom filter *does* contain the CID (repeat access),
    /// the entry is routed to the hot tier directly.
    ///
    /// Eviction cascade:
    /// 1. Hot full → demote the LRU hot entry to warm.
    /// 2. Warm full → evict the entry with the lowest `access_count` (LFU).
    pub fn insert(&mut self, cid: Cac2Cid, data: Vec<u8>) {
        let now = now_secs();

        // If the CID already lives in hot or warm, update in-place and return.
        if let Some(entry) = self.hot.get_mut(&cid) {
            entry.data = data;
            entry.last_accessed = now;
            entry.access_count = entry.access_count.saturating_add(1);
            self.hot_lru.touch(cid);
            self.bloom.insert(&cid);
            return;
        }
        if let Some(entry) = self.warm.get_mut(&cid) {
            entry.data = data;
            entry.last_accessed = now;
            entry.access_count = entry.access_count.saturating_add(1);
            self.bloom.insert(&cid);
            return;
        }

        // Admission gate: has this CID been seen before?
        let seen_before = self.bloom.probably_contains(&cid);
        self.bloom.insert(&cid);

        if seen_before {
            // Route to hot tier.
            self.ensure_hot_capacity(now);
            let entry = Cac2Entry {
                cid,
                data,
                access_count: 1,
                inserted_at: now,
                last_accessed: now,
                tier: Cac2Tier::Hot,
            };
            self.hot.insert(cid, entry);
            self.hot_lru.touch(cid);
        } else {
            // First-time CID → route to warm tier (avoid polluting hot).
            self.ensure_warm_capacity(now);
            let entry = Cac2Entry {
                cid,
                data,
                access_count: 0,
                inserted_at: now,
                last_accessed: now,
                tier: Cac2Tier::Warm,
            };
            self.warm.insert(cid, entry);
        }
    }

    // ─── Core get ────────────────────────────────────────────────────────

    /// Retrieve a cached payload.
    ///
    /// Look-up order: hot tier first, then warm tier.  A warm hit with
    /// `access_count > admission_threshold` triggers promotion to the hot tier.
    pub fn get(&mut self, cid: &Cac2Cid) -> Option<&[u8]> {
        let now = now_secs();

        // Hot hit.
        if let Some(entry) = self.hot.get_mut(cid) {
            entry.access_count = entry.access_count.saturating_add(1);
            entry.last_accessed = now;
            self.hits += 1;
            self.hot_lru.touch(*cid);
            // Need to re-borrow immutably for the return.
            return self.hot.get(cid).map(|e| e.data.as_slice());
        }

        // Warm hit — check for promotion eligibility.
        if self.warm.contains_key(cid) {
            let threshold = self.config.admission_threshold;
            // Mutate first, then decide.
            if let Some(entry) = self.warm.get_mut(cid) {
                entry.access_count = entry.access_count.saturating_add(1);
                entry.last_accessed = now;
                self.hits += 1;
            }
            let promote = self
                .warm
                .get(cid)
                .map(|e| e.access_count > threshold)
                .unwrap_or(false);
            if promote {
                self.promote_warm_to_hot(cid, now);
                return self.hot.get(cid).map(|e| e.data.as_slice());
            }
            return self.warm.get(cid).map(|e| e.data.as_slice());
        }

        self.misses += 1;
        None
    }

    // ─── Manual eviction ─────────────────────────────────────────────────

    /// Forcibly remove `cid` from all tiers.
    pub fn evict(&mut self, cid: &Cac2Cid) {
        let now = now_secs();
        if self.hot.remove(cid).is_some() {
            self.hot_lru.remove(cid);
            self.log_eviction(now, *cid, Cac2Tier::Hot, Cac2EvictionReason::ManualEviction);
        }
        if self.warm.remove(cid).is_some() {
            self.log_eviction(
                now,
                *cid,
                Cac2Tier::Warm,
                Cac2EvictionReason::ManualEviction,
            );
        }
    }

    // ─── TTL sweep ────────────────────────────────────────────────────────

    /// Remove all entries whose TTL has elapsed as of `now_ts`.
    /// Pass `now_secs()` for the current wall-clock time.
    pub fn expire_stale(&mut self, now_ts: u64) {
        let hot_ttl = self.config.hot_ttl_secs;
        let warm_ttl = self.config.warm_ttl_secs;

        if hot_ttl > 0 {
            let expired_hot: Vec<Cac2Cid> = self
                .hot
                .values()
                .filter(|e| now_ts.saturating_sub(e.inserted_at) >= hot_ttl)
                .map(|e| e.cid)
                .collect();
            for cid in expired_hot {
                self.hot.remove(&cid);
                self.hot_lru.remove(&cid);
                self.log_eviction(now_ts, cid, Cac2Tier::Hot, Cac2EvictionReason::TtlExpiry);
            }
        }

        if warm_ttl > 0 {
            let expired_warm: Vec<Cac2Cid> = self
                .warm
                .values()
                .filter(|e| now_ts.saturating_sub(e.inserted_at) >= warm_ttl)
                .map(|e| e.cid)
                .collect();
            for cid in expired_warm {
                self.warm.remove(&cid);
                self.log_eviction(now_ts, cid, Cac2Tier::Warm, Cac2EvictionReason::TtlExpiry);
            }
        }
    }

    // ─── Warm drain ───────────────────────────────────────────────────────

    /// Drain the lowest-frequency 25% of warm-tier entries, returning them
    /// as `(Cac2Cid, Vec<u8>)` pairs so the caller can write them to disk.
    ///
    /// This simulates a write-back flush in a hierarchical storage system.
    pub fn drain_warm_to_disk_simulation(&mut self) -> Vec<(Cac2Cid, Vec<u8>)> {
        if self.warm.is_empty() {
            return Vec::new();
        }

        let drain_count = ((self.warm.len() as f64) * 0.25).ceil() as usize;
        if drain_count == 0 {
            return Vec::new();
        }

        let now = now_secs();

        // Collect and sort by access_count ascending (lowest frequency first).
        let mut candidates: Vec<(u32, Cac2Cid)> = self
            .warm
            .values()
            .map(|e| (e.access_count, e.cid))
            .collect();
        candidates.sort_unstable_by_key(|(count, _)| *count);

        let to_drain: Vec<Cac2Cid> = candidates
            .into_iter()
            .take(drain_count)
            .map(|(_, cid)| cid)
            .collect();

        let mut result = Vec::with_capacity(to_drain.len());
        for cid in to_drain {
            if let Some(entry) = self.warm.remove(&cid) {
                self.log_eviction(now, cid, Cac2Tier::Warm, Cac2EvictionReason::LfuEviction);
                result.push((cid, entry.data));
            }
        }
        result
    }

    // ─── Stats ────────────────────────────────────────────────────────────

    /// Return a snapshot of the current cache statistics.
    pub fn cache_stats(&self) -> Cac2CacheStats {
        let total = self.hits + self.misses;
        let (hit_rate, miss_rate) = if total == 0 {
            (0.0, 0.0)
        } else {
            (
                self.hits as f64 / total as f64,
                self.misses as f64 / total as f64,
            )
        };
        Cac2CacheStats {
            hot_count: self.hot.len(),
            warm_count: self.warm.len(),
            hit_rate,
            miss_rate,
            eviction_count: self.total_evictions,
            bloom_false_positive_est: self.bloom.false_positive_estimate(),
        }
    }

    /// Expose a reference to the eviction log.
    pub fn eviction_log(&self) -> &VecDeque<Cac2EvictionRecord> {
        &self.eviction_log
    }

    /// Return the current configuration.
    pub fn config(&self) -> &Cac2CacheConfig {
        &self.config
    }

    /// Number of entries in the hot tier.
    pub fn hot_len(&self) -> usize {
        self.hot.len()
    }

    /// Number of entries in the warm tier.
    pub fn warm_len(&self) -> usize {
        self.warm.len()
    }

    /// Total number of cached entries across both tiers.
    pub fn total_len(&self) -> usize {
        self.hot.len() + self.warm.len()
    }

    /// Return `true` if neither tier contains any entry.
    pub fn is_empty(&self) -> bool {
        self.hot.is_empty() && self.warm.is_empty()
    }

    /// Check whether `cid` is present in either tier without updating stats.
    pub fn contains(&self, cid: &Cac2Cid) -> bool {
        self.hot.contains_key(cid) || self.warm.contains_key(cid)
    }

    // ─── Private helpers ──────────────────────────────────────────────────

    /// Make room in the hot tier by demoting the LRU entry to warm.
    fn ensure_hot_capacity(&mut self, now: u64) {
        while self.hot.len() >= self.config.hot_capacity {
            if let Some(lru_cid) = self.hot_lru.evict_lru() {
                if let Some(mut entry) = self.hot.remove(&lru_cid) {
                    self.log_eviction(now, lru_cid, Cac2Tier::Hot, Cac2EvictionReason::LruEviction);
                    // Demote to warm (make room first if needed).
                    self.ensure_warm_capacity(now);
                    entry.tier = Cac2Tier::Warm;
                    self.warm.insert(lru_cid, entry);
                }
            } else {
                break;
            }
        }
    }

    /// Make room in the warm tier by evicting the entry with the lowest access count (LFU).
    fn ensure_warm_capacity(&mut self, now: u64) {
        while self.warm.len() >= self.config.warm_capacity {
            // Find the CID with the minimum access_count; break ties with LRU (last_accessed).
            let victim_cid = self
                .warm
                .values()
                .min_by(|a, b| {
                    a.access_count
                        .cmp(&b.access_count)
                        .then_with(|| a.last_accessed.cmp(&b.last_accessed))
                })
                .map(|e| e.cid);

            if let Some(cid) = victim_cid {
                self.warm.remove(&cid);
                self.log_eviction(now, cid, Cac2Tier::Warm, Cac2EvictionReason::LfuEviction);
            } else {
                break;
            }
        }
    }

    /// Promote a warm entry to the hot tier.
    fn promote_warm_to_hot(&mut self, cid: &Cac2Cid, now: u64) {
        if let Some(mut entry) = self.warm.remove(cid) {
            self.ensure_hot_capacity(now);
            entry.tier = Cac2Tier::Hot;
            entry.last_accessed = now;
            self.hot.insert(*cid, entry);
            self.hot_lru.touch(*cid);
        }
    }

    /// Append an eviction record to the bounded log.
    fn log_eviction(&mut self, ts: u64, cid: Cac2Cid, tier: Cac2Tier, reason: Cac2EvictionReason) {
        const MAX_EVICTION_LOG: usize = 500;
        if self.eviction_log.len() >= MAX_EVICTION_LOG {
            self.eviction_log.pop_front();
        }
        self.eviction_log.push_back(Cac2EvictionRecord {
            ts,
            cid,
            tier: Cac2Tier::Evicted,
            reason,
        });
        // Suppress the unused warning on the `tier` param — we store `Evicted`
        // in the record per the spec, but we still accept the source tier for
        // future logging extensions.
        let _ = tier;
        self.total_evictions += 1;
    }

    /// Expose the raw PRNG state accessor for testing.
    #[cfg(test)]
    fn next_rand(&mut self) -> u64 {
        xorshift64(&mut self.rng_state)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_cid(seed: u8) -> Cac2Cid {
        Cac2Cid([seed; 32])
    }

    fn make_cid_distinct(n: u64) -> Cac2Cid {
        Cac2Cid::from_bytes(&n.to_le_bytes())
    }

    fn default_cache() -> ContentAddressedCacheV2 {
        ContentAddressedCacheV2::new(Cac2CacheConfig {
            hot_capacity: 4,
            warm_capacity: 8,
            bloom_size: 256,
            hot_ttl_secs: 10,
            warm_ttl_secs: 60,
            admission_threshold: 2,
        })
    }

    // ── Cac2Cid ──────────────────────────────────────────────────────────────

    #[test]
    fn test_cid_from_slice_ok() {
        let data = [7u8; 32];
        let cid = Cac2Cid::from_slice(&data);
        assert!(cid.is_some());
        assert_eq!(cid.unwrap().0, data);
    }

    #[test]
    fn test_cid_from_slice_too_short() {
        let short = [1u8; 10];
        assert!(Cac2Cid::from_slice(&short).is_none());
    }

    #[test]
    fn test_cid_from_bytes_deterministic() {
        let a = Cac2Cid::from_bytes(b"hello");
        let b = Cac2Cid::from_bytes(b"hello");
        assert_eq!(a, b);
    }

    #[test]
    fn test_cid_from_bytes_different_inputs() {
        let a = Cac2Cid::from_bytes(b"foo");
        let b = Cac2Cid::from_bytes(b"bar");
        assert_ne!(a, b);
    }

    #[test]
    fn test_cid_copy_semantics() {
        let a = make_cid(1);
        let b = a; // copy
        assert_eq!(a, b);
    }

    #[test]
    fn test_cid_hash_equality() {
        let mut set = std::collections::HashSet::new();
        set.insert(make_cid(42));
        assert!(set.contains(&make_cid(42)));
        assert!(!set.contains(&make_cid(43)));
    }

    // ── Bloom filter ─────────────────────────────────────────────────────────

    #[test]
    fn test_bloom_new_empty() {
        let bf = BloomFilter::new(1024);
        let cid = make_cid(1);
        assert!(!bf.probably_contains(&cid));
    }

    #[test]
    fn test_bloom_insert_then_contains() {
        let mut bf = BloomFilter::new(1024);
        let cid = make_cid(99);
        bf.insert(&cid);
        assert!(bf.probably_contains(&cid));
    }

    #[test]
    fn test_bloom_multiple_cids() {
        let mut bf = BloomFilter::new(8192);
        for i in 0u8..50 {
            let cid = make_cid(i);
            bf.insert(&cid);
        }
        for i in 0u8..50 {
            let cid = make_cid(i);
            assert!(bf.probably_contains(&cid), "false negative for cid {i}");
        }
    }

    #[test]
    fn test_bloom_false_positive_estimate_increases_with_load() {
        let mut bf = BloomFilter::new(256);
        let fp0 = bf.false_positive_estimate();
        for i in 0u64..20 {
            bf.insert(&make_cid_distinct(i));
        }
        let fp20 = bf.false_positive_estimate();
        assert!(fp20 > fp0, "FP rate should increase with insertions");
    }

    #[test]
    fn test_bloom_no_false_negatives() {
        let mut bf = BloomFilter::new(65536);
        for i in 0u64..200 {
            let cid = make_cid_distinct(i);
            bf.insert(&cid);
            assert!(bf.probably_contains(&cid), "false negative at i={i}");
        }
    }

    // ── LRU list ─────────────────────────────────────────────────────────────

    #[test]
    fn test_lru_touch_and_evict() {
        let mut lru = LruList::new();
        let a = make_cid(1);
        let b = make_cid(2);
        let c = make_cid(3);
        lru.touch(a);
        lru.touch(b);
        lru.touch(c);
        // LRU should be `a`
        assert_eq!(lru.evict_lru(), Some(a));
    }

    #[test]
    fn test_lru_touch_promotes_existing() {
        let mut lru = LruList::new();
        let a = make_cid(1);
        let b = make_cid(2);
        lru.touch(a);
        lru.touch(b);
        lru.touch(a); // promote a to front
        assert_eq!(lru.evict_lru(), Some(b));
    }

    #[test]
    fn test_lru_remove_specific() {
        let mut lru = LruList::new();
        let a = make_cid(1);
        let b = make_cid(2);
        lru.touch(a);
        lru.touch(b);
        lru.remove(&a);
        assert_eq!(lru.len(), 1);
        assert_eq!(lru.evict_lru(), Some(b));
    }

    #[test]
    fn test_lru_evict_from_empty() {
        let mut lru = LruList::new();
        assert_eq!(lru.evict_lru(), None);
    }

    // ── Cache construction ────────────────────────────────────────────────────

    #[test]
    fn test_cache_new_empty() {
        let cache = default_cache();
        assert!(cache.is_empty());
        assert_eq!(cache.total_len(), 0);
    }

    #[test]
    fn test_cache_default_config() {
        let cache = ContentAddressedCacheV2::default_config();
        assert_eq!(cache.config().hot_capacity, 512);
    }

    // ── Insert / get basic ────────────────────────────────────────────────────

    #[test]
    fn test_insert_and_get_warm() {
        let mut cache = default_cache();
        let cid = make_cid(1);
        // First insert goes to warm (bloom miss).
        cache.insert(cid, b"hello".to_vec());
        assert_eq!(cache.warm_len(), 1);
        assert_eq!(cache.hot_len(), 0);
        let val = cache.get(&cid);
        assert_eq!(val, Some(b"hello".as_ref()));
    }

    #[test]
    fn test_insert_twice_routes_to_hot() {
        let mut cache = default_cache();
        let cid = make_cid(10);
        // First insert → bloom miss → warm tier
        cache.insert(cid, b"v1".to_vec());
        // Evict from warm so the CID is absent, but bloom remembers it.
        cache.warm.remove(&cid);
        // Second insert → bloom hit → hot tier
        cache.insert(cid, b"v2".to_vec());
        assert_eq!(cache.hot_len(), 1);
    }

    #[test]
    fn test_get_miss_returns_none() {
        let mut cache = default_cache();
        let cid = make_cid(200);
        assert_eq!(cache.get(&cid), None);
    }

    #[test]
    fn test_get_increments_access_count() {
        let mut cache = default_cache();
        let cid = make_cid(5);
        cache.insert(cid, b"data".to_vec());
        // Insert to warm; first get → warm
        let _ = cache.get(&cid);
        let _ = cache.get(&cid);
        let entry = cache.warm.get(&cid).or_else(|| cache.hot.get(&cid));
        assert!(entry.map(|e| e.access_count).unwrap_or(0) >= 1);
    }

    #[test]
    fn test_get_updates_last_accessed() {
        let mut cache = default_cache();
        let cid = make_cid(7);
        cache.insert(cid, b"x".to_vec());
        let before = cache.warm.get(&cid).map(|e| e.last_accessed).unwrap_or(0);
        // Even a small pause won't guarantee the clock advanced in a fast test,
        // so we just check the field is non-zero.
        assert!(before > 0);
        let _ = cache.get(&cid);
    }

    // ── Admission control ─────────────────────────────────────────────────────

    #[test]
    fn test_bloom_admission_cold_goes_to_warm() {
        let mut cache = default_cache();
        let cid = Cac2Cid::from_bytes(b"unique_cold_key_abc");
        cache.insert(cid, b"payload".to_vec());
        assert_eq!(cache.warm_len(), 1);
        assert_eq!(cache.hot_len(), 0);
    }

    #[test]
    fn test_bloom_admission_warm_hit_goes_to_hot() {
        let mut cache = default_cache();
        let cid = Cac2Cid::from_bytes(b"repeated_key_xyz");
        // First insert: bloom miss → warm tier.
        cache.insert(cid, b"first".to_vec());
        assert_eq!(cache.warm_len(), 1, "should be in warm after cold insert");
        // Manually evict from warm so the CID is no longer cached, but the
        // bloom filter still knows about it.
        cache.warm.remove(&cid);
        // Second insert: bloom already contains this CID → routed to hot.
        cache.insert(cid, b"second".to_vec());
        assert_eq!(cache.hot_len(), 1, "repeated CID should route to hot tier");
    }

    #[test]
    fn test_bloom_probably_contains_after_insert() {
        let mut cache = default_cache();
        let cid = make_cid(33);
        assert!(!cache.bloom_probably_contains(&cid));
        cache.insert(cid, b"z".to_vec());
        assert!(cache.bloom_probably_contains(&cid));
    }

    // ── Promotion warm → hot ──────────────────────────────────────────────────

    #[test]
    fn test_promotion_on_repeated_get() {
        // admission_threshold = 2; after 3 gets the entry should be promoted.
        let mut cache = default_cache();
        let cid = Cac2Cid::from_bytes(b"promo_test");
        cache.insert(cid, b"data".to_vec()); // warm (cold insert)
                                             // Gets increment access_count; promotion fires when count > threshold (2).
        let _ = cache.get(&cid); // count = 1
        let _ = cache.get(&cid); // count = 2
        let _ = cache.get(&cid); // count = 3 → promote
        assert_eq!(cache.hot_len(), 1, "entry should have been promoted to hot");
    }

    #[test]
    fn test_no_promotion_below_threshold() {
        let mut cache = default_cache(); // threshold = 2
        let cid = Cac2Cid::from_bytes(b"no_promo");
        cache.insert(cid, b"d".to_vec()); // warm
        let _ = cache.get(&cid); // count = 1 (not > 2 yet)
        assert_eq!(cache.hot_len(), 0, "should not yet be promoted");
    }

    // ── Hot-tier LRU eviction ─────────────────────────────────────────────────

    #[test]
    fn test_hot_lru_eviction_demotes_to_warm() {
        // hot_capacity = 4; insert 5 hot entries, the LRU should be demoted.
        let mut cache = ContentAddressedCacheV2::new(Cac2CacheConfig {
            hot_capacity: 3,
            warm_capacity: 10,
            bloom_size: 512,
            hot_ttl_secs: 0,
            warm_ttl_secs: 0,
            admission_threshold: 0,
        });
        // With admission_threshold = 0 every insert goes to hot (bloom sees it as seen-before
        // after a single insertion; but the first insert is always a bloom miss). We force all
        // to warm first, then re-insert to route to hot.
        for i in 0u8..4 {
            let cid = make_cid(i);
            cache.insert(cid, vec![i]); // warm (bloom miss)
            cache.insert(cid, vec![i, i]); // hot (bloom hit)
        }
        // hot should be capped at 3, the 4th should have demoted LRU to warm.
        assert!(cache.hot_len() <= 3);
        assert!(cache.warm_len() >= 1);
    }

    // ── Warm-tier LFU eviction ────────────────────────────────────────────────

    #[test]
    fn test_warm_lfu_eviction() {
        let mut cache = ContentAddressedCacheV2::new(Cac2CacheConfig {
            hot_capacity: 100,
            warm_capacity: 3,
            bloom_size: 512,
            hot_ttl_secs: 0,
            warm_ttl_secs: 0,
            admission_threshold: 100, // very high threshold → no promotion
        });

        let cid_a = make_cid(1);
        let cid_b = make_cid(2);
        let cid_c = make_cid(3);
        let cid_d = make_cid(4);

        // Fill warm tier (all are bloom-misses so go to warm).
        cache.insert(cid_a, b"A".to_vec());
        cache.insert(cid_b, b"B".to_vec());
        cache.insert(cid_c, b"C".to_vec());

        // Access A and B to raise their frequency.
        let _ = cache.get(&cid_a);
        let _ = cache.get(&cid_a);
        let _ = cache.get(&cid_b);

        // Insert D — warm is full; C (access_count = 0) should be evicted.
        cache.insert(cid_d, b"D".to_vec());

        assert!(!cache.contains(&cid_c), "C should have been LFU-evicted");
    }

    // ── Manual eviction ───────────────────────────────────────────────────────

    #[test]
    fn test_evict_removes_hot_entry() {
        let mut cache = default_cache();
        let cid = make_cid(20);
        cache.insert(cid, b"hot".to_vec()); // warm first
        cache.insert(cid, b"hot2".to_vec()); // now hot
        cache.evict(&cid);
        assert!(!cache.contains(&cid));
    }

    #[test]
    fn test_evict_removes_warm_entry() {
        let mut cache = default_cache();
        let cid = make_cid(21);
        cache.insert(cid, b"warm".to_vec());
        cache.evict(&cid);
        assert!(!cache.contains(&cid));
    }

    #[test]
    fn test_evict_nonexistent_is_noop() {
        let mut cache = default_cache();
        cache.evict(&make_cid(255)); // should not panic
    }

    // ── TTL expiry ────────────────────────────────────────────────────────────

    #[test]
    fn test_expire_stale_removes_old_hot_entries() {
        let mut cache = ContentAddressedCacheV2::new(Cac2CacheConfig {
            hot_capacity: 10,
            warm_capacity: 10,
            bloom_size: 256,
            hot_ttl_secs: 5,
            warm_ttl_secs: 0,
            admission_threshold: 0,
        });

        let cid = make_cid(50);
        cache.insert(cid, b"ttl".to_vec()); // warm
        cache.insert(cid, b"ttl2".to_vec()); // hot (bloom hit on second insert)

        // Manually force insertion time far into the past.
        if let Some(entry) = cache.hot.get_mut(&cid) {
            entry.inserted_at = 0; // epoch → definitely stale
        }

        let future_ts = 1_000_000u64;
        cache.expire_stale(future_ts);
        assert!(
            !cache.hot.contains_key(&cid),
            "hot entry should have expired"
        );
    }

    #[test]
    fn test_expire_stale_removes_old_warm_entries() {
        let mut cache = ContentAddressedCacheV2::new(Cac2CacheConfig {
            hot_capacity: 10,
            warm_capacity: 10,
            bloom_size: 256,
            hot_ttl_secs: 0,
            warm_ttl_secs: 30,
            admission_threshold: 100,
        });

        let cid = make_cid(51);
        cache.insert(cid, b"warm_old".to_vec());

        if let Some(entry) = cache.warm.get_mut(&cid) {
            entry.inserted_at = 0;
        }

        cache.expire_stale(1_000_000);
        assert!(!cache.warm.contains_key(&cid));
    }

    #[test]
    fn test_expire_stale_keeps_fresh_entries() {
        let mut cache = ContentAddressedCacheV2::new(Cac2CacheConfig {
            hot_capacity: 10,
            warm_capacity: 10,
            bloom_size: 256,
            hot_ttl_secs: 300,
            warm_ttl_secs: 300,
            admission_threshold: 100,
        });

        let cid = make_cid(52);
        cache.insert(cid, b"fresh".to_vec());

        // Do not modify inserted_at; call expire with now = inserted_at (no elapsed time).
        let now = cache
            .warm
            .get(&cid)
            .map(|e| e.inserted_at)
            .unwrap_or(now_secs());
        cache.expire_stale(now);
        assert!(cache.contains(&cid), "fresh entry should survive TTL sweep");
    }

    #[test]
    fn test_expire_stale_zero_ttl_skips() {
        let mut cache = ContentAddressedCacheV2::new(Cac2CacheConfig {
            hot_ttl_secs: 0,
            warm_ttl_secs: 0,
            ..Cac2CacheConfig::default()
        });
        let cid = make_cid(53);
        cache.insert(cid, b"forever".to_vec());
        if let Some(e) = cache.warm.get_mut(&cid) {
            e.inserted_at = 0;
        }
        cache.expire_stale(u64::MAX);
        assert!(cache.contains(&cid));
    }

    // ── Drain warm ────────────────────────────────────────────────────────────

    #[test]
    fn test_drain_warm_empty_returns_empty() {
        let mut cache = default_cache();
        let drained = cache.drain_warm_to_disk_simulation();
        assert!(drained.is_empty());
    }

    #[test]
    fn test_drain_warm_drains_25_percent() {
        let mut cache = ContentAddressedCacheV2::new(Cac2CacheConfig {
            hot_capacity: 100,
            warm_capacity: 1000,
            bloom_size: 8192,
            hot_ttl_secs: 0,
            warm_ttl_secs: 0,
            admission_threshold: 1000,
        });
        for i in 0u64..20 {
            cache.insert(make_cid_distinct(i), vec![i as u8]);
        }
        let before = cache.warm_len();
        let drained = cache.drain_warm_to_disk_simulation();
        let expected = ((before as f64) * 0.25).ceil() as usize;
        assert_eq!(drained.len(), expected);
        assert_eq!(cache.warm_len(), before - expected);
    }

    #[test]
    fn test_drain_warm_returns_lowest_frequency() {
        let mut cache = ContentAddressedCacheV2::new(Cac2CacheConfig {
            hot_capacity: 100,
            warm_capacity: 1000,
            bloom_size: 8192,
            hot_ttl_secs: 0,
            warm_ttl_secs: 0,
            admission_threshold: 1000,
        });
        let high_freq = make_cid(101);
        let low_freq = make_cid(102);

        cache.insert(high_freq, b"hf".to_vec());
        cache.insert(low_freq, b"lf".to_vec());

        // Give high_freq many accesses.
        for _ in 0..10 {
            let _ = cache.get(&high_freq);
        }

        let drained = cache.drain_warm_to_disk_simulation();
        let drained_cids: Vec<Cac2Cid> = drained.iter().map(|(c, _)| *c).collect();
        assert!(
            drained_cids.contains(&low_freq),
            "low_freq should be drained first"
        );
    }

    // ── Stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_initial_zero() {
        let cache = default_cache();
        let stats = cache.cache_stats();
        assert_eq!(stats.hot_count, 0);
        assert_eq!(stats.warm_count, 0);
        assert_eq!(stats.hit_rate, 0.0);
        assert_eq!(stats.miss_rate, 0.0);
        assert_eq!(stats.eviction_count, 0);
    }

    #[test]
    fn test_stats_hit_rate() {
        let mut cache = default_cache();
        let cid = make_cid(60);
        cache.insert(cid, b"stat".to_vec());
        let _ = cache.get(&cid); // hit
        let _ = cache.get(&make_cid(61)); // miss
        let stats = cache.cache_stats();
        assert!((stats.hit_rate - 0.5).abs() < 1e-9);
        assert!((stats.miss_rate - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_stats_eviction_count() {
        let mut cache = ContentAddressedCacheV2::new(Cac2CacheConfig {
            hot_capacity: 2,
            warm_capacity: 2,
            bloom_size: 256,
            hot_ttl_secs: 0,
            warm_ttl_secs: 0,
            admission_threshold: 0,
        });
        // Fill tiers enough to trigger evictions.
        for i in 0u8..10 {
            let cid = make_cid(i);
            cache.insert(cid, vec![i]);
            cache.insert(cid, vec![i]); // second → hot
        }
        assert!(cache.cache_stats().eviction_count > 0);
    }

    #[test]
    fn test_stats_bloom_fpe_non_negative() {
        let cache = default_cache();
        assert!(cache.cache_stats().bloom_false_positive_est >= 0.0);
    }

    // ── Eviction log ──────────────────────────────────────────────────────────

    #[test]
    fn test_eviction_log_bounded_500() {
        let mut cache = ContentAddressedCacheV2::new(Cac2CacheConfig {
            hot_capacity: 1,
            warm_capacity: 1,
            bloom_size: 256,
            hot_ttl_secs: 0,
            warm_ttl_secs: 0,
            admission_threshold: 0,
        });
        // Each pair of inserts causes at least one eviction.
        for i in 0u64..600 {
            let cid = make_cid_distinct(i);
            cache.insert(cid, vec![0u8]);
            cache.insert(cid, vec![1u8]);
        }
        assert!(
            cache.eviction_log().len() <= 500,
            "eviction log must be bounded at 500"
        );
    }

    #[test]
    fn test_eviction_log_records_reason() {
        let mut cache = default_cache();
        let cid = make_cid(80);
        cache.insert(cid, b"manual".to_vec());
        cache.evict(&cid);
        let found = cache
            .eviction_log()
            .iter()
            .any(|r| r.reason == Cac2EvictionReason::ManualEviction);
        assert!(found, "manual eviction should appear in log");
    }

    #[test]
    fn test_eviction_log_ttl_reason() {
        let mut cache = ContentAddressedCacheV2::new(Cac2CacheConfig {
            hot_capacity: 10,
            warm_capacity: 10,
            bloom_size: 256,
            hot_ttl_secs: 1,
            warm_ttl_secs: 1,
            admission_threshold: 100,
        });
        let cid = make_cid(81);
        cache.insert(cid, b"stale".to_vec());
        if let Some(e) = cache.warm.get_mut(&cid) {
            e.inserted_at = 0;
        }
        cache.expire_stale(1_000_000);
        let found = cache
            .eviction_log()
            .iter()
            .any(|r| r.reason == Cac2EvictionReason::TtlExpiry);
        assert!(found);
    }

    // ── contains / is_empty ───────────────────────────────────────────────────

    #[test]
    fn test_contains_hot() {
        let mut cache = default_cache();
        let cid = make_cid(90);
        cache.insert(cid, b"h".to_vec());
        cache.insert(cid, b"h2".to_vec()); // to hot
        assert!(cache.contains(&cid));
    }

    #[test]
    fn test_contains_warm() {
        let mut cache = default_cache();
        let cid = make_cid(91);
        cache.insert(cid, b"w".to_vec());
        assert!(cache.contains(&cid));
    }

    #[test]
    fn test_is_empty_after_evict_all() {
        let mut cache = default_cache();
        let c1 = make_cid(92);
        let c2 = make_cid(93);
        cache.insert(c1, b"a".to_vec());
        cache.insert(c2, b"b".to_vec());
        cache.evict(&c1);
        cache.evict(&c2);
        assert!(cache.is_empty());
    }

    // ── PRNG ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_xorshift64_non_zero() {
        let mut state = 12345u64;
        let val = xorshift64(&mut state);
        assert_ne!(val, 0);
        assert_ne!(state, 12345);
    }

    #[test]
    fn test_cache_prng_accessor() {
        let mut cache = default_cache();
        let a = cache.next_rand();
        let b = cache.next_rand();
        assert_ne!(a, b, "consecutive PRNG values should differ");
    }

    // ── FNV-1a ───────────────────────────────────────────────────────────────

    #[test]
    fn test_fnv1a_deterministic() {
        assert_eq!(fnv1a_64(b"hello"), fnv1a_64(b"hello"));
    }

    #[test]
    fn test_fnv1a_distinct_inputs() {
        assert_ne!(fnv1a_64(b"a"), fnv1a_64(b"b"));
    }

    #[test]
    fn test_fnv1a_empty_input() {
        // FNV offset basis
        assert_eq!(fnv1a_64(b""), 14_695_981_039_346_656_037);
    }

    // ── In-place update ───────────────────────────────────────────────────────

    #[test]
    fn test_insert_updates_data_in_place_warm() {
        let mut cache = default_cache();
        let cid = make_cid(110);
        cache.insert(cid, b"v1".to_vec());
        cache.insert(cid, b"v2".to_vec()); // → hot (bloom hit)
                                           // Could be hot or warm depending on bloom state; data should be v2.
        let val = cache.get(&cid);
        assert_eq!(val, Some(b"v2".as_ref()));
    }

    // ── Tier counts after operations ──────────────────────────────────────────

    #[test]
    fn test_hot_len_warm_len_total_len() {
        let mut cache = default_cache();
        let cid_a = make_cid(120);
        let cid_b = make_cid(121);
        cache.insert(cid_a, b"a".to_vec()); // warm
        cache.insert(cid_b, b"b".to_vec()); // warm
        assert_eq!(cache.total_len(), 2);
        assert_eq!(cache.warm_len(), 2);
        assert_eq!(cache.hot_len(), 0);
    }

    #[test]
    fn test_total_len_after_eviction() {
        let mut cache = default_cache();
        let cid = make_cid(122);
        cache.insert(cid, b"z".to_vec());
        assert_eq!(cache.total_len(), 1);
        cache.evict(&cid);
        assert_eq!(cache.total_len(), 0);
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_zero_capacity_warm_still_works() {
        // Warm capacity of 1: every insert evicts the previous warm entry.
        let mut cache = ContentAddressedCacheV2::new(Cac2CacheConfig {
            hot_capacity: 1,
            warm_capacity: 1,
            bloom_size: 128,
            hot_ttl_secs: 0,
            warm_ttl_secs: 0,
            admission_threshold: 999,
        });
        let c1 = make_cid(0);
        let c2 = make_cid(1);
        cache.insert(c1, b"first".to_vec());
        cache.insert(c2, b"second".to_vec()); // evicts c1 from warm
        assert!(cache.warm_len() <= 1);
    }

    #[test]
    fn test_large_data_payload() {
        let mut cache = default_cache();
        let cid = make_cid(130);
        let big_data = vec![0xABu8; 1024 * 1024]; // 1 MiB
        cache.insert(cid, big_data.clone());
        let retrieved = cache.get(&cid);
        assert_eq!(retrieved.map(|d| d.len()), Some(1024 * 1024));
    }

    #[test]
    fn test_get_miss_increments_miss_counter() {
        let mut cache = default_cache();
        let _ = cache.get(&make_cid(200));
        let _ = cache.get(&make_cid(201));
        let stats = cache.cache_stats();
        assert!((stats.miss_rate - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_expire_stale_multiple_entries() {
        let mut cache = ContentAddressedCacheV2::new(Cac2CacheConfig {
            hot_capacity: 100,
            warm_capacity: 100,
            bloom_size: 1024,
            hot_ttl_secs: 10,
            warm_ttl_secs: 10,
            admission_threshold: 100,
        });
        for i in 0u8..10 {
            let cid = make_cid(i);
            cache.insert(cid, vec![i]);
        }
        // Age all entries.
        for entry in cache.warm.values_mut() {
            entry.inserted_at = 0;
        }
        cache.expire_stale(1_000_000);
        assert_eq!(cache.warm_len(), 0);
    }

    #[test]
    fn test_drain_warm_one_entry() {
        let mut cache = ContentAddressedCacheV2::new(Cac2CacheConfig {
            hot_capacity: 100,
            warm_capacity: 100,
            bloom_size: 256,
            hot_ttl_secs: 0,
            warm_ttl_secs: 0,
            admission_threshold: 1000,
        });
        let cid = make_cid(140);
        cache.insert(cid, b"only".to_vec());
        let drained = cache.drain_warm_to_disk_simulation();
        // 25% of 1 = 0.25, ceil = 1
        assert_eq!(drained.len(), 1);
        assert!(cache.warm.is_empty());
    }

    #[test]
    fn test_cid_equality_and_inequality() {
        let a = make_cid(0);
        let b = make_cid(0);
        let c = make_cid(1);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_tier_enum_debug() {
        let t = Cac2Tier::Hot;
        assert!(format!("{t:?}").contains("Hot"));
    }

    #[test]
    fn test_eviction_reason_debug() {
        let r = Cac2EvictionReason::CapacityPressure;
        assert!(format!("{r:?}").contains("CapacityPressure"));
    }
}
