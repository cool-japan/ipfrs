//! Content-Addressable Cache with LRU eviction and TTL support.
//!
//! Provides a high-performance, in-memory cache keyed by content-derived CIDs (FNV-1a).
//! Supports multiple eviction policies, TTL expiration, tag-based grouping, and rich stats.

use std::collections::HashMap;

// ─── FNV-1a helpers ──────────────────────────────────────────────────────────

/// Compute FNV-1a 64-bit hash over `data`.
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

/// Derive a hex-encoded CID string from raw bytes.
fn compute_cid(data: &[u8]) -> String {
    format!("{:016x}", fnv1a_64(data))
}

// ─── Monotonic timestamp ─────────────────────────────────────────────────────

/// Return microseconds since an arbitrary epoch (uses `std::time::UNIX_EPOCH`).
fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

// ─── Public Types ─────────────────────────────────────────────────────────────

/// A single entry stored in the cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheEntry {
    /// Content-derived identifier.
    pub cid: String,
    /// Raw payload.
    pub data: Vec<u8>,
    /// Microsecond timestamp at insertion time.
    pub inserted_at: u64,
    /// Microsecond timestamp of most-recent access.
    pub last_accessed: u64,
    /// Total number of times this entry has been retrieved.
    pub access_count: u64,
    /// Optional TTL in microseconds (measured from `inserted_at`).
    pub ttl_us: Option<u64>,
    /// Payload size in bytes (mirrors `data.len()`).
    pub size_bytes: usize,
    /// Arbitrary string tags for group-based operations.
    pub tags: Vec<String>,
}

impl CacheEntry {
    /// Return `true` when the entry has passed its TTL deadline.
    pub fn is_expired(&self, now: u64) -> bool {
        match self.ttl_us {
            Some(ttl) => now.saturating_sub(self.inserted_at) >= ttl,
            None => false,
        }
    }

    /// Remaining TTL in microseconds, or `u64::MAX` for entries without TTL.
    pub fn remaining_ttl_us(&self, now: u64) -> u64 {
        match self.ttl_us {
            Some(ttl) => {
                let elapsed = now.saturating_sub(self.inserted_at);
                ttl.saturating_sub(elapsed)
            }
            None => u64::MAX,
        }
    }
}

/// Eviction policy controlling which entry is removed when the cache is full.
#[derive(Debug, Clone)]
pub enum EvictionPolicy {
    /// Least-Recently-Used: evict the entry accessed furthest in the past.
    Lru,
    /// Least-Frequently-Used: evict the entry with the lowest `access_count`.
    Lfu,
    /// TTL-first: evict the entry whose TTL expires soonest.
    TtlFirst,
    /// Size-weighted: evict the largest entry.
    SizeWeighted,
    /// Tagged: evict entries bearing `tag` first, then fall back to LRU.
    Tagged(String),
}

/// Configuration for [`ContentAddressableCache`].
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries.
    pub max_entries: usize,
    /// Maximum total payload bytes.
    pub max_bytes: usize,
    /// Default TTL applied when an entry is inserted without an explicit TTL.
    pub default_ttl_us: Option<u64>,
    /// Eviction policy used when the cache is over capacity.
    pub policy: EvictionPolicy,
    /// When `true` the cache maintains hit/miss/eviction counters.
    pub enable_stats: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1024,
            max_bytes: 64 * 1024 * 1024, // 64 MiB
            default_ttl_us: None,
            policy: EvictionPolicy::Lru,
            enable_stats: true,
        }
    }
}

/// Snapshot of cache performance counters.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub insertions: u64,
    pub expirations: u64,
    pub current_entries: usize,
    pub current_bytes: usize,
    /// Ratio of hits to total lookups; `0.0` when no lookups have occurred.
    pub hit_rate: f64,
}

/// Errors that can arise during cache operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheError {
    /// The provided CID does not match the CID derived from the data.
    CidMismatch { expected: String, got: String },
    /// The entry's payload exceeds the configured `max_bytes` limit.
    EntryTooLarge(usize),
    /// The cache is at capacity and eviction could not free enough space.
    CacheAtCapacity,
    /// No entry exists for the requested CID.
    EntryNotFound,
    /// The entry exists but its TTL has expired (it has been removed).
    TtlExpired,
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CacheError::CidMismatch { expected, got } => {
                write!(f, "CID mismatch: expected {expected}, got {got}")
            }
            CacheError::EntryTooLarge(sz) => write!(f, "entry too large: {sz} bytes"),
            CacheError::CacheAtCapacity => write!(f, "cache at capacity"),
            CacheError::EntryNotFound => write!(f, "entry not found"),
            CacheError::TtlExpired => write!(f, "entry TTL expired"),
        }
    }
}

impl std::error::Error for CacheError {}

// ─── LRU linked-list node ──────────────────────────────────────────────────

/// A node in the index-based doubly-linked LRU list.
#[derive(Debug, Clone)]
pub struct LruNode {
    pub cid: String,
    pub prev: Option<usize>,
    pub next: Option<usize>,
}

// ─── Core cache struct ────────────────────────────────────────────────────────

/// High-performance content-addressable cache.
///
/// Keyed by FNV-1a-derived CIDs; supports LRU, LFU, TTL-first, size-weighted,
/// and tag-targeted eviction policies. Thread-safety must be provided by the caller.
pub struct ContentAddressableCache {
    config: CacheConfig,

    /// The actual cached data, keyed by CID.
    entries: HashMap<String, CacheEntry>,

    /// Maps CID → slot index inside `lru_arena`.
    lru_index: HashMap<String, usize>,

    /// Arena of optional LRU nodes; freed slots are kept as `None` for reuse.
    lru_arena: Vec<Option<LruNode>>,

    /// Free-list of arena indices available for reuse.
    lru_free: Vec<usize>,

    /// Most-recently-used node (head of the list).
    lru_head: Option<usize>,

    /// Least-recently-used node (tail of the list).
    lru_tail: Option<usize>,

    /// Running byte total of all entry payloads.
    total_bytes: usize,

    // ── Stats ──────────────────────────────────────────────────────────
    hits: u64,
    misses: u64,
    evictions: u64,
    insertions: u64,
    expirations: u64,
}

impl ContentAddressableCache {
    // ── Construction ──────────────────────────────────────────────────

    /// Create a new cache with the given configuration.
    pub fn new(config: CacheConfig) -> Self {
        let cap = config.max_entries;
        Self {
            config,
            entries: HashMap::with_capacity(cap),
            lru_index: HashMap::with_capacity(cap),
            lru_arena: Vec::with_capacity(cap),
            lru_free: Vec::new(),
            lru_head: None,
            lru_tail: None,
            total_bytes: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            insertions: 0,
            expirations: 0,
        }
    }

    // ── Public API ────────────────────────────────────────────────────

    /// Insert `data` into the cache.
    ///
    /// Computes the CID automatically; applies `ttl_us` (or the configured
    /// default) and the given `tags`. Returns the derived CID on success.
    pub fn insert(
        &mut self,
        data: Vec<u8>,
        ttl_us: Option<u64>,
        tags: Vec<String>,
    ) -> Result<String, CacheError> {
        let cid = compute_cid(&data);
        self.insert_with_cid(cid.clone(), data, ttl_us, tags)?;
        Ok(cid)
    }

    /// Insert `data` under a caller-supplied `cid`.
    ///
    /// Verifies that the CID matches the data; returns
    /// [`CacheError::CidMismatch`] otherwise.
    pub fn insert_with_cid(
        &mut self,
        cid: String,
        data: Vec<u8>,
        ttl_us: Option<u64>,
        tags: Vec<String>,
    ) -> Result<(), CacheError> {
        // Verify CID.
        let expected = compute_cid(&data);
        if expected != cid {
            return Err(CacheError::CidMismatch { expected, got: cid });
        }

        let size = data.len();
        if size > self.config.max_bytes {
            return Err(CacheError::EntryTooLarge(size));
        }

        // If the entry already exists just update it in-place.
        if self.entries.contains_key(&cid) {
            let now = now_us();
            let entry = self
                .entries
                .get_mut(&cid)
                .ok_or(CacheError::EntryNotFound)?;
            self.total_bytes -= entry.size_bytes;
            entry.data = data;
            entry.inserted_at = now;
            entry.last_accessed = now;
            entry.ttl_us = ttl_us.or(self.config.default_ttl_us);
            entry.tags = tags;
            entry.size_bytes = size;
            entry.access_count = 0;
            self.total_bytes += size;
            self.lru_touch(&cid.clone());
            return Ok(());
        }

        // Evict expired entries first, then evict by policy until there is room.
        self.expire_ttl_internal();
        self.evict_to_fit_internal(size)?;

        let now = now_us();
        let effective_ttl = ttl_us.or(self.config.default_ttl_us);
        let entry = CacheEntry {
            cid: cid.clone(),
            data,
            inserted_at: now,
            last_accessed: now,
            access_count: 0,
            ttl_us: effective_ttl,
            size_bytes: size,
            tags,
        };

        self.total_bytes += size;
        self.entries.insert(cid.clone(), entry);
        self.lru_push_front(&cid);

        if self.config.enable_stats {
            self.insertions += 1;
        }
        Ok(())
    }

    /// Retrieve the data payload for `cid`.
    ///
    /// Updates LRU position and increments `access_count`. Returns
    /// [`CacheError::TtlExpired`] (and removes the entry) if the entry has
    /// expired.
    pub fn get(&mut self, cid: &str) -> Result<&[u8], CacheError> {
        let now = now_us();

        // Check existence & expiry before borrowing mutably.
        let expired = self.entries.get(cid).map(|e| e.is_expired(now));
        match expired {
            None => {
                if self.config.enable_stats {
                    self.misses += 1;
                }
                return Err(CacheError::EntryNotFound);
            }
            Some(true) => {
                self.remove_internal(cid);
                if self.config.enable_stats {
                    self.expirations += 1;
                    self.misses += 1;
                }
                return Err(CacheError::TtlExpired);
            }
            Some(false) => {}
        }

        // Update metadata.
        if let Some(entry) = self.entries.get_mut(cid) {
            entry.last_accessed = now;
            entry.access_count += 1;
        }
        self.lru_touch(cid);

        if self.config.enable_stats {
            self.hits += 1;
        }

        self.entries
            .get(cid)
            .map(|e| e.data.as_slice())
            .ok_or(CacheError::EntryNotFound)
    }

    /// Retrieve the full [`CacheEntry`] for `cid` (read-only, does NOT update
    /// LRU or access counts).
    pub fn get_entry(&self, cid: &str) -> Result<&CacheEntry, CacheError> {
        self.entries.get(cid).ok_or(CacheError::EntryNotFound)
    }

    /// Return `true` when an entry for `cid` exists and has not expired.
    /// Does **not** update LRU position.
    pub fn contains(&self, cid: &str) -> bool {
        match self.entries.get(cid) {
            Some(e) => !e.is_expired(now_us()),
            None => false,
        }
    }

    /// Remove and return the entry for `cid`.
    pub fn remove(&mut self, cid: &str) -> Result<CacheEntry, CacheError> {
        match self.remove_internal(cid) {
            Some(e) => Ok(e),
            None => Err(CacheError::EntryNotFound),
        }
    }

    /// Remove all entries bearing `tag` and return their CIDs.
    pub fn remove_by_tag(&mut self, tag: &str) -> Vec<String> {
        let cids: Vec<String> = self
            .entries
            .values()
            .filter(|e| e.tags.iter().any(|t| t == tag))
            .map(|e| e.cid.clone())
            .collect();

        for cid in &cids {
            self.remove_internal(cid);
        }
        cids
    }

    /// Scan the cache and remove all TTL-expired entries.
    ///
    /// Returns the CIDs that were removed.
    pub fn expire_ttl(&mut self) -> Vec<String> {
        self.expire_ttl_internal()
    }

    /// Evict entries per policy until `needed_bytes` additional bytes can be
    /// accommodated without exceeding `max_bytes`.
    ///
    /// Returns the CIDs evicted.
    pub fn evict_to_fit(&mut self, needed_bytes: usize) -> Vec<String> {
        let mut evicted = Vec::new();
        while self.total_bytes + needed_bytes > self.config.max_bytes
            || self.entries.len() >= self.config.max_entries
        {
            if self.entries.is_empty() {
                break;
            }
            if let Some(cid) = self.pick_eviction_candidate() {
                self.remove_internal(&cid);
                if self.config.enable_stats {
                    self.evictions += 1;
                }
                evicted.push(cid);
            } else {
                break;
            }
        }
        evicted
    }

    /// Return a snapshot of current cache statistics.
    pub fn stats(&self) -> CacheStats {
        let total_lookups = self.hits + self.misses;
        let hit_rate = if total_lookups > 0 {
            self.hits as f64 / total_lookups as f64
        } else {
            0.0
        };
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            insertions: self.insertions,
            expirations: self.expirations,
            current_entries: self.entries.len(),
            current_bytes: self.total_bytes,
            hit_rate,
        }
    }

    /// Remove all entries and return how many were cleared.
    pub fn clear(&mut self) -> usize {
        let count = self.entries.len();
        self.entries.clear();
        self.lru_index.clear();
        self.lru_arena.clear();
        self.lru_free.clear();
        self.lru_head = None;
        self.lru_tail = None;
        self.total_bytes = 0;
        count
    }

    /// Return the CIDs of all currently-cached entries.
    pub fn cids(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    // ── Internal helpers ──────────────────────────────────────────────

    fn expire_ttl_internal(&mut self) -> Vec<String> {
        let now = now_us();
        let expired: Vec<String> = self
            .entries
            .values()
            .filter(|e| e.is_expired(now))
            .map(|e| e.cid.clone())
            .collect();

        for cid in &expired {
            self.remove_internal(cid);
            if self.config.enable_stats {
                self.expirations += 1;
            }
        }
        expired
    }

    fn evict_to_fit_internal(&mut self, needed_bytes: usize) -> Result<(), CacheError> {
        let max_attempts = self.entries.len() + 1;
        let mut attempts = 0;

        while (self.total_bytes + needed_bytes > self.config.max_bytes
            || self.entries.len() >= self.config.max_entries)
            && !self.entries.is_empty()
        {
            if attempts >= max_attempts {
                return Err(CacheError::CacheAtCapacity);
            }
            attempts += 1;

            if let Some(cid) = self.pick_eviction_candidate() {
                self.remove_internal(&cid);
                if self.config.enable_stats {
                    self.evictions += 1;
                }
            } else {
                return Err(CacheError::CacheAtCapacity);
            }
        }

        if needed_bytes > self.config.max_bytes {
            return Err(CacheError::EntryTooLarge(needed_bytes));
        }
        Ok(())
    }

    /// Choose the CID of the best eviction candidate according to the policy.
    fn pick_eviction_candidate(&self) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }

        match &self.config.policy {
            EvictionPolicy::Lru => {
                // The LRU tail is the least-recently-used entry.
                self.lru_tail
                    .and_then(|idx| self.lru_arena.get(idx))
                    .and_then(|node| node.as_ref())
                    .map(|n| n.cid.clone())
            }
            EvictionPolicy::Lfu => self
                .entries
                .values()
                .min_by_key(|e| e.access_count)
                .map(|e| e.cid.clone()),
            EvictionPolicy::TtlFirst => {
                let now = now_us();
                self.entries
                    .values()
                    .min_by_key(|e| e.remaining_ttl_us(now))
                    .map(|e| e.cid.clone())
            }
            EvictionPolicy::SizeWeighted => self
                .entries
                .values()
                .max_by_key(|e| e.size_bytes)
                .map(|e| e.cid.clone()),
            EvictionPolicy::Tagged(tag) => {
                // Prefer tagged entries; fall back to LRU tail.
                let tagged = self
                    .entries
                    .values()
                    .find(|e| e.tags.iter().any(|t| t == tag))
                    .map(|e| e.cid.clone());

                tagged.or_else(|| {
                    self.lru_tail
                        .and_then(|idx| self.lru_arena.get(idx))
                        .and_then(|node| node.as_ref())
                        .map(|n| n.cid.clone())
                })
            }
        }
    }

    /// Remove an entry by CID, returning it if it existed.
    fn remove_internal(&mut self, cid: &str) -> Option<CacheEntry> {
        let entry = self.entries.remove(cid)?;
        self.total_bytes -= entry.size_bytes;
        self.lru_remove(cid);
        Some(entry)
    }

    // ── LRU linked-list operations ────────────────────────────────────

    /// Allocate a new arena slot (or reuse a freed one) and return its index.
    fn lru_alloc(&mut self, node: LruNode) -> usize {
        if let Some(idx) = self.lru_free.pop() {
            self.lru_arena[idx] = Some(node);
            idx
        } else {
            let idx = self.lru_arena.len();
            self.lru_arena.push(Some(node));
            idx
        }
    }

    /// Insert a new CID at the head (most-recently-used position).
    fn lru_push_front(&mut self, cid: &str) {
        let node = LruNode {
            cid: cid.to_owned(),
            prev: None,
            next: self.lru_head,
        };
        let idx = self.lru_alloc(node);

        if let Some(old_head) = self.lru_head {
            if let Some(Some(n)) = self.lru_arena.get_mut(old_head) {
                n.prev = Some(idx);
            }
        }

        self.lru_head = Some(idx);
        if self.lru_tail.is_none() {
            self.lru_tail = Some(idx);
        }

        self.lru_index.insert(cid.to_owned(), idx);
    }

    /// Move an existing entry to the head (access touch).
    fn lru_touch(&mut self, cid: &str) {
        // Detach from current position.
        let idx = match self.lru_index.get(cid).copied() {
            Some(i) => i,
            None => return,
        };

        let (prev, next) = {
            match self.lru_arena.get(idx).and_then(|n| n.as_ref()) {
                Some(n) => (n.prev, n.next),
                None => return,
            }
        };

        // Already at head — nothing to do.
        if prev.is_none() {
            return;
        }

        // Splice out.
        if let Some(p) = prev {
            if let Some(Some(pn)) = self.lru_arena.get_mut(p) {
                pn.next = next;
            }
        }
        if let Some(nx) = next {
            if let Some(Some(nn)) = self.lru_arena.get_mut(nx) {
                nn.prev = prev;
            }
        } else {
            // Was tail.
            self.lru_tail = prev;
        }

        // Re-link at head.
        if let Some(Some(n)) = self.lru_arena.get_mut(idx) {
            n.prev = None;
            n.next = self.lru_head;
        }
        if let Some(old_head) = self.lru_head {
            if let Some(Some(h)) = self.lru_arena.get_mut(old_head) {
                h.prev = Some(idx);
            }
        }
        self.lru_head = Some(idx);
    }

    /// Remove the LRU node associated with `cid`.
    fn lru_remove(&mut self, cid: &str) {
        let idx = match self.lru_index.remove(cid) {
            Some(i) => i,
            None => return,
        };

        let (prev, next) = {
            match self.lru_arena.get(idx).and_then(|n| n.as_ref()) {
                Some(n) => (n.prev, n.next),
                None => return,
            }
        };

        if let Some(p) = prev {
            if let Some(Some(pn)) = self.lru_arena.get_mut(p) {
                pn.next = next;
            }
        } else {
            self.lru_head = next;
        }

        if let Some(nx) = next {
            if let Some(Some(nn)) = self.lru_arena.get_mut(nx) {
                nn.prev = prev;
            }
        } else {
            self.lru_tail = prev;
        }

        self.lru_arena[idx] = None;
        self.lru_free.push(idx);
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Inline xorshift64 PRNG for test data (no `rand` crate) ───────────────

    struct Xorshift64 {
        state: u64,
    }

    impl Xorshift64 {
        fn new(seed: u64) -> Self {
            Self {
                state: if seed == 0 { 1 } else { seed },
            }
        }

        fn next(&mut self) -> u64 {
            self.state ^= self.state << 13;
            self.state ^= self.state >> 7;
            self.state ^= self.state << 17;
            self.state
        }

        fn bytes(&mut self, len: usize) -> Vec<u8> {
            let mut out = Vec::with_capacity(len);
            while out.len() < len {
                let v = self.next().to_le_bytes();
                for &b in &v {
                    if out.len() < len {
                        out.push(b);
                    }
                }
            }
            out
        }
    }

    fn default_cache() -> ContentAddressableCache {
        ContentAddressableCache::new(CacheConfig {
            max_entries: 64,
            max_bytes: 1024 * 1024,
            ..Default::default()
        })
    }

    // ── 1: Basic insert / get ─────────────────────────────────────────────────

    #[test]
    fn test_insert_returns_valid_cid() {
        let mut c = default_cache();
        let data = b"hello world".to_vec();
        let cid = c.insert(data.clone(), None, vec![]).unwrap();
        assert_eq!(cid.len(), 16);
        assert_eq!(cid, compute_cid(&data));
    }

    #[test]
    fn test_get_returns_correct_data() {
        let mut c = default_cache();
        let data = b"test data".to_vec();
        let cid = c.insert(data.clone(), None, vec![]).unwrap();
        assert_eq!(c.get(&cid).unwrap(), data.as_slice());
    }

    #[test]
    fn test_get_missing_returns_err() {
        let mut c = default_cache();
        assert_eq!(c.get("deadbeef00000000"), Err(CacheError::EntryNotFound));
    }

    #[test]
    fn test_contains_true_after_insert() {
        let mut c = default_cache();
        let cid = c.insert(b"abc".to_vec(), None, vec![]).unwrap();
        assert!(c.contains(&cid));
    }

    #[test]
    fn test_contains_false_for_missing() {
        let c = default_cache();
        assert!(!c.contains("0000000000000000"));
    }

    // ── 2: insert_with_cid ───────────────────────────────────────────────────

    #[test]
    fn test_insert_with_cid_success() {
        let mut c = default_cache();
        let data = b"verified".to_vec();
        let cid = compute_cid(&data);
        c.insert_with_cid(cid.clone(), data, None, vec![]).unwrap();
        assert!(c.contains(&cid));
    }

    #[test]
    fn test_insert_with_cid_mismatch_errors() {
        let mut c = default_cache();
        let data = b"real data".to_vec();
        let bad_cid = "aaaaaaaaaaaaaaaa".to_string();
        let res = c.insert_with_cid(bad_cid.clone(), data.clone(), None, vec![]);
        assert!(matches!(
            res,
            Err(CacheError::CidMismatch { got, .. }) if got == bad_cid
        ));
    }

    // ── 3: remove ────────────────────────────────────────────────────────────

    #[test]
    fn test_remove_existing_entry() {
        let mut c = default_cache();
        let cid = c.insert(b"remove me".to_vec(), None, vec![]).unwrap();
        let entry = c.remove(&cid).unwrap();
        assert_eq!(entry.cid, cid);
        assert!(!c.contains(&cid));
    }

    #[test]
    fn test_remove_missing_returns_err() {
        let mut c = default_cache();
        assert_eq!(c.remove("0000000000000001"), Err(CacheError::EntryNotFound));
    }

    // ── 4: get_entry ─────────────────────────────────────────────────────────

    #[test]
    fn test_get_entry_returns_metadata() {
        let mut c = default_cache();
        let data = b"entry meta".to_vec();
        let cid = c
            .insert(data.clone(), Some(999_999), vec!["tag1".into()])
            .unwrap();
        let entry = c.get_entry(&cid).unwrap();
        assert_eq!(entry.size_bytes, data.len());
        assert_eq!(entry.ttl_us, Some(999_999));
        assert!(entry.tags.contains(&"tag1".to_string()));
    }

    // ── 5: Access count increments ───────────────────────────────────────────

    #[test]
    fn test_access_count_increments_on_get() {
        let mut c = default_cache();
        let cid = c.insert(b"counted".to_vec(), None, vec![]).unwrap();
        c.get(&cid).unwrap();
        c.get(&cid).unwrap();
        assert_eq!(c.get_entry(&cid).unwrap().access_count, 2);
    }

    // ── 6: LRU ordering ──────────────────────────────────────────────────────

    #[test]
    fn test_lru_evicts_least_recently_used() {
        // max_entries = 3; insert a, b, c (c is MRU); then insert d → a evicted
        let mut c = ContentAddressableCache::new(CacheConfig {
            max_entries: 3,
            max_bytes: 1024 * 1024,
            policy: EvictionPolicy::Lru,
            ..Default::default()
        });

        let a = c.insert(b"aaaa".to_vec(), None, vec![]).unwrap();
        let b = c.insert(b"bbbb".to_vec(), None, vec![]).unwrap();
        let cc = c.insert(b"cccc".to_vec(), None, vec![]).unwrap();

        // Touch b so order is c (MRU) → b → a (LRU).
        // Actually insertion order: a inserted first (tail), c most recent (head).
        // Access b to move it to front: order becomes b → c → a (LRU tail = a).
        c.get(&b).unwrap();

        // Insert d → should evict a (LRU tail).
        let d = c.insert(b"dddd".to_vec(), None, vec![]).unwrap();
        assert!(!c.contains(&a), "a should have been evicted");
        assert!(c.contains(&b));
        assert!(c.contains(&cc));
        assert!(c.contains(&d));
    }

    #[test]
    fn test_lru_touch_moves_to_front() {
        let mut c = ContentAddressableCache::new(CacheConfig {
            max_entries: 3,
            max_bytes: 1024 * 1024,
            policy: EvictionPolicy::Lru,
            ..Default::default()
        });

        let a = c.insert(b"aaaa".to_vec(), None, vec![]).unwrap();
        let _b = c.insert(b"bbbb".to_vec(), None, vec![]).unwrap();
        let _cc = c.insert(b"cccc".to_vec(), None, vec![]).unwrap();

        // Touch a to move it to MRU.
        c.get(&a).unwrap();

        // Insert d — should NOT evict a.
        c.insert(b"dddd".to_vec(), None, vec![]).unwrap();
        assert!(c.contains(&a));
    }

    // ── 7: LFU eviction ──────────────────────────────────────────────────────

    #[test]
    fn test_lfu_evicts_least_frequently_used() {
        let mut c = ContentAddressableCache::new(CacheConfig {
            max_entries: 3,
            max_bytes: 1024 * 1024,
            policy: EvictionPolicy::Lfu,
            ..Default::default()
        });

        let a = c.insert(b"aaaa".to_vec(), None, vec![]).unwrap();
        let b = c.insert(b"bbbb".to_vec(), None, vec![]).unwrap();
        let cc = c.insert(b"cccc".to_vec(), None, vec![]).unwrap();

        // Access b and c multiple times so a stays at count 0.
        c.get(&b).unwrap();
        c.get(&b).unwrap();
        c.get(&cc).unwrap();

        // Insert d → a (access_count 0) should be evicted.
        let d = c.insert(b"dddd".to_vec(), None, vec![]).unwrap();
        assert!(!c.contains(&a));
        assert!(c.contains(&b));
        assert!(c.contains(&cc));
        assert!(c.contains(&d));
    }

    // ── 8: TTL expiration ────────────────────────────────────────────────────

    #[test]
    fn test_ttl_expired_entry_removed_on_get() {
        let mut c = default_cache();
        // TTL of 1 microsecond — will have expired by the next syscall.
        let cid = c.insert(b"expire me".to_vec(), Some(1), vec![]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert_eq!(c.get(&cid), Err(CacheError::TtlExpired));
        assert!(!c.contains(&cid));
    }

    #[test]
    fn test_expire_ttl_removes_expired_entries() {
        let mut c = default_cache();
        // Use a TTL long enough to survive the insert but short enough to expire after sleep.
        let cid1 = c.insert(b"exp1".to_vec(), Some(10_000), vec![]).unwrap();
        let cid2 = c.insert(b"keep".to_vec(), None, vec![]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let removed = c.expire_ttl();
        assert!(removed.contains(&cid1));
        assert!(!removed.contains(&cid2));
        assert!(!c.contains(&cid1));
        assert!(c.contains(&cid2));
    }

    #[test]
    fn test_contains_returns_false_for_expired() {
        let mut c = default_cache();
        let cid = c.insert(b"exp_check".to_vec(), Some(1), vec![]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(!c.contains(&cid));
    }

    #[test]
    fn test_default_ttl_applied() {
        let mut c = ContentAddressableCache::new(CacheConfig {
            default_ttl_us: Some(1),
            ..Default::default()
        });
        let cid = c.insert(b"default_ttl".to_vec(), None, vec![]).unwrap();
        let entry = c.get_entry(&cid).unwrap();
        assert_eq!(entry.ttl_us, Some(1));
    }

    // ── 9: TTLFirst eviction ─────────────────────────────────────────────────

    #[test]
    fn test_ttlfirst_evicts_soonest_expiring() {
        let mut c = ContentAddressableCache::new(CacheConfig {
            max_entries: 3,
            max_bytes: 1024 * 1024,
            policy: EvictionPolicy::TtlFirst,
            ..Default::default()
        });

        // short_ttl will expire first.
        let short = c.insert(b"short".to_vec(), Some(10), vec![]).unwrap();
        let _mid = c
            .insert(b"middl".to_vec(), Some(100_000_000), vec![])
            .unwrap();
        let _long = c.insert(b"longg".to_vec(), None, vec![]).unwrap();

        // Insert d → short should be evicted (soonest TTL).
        c.insert(b"ddddd".to_vec(), None, vec![]).unwrap();
        assert!(!c.contains(&short));
    }

    // ── 10: SizeWeighted eviction ────────────────────────────────────────────

    #[test]
    fn test_size_weighted_evicts_largest() {
        let mut c = ContentAddressableCache::new(CacheConfig {
            max_entries: 3,
            max_bytes: 1024 * 1024,
            policy: EvictionPolicy::SizeWeighted,
            ..Default::default()
        });

        let small = c.insert(b"s".to_vec(), None, vec![]).unwrap();
        let large = c.insert(vec![0u8; 512], None, vec![]).unwrap();
        let _medium = c.insert(vec![0u8; 128], None, vec![]).unwrap();

        c.insert(b"new".to_vec(), None, vec![]).unwrap();
        assert!(!c.contains(&large), "largest should be evicted");
        assert!(c.contains(&small));
    }

    // ── 11: Tagged eviction ──────────────────────────────────────────────────

    #[test]
    fn test_tagged_evicts_tagged_entries_first() {
        let mut c = ContentAddressableCache::new(CacheConfig {
            max_entries: 3,
            max_bytes: 1024 * 1024,
            policy: EvictionPolicy::Tagged("evict_me".to_string()),
            ..Default::default()
        });

        let tagged = c
            .insert(b"tagged_entry".to_vec(), None, vec!["evict_me".into()])
            .unwrap();
        let _untagged1 = c.insert(b"untagged_a".to_vec(), None, vec![]).unwrap();
        let _untagged2 = c.insert(b"untagged_b".to_vec(), None, vec![]).unwrap();

        c.insert(b"trigger_evict".to_vec(), None, vec![]).unwrap();
        assert!(!c.contains(&tagged));
    }

    #[test]
    fn test_tagged_falls_back_to_lru_when_no_tagged_entry() {
        let mut c = ContentAddressableCache::new(CacheConfig {
            max_entries: 3,
            max_bytes: 1024 * 1024,
            policy: EvictionPolicy::Tagged("missing_tag".to_string()),
            ..Default::default()
        });

        let a = c.insert(b"aaaaaa".to_vec(), None, vec![]).unwrap();
        let _b = c.insert(b"bbbbbb".to_vec(), None, vec![]).unwrap();
        let _cc = c.insert(b"cccccc".to_vec(), None, vec![]).unwrap();

        c.insert(b"dddddd".to_vec(), None, vec![]).unwrap();
        // LRU tail (a) should be evicted as fallback.
        assert!(!c.contains(&a));
    }

    // ── 12: remove_by_tag ────────────────────────────────────────────────────

    #[test]
    fn test_remove_by_tag_removes_all_tagged() {
        let mut c = default_cache();
        let t1 = c
            .insert(b"t1data".to_vec(), None, vec!["group".into()])
            .unwrap();
        let t2 = c
            .insert(b"t2data".to_vec(), None, vec!["group".into()])
            .unwrap();
        let keep = c
            .insert(b"keepdt".to_vec(), None, vec!["other".into()])
            .unwrap();

        let removed = c.remove_by_tag("group");
        assert!(removed.contains(&t1));
        assert!(removed.contains(&t2));
        assert!(!c.contains(&t1));
        assert!(!c.contains(&t2));
        assert!(c.contains(&keep));
    }

    #[test]
    fn test_remove_by_tag_returns_empty_when_no_match() {
        let mut c = default_cache();
        c.insert(b"data".to_vec(), None, vec!["other".into()])
            .unwrap();
        assert!(c.remove_by_tag("nonexistent").is_empty());
    }

    // ── 13: Capacity enforcement ─────────────────────────────────────────────

    #[test]
    fn test_entry_count_never_exceeds_max() {
        let mut c = ContentAddressableCache::new(CacheConfig {
            max_entries: 10,
            max_bytes: 1024 * 1024,
            ..Default::default()
        });
        let mut rng = Xorshift64::new(42);
        for _ in 0..20 {
            let data = rng.bytes(16);
            let _ = c.insert(data, None, vec![]);
        }
        assert!(c.stats().current_entries <= 10);
    }

    #[test]
    fn test_byte_limit_never_exceeded() {
        let limit = 1024_usize;
        let mut c = ContentAddressableCache::new(CacheConfig {
            max_entries: 1024,
            max_bytes: limit,
            ..Default::default()
        });
        let mut rng = Xorshift64::new(99);
        for _ in 0..50 {
            let data = rng.bytes(64);
            let _ = c.insert(data, None, vec![]);
        }
        assert!(c.stats().current_bytes <= limit);
    }

    #[test]
    fn test_entry_too_large_rejected() {
        let mut c = ContentAddressableCache::new(CacheConfig {
            max_entries: 64,
            max_bytes: 100,
            ..Default::default()
        });
        let res = c.insert(vec![0u8; 200], None, vec![]);
        assert_eq!(res, Err(CacheError::EntryTooLarge(200)));
    }

    // ── 14: Stats accuracy ───────────────────────────────────────────────────

    #[test]
    fn test_stats_hits_and_misses() {
        let mut c = default_cache();
        let cid = c.insert(b"stats_test".to_vec(), None, vec![]).unwrap();
        c.get(&cid).unwrap();
        c.get(&cid).unwrap();
        let _ = c.get("000000000000cafe");
        let s = c.stats();
        assert_eq!(s.hits, 2);
        assert_eq!(s.misses, 1);
    }

    #[test]
    fn test_stats_hit_rate() {
        let mut c = default_cache();
        let cid = c.insert(b"hr".to_vec(), None, vec![]).unwrap();
        c.get(&cid).unwrap(); // hit
        let _ = c.get("nope00000000cafe"); // miss
        let s = c.stats();
        assert!((s.hit_rate - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_insertions_count() {
        let mut c = default_cache();
        c.insert(b"a".to_vec(), None, vec![]).unwrap();
        c.insert(b"b".to_vec(), None, vec![]).unwrap();
        c.insert(b"c".to_vec(), None, vec![]).unwrap();
        assert_eq!(c.stats().insertions, 3);
    }

    #[test]
    fn test_stats_eviction_count() {
        let mut c = ContentAddressableCache::new(CacheConfig {
            max_entries: 2,
            max_bytes: 1024 * 1024,
            ..Default::default()
        });
        c.insert(b"first_".to_vec(), None, vec![]).unwrap();
        c.insert(b"second".to_vec(), None, vec![]).unwrap();
        c.insert(b"third_".to_vec(), None, vec![]).unwrap(); // triggers eviction
        assert!(c.stats().evictions >= 1);
    }

    #[test]
    fn test_stats_current_bytes() {
        let mut c = default_cache();
        let data = vec![0u8; 128];
        c.insert(data, None, vec![]).unwrap();
        assert_eq!(c.stats().current_bytes, 128);
    }

    #[test]
    fn test_stats_current_entries() {
        let mut c = default_cache();
        c.insert(b"e1".to_vec(), None, vec![]).unwrap();
        c.insert(b"e2".to_vec(), None, vec![]).unwrap();
        assert_eq!(c.stats().current_entries, 2);
    }

    #[test]
    fn test_stats_expirations_count() {
        let mut c = default_cache();
        c.insert(b"will_expire".to_vec(), Some(1), vec![]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        c.expire_ttl();
        assert_eq!(c.stats().expirations, 1);
    }

    // ── 15: clear ────────────────────────────────────────────────────────────

    #[test]
    fn test_clear_removes_all() {
        let mut c = default_cache();
        c.insert(b"x".to_vec(), None, vec![]).unwrap();
        c.insert(b"y".to_vec(), None, vec![]).unwrap();
        let n = c.clear();
        assert_eq!(n, 2);
        assert_eq!(c.stats().current_entries, 0);
        assert_eq!(c.stats().current_bytes, 0);
    }

    // ── 16: cids() ───────────────────────────────────────────────────────────

    #[test]
    fn test_cids_returns_all_current_cids() {
        let mut c = default_cache();
        let cid1 = c.insert(b"cid_test_1".to_vec(), None, vec![]).unwrap();
        let cid2 = c.insert(b"cid_test_2".to_vec(), None, vec![]).unwrap();
        let cids = c.cids();
        assert!(cids.contains(&cid1));
        assert!(cids.contains(&cid2));
        assert_eq!(cids.len(), 2);
    }

    // ── 17: evict_to_fit ─────────────────────────────────────────────────────

    #[test]
    fn test_evict_to_fit_makes_room() {
        let limit = 256_usize;
        let mut c = ContentAddressableCache::new(CacheConfig {
            max_entries: 64,
            max_bytes: limit,
            ..Default::default()
        });
        // Fill to capacity with 32-byte payloads.
        let mut rng = Xorshift64::new(77);
        for _ in 0..8 {
            let data = rng.bytes(32);
            c.insert(data, None, vec![]).unwrap();
        }
        assert!(c.stats().current_bytes <= limit);
        // Now request 32 more bytes.
        let evicted = c.evict_to_fit(32);
        assert!(!evicted.is_empty() || c.stats().current_bytes + 32 <= limit);
    }

    // ── 18: Duplicate insert updates in place ────────────────────────────────

    #[test]
    fn test_duplicate_insert_updates_entry() {
        let mut c = default_cache();
        let data = b"same data".to_vec();
        let cid = c
            .insert(data.clone(), Some(1_000_000), vec!["old".into()])
            .unwrap();
        // Re-insert same data with different TTL and tags.
        c.insert(data.clone(), Some(9_999_999), vec!["new".into()])
            .unwrap();
        let entry = c.get_entry(&cid).unwrap();
        assert_eq!(entry.ttl_us, Some(9_999_999));
        assert!(entry.tags.contains(&"new".to_string()));
        // Entry count should still be 1.
        assert_eq!(c.stats().current_entries, 1);
    }

    // ── 19: CID determinism ──────────────────────────────────────────────────

    #[test]
    fn test_cid_is_deterministic() {
        let data = b"deterministic payload";
        let c1 = compute_cid(data);
        let c2 = compute_cid(data);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_different_data_different_cid() {
        let c1 = compute_cid(b"apple");
        let c2 = compute_cid(b"orange");
        assert_ne!(c1, c2);
    }

    // ── 20: LRU list integrity under many operations ─────────────────────────

    #[test]
    fn test_lru_integrity_many_inserts() {
        let mut c = ContentAddressableCache::new(CacheConfig {
            max_entries: 8,
            max_bytes: 1024 * 1024,
            policy: EvictionPolicy::Lru,
            ..Default::default()
        });
        let mut rng = Xorshift64::new(1234);
        let mut inserted = Vec::new();
        for _ in 0..20 {
            let data = rng.bytes(16);
            if let Ok(cid) = c.insert(data, None, vec![]) {
                inserted.push(cid);
            }
        }
        assert!(c.stats().current_entries <= 8);
    }

    // ── 21: Zero-byte payload ────────────────────────────────────────────────

    #[test]
    fn test_zero_byte_insert_and_get() {
        let mut c = default_cache();
        let cid = c.insert(vec![], None, vec![]).unwrap();
        assert_eq!(c.get(&cid).unwrap(), &[] as &[u8]);
    }

    // ── 22: Large single entry at max_bytes limit ─────────────────────────────

    #[test]
    fn test_insert_exactly_at_byte_limit() {
        let limit = 512;
        let mut c = ContentAddressableCache::new(CacheConfig {
            max_entries: 64,
            max_bytes: limit,
            ..Default::default()
        });
        let data = vec![0u8; limit];
        c.insert(data, None, vec![]).unwrap();
        assert_eq!(c.stats().current_bytes, limit);
    }

    // ── 23: Tag with multiple tagged entries ─────────────────────────────────

    #[test]
    fn test_entry_can_have_multiple_tags() {
        let mut c = default_cache();
        let cid = c
            .insert(
                b"multi_tag".to_vec(),
                None,
                vec!["a".into(), "b".into(), "c".into()],
            )
            .unwrap();
        let entry = c.get_entry(&cid).unwrap();
        assert_eq!(entry.tags.len(), 3);
    }

    // ── 24: Stats disabled ───────────────────────────────────────────────────

    #[test]
    fn test_stats_disabled_no_counters() {
        let mut c = ContentAddressableCache::new(CacheConfig {
            enable_stats: false,
            ..Default::default()
        });
        let cid = c.insert(b"no_stats".to_vec(), None, vec![]).unwrap();
        c.get(&cid).unwrap();
        let s = c.stats();
        // Stats counters remain at default (0) since disabled.
        assert_eq!(s.hits, 0);
        assert_eq!(s.insertions, 0);
    }

    // ── 25: LRU order after remove ───────────────────────────────────────────

    #[test]
    fn test_lru_order_after_remove() {
        let mut c = ContentAddressableCache::new(CacheConfig {
            max_entries: 4,
            max_bytes: 1024 * 1024,
            policy: EvictionPolicy::Lru,
            ..Default::default()
        });

        let a = c.insert(b"aaaaa_lru".to_vec(), None, vec![]).unwrap();
        let b = c.insert(b"bbbbb_lru".to_vec(), None, vec![]).unwrap();
        let cc = c.insert(b"ccccc_lru".to_vec(), None, vec![]).unwrap();

        // Remove the MRU (c) and the tail (a).
        c.remove(&cc).unwrap();
        c.remove(&a).unwrap();

        // Only b remains; insert 4 more — b should survive until the 4th.
        c.insert(b"d_lru_test".to_vec(), None, vec![]).unwrap();
        c.insert(b"e_lru_test".to_vec(), None, vec![]).unwrap();
        c.insert(b"f_lru_test".to_vec(), None, vec![]).unwrap();
        // At max_entries = 4 with b + d + e + f, inserting g evicts b (LRU tail).
        c.insert(b"g_lru_test".to_vec(), None, vec![]).unwrap();
        assert!(!c.contains(&b));
    }

    // ── 26: Mixed TTL and non-TTL entries ────────────────────────────────────

    #[test]
    fn test_mixed_ttl_and_no_ttl() {
        let mut c = default_cache();
        // Insert both entries first, then let TTL expire.
        let exp = c
            .insert(b"expiring_x".to_vec(), Some(10_000), vec![])
            .unwrap();
        let keep = c.insert(b"permanent_y".to_vec(), None, vec![]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        c.expire_ttl();
        assert!(!c.contains(&exp));
        assert!(c.contains(&keep));
    }

    // ── 27: get miss updates miss counter ───────────────────────────────────

    #[test]
    fn test_miss_counter_increments() {
        let mut c = default_cache();
        let _ = c.get("missing_cid_cafe01");
        let _ = c.get("missing_cid_cafe02");
        assert_eq!(c.stats().misses, 2);
    }

    // ── 28: expiration triggers miss counter ─────────────────────────────────

    #[test]
    fn test_expired_get_counts_as_miss() {
        let mut c = default_cache();
        let cid = c.insert(b"short_live".to_vec(), Some(1), vec![]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let _ = c.get(&cid);
        assert_eq!(c.stats().misses, 1);
    }

    // ── 29: evict_to_fit with zero needed ────────────────────────────────────

    #[test]
    fn test_evict_to_fit_zero_needed() {
        let mut c = default_cache();
        c.insert(b"zero_fit".to_vec(), None, vec![]).unwrap();
        let evicted = c.evict_to_fit(0);
        assert!(evicted.is_empty());
    }

    // ── 30: CID format is 16 hex chars ────────────────────────────────────────

    #[test]
    fn test_cid_is_16_hex_chars() {
        let mut rng = Xorshift64::new(55);
        let mut c = default_cache();
        for _ in 0..10 {
            let data = rng.bytes(32);
            let cid = c.insert(data, None, vec![]).unwrap();
            assert_eq!(cid.len(), 16, "CID should be 16 hex chars");
            assert!(cid.chars().all(|ch| ch.is_ascii_hexdigit()));
        }
    }

    // ── 31: Multiple tags on remove_by_tag ───────────────────────────────────

    #[test]
    fn test_remove_by_tag_multi_tag_entries() {
        let mut c = default_cache();
        let cid = c
            .insert(
                b"multi_tagged_e".to_vec(),
                None,
                vec!["x".into(), "y".into()],
            )
            .unwrap();
        let removed = c.remove_by_tag("x");
        assert!(removed.contains(&cid));
        let removed2 = c.remove_by_tag("y");
        assert!(removed2.is_empty()); // already removed
    }

    // ── 32: Clear resets byte count ──────────────────────────────────────────

    #[test]
    fn test_clear_resets_byte_count() {
        let mut c = default_cache();
        c.insert(vec![0u8; 256], None, vec![]).unwrap();
        c.clear();
        assert_eq!(c.stats().current_bytes, 0);
    }

    // ── 33: get_entry on missing CID ─────────────────────────────────────────

    #[test]
    fn test_get_entry_missing_returns_err() {
        let c = default_cache();
        assert_eq!(
            c.get_entry("cafecafecafecafe"),
            Err(CacheError::EntryNotFound)
        );
    }

    // ── 34: insert after clear works ─────────────────────────────────────────

    #[test]
    fn test_insert_after_clear() {
        let mut c = default_cache();
        c.insert(b"before".to_vec(), None, vec![]).unwrap();
        c.clear();
        let cid = c.insert(b"after ".to_vec(), None, vec![]).unwrap();
        assert!(c.contains(&cid));
    }

    // ── 35: LFU with all equal counts falls back to any entry ────────────────

    #[test]
    fn test_lfu_equal_counts_evicts_one() {
        let mut c = ContentAddressableCache::new(CacheConfig {
            max_entries: 2,
            max_bytes: 1024 * 1024,
            policy: EvictionPolicy::Lfu,
            ..Default::default()
        });
        c.insert(b"lfu_a".to_vec(), None, vec![]).unwrap();
        c.insert(b"lfu_b".to_vec(), None, vec![]).unwrap();
        // Insert third — one of the two equal-count entries gets evicted.
        c.insert(b"lfu_c".to_vec(), None, vec![]).unwrap();
        assert_eq!(c.stats().current_entries, 2);
    }

    // ── 36: bytes freed when entry removed ───────────────────────────────────

    #[test]
    fn test_bytes_freed_on_remove() {
        let mut c = default_cache();
        let cid = c.insert(vec![0u8; 100], None, vec![]).unwrap();
        assert_eq!(c.stats().current_bytes, 100);
        c.remove(&cid).unwrap();
        assert_eq!(c.stats().current_bytes, 0);
    }

    // ── 37: entries freed on expire_ttl ──────────────────────────────────────

    #[test]
    fn test_bytes_freed_on_expire() {
        let mut c = default_cache();
        c.insert(vec![0u8; 200], Some(1), vec![]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        c.expire_ttl();
        assert_eq!(c.stats().current_bytes, 0);
    }

    // ── 38: size-weighted with equal sizes falls back to any ─────────────────

    #[test]
    fn test_size_weighted_equal_sizes() {
        let mut c = ContentAddressableCache::new(CacheConfig {
            max_entries: 2,
            max_bytes: 1024 * 1024,
            policy: EvictionPolicy::SizeWeighted,
            ..Default::default()
        });
        c.insert(b"sw_aa".to_vec(), None, vec![]).unwrap();
        c.insert(b"sw_bb".to_vec(), None, vec![]).unwrap();
        c.insert(b"sw_cc".to_vec(), None, vec![]).unwrap();
        assert_eq!(c.stats().current_entries, 2);
    }

    // ── 39: Byte-limit enforcement with exact fit ─────────────────────────────

    #[test]
    fn test_byte_limit_exact_fit() {
        let limit = 100_usize;
        let mut c = ContentAddressableCache::new(CacheConfig {
            max_entries: 64,
            max_bytes: limit,
            ..Default::default()
        });
        // Insert two 50-byte entries.
        c.insert(vec![1u8; 50], None, vec![]).unwrap();
        c.insert(vec![2u8; 50], None, vec![]).unwrap();
        assert_eq!(c.stats().current_bytes, 100);
        // Insert one more 50-byte entry — one existing entry gets evicted.
        c.insert(vec![3u8; 50], None, vec![]).unwrap();
        assert!(c.stats().current_bytes <= 100);
    }

    // ── 40: last_accessed updates on get ─────────────────────────────────────

    #[test]
    fn test_last_accessed_updates_on_get() {
        let mut c = default_cache();
        let cid = c.insert(b"access_ts".to_vec(), None, vec![]).unwrap();
        let before = c.get_entry(&cid).unwrap().last_accessed;
        std::thread::sleep(std::time::Duration::from_millis(2));
        c.get(&cid).unwrap();
        let after = c.get_entry(&cid).unwrap().last_accessed;
        assert!(after >= before);
    }

    // ── 41: inserted_at is set on insert ─────────────────────────────────────

    #[test]
    fn test_inserted_at_is_set() {
        let mut c = default_cache();
        let cid = c.insert(b"timestamp".to_vec(), None, vec![]).unwrap();
        let entry = c.get_entry(&cid).unwrap();
        assert!(entry.inserted_at > 0);
    }

    // ── 42: Expire TTL on empty cache ────────────────────────────────────────

    #[test]
    fn test_expire_ttl_empty_cache() {
        let mut c = default_cache();
        let removed = c.expire_ttl();
        assert!(removed.is_empty());
    }

    // ── 43: remove_by_tag on empty cache ─────────────────────────────────────

    #[test]
    fn test_remove_by_tag_empty_cache() {
        let mut c = default_cache();
        assert!(c.remove_by_tag("any_tag").is_empty());
    }

    // ── 44: TTLFirst with all no-TTL entries falls back to max u64 ────────────

    #[test]
    fn test_ttl_first_no_ttl_entries() {
        let mut c = ContentAddressableCache::new(CacheConfig {
            max_entries: 2,
            max_bytes: 1024 * 1024,
            policy: EvictionPolicy::TtlFirst,
            ..Default::default()
        });
        c.insert(b"no_ttl_x".to_vec(), None, vec![]).unwrap();
        c.insert(b"no_ttl_y".to_vec(), None, vec![]).unwrap();
        c.insert(b"no_ttl_z".to_vec(), None, vec![]).unwrap();
        // Should evict one entry (the one with max remaining TTL = u64::MAX, any of them).
        assert_eq!(c.stats().current_entries, 2);
    }

    // ── 45: CacheError display messages ──────────────────────────────────────

    #[test]
    fn test_cache_error_display() {
        let e = CacheError::EntryNotFound;
        assert!(!e.to_string().is_empty());
        let e2 = CacheError::EntryTooLarge(999);
        assert!(e2.to_string().contains("999"));
        let e3 = CacheError::CidMismatch {
            expected: "aaa".into(),
            got: "bbb".into(),
        };
        assert!(e3.to_string().contains("aaa"));
    }

    // ── 46: CacheStats default hit_rate is 0.0 ────────────────────────────────

    #[test]
    fn test_stats_hit_rate_zero_when_no_lookups() {
        let c = default_cache();
        assert_eq!(c.stats().hit_rate, 0.0);
    }

    // ── 47: Arena slot reuse after remove ────────────────────────────────────

    #[test]
    fn test_lru_arena_slot_reused() {
        let mut c = ContentAddressableCache::new(CacheConfig {
            max_entries: 4,
            max_bytes: 1024 * 1024,
            ..Default::default()
        });
        let a = c.insert(b"arena_a".to_vec(), None, vec![]).unwrap();
        c.insert(b"arena_b".to_vec(), None, vec![]).unwrap();
        c.remove(&a).unwrap();
        // After removing a, its arena slot should be in the free list.
        assert!(!c.lru_free.is_empty());
        // Inserting another entry should reuse the freed slot.
        c.insert(b"arena_c".to_vec(), None, vec![]).unwrap();
        assert!(c.lru_free.is_empty() || c.lru_arena.len() <= 3);
    }

    // ── 48: Stress test with random operations ────────────────────────────────

    #[test]
    fn test_stress_random_operations() {
        let mut c = ContentAddressableCache::new(CacheConfig {
            max_entries: 32,
            max_bytes: 4096,
            policy: EvictionPolicy::Lru,
            ..Default::default()
        });
        let mut rng = Xorshift64::new(999);
        let mut cids: Vec<String> = Vec::new();

        for _ in 0..200 {
            let op = rng.next() % 4;
            match op {
                0 => {
                    // Insert.
                    let sz = ((rng.next() % 128) + 1) as usize;
                    let data = rng.bytes(sz);
                    if let Ok(cid) = c.insert(data, None, vec![]) {
                        cids.push(cid);
                    }
                }
                1 => {
                    // Get.
                    if !cids.is_empty() {
                        let idx = (rng.next() as usize) % cids.len();
                        let _ = c.get(&cids[idx].clone());
                    }
                }
                2 => {
                    // Remove.
                    if !cids.is_empty() {
                        let idx = (rng.next() as usize) % cids.len();
                        let cid = cids.swap_remove(idx);
                        let _ = c.remove(&cid);
                    }
                }
                _ => {
                    c.expire_ttl();
                }
            }
        }
        // Invariant: bytes and entries within bounds.
        let s = c.stats();
        assert!(s.current_entries <= 32);
        assert!(s.current_bytes <= 4096);
    }

    // ── 49: insert same CID twice preserves single entry ─────────────────────

    #[test]
    fn test_duplicate_cid_single_entry() {
        let mut c = default_cache();
        let data = b"unique_content".to_vec();
        c.insert(data.clone(), None, vec![]).unwrap();
        c.insert(data, None, vec![]).unwrap();
        assert_eq!(c.stats().current_entries, 1);
    }

    // ── 50: Entry size in stats matches data len ──────────────────────────────

    #[test]
    fn test_entry_size_bytes_matches_data_len() {
        let mut c = default_cache();
        let data = vec![42u8; 77];
        let cid = c.insert(data.clone(), None, vec![]).unwrap();
        let entry = c.get_entry(&cid).unwrap();
        assert_eq!(entry.size_bytes, data.len());
        assert_eq!(c.stats().current_bytes, 77);
    }

    // ── 51: LRU head/tail consistency after single entry ────────────────────

    #[test]
    fn test_lru_single_entry_head_equals_tail() {
        let mut c = default_cache();
        c.insert(b"only_one_entry".to_vec(), None, vec![]).unwrap();
        assert!(c.lru_head.is_some());
        assert!(c.lru_tail.is_some());
        assert_eq!(c.lru_head, c.lru_tail);
    }

    // ── 52: LRU head/tail cleared after all entries removed ─────────────────

    #[test]
    fn test_lru_empty_after_clear() {
        let mut c = default_cache();
        c.insert(b"entry_to_clear".to_vec(), None, vec![]).unwrap();
        c.clear();
        assert!(c.lru_head.is_none());
        assert!(c.lru_tail.is_none());
    }

    // ── 53: hit_rate is 1.0 when all lookups are hits ─────────────────────────

    #[test]
    fn test_hit_rate_all_hits() {
        let mut c = default_cache();
        let cid = c.insert(b"all_hits".to_vec(), None, vec![]).unwrap();
        c.get(&cid).unwrap();
        c.get(&cid).unwrap();
        c.get(&cid).unwrap();
        assert!((c.stats().hit_rate - 1.0).abs() < f64::EPSILON);
    }

    // ── 54: Large number of entries all retrievable ───────────────────────────

    #[test]
    fn test_many_entries_all_retrievable() {
        let mut c = ContentAddressableCache::new(CacheConfig {
            max_entries: 512,
            max_bytes: 64 * 1024 * 1024,
            ..Default::default()
        });
        let mut rng = Xorshift64::new(7777);
        let mut kv: Vec<(String, Vec<u8>)> = Vec::new();
        for _ in 0..256 {
            let data = rng.bytes(64);
            if let Ok(cid) = c.insert(data.clone(), None, vec![]) {
                kv.push((cid, data));
            }
        }
        for (cid, data) in &kv {
            if c.contains(cid) {
                assert_eq!(c.get(cid).unwrap(), data.as_slice());
            }
        }
    }
}
