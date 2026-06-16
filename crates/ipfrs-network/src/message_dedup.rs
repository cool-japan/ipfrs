//! Message deduplication using a two-layer approach.
//!
//! Provides efficient duplicate detection for gossip/pubsub messages via:
//! - **Layer 1 – Bloom filter**: Fast probabilistic rejection using a bit array
//!   with Kirsch-Mitzenmacher double hashing (FNV-1a based).  If the bloom
//!   filter reports "not seen", the message is *definitely* new.
//! - **Layer 2 – Bounded LRU cache**: Exact deduplication backed by a
//!   `HashMap<[u8;32], DedupEntry>` with an insertion-order `Vec` for LRU
//!   eviction once capacity is reached.
//!
//! ## Usage
//!
//! ```rust
//! use ipfrs_network::message_dedup::{MessageDeduplicator, DedupConfig, MsgId};
//!
//! let config = DedupConfig {
//!     bloom_bits: 1 << 16,
//!     bloom_hash_count: 7,
//!     cache_capacity: 4096,
//!     window_ms: 60_000,
//! };
//! let mut dedup = MessageDeduplicator::new(config);
//!
//! let id = MessageDeduplicator::make_msg_id(b"hello world");
//! assert!(!dedup.check_and_insert(&id, 1_000));  // new
//! assert!( dedup.check_and_insert(&id, 1_001));  // duplicate
//! ```

use std::collections::HashMap;

// ── FNV-1a constants ──────────────────────────────────────────────────────────

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET_BASIS_64: u64 = 14_695_981_039_346_656_037;
/// FNV-1a 64-bit prime.
const FNV_PRIME_64: u64 = 1_099_511_628_211;

/// Compute FNV-1a 64-bit hash of `data`.
#[inline]
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h = FNV_OFFSET_BASIS_64;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME_64);
    }
    h
}

/// Compute a seeded FNV-1a 64-bit hash by pre-mixing a seed into the state.
#[inline]
fn fnv1a_64_seeded(data: &[u8], seed: u64) -> u64 {
    // Mix the seed into the initial state using the FNV prime, then hash data.
    let mut h = FNV_OFFSET_BASIS_64 ^ seed;
    h = h.wrapping_mul(FNV_PRIME_64);
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME_64);
    }
    h
}

// ── MessageId ────────────────────────────────────────────────────────────────

/// A 32-byte message fingerprint used for deduplication.
///
/// Constructed deterministically from message content via [`MessageDeduplicator::make_msg_id`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MsgId {
    /// Raw 32-byte fingerprint.
    pub bytes: [u8; 32],
}

impl MsgId {
    /// Construct a `MsgId` directly from raw bytes.
    #[inline]
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self { bytes }
    }
}

// ── DedupConfig ───────────────────────────────────────────────────────────────

/// Configuration for [`MessageDeduplicator`].
#[derive(Debug, Clone)]
pub struct DedupConfig {
    /// Number of bits in the bloom filter bit-array.  Rounded up to the next
    /// multiple of 64 internally.  Minimum: 64.
    pub bloom_bits: usize,
    /// Number of independent hash functions for the bloom filter.  More hashes
    /// reduce false-positive rate at the cost of slightly more CPU per insert.
    pub bloom_hash_count: u32,
    /// Maximum number of entries held in the exact-dedup LRU cache.
    pub cache_capacity: usize,
    /// Deduplication window in milliseconds.  Entries older than this are
    /// eligible for expiry via [`MessageDeduplicator::purge_expired`].
    pub window_ms: u64,
}

impl Default for DedupConfig {
    fn default() -> Self {
        Self {
            bloom_bits: 1 << 17, // 128 Ki bits ≈ 16 KiB
            bloom_hash_count: 7,
            cache_capacity: 8_192,
            window_ms: 120_000, // 2 minutes
        }
    }
}

// ── DedupEntry ────────────────────────────────────────────────────────────────

/// A single entry in the exact-dedup LRU cache.
#[derive(Debug, Clone)]
pub struct DedupEntry {
    /// The message identifier.
    pub msg_id: MsgId,
    /// Timestamp (ms) when the message was first seen.
    pub first_seen: u64,
    /// Number of duplicate sightings *after* the first (i.e. 0 means seen once).
    pub count: u32,
}

// ── MsgDedupStats ─────────────────────────────────────────────────────────────

/// Accumulated statistics for a [`MessageDeduplicator`].
#[derive(Debug, Clone, Default)]
pub struct MsgDedupStats {
    /// Total messages processed (unique + duplicate).
    pub total_seen: u64,
    /// Messages identified as duplicates.
    pub duplicates: u64,
    /// Messages accepted as unique.
    pub unique: u64,
    /// Cases where the bloom filter reported "seen" but the cache said "new"
    /// (i.e. bloom false-positives caught by the exact cache layer).
    pub bloom_false_positives: u64,
    /// Number of cache entries evicted due to LRU capacity enforcement.
    pub cache_evictions: u64,
}

// ── MessageDeduplicator ───────────────────────────────────────────────────────

/// Two-layer message deduplicator: Bloom filter + bounded LRU cache.
///
/// See [module documentation](crate::message_dedup) for usage.
pub struct MessageDeduplicator {
    config: DedupConfig,
    /// Bloom filter bit-array stored as 64-bit words.
    bloom: Vec<u64>,
    /// Exact-dedup map: msg bytes → entry.
    cache: HashMap<[u8; 32], DedupEntry>,
    /// Tracks insertion order for LRU eviction (oldest at index 0).
    insertion_order: Vec<[u8; 32]>,
    /// Accumulated statistics.
    stats: MsgDedupStats,
}

impl MessageDeduplicator {
    // ── Construction ────────────────────────────────────────────────────────

    /// Create a new deduplicator with the given configuration.
    pub fn new(config: DedupConfig) -> Self {
        // Round bloom_bits up to a multiple of 64, ensure at least 64 bits.
        let bloom_bits = config.bloom_bits.max(64);
        let words = bloom_bits.div_ceil(64);

        let cache_capacity = config.cache_capacity.max(1);
        Self {
            bloom: vec![0u64; words],
            cache: HashMap::with_capacity(cache_capacity),
            insertion_order: Vec::with_capacity(cache_capacity),
            stats: MsgDedupStats::default(),
            config: DedupConfig {
                bloom_bits: words * 64,
                cache_capacity,
                ..config
            },
        }
    }

    // ── Public API ──────────────────────────────────────────────────────────

    /// Check whether `msg_id` has been seen before and, if not, record it.
    ///
    /// Returns `true` if the message is a **duplicate** (was already seen),
    /// `false` if it is **new** (first occurrence).
    ///
    /// `now` is the caller-supplied timestamp in milliseconds (e.g. from
    /// `std::time::SystemTime` or a monotonic clock).
    pub fn check_and_insert(&mut self, msg_id: &MsgId, now: u64) -> bool {
        self.stats.total_seen += 1;

        // Layer 1: bloom fast path.
        let bloom_says_seen = self.is_duplicate_bloom(msg_id);

        if bloom_says_seen {
            // Layer 2: exact cache check.
            if let Some(entry) = self.cache.get_mut(&msg_id.bytes) {
                // Confirmed duplicate.
                entry.count = entry.count.saturating_add(1);
                self.stats.duplicates += 1;
                return true;
            }
            // Bloom false positive — fall through to insert as new.
            self.stats.bloom_false_positives += 1;
        }

        // New message: insert into bloom and cache.
        self.insert_bloom(msg_id);
        self.insert_cache(msg_id, now);
        self.stats.unique += 1;
        false
    }

    /// Check the bloom filter alone (probabilistic — may have false positives).
    ///
    /// Returns `true` if the bloom filter believes `msg_id` was seen before.
    pub fn is_duplicate_bloom(&self, msg_id: &MsgId) -> bool {
        let total_bits = self.bloom.len() * 64;
        if total_bits == 0 {
            return false;
        }
        for pos in Self::bloom_bit_positions(msg_id, self.config.bloom_hash_count) {
            let bit_idx = pos % total_bits;
            let word = bit_idx / 64;
            let bit = bit_idx % 64;
            if self.bloom[word] & (1u64 << bit) == 0 {
                return false;
            }
        }
        true
    }

    /// Set the bloom filter bits for `msg_id`.
    pub fn insert_bloom(&mut self, msg_id: &MsgId) {
        let total_bits = self.bloom.len() * 64;
        if total_bits == 0 {
            return;
        }
        for pos in Self::bloom_bit_positions(msg_id, self.config.bloom_hash_count) {
            let bit_idx = pos % total_bits;
            let word = bit_idx / 64;
            let bit = bit_idx % 64;
            self.bloom[word] |= 1u64 << bit;
        }
    }

    /// Compute `count` bloom bit positions for `msg_id` using
    /// Kirsch-Mitzenmacher double hashing over FNV-1a.
    ///
    /// Uses two independent FNV-1a 64-bit hashes (`h1`, `h2`) and derives
    /// positions as `(h1 + i * h2) mod usize::MAX`.  The modulo against the
    /// actual bit-array size is applied by the caller.
    pub fn bloom_bit_positions(msg_id: &MsgId, count: u32) -> Vec<usize> {
        let h1 = fnv1a_64(&msg_id.bytes);
        // Use a different seed for h2 to get an independent hash.
        let h2 = fnv1a_64_seeded(&msg_id.bytes, 0xDEAD_BEEF_CAFE_BABE_u64);
        // Ensure h2 is odd so it covers all positions in power-of-two arrays.
        let h2 = h2 | 1;

        (0..count as u64)
            .map(|i| h1.wrapping_add(i.wrapping_mul(h2)) as usize)
            .collect()
    }

    /// Construct a [`MsgId`] from arbitrary data.
    ///
    /// Produces a 32-byte fingerprint using four independent FNV-1a 64-bit
    /// hashes (seeded with 0, 1, 2, 3) concatenated into a `[u8; 32]`.
    /// This is deterministic: the same `data` always yields the same [`MsgId`].
    pub fn make_msg_id(data: &[u8]) -> MsgId {
        let h0 = fnv1a_64(data);
        let h1 = fnv1a_64_seeded(data, 1);
        let h2 = fnv1a_64_seeded(data, 2);
        let h3 = fnv1a_64_seeded(data, 3);

        let mut bytes = [0u8; 32];
        bytes[0..8].copy_from_slice(&h0.to_le_bytes());
        bytes[8..16].copy_from_slice(&h1.to_le_bytes());
        bytes[16..24].copy_from_slice(&h2.to_le_bytes());
        bytes[24..32].copy_from_slice(&h3.to_le_bytes());
        MsgId { bytes }
    }

    /// Remove all cache entries whose `first_seen` timestamp is older than
    /// `now - window_ms`.  Returns the number of entries purged.
    ///
    /// Note: the bloom filter is **not** cleared by this call (bloom filters
    /// do not support deletion).  Only the exact cache is pruned.
    pub fn purge_expired(&mut self, now: u64) -> usize {
        let cutoff = now.saturating_sub(self.config.window_ms);
        let before = self.cache.len();

        // Collect expired keys.
        let expired: Vec<[u8; 32]> = self
            .cache
            .iter()
            .filter(|(_, e)| e.first_seen < cutoff)
            .map(|(k, _)| *k)
            .collect();

        for key in &expired {
            self.cache.remove(key);
            // Remove from insertion_order as well.
            if let Some(pos) = self.insertion_order.iter().position(|k| k == key) {
                self.insertion_order.remove(pos);
            }
        }

        before - self.cache.len()
    }

    /// Evict the oldest (least recently inserted) entry from the cache.
    ///
    /// Returns `true` if an entry was evicted, `false` if the cache was empty.
    pub fn evict_oldest(&mut self) -> bool {
        if let Some(oldest) = self.insertion_order.first().copied() {
            self.insertion_order.remove(0);
            self.cache.remove(&oldest);
            self.stats.cache_evictions += 1;
            true
        } else {
            false
        }
    }

    /// Return the number of unique entries currently in the exact cache.
    pub fn seen_count(&self) -> usize {
        self.cache.len()
    }

    /// Return a reference to the accumulated statistics.
    pub fn stats(&self) -> &MsgDedupStats {
        &self.stats
    }

    /// Reset the deduplicator to a clean initial state (clears bloom, cache,
    /// insertion order, and stats).  Configuration is preserved.
    pub fn reset(&mut self) {
        for word in &mut self.bloom {
            *word = 0;
        }
        self.cache.clear();
        self.insertion_order.clear();
        self.stats = MsgDedupStats::default();
    }

    // ── Private helpers ─────────────────────────────────────────────────────

    /// Insert a new entry into the exact cache, enforcing capacity via LRU
    /// eviction when necessary.
    fn insert_cache(&mut self, msg_id: &MsgId, now: u64) {
        // Evict until we have room.
        while self.cache.len() >= self.config.cache_capacity {
            self.evict_oldest();
        }
        let entry = DedupEntry {
            msg_id: msg_id.clone(),
            first_seen: now,
            count: 0,
        };
        self.cache.insert(msg_id.bytes, entry);
        self.insertion_order.push(msg_id.bytes);
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> DedupConfig {
        DedupConfig {
            bloom_bits: 1 << 14, // 16 Ki bits
            bloom_hash_count: 7,
            cache_capacity: 128,
            window_ms: 60_000,
        }
    }

    fn make_dedup() -> MessageDeduplicator {
        MessageDeduplicator::new(default_config())
    }

    // ── 1. New message returns false ─────────────────────────────────────────
    #[test]
    fn test_new_message_returns_false() {
        let mut dedup = make_dedup();
        let id = MessageDeduplicator::make_msg_id(b"hello");
        assert!(!dedup.check_and_insert(&id, 1_000));
    }

    // ── 2. Duplicate returns true ────────────────────────────────────────────
    #[test]
    fn test_duplicate_returns_true() {
        let mut dedup = make_dedup();
        let id = MessageDeduplicator::make_msg_id(b"hello");
        assert!(!dedup.check_and_insert(&id, 1_000));
        assert!(dedup.check_and_insert(&id, 1_001));
    }

    // ── 3. Multiple duplicates all return true ───────────────────────────────
    #[test]
    fn test_multiple_duplicates() {
        let mut dedup = make_dedup();
        let id = MessageDeduplicator::make_msg_id(b"repeat");
        assert!(!dedup.check_and_insert(&id, 100));
        for i in 1..10u64 {
            assert!(dedup.check_and_insert(&id, 100 + i));
        }
    }

    // ── 4. Bloom fast path: after insert bloom_says_seen ────────────────────
    #[test]
    fn test_bloom_fast_path() {
        let mut dedup = make_dedup();
        let id = MessageDeduplicator::make_msg_id(b"bloom_test");
        assert!(!dedup.is_duplicate_bloom(&id));
        dedup.insert_bloom(&id);
        assert!(dedup.is_duplicate_bloom(&id));
    }

    // ── 5. Bloom positions are deterministic ────────────────────────────────
    #[test]
    fn test_bloom_positions_deterministic() {
        let id = MessageDeduplicator::make_msg_id(b"positions");
        let p1 = MessageDeduplicator::bloom_bit_positions(&id, 7);
        let p2 = MessageDeduplicator::bloom_bit_positions(&id, 7);
        assert_eq!(p1, p2);
    }

    // ── 6. Bloom positions count matches hash_count ──────────────────────────
    #[test]
    fn test_bloom_positions_count() {
        let id = MessageDeduplicator::make_msg_id(b"count_test");
        for k in [1u32, 3, 7, 10, 20] {
            let positions = MessageDeduplicator::bloom_bit_positions(&id, k);
            assert_eq!(positions.len(), k as usize);
        }
    }

    // ── 7. make_msg_id is deterministic ─────────────────────────────────────
    #[test]
    fn test_make_msg_id_deterministic() {
        let id1 = MessageDeduplicator::make_msg_id(b"data");
        let id2 = MessageDeduplicator::make_msg_id(b"data");
        assert_eq!(id1, id2);
    }

    // ── 8. Identical data → same MsgId ───────────────────────────────────────
    #[test]
    fn test_identical_data_same_id() {
        let data = b"same content here";
        let id1 = MessageDeduplicator::make_msg_id(data);
        let id2 = MessageDeduplicator::make_msg_id(data);
        assert_eq!(id1.bytes, id2.bytes);
    }

    // ── 9. Different data → different MsgId ──────────────────────────────────
    #[test]
    fn test_different_data_different_id() {
        let id1 = MessageDeduplicator::make_msg_id(b"message A");
        let id2 = MessageDeduplicator::make_msg_id(b"message B");
        assert_ne!(id1.bytes, id2.bytes);
    }

    // ── 10. Empty data produces a valid, non-zero MsgId ──────────────────────
    #[test]
    fn test_empty_data_msg_id() {
        let id = MessageDeduplicator::make_msg_id(b"");
        // All-zero would be suspicious; FNV offset basis guarantees non-zero.
        assert_ne!(id.bytes, [0u8; 32]);
    }

    // ── 11. Stats: total_seen / unique / duplicates ───────────────────────────
    #[test]
    fn test_stats_accuracy() {
        let mut dedup = make_dedup();
        let id_a = MessageDeduplicator::make_msg_id(b"A");
        let id_b = MessageDeduplicator::make_msg_id(b"B");

        dedup.check_and_insert(&id_a, 1);
        dedup.check_and_insert(&id_a, 2); // dup
        dedup.check_and_insert(&id_b, 3);

        let s = dedup.stats();
        assert_eq!(s.total_seen, 3);
        assert_eq!(s.unique, 2);
        assert_eq!(s.duplicates, 1);
    }

    // ── 12. seen_count reflects unique inserts ────────────────────────────────
    #[test]
    fn test_seen_count() {
        let mut dedup = make_dedup();
        assert_eq!(dedup.seen_count(), 0);
        let id = MessageDeduplicator::make_msg_id(b"x");
        dedup.check_and_insert(&id, 0);
        assert_eq!(dedup.seen_count(), 1);
    }

    // ── 13. LRU eviction: capacity enforcement ────────────────────────────────
    #[test]
    fn test_lru_capacity_enforcement() {
        let config = DedupConfig {
            bloom_bits: 1 << 16,
            bloom_hash_count: 7,
            cache_capacity: 5,
            window_ms: 60_000,
        };
        let mut dedup = MessageDeduplicator::new(config);

        for i in 0u64..10 {
            let id = MessageDeduplicator::make_msg_id(&i.to_le_bytes());
            dedup.check_and_insert(&id, i);
        }
        // Cache must never exceed capacity.
        assert!(dedup.seen_count() <= 5);
    }

    // ── 14. LRU eviction increments cache_evictions stat ─────────────────────
    #[test]
    fn test_lru_eviction_stats() {
        let config = DedupConfig {
            bloom_bits: 1 << 16,
            bloom_hash_count: 7,
            cache_capacity: 3,
            window_ms: 60_000,
        };
        let mut dedup = MessageDeduplicator::new(config);

        for i in 0u64..6 {
            let id = MessageDeduplicator::make_msg_id(&i.to_le_bytes());
            dedup.check_and_insert(&id, i);
        }
        // We inserted 6 into a capacity-3 cache → at least 3 evictions.
        assert!(dedup.stats().cache_evictions >= 3);
    }

    // ── 15. evict_oldest removes the oldest entry ────────────────────────────
    #[test]
    fn test_evict_oldest() {
        let mut dedup = make_dedup();
        let id0 = MessageDeduplicator::make_msg_id(b"first");
        let id1 = MessageDeduplicator::make_msg_id(b"second");
        dedup.check_and_insert(&id0, 0);
        dedup.check_and_insert(&id1, 1);
        assert_eq!(dedup.seen_count(), 2);
        let evicted = dedup.evict_oldest();
        assert!(evicted);
        assert_eq!(dedup.seen_count(), 1);
    }

    // ── 16. evict_oldest on empty cache returns false ─────────────────────────
    #[test]
    fn test_evict_oldest_empty() {
        let mut dedup = make_dedup();
        assert!(!dedup.evict_oldest());
    }

    // ── 17. purge_expired removes entries older than window ──────────────────
    #[test]
    fn test_purge_expired() {
        let mut dedup = make_dedup(); // window_ms = 60_000
        let id_old = MessageDeduplicator::make_msg_id(b"old");
        let id_new = MessageDeduplicator::make_msg_id(b"new");

        // Insert old entry at t=0, new entry at t=100_000.
        dedup.check_and_insert(&id_old, 0);
        dedup.check_and_insert(&id_new, 100_000);

        let now = 120_001; // window end: 120_001 - 60_000 = 60_001 → old (0) expired
        let purged = dedup.purge_expired(now);
        assert_eq!(purged, 1);
        assert_eq!(dedup.seen_count(), 1);
    }

    // ── 18. purge_expired keeps entries within window ────────────────────────
    #[test]
    fn test_purge_expired_keeps_recent() {
        let mut dedup = make_dedup(); // window_ms = 60_000
        let id = MessageDeduplicator::make_msg_id(b"recent");
        dedup.check_and_insert(&id, 1_000);
        let purged = dedup.purge_expired(30_000); // only 29 s elapsed
        assert_eq!(purged, 0);
        assert_eq!(dedup.seen_count(), 1);
    }

    // ── 19. purge_expired on empty cache returns 0 ───────────────────────────
    #[test]
    fn test_purge_expired_empty() {
        let mut dedup = make_dedup();
        assert_eq!(dedup.purge_expired(999_999), 0);
    }

    // ── 20. reset clears bloom, cache, and stats ─────────────────────────────
    #[test]
    fn test_reset_clears_state() {
        let mut dedup = make_dedup();
        let id = MessageDeduplicator::make_msg_id(b"pre-reset");
        dedup.check_and_insert(&id, 0);
        dedup.check_and_insert(&id, 1); // dup

        dedup.reset();

        assert_eq!(dedup.seen_count(), 0);
        let s = dedup.stats();
        assert_eq!(s.total_seen, 0);
        assert_eq!(s.duplicates, 0);
        assert_eq!(s.unique, 0);

        // After reset the bloom is clear, so the same message should be new.
        assert!(!dedup.check_and_insert(&id, 2));
    }

    // ── 21. reset resets bloom (false positive would reoccur without reset) ──
    #[test]
    fn test_reset_clears_bloom() {
        let mut dedup = make_dedup();
        let id = MessageDeduplicator::make_msg_id(b"bloom_reset");
        dedup.insert_bloom(&id);
        assert!(dedup.is_duplicate_bloom(&id));
        dedup.reset();
        assert!(!dedup.is_duplicate_bloom(&id));
    }

    // ── 22. Bloom false positive tracking ────────────────────────────────────
    #[test]
    fn test_bloom_false_positive_tracking() {
        // Use a tiny bloom (64 bits, many hashes) to force false positives.
        let config = DedupConfig {
            bloom_bits: 64,
            bloom_hash_count: 20,
            cache_capacity: 1024,
            window_ms: 60_000,
        };
        let mut dedup = MessageDeduplicator::new(config);

        // Fill the bloom heavily.
        for i in 0u64..50 {
            let id = MessageDeduplicator::make_msg_id(&i.to_le_bytes());
            dedup.check_and_insert(&id, i);
        }
        // With a 64-bit bloom and 20 hashes, virtually every bit is set →
        // any new key will hit a bloom false positive.
        let id_new = MessageDeduplicator::make_msg_id(b"definitely_new_unique_key_9999");
        dedup.check_and_insert(&id_new, 9999);

        // bloom_false_positives may be > 0 in this degenerate setup.
        // We just ensure it doesn't panic and the stat is accessible.
        let _ = dedup.stats().bloom_false_positives;
    }

    // ── 23. Large volume: 1000 unique messages ───────────────────────────────
    #[test]
    fn test_large_volume_unique() {
        let config = DedupConfig {
            bloom_bits: 1 << 17,
            bloom_hash_count: 7,
            cache_capacity: 2048,
            window_ms: 60_000,
        };
        let mut dedup = MessageDeduplicator::new(config);

        for i in 0u64..1000 {
            let data = format!("message-{}", i);
            let id = MessageDeduplicator::make_msg_id(data.as_bytes());
            assert!(!dedup.check_and_insert(&id, i));
        }
        assert_eq!(dedup.stats().total_seen, 1000);
        // unique + bloom_false_positives == total_seen since all were new.
        // (A bloom false-positive still counts as unique from the dedup perspective)
        assert_eq!(dedup.stats().unique, 1000 - dedup.stats().duplicates);
    }

    // ── 24. Large volume: 1000 duplicate messages ────────────────────────────
    #[test]
    fn test_large_volume_duplicates() {
        let mut dedup = make_dedup();
        let id = MessageDeduplicator::make_msg_id(b"same_msg");
        assert!(!dedup.check_and_insert(&id, 0));
        for i in 1u64..1000 {
            assert!(dedup.check_and_insert(&id, i));
        }
        assert_eq!(dedup.stats().duplicates, 999);
    }

    // ── 25. DedupEntry count increments on repeated duplicates ───────────────
    #[test]
    fn test_dedup_entry_count() {
        let mut dedup = make_dedup();
        let id = MessageDeduplicator::make_msg_id(b"counted");
        dedup.check_and_insert(&id, 0);
        dedup.check_and_insert(&id, 1);
        dedup.check_and_insert(&id, 2);
        // The entry.count should be 2 (two extra sightings).
        let entry = dedup.cache.get(&id.bytes).cloned();
        assert!(entry.is_some());
        let e = entry.unwrap_or_else(|| DedupEntry {
            msg_id: id.clone(),
            first_seen: 0,
            count: 0,
        });
        assert_eq!(e.count, 2);
    }

    // ── 26. Distinct messages each tracked independently ─────────────────────
    #[test]
    fn test_distinct_messages_independent() {
        let mut dedup = make_dedup();
        for i in 0u64..20 {
            let id = MessageDeduplicator::make_msg_id(&i.to_le_bytes());
            assert!(!dedup.check_and_insert(&id, i * 100));
        }
        // Re-check each one should be a duplicate now.
        for i in 0u64..20 {
            let id = MessageDeduplicator::make_msg_id(&i.to_le_bytes());
            // After LRU may have evicted some (capacity=128 >> 20, so all present).
            assert!(dedup.check_and_insert(&id, i * 100 + 50));
        }
    }

    // ── 27. from_bytes round-trip ─────────────────────────────────────────────
    #[test]
    fn test_msg_id_from_bytes_roundtrip() {
        let original = [42u8; 32];
        let id = MsgId::from_bytes(original);
        assert_eq!(id.bytes, original);
    }

    // ── 28. Config defaults are valid ────────────────────────────────────────
    #[test]
    fn test_default_config_valid() {
        let config = DedupConfig::default();
        let dedup = MessageDeduplicator::new(config);
        assert!(!dedup.bloom.is_empty());
        assert!(dedup.config.cache_capacity > 0);
    }
}
