//! Content deduplication index for tracking and merging identical content blocks.
//!
//! This module provides a hash-based content deduplication index that:
//! - Tracks identical content blocks via 32-byte approximate hashes
//! - Manages reference counts per unique content block
//! - Reclaims storage by detecting and merging duplicates
//! - Supports eviction when the index is full
//!
//! # Hash Implementation
//!
//! The 32-byte `ContentHash` is composed of four 8-byte segments:
//! - Bytes  0– 7: FNV-1a 64-bit hash of the data
//! - Bytes  8–15: DJB2 64-bit xorshifted hash of the data
//! - Bytes 16–23: data length as little-endian u64
//! - Bytes 24–31: FNV-1a 64-bit hash of the reversed data
//!
//! This is a fast approximate hash, not a cryptographic hash.

use std::collections::HashMap;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can arise when operating on a [`ContentDeduplicationIndex`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum DedupIndexError {
    /// The given key already exists in the index.
    #[error("key already exists: {0}")]
    KeyAlreadyExists(String),

    /// No entry was found for the given key or hash.
    #[error("entry not found: {0}")]
    EntryNotFound(String),

    /// The data size is outside the configured range.
    #[error("invalid size {size}: must be in [{min}, {max}]")]
    InvalidSize {
        /// Actual size of the data.
        size: usize,
        /// Configured minimum block size.
        min: usize,
        /// Configured maximum block size.
        max: usize,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// ContentHash
// ─────────────────────────────────────────────────────────────────────────────

/// A 32-byte approximate hash used to identify content blocks.
///
/// Not cryptographic — uses FNV-1a and DJB2 mixing for speed.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ContentHash(pub [u8; 32]);

impl std::fmt::Debug for ContentHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ContentHash(")?;
        for byte in &self.0 {
            write!(f, "{:02x}", byte)?;
        }
        write!(f, ")")
    }
}

impl std::fmt::Display for ContentHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in &self.0 {
            write!(f, "{:02x}", byte)?;
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Hash primitives
// ─────────────────────────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash.
#[inline]
fn fnv1a_64(data: &[u8]) -> u64 {
    const OFFSET: u64 = 14695981039346656037;
    const PRIME: u64 = 1099511628211;
    data.iter()
        .fold(OFFSET, |h, &b| (h ^ (b as u64)).wrapping_mul(PRIME))
}

/// DJB2 64-bit hash (xorshift variant).
#[inline]
fn djb2_64(data: &[u8]) -> u64 {
    data.iter().fold(5381u64, |h, &b| {
        h.wrapping_shl(5).wrapping_add(h).wrapping_add(b as u64) ^ (h >> 33)
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// DedupEntry
// ─────────────────────────────────────────────────────────────────────────────

/// A single entry in the deduplication index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DedupEntry {
    /// Hash of the content.
    pub hash: ContentHash,
    /// The canonical storage key for this content.
    pub canonical_key: String,
    /// Number of keys that currently reference this content.
    pub ref_count: u32,
    /// Size of the content in bytes.
    pub byte_size: u64,
    /// Unix timestamp (seconds) when this entry was first seen.
    pub first_seen: u64,
    /// Unix timestamp (seconds) of the most recent access.
    pub last_accessed: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// DedupResult
// ─────────────────────────────────────────────────────────────────────────────

/// The outcome of an [`ContentDeduplicationIndex::insert`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentDedupResult {
    /// Content is new — no prior entry existed for this hash.
    New {
        /// The key that was inserted.
        key: String,
        /// Hash computed for the inserted content.
        hash: ContentHash,
    },
    /// Content already existed — this insertion is a duplicate.
    Duplicate {
        /// The canonical key of the pre-existing entry.
        original_key: String,
        /// Bytes saved by not storing this content again.
        saved_bytes: u64,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// DedupConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`ContentDeduplicationIndex`].
#[derive(Debug, Clone)]
pub struct ContentDedupConfig {
    /// Maximum number of unique content entries held in the index.
    pub max_entries: usize,
    /// Minimum content size (in bytes) eligible for deduplication.
    /// Blocks smaller than this are passed through as `New` without indexing.
    pub min_block_size: usize,
    /// Maximum content size (in bytes) eligible for deduplication.
    /// Blocks larger than this are passed through as `New` without indexing.
    pub max_block_size: usize,
    /// When `true`, each duplicate insertion increments `ref_count`.
    pub enable_ref_counting: bool,
}

impl Default for ContentDedupConfig {
    fn default() -> Self {
        Self {
            max_entries: 100_000,
            min_block_size: 64,
            max_block_size: 64 * 1024 * 1024, // 64 MiB
            enable_ref_counting: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DedupStats
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregated statistics for a [`ContentDeduplicationIndex`].
#[derive(Debug, Clone, PartialEq)]
pub struct ContentDedupStats {
    /// Number of unique content entries currently in the index.
    pub total_entries: usize,
    /// Total number of key→hash mappings (includes duplicates).
    pub total_keys: usize,
    /// Total bytes saved by deduplication across all duplicate insertions.
    pub total_saved_bytes: u64,
    /// Total number of insert calls (including out-of-range pass-throughs).
    pub total_insertions: u64,
    /// Number of insert calls that resolved to a duplicate.
    pub total_duplicates: u64,
    /// Ratio of duplicates to total insertions (`total_duplicates / max(1, total_insertions)`).
    pub dedup_ratio: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// ContentDeduplicationIndex
// ─────────────────────────────────────────────────────────────────────────────

/// Hash-based content deduplication index.
///
/// Tracks identical content blocks, reference counts, and can reclaim storage
/// by merging duplicates.
///
/// # Example
///
/// ```rust
/// use ipfrs_storage::content_dedup_index::{
///     ContentDeduplicationIndex, ContentDedupConfig, ContentDedupResult,
/// };
///
/// let config = ContentDedupConfig::default();
/// let mut idx = ContentDeduplicationIndex::new(config);
///
/// let data = b"hello, world!";
/// let result = idx.insert("key1".into(), data, 1000).unwrap();
/// assert!(matches!(result, ContentDedupResult::New { .. }));
///
/// // Insert the same content under a different key — it's a duplicate
/// let result2 = idx.insert("key2".into(), data, 1001).unwrap();
/// assert!(matches!(result2, ContentDedupResult::Duplicate { .. }));
/// ```
#[derive(Debug)]
pub struct ContentDeduplicationIndex {
    /// Runtime configuration.
    pub config: ContentDedupConfig,
    /// Map from content hash to dedup entry.
    entries: HashMap<ContentHash, DedupEntry>,
    /// Map from storage key to content hash.
    key_to_hash: HashMap<String, ContentHash>,
    /// Cumulative bytes saved by deduplication.
    total_saved_bytes: u64,
    /// Total number of insert calls.
    total_insertions: u64,
    /// Number of insert calls that were duplicates.
    total_duplicates: u64,
}

impl ContentDeduplicationIndex {
    /// Create a new index with the supplied configuration.
    pub fn new(config: ContentDedupConfig) -> Self {
        Self {
            entries: HashMap::new(),
            key_to_hash: HashMap::new(),
            total_saved_bytes: 0,
            total_insertions: 0,
            total_duplicates: 0,
            config,
        }
    }

    /// Compute the 32-byte approximate `ContentHash` for `data`.
    ///
    /// Layout:
    /// - Bytes  0– 7: FNV-1a 64-bit
    /// - Bytes  8–15: DJB2 64-bit xorshifted
    /// - Bytes 16–23: `data.len()` as little-endian u64
    /// - Bytes 24–31: FNV-1a 64-bit of reversed data
    pub fn compute_hash(data: &[u8]) -> ContentHash {
        let fnv_forward = fnv1a_64(data);
        let djb2 = djb2_64(data);
        let length = data.len() as u64;

        // Compute FNV-1a over data traversed in reverse without allocating.
        let fnv_reverse = data.iter().rev().copied().collect::<Vec<u8>>();
        let fnv_rev = fnv1a_64(&fnv_reverse);

        let mut raw = [0u8; 32];
        raw[0..8].copy_from_slice(&fnv_forward.to_le_bytes());
        raw[8..16].copy_from_slice(&djb2.to_le_bytes());
        raw[16..24].copy_from_slice(&length.to_le_bytes());
        raw[24..32].copy_from_slice(&fnv_rev.to_le_bytes());
        ContentHash(raw)
    }

    /// Insert `data` under `key` into the deduplication index.
    ///
    /// Returns:
    /// - `Ok(ContentDedupResult::New)` when the content has not been seen before.
    /// - `Ok(ContentDedupResult::Duplicate)` when identical content already exists.
    ///
    /// Blocks whose size falls outside `[min_block_size, max_block_size]` are
    /// returned as `New` without modifying the index (pass-through behaviour).
    pub fn insert(
        &mut self,
        key: String,
        data: &[u8],
        now: u64,
    ) -> Result<ContentDedupResult, DedupIndexError> {
        self.total_insertions += 1;

        let hash = Self::compute_hash(data);

        // Pass-through for out-of-range blocks — don't index them.
        if data.len() < self.config.min_block_size || data.len() > self.config.max_block_size {
            return Ok(ContentDedupResult::New { key, hash });
        }

        // Check whether this hash is already known.
        if let Some(entry) = self.entries.get_mut(&hash) {
            // Duplicate found.
            if self.config.enable_ref_counting {
                entry.ref_count += 1;
            }
            entry.last_accessed = now;
            let original_key = entry.canonical_key.clone();
            let saved_bytes = data.len() as u64;

            self.key_to_hash.insert(key, hash);
            self.total_saved_bytes += saved_bytes;
            self.total_duplicates += 1;

            return Ok(ContentDedupResult::Duplicate {
                original_key,
                saved_bytes,
            });
        }

        // New content — evict if at capacity.
        if self.entries.len() >= self.config.max_entries {
            self.evict_one();
        }

        let entry = DedupEntry {
            hash: hash.clone(),
            canonical_key: key.clone(),
            ref_count: 1,
            byte_size: data.len() as u64,
            first_seen: now,
            last_accessed: now,
        };
        self.entries.insert(hash.clone(), entry);
        self.key_to_hash.insert(key.clone(), hash.clone());

        Ok(ContentDedupResult::New { key, hash })
    }

    /// Remove the key from the index.
    ///
    /// Decrements the reference count of the associated entry; if the count
    /// reaches zero and no other key points to the same hash, the entry is
    /// removed entirely.
    ///
    /// Returns `true` if `key` was present, `false` otherwise.
    pub fn remove(&mut self, key: &str) -> bool {
        let hash = match self.key_to_hash.remove(key) {
            Some(h) => h,
            None => return false,
        };

        // Count remaining keys that still point to this hash.
        let remaining_refs = self.key_to_hash.values().filter(|h| **h == hash).count();

        if let Some(entry) = self.entries.get_mut(&hash) {
            if self.config.enable_ref_counting && entry.ref_count > 0 {
                entry.ref_count -= 1;
            }
            // Remove the entry only when no keys (including the one we just
            // removed) still reference it.
            if remaining_refs == 0 {
                self.entries.remove(&hash);
            }
        }

        true
    }

    /// Look up a dedup entry by its storage key.
    pub fn lookup_by_key(&self, key: &str) -> Option<&DedupEntry> {
        let hash = self.key_to_hash.get(key)?;
        self.entries.get(hash)
    }

    /// Look up a dedup entry directly by its content hash.
    pub fn lookup_by_hash(&self, hash: &ContentHash) -> Option<&DedupEntry> {
        self.entries.get(hash)
    }

    /// Return `true` if `data` has been seen before (i.e., a matching hash
    /// exists in the index and the block size is within range).
    pub fn is_duplicate(&self, data: &[u8]) -> bool {
        if data.len() < self.config.min_block_size || data.len() > self.config.max_block_size {
            return false;
        }
        let hash = Self::compute_hash(data);
        self.entries.contains_key(&hash)
    }

    /// Scan `key_to_hash` and count keys whose hash is already represented by
    /// a canonical key that differs from themselves.  Returns the total number
    /// of duplicate key→hash mappings merged (conceptually deduplicated).
    ///
    /// This method does not remove any data; it reports how many duplicate
    /// entries currently exist and is idempotent.
    pub fn merge_duplicates(&mut self) -> u64 {
        let mut count = 0u64;
        for (key, hash) in &self.key_to_hash {
            if let Some(entry) = self.entries.get(hash) {
                if entry.canonical_key != *key {
                    count += 1;
                }
            }
        }
        count
    }

    /// Return all `(duplicate_key, canonical_key)` pairs where the duplicate
    /// key differs from the canonical key.
    pub fn deduplicated_keys(&self) -> Vec<(&str, &str)> {
        self.key_to_hash
            .iter()
            .filter_map(|(key, hash)| {
                let entry = self.entries.get(hash)?;
                if entry.canonical_key == *key {
                    return None;
                }
                Some((key.as_str(), entry.canonical_key.as_str()))
            })
            .collect()
    }

    /// Return a snapshot of index statistics.
    pub fn stats(&self) -> ContentDedupStats {
        let total_insertions = self.total_insertions;
        let total_duplicates = self.total_duplicates;
        let dedup_ratio = total_duplicates as f64 / total_insertions.max(1) as f64;
        ContentDedupStats {
            total_entries: self.entries.len(),
            total_keys: self.key_to_hash.len(),
            total_saved_bytes: self.total_saved_bytes,
            total_insertions,
            total_duplicates,
            dedup_ratio,
        }
    }

    /// Return the number of unique content hashes in the index.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` when the index contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return the number of key→hash mappings (including duplicates).
    pub fn key_count(&self) -> usize {
        self.key_to_hash.len()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// Evict the entry with the smallest ref_count, breaking ties by choosing
    /// the oldest entry (`first_seen`).
    fn evict_one(&mut self) {
        // Find the candidate hash.
        let victim = self
            .entries
            .iter()
            .min_by(|a, b| {
                a.1.ref_count
                    .cmp(&b.1.ref_count)
                    .then_with(|| a.1.first_seen.cmp(&b.1.first_seen))
            })
            .map(|(hash, _)| hash.clone());

        if let Some(victim_hash) = victim {
            self.entries.remove(&victim_hash);
            // Remove all key→hash mappings that pointed to the evicted entry.
            self.key_to_hash.retain(|_, h| *h != victim_hash);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::content_dedup_index::{
        ContentDedupConfig, ContentDedupResult, ContentDeduplicationIndex, ContentHash,
        DedupIndexError,
    };

    // ── helpers ──────────────────────────────────────────────────────────────

    fn default_config() -> ContentDedupConfig {
        ContentDedupConfig::default()
    }

    fn small_config(max_entries: usize) -> ContentDedupConfig {
        ContentDedupConfig {
            max_entries,
            min_block_size: 4,
            max_block_size: 1024,
            enable_ref_counting: true,
        }
    }

    fn make_data(seed: u8, len: usize) -> Vec<u8> {
        (0..len).map(|i| seed.wrapping_add(i as u8)).collect()
    }

    // ── ContentHash tests ─────────────────────────────────────────────────────

    #[test]
    fn test_content_hash_is_32_bytes() {
        let h = ContentDeduplicationIndex::compute_hash(b"hello");
        assert_eq!(h.0.len(), 32);
    }

    #[test]
    fn test_same_data_same_hash() {
        let h1 = ContentDeduplicationIndex::compute_hash(b"deterministic");
        let h2 = ContentDeduplicationIndex::compute_hash(b"deterministic");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_different_data_different_hash() {
        let h1 = ContentDeduplicationIndex::compute_hash(b"foo");
        let h2 = ContentDeduplicationIndex::compute_hash(b"bar");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hash_encodes_length_in_bytes_16_to_23() {
        let h = ContentDeduplicationIndex::compute_hash(b"abcd");
        let len = u64::from_le_bytes(h.0[16..24].try_into().unwrap());
        assert_eq!(len, 4u64);
    }

    #[test]
    fn test_hash_empty_data() {
        let h = ContentDeduplicationIndex::compute_hash(b"");
        let len = u64::from_le_bytes(h.0[16..24].try_into().unwrap());
        assert_eq!(len, 0u64);
    }

    #[test]
    fn test_content_hash_debug_format() {
        let h = ContentHash([0u8; 32]);
        let s = format!("{:?}", h);
        assert!(s.starts_with("ContentHash("));
        assert!(s.ends_with(')'));
    }

    #[test]
    fn test_content_hash_display_is_hex() {
        let h = ContentHash([0xABu8; 32]);
        let s = format!("{}", h);
        assert_eq!(s.len(), 64); // 32 bytes * 2 hex chars
        assert!(s.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_content_hash_clone_eq() {
        let h = ContentDeduplicationIndex::compute_hash(b"clone-me");
        let h2 = h.clone();
        assert_eq!(h, h2);
    }

    // ── Basic insert/lookup tests ─────────────────────────────────────────────

    #[test]
    fn test_insert_new_returns_new_variant() {
        let mut idx = ContentDeduplicationIndex::new(default_config());
        let result = idx.insert("k1".into(), b"some data here!!", 100).unwrap();
        assert!(matches!(result, ContentDedupResult::New { .. }));
    }

    #[test]
    fn test_insert_duplicate_returns_duplicate_variant() {
        let mut idx = ContentDeduplicationIndex::new(default_config());
        let data = make_data(7, 100);
        idx.insert("k1".into(), &data, 100).unwrap();
        let result = idx.insert("k2".into(), &data, 200).unwrap();
        assert!(matches!(result, ContentDedupResult::Duplicate { .. }));
    }

    #[test]
    fn test_duplicate_saved_bytes_matches_data_len() {
        let mut idx = ContentDeduplicationIndex::new(default_config());
        let data = make_data(1, 512);
        idx.insert("k1".into(), &data, 100).unwrap();
        let result = idx.insert("k2".into(), &data, 101).unwrap();
        if let ContentDedupResult::Duplicate { saved_bytes, .. } = result {
            assert_eq!(saved_bytes, 512);
        } else {
            panic!("expected Duplicate");
        }
    }

    #[test]
    fn test_duplicate_original_key_is_canonical() {
        let mut idx = ContentDeduplicationIndex::new(default_config());
        let data = make_data(2, 256);
        idx.insert("canonical".into(), &data, 1).unwrap();
        let result = idx.insert("alias".into(), &data, 2).unwrap();
        if let ContentDedupResult::Duplicate { original_key, .. } = result {
            assert_eq!(original_key, "canonical");
        } else {
            panic!("expected Duplicate");
        }
    }

    #[test]
    fn test_lookup_by_key_after_insert() {
        let mut idx = ContentDeduplicationIndex::new(default_config());
        let data = make_data(3, 128);
        idx.insert("mykey".into(), &data, 10).unwrap();
        let entry = idx.lookup_by_key("mykey").unwrap();
        assert_eq!(entry.canonical_key, "mykey");
        assert_eq!(entry.byte_size, 128);
    }

    #[test]
    fn test_lookup_by_hash_after_insert() {
        let mut idx = ContentDeduplicationIndex::new(default_config());
        let data = make_data(4, 64);
        let hash = ContentDeduplicationIndex::compute_hash(&data);
        idx.insert("hkey".into(), &data, 5).unwrap();
        assert!(idx.lookup_by_hash(&hash).is_some());
    }

    #[test]
    fn test_lookup_by_key_missing_returns_none() {
        let idx = ContentDeduplicationIndex::new(default_config());
        assert!(idx.lookup_by_key("nonexistent").is_none());
    }

    // ── is_duplicate tests ────────────────────────────────────────────────────

    #[test]
    fn test_is_duplicate_false_before_insert() {
        let idx = ContentDeduplicationIndex::new(default_config());
        assert!(!idx.is_duplicate(b"some bytes that are long enough to matter!"));
    }

    #[test]
    fn test_is_duplicate_true_after_insert() {
        let mut idx = ContentDeduplicationIndex::new(default_config());
        let data = make_data(5, 200);
        idx.insert("k1".into(), &data, 1).unwrap();
        assert!(idx.is_duplicate(&data));
    }

    // ── out-of-range pass-through tests ──────────────────────────────────────

    #[test]
    fn test_below_min_block_size_passthrough() {
        let config = ContentDedupConfig {
            min_block_size: 64,
            ..default_config()
        };
        let mut idx = ContentDeduplicationIndex::new(config);
        // data.len() = 10 < 64
        let result = idx.insert("tiny".into(), b"tooshort!!", 0).unwrap();
        assert!(matches!(result, ContentDedupResult::New { .. }));
        assert_eq!(idx.len(), 0, "tiny block must not be indexed");
    }

    #[test]
    fn test_above_max_block_size_passthrough() {
        let config = ContentDedupConfig {
            max_block_size: 128,
            min_block_size: 4,
            ..default_config()
        };
        let mut idx = ContentDeduplicationIndex::new(config);
        let big = make_data(9, 256);
        let result = idx.insert("big".into(), &big, 0).unwrap();
        assert!(matches!(result, ContentDedupResult::New { .. }));
        assert_eq!(idx.len(), 0, "oversized block must not be indexed");
    }

    #[test]
    fn test_passthrough_does_not_mark_duplicate() {
        let config = ContentDedupConfig {
            min_block_size: 64,
            max_block_size: 1024,
            ..default_config()
        };
        let mut idx = ContentDeduplicationIndex::new(config);
        let tiny = b"abc";
        idx.insert("a".into(), tiny, 0).unwrap();
        // Still not a duplicate because it was never indexed.
        let result = idx.insert("b".into(), tiny, 1).unwrap();
        assert!(matches!(result, ContentDedupResult::New { .. }));
    }

    // ── remove tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_remove_existing_key_returns_true() {
        let mut idx = ContentDeduplicationIndex::new(default_config());
        let data = make_data(6, 100);
        idx.insert("rem".into(), &data, 0).unwrap();
        assert!(idx.remove("rem"));
    }

    #[test]
    fn test_remove_nonexistent_key_returns_false() {
        let mut idx = ContentDeduplicationIndex::new(default_config());
        assert!(!idx.remove("ghost"));
    }

    #[test]
    fn test_remove_last_ref_deletes_entry() {
        let mut idx = ContentDeduplicationIndex::new(default_config());
        let data = make_data(7, 80);
        idx.insert("only".into(), &data, 0).unwrap();
        idx.remove("only");
        assert!(idx.lookup_by_key("only").is_none());
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn test_remove_one_of_two_refs_keeps_entry() {
        let mut idx = ContentDeduplicationIndex::new(default_config());
        let data = make_data(8, 80);
        idx.insert("a".into(), &data, 0).unwrap();
        idx.insert("b".into(), &data, 1).unwrap();
        idx.remove("b");
        // Entry still present because "a" still holds a reference.
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_remove_decrements_ref_count() {
        let config = ContentDedupConfig {
            enable_ref_counting: true,
            ..small_config(100)
        };
        let mut idx = ContentDeduplicationIndex::new(config);
        let data = make_data(9, 80);
        idx.insert("x1".into(), &data, 0).unwrap();
        idx.insert("x2".into(), &data, 1).unwrap(); // ref_count = 2
        idx.remove("x2"); // ref_count should drop to 1
        let entry = idx.lookup_by_key("x1").unwrap();
        assert_eq!(entry.ref_count, 1);
    }

    // ── ref counting tests ────────────────────────────────────────────────────

    #[test]
    fn test_ref_counting_increments_on_duplicate() {
        let config = ContentDedupConfig {
            enable_ref_counting: true,
            ..small_config(100)
        };
        let mut idx = ContentDeduplicationIndex::new(config);
        let data = make_data(10, 80);
        idx.insert("r1".into(), &data, 0).unwrap();
        idx.insert("r2".into(), &data, 1).unwrap();
        idx.insert("r3".into(), &data, 2).unwrap();
        let entry = idx.lookup_by_key("r1").unwrap();
        assert_eq!(entry.ref_count, 3);
    }

    #[test]
    fn test_ref_counting_disabled_stays_at_one() {
        let config = ContentDedupConfig {
            enable_ref_counting: false,
            ..small_config(100)
        };
        let mut idx = ContentDeduplicationIndex::new(config);
        let data = make_data(11, 80);
        idx.insert("d1".into(), &data, 0).unwrap();
        idx.insert("d2".into(), &data, 1).unwrap();
        let entry = idx.lookup_by_key("d1").unwrap();
        assert_eq!(entry.ref_count, 1);
    }

    // ── eviction tests ────────────────────────────────────────────────────────

    #[test]
    fn test_eviction_at_capacity() {
        let config = small_config(2);
        let mut idx = ContentDeduplicationIndex::new(config);
        idx.insert("e1".into(), &make_data(1, 10), 1).unwrap();
        idx.insert("e2".into(), &make_data(2, 10), 2).unwrap();
        // Inserting a third unique entry triggers eviction.
        idx.insert("e3".into(), &make_data(3, 10), 3).unwrap();
        assert_eq!(
            idx.len(),
            2,
            "index must stay at max_entries=2 after eviction"
        );
    }

    #[test]
    fn test_eviction_removes_lowest_ref_count() {
        let config = ContentDedupConfig {
            max_entries: 2,
            min_block_size: 4,
            max_block_size: 1024,
            enable_ref_counting: true,
        };
        let mut idx = ContentDeduplicationIndex::new(config);
        let data1 = make_data(20, 10);
        let data2 = make_data(21, 10);
        idx.insert("a".into(), &data1, 1).unwrap();
        idx.insert("b".into(), &data2, 2).unwrap();
        // Boost ref_count for data2.
        idx.insert("b_dup".into(), &data2, 3).unwrap();
        // data1 has ref_count=1, data2 has ref_count=2 — next unique evicts data1.
        let data3 = make_data(22, 10);
        idx.insert("c".into(), &data3, 4).unwrap();
        // data2 and data3 should survive; data1 was evicted.
        assert!(idx
            .lookup_by_hash(&ContentDeduplicationIndex::compute_hash(&data2))
            .is_some());
        assert!(idx
            .lookup_by_hash(&ContentDeduplicationIndex::compute_hash(&data3))
            .is_some());
    }

    // ── merge_duplicates / deduplicated_keys tests ───────────────────────────

    #[test]
    fn test_merge_duplicates_returns_zero_when_no_dups() {
        let mut idx = ContentDeduplicationIndex::new(default_config());
        idx.insert("u1".into(), &make_data(30, 100), 1).unwrap();
        idx.insert("u2".into(), &make_data(31, 100), 2).unwrap();
        assert_eq!(idx.merge_duplicates(), 0);
    }

    #[test]
    fn test_merge_duplicates_counts_duplicates() {
        let mut idx = ContentDeduplicationIndex::new(default_config());
        let data = make_data(32, 100);
        idx.insert("orig".into(), &data, 1).unwrap();
        idx.insert("dup1".into(), &data, 2).unwrap();
        idx.insert("dup2".into(), &data, 3).unwrap();
        // "dup1" and "dup2" differ from canonical "orig"
        assert_eq!(idx.merge_duplicates(), 2);
    }

    #[test]
    fn test_deduplicated_keys_correct_pairs() {
        let mut idx = ContentDeduplicationIndex::new(default_config());
        let data = make_data(33, 100);
        idx.insert("canon".into(), &data, 1).unwrap();
        idx.insert("alias".into(), &data, 2).unwrap();
        let pairs = idx.deduplicated_keys();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, "alias");
        assert_eq!(pairs[0].1, "canon");
    }

    #[test]
    fn test_deduplicated_keys_empty_when_all_unique() {
        let mut idx = ContentDeduplicationIndex::new(default_config());
        idx.insert("u1".into(), &make_data(40, 100), 1).unwrap();
        idx.insert("u2".into(), &make_data(41, 100), 2).unwrap();
        assert!(idx.deduplicated_keys().is_empty());
    }

    // ── stats tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_stats_initial_zero() {
        let idx = ContentDeduplicationIndex::new(default_config());
        let s = idx.stats();
        assert_eq!(s.total_entries, 0);
        assert_eq!(s.total_keys, 0);
        assert_eq!(s.total_saved_bytes, 0);
        assert_eq!(s.total_insertions, 0);
        assert_eq!(s.total_duplicates, 0);
        assert_eq!(s.dedup_ratio, 0.0);
    }

    #[test]
    fn test_stats_after_insert() {
        let mut idx = ContentDeduplicationIndex::new(default_config());
        idx.insert("s1".into(), &make_data(50, 128), 1).unwrap();
        let s = idx.stats();
        assert_eq!(s.total_entries, 1);
        assert_eq!(s.total_keys, 1);
        assert_eq!(s.total_insertions, 1);
        assert_eq!(s.total_duplicates, 0);
    }

    #[test]
    fn test_stats_total_saved_bytes_accumulates() {
        let mut idx = ContentDeduplicationIndex::new(default_config());
        let data = make_data(51, 200);
        idx.insert("p1".into(), &data, 1).unwrap();
        idx.insert("p2".into(), &data, 2).unwrap();
        idx.insert("p3".into(), &data, 3).unwrap();
        let s = idx.stats();
        // Two duplicate insertions each save 200 bytes.
        assert_eq!(s.total_saved_bytes, 400);
    }

    #[test]
    fn test_stats_dedup_ratio_calculation() {
        let mut idx = ContentDeduplicationIndex::new(default_config());
        let data = make_data(52, 200);
        idx.insert("q1".into(), &data, 1).unwrap();
        idx.insert("q2".into(), &data, 2).unwrap(); // 1 duplicate / 2 insertions
        let s = idx.stats();
        let expected = 1.0 / 2.0;
        assert!((s.dedup_ratio - expected).abs() < 1e-10);
    }

    #[test]
    fn test_stats_dedup_ratio_zero_insertions() {
        let idx = ContentDeduplicationIndex::new(default_config());
        let s = idx.stats();
        // dedup_ratio = 0 / max(1,0) = 0
        assert_eq!(s.dedup_ratio, 0.0);
    }

    // ── is_empty / len / key_count ────────────────────────────────────────────

    #[test]
    fn test_is_empty_on_new_index() {
        let idx = ContentDeduplicationIndex::new(default_config());
        assert!(idx.is_empty());
    }

    #[test]
    fn test_len_after_inserts() {
        let mut idx = ContentDeduplicationIndex::new(default_config());
        idx.insert("L1".into(), &make_data(60, 100), 1).unwrap();
        idx.insert("L2".into(), &make_data(61, 100), 2).unwrap();
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn test_key_count_includes_duplicates() {
        let mut idx = ContentDeduplicationIndex::new(default_config());
        let data = make_data(62, 100);
        idx.insert("K1".into(), &data, 1).unwrap();
        idx.insert("K2".into(), &data, 2).unwrap();
        // 1 unique entry, but 2 keys.
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.key_count(), 2);
    }

    // ── DedupIndexError tests ─────────────────────────────────────────────────

    #[test]
    fn test_error_display_key_already_exists() {
        let e = DedupIndexError::KeyAlreadyExists("foo".into());
        assert!(e.to_string().contains("foo"));
    }

    #[test]
    fn test_error_display_entry_not_found() {
        let e = DedupIndexError::EntryNotFound("bar".into());
        assert!(e.to_string().contains("bar"));
    }

    #[test]
    fn test_error_display_invalid_size() {
        let e = DedupIndexError::InvalidSize {
            size: 10,
            min: 64,
            max: 1024,
        };
        let s = e.to_string();
        assert!(s.contains("10"));
        assert!(s.contains("64"));
        assert!(s.contains("1024"));
    }

    // ── last_accessed timestamp test ──────────────────────────────────────────

    #[test]
    fn test_last_accessed_updates_on_duplicate() {
        let mut idx = ContentDeduplicationIndex::new(default_config());
        let data = make_data(70, 100);
        idx.insert("t1".into(), &data, 1000).unwrap();
        idx.insert("t2".into(), &data, 2000).unwrap();
        let entry = idx.lookup_by_key("t1").unwrap();
        assert_eq!(entry.last_accessed, 2000);
    }

    #[test]
    fn test_first_seen_preserved_on_duplicate() {
        let mut idx = ContentDeduplicationIndex::new(default_config());
        let data = make_data(71, 100);
        idx.insert("fs1".into(), &data, 500).unwrap();
        idx.insert("fs2".into(), &data, 600).unwrap();
        let entry = idx.lookup_by_key("fs1").unwrap();
        assert_eq!(entry.first_seen, 500);
    }
}
