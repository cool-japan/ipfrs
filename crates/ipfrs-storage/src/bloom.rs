//! Bloom filter for probabilistic block existence checks.
//!
//! Provides fast probabilistic `has()` checks with configurable false positive rates.
//! A bloom filter can quickly tell if a block definitely doesn't exist,
//! avoiding expensive disk lookups for cache misses.
//!
//! # Example
//!
//! ```rust,ignore
//! use ipfrs_storage::bloom::BloomFilter;
//!
//! let mut filter = BloomFilter::new(1_000_000, 0.01); // 1M items, 1% FPR
//! filter.insert(b"block_cid_bytes");
//! assert!(filter.contains(b"block_cid_bytes"));
//! assert!(!filter.contains(b"unknown")); // Probably false, might be true
//! ```

use ipfrs_core::{Cid, Error, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Default false positive rate (1%)
const DEFAULT_FALSE_POSITIVE_RATE: f64 = 0.01;

/// Bloom filter for fast probabilistic existence checks.
///
/// Uses multiple hash functions to minimize false positives while
/// maintaining constant-time lookups regardless of dataset size.
pub struct BloomFilter {
    /// Bit array for the bloom filter
    inner: RwLock<BloomFilterInner>,
    /// Configuration
    config: BloomConfig,
}

/// Inner mutable state of the bloom filter
#[derive(Serialize, Deserialize)]
struct BloomFilterInner {
    /// Bit vector
    bits: Vec<u64>,
    /// Number of items inserted
    count: usize,
}

/// Bloom filter configuration
#[derive(Debug, Clone)]
pub struct BloomConfig {
    /// Expected number of items
    pub expected_items: usize,
    /// Desired false positive rate (0.0 - 1.0)
    pub false_positive_rate: f64,
    /// Number of hash functions to use
    pub num_hashes: usize,
    /// Size of the bit array in bits
    pub num_bits: usize,
}

impl BloomConfig {
    /// Create a new configuration with given parameters
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        // Calculate optimal parameters
        // m = -n * ln(p) / (ln(2)^2) where m = bits, n = items, p = FPR
        let ln2_squared = std::f64::consts::LN_2 * std::f64::consts::LN_2;
        let num_bits =
            (-((expected_items as f64) * false_positive_rate.ln()) / ln2_squared).ceil() as usize;

        // k = (m/n) * ln(2) where k = hash functions
        let num_hashes =
            ((num_bits as f64 / expected_items as f64) * std::f64::consts::LN_2).ceil() as usize;

        // Ensure minimum values
        let num_bits = num_bits.max(64);
        let num_hashes = num_hashes.clamp(1, 16); // Cap at 16 hash functions

        Self {
            expected_items,
            false_positive_rate,
            num_hashes,
            num_bits,
        }
    }

    /// Create a configuration for low memory usage
    pub fn low_memory(expected_items: usize) -> Self {
        Self::new(expected_items, 0.05) // 5% FPR for smaller filter
    }

    /// Create a configuration for high accuracy
    pub fn high_accuracy(expected_items: usize) -> Self {
        Self::new(expected_items, 0.001) // 0.1% FPR
    }

    /// Calculate memory usage in bytes
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        // Round up to u64 boundary
        self.num_bits.div_ceil(64) * 8
    }
}

impl Default for BloomConfig {
    fn default() -> Self {
        Self::new(100_000, DEFAULT_FALSE_POSITIVE_RATE)
    }
}

impl BloomFilter {
    /// Create a new bloom filter with the given expected item count and false positive rate.
    ///
    /// # Arguments
    /// * `expected_items` - Expected number of items to be stored
    /// * `false_positive_rate` - Desired false positive rate (0.0 - 1.0)
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        let config = BloomConfig::new(expected_items, false_positive_rate);
        Self::with_config(config)
    }

    /// Create a bloom filter with custom configuration
    pub fn with_config(config: BloomConfig) -> Self {
        let num_u64s = config.num_bits.div_ceil(64);
        let inner = BloomFilterInner {
            bits: vec![0u64; num_u64s],
            count: 0,
        };
        Self {
            inner: RwLock::new(inner),
            config,
        }
    }

    /// Insert a CID into the bloom filter
    #[inline]
    pub fn insert_cid(&self, cid: &Cid) {
        self.insert(&cid.to_bytes());
    }

    /// Check if a CID might be in the bloom filter
    ///
    /// Returns `true` if the CID might be present (may be a false positive),
    /// Returns `false` if the CID is definitely not present.
    #[inline]
    pub fn contains_cid(&self, cid: &Cid) -> bool {
        self.contains(&cid.to_bytes())
    }

    /// Insert raw bytes into the bloom filter
    pub fn insert(&self, data: &[u8]) {
        let mut inner = self.inner.write();
        let hashes = self.compute_hashes(data);

        for hash in hashes {
            let bit_index = hash % self.config.num_bits;
            let word_index = bit_index / 64;
            let bit_offset = bit_index % 64;
            inner.bits[word_index] |= 1u64 << bit_offset;
        }
        inner.count += 1;
    }

    /// Check if raw bytes might be in the bloom filter
    pub fn contains(&self, data: &[u8]) -> bool {
        let inner = self.inner.read();
        let hashes = self.compute_hashes(data);

        for hash in hashes {
            let bit_index = hash % self.config.num_bits;
            let word_index = bit_index / 64;
            let bit_offset = bit_index % 64;
            if inner.bits[word_index] & (1u64 << bit_offset) == 0 {
                return false;
            }
        }
        true
    }

    /// Compute hash values for data using double hashing technique
    fn compute_hashes(&self, data: &[u8]) -> Vec<usize> {
        // Use FNV-1a for h1 and a different seed for h2
        let h1 = fnv1a_hash(data);
        let h2 = fnv1a_hash_with_seed(data, 0x811c_9dc5);

        let mut hashes = Vec::with_capacity(self.config.num_hashes);
        for i in 0..self.config.num_hashes {
            // Double hashing: h(i) = h1 + i * h2
            let hash = h1.wrapping_add((i as u64).wrapping_mul(h2));
            hashes.push(hash as usize);
        }
        hashes
    }

    /// Get the number of items inserted
    #[inline]
    pub fn count(&self) -> usize {
        self.inner.read().count
    }

    /// Get the fill ratio (proportion of bits set)
    pub fn fill_ratio(&self) -> f64 {
        let inner = self.inner.read();
        let set_bits: usize = inner.bits.iter().map(|w| w.count_ones() as usize).sum();
        set_bits as f64 / self.config.num_bits as f64
    }

    /// Estimate the actual false positive rate based on current fill
    pub fn estimated_fpr(&self) -> f64 {
        let fill = self.fill_ratio();
        fill.powi(self.config.num_hashes as i32)
    }

    /// Get memory usage in bytes
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        self.config.memory_bytes()
    }

    /// Clear the bloom filter
    pub fn clear(&self) {
        let mut inner = self.inner.write();
        for word in inner.bits.iter_mut() {
            *word = 0;
        }
        inner.count = 0;
    }

    /// Save the bloom filter to a file
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let inner = self.inner.read();
        let data = oxicode::serde::encode_to_vec(&*inner, oxicode::config::standard())
            .map_err(|e| Error::Serialization(format!("Failed to serialize bloom filter: {e}")))?;
        std::fs::write(path, data)
            .map_err(|e| Error::Storage(format!("Failed to write bloom filter: {e}")))?;
        Ok(())
    }

    /// Load the bloom filter from a file
    pub fn load_from_file(path: &Path, config: BloomConfig) -> Result<Self> {
        let data = std::fs::read(path)
            .map_err(|e| Error::Storage(format!("Failed to read bloom filter: {e}")))?;
        let inner: BloomFilterInner =
            oxicode::serde::decode_owned_from_slice(&data, oxicode::config::standard())
                .map(|(v, _)| v)
                .map_err(|e| {
                    Error::Deserialization(format!("Failed to deserialize bloom filter: {e}"))
                })?;

        // Verify the loaded filter matches expected config
        let expected_words = config.num_bits.div_ceil(64);
        if inner.bits.len() != expected_words {
            return Err(Error::InvalidData(format!(
                "Bloom filter size mismatch: expected {} words, got {}",
                expected_words,
                inner.bits.len()
            )));
        }

        Ok(Self {
            inner: RwLock::new(inner),
            config,
        })
    }

    /// Get bloom filter statistics
    pub fn stats(&self) -> BloomStats {
        BloomStats {
            count: self.count(),
            memory_bytes: self.memory_bytes(),
            fill_ratio: self.fill_ratio(),
            estimated_fpr: self.estimated_fpr(),
            num_bits: self.config.num_bits,
            num_hashes: self.config.num_hashes,
        }
    }
}

/// Statistics about a bloom filter
#[derive(Debug, Clone)]
pub struct BloomStats {
    /// Number of items inserted
    pub count: usize,
    /// Memory usage in bytes
    pub memory_bytes: usize,
    /// Proportion of bits set (0.0 - 1.0)
    pub fill_ratio: f64,
    /// Estimated false positive rate
    pub estimated_fpr: f64,
    /// Total number of bits
    pub num_bits: usize,
    /// Number of hash functions
    pub num_hashes: usize,
}

/// FNV-1a hash function
#[inline]
fn fnv1a_hash(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;

    let mut hash = FNV_OFFSET;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// FNV-1a hash with custom seed
#[inline]
fn fnv1a_hash_with_seed(data: &[u8], seed: u64) -> u64 {
    const FNV_PRIME: u64 = 0x0100_0000_01b3;

    let mut hash = seed;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Convenience constructor: create a `BloomFilter` backed by a fixed bit-count.
///
/// Rounds `bits` up to the next multiple of 64 and uses a two-hash (FNV-1a +
/// multiplicative) scheme with 7 probes — chosen for ~1 % FPR at 100 k elements
/// in a 1 M-bit filter.
impl BloomFilter {
    /// Create a filter with exactly `bits` capacity (rounded up to 64-bit boundary).
    ///
    /// Uses a fixed 7-probe two-hash scheme suitable for general-purpose deduplication.
    pub fn new_with_bits(bits: usize) -> Self {
        // Round up to next multiple of 64
        let rounded = bits.div_ceil(64) * 64;
        let config = BloomConfig {
            expected_items: 100_000,
            false_positive_rate: 0.01,
            num_hashes: 7,
            num_bits: rounded,
        };
        Self::with_config(config)
    }

    /// Number of elements inserted so far (alias for `count()`).
    #[inline]
    pub fn len(&self) -> usize {
        self.count()
    }

    /// Whether no elements have been inserted.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Whether no elements have been inserted (semantic alias, kept for test clarity).
    #[inline]
    pub fn is_bloom_empty(&self) -> bool {
        self.is_empty()
    }

    /// Total number of bits in the filter.
    #[inline]
    pub fn bit_count(&self) -> usize {
        self.config.num_bits
    }

    /// Probabilistic check: returns `false` iff the key is *definitely* absent.
    #[inline]
    pub fn may_contain(&self, key: &[u8]) -> bool {
        self.contains(key)
    }

    /// Fraction of bits currently set (0.0 – 1.0).
    #[inline]
    pub fn estimated_fill_ratio(&self) -> f64 {
        self.fill_ratio()
    }
}

// ─── BloomFilterConfig ────────────────────────────────────────────────────────

/// High-level configuration for the CID-oriented bloom filter layer.
#[derive(Debug, Clone)]
pub struct BloomFilterConfig {
    /// Total number of bits in the underlying bit array (default: 1 048 576 = 1 M bits).
    pub bits: usize,
    /// Expected number of elements to be inserted (used for documentation / stats only).
    pub expected_elements: usize,
}

impl Default for BloomFilterConfig {
    fn default() -> Self {
        Self {
            bits: 1_048_576,
            expected_elements: 100_000,
        }
    }
}

// ─── BloomSnapshot ────────────────────────────────────────────────────────────

/// Point-in-time snapshot of `CidBloomFilter` state.
#[derive(Debug, Clone)]
pub struct BloomSnapshot {
    /// Fraction of bits that are set (0.0 – 1.0).
    pub fill_ratio: f64,
    /// Estimated number of distinct elements inserted (via fill-ratio formula).
    pub estimated_elements: usize,
    /// Total capacity in bits.
    pub bit_count: usize,
}

// ─── CidBloomFilter ───────────────────────────────────────────────────────────

/// CID-specific wrapper around [`BloomFilter`] for write-time deduplication.
///
/// Converts CID strings to bytes and delegates to the inner filter.  All
/// operations are thread-safe via the `parking_lot::RwLock` inside `BloomFilter`.
pub struct CidBloomFilter {
    inner: BloomFilter,
    config: BloomFilterConfig,
}

impl CidBloomFilter {
    /// Create a new `CidBloomFilter` with the given configuration.
    pub fn new(config: BloomFilterConfig) -> Self {
        let filter = BloomFilter::new_with_bits(config.bits);
        Self {
            inner: filter,
            config,
        }
    }

    /// Create a `CidBloomFilter` with default configuration (1 M-bit filter).
    pub fn default_config() -> Self {
        Self::new(BloomFilterConfig::default())
    }

    /// Insert a CID (as a UTF-8 string) into the filter.
    #[inline]
    pub fn insert_cid(&self, cid: &str) {
        self.inner.insert(cid.as_bytes());
    }

    /// Returns `false` iff the CID is *definitely* not in the filter.
    #[inline]
    pub fn may_contain_cid(&self, cid: &str) -> bool {
        self.inner.may_contain(cid.as_bytes())
    }

    /// Take a snapshot of the current filter state.
    pub fn snapshot(&self) -> BloomSnapshot {
        let fill = self.inner.estimated_fill_ratio();
        let bit_count = self.inner.bit_count();

        // Estimate elements from fill ratio:
        //   fill ≈ 1 - exp(-k * n / m)  ⟹  n ≈ -m/k * ln(1 - fill)
        // k = num_hashes, m = bit_count
        let k = self.inner.config.num_hashes as f64;
        let m = bit_count as f64;
        let estimated_elements = if fill >= 1.0 {
            usize::MAX
        } else {
            let est = -(m / k) * (1.0 - fill).ln();
            est.round() as usize
        };

        BloomSnapshot {
            fill_ratio: fill,
            estimated_elements,
            bit_count,
        }
    }

    /// Clear the filter (all bits zeroed, count reset to zero).
    #[inline]
    pub fn reset(&self) {
        self.inner.clear();
    }

    /// Access the underlying `BloomFilter` directly.
    #[inline]
    pub fn inner(&self) -> &BloomFilter {
        &self.inner
    }

    /// Return the configuration this filter was created with.
    #[inline]
    pub fn config(&self) -> &BloomFilterConfig {
        &self.config
    }
}

impl Default for CidBloomFilter {
    fn default() -> Self {
        Self::default_config()
    }
}

/// Block store wrapper that uses a bloom filter for fast negative lookups
use crate::traits::BlockStore;
use async_trait::async_trait;
use ipfrs_core::Block;

pub struct BloomBlockStore<S: BlockStore> {
    store: S,
    filter: BloomFilter,
}

impl<S: BlockStore> BloomBlockStore<S> {
    /// Create a new bloom-filtered block store
    pub fn new(store: S, expected_items: usize, false_positive_rate: f64) -> Self {
        Self {
            store,
            filter: BloomFilter::new(expected_items, false_positive_rate),
        }
    }

    /// Create with custom bloom filter configuration
    pub fn with_config(store: S, config: BloomConfig) -> Self {
        Self {
            store,
            filter: BloomFilter::with_config(config),
        }
    }

    /// Rebuild the bloom filter from the store's contents
    pub fn rebuild_filter(&self) -> Result<()> {
        self.filter.clear();
        for cid in self.store.list_cids()? {
            self.filter.insert_cid(&cid);
        }
        Ok(())
    }

    /// Get bloom filter statistics
    pub fn bloom_stats(&self) -> BloomStats {
        self.filter.stats()
    }

    /// Get reference to underlying store
    #[inline]
    pub fn store(&self) -> &S {
        &self.store
    }
}

#[async_trait]
impl<S: BlockStore> BlockStore for BloomBlockStore<S> {
    async fn put(&self, block: &Block) -> Result<()> {
        self.filter.insert_cid(block.cid());
        self.store.put(block).await
    }

    async fn put_many(&self, blocks: &[Block]) -> Result<()> {
        for block in blocks {
            self.filter.insert_cid(block.cid());
        }
        self.store.put_many(blocks).await
    }

    async fn get(&self, cid: &Cid) -> Result<Option<Block>> {
        // Fast path: if bloom filter says no, definitely not there
        if !self.filter.contains_cid(cid) {
            return Ok(None);
        }
        // May be a false positive, check actual store
        self.store.get(cid).await
    }

    async fn has(&self, cid: &Cid) -> Result<bool> {
        // Fast path: if bloom filter says no, definitely not there
        if !self.filter.contains_cid(cid) {
            return Ok(false);
        }
        // May be a false positive, check actual store
        self.store.has(cid).await
    }

    async fn has_many(&self, cids: &[Cid]) -> Result<Vec<bool>> {
        // Check bloom filter first, only query store for maybes
        let mut results = Vec::with_capacity(cids.len());
        let mut to_check = Vec::new();
        let mut indices = Vec::new();

        for (i, cid) in cids.iter().enumerate() {
            if self.filter.contains_cid(cid) {
                to_check.push(*cid);
                indices.push(i);
            }
            results.push(false); // Default to false
        }

        // Only query store for CIDs that passed bloom filter
        if !to_check.is_empty() {
            let store_results = self.store.has_many(&to_check).await?;
            for (idx, exists) in indices.into_iter().zip(store_results) {
                results[idx] = exists;
            }
        }

        Ok(results)
    }

    async fn delete(&self, cid: &Cid) -> Result<()> {
        // Note: We don't remove from bloom filter (standard bloom filters don't support deletion)
        // The filter may have false positives for deleted items until rebuild
        self.store.delete(cid).await
    }

    async fn delete_many(&self, cids: &[Cid]) -> Result<()> {
        self.store.delete_many(cids).await
    }

    fn list_cids(&self) -> Result<Vec<Cid>> {
        self.store.list_cids()
    }

    fn len(&self) -> usize {
        self.store.len()
    }

    fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    async fn flush(&self) -> Result<()> {
        self.store.flush().await
    }

    async fn close(&self) -> Result<()> {
        self.store.close().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_filter_basic() {
        let filter = BloomFilter::new(1000, 0.01);

        filter.insert(b"hello");
        filter.insert(b"world");

        assert!(filter.contains(b"hello"));
        assert!(filter.contains(b"world"));
        assert!(!filter.contains(b"foo")); // Might be false positive, but unlikely
    }

    #[test]
    fn test_bloom_filter_false_positive_rate() {
        let filter = BloomFilter::new(10000, 0.01);

        // Insert 10000 items
        for i in 0i32..10000 {
            filter.insert(&i.to_le_bytes());
        }

        // Check false positives on items not inserted
        let mut false_positives = 0;
        for i in 10000i32..20000 {
            if filter.contains(&i.to_le_bytes()) {
                false_positives += 1;
            }
        }

        // Should be around 1% false positives (allow some margin)
        let fpr = false_positives as f64 / 10000.0;
        assert!(fpr < 0.03, "False positive rate {} too high", fpr);
    }

    #[test]
    fn test_bloom_config_memory() {
        let config = BloomConfig::new(1_000_000, 0.01);
        let memory_mb = config.memory_bytes() as f64 / (1024.0 * 1024.0);
        // Should be less than 10MB for 1M items (verified target)
        assert!(
            memory_mb < 10.0,
            "Memory {} MB exceeds 10MB target",
            memory_mb
        );
    }

    #[test]
    fn test_bloom_filter_stats() {
        let filter = BloomFilter::new(1000, 0.01);

        for i in 0i32..100 {
            filter.insert(&i.to_le_bytes());
        }

        let stats = filter.stats();
        assert_eq!(stats.count, 100);
        assert!(stats.fill_ratio > 0.0);
        assert!(stats.fill_ratio < 1.0);
    }

    // ── Tests for the new deduplication layer ────────────────────────────────

    /// 1. new_with_bits rounds bits up to 64-bit boundary correctly.
    #[test]
    fn test_new_with_bits_rounding() {
        let f = BloomFilter::new_with_bits(1);
        assert_eq!(f.bit_count(), 64, "1 bit should round up to 64");

        let f2 = BloomFilter::new_with_bits(65);
        assert_eq!(f2.bit_count(), 128, "65 bits should round up to 128");

        let f3 = BloomFilter::new_with_bits(1_048_576);
        assert_eq!(
            f3.bit_count(),
            1_048_576,
            "exact multiple must stay unchanged"
        );
    }

    /// 2. Zero false negatives: every inserted item is found.
    #[test]
    fn test_zero_false_negatives() {
        let filter = BloomFilter::new_with_bits(1_048_576);
        let items: Vec<String> = (0..500).map(|i| format!("item-{}", i)).collect();

        for item in &items {
            filter.insert(item.as_bytes());
        }
        for item in &items {
            assert!(
                filter.may_contain(item.as_bytes()),
                "False negative detected for '{}'",
                item
            );
        }
    }

    /// 3. may_contain returns false for items that were never inserted
    ///    (for clearly distinct keys this is deterministic).
    #[test]
    fn test_absent_keys_not_found() {
        let filter = BloomFilter::new_with_bits(1_048_576);
        // Nothing inserted — no key should be found.
        assert!(!filter.may_contain(b"never-inserted-key-abc"));
        assert!(!filter.may_contain(b"another-absent-key-xyz"));
    }

    /// 4. False-positive rate is < 1 % for 1 000 elements in a 1 M-bit filter.
    #[test]
    fn test_false_positive_rate_under_one_percent() {
        let filter = BloomFilter::new_with_bits(1_048_576);

        // Insert 1 000 items using a prefix that won't overlap with the probe set.
        for i in 0u32..1_000 {
            filter.insert(format!("inserted-{}", i).as_bytes());
        }

        // Probe 5 000 distinct keys that were NOT inserted.
        let mut false_positives = 0usize;
        let total = 5_000usize;
        for i in 0u32..total as u32 {
            if filter.may_contain(format!("probe-{}", i).as_bytes()) {
                false_positives += 1;
            }
        }
        let fpr = false_positives as f64 / total as f64;
        assert!(
            fpr < 0.01,
            "FPR {:.4} ≥ 1 % for 1 000 elements in 1 M-bit filter",
            fpr
        );
    }

    /// 5. clear() zeroes all bits and resets the element counter.
    #[test]
    fn test_clear_resets_filter() {
        let filter = BloomFilter::new_with_bits(1_048_576);
        filter.insert(b"key-a");
        filter.insert(b"key-b");
        assert!(filter.may_contain(b"key-a"));
        assert_eq!(filter.len(), 2);

        filter.clear();

        assert_eq!(filter.len(), 0);
        assert_eq!(filter.estimated_fill_ratio(), 0.0);
        assert!(
            !filter.may_contain(b"key-a"),
            "key-a should be absent after clear"
        );
        assert!(
            !filter.may_contain(b"key-b"),
            "key-b should be absent after clear"
        );
    }

    /// 6. estimated_fill_ratio grows monotonically with insertions.
    #[test]
    fn test_fill_ratio_grows_with_insertions() {
        let filter = BloomFilter::new_with_bits(1_048_576);
        let mut prev = filter.estimated_fill_ratio();

        for i in 0u32..200 {
            filter.insert(format!("grow-{}", i).as_bytes());
            let current = filter.estimated_fill_ratio();
            assert!(
                current >= prev,
                "fill_ratio decreased after insertion {} ({} < {})",
                i,
                current,
                prev
            );
            prev = current;
        }
        assert!(prev > 0.0, "fill_ratio must be positive after insertions");
    }

    /// 7. bit_count() and len() accessors return consistent values.
    #[test]
    fn test_accessors_consistency() {
        let filter = BloomFilter::new_with_bits(1_048_576);
        assert_eq!(filter.bit_count(), 1_048_576);
        assert_eq!(filter.len(), 0);

        filter.insert(b"x");
        assert_eq!(filter.len(), 1);
    }

    /// 8. CidBloomFilter – inserted CIDs are always found (zero false negatives).
    #[test]
    fn test_cid_bloom_zero_false_negatives() {
        let cbf = CidBloomFilter::default_config();
        let cids: Vec<String> = (0..300).map(|i| format!("Qm{:044}", i)).collect();

        for cid in &cids {
            cbf.insert_cid(cid);
        }
        for cid in &cids {
            assert!(
                cbf.may_contain_cid(cid),
                "CidBloomFilter false negative for '{}'",
                cid
            );
        }
    }

    /// 9. CidBloomFilter – absent CIDs are not found by default.
    #[test]
    fn test_cid_bloom_absent_cids() {
        let cbf = CidBloomFilter::default_config();
        assert!(!cbf.may_contain_cid("QmNeverInserted000000000000000000000000000000000"));
    }

    /// 10. CidBloomFilter::reset() clears the filter completely.
    #[test]
    fn test_cid_bloom_reset() {
        let cbf = CidBloomFilter::default_config();
        cbf.insert_cid("QmSomeTestCid0000000000000000000000000000000000");
        assert!(cbf.may_contain_cid("QmSomeTestCid0000000000000000000000000000000000"));

        cbf.reset();

        assert!(
            !cbf.may_contain_cid("QmSomeTestCid0000000000000000000000000000000000"),
            "CID should be absent after reset"
        );
        let snap = cbf.snapshot();
        assert_eq!(snap.fill_ratio, 0.0, "fill_ratio must be 0 after reset");
    }

    /// 11. BloomSnapshot reflects correct bit_count and fill_ratio direction.
    #[test]
    fn test_bloom_snapshot_fields() {
        let cbf = CidBloomFilter::new(BloomFilterConfig {
            bits: 1_048_576,
            expected_elements: 100_000,
        });

        let snap_before = cbf.snapshot();
        assert_eq!(snap_before.bit_count, 1_048_576);
        assert_eq!(snap_before.fill_ratio, 0.0);

        for i in 0u32..100 {
            cbf.insert_cid(&format!("Qm{:044}", i));
        }

        let snap_after = cbf.snapshot();
        assert!(
            snap_after.fill_ratio > 0.0,
            "fill_ratio must increase after insertions"
        );
        assert_eq!(snap_after.bit_count, 1_048_576);
        assert!(
            snap_after.estimated_elements > 0,
            "estimated_elements must be positive after insertions"
        );
    }

    /// 12. BloomFilterConfig default values are as specified.
    #[test]
    fn test_bloom_filter_config_defaults() {
        let cfg = BloomFilterConfig::default();
        assert_eq!(cfg.bits, 1_048_576, "default bits should be 1 048 576");
        assert_eq!(
            cfg.expected_elements, 100_000,
            "default expected_elements should be 100 000"
        );
    }

    /// 13. CidBloomFilter::snapshot() estimated_elements grows with insertions.
    #[test]
    fn test_snapshot_estimated_elements_grows() {
        let cbf = CidBloomFilter::default_config();
        let snap0 = cbf.snapshot();
        assert_eq!(snap0.estimated_elements, 0);

        for i in 0u32..500 {
            cbf.insert_cid(&format!("Qm{:044}", i));
        }
        let snap1 = cbf.snapshot();
        assert!(
            snap1.estimated_elements > 0,
            "estimated_elements should be > 0 after 500 insertions"
        );
    }

    /// 14. BloomFilter::is_bloom_empty() reflects insertion state.
    #[test]
    fn test_is_bloom_empty() {
        let f = BloomFilter::new_with_bits(1_048_576);
        assert!(f.is_bloom_empty(), "freshly created filter must be empty");
        f.insert(b"one");
        assert!(
            !f.is_bloom_empty(),
            "filter must not be empty after one insertion"
        );
        f.clear();
        assert!(f.is_bloom_empty(), "filter must be empty after clear");
    }
}
