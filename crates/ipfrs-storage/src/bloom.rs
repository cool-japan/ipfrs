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
}
