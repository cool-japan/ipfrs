//! Bloom filter for probabilistic duplicate detection across peers.
//!
//! Provides a compact probabilistic set for tracking seen message IDs and
//! CIDs across peers, enabling fast duplicate detection with configurable
//! false-positive rates.

use std::collections::HashMap;

// FNV-1a 64-bit offset basis and prime
const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME: u64 = 1_099_511_628_211;

/// Configuration for a bloom filter.
#[derive(Debug, Clone)]
pub struct BloomConfig {
    /// Expected number of elements.
    pub capacity: usize,
    /// Target false-positive rate (between 0 and 1).
    pub false_positive_rate: f64,
}

impl Default for BloomConfig {
    fn default() -> Self {
        Self {
            capacity: 10_000,
            false_positive_rate: 0.01,
        }
    }
}

impl BloomConfig {
    /// Compute the optimal number of bits for this configuration.
    ///
    /// Formula: `bits = -(capacity * ln(fpr)) / (ln(2)^2)`, rounded up; min 64.
    pub fn optimal_bits(&self) -> usize {
        let capacity = self.capacity as f64;
        let fpr = self.false_positive_rate.max(f64::EPSILON);
        let ln2_sq = std::f64::consts::LN_2 * std::f64::consts::LN_2;
        let bits = -(capacity * fpr.ln()) / ln2_sq;
        (bits.ceil() as usize).max(64)
    }

    /// Compute the optimal number of hash functions for this configuration.
    ///
    /// Formula: `k = (bits/capacity) * ln(2)`, rounded up; min 1, max 32.
    pub fn optimal_hashes(&self) -> u32 {
        let bits = self.optimal_bits() as f64;
        let capacity = self.capacity as f64;
        let k = (bits / capacity) * std::f64::consts::LN_2;
        (k.ceil() as u32).clamp(1, 32)
    }
}

/// A single bloom filter backed by a bit array.
#[derive(Debug, Clone)]
pub struct BloomFilter {
    /// Bit array stored as bytes.
    bits: Vec<u8>,
    /// Total number of bits in the filter.
    num_bits: usize,
    /// Number of hash functions to apply per element.
    num_hashes: u32,
    /// Number of elements inserted so far.
    insertions: u64,
}

impl BloomFilter {
    /// Create a new bloom filter from a configuration.
    pub fn new(config: &BloomConfig) -> Self {
        let num_bits = config.optimal_bits();
        let num_hashes = config.optimal_hashes();
        let byte_count = num_bits.div_ceil(8);
        Self {
            bits: vec![0u8; byte_count],
            num_bits,
            num_hashes,
            insertions: 0,
        }
    }

    /// Insert an item into the filter.
    pub fn insert(&mut self, item: &[u8]) {
        for i in 0..self.num_hashes {
            let pos = self.hash_position(item, i);
            let byte_idx = pos / 8;
            let bit_idx = pos % 8;
            self.bits[byte_idx] |= 1u8 << bit_idx;
        }
        self.insertions += 1;
    }

    /// Check whether an item is (probably) in the filter.
    ///
    /// Returns `false` if the item was definitely not inserted. Returns
    /// `true` if the item was probably inserted (may be a false positive).
    pub fn contains(&self, item: &[u8]) -> bool {
        for i in 0..self.num_hashes {
            let pos = self.hash_position(item, i);
            let byte_idx = pos / 8;
            let bit_idx = pos % 8;
            if self.bits[byte_idx] & (1u8 << bit_idx) == 0 {
                return false;
            }
        }
        true
    }

    /// Clear the filter, resetting all bits and the insertion counter.
    pub fn clear(&mut self) {
        self.bits.iter_mut().for_each(|b| *b = 0);
        self.insertions = 0;
    }

    /// Estimate the current false-positive rate based on insertions.
    ///
    /// Formula: `(1 - e^(-k*n/m))^k`.
    pub fn estimated_fpr(&self) -> f64 {
        let k = self.num_hashes as f64;
        let n = self.insertions as f64;
        let m = self.num_bits as f64;
        if m == 0.0 {
            return 1.0;
        }
        (1.0_f64 - (-k * n / m).exp()).powf(k)
    }

    /// Return the number of elements inserted.
    pub fn insertions(&self) -> u64 {
        self.insertions
    }

    /// Return the total number of bits in the filter.
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// Return the number of hash functions used.
    pub fn num_hashes(&self) -> u32 {
        self.num_hashes
    }

    // Compute the bit position for item at hash index `i` using double-hashing
    // (Kirsch-Mitzenmacker scheme): h(i) = h1 + i*h2.
    fn hash_position(&self, data: &[u8], i: u32) -> usize {
        let h1 = fnv1a_seeded(data, 0);
        let h2 = fnv1a_seeded(data, 0x9e3779b9); // golden ratio seed
        let combined = h1.wrapping_add((i as u64).wrapping_mul(h2));
        (combined % self.num_bits as u64) as usize
    }
}

/// FNV-1a seeded hash.
///
/// Starts with `FNV_OFFSET_BASIS XOR seed as u64`, then processes data bytes.
pub fn fnv1a_seeded(data: &[u8], seed: u32) -> u64 {
    let mut hash = FNV_OFFSET_BASIS ^ (seed as u64);
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// A collection of named bloom filters sharing the same configuration.
///
/// Useful for tracking seen message IDs or CIDs independently per topic,
/// protocol, or peer group.
#[derive(Debug)]
pub struct PeerBloomFilter {
    filters: HashMap<String, BloomFilter>,
    config: BloomConfig,
}

impl PeerBloomFilter {
    /// Create a new `PeerBloomFilter` with the given configuration.
    pub fn new(config: BloomConfig) -> Self {
        Self {
            filters: HashMap::new(),
            config,
        }
    }

    /// Get the named filter, creating it if it does not exist.
    pub fn get_or_create(&mut self, name: &str) -> &mut BloomFilter {
        let config = &self.config;
        self.filters
            .entry(name.to_string())
            .or_insert_with(|| BloomFilter::new(config))
    }

    /// Insert `item` into the named filter, creating the filter if needed.
    pub fn insert(&mut self, name: &str, item: &[u8]) {
        self.get_or_create(name).insert(item);
    }

    /// Check whether `item` is (probably) in the named filter.
    ///
    /// Returns `false` if the filter does not exist or the item is absent.
    pub fn contains(&self, name: &str, item: &[u8]) -> bool {
        match self.filters.get(name) {
            Some(filter) => filter.contains(item),
            None => false,
        }
    }

    /// Clear the named filter.
    ///
    /// Returns `false` if the filter was not found.
    pub fn clear(&mut self, name: &str) -> bool {
        match self.filters.get_mut(name) {
            Some(filter) => {
                filter.clear();
                true
            }
            None => false,
        }
    }

    /// Remove the named filter entirely.
    ///
    /// Returns `false` if the filter was not found.
    pub fn remove_filter(&mut self, name: &str) -> bool {
        self.filters.remove(name).is_some()
    }

    /// Return the number of named filters currently managed.
    pub fn filter_count(&self) -> usize {
        self.filters.len()
    }

    /// Return aggregate statistics across all managed filters.
    pub fn stats(&self) -> BloomStats {
        let total_filters = self.filters.len();
        let total_insertions = self.filters.values().map(|f| f.insertions).sum();
        let avg_estimated_fpr = if total_filters == 0 {
            0.0
        } else {
            let sum: f64 = self.filters.values().map(|f| f.estimated_fpr()).sum();
            sum / total_filters as f64
        };
        BloomStats {
            total_filters,
            total_insertions,
            avg_estimated_fpr,
        }
    }
}

/// Aggregate statistics across all filters in a [`PeerBloomFilter`].
#[derive(Debug, Clone)]
pub struct BloomStats {
    /// Total number of named filters.
    pub total_filters: usize,
    /// Sum of insertions across all filters.
    pub total_insertions: u64,
    /// Average estimated false-positive rate across all filters.
    pub avg_estimated_fpr: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // BloomConfig tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_bloom_config_default() {
        let cfg = BloomConfig::default();
        assert_eq!(cfg.capacity, 10_000);
        assert!((cfg.false_positive_rate - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_optimal_bits_minimum() {
        // Very small capacity should still yield at least 64 bits.
        let cfg = BloomConfig {
            capacity: 1,
            false_positive_rate: 0.5,
        };
        assert!(cfg.optimal_bits() >= 64);
    }

    #[test]
    fn test_optimal_bits_formula() {
        let cfg = BloomConfig {
            capacity: 1_000,
            false_positive_rate: 0.01,
        };
        let bits = cfg.optimal_bits();
        // Known result for capacity=1000, fpr=0.01 is ~9586 bits.
        assert!(bits > 9_000 && bits < 10_200, "bits={bits}");
    }

    #[test]
    fn test_optimal_hashes_minimum() {
        let cfg = BloomConfig {
            capacity: 1,
            false_positive_rate: 0.99,
        };
        assert!(cfg.optimal_hashes() >= 1);
    }

    #[test]
    fn test_optimal_hashes_maximum() {
        // Extreme configs should be capped at 32.
        let cfg = BloomConfig {
            capacity: 1,
            false_positive_rate: 1e-300,
        };
        assert!(cfg.optimal_hashes() <= 32);
    }

    #[test]
    fn test_optimal_hashes_formula() {
        let cfg = BloomConfig {
            capacity: 1_000,
            false_positive_rate: 0.01,
        };
        // Known result for capacity=1000, fpr=0.01 is ~7 hashes.
        let k = cfg.optimal_hashes();
        assert!((6..=8).contains(&k), "k={k}");
    }

    // -------------------------------------------------------------------------
    // BloomFilter tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_bloom_filter_new() {
        let cfg = BloomConfig::default();
        let filter = BloomFilter::new(&cfg);
        assert_eq!(filter.insertions(), 0);
        assert!(filter.num_bits() >= 64);
        assert!(filter.num_hashes() >= 1);
    }

    #[test]
    fn test_bloom_filter_insert_and_contains() {
        let cfg = BloomConfig {
            capacity: 100,
            false_positive_rate: 0.01,
        };
        let mut filter = BloomFilter::new(&cfg);
        let item = b"hello_world";
        assert!(!filter.contains(item));
        filter.insert(item);
        assert!(filter.contains(item));
    }

    #[test]
    fn test_bloom_filter_insertions_counter() {
        let cfg = BloomConfig {
            capacity: 100,
            false_positive_rate: 0.01,
        };
        let mut filter = BloomFilter::new(&cfg);
        for i in 0u64..10 {
            filter.insert(&i.to_le_bytes());
        }
        assert_eq!(filter.insertions(), 10);
    }

    #[test]
    fn test_bloom_filter_contains_empty() {
        let cfg = BloomConfig::default();
        let filter = BloomFilter::new(&cfg);
        assert!(!filter.contains(b"anything"));
        assert!(!filter.contains(b""));
    }

    #[test]
    fn test_bloom_filter_clear_resets() {
        let cfg = BloomConfig {
            capacity: 100,
            false_positive_rate: 0.01,
        };
        let mut filter = BloomFilter::new(&cfg);
        filter.insert(b"item_a");
        filter.insert(b"item_b");
        assert_eq!(filter.insertions(), 2);
        filter.clear();
        assert_eq!(filter.insertions(), 0);
        // After clear the item should no longer be detected.
        assert!(!filter.contains(b"item_a"));
        assert!(!filter.contains(b"item_b"));
    }

    #[test]
    fn test_bloom_filter_no_false_negatives() {
        // Bloom filters must never produce false negatives.
        let cfg = BloomConfig {
            capacity: 1_000,
            false_positive_rate: 0.01,
        };
        let mut filter = BloomFilter::new(&cfg);
        for i in 0u64..500 {
            let bytes = i.to_le_bytes();
            filter.insert(&bytes);
            assert!(filter.contains(&bytes), "false negative for i={i}");
        }
    }

    #[test]
    fn test_bloom_filter_low_false_positive_rate() {
        let cfg = BloomConfig {
            capacity: 1_000,
            false_positive_rate: 0.01,
        };
        let mut filter = BloomFilter::new(&cfg);
        // Insert 500 known items.
        for i in 0u64..500 {
            filter.insert(&i.to_le_bytes());
        }
        // Check 500 items that were not inserted.
        let mut fp_count = 0usize;
        for i in 10_000u64..10_500 {
            if filter.contains(&i.to_le_bytes()) {
                fp_count += 1;
            }
        }
        // False positive rate should be well below 5% for these parameters.
        assert!(fp_count < 25, "too many false positives: {fp_count}/500");
    }

    #[test]
    fn test_estimated_fpr_increases_with_insertions() {
        let cfg = BloomConfig {
            capacity: 1_000,
            false_positive_rate: 0.01,
        };
        let mut filter = BloomFilter::new(&cfg);
        let fpr0 = filter.estimated_fpr();
        for i in 0u64..100 {
            filter.insert(&i.to_le_bytes());
        }
        let fpr100 = filter.estimated_fpr();
        for i in 100u64..500 {
            filter.insert(&i.to_le_bytes());
        }
        let fpr500 = filter.estimated_fpr();
        assert!(fpr0 < fpr100, "fpr should increase after insertions");
        assert!(fpr100 < fpr500, "fpr should increase with more insertions");
    }

    #[test]
    fn test_estimated_fpr_zero_for_empty_filter() {
        let cfg = BloomConfig::default();
        let filter = BloomFilter::new(&cfg);
        assert!(filter.estimated_fpr() < 1e-10);
    }

    #[test]
    fn test_bloom_filter_empty_slice() {
        let cfg = BloomConfig {
            capacity: 100,
            false_positive_rate: 0.01,
        };
        let mut filter = BloomFilter::new(&cfg);
        filter.insert(b"");
        assert!(filter.contains(b""));
        assert!(!filter.contains(b"x"));
    }

    #[test]
    fn test_bloom_filter_large_item() {
        let cfg = BloomConfig {
            capacity: 100,
            false_positive_rate: 0.01,
        };
        let mut filter = BloomFilter::new(&cfg);
        let large_item = vec![42u8; 1024];
        filter.insert(&large_item);
        assert!(filter.contains(&large_item));
        let other_item = vec![43u8; 1024];
        // Different item should not be present (this is deterministic).
        let _ = filter.contains(&other_item); // just ensure no panic
    }

    // -------------------------------------------------------------------------
    // fnv1a_seeded tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_fnv1a_seeded_different_seeds_differ() {
        let data = b"test_data";
        let h0 = fnv1a_seeded(data, 0);
        let h1 = fnv1a_seeded(data, 1);
        let h2 = fnv1a_seeded(data, 2);
        assert_ne!(h0, h1);
        assert_ne!(h1, h2);
        assert_ne!(h0, h2);
    }

    #[test]
    fn test_fnv1a_seeded_deterministic() {
        let data = b"determinism";
        assert_eq!(fnv1a_seeded(data, 7), fnv1a_seeded(data, 7));
    }

    // -------------------------------------------------------------------------
    // PeerBloomFilter tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_peer_bloom_filter_new() {
        let pbf = PeerBloomFilter::new(BloomConfig::default());
        assert_eq!(pbf.filter_count(), 0);
    }

    #[test]
    fn test_peer_bloom_filter_get_or_create_auto_creates() {
        let mut pbf = PeerBloomFilter::new(BloomConfig::default());
        assert_eq!(pbf.filter_count(), 0);
        let _ = pbf.get_or_create("alpha");
        assert_eq!(pbf.filter_count(), 1);
        // Calling again for the same name should not create a second filter.
        let _ = pbf.get_or_create("alpha");
        assert_eq!(pbf.filter_count(), 1);
    }

    #[test]
    fn test_peer_bloom_filter_insert_contains() {
        let mut pbf = PeerBloomFilter::new(BloomConfig {
            capacity: 100,
            false_positive_rate: 0.01,
        });
        pbf.insert("msgs", b"cid_abc");
        assert!(pbf.contains("msgs", b"cid_abc"));
        assert!(!pbf.contains("msgs", b"cid_xyz"));
    }

    #[test]
    fn test_peer_bloom_filter_contains_missing_filter() {
        let pbf = PeerBloomFilter::new(BloomConfig::default());
        assert!(!pbf.contains("nonexistent", b"anything"));
    }

    #[test]
    fn test_peer_bloom_filter_clear_existing() {
        let mut pbf = PeerBloomFilter::new(BloomConfig {
            capacity: 100,
            false_positive_rate: 0.01,
        });
        pbf.insert("f1", b"item");
        assert!(pbf.contains("f1", b"item"));
        let cleared = pbf.clear("f1");
        assert!(cleared);
        assert!(!pbf.contains("f1", b"item"));
    }

    #[test]
    fn test_peer_bloom_filter_clear_missing_returns_false() {
        let mut pbf = PeerBloomFilter::new(BloomConfig::default());
        assert!(!pbf.clear("ghost"));
    }

    #[test]
    fn test_peer_bloom_filter_remove_filter() {
        let mut pbf = PeerBloomFilter::new(BloomConfig::default());
        pbf.insert("to_remove", b"data");
        assert_eq!(pbf.filter_count(), 1);
        let removed = pbf.remove_filter("to_remove");
        assert!(removed);
        assert_eq!(pbf.filter_count(), 0);
        // Removing again should return false.
        assert!(!pbf.remove_filter("to_remove"));
    }

    #[test]
    fn test_peer_bloom_filter_multiple_filters_independent() {
        let mut pbf = PeerBloomFilter::new(BloomConfig {
            capacity: 200,
            false_positive_rate: 0.01,
        });
        pbf.insert("filter_a", b"shared_key");
        pbf.insert("filter_b", b"only_b");

        // filter_a contains "shared_key", filter_b does not.
        assert!(pbf.contains("filter_a", b"shared_key"));
        assert!(!pbf.contains("filter_b", b"shared_key"));
        // "only_b" was only inserted into filter_b.
        assert!(!pbf.contains("filter_a", b"only_b"));
        assert!(pbf.contains("filter_b", b"only_b"));
    }

    #[test]
    fn test_peer_bloom_filter_multiple_filters_isolated() {
        let mut pbf = PeerBloomFilter::new(BloomConfig {
            capacity: 1_000,
            false_positive_rate: 0.001,
        });
        // Insert distinct items into separate filters.
        for i in 0u64..50 {
            pbf.insert("even", &(i * 2).to_le_bytes());
            pbf.insert("odd", &(i * 2 + 1).to_le_bytes());
        }
        // Verify filters are separate.
        assert_eq!(pbf.filter_count(), 2);
        // Even filter should not contain odd items (with overwhelming probability).
        let mut cross_hits = 0usize;
        for i in 0u64..50 {
            if pbf.contains("even", &(i * 2 + 1).to_le_bytes()) {
                cross_hits += 1;
            }
        }
        assert!(cross_hits < 5, "cross_hits={cross_hits}");
    }

    #[test]
    fn test_peer_bloom_filter_filter_count() {
        let mut pbf = PeerBloomFilter::new(BloomConfig::default());
        assert_eq!(pbf.filter_count(), 0);
        pbf.insert("a", b"x");
        pbf.insert("b", b"y");
        pbf.insert("c", b"z");
        assert_eq!(pbf.filter_count(), 3);
        pbf.remove_filter("b");
        assert_eq!(pbf.filter_count(), 2);
    }

    // -------------------------------------------------------------------------
    // BloomStats tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_stats_empty() {
        let pbf = PeerBloomFilter::new(BloomConfig::default());
        let stats = pbf.stats();
        assert_eq!(stats.total_filters, 0);
        assert_eq!(stats.total_insertions, 0);
        assert!((stats.avg_estimated_fpr - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats_multiple_filters() {
        let mut pbf = PeerBloomFilter::new(BloomConfig {
            capacity: 500,
            false_positive_rate: 0.01,
        });
        for i in 0u64..10 {
            pbf.insert("f1", &i.to_le_bytes());
        }
        for i in 0u64..20 {
            pbf.insert("f2", &i.to_le_bytes());
        }
        let stats = pbf.stats();
        assert_eq!(stats.total_filters, 2);
        assert_eq!(stats.total_insertions, 30);
        assert!(stats.avg_estimated_fpr >= 0.0);
        assert!(stats.avg_estimated_fpr <= 1.0);
    }

    #[test]
    fn test_stats_fpr_reflects_insertions() {
        let mut pbf = PeerBloomFilter::new(BloomConfig {
            capacity: 100,
            false_positive_rate: 0.01,
        });
        pbf.insert("f", b"a");
        let fpr_early = pbf.stats().avg_estimated_fpr;
        for i in 0u64..90 {
            pbf.insert("f", &i.to_le_bytes());
        }
        let fpr_late = pbf.stats().avg_estimated_fpr;
        assert!(fpr_late > fpr_early);
    }
}
