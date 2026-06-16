//! Storage Block Packer — packs multiple small blocks into larger "pack files"
//! to reduce storage overhead and improve sequential read performance,
//! similar to Git's packfile format.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// FNV-1a 64-bit hash
// ---------------------------------------------------------------------------

/// Computes the FNV-1a 64-bit hash of `bytes`.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;

    let mut hash = OFFSET_BASIS;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

// ---------------------------------------------------------------------------
// PackEntry
// ---------------------------------------------------------------------------

/// A single block entry within a pack file.
#[derive(Debug, Clone, PartialEq)]
pub struct PackEntry {
    /// Content identifier for this block.
    pub cid: String,
    /// Byte offset of this block within the pack file.
    pub offset: u64,
    /// Size of this block in bytes.
    pub size_bytes: u64,
    /// FNV-1a 64-bit hash of the CID bytes, for integrity verification.
    pub checksum: u64,
}

// ---------------------------------------------------------------------------
// Pack
// ---------------------------------------------------------------------------

/// A single pack file that contains multiple block entries stored sequentially.
#[derive(Debug, Clone)]
pub struct Pack {
    /// Unique identifier for this pack.
    pub pack_id: u64,
    /// Entries within this pack, sorted by offset ascending.
    pub entries: Vec<PackEntry>,
    /// Sum of all entry sizes in bytes (not including header overhead).
    pub total_size_bytes: u64,
    /// Unix timestamp (seconds) when this pack was created.
    pub created_at_secs: u64,
}

impl Pack {
    /// Returns the number of entries in this pack.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if this pack contains a block with the given CID.
    pub fn contains(&self, cid: &str) -> bool {
        self.entries.iter().any(|e| e.cid == cid)
    }

    /// Returns a reference to the `PackEntry` for the given CID, or `None`.
    pub fn find(&self, cid: &str) -> Option<&PackEntry> {
        self.entries.iter().find(|e| e.cid == cid)
    }
}

// ---------------------------------------------------------------------------
// PackerConfig
// ---------------------------------------------------------------------------

/// Configuration for the `StorageBlockPacker`.
#[derive(Debug, Clone)]
pub struct PackerConfig {
    /// Maximum total size (in bytes) of a single pack file. Default: 64 MiB.
    pub max_pack_size_bytes: u64,
    /// Blocks strictly smaller than this threshold are candidates for packing.
    /// Default: 64 KiB.
    pub min_block_size_bytes: u64,
    /// Maximum number of entries per pack. Default: 1024.
    pub max_entries_per_pack: usize,
}

impl Default for PackerConfig {
    fn default() -> Self {
        Self {
            max_pack_size_bytes: 67_108_864, // 64 MiB
            min_block_size_bytes: 65_536,    // 64 KiB
            max_entries_per_pack: 1024,
        }
    }
}

// ---------------------------------------------------------------------------
// PackerStats
// ---------------------------------------------------------------------------

/// Aggregate statistics over all packs managed by a `StorageBlockPacker`.
#[derive(Debug, Clone, PartialEq)]
pub struct PackerStats {
    /// Total number of packs.
    pub total_packs: usize,
    /// Total number of entries across all packs.
    pub total_entries: usize,
    /// Total bytes packed across all packs.
    pub total_packed_bytes: u64,
    /// Average pack utilization: mean(pack.total_size_bytes / max_pack_size_bytes).
    /// Returns 0.0 if there are no packs.
    pub avg_pack_utilization: f64,
}

// ---------------------------------------------------------------------------
// StorageBlockPacker
// ---------------------------------------------------------------------------

/// Packs multiple small blocks into larger pack files for efficient storage.
pub struct StorageBlockPacker {
    /// All packs, keyed by pack_id.
    pub packs: HashMap<u64, Pack>,
    /// The next pack_id to assign.
    pub next_pack_id: u64,
    /// Configuration controlling packing behavior.
    pub config: PackerConfig,
}

impl StorageBlockPacker {
    /// Creates a new `StorageBlockPacker` with the given configuration.
    pub fn new(config: PackerConfig) -> Self {
        Self {
            packs: HashMap::new(),
            next_pack_id: 1,
            config,
        }
    }

    /// Packs a list of `(cid, size_bytes)` blocks into one or more pack files.
    ///
    /// Only blocks where `size_bytes < min_block_size_bytes` are eligible.
    /// Blocks are greedily packed: a new pack is started whenever the current
    /// pack would exceed `max_pack_size_bytes` or `max_entries_per_pack`.
    ///
    /// Returns the list of newly created pack IDs (empty if no eligible blocks).
    pub fn pack(&mut self, blocks: Vec<(String, u64)>, now_secs: u64) -> Vec<u64> {
        // Filter to only eligible (small) blocks
        let eligible: Vec<(String, u64)> = blocks
            .into_iter()
            .filter(|(_, size)| *size < self.config.min_block_size_bytes)
            .collect();

        if eligible.is_empty() {
            return Vec::new();
        }

        let mut created_ids: Vec<u64> = Vec::new();

        // State for the current pack being built
        let mut current_entries: Vec<PackEntry> = Vec::new();
        let mut current_size: u64 = 0;
        let mut current_offset: u64 = 0;

        for (cid, size_bytes) in eligible {
            // Determine if adding this block would exceed limits
            let would_exceed_size = current_size + size_bytes > self.config.max_pack_size_bytes;
            let would_exceed_entries = current_entries.len() >= self.config.max_entries_per_pack;

            if !current_entries.is_empty() && (would_exceed_size || would_exceed_entries) {
                // Flush the current pack
                let pack_id = self.next_pack_id;
                self.next_pack_id += 1;

                let pack = Pack {
                    pack_id,
                    total_size_bytes: current_size,
                    entries: current_entries,
                    created_at_secs: now_secs,
                };
                self.packs.insert(pack_id, pack);
                created_ids.push(pack_id);

                // Reset for the next pack
                current_entries = Vec::new();
                current_size = 0;
                current_offset = 0;
            }

            let checksum = fnv1a(cid.as_bytes());
            let entry = PackEntry {
                cid,
                offset: current_offset,
                size_bytes,
                checksum,
            };
            current_offset += size_bytes;
            current_size += size_bytes;
            current_entries.push(entry);
        }

        // Flush any remaining entries
        if !current_entries.is_empty() {
            let pack_id = self.next_pack_id;
            self.next_pack_id += 1;

            let pack = Pack {
                pack_id,
                total_size_bytes: current_size,
                entries: current_entries,
                created_at_secs: now_secs,
            };
            self.packs.insert(pack_id, pack);
            created_ids.push(pack_id);
        }

        created_ids
    }

    /// Searches all packs for the given CID.
    ///
    /// Returns `Some((&Pack, &PackEntry))` if found, `None` otherwise.
    pub fn find_pack(&self, cid: &str) -> Option<(&Pack, &PackEntry)> {
        for pack in self.packs.values() {
            if let Some(entry) = pack.find(cid) {
                return Some((pack, entry));
            }
        }
        None
    }

    /// Returns a reference to the pack with the given ID, or `None`.
    pub fn get_pack(&self, pack_id: u64) -> Option<&Pack> {
        self.packs.get(&pack_id)
    }

    /// Removes the pack with the given ID.
    ///
    /// Returns `true` if the pack existed and was removed, `false` otherwise.
    pub fn delete_pack(&mut self, pack_id: u64) -> bool {
        self.packs.remove(&pack_id).is_some()
    }

    /// Computes aggregate statistics over all packs.
    pub fn stats(&self) -> PackerStats {
        let total_packs = self.packs.len();
        let total_entries = self.packs.values().map(|p| p.entry_count()).sum();
        let total_packed_bytes = self.packs.values().map(|p| p.total_size_bytes).sum();

        let avg_pack_utilization = if total_packs == 0 {
            0.0
        } else {
            let sum: f64 = self
                .packs
                .values()
                .map(|p| p.total_size_bytes as f64 / self.config.max_pack_size_bytes as f64)
                .sum();
            sum / total_packs as f64
        };

        PackerStats {
            total_packs,
            total_entries,
            total_packed_bytes,
            avg_pack_utilization,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_packer() -> StorageBlockPacker {
        StorageBlockPacker::new(PackerConfig::default())
    }

    fn small_block(cid: &str, kb: u64) -> (String, u64) {
        (cid.to_string(), kb * 1024)
    }

    fn large_block(cid: &str) -> (String, u64) {
        // Blocks >= 64 KiB should be filtered out (default min_block_size_bytes = 65536)
        (cid.to_string(), 65_536)
    }

    // 1. new() starts empty
    #[test]
    fn test_new_starts_empty() {
        let packer = default_packer();
        assert!(packer.packs.is_empty());
        assert_eq!(packer.next_pack_id, 1);
    }

    // 2. pack returns empty for no eligible blocks
    #[test]
    fn test_pack_empty_input_returns_empty() {
        let mut packer = default_packer();
        let ids = packer.pack(vec![], 100);
        assert!(ids.is_empty());
    }

    // 3. pack filters blocks >= min_block_size_bytes
    #[test]
    fn test_pack_filters_large_blocks() {
        let mut packer = default_packer();
        let blocks = vec![large_block("cid-large")];
        let ids = packer.pack(blocks, 100);
        assert!(ids.is_empty(), "large block should be filtered out");
        assert!(packer.packs.is_empty());
    }

    // 4. pack creates single pack for small set
    #[test]
    fn test_pack_creates_single_pack() {
        let mut packer = default_packer();
        let blocks = vec![small_block("cid-a", 1), small_block("cid-b", 2)];
        let ids = packer.pack(blocks, 100);
        assert_eq!(ids.len(), 1);
        assert_eq!(packer.packs.len(), 1);
    }

    // 5. pack creates multiple packs when size exceeded
    #[test]
    fn test_pack_creates_multiple_packs_on_size_overflow() {
        let config = PackerConfig {
            max_pack_size_bytes: 10_000,
            min_block_size_bytes: 65_536,
            max_entries_per_pack: 1024,
        };
        let mut packer = StorageBlockPacker::new(config);
        // Each block is 6000 bytes; two blocks exceed 10000 bytes
        let blocks = vec![
            ("cid-a".to_string(), 6000u64),
            ("cid-b".to_string(), 6000u64),
            ("cid-c".to_string(), 6000u64),
        ];
        let ids = packer.pack(blocks, 200);
        // First pack: cid-a (6000). cid-b would make 12000 > 10000, so flush.
        // Second pack: cid-b (6000). cid-c would make 12000 > 10000, so flush.
        // Third pack: cid-c (6000).
        assert_eq!(ids.len(), 3);
        assert_eq!(packer.packs.len(), 3);
    }

    // 6. pack creates new pack when max_entries reached
    #[test]
    fn test_pack_creates_new_pack_on_max_entries() {
        let config = PackerConfig {
            max_pack_size_bytes: 67_108_864,
            min_block_size_bytes: 65_536,
            max_entries_per_pack: 2,
        };
        let mut packer = StorageBlockPacker::new(config);
        let blocks = vec![
            ("cid-1".to_string(), 100u64),
            ("cid-2".to_string(), 100u64),
            ("cid-3".to_string(), 100u64),
        ];
        let ids = packer.pack(blocks, 300);
        // Pack 1: cid-1, cid-2 (at max_entries). cid-3 triggers new pack.
        // Pack 2: cid-3
        assert_eq!(ids.len(), 2);
        let pack1 = packer.get_pack(ids[0]).expect("pack 1 should exist");
        assert_eq!(pack1.entry_count(), 2);
        let pack2 = packer.get_pack(ids[1]).expect("pack 2 should exist");
        assert_eq!(pack2.entry_count(), 1);
    }

    // 7. PackEntry offset computed correctly
    #[test]
    fn test_pack_entry_offsets() {
        let mut packer = default_packer();
        let blocks = vec![
            ("cid-x".to_string(), 100u64),
            ("cid-y".to_string(), 250u64),
            ("cid-z".to_string(), 50u64),
        ];
        let ids = packer.pack(blocks, 100);
        assert_eq!(ids.len(), 1);
        let pack = packer.get_pack(ids[0]).expect("pack should exist");

        let x = pack.find("cid-x").expect("cid-x not found");
        assert_eq!(x.offset, 0);

        let y = pack.find("cid-y").expect("cid-y not found");
        assert_eq!(y.offset, 100);

        let z = pack.find("cid-z").expect("cid-z not found");
        assert_eq!(z.offset, 350);
    }

    // 8. PackEntry checksum = fnv1a(cid)
    #[test]
    fn test_pack_entry_checksum() {
        let mut packer = default_packer();
        let cid = "QmTestChecksum";
        let blocks = vec![(cid.to_string(), 500u64)];
        let ids = packer.pack(blocks, 100);
        let pack = packer.get_pack(ids[0]).expect("pack should exist");
        let entry = pack.find(cid).expect("entry not found");
        assert_eq!(entry.checksum, fnv1a(cid.as_bytes()));
    }

    // 9. Pack.contains correct
    #[test]
    fn test_pack_contains() {
        let mut packer = default_packer();
        let blocks = vec![small_block("cid-p", 1), small_block("cid-q", 2)];
        let ids = packer.pack(blocks, 100);
        let pack = packer.get_pack(ids[0]).expect("pack should exist");
        assert!(pack.contains("cid-p"));
        assert!(pack.contains("cid-q"));
        assert!(!pack.contains("cid-missing"));
    }

    // 10. Pack.find returns correct entry
    #[test]
    fn test_pack_find_returns_correct_entry() {
        let mut packer = default_packer();
        let blocks = vec![("cid-find".to_string(), 1234u64)];
        let ids = packer.pack(blocks, 100);
        let pack = packer.get_pack(ids[0]).expect("pack should exist");
        let entry = pack.find("cid-find").expect("entry not found");
        assert_eq!(entry.cid, "cid-find");
        assert_eq!(entry.size_bytes, 1234);
        assert!(pack.find("nonexistent").is_none());
    }

    // 11. Pack.entry_count correct
    #[test]
    fn test_pack_entry_count() {
        let mut packer = default_packer();
        let blocks = vec![
            small_block("c1", 1),
            small_block("c2", 2),
            small_block("c3", 3),
        ];
        let ids = packer.pack(blocks, 100);
        let pack = packer.get_pack(ids[0]).expect("pack should exist");
        assert_eq!(pack.entry_count(), 3);
    }

    // 12. find_pack searches across packs
    #[test]
    fn test_find_pack_searches_across_packs() {
        let config = PackerConfig {
            max_pack_size_bytes: 10_000,
            min_block_size_bytes: 65_536,
            max_entries_per_pack: 1024,
        };
        let mut packer = StorageBlockPacker::new(config);
        // Force two packs
        let blocks = vec![
            ("cid-first".to_string(), 6000u64),
            ("cid-second".to_string(), 6000u64),
        ];
        let ids = packer.pack(blocks, 100);
        assert_eq!(ids.len(), 2);

        let result = packer.find_pack("cid-first");
        assert!(result.is_some());
        let (_, entry) = result.expect("should find cid-first");
        assert_eq!(entry.cid, "cid-first");

        let result2 = packer.find_pack("cid-second");
        assert!(result2.is_some());
        let (_, entry2) = result2.expect("should find cid-second");
        assert_eq!(entry2.cid, "cid-second");
    }

    // 13. find_pack returns None for unknown cid
    #[test]
    fn test_find_pack_returns_none_for_unknown() {
        let mut packer = default_packer();
        packer.pack(vec![small_block("known", 1)], 100);
        assert!(packer.find_pack("unknown-cid").is_none());
    }

    // 14. get_pack Some/None
    #[test]
    fn test_get_pack_some_and_none() {
        let mut packer = default_packer();
        let ids = packer.pack(vec![small_block("cid-gp", 1)], 100);
        assert!(packer.get_pack(ids[0]).is_some());
        assert!(packer.get_pack(9999).is_none());
    }

    // 15. delete_pack true/false
    #[test]
    fn test_delete_pack_true_false() {
        let mut packer = default_packer();
        let ids = packer.pack(vec![small_block("cid-del", 1)], 100);
        let pack_id = ids[0];
        assert!(
            packer.delete_pack(pack_id),
            "should return true when pack exists"
        );
        assert!(
            !packer.delete_pack(pack_id),
            "should return false when already deleted"
        );
        assert!(packer.get_pack(pack_id).is_none());
    }

    // 16. stats total_packs correct
    #[test]
    fn test_stats_total_packs() {
        let config = PackerConfig {
            max_pack_size_bytes: 10_000,
            min_block_size_bytes: 65_536,
            max_entries_per_pack: 1024,
        };
        let mut packer = StorageBlockPacker::new(config);
        packer.pack(
            vec![("c1".to_string(), 6000u64), ("c2".to_string(), 6000u64)],
            100,
        );
        let stats = packer.stats();
        assert_eq!(stats.total_packs, 2);
    }

    // 17. stats total_entries correct
    #[test]
    fn test_stats_total_entries() {
        let mut packer = default_packer();
        packer.pack(
            vec![
                small_block("e1", 1),
                small_block("e2", 2),
                small_block("e3", 3),
            ],
            100,
        );
        let stats = packer.stats();
        assert_eq!(stats.total_entries, 3);
    }

    // 18. stats total_packed_bytes correct
    #[test]
    fn test_stats_total_packed_bytes() {
        let mut packer = default_packer();
        packer.pack(
            vec![
                ("b1".to_string(), 1000u64),
                ("b2".to_string(), 2000u64),
                ("b3".to_string(), 3000u64),
            ],
            100,
        );
        let stats = packer.stats();
        assert_eq!(stats.total_packed_bytes, 6000);
    }

    // 19. stats avg_pack_utilization computed
    #[test]
    fn test_stats_avg_pack_utilization() {
        let config = PackerConfig {
            max_pack_size_bytes: 10_000,
            min_block_size_bytes: 65_536,
            max_entries_per_pack: 1024,
        };
        let mut packer = StorageBlockPacker::new(config);
        // Force a single pack with 5000 bytes → utilization = 0.5
        packer.pack(vec![("u1".to_string(), 5000u64)], 100);
        let stats = packer.stats();
        let expected = 5000.0_f64 / 10_000.0_f64;
        assert!(
            (stats.avg_pack_utilization - expected).abs() < 1e-10,
            "expected {expected}, got {}",
            stats.avg_pack_utilization
        );
    }

    // 20. stats avg_pack_utilization is 0.0 when no packs
    #[test]
    fn test_stats_avg_utilization_empty() {
        let packer = default_packer();
        let stats = packer.stats();
        assert_eq!(stats.avg_pack_utilization, 0.0);
    }

    // 21. pack_id monotonically increasing
    #[test]
    fn test_pack_id_monotonically_increasing() {
        let config = PackerConfig {
            max_pack_size_bytes: 10_000,
            min_block_size_bytes: 65_536,
            max_entries_per_pack: 1024,
        };
        let mut packer = StorageBlockPacker::new(config);
        let ids = packer.pack(
            vec![
                ("m1".to_string(), 6000u64),
                ("m2".to_string(), 6000u64),
                ("m3".to_string(), 6000u64),
            ],
            100,
        );
        assert_eq!(ids.len(), 3);
        assert!(ids[0] < ids[1], "pack IDs must be monotonically increasing");
        assert!(ids[1] < ids[2], "pack IDs must be monotonically increasing");
    }

    // 22. fnv1a produces deterministic results
    #[test]
    fn test_fnv1a_deterministic() {
        let a = fnv1a(b"hello");
        let b = fnv1a(b"hello");
        assert_eq!(a, b);
        let c = fnv1a(b"world");
        assert_ne!(a, c);
    }

    // 23. Mixed eligible/ineligible blocks — only eligible packed
    #[test]
    fn test_pack_mixed_blocks() {
        let mut packer = default_packer();
        let blocks = vec![
            ("small".to_string(), 1000u64),   // eligible (< 65536)
            ("large".to_string(), 65_536u64), // ineligible (== min_block_size_bytes, not <)
            ("tiny".to_string(), 512u64),     // eligible
        ];
        let ids = packer.pack(blocks, 100);
        assert_eq!(ids.len(), 1);
        let pack = packer.get_pack(ids[0]).expect("pack should exist");
        assert_eq!(pack.entry_count(), 2);
        assert!(pack.contains("small"));
        assert!(pack.contains("tiny"));
        assert!(!pack.contains("large"));
    }

    // 24. Pack created_at_secs matches provided timestamp
    #[test]
    fn test_pack_created_at_secs() {
        let mut packer = default_packer();
        let ts = 999_999_u64;
        let ids = packer.pack(vec![small_block("time-cid", 1)], ts);
        let pack = packer.get_pack(ids[0]).expect("pack should exist");
        assert_eq!(pack.created_at_secs, ts);
    }

    // 25. entries in pack are sorted by offset ascending
    #[test]
    fn test_pack_entries_sorted_by_offset() {
        let mut packer = default_packer();
        let blocks = vec![
            ("first".to_string(), 100u64),
            ("second".to_string(), 200u64),
            ("third".to_string(), 300u64),
        ];
        let ids = packer.pack(blocks, 100);
        let pack = packer.get_pack(ids[0]).expect("pack should exist");
        let offsets: Vec<u64> = pack.entries.iter().map(|e| e.offset).collect();
        let mut sorted = offsets.clone();
        sorted.sort_unstable();
        assert_eq!(offsets, sorted, "entries should be sorted by offset");
    }
}
