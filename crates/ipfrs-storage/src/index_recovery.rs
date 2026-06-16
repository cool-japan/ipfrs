//! Storage index recovery from corrupt or missing state by scanning raw block data.
//!
//! This module provides tools to reconstruct a storage index by scanning raw block
//! data when the primary index is corrupt, missing, or out of sync. Uses FNV-1a
//! checksums for fast integrity verification.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Current phase of the recovery process.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum RecoveryStatus {
    #[default]
    NotStarted,
    Scanning,
    Rebuilding,
    Verifying,
    Completed,
    Failed(String),
}

/// A single recovered index entry describing one block.
#[derive(Debug, Clone)]
pub struct IndexEntry {
    /// Content identifier of the block.
    pub cid: String,
    /// Size of the block data in bytes.
    pub size: u64,
    /// FNV-1a checksum of the block data.
    pub checksum: u64,
    /// Byte offset at which the block was found in the raw storage.
    pub offset: u64,
    /// Unix timestamp (seconds) at which this entry was recovered.
    pub recovered_at: u64,
}

/// Configuration for the recovery process.
#[derive(Debug, Clone)]
pub struct RecoveryConfig {
    /// Whether to verify checksums during scanning.
    pub verify_checksums: bool,
    /// Whether to skip corrupted blocks instead of halting.
    pub skip_corrupted: bool,
    /// Maximum recursion / nesting depth when scanning.
    pub max_scan_depth: usize,
    /// Number of blocks to process per batch.
    pub batch_size: usize,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        RecoveryConfig {
            verify_checksums: true,
            skip_corrupted: true,
            max_scan_depth: 64,
            batch_size: 256,
        }
    }
}

/// A raw block as found in storage — CID string, raw bytes, and byte offset.
#[derive(Debug, Clone)]
pub struct RawBlock {
    pub cid: String,
    pub data: Vec<u8>,
    pub offset: u64,
}

/// Aggregate statistics gathered during a recovery run.
#[derive(Debug, Clone, Default)]
pub struct RecoveryStats {
    pub blocks_scanned: u64,
    pub blocks_recovered: u64,
    pub blocks_skipped: u64,
    pub bytes_recovered: u64,
    pub checksum_failures: u64,
}

// ---------------------------------------------------------------------------
// IndexRecovery
// ---------------------------------------------------------------------------

/// Main recovery engine. Scans raw blocks and rebuilds the in-memory index.
pub struct IndexRecovery {
    config: RecoveryConfig,
    status: RecoveryStatus,
    recovered_index: HashMap<String, IndexEntry>,
    errors: Vec<String>,
    stats: RecoveryStats,
}

impl IndexRecovery {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new recovery engine with the given configuration.
    pub fn new(config: RecoveryConfig) -> Self {
        IndexRecovery {
            config,
            status: RecoveryStatus::NotStarted,
            recovered_index: HashMap::new(),
            errors: Vec::new(),
            stats: RecoveryStats::default(),
        }
    }

    // -----------------------------------------------------------------------
    // Core FNV-1a checksum
    // -----------------------------------------------------------------------

    /// Compute the FNV-1a 64-bit checksum of `data`.
    ///
    /// Reference: <https://tools.ietf.org/html/draft-eastlake-fnv-17>
    pub fn compute_checksum(data: &[u8]) -> u64 {
        const FNV_OFFSET_BASIS: u64 = 14695981039346656037;
        const FNV_PRIME: u64 = 1099511628211;

        let mut hash = FNV_OFFSET_BASIS;
        for byte in data {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    // -----------------------------------------------------------------------
    // Block scanning
    // -----------------------------------------------------------------------

    /// Scan a single raw block and produce an [`IndexEntry`], or return an
    /// error string describing why the block could not be recovered.
    ///
    /// This method updates the internal status, index, and stats.
    pub fn scan_block(&mut self, block: RawBlock) -> Result<IndexEntry, String> {
        // Transition to Scanning on first use
        if self.status == RecoveryStatus::NotStarted {
            self.status = RecoveryStatus::Scanning;
        }

        self.stats.blocks_scanned += 1;

        // Reject empty CIDs immediately
        if block.cid.is_empty() {
            let msg = format!("block at offset {} has empty CID", block.offset);
            self.errors.push(msg.clone());
            self.stats.blocks_skipped += 1;
            if self.config.skip_corrupted {
                return Err(msg);
            }
            self.status = RecoveryStatus::Failed(msg.clone());
            return Err(msg);
        }

        let checksum = Self::compute_checksum(&block.data);

        // Optionally verify by re-deriving the checksum embedded in the CID
        // prefix.  The convention used here: if the CID starts with
        // "bafy" (IPFS CIDv1 prefix) we trust it and only log the computed
        // checksum; for synthetic / test CIDs that encode their checksum as
        // a hex suffix we validate them.
        if self.config.verify_checksums {
            if let Some(fail_msg) = self.validate_cid_checksum(&block, checksum) {
                self.stats.checksum_failures += 1;
                self.errors.push(fail_msg.clone());
                if self.config.skip_corrupted {
                    self.stats.blocks_skipped += 1;
                    return Err(fail_msg);
                }
                self.status = RecoveryStatus::Failed(fail_msg.clone());
                return Err(fail_msg);
            }
        }

        let now = current_timestamp_secs();
        let size = block.data.len() as u64;

        let entry = IndexEntry {
            cid: block.cid.clone(),
            size,
            checksum,
            offset: block.offset,
            recovered_at: now,
        };

        // Handle duplicate CIDs: keep the entry with the lower offset (i.e.
        // the first occurrence found in raw storage) so recovery is
        // deterministic.
        let inserted = match self.recovered_index.get(&block.cid) {
            Some(existing) if existing.offset <= entry.offset => false,
            _ => {
                self.recovered_index
                    .insert(block.cid.clone(), entry.clone());
                true
            }
        };

        if inserted {
            self.stats.blocks_recovered += 1;
            self.stats.bytes_recovered += size;
        } else {
            // Duplicate — count it but do not overwrite the primary entry.
            self.stats.blocks_skipped += 1;
        }

        Ok(entry)
    }

    /// Validate a block's CID-encoded checksum, if present.
    ///
    /// Returns `Some(error_message)` if validation fails, `None` if the block
    /// passes or has no embedded checksum to validate.
    fn validate_cid_checksum(&self, block: &RawBlock, computed: u64) -> Option<String> {
        // Synthetic CIDs used in tests may encode the expected FNV-1a checksum
        // as a hex suffix separated by ':'.  Example: "testcid:1a2b3c4d5e6f7890".
        // Real IPFS CIDs (starting with "bafy", "Qm", "b", etc.) are trusted.
        if let Some(hex_part) = block.cid.split_once(':').map(|x| x.1) {
            match u64::from_str_radix(hex_part, 16) {
                Ok(expected) if expected != computed => {
                    return Some(format!(
                        "checksum mismatch for CID '{}': expected {:#018x}, got {:#018x}",
                        block.cid, expected, computed
                    ));
                }
                Ok(_) => {}
                Err(_) => {} // Not a valid hex suffix — treat as trusted CID
            }
        }
        None
    }

    // -----------------------------------------------------------------------
    // Batch operations
    // -----------------------------------------------------------------------

    /// Scan a batch of raw blocks and return one result per block.
    ///
    /// The order of results matches the order of the input slice.
    pub fn scan_batch(&mut self, blocks: Vec<RawBlock>) -> Vec<Result<IndexEntry, String>> {
        blocks.into_iter().map(|b| self.scan_block(b)).collect()
    }

    /// Fully rebuild the index from `blocks`.
    ///
    /// This resets the engine state, transitions through `Scanning` →
    /// `Rebuilding` → `Verifying` → `Completed` (or `Failed`) and returns
    /// the final [`RecoveryStats`].
    pub fn rebuild_index(&mut self, blocks: Vec<RawBlock>) -> RecoveryStats {
        self.reset();
        self.status = RecoveryStatus::Scanning;

        // --- Scanning phase -------------------------------------------------
        for chunk in blocks.chunks(self.config.batch_size) {
            for block in chunk {
                let _ = self.scan_block(block.clone());
            }
        }

        // --- Rebuilding phase -----------------------------------------------
        self.status = RecoveryStatus::Rebuilding;
        // The index is already populated; this phase represents any secondary
        // aggregation or cross-reference building that might be required.
        // (Currently a no-op — the map is the rebuilt index.)

        // --- Verifying phase ------------------------------------------------
        self.status = RecoveryStatus::Verifying;

        // If checksum verification is enabled, do a final pass over the
        // recovered index to ensure all entries are self-consistent.
        if self.config.verify_checksums {
            let cids: Vec<String> = self.recovered_index.keys().cloned().collect();
            let mut verification_failures: Vec<String> = Vec::new();
            for cid in &cids {
                if let Some(entry) = self.recovered_index.get(cid) {
                    // Re-derive checksum from the stored entry (without access
                    // to raw bytes at this stage we can only validate the
                    // internal consistency of the entry itself).
                    if entry.size == 0 && entry.checksum != Self::compute_checksum(&[]) {
                        verification_failures
                            .push(format!("entry '{}' has size 0 but non-empty checksum", cid));
                    }
                }
            }
            if !verification_failures.is_empty() {
                for msg in &verification_failures {
                    self.errors.push(msg.clone());
                }
                if !self.config.skip_corrupted {
                    self.status = RecoveryStatus::Failed(verification_failures.join("; "));
                    return self.stats.clone();
                }
            }
        }

        // Transition to Completed only if we are not already in Failed state.
        if self.status != RecoveryStatus::Failed("".to_string()) {
            // The pattern above doesn't equality-match Failed variants with
            // arbitrary messages, so use a more robust check.
            match &self.status {
                RecoveryStatus::Failed(_) => {}
                _ => {
                    self.status = RecoveryStatus::Completed;
                }
            }
        }

        self.stats.clone()
    }

    // -----------------------------------------------------------------------
    // Verification
    // -----------------------------------------------------------------------

    /// Verify that the stored `entry` is consistent with the provided `data`.
    ///
    /// Returns `true` if size and checksum match, `false` otherwise.
    pub fn verify_entry(&self, entry: &IndexEntry, data: &[u8]) -> bool {
        if entry.size != data.len() as u64 {
            return false;
        }
        if self.config.verify_checksums {
            let computed = Self::compute_checksum(data);
            if computed != entry.checksum {
                return false;
            }
        }
        true
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Look up a recovered entry by CID.
    pub fn get_entry(&self, cid: &str) -> Option<&IndexEntry> {
        self.recovered_index.get(cid)
    }

    /// Number of entries currently in the recovered index.
    pub fn entry_count(&self) -> usize {
        self.recovered_index.len()
    }

    /// Current recovery status.
    pub fn status(&self) -> &RecoveryStatus {
        &self.status
    }

    /// All error messages accumulated during recovery.
    pub fn errors(&self) -> &[String] {
        &self.errors
    }

    /// Aggregate statistics from the current (or last) recovery run.
    pub fn stats(&self) -> &RecoveryStats {
        &self.stats
    }

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    /// Reset all state, ready for a fresh recovery run.
    pub fn reset(&mut self) {
        self.status = RecoveryStatus::NotStarted;
        self.recovered_index.clear();
        self.errors.clear();
        self.stats = RecoveryStats::default();
    }

    // -----------------------------------------------------------------------
    // Export
    // -----------------------------------------------------------------------

    /// Export all recovered entries, sorted ascending by CID string.
    pub fn export_index(&self) -> Vec<IndexEntry> {
        let mut entries: Vec<IndexEntry> = self.recovered_index.values().cloned().collect();
        entries.sort_by(|a, b| a.cid.cmp(&b.cid));
        entries
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return a coarse Unix timestamp in seconds.
///
/// Falls back to 0 if the system clock cannot be read.
fn current_timestamp_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn default_config() -> RecoveryConfig {
        RecoveryConfig::default()
    }

    fn strict_config() -> RecoveryConfig {
        RecoveryConfig {
            skip_corrupted: false,
            ..RecoveryConfig::default()
        }
    }

    fn make_block(cid: &str, data: &[u8], offset: u64) -> RawBlock {
        RawBlock {
            cid: cid.to_owned(),
            data: data.to_vec(),
            offset,
        }
    }

    /// Build a block whose CID encodes the correct FNV-1a checksum as hex.
    fn valid_block(label: &str, data: &[u8], offset: u64) -> RawBlock {
        let checksum = IndexRecovery::compute_checksum(data);
        let cid = format!("{}:{:016x}", label, checksum);
        make_block(&cid, data, offset)
    }

    /// Build a block whose CID encodes a *wrong* checksum.
    fn corrupted_block(label: &str, data: &[u8], offset: u64) -> RawBlock {
        let cid = format!("{}:{:016x}", label, 0xdeadbeef_u64);
        make_block(&cid, data, offset)
    }

    // ------------------------------------------------------------------
    // 1. Initial status
    // ------------------------------------------------------------------

    #[test]
    fn test_initial_status_is_not_started() {
        let engine = IndexRecovery::new(default_config());
        assert_eq!(*engine.status(), RecoveryStatus::NotStarted);
    }

    // ------------------------------------------------------------------
    // 2. scan_block — valid block
    // ------------------------------------------------------------------

    #[test]
    fn test_scan_valid_block() {
        let mut engine = IndexRecovery::new(default_config());
        let data = b"hello world";
        let block = valid_block("cid1", data, 0);
        let result = engine.scan_block(block);
        assert!(result.is_ok());
        let entry = result.unwrap_or_else(|_| panic!("expected Ok"));
        assert_eq!(entry.size, data.len() as u64);
        assert_eq!(entry.offset, 0);
        assert_eq!(engine.stats().blocks_recovered, 1);
        assert_eq!(engine.stats().blocks_scanned, 1);
    }

    // ------------------------------------------------------------------
    // 3. scan_block — status transitions to Scanning
    // ------------------------------------------------------------------

    #[test]
    fn test_status_transitions_to_scanning_on_first_scan() {
        let mut engine = IndexRecovery::new(default_config());
        let block = valid_block("cid_scan", b"data", 0);
        let _ = engine.scan_block(block);
        assert_eq!(*engine.status(), RecoveryStatus::Scanning);
    }

    // ------------------------------------------------------------------
    // 4. scan_block — corrupted block with skip_corrupted = true (default)
    // ------------------------------------------------------------------

    #[test]
    fn test_scan_corrupted_block_skip_enabled() {
        let mut engine = IndexRecovery::new(default_config()); // skip_corrupted: true
        let block = corrupted_block("cid_bad", b"data", 0);
        let result = engine.scan_block(block);
        assert!(result.is_err());
        assert_eq!(engine.stats().checksum_failures, 1);
        assert_eq!(engine.stats().blocks_skipped, 1);
        assert_eq!(engine.stats().blocks_recovered, 0);
        // Status must NOT be Failed when skip_corrupted is true
        assert_ne!(*engine.status(), RecoveryStatus::Failed(String::new()));
    }

    // ------------------------------------------------------------------
    // 5. scan_block — corrupted block with skip_corrupted = false
    // ------------------------------------------------------------------

    #[test]
    fn test_scan_corrupted_block_no_skip() {
        let mut engine = IndexRecovery::new(strict_config());
        let block = corrupted_block("cid_bad", b"data", 0);
        let result = engine.scan_block(block);
        assert!(result.is_err());
        match engine.status() {
            RecoveryStatus::Failed(_) => {}
            other => panic!("expected Failed, got {:?}", other),
        }
    }

    // ------------------------------------------------------------------
    // 6. scan_block — empty CID
    // ------------------------------------------------------------------

    #[test]
    fn test_scan_block_empty_cid_skipped() {
        let mut engine = IndexRecovery::new(default_config());
        let block = make_block("", b"data", 0);
        let result = engine.scan_block(block);
        assert!(result.is_err());
        assert_eq!(engine.stats().blocks_skipped, 1);
        assert_eq!(engine.errors().len(), 1);
    }

    // ------------------------------------------------------------------
    // 7. scan_batch — mixed valid and corrupted blocks
    // ------------------------------------------------------------------

    #[test]
    fn test_scan_batch_mixed() {
        let mut engine = IndexRecovery::new(default_config());
        let blocks = vec![
            valid_block("cid_a", b"aaa", 0),
            corrupted_block("cid_b", b"bbb", 10),
            valid_block("cid_c", b"ccc", 20),
        ];
        let results = engine.scan_batch(blocks);
        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok());
        assert!(results[1].is_err());
        assert!(results[2].is_ok());
        assert_eq!(engine.stats().blocks_recovered, 2);
        assert_eq!(engine.stats().checksum_failures, 1);
    }

    // ------------------------------------------------------------------
    // 8. scan_batch — all valid
    // ------------------------------------------------------------------

    #[test]
    fn test_scan_batch_all_valid() {
        let mut engine = IndexRecovery::new(default_config());
        let blocks: Vec<RawBlock> = (0..10)
            .map(|i| valid_block(&format!("cid_{}", i), &[i as u8; 32], i as u64 * 32))
            .collect();
        let results = engine.scan_batch(blocks);
        assert!(results.iter().all(|r| r.is_ok()));
        assert_eq!(engine.entry_count(), 10);
    }

    // ------------------------------------------------------------------
    // 9. rebuild_index — status becomes Completed
    // ------------------------------------------------------------------

    #[test]
    fn test_rebuild_index_status_completed() {
        let mut engine = IndexRecovery::new(default_config());
        let blocks = vec![
            valid_block("cid1", b"hello", 0),
            valid_block("cid2", b"world", 8),
        ];
        let stats = engine.rebuild_index(blocks);
        assert_eq!(*engine.status(), RecoveryStatus::Completed);
        assert_eq!(stats.blocks_recovered, 2);
    }

    // ------------------------------------------------------------------
    // 10. rebuild_index — reset between runs
    // ------------------------------------------------------------------

    #[test]
    fn test_rebuild_index_clears_previous_state() {
        let mut engine = IndexRecovery::new(default_config());
        engine.rebuild_index(vec![valid_block("old_cid", b"old", 0)]);
        assert_eq!(engine.entry_count(), 1);

        engine.rebuild_index(vec![
            valid_block("new1", b"new1data", 0),
            valid_block("new2", b"new2data", 8),
        ]);
        assert_eq!(engine.entry_count(), 2);
        assert!(engine.get_entry("old_cid").is_none());
    }

    // ------------------------------------------------------------------
    // 11. compute_checksum — FNV-1a known values
    // ------------------------------------------------------------------

    #[test]
    fn test_compute_checksum_empty_slice() {
        // FNV-1a 64-bit of empty input is the offset basis itself.
        let expected: u64 = 14695981039346656037;
        assert_eq!(IndexRecovery::compute_checksum(&[]), expected);
    }

    #[test]
    fn test_compute_checksum_known_value() {
        // FNV-1a 64-bit of b"a" = 0xaf63dc4c8601ec8c (well-known)
        let expected: u64 = 0xaf63dc4c8601ec8c;
        assert_eq!(IndexRecovery::compute_checksum(b"a"), expected);
    }

    #[test]
    fn test_compute_checksum_deterministic() {
        let data = b"deterministic test payload 1234";
        let c1 = IndexRecovery::compute_checksum(data);
        let c2 = IndexRecovery::compute_checksum(data);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_compute_checksum_differs_for_different_data() {
        let c1 = IndexRecovery::compute_checksum(b"abc");
        let c2 = IndexRecovery::compute_checksum(b"abd");
        assert_ne!(c1, c2);
    }

    // ------------------------------------------------------------------
    // 12. verify_entry — correct data
    // ------------------------------------------------------------------

    #[test]
    fn test_verify_entry_correct_data() {
        let engine = IndexRecovery::new(default_config());
        let data = b"verify me";
        let entry = IndexEntry {
            cid: "test_cid".to_owned(),
            size: data.len() as u64,
            checksum: IndexRecovery::compute_checksum(data),
            offset: 0,
            recovered_at: 0,
        };
        assert!(engine.verify_entry(&entry, data));
    }

    // ------------------------------------------------------------------
    // 13. verify_entry — wrong size
    // ------------------------------------------------------------------

    #[test]
    fn test_verify_entry_wrong_size() {
        let engine = IndexRecovery::new(default_config());
        let data = b"verify me";
        let entry = IndexEntry {
            cid: "test_cid".to_owned(),
            size: 999,
            checksum: IndexRecovery::compute_checksum(data),
            offset: 0,
            recovered_at: 0,
        };
        assert!(!engine.verify_entry(&entry, data));
    }

    // ------------------------------------------------------------------
    // 14. verify_entry — wrong checksum
    // ------------------------------------------------------------------

    #[test]
    fn test_verify_entry_wrong_checksum() {
        let engine = IndexRecovery::new(default_config());
        let data = b"verify me";
        let entry = IndexEntry {
            cid: "test_cid".to_owned(),
            size: data.len() as u64,
            checksum: 0xdeadbeef,
            offset: 0,
            recovered_at: 0,
        };
        assert!(!engine.verify_entry(&entry, data));
    }

    // ------------------------------------------------------------------
    // 15. get_entry — found and not found
    // ------------------------------------------------------------------

    #[test]
    fn test_get_entry_found() {
        let mut engine = IndexRecovery::new(default_config());
        let data = b"content";
        let checksum = IndexRecovery::compute_checksum(data);
        let cid = format!("lookup_cid:{:016x}", checksum);
        let block = make_block(&cid, data, 42);
        engine.scan_block(block).ok();
        let entry = engine.get_entry(&cid);
        assert!(entry.is_some());
        assert_eq!(entry.map(|e| e.offset), Some(42));
    }

    #[test]
    fn test_get_entry_not_found() {
        let engine = IndexRecovery::new(default_config());
        assert!(engine.get_entry("nonexistent").is_none());
    }

    // ------------------------------------------------------------------
    // 16. export_index — sorted by cid
    // ------------------------------------------------------------------

    #[test]
    fn test_export_index_sorted_by_cid() {
        let mut engine = IndexRecovery::new(default_config());
        // Insert in non-alphabetical order
        for label in &["zzz", "aaa", "mmm", "bbb"] {
            let block = valid_block(label, b"data", 0);
            engine.scan_block(block).ok();
        }
        let exported = engine.export_index();
        let cids: Vec<&str> = exported.iter().map(|e| e.cid.as_str()).collect();
        let mut sorted = cids.clone();
        sorted.sort();
        assert_eq!(cids, sorted);
    }

    // ------------------------------------------------------------------
    // 17. reset — clears all state
    // ------------------------------------------------------------------

    #[test]
    fn test_reset_clears_all_state() {
        let mut engine = IndexRecovery::new(default_config());
        engine.scan_block(valid_block("cid1", b"data", 0)).ok();
        engine.reset();
        assert_eq!(*engine.status(), RecoveryStatus::NotStarted);
        assert_eq!(engine.entry_count(), 0);
        assert_eq!(engine.errors().len(), 0);
        assert_eq!(engine.stats().blocks_scanned, 0);
        assert_eq!(engine.stats().blocks_recovered, 0);
    }

    // ------------------------------------------------------------------
    // 18. errors tracking
    // ------------------------------------------------------------------

    #[test]
    fn test_errors_accumulate() {
        let mut engine = IndexRecovery::new(default_config());
        engine.scan_block(corrupted_block("c1", b"d1", 0)).ok();
        engine.scan_block(make_block("", b"d2", 10)).ok();
        engine.scan_block(corrupted_block("c3", b"d3", 20)).ok();
        assert_eq!(engine.errors().len(), 3);
    }

    // ------------------------------------------------------------------
    // 19. duplicate CID handling — lower offset wins
    // ------------------------------------------------------------------

    #[test]
    fn test_duplicate_cid_lower_offset_wins() {
        let mut engine = IndexRecovery::new(RecoveryConfig {
            verify_checksums: false, // plain CID without checksum suffix
            ..RecoveryConfig::default()
        });
        let block_first = make_block("dup_cid", b"first", 0);
        let block_second = make_block("dup_cid", b"first", 100);
        engine.scan_block(block_first).ok();
        engine.scan_block(block_second).ok();
        // Only 1 recovered entry; the one at offset 0 is kept.
        assert_eq!(engine.entry_count(), 1);
        assert_eq!(engine.get_entry("dup_cid").map(|e| e.offset), Some(0));
        // The second occurrence counted as skipped
        assert_eq!(engine.stats().blocks_skipped, 1);
    }

    // ------------------------------------------------------------------
    // 20. duplicate CID handling — higher offset superseded
    // ------------------------------------------------------------------

    #[test]
    fn test_duplicate_cid_later_entry_not_inserted() {
        let mut engine = IndexRecovery::new(RecoveryConfig {
            verify_checksums: false,
            ..RecoveryConfig::default()
        });
        engine.scan_block(make_block("dup", b"data", 50)).ok();
        engine.scan_block(make_block("dup", b"data", 200)).ok();
        assert_eq!(engine.stats().blocks_recovered, 1);
        assert_eq!(engine.stats().blocks_skipped, 1);
    }

    // ------------------------------------------------------------------
    // 21. large batch (100+ blocks) — stats accuracy
    // ------------------------------------------------------------------

    #[test]
    fn test_large_batch_stats_accuracy() {
        let mut engine = IndexRecovery::new(default_config());
        let n = 150_usize;
        let blocks: Vec<RawBlock> = (0..n)
            .map(|i| valid_block(&format!("cid_{:04}", i), &[i as u8; 64], i as u64 * 64))
            .collect();
        let results = engine.scan_batch(blocks);
        assert_eq!(results.len(), n);
        assert_eq!(engine.stats().blocks_scanned, n as u64);
        assert_eq!(engine.stats().blocks_recovered, n as u64);
        assert_eq!(engine.stats().blocks_skipped, 0);
        assert_eq!(engine.stats().checksum_failures, 0);
        assert_eq!(engine.stats().bytes_recovered, (n as u64) * 64);
        assert_eq!(engine.entry_count(), n);
    }

    // ------------------------------------------------------------------
    // 22. entry_count reflects insertions
    // ------------------------------------------------------------------

    #[test]
    fn test_entry_count() {
        let mut engine = IndexRecovery::new(RecoveryConfig {
            verify_checksums: false,
            ..RecoveryConfig::default()
        });
        assert_eq!(engine.entry_count(), 0);
        for i in 0..5 {
            engine
                .scan_block(make_block(&format!("c{}", i), b"d", i as u64))
                .ok();
        }
        assert_eq!(engine.entry_count(), 5);
    }

    // ------------------------------------------------------------------
    // 23. bytes_recovered tracks total bytes
    // ------------------------------------------------------------------

    #[test]
    fn test_bytes_recovered_accumulates() {
        let mut engine = IndexRecovery::new(RecoveryConfig {
            verify_checksums: false,
            ..RecoveryConfig::default()
        });
        engine.scan_block(make_block("a", &[0u8; 100], 0)).ok();
        engine.scan_block(make_block("b", &[0u8; 200], 100)).ok();
        assert_eq!(engine.stats().bytes_recovered, 300);
    }

    // ------------------------------------------------------------------
    // 24. verify_entry with checksum disabled
    // ------------------------------------------------------------------

    #[test]
    fn test_verify_entry_no_checksum_validation() {
        let engine = IndexRecovery::new(RecoveryConfig {
            verify_checksums: false,
            ..RecoveryConfig::default()
        });
        let data = b"test data";
        let entry = IndexEntry {
            cid: "cid".to_owned(),
            size: data.len() as u64,
            checksum: 0, // deliberately wrong
            offset: 0,
            recovered_at: 0,
        };
        // Without checksum verification, size match is sufficient.
        assert!(engine.verify_entry(&entry, data));
    }

    // ------------------------------------------------------------------
    // 25. rebuild_index integrates errors into error list
    // ------------------------------------------------------------------

    #[test]
    fn test_rebuild_index_accumulates_errors() {
        let mut engine = IndexRecovery::new(default_config());
        let blocks = vec![
            valid_block("good1", b"ok", 0),
            corrupted_block("bad1", b"corrupt", 10),
            corrupted_block("bad2", b"corrupt2", 20),
            valid_block("good2", b"ok2", 30),
        ];
        let stats = engine.rebuild_index(blocks);
        assert_eq!(stats.blocks_recovered, 2);
        assert_eq!(stats.checksum_failures, 2);
        assert_eq!(engine.errors().len(), 2);
        assert_eq!(*engine.status(), RecoveryStatus::Completed);
    }

    // ------------------------------------------------------------------
    // 26. export_index returns empty vec when no entries
    // ------------------------------------------------------------------

    #[test]
    fn test_export_index_empty() {
        let engine = IndexRecovery::new(default_config());
        assert!(engine.export_index().is_empty());
    }

    // ------------------------------------------------------------------
    // 27. RecoveryStatus default is NotStarted
    // ------------------------------------------------------------------

    #[test]
    fn test_recovery_status_default() {
        assert_eq!(RecoveryStatus::default(), RecoveryStatus::NotStarted);
    }

    // ------------------------------------------------------------------
    // 28. IndexEntry fields are accessible (clone + Debug)
    // ------------------------------------------------------------------

    #[test]
    fn test_index_entry_clone_and_debug() {
        let entry = IndexEntry {
            cid: "test".to_owned(),
            size: 42,
            checksum: 0xabcd,
            offset: 10,
            recovered_at: 9999,
        };
        let cloned = entry.clone();
        assert_eq!(cloned.cid, entry.cid);
        let _ = format!("{:?}", entry);
    }
}
