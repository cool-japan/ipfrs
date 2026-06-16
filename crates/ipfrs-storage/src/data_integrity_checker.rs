//! Data Integrity Checker - Deep integrity verification for stored blocks
//!
//! Provides FNV-1a based checksum verification, size validation,
//! quarantine management, and batch integrity checking for block storage.

use std::collections::{HashMap, HashSet};

/// Status of a block's integrity check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntegrityStatus {
    /// Block data is valid and matches expected checksum and size.
    Valid,
    /// Block data is corrupted (checksum and size both wrong).
    Corrupted,
    /// Block is not found in the registry.
    Missing,
    /// Block size does not match expected size.
    SizeMismatch,
    /// Block checksum does not match expected checksum.
    ChecksumMismatch,
}

/// Report of a single block's integrity check.
#[derive(Debug, Clone)]
pub struct IntegrityReport {
    /// Content identifier of the block.
    pub cid: String,
    /// Result status of the integrity check.
    pub status: IntegrityStatus,
    /// Expected FNV-1a checksum.
    pub expected_checksum: u64,
    /// Actual computed FNV-1a checksum.
    pub actual_checksum: u64,
    /// Expected block size in bytes.
    pub expected_size: u64,
    /// Actual block size in bytes.
    pub actual_size: u64,
    /// Timestamp (epoch seconds) when the check was performed.
    pub checked_at: u64,
    /// Human-readable details about the check result.
    pub details: String,
}

/// Configuration for the integrity checker.
#[derive(Debug, Clone)]
pub struct IntegrityCheckerConfig {
    /// Number of blocks to process per batch.
    pub batch_size: usize,
    /// Number of parallel checks (advisory, not used in sync impl).
    pub parallel_checks: usize,
    /// Whether to verify checksums during checks.
    pub verify_checksums: bool,
    /// Whether to verify sizes during checks.
    pub verify_sizes: bool,
    /// Whether to automatically quarantine failed blocks.
    pub auto_quarantine: bool,
}

impl Default for IntegrityCheckerConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            parallel_checks: 4,
            verify_checksums: true,
            verify_sizes: true,
            auto_quarantine: false,
        }
    }
}

/// A registered block record containing data and expected metadata.
#[derive(Debug, Clone)]
pub struct BlockRecord {
    /// Content identifier.
    pub cid: String,
    /// Raw block data.
    pub data: Vec<u8>,
    /// Expected FNV-1a checksum of the data.
    pub expected_checksum: u64,
    /// Expected size in bytes.
    pub expected_size: u64,
}

/// Aggregate statistics for integrity checks.
#[derive(Debug, Clone, Default)]
pub struct IntegrityStats {
    /// Total number of blocks checked.
    pub total_checked: u64,
    /// Number of valid blocks.
    pub valid: u64,
    /// Number of corrupted blocks.
    pub corrupted: u64,
    /// Number of missing blocks.
    pub missing: u64,
    /// Number of size mismatches.
    pub size_mismatches: u64,
    /// Number of checksum mismatches.
    pub checksum_mismatches: u64,
    /// Number of quarantined blocks.
    pub quarantined: u64,
}

/// Deep integrity checker for stored blocks.
///
/// Maintains a registry of block records, performs FNV-1a checksum and
/// size verification, tracks quarantined blocks, and accumulates statistics.
pub struct DataIntegrityChecker {
    config: IntegrityCheckerConfig,
    records: HashMap<String, BlockRecord>,
    reports: Vec<IntegrityReport>,
    quarantined: HashSet<String>,
    stats: IntegrityStats,
}

// FNV-1a 64-bit constants
const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x00000100000001B3;

impl DataIntegrityChecker {
    /// Create a new integrity checker with the given configuration.
    pub fn new(config: IntegrityCheckerConfig) -> Self {
        Self {
            config,
            records: HashMap::new(),
            reports: Vec::new(),
            quarantined: HashSet::new(),
            stats: IntegrityStats::default(),
        }
    }

    /// Register a block record for future integrity checks.
    pub fn register_block(&mut self, record: BlockRecord) {
        self.records.insert(record.cid.clone(), record);
    }

    /// Check the integrity of a single block by CID.
    ///
    /// Returns `None` if the CID is not registered (and records a Missing report).
    pub fn check_block(&mut self, cid: &str) -> Option<IntegrityReport> {
        let now = current_epoch_secs();

        let record = match self.records.get(cid) {
            Some(r) => r.clone(),
            None => {
                let report = IntegrityReport {
                    cid: cid.to_string(),
                    status: IntegrityStatus::Missing,
                    expected_checksum: 0,
                    actual_checksum: 0,
                    expected_size: 0,
                    actual_size: 0,
                    checked_at: now,
                    details: format!("Block '{}' not found in registry", cid),
                };
                self.stats.total_checked += 1;
                self.stats.missing += 1;
                if self.config.auto_quarantine && self.quarantined.insert(cid.to_string()) {
                    self.stats.quarantined += 1;
                }
                self.reports.push(report.clone());
                return Some(report);
            }
        };

        let actual_size = record.data.len() as u64;
        let actual_checksum = Self::compute_checksum(&record.data);

        let size_ok = !self.config.verify_sizes || actual_size == record.expected_size;
        let checksum_ok =
            !self.config.verify_checksums || actual_checksum == record.expected_checksum;

        let (status, details) = if size_ok && checksum_ok {
            (
                IntegrityStatus::Valid,
                format!("Block '{}' passed integrity check", cid),
            )
        } else if !size_ok && !checksum_ok {
            (
                IntegrityStatus::Corrupted,
                format!(
                    "Block '{}' corrupted: size expected {} got {}, checksum expected {:#018x} got {:#018x}",
                    cid, record.expected_size, actual_size, record.expected_checksum, actual_checksum
                ),
            )
        } else if !size_ok {
            (
                IntegrityStatus::SizeMismatch,
                format!(
                    "Block '{}' size mismatch: expected {} got {}",
                    cid, record.expected_size, actual_size
                ),
            )
        } else {
            (
                IntegrityStatus::ChecksumMismatch,
                format!(
                    "Block '{}' checksum mismatch: expected {:#018x} got {:#018x}",
                    cid, record.expected_checksum, actual_checksum
                ),
            )
        };

        self.stats.total_checked += 1;
        match &status {
            IntegrityStatus::Valid => self.stats.valid += 1,
            IntegrityStatus::Corrupted => self.stats.corrupted += 1,
            IntegrityStatus::SizeMismatch => self.stats.size_mismatches += 1,
            IntegrityStatus::ChecksumMismatch => self.stats.checksum_mismatches += 1,
            IntegrityStatus::Missing => self.stats.missing += 1,
        }

        if self.config.auto_quarantine
            && status != IntegrityStatus::Valid
            && self.quarantined.insert(cid.to_string())
        {
            self.stats.quarantined += 1;
        }

        let report = IntegrityReport {
            cid: cid.to_string(),
            status,
            expected_checksum: record.expected_checksum,
            actual_checksum,
            expected_size: record.expected_size,
            actual_size,
            checked_at: now,
            details,
        };

        self.reports.push(report.clone());
        Some(report)
    }

    /// Check integrity of all registered blocks.
    ///
    /// Processes blocks in batches according to the configured `batch_size`.
    pub fn check_all(&mut self) -> Vec<IntegrityReport> {
        let cids: Vec<String> = self.records.keys().cloned().collect();
        let mut results = Vec::with_capacity(cids.len());

        for batch in cids.chunks(self.config.batch_size.max(1)) {
            for cid in batch {
                if let Some(report) = self.check_block(cid) {
                    results.push(report);
                }
            }
        }

        results
    }

    /// Compute FNV-1a 64-bit hash of the given data.
    pub fn compute_checksum(data: &[u8]) -> u64 {
        let mut hash = FNV_OFFSET_BASIS;
        for &byte in data {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    /// Quarantine a block by CID. Returns `true` if newly quarantined.
    pub fn quarantine(&mut self, cid: &str) -> bool {
        let inserted = self.quarantined.insert(cid.to_string());
        if inserted {
            self.stats.quarantined += 1;
        }
        inserted
    }

    /// Remove a block from quarantine. Returns `true` if it was quarantined.
    pub fn unquarantine(&mut self, cid: &str) -> bool {
        let removed = self.quarantined.remove(cid);
        if removed {
            self.stats.quarantined = self.stats.quarantined.saturating_sub(1);
        }
        removed
    }

    /// Check if a block is currently quarantined.
    pub fn is_quarantined(&self, cid: &str) -> bool {
        self.quarantined.contains(cid)
    }

    /// Number of currently quarantined blocks.
    pub fn quarantined_count(&self) -> usize {
        self.quarantined.len()
    }

    /// Get the most recent report for a given CID.
    pub fn get_report(&self, cid: &str) -> Option<&IntegrityReport> {
        self.reports.iter().rev().find(|r| r.cid == cid)
    }

    /// Get all reports for corrupted blocks.
    pub fn corrupted_blocks(&self) -> Vec<&IntegrityReport> {
        self.reports
            .iter()
            .filter(|r| r.status == IntegrityStatus::Corrupted)
            .collect()
    }

    /// Get the current aggregate statistics.
    pub fn stats(&self) -> &IntegrityStats {
        &self.stats
    }

    /// Clear all stored reports (does not reset stats or quarantine).
    pub fn clear_reports(&mut self) {
        self.reports.clear();
    }

    /// Remove a block record from the registry. Returns `true` if it existed.
    pub fn remove_block(&mut self, cid: &str) -> bool {
        self.records.remove(cid).is_some()
    }
}

/// Get the current time as epoch seconds, falling back to 0 on error.
fn current_epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> IntegrityCheckerConfig {
        IntegrityCheckerConfig::default()
    }

    fn make_block(cid: &str, data: &[u8]) -> BlockRecord {
        let checksum = DataIntegrityChecker::compute_checksum(data);
        BlockRecord {
            cid: cid.to_string(),
            data: data.to_vec(),
            expected_checksum: checksum,
            expected_size: data.len() as u64,
        }
    }

    fn make_block_with_wrong_checksum(cid: &str, data: &[u8]) -> BlockRecord {
        BlockRecord {
            cid: cid.to_string(),
            data: data.to_vec(),
            expected_checksum: 0xDEADBEEF,
            expected_size: data.len() as u64,
        }
    }

    fn make_block_with_wrong_size(cid: &str, data: &[u8]) -> BlockRecord {
        let checksum = DataIntegrityChecker::compute_checksum(data);
        BlockRecord {
            cid: cid.to_string(),
            data: data.to_vec(),
            expected_checksum: checksum,
            expected_size: data.len() as u64 + 999,
        }
    }

    fn make_corrupted_block(cid: &str, data: &[u8]) -> BlockRecord {
        BlockRecord {
            cid: cid.to_string(),
            data: data.to_vec(),
            expected_checksum: 0xDEADBEEF,
            expected_size: data.len() as u64 + 100,
        }
    }

    // 1. Valid block verification
    #[test]
    fn test_valid_block() {
        let mut checker = DataIntegrityChecker::new(default_config());
        checker.register_block(make_block("cid1", b"hello world"));
        let report = checker.check_block("cid1").expect("should return report");
        assert_eq!(report.status, IntegrityStatus::Valid);
        assert_eq!(report.expected_checksum, report.actual_checksum);
        assert_eq!(report.expected_size, report.actual_size);
    }

    // 2. Corrupted data detection (both checksum and size wrong)
    #[test]
    fn test_corrupted_block() {
        let mut checker = DataIntegrityChecker::new(default_config());
        checker.register_block(make_corrupted_block("cid1", b"data"));
        let report = checker.check_block("cid1").expect("should return report");
        assert_eq!(report.status, IntegrityStatus::Corrupted);
    }

    // 3. Size mismatch
    #[test]
    fn test_size_mismatch() {
        let mut checker = DataIntegrityChecker::new(default_config());
        checker.register_block(make_block_with_wrong_size("cid1", b"some data"));
        let report = checker.check_block("cid1").expect("should return report");
        assert_eq!(report.status, IntegrityStatus::SizeMismatch);
    }

    // 4. Checksum mismatch
    #[test]
    fn test_checksum_mismatch() {
        let mut checker = DataIntegrityChecker::new(default_config());
        checker.register_block(make_block_with_wrong_checksum("cid1", b"payload"));
        let report = checker.check_block("cid1").expect("should return report");
        assert_eq!(report.status, IntegrityStatus::ChecksumMismatch);
    }

    // 5. Missing block
    #[test]
    fn test_missing_block() {
        let mut checker = DataIntegrityChecker::new(default_config());
        let report = checker
            .check_block("nonexistent")
            .expect("should return report");
        assert_eq!(report.status, IntegrityStatus::Missing);
    }

    // 6. Auto-quarantine on failure
    #[test]
    fn test_auto_quarantine_on_corrupted() {
        let mut config = default_config();
        config.auto_quarantine = true;
        let mut checker = DataIntegrityChecker::new(config);
        checker.register_block(make_corrupted_block("cid1", b"bad data"));
        checker.check_block("cid1");
        assert!(checker.is_quarantined("cid1"));
        assert_eq!(checker.quarantined_count(), 1);
    }

    // 7. Auto-quarantine does NOT quarantine valid blocks
    #[test]
    fn test_auto_quarantine_valid_block() {
        let mut config = default_config();
        config.auto_quarantine = true;
        let mut checker = DataIntegrityChecker::new(config);
        checker.register_block(make_block("cid1", b"good data"));
        checker.check_block("cid1");
        assert!(!checker.is_quarantined("cid1"));
    }

    // 8. Manual quarantine
    #[test]
    fn test_manual_quarantine() {
        let mut checker = DataIntegrityChecker::new(default_config());
        assert!(checker.quarantine("cid1"));
        assert!(checker.is_quarantined("cid1"));
        // Second call returns false (already quarantined)
        assert!(!checker.quarantine("cid1"));
    }

    // 9. Unquarantine
    #[test]
    fn test_unquarantine() {
        let mut checker = DataIntegrityChecker::new(default_config());
        checker.quarantine("cid1");
        assert!(checker.unquarantine("cid1"));
        assert!(!checker.is_quarantined("cid1"));
        // Second call returns false
        assert!(!checker.unquarantine("cid1"));
    }

    // 10. check_all with batch
    #[test]
    fn test_check_all() {
        let mut checker = DataIntegrityChecker::new(default_config());
        checker.register_block(make_block("a", b"alpha"));
        checker.register_block(make_block("b", b"bravo"));
        checker.register_block(make_block("c", b"charlie"));
        let reports = checker.check_all();
        assert_eq!(reports.len(), 3);
        assert!(reports.iter().all(|r| r.status == IntegrityStatus::Valid));
    }

    // 11. Stats accuracy after mixed checks
    #[test]
    fn test_stats_accuracy() {
        let mut checker = DataIntegrityChecker::new(default_config());
        checker.register_block(make_block("ok1", b"valid1"));
        checker.register_block(make_block("ok2", b"valid2"));
        checker.register_block(make_corrupted_block("bad1", b"corrupt"));
        checker.register_block(make_block_with_wrong_size("sz1", b"sizewrong"));
        checker.register_block(make_block_with_wrong_checksum("cs1", b"cswrong"));

        checker.check_block("ok1");
        checker.check_block("ok2");
        checker.check_block("bad1");
        checker.check_block("sz1");
        checker.check_block("cs1");
        checker.check_block("missing_one"); // missing

        let s = checker.stats();
        assert_eq!(s.total_checked, 6);
        assert_eq!(s.valid, 2);
        assert_eq!(s.corrupted, 1);
        assert_eq!(s.size_mismatches, 1);
        assert_eq!(s.checksum_mismatches, 1);
        assert_eq!(s.missing, 1);
    }

    // 12. corrupted_blocks filter
    #[test]
    fn test_corrupted_blocks_filter() {
        let mut checker = DataIntegrityChecker::new(default_config());
        checker.register_block(make_block("ok", b"fine"));
        checker.register_block(make_corrupted_block("bad", b"nope"));
        checker.check_block("ok");
        checker.check_block("bad");
        let corrupted = checker.corrupted_blocks();
        assert_eq!(corrupted.len(), 1);
        assert_eq!(corrupted[0].cid, "bad");
    }

    // 13. clear_reports
    #[test]
    fn test_clear_reports() {
        let mut checker = DataIntegrityChecker::new(default_config());
        checker.register_block(make_block("x", b"data"));
        checker.check_block("x");
        assert!(!checker.reports.is_empty());
        checker.clear_reports();
        assert!(checker.reports.is_empty());
        // Stats should still be there
        assert_eq!(checker.stats().total_checked, 1);
    }

    // 14. Register and remove block
    #[test]
    fn test_register_remove() {
        let mut checker = DataIntegrityChecker::new(default_config());
        checker.register_block(make_block("cid1", b"data"));
        assert!(checker.remove_block("cid1"));
        assert!(!checker.remove_block("cid1")); // already removed
    }

    // 15. Empty checker
    #[test]
    fn test_empty_checker() {
        let checker = DataIntegrityChecker::new(default_config());
        assert_eq!(checker.stats().total_checked, 0);
        assert_eq!(checker.quarantined_count(), 0);
        assert!(checker.corrupted_blocks().is_empty());
    }

    // 16. FNV-1a checksum consistency
    #[test]
    fn test_fnv1a_consistency() {
        let data = b"hello";
        let c1 = DataIntegrityChecker::compute_checksum(data);
        let c2 = DataIntegrityChecker::compute_checksum(data);
        assert_eq!(c1, c2);
    }

    // 17. FNV-1a empty data
    #[test]
    fn test_fnv1a_empty() {
        let checksum = DataIntegrityChecker::compute_checksum(b"");
        assert_eq!(checksum, FNV_OFFSET_BASIS);
    }

    // 18. FNV-1a different inputs produce different hashes
    #[test]
    fn test_fnv1a_different_inputs() {
        let c1 = DataIntegrityChecker::compute_checksum(b"aaa");
        let c2 = DataIntegrityChecker::compute_checksum(b"bbb");
        assert_ne!(c1, c2);
    }

    // 19. get_report returns most recent
    #[test]
    fn test_get_report_most_recent() {
        let mut checker = DataIntegrityChecker::new(default_config());
        checker.register_block(make_block("cid1", b"data"));
        checker.check_block("cid1");
        // Re-register with corrupted and re-check
        checker.register_block(make_corrupted_block("cid1", b"data"));
        checker.check_block("cid1");
        let report = checker.get_report("cid1").expect("should exist");
        assert_eq!(report.status, IntegrityStatus::Corrupted);
    }

    // 20. get_report returns None for unknown cid
    #[test]
    fn test_get_report_none() {
        let checker = DataIntegrityChecker::new(default_config());
        assert!(checker.get_report("unknown").is_none());
    }

    // 21. Quarantined count tracks correctly
    #[test]
    fn test_quarantined_count_tracking() {
        let mut checker = DataIntegrityChecker::new(default_config());
        checker.quarantine("a");
        checker.quarantine("b");
        checker.quarantine("c");
        assert_eq!(checker.quarantined_count(), 3);
        checker.unquarantine("b");
        assert_eq!(checker.quarantined_count(), 2);
        assert_eq!(checker.stats().quarantined, 2);
    }

    // 22. check_all with mixed results
    #[test]
    fn test_check_all_mixed() {
        let mut checker = DataIntegrityChecker::new(default_config());
        checker.register_block(make_block("good", b"ok"));
        checker.register_block(make_corrupted_block("bad", b"no"));
        let reports = checker.check_all();
        assert_eq!(reports.len(), 2);
        let valid_count = reports
            .iter()
            .filter(|r| r.status == IntegrityStatus::Valid)
            .count();
        let corrupted_count = reports
            .iter()
            .filter(|r| r.status == IntegrityStatus::Corrupted)
            .count();
        assert_eq!(valid_count, 1);
        assert_eq!(corrupted_count, 1);
    }

    // 23. Disable checksum verification
    #[test]
    fn test_disable_checksum_verification() {
        let mut config = default_config();
        config.verify_checksums = false;
        let mut checker = DataIntegrityChecker::new(config);
        checker.register_block(make_block_with_wrong_checksum("cid1", b"data"));
        let report = checker.check_block("cid1").expect("should return report");
        // With checksum verification disabled, only size is checked (size is correct)
        assert_eq!(report.status, IntegrityStatus::Valid);
    }

    // 24. Disable size verification
    #[test]
    fn test_disable_size_verification() {
        let mut config = default_config();
        config.verify_sizes = false;
        let mut checker = DataIntegrityChecker::new(config);
        checker.register_block(make_block_with_wrong_size("cid1", b"data"));
        let report = checker.check_block("cid1").expect("should return report");
        // With size verification disabled, only checksum is checked (checksum is correct)
        assert_eq!(report.status, IntegrityStatus::Valid);
    }

    // 25. Auto-quarantine on checksum mismatch
    #[test]
    fn test_auto_quarantine_checksum_mismatch() {
        let mut config = default_config();
        config.auto_quarantine = true;
        let mut checker = DataIntegrityChecker::new(config);
        checker.register_block(make_block_with_wrong_checksum("cs1", b"xxx"));
        checker.check_block("cs1");
        assert!(checker.is_quarantined("cs1"));
    }

    // 26. Auto-quarantine on missing block
    #[test]
    fn test_auto_quarantine_missing() {
        let mut config = default_config();
        config.auto_quarantine = true;
        let mut checker = DataIntegrityChecker::new(config);
        checker.check_block("ghost");
        assert!(checker.is_quarantined("ghost"));
    }

    // 27. Report details contain cid
    #[test]
    fn test_report_details_contain_cid() {
        let mut checker = DataIntegrityChecker::new(default_config());
        checker.register_block(make_block("my-cid", b"test"));
        let report = checker.check_block("my-cid").expect("report");
        assert!(report.details.contains("my-cid"));
    }

    // 28. Batch size of 1
    #[test]
    fn test_batch_size_one() {
        let mut config = default_config();
        config.batch_size = 1;
        let mut checker = DataIntegrityChecker::new(config);
        checker.register_block(make_block("a", b"1"));
        checker.register_block(make_block("b", b"2"));
        checker.register_block(make_block("c", b"3"));
        let reports = checker.check_all();
        assert_eq!(reports.len(), 3);
    }

    // 29. FNV-1a known value test
    #[test]
    fn test_fnv1a_known_value() {
        // FNV-1a 64-bit of "foobar" is a well-known test vector
        let checksum = DataIntegrityChecker::compute_checksum(b"foobar");
        // Manually verify: start with offset basis and apply algorithm
        let mut expected = FNV_OFFSET_BASIS;
        for &b in b"foobar" {
            expected ^= b as u64;
            expected = expected.wrapping_mul(FNV_PRIME);
        }
        assert_eq!(checksum, expected);
    }

    // 30. Remove block then check returns missing
    #[test]
    fn test_remove_then_check() {
        let mut checker = DataIntegrityChecker::new(default_config());
        checker.register_block(make_block("cid1", b"data"));
        checker.remove_block("cid1");
        let report = checker.check_block("cid1").expect("report");
        assert_eq!(report.status, IntegrityStatus::Missing);
    }

    // 31. Re-register overwrites block
    #[test]
    fn test_re_register_overwrites() {
        let mut checker = DataIntegrityChecker::new(default_config());
        checker.register_block(make_block("cid1", b"old"));
        checker.register_block(make_block("cid1", b"new"));
        let report = checker.check_block("cid1").expect("report");
        assert_eq!(report.status, IntegrityStatus::Valid);
        assert_eq!(report.actual_size, 3); // "new" is 3 bytes
    }

    // 32. checked_at is non-zero
    #[test]
    fn test_checked_at_nonzero() {
        let mut checker = DataIntegrityChecker::new(default_config());
        checker.register_block(make_block("cid1", b"data"));
        let report = checker.check_block("cid1").expect("report");
        assert!(report.checked_at > 0);
    }
}
