//! Block Integrity Scanner
//!
//! Scans stored blocks for integrity issues including CID mismatches,
//! size violations, corruption markers (magic byte mismatches), and
//! missing blocks.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// FNV-1a hash (64-bit, no external dependency)
// ---------------------------------------------------------------------------

const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME: u64 = 1_099_511_628_211;

fn fnv1a(data: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ---------------------------------------------------------------------------
// IntegrityIssue
// ---------------------------------------------------------------------------

/// Describes a single integrity problem found during a block scan.
#[derive(Clone, Debug, PartialEq)]
pub enum IntegrityIssue {
    /// The CID stored in the registry does not match the CID computed from
    /// the data presented at scan time.
    CidMismatch {
        /// CID that was registered (the "expected" CID).
        cid: String,
        /// CID computed from the actual data presented at scan time.
        /// In production callers replace this with the real recomputed CID;
        /// the scanner itself uses the registered CID as a placeholder so
        /// that the type signature is stable.
        computed_cid: String,
    },
    /// The size of the actual data does not match the expected size stored
    /// in the registry, or exceeds the configured `max_size_bytes`.
    SizeViolation {
        /// CID of the block that triggered the violation.
        cid: String,
        /// Expected size in bytes (from registry or `max_size_bytes`).
        expected: u64,
        /// Actual size in bytes.
        actual: u64,
    },
    /// The actual data does not begin with the configured magic bytes,
    /// indicating likely corruption.
    CorruptionMarker {
        /// CID of the affected block.
        cid: String,
        /// Byte offset where the mismatch was detected (always 0 here,
        /// since the prefix is checked from the start of the data).
        offset: usize,
    },
    /// No registry entry was found for the requested CID.
    MissingBlock {
        /// The CID that was not found in the registry.
        cid: String,
    },
}

// ---------------------------------------------------------------------------
// ScanRecord
// ---------------------------------------------------------------------------

/// Metadata stored in the scanner's registry for a single known block.
#[derive(Clone, Debug, PartialEq)]
pub struct ScanRecord {
    /// Content identifier for this block.
    pub cid: String,
    /// Expected size in bytes as reported when the block was registered.
    pub expected_size: u64,
    /// FNV-1a hash of the raw data bytes at registration time.
    pub data_hash: u64,
    /// Unix timestamp (seconds) at which the block was registered.
    pub registered_at_secs: u64,
}

// ---------------------------------------------------------------------------
// ScanResult
// ---------------------------------------------------------------------------

/// The outcome of scanning a single block.
#[derive(Clone, Debug, PartialEq)]
pub struct ScanResult {
    /// CID of the scanned block.
    pub cid: String,
    /// All integrity issues detected during the scan.  Empty means healthy.
    pub issues: Vec<IntegrityIssue>,
    /// Wall-clock time spent on this scan, in microseconds.
    pub scan_duration_us: u64,
}

impl ScanResult {
    /// Returns `true` when no integrity issues were found.
    pub fn is_healthy(&self) -> bool {
        self.issues.is_empty()
    }
}

// ---------------------------------------------------------------------------
// ScannerConfig
// ---------------------------------------------------------------------------

/// Tunable parameters for [`BlockIntegrityScanner`].
#[derive(Clone, Debug)]
pub struct ScannerConfig {
    /// Blocks larger than this value (in bytes) are flagged with a
    /// [`IntegrityIssue::SizeViolation`].
    /// Default: `256 * 1024 * 1024` (256 MiB).
    pub max_size_bytes: u64,
    /// Expected prefix bytes for every block.  When non-empty, blocks whose
    /// data does not start with these bytes are flagged with a
    /// [`IntegrityIssue::CorruptionMarker`].
    /// Default: empty (check skipped).
    pub magic_bytes: Vec<u8>,
    /// When `true`, the FNV-1a hash of the presented data is compared with
    /// the hash stored at registration time; a difference causes a
    /// [`IntegrityIssue::CidMismatch`].
    /// Default: `true`.
    pub verify_cid_hash: bool,
}

impl Default for ScannerConfig {
    fn default() -> Self {
        Self {
            max_size_bytes: 256 * 1024 * 1024,
            magic_bytes: Vec::new(),
            verify_cid_hash: true,
        }
    }
}

// ---------------------------------------------------------------------------
// ScannerStats
// ---------------------------------------------------------------------------

/// Aggregate statistics accumulated across all scan operations.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScannerStats {
    /// Total number of blocks scanned (including missing-block probes).
    pub total_scanned: u64,
    /// Number of blocks that were healthy (no issues).
    pub healthy: u64,
    /// Number of blocks that had at least one issue.
    pub with_issues: u64,
}

impl ScannerStats {
    /// Fraction of scanned blocks that had at least one issue.
    ///
    /// Returns `0.0` when no blocks have been scanned yet.
    pub fn issue_rate(&self) -> f64 {
        if self.total_scanned == 0 {
            0.0
        } else {
            self.with_issues as f64 / self.total_scanned as f64
        }
    }
}

// ---------------------------------------------------------------------------
// BlockIntegrityScanner
// ---------------------------------------------------------------------------

/// Scans stored blocks for integrity issues.
///
/// Callers first [`register_block`](BlockIntegrityScanner::register_block)
/// each known block, then call
/// [`scan_block`](BlockIntegrityScanner::scan_block) (or
/// [`scan_all`](BlockIntegrityScanner::scan_all)) whenever they want to
/// verify the live data against the registry.
pub struct BlockIntegrityScanner {
    /// Registry of all known blocks, keyed by CID string.
    pub registry: HashMap<String, ScanRecord>,
    /// Configuration controlling which checks are performed.
    pub config: ScannerConfig,
    /// Running statistics.
    pub stats: ScannerStats,
}

impl BlockIntegrityScanner {
    /// Creates a new scanner with the supplied configuration and an empty
    /// registry.
    pub fn new(config: ScannerConfig) -> Self {
        Self {
            registry: HashMap::new(),
            config,
            stats: ScannerStats::default(),
        }
    }

    /// Registers a block in the scanner's registry.
    ///
    /// The FNV-1a hash of `data` is computed and stored alongside the
    /// supplied metadata so that future scans can detect hash mismatches.
    ///
    /// # Parameters
    /// - `cid` – Content identifier for the block.
    /// - `expected_size` – Expected size of the block in bytes.
    /// - `data` – Raw block data used to compute the stored hash.
    /// - `now_secs` – Current Unix timestamp in seconds (caller-supplied
    ///   so the scanner remains deterministic in tests).
    pub fn register_block(&mut self, cid: String, expected_size: u64, data: &[u8], now_secs: u64) {
        let data_hash = fnv1a(data);
        self.registry.insert(
            cid.clone(),
            ScanRecord {
                cid,
                expected_size,
                data_hash,
                registered_at_secs: now_secs,
            },
        );
    }

    /// Scans a single block and returns a [`ScanResult`] describing any
    /// integrity issues found.
    ///
    /// The following checks are performed in order:
    ///
    /// 1. **Missing block** – If `cid` is not in the registry the result
    ///    contains a single [`IntegrityIssue::MissingBlock`] and all other
    ///    checks are skipped.
    /// 2. **Size mismatch** – If `actual_data.len() as u64 != expected_size`
    ///    a [`IntegrityIssue::SizeViolation`] is added.
    /// 3. **Hash verification** – When `verify_cid_hash` is `true`, the
    ///    FNV-1a hash of `actual_data` is compared with the stored hash; a
    ///    difference adds a [`IntegrityIssue::CidMismatch`].
    /// 4. **Magic bytes** – When `magic_bytes` is non-empty and
    ///    `actual_data` does not start with those bytes, a
    ///    [`IntegrityIssue::CorruptionMarker`] is added at `offset = 0`.
    /// 5. **Max size** – If `actual_data.len() as u64 > max_size_bytes` a
    ///    [`IntegrityIssue::SizeViolation`] is added (using `max_size_bytes`
    ///    as `expected`).
    ///
    /// Stats are updated before returning.
    pub fn scan_block(&mut self, cid: &str, actual_data: &[u8], scan_time_us: u64) -> ScanResult {
        let record = match self.registry.get(cid) {
            Some(r) => r.clone(),
            None => {
                self.stats.total_scanned += 1;
                self.stats.with_issues += 1;
                return ScanResult {
                    cid: cid.to_string(),
                    issues: vec![IntegrityIssue::MissingBlock {
                        cid: cid.to_string(),
                    }],
                    scan_duration_us: scan_time_us,
                };
            }
        };

        let mut issues = Vec::new();
        let actual_len = actual_data.len() as u64;

        // Check 2: size mismatch against registered expected_size
        if actual_len != record.expected_size {
            issues.push(IntegrityIssue::SizeViolation {
                cid: cid.to_string(),
                expected: record.expected_size,
                actual: actual_len,
            });
        }

        // Check 3: hash verification (CidMismatch)
        if self.config.verify_cid_hash {
            let actual_hash = fnv1a(actual_data);
            if actual_hash != record.data_hash {
                // In production callers replace `computed_cid` with the real
                // recomputed CID; the scanner uses `cid` as a placeholder.
                issues.push(IntegrityIssue::CidMismatch {
                    cid: cid.to_string(),
                    computed_cid: cid.to_string(),
                });
            }
        }

        // Check 4: magic bytes prefix
        if !self.config.magic_bytes.is_empty() && !actual_data.starts_with(&self.config.magic_bytes)
        {
            issues.push(IntegrityIssue::CorruptionMarker {
                cid: cid.to_string(),
                offset: 0,
            });
        }

        // Check 5: max size
        if actual_len > self.config.max_size_bytes {
            issues.push(IntegrityIssue::SizeViolation {
                cid: cid.to_string(),
                expected: self.config.max_size_bytes,
                actual: actual_len,
            });
        }

        // Update stats
        self.stats.total_scanned += 1;
        if issues.is_empty() {
            self.stats.healthy += 1;
        } else {
            self.stats.with_issues += 1;
        }

        ScanResult {
            cid: cid.to_string(),
            issues,
            scan_duration_us: scan_time_us,
        }
    }

    /// Scans all supplied blocks and returns one [`ScanResult`] per block.
    ///
    /// Each element in `blocks` is a `(cid, data)` pair.  The same
    /// `scan_time_us` value is recorded for every block in the batch.
    pub fn scan_all(&mut self, blocks: &[(String, Vec<u8>)], scan_time_us: u64) -> Vec<ScanResult> {
        blocks
            .iter()
            .map(|(cid, data)| self.scan_block(cid, data, scan_time_us))
            .collect()
    }

    /// Returns references to all [`ScanRecord`]s in the registry.
    ///
    /// This does **not** filter by scan history; it simply exposes the full
    /// contents of the registry at the time of the call.
    pub fn healthy_blocks(&self) -> Vec<&ScanRecord> {
        self.registry.values().collect()
    }

    /// Returns a reference to the scanner's accumulated statistics.
    pub fn stats(&self) -> &ScannerStats {
        &self.stats
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_scanner() -> BlockIntegrityScanner {
        BlockIntegrityScanner::new(ScannerConfig::default())
    }

    /// Helper: build scanner, register a block, return (scanner, data).
    fn scanner_with_block(cid: &str, data: &[u8]) -> (BlockIntegrityScanner, Vec<u8>) {
        let mut s = default_scanner();
        s.register_block(cid.to_string(), data.len() as u64, data, 1_000);
        (s, data.to_vec())
    }

    // -----------------------------------------------------------------------
    // 1. Register + scan healthy block
    // -----------------------------------------------------------------------
    #[test]
    fn test_register_and_scan_healthy() {
        let data = b"hello ipfrs";
        let (mut s, d) = scanner_with_block("cid1", data);
        let result = s.scan_block("cid1", &d, 42);
        assert!(
            result.is_healthy(),
            "expected no issues, got {:?}",
            result.issues
        );
        assert_eq!(result.cid, "cid1");
        assert_eq!(result.scan_duration_us, 42);
    }

    // -----------------------------------------------------------------------
    // 2. is_healthy returns true for empty issues
    // -----------------------------------------------------------------------
    #[test]
    fn test_is_healthy_true() {
        let result = ScanResult {
            cid: "x".to_string(),
            issues: vec![],
            scan_duration_us: 0,
        };
        assert!(result.is_healthy());
    }

    // -----------------------------------------------------------------------
    // 3. is_healthy returns false when issues present
    // -----------------------------------------------------------------------
    #[test]
    fn test_is_healthy_false() {
        let result = ScanResult {
            cid: "x".to_string(),
            issues: vec![IntegrityIssue::MissingBlock {
                cid: "x".to_string(),
            }],
            scan_duration_us: 0,
        };
        assert!(!result.is_healthy());
    }

    // -----------------------------------------------------------------------
    // 4. Missing block
    // -----------------------------------------------------------------------
    #[test]
    fn test_missing_block() {
        let mut s = default_scanner();
        let result = s.scan_block("not-registered", b"anything", 10);
        assert!(!result.is_healthy());
        assert_eq!(result.issues.len(), 1);
        assert_eq!(
            result.issues[0],
            IntegrityIssue::MissingBlock {
                cid: "not-registered".to_string()
            }
        );
    }

    // -----------------------------------------------------------------------
    // 5. Size violation — actual != expected
    // -----------------------------------------------------------------------
    #[test]
    fn test_size_violation_wrong_length() {
        let data = b"original data";
        let (mut s, _) = scanner_with_block("cid-sz", data);
        // Present shorter data
        let result = s.scan_block("cid-sz", b"short", 0);
        let has_size_violation = result.issues.iter().any(|i| {
            matches!(i, IntegrityIssue::SizeViolation { expected, actual, .. }
                if *expected == data.len() as u64 && *actual == 5)
        });
        assert!(
            has_size_violation,
            "expected SizeViolation, got {:?}",
            result.issues
        );
    }

    // -----------------------------------------------------------------------
    // 6. Hash mismatch triggers CidMismatch
    // -----------------------------------------------------------------------
    #[test]
    fn test_hash_mismatch_triggers_cid_mismatch() {
        let original = b"block content v1";
        let (mut s, _) = scanner_with_block("cid-hash", original);
        // Same length but different content
        let tampered = b"block content v2";
        assert_eq!(original.len(), tampered.len());
        let result = s.scan_block("cid-hash", tampered, 0);
        let has_cid_mismatch = result
            .issues
            .iter()
            .any(|i| matches!(i, IntegrityIssue::CidMismatch { .. }));
        assert!(
            has_cid_mismatch,
            "expected CidMismatch, got {:?}",
            result.issues
        );
    }

    // -----------------------------------------------------------------------
    // 7. No CidMismatch when verify_cid_hash is false
    // -----------------------------------------------------------------------
    #[test]
    fn test_no_cid_mismatch_when_verify_disabled() {
        let config = ScannerConfig {
            verify_cid_hash: false,
            ..ScannerConfig::default()
        };
        let mut s = BlockIntegrityScanner::new(config);
        let original = b"block content v1";
        s.register_block("cid-nv".to_string(), original.len() as u64, original, 0);
        let tampered = b"block content v2";
        let result = s.scan_block("cid-nv", tampered, 0);
        let has_cid_mismatch = result
            .issues
            .iter()
            .any(|i| matches!(i, IntegrityIssue::CidMismatch { .. }));
        assert!(
            !has_cid_mismatch,
            "did not expect CidMismatch when verify disabled"
        );
    }

    // -----------------------------------------------------------------------
    // 8. Magic bytes check — mismatch triggers CorruptionMarker
    // -----------------------------------------------------------------------
    #[test]
    fn test_magic_bytes_mismatch_triggers_corruption_marker() {
        let config = ScannerConfig {
            magic_bytes: vec![0xDE, 0xAD, 0xBE, 0xEF],
            ..ScannerConfig::default()
        };
        let mut s = BlockIntegrityScanner::new(config);
        let data = b"\x00\x00\x00\x00extra";
        s.register_block("cid-magic".to_string(), data.len() as u64, data, 0);
        let result = s.scan_block("cid-magic", data, 0);
        let has_corruption = result
            .issues
            .iter()
            .any(|i| matches!(i, IntegrityIssue::CorruptionMarker { offset: 0, .. }));
        assert!(
            has_corruption,
            "expected CorruptionMarker, got {:?}",
            result.issues
        );
    }

    // -----------------------------------------------------------------------
    // 9. Magic bytes check — matching prefix is fine
    // -----------------------------------------------------------------------
    #[test]
    fn test_magic_bytes_match_no_corruption() {
        let config = ScannerConfig {
            magic_bytes: vec![0xDE, 0xAD, 0xBE, 0xEF],
            ..ScannerConfig::default()
        };
        let mut s = BlockIntegrityScanner::new(config);
        let data = b"\xDE\xAD\xBE\xEFrest";
        s.register_block("cid-ok-magic".to_string(), data.len() as u64, data, 0);
        let result = s.scan_block("cid-ok-magic", data, 0);
        let has_corruption = result
            .issues
            .iter()
            .any(|i| matches!(i, IntegrityIssue::CorruptionMarker { .. }));
        assert!(
            !has_corruption,
            "did not expect CorruptionMarker, got {:?}",
            result.issues
        );
    }

    // -----------------------------------------------------------------------
    // 10. Max size violation
    // -----------------------------------------------------------------------
    #[test]
    fn test_max_size_violation() {
        let config = ScannerConfig {
            max_size_bytes: 8,
            ..ScannerConfig::default()
        }; // very small for testing
        let mut s = BlockIntegrityScanner::new(config);
        // Data is 10 bytes — exceeds max_size_bytes=8
        let data = b"0123456789";
        s.register_block("cid-big".to_string(), data.len() as u64, data, 0);
        let result = s.scan_block("cid-big", data, 0);
        let has_max_violation = result.issues.iter().any(|i| {
            matches!(
                i,
                IntegrityIssue::SizeViolation {
                    expected: 8,
                    actual: 10,
                    ..
                }
            )
        });
        assert!(
            has_max_violation,
            "expected max-size SizeViolation, got {:?}",
            result.issues
        );
    }

    // -----------------------------------------------------------------------
    // 11. scan_all returns one result per block
    // -----------------------------------------------------------------------
    #[test]
    fn test_scan_all_returns_one_result_per_block() {
        let mut s = default_scanner();
        let data_a = b"block-a";
        let data_b = b"block-b";
        s.register_block("a".to_string(), data_a.len() as u64, data_a, 0);
        s.register_block("b".to_string(), data_b.len() as u64, data_b, 0);
        let blocks = vec![
            ("a".to_string(), data_a.to_vec()),
            ("b".to_string(), data_b.to_vec()),
        ];
        let results = s.scan_all(&blocks, 5);
        assert_eq!(results.len(), 2);
        assert!(results[0].is_healthy(), "block a should be healthy");
        assert!(results[1].is_healthy(), "block b should be healthy");
    }

    // -----------------------------------------------------------------------
    // 12. scan_all with a missing block
    // -----------------------------------------------------------------------
    #[test]
    fn test_scan_all_includes_missing_block() {
        let mut s = default_scanner();
        let blocks = vec![("ghost".to_string(), b"data".to_vec())];
        let results = s.scan_all(&blocks, 0);
        assert_eq!(results.len(), 1);
        assert!(!results[0].is_healthy());
        assert!(results[0]
            .issues
            .iter()
            .any(|i| matches!(i, IntegrityIssue::MissingBlock { .. })));
    }

    // -----------------------------------------------------------------------
    // 13. issue_rate when no scans
    // -----------------------------------------------------------------------
    #[test]
    fn test_issue_rate_no_scans() {
        let s = default_scanner();
        assert_eq!(s.stats().issue_rate(), 0.0);
    }

    // -----------------------------------------------------------------------
    // 14. issue_rate calculation
    // -----------------------------------------------------------------------
    #[test]
    fn test_issue_rate_calculation() {
        let mut s = default_scanner();
        let data = b"some data";
        s.register_block("ok".to_string(), data.len() as u64, data, 0);
        // healthy scan
        s.scan_block("ok", data, 0);
        // missing-block scan
        s.scan_block("bad", b"x", 0);

        let stats = s.stats();
        assert_eq!(stats.total_scanned, 2);
        assert_eq!(stats.healthy, 1);
        assert_eq!(stats.with_issues, 1);
        let rate = stats.issue_rate();
        assert!(
            (rate - 0.5).abs() < f64::EPSILON,
            "expected 0.5, got {rate}"
        );
    }

    // -----------------------------------------------------------------------
    // 15. healthy_blocks count
    // -----------------------------------------------------------------------
    #[test]
    fn test_healthy_blocks_count() {
        let mut s = default_scanner();
        s.register_block("c1".to_string(), 4, b"aaaa", 0);
        s.register_block("c2".to_string(), 4, b"bbbb", 0);
        s.register_block("c3".to_string(), 4, b"cccc", 0);
        // healthy_blocks returns all registry entries regardless of scans
        assert_eq!(s.healthy_blocks().len(), 3);
    }

    // -----------------------------------------------------------------------
    // 16. Stats update correctly across multiple scans
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_accumulate_correctly() {
        let mut s = default_scanner();
        let data = b"payload";
        for i in 0..5_u32 {
            s.register_block(format!("cid-{i}"), data.len() as u64, data, 0);
        }
        // Scan all five healthy
        for i in 0..5_u32 {
            s.scan_block(&format!("cid-{i}"), data, 0);
        }
        // Scan one missing
        s.scan_block("missing", b"x", 0);

        let st = s.stats();
        assert_eq!(st.total_scanned, 6);
        assert_eq!(st.healthy, 5);
        assert_eq!(st.with_issues, 1);
    }

    // -----------------------------------------------------------------------
    // 17. Multiple issues in a single scan
    // -----------------------------------------------------------------------
    #[test]
    fn test_multiple_issues_in_single_scan() {
        let config = ScannerConfig {
            magic_bytes: vec![0xFF],
            max_size_bytes: 3,
            ..ScannerConfig::default()
        };
        let mut s = BlockIntegrityScanner::new(config);

        // Register a 5-byte block that starts with 0xFF
        let original = b"\xFF\x00\x01\x02\x03";
        s.register_block("multi".to_string(), original.len() as u64, original, 0);

        // Present 5-byte block that does NOT start with 0xFF and has different hash
        let tampered = b"\x00\x00\x01\x02\x03";
        let result = s.scan_block("multi", tampered, 100);

        // Should see: CidMismatch, CorruptionMarker, SizeViolation(max)
        let has_cid = result
            .issues
            .iter()
            .any(|i| matches!(i, IntegrityIssue::CidMismatch { .. }));
        let has_cor = result
            .issues
            .iter()
            .any(|i| matches!(i, IntegrityIssue::CorruptionMarker { .. }));
        let has_sz = result
            .issues
            .iter()
            .any(|i| matches!(i, IntegrityIssue::SizeViolation { .. }));
        assert!(has_cid, "expected CidMismatch");
        assert!(has_cor, "expected CorruptionMarker");
        assert!(has_sz, "expected SizeViolation");
        assert_eq!(result.scan_duration_us, 100);
    }

    // -----------------------------------------------------------------------
    // 18. FNV-1a hash is deterministic
    // -----------------------------------------------------------------------
    #[test]
    fn test_fnv1a_deterministic() {
        let data = b"deterministic";
        assert_eq!(fnv1a(data), fnv1a(data));
    }

    // -----------------------------------------------------------------------
    // 19. Registering same CID twice overwrites the record
    // -----------------------------------------------------------------------
    #[test]
    fn test_register_overwrites_existing() {
        let mut s = default_scanner();
        s.register_block("dup".to_string(), 4, b"aaaa", 100);
        s.register_block("dup".to_string(), 8, b"bbbbbbbb", 200);
        let record = s.registry.get("dup").expect("record should exist");
        assert_eq!(record.expected_size, 8);
        assert_eq!(record.registered_at_secs, 200);
    }

    // -----------------------------------------------------------------------
    // 20. Empty magic_bytes skips corruption check
    // -----------------------------------------------------------------------
    #[test]
    fn test_empty_magic_bytes_skips_corruption_check() {
        let config = ScannerConfig {
            magic_bytes: vec![],
            ..ScannerConfig::default()
        };
        let mut s = BlockIntegrityScanner::new(config);
        let data = b"anything goes here";
        s.register_block("cid-em".to_string(), data.len() as u64, data, 0);
        let result = s.scan_block("cid-em", data, 0);
        let has_corruption = result
            .issues
            .iter()
            .any(|i| matches!(i, IntegrityIssue::CorruptionMarker { .. }));
        assert!(!has_corruption);
        assert!(result.is_healthy());
    }
}
