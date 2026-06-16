//! Storage block integrity verifier using FNV-1a checksums.
//!
//! Tracks verification results and produces detailed reports for
//! auditing and self-healing workflows.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// FNV-1a checksum (64-bit)
// ---------------------------------------------------------------------------

#[inline]
fn fnv1a_checksum(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// ---------------------------------------------------------------------------
// VerificationResult
// ---------------------------------------------------------------------------

/// The outcome of verifying a single storage block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerificationResult {
    /// Checksum matches — block is intact.
    Ok,
    /// Checksum mismatch — block content has changed since registration.
    Corrupted {
        /// The FNV-1a checksum recorded at write time.
        expected: u64,
        /// The FNV-1a checksum computed from the current content.
        actual: u64,
    },
    /// Block was not found in the registry, or content was absent.
    Missing,
}

// ---------------------------------------------------------------------------
// BlockRecord
// ---------------------------------------------------------------------------

/// Metadata and last verification state for a single registered block.
#[derive(Clone, Debug)]
pub struct BlockRecord {
    /// Unique block identifier assigned by the verifier.
    pub block_id: u64,
    /// Content Identifier (CID) string for this block.
    pub cid: String,
    /// FNV-1a checksum of the block content at registration time.
    pub expected_checksum: u64,
    /// Size of the block content in bytes.
    pub size_bytes: u64,
    /// Logical tick at which the last verification was performed, if any.
    pub verified_at_tick: Option<u64>,
    /// Result of the most recent verification, if any.
    pub last_result: Option<VerificationResult>,
}

// ---------------------------------------------------------------------------
// VerificationReport
// ---------------------------------------------------------------------------

/// Summary of a batch verification run.
#[derive(Clone, Debug)]
pub struct VerificationReport {
    /// Total number of blocks checked in this batch.
    pub total_checked: usize,
    /// Number of blocks whose checksums matched.
    pub ok_count: usize,
    /// Number of blocks with checksum mismatches.
    pub corrupted_count: usize,
    /// Number of missing blocks (not registered or content absent).
    pub missing_count: usize,
    /// Block IDs of corrupted blocks, sorted ascending.
    pub corrupted_ids: Vec<u64>,
    /// Block IDs of missing blocks, sorted ascending.
    pub missing_ids: Vec<u64>,
}

impl VerificationReport {
    /// Returns `true` when no corruption or missing blocks were found.
    #[inline]
    pub fn is_clean(&self) -> bool {
        self.corrupted_count == 0 && self.missing_count == 0
    }
}

// ---------------------------------------------------------------------------
// VerifierStats
// ---------------------------------------------------------------------------

/// Cumulative statistics across all verification operations.
#[derive(Clone, Debug, Default)]
pub struct VerifierStats {
    /// Total number of distinct blocks currently registered.
    pub total_blocks_registered: usize,
    /// Total number of individual block verifications performed.
    pub total_verifications_run: u64,
    /// Total verifications that returned `Ok`.
    pub total_ok: u64,
    /// Total verifications that returned `Corrupted`.
    pub total_corrupted: u64,
    /// Total verifications that returned `Missing`.
    pub total_missing: u64,
}

// ---------------------------------------------------------------------------
// StorageBlockVerifier
// ---------------------------------------------------------------------------

/// Verifies storage block integrity using FNV-1a checksums.
///
/// Tracks per-block verification history and accumulates lifetime statistics
/// suitable for auditing and self-healing pipelines.
pub struct StorageBlockVerifier {
    records: HashMap<u64, BlockRecord>,
    next_block_id: u64,
    stats: VerifierStats,
}

impl StorageBlockVerifier {
    /// Creates a new, empty verifier.
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
            next_block_id: 0,
            stats: VerifierStats::default(),
        }
    }

    /// Registers a block, computing its FNV-1a checksum from `content`.
    ///
    /// Returns the newly assigned `block_id`.
    pub fn register(&mut self, cid: String, content: &[u8]) -> u64 {
        let block_id = self.next_block_id;
        self.next_block_id += 1;

        let expected_checksum = fnv1a_checksum(content);
        let record = BlockRecord {
            block_id,
            cid,
            expected_checksum,
            size_bytes: content.len() as u64,
            verified_at_tick: None,
            last_result: None,
        };

        self.records.insert(block_id, record);
        self.stats.total_blocks_registered += 1;

        block_id
    }

    /// Verifies a single block against its registered checksum.
    ///
    /// - If `block_id` is unknown → `Missing` (stats updated, **no** record mutation).
    /// - If `content` is `None` → `Missing` (record updated).
    /// - Otherwise computes the FNV-1a checksum and returns `Ok` or `Corrupted`.
    ///
    /// Always increments `stats.total_verifications_run`.
    pub fn verify_block(
        &mut self,
        block_id: u64,
        content: Option<&[u8]>,
        current_tick: u64,
    ) -> VerificationResult {
        // Block not registered at all.
        if !self.records.contains_key(&block_id) {
            self.stats.total_missing += 1;
            self.stats.total_verifications_run += 1;
            return VerificationResult::Missing;
        }

        self.stats.total_verifications_run += 1;

        let content_bytes = match content {
            None => {
                self.stats.total_missing += 1;
                let record = self
                    .records
                    .get_mut(&block_id)
                    .expect("key existence checked above");
                record.verified_at_tick = Some(current_tick);
                record.last_result = Some(VerificationResult::Missing);
                return VerificationResult::Missing;
            }
            Some(bytes) => bytes,
        };

        let record = self
            .records
            .get_mut(&block_id)
            .expect("key existence checked above");

        let actual = fnv1a_checksum(content_bytes);
        let result = if actual == record.expected_checksum {
            self.stats.total_ok += 1;
            VerificationResult::Ok
        } else {
            self.stats.total_corrupted += 1;
            VerificationResult::Corrupted {
                expected: record.expected_checksum,
                actual,
            }
        };

        record.verified_at_tick = Some(current_tick);
        record.last_result = Some(result.clone());

        result
    }

    /// Verifies a batch of `(block_id, content)` pairs and returns a summary report.
    ///
    /// `corrupted_ids` and `missing_ids` in the returned report are sorted ascending.
    pub fn verify_batch(
        &mut self,
        batch: Vec<(u64, Option<Vec<u8>>)>,
        current_tick: u64,
    ) -> VerificationReport {
        let mut ok_count = 0usize;
        let mut corrupted_count = 0usize;
        let mut missing_count = 0usize;
        let mut corrupted_ids: Vec<u64> = Vec::new();
        let mut missing_ids: Vec<u64> = Vec::new();

        let total_checked = batch.len();

        for (block_id, content) in batch {
            let result = self.verify_block(block_id, content.as_deref(), current_tick);
            match result {
                VerificationResult::Ok => ok_count += 1,
                VerificationResult::Corrupted { .. } => {
                    corrupted_count += 1;
                    corrupted_ids.push(block_id);
                }
                VerificationResult::Missing => {
                    missing_count += 1;
                    missing_ids.push(block_id);
                }
            }
        }

        corrupted_ids.sort_unstable();
        missing_ids.sort_unstable();

        VerificationReport {
            total_checked,
            ok_count,
            corrupted_count,
            missing_count,
            corrupted_ids,
            missing_ids,
        }
    }

    /// Returns a reference to the record for `block_id`, if it exists.
    pub fn get_record(&self, block_id: u64) -> Option<&BlockRecord> {
        self.records.get(&block_id)
    }

    /// Returns a reference to the cumulative statistics.
    pub fn stats(&self) -> &VerifierStats {
        &self.stats
    }

    /// Returns block IDs that have never been verified, sorted ascending.
    pub fn unverified_blocks(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self
            .records
            .values()
            .filter(|r| r.verified_at_tick.is_none())
            .map(|r| r.block_id)
            .collect();
        ids.sort_unstable();
        ids
    }

    /// Returns block IDs whose most recent verification was `Corrupted`, sorted ascending.
    pub fn corrupted_blocks(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self
            .records
            .values()
            .filter(|r| matches!(&r.last_result, Some(VerificationResult::Corrupted { .. })))
            .map(|r| r.block_id)
            .collect();
        ids.sort_unstable();
        ids
    }
}

impl Default for StorageBlockVerifier {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper
    // -----------------------------------------------------------------------

    fn make_verifier() -> StorageBlockVerifier {
        StorageBlockVerifier::new()
    }

    fn expected_checksum(data: &[u8]) -> u64 {
        fnv1a_checksum(data)
    }

    // -----------------------------------------------------------------------
    // register
    // -----------------------------------------------------------------------

    #[test]
    fn test_register_computes_correct_checksum() {
        let mut v = make_verifier();
        let content = b"hello world";
        let id = v.register("cid-1".into(), content);
        let record = v.get_record(id).expect("record must exist");
        assert_eq!(record.expected_checksum, expected_checksum(content));
    }

    #[test]
    fn test_register_stores_size() {
        let mut v = make_verifier();
        let content = b"abcde";
        let id = v.register("cid-size".into(), content);
        let record = v.get_record(id).expect("record must exist");
        assert_eq!(record.size_bytes, 5);
    }

    #[test]
    fn test_register_stores_cid() {
        let mut v = make_verifier();
        let id = v.register("my-cid".into(), b"data");
        let record = v.get_record(id).expect("record must exist");
        assert_eq!(record.cid, "my-cid");
    }

    #[test]
    fn test_register_increments_block_id() {
        let mut v = make_verifier();
        let id0 = v.register("cid-0".into(), b"a");
        let id1 = v.register("cid-1".into(), b"b");
        let id2 = v.register("cid-2".into(), b"c");
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }

    #[test]
    fn test_register_increments_stats() {
        let mut v = make_verifier();
        v.register("a".into(), b"x");
        v.register("b".into(), b"y");
        assert_eq!(v.stats().total_blocks_registered, 2);
    }

    #[test]
    fn test_register_no_verified_at_tick() {
        let mut v = make_verifier();
        let id = v.register("cid".into(), b"data");
        let record = v.get_record(id).expect("record must exist");
        assert!(record.verified_at_tick.is_none());
        assert!(record.last_result.is_none());
    }

    // -----------------------------------------------------------------------
    // verify_block — Ok
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_block_ok_when_content_matches() {
        let mut v = make_verifier();
        let content = b"consistent data";
        let id = v.register("cid".into(), content);
        let result = v.verify_block(id, Some(content), 1);
        assert_eq!(result, VerificationResult::Ok);
    }

    #[test]
    fn test_verify_block_ok_updates_record() {
        let mut v = make_verifier();
        let content = b"data";
        let id = v.register("cid".into(), content);
        v.verify_block(id, Some(content), 42);
        let record = v.get_record(id).expect("record must exist");
        assert_eq!(record.verified_at_tick, Some(42));
        assert_eq!(record.last_result, Some(VerificationResult::Ok));
    }

    // -----------------------------------------------------------------------
    // verify_block — Corrupted
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_block_corrupted_when_content_changed() {
        let mut v = make_verifier();
        let id = v.register("cid".into(), b"original");
        let result = v.verify_block(id, Some(b"modified"), 1);
        assert!(matches!(result, VerificationResult::Corrupted { .. }));
    }

    #[test]
    fn test_verify_block_corrupted_expected_and_actual() {
        let mut v = make_verifier();
        let original = b"original";
        let modified = b"modified!!";
        let id = v.register("cid".into(), original);
        let expected_cs = expected_checksum(original);
        let actual_cs = expected_checksum(modified);
        let result = v.verify_block(id, Some(modified), 1);
        assert_eq!(
            result,
            VerificationResult::Corrupted {
                expected: expected_cs,
                actual: actual_cs,
            }
        );
    }

    #[test]
    fn test_verify_block_corrupted_updates_record() {
        let mut v = make_verifier();
        let id = v.register("cid".into(), b"a");
        v.verify_block(id, Some(b"b"), 7);
        let record = v.get_record(id).expect("record must exist");
        assert_eq!(record.verified_at_tick, Some(7));
        assert!(matches!(
            record.last_result,
            Some(VerificationResult::Corrupted { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // verify_block — Missing
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_block_missing_when_not_registered() {
        let mut v = make_verifier();
        let result = v.verify_block(999, Some(b"some data"), 1);
        assert_eq!(result, VerificationResult::Missing);
    }

    #[test]
    fn test_verify_block_missing_when_content_none() {
        let mut v = make_verifier();
        let id = v.register("cid".into(), b"data");
        let result = v.verify_block(id, None, 5);
        assert_eq!(result, VerificationResult::Missing);
    }

    #[test]
    fn test_verify_block_none_updates_record() {
        let mut v = make_verifier();
        let id = v.register("cid".into(), b"data");
        v.verify_block(id, None, 10);
        let record = v.get_record(id).expect("record must exist");
        assert_eq!(record.verified_at_tick, Some(10));
        assert_eq!(record.last_result, Some(VerificationResult::Missing));
    }

    // -----------------------------------------------------------------------
    // stats accumulation
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_total_verifications_run() {
        let mut v = make_verifier();
        let id = v.register("cid".into(), b"data");
        v.verify_block(id, Some(b"data"), 1);
        v.verify_block(id, Some(b"wrong"), 2);
        v.verify_block(id, None, 3);
        v.verify_block(9999, Some(b"x"), 4); // missing id
        assert_eq!(v.stats().total_verifications_run, 4);
    }

    #[test]
    fn test_stats_total_ok() {
        let mut v = make_verifier();
        let id = v.register("cid".into(), b"data");
        v.verify_block(id, Some(b"data"), 1);
        v.verify_block(id, Some(b"data"), 2);
        assert_eq!(v.stats().total_ok, 2);
    }

    #[test]
    fn test_stats_total_corrupted() {
        let mut v = make_verifier();
        let id = v.register("cid".into(), b"data");
        v.verify_block(id, Some(b"bad"), 1);
        v.verify_block(id, Some(b"worse"), 2);
        assert_eq!(v.stats().total_corrupted, 2);
    }

    #[test]
    fn test_stats_total_missing() {
        let mut v = make_verifier();
        let id = v.register("cid".into(), b"data");
        v.verify_block(id, None, 1); // missing (None)
        v.verify_block(9999, Some(b"x"), 2); // missing (unknown id)
        assert_eq!(v.stats().total_missing, 2);
    }

    #[test]
    fn test_stats_accumulate_across_calls() {
        let mut v = make_verifier();
        let id = v.register("cid".into(), b"hello");
        v.verify_block(id, Some(b"hello"), 1); // ok
        v.verify_block(id, Some(b"world"), 2); // corrupted
        v.verify_block(id, None, 3); // missing
        let s = v.stats();
        assert_eq!(s.total_ok, 1);
        assert_eq!(s.total_corrupted, 1);
        assert_eq!(s.total_missing, 1);
        assert_eq!(s.total_verifications_run, 3);
    }

    // -----------------------------------------------------------------------
    // verify_batch
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_batch_report_counts() {
        let mut v = make_verifier();
        let id_ok = v.register("ok".into(), b"good");
        let id_bad = v.register("bad".into(), b"original");
        let id_reg = v.register("none".into(), b"data");

        let batch = vec![
            (id_ok, Some(b"good".to_vec())),
            (id_bad, Some(b"corrupted".to_vec())),
            (id_reg, None),
            (9999, Some(b"x".to_vec())), // unregistered → missing
        ];

        let report = v.verify_batch(batch, 1);
        assert_eq!(report.total_checked, 4);
        assert_eq!(report.ok_count, 1);
        assert_eq!(report.corrupted_count, 1);
        assert_eq!(report.missing_count, 2);
    }

    #[test]
    fn test_verify_batch_sorted_ids() {
        let mut v = make_verifier();
        // Register three blocks; ids will be 0, 1, 2.
        let id0 = v.register("c0".into(), b"a");
        let id1 = v.register("c1".into(), b"b");
        let id2 = v.register("c2".into(), b"c");

        // Submit in reverse order with corrupted/missing content.
        let batch = vec![
            (id2, Some(b"z".to_vec())), // corrupted
            (id0, Some(b"z".to_vec())), // corrupted
            (id1, None),                // missing
        ];
        let report = v.verify_batch(batch, 1);
        assert_eq!(report.corrupted_ids, vec![id0, id2]);
        assert_eq!(report.missing_ids, vec![id1]);
    }

    // -----------------------------------------------------------------------
    // is_clean
    // -----------------------------------------------------------------------

    #[test]
    fn test_report_is_clean_true() {
        let mut v = make_verifier();
        let id = v.register("cid".into(), b"data");
        let report = v.verify_batch(vec![(id, Some(b"data".to_vec()))], 1);
        assert!(report.is_clean());
    }

    #[test]
    fn test_report_is_clean_false_when_corrupted() {
        let mut v = make_verifier();
        let id = v.register("cid".into(), b"data");
        let report = v.verify_batch(vec![(id, Some(b"wrong".to_vec()))], 1);
        assert!(!report.is_clean());
    }

    #[test]
    fn test_report_is_clean_false_when_missing() {
        let mut v = make_verifier();
        let id = v.register("cid".into(), b"data");
        let report = v.verify_batch(vec![(id, None)], 1);
        assert!(!report.is_clean());
    }

    // -----------------------------------------------------------------------
    // unverified_blocks
    // -----------------------------------------------------------------------

    #[test]
    fn test_unverified_blocks_returns_all_initially() {
        let mut v = make_verifier();
        v.register("a".into(), b"1");
        v.register("b".into(), b"2");
        v.register("c".into(), b"3");
        let unverified = v.unverified_blocks();
        assert_eq!(unverified, vec![0, 1, 2]);
    }

    #[test]
    fn test_unverified_blocks_excludes_verified() {
        let mut v = make_verifier();
        let id0 = v.register("a".into(), b"1");
        v.register("b".into(), b"2");
        v.verify_block(id0, Some(b"1"), 1);
        let unverified = v.unverified_blocks();
        assert_eq!(unverified, vec![1]);
    }

    #[test]
    fn test_unverified_blocks_empty_when_all_verified() {
        let mut v = make_verifier();
        let id0 = v.register("a".into(), b"x");
        let id1 = v.register("b".into(), b"y");
        v.verify_block(id0, Some(b"x"), 1);
        v.verify_block(id1, Some(b"y"), 1);
        assert!(v.unverified_blocks().is_empty());
    }

    // -----------------------------------------------------------------------
    // corrupted_blocks
    // -----------------------------------------------------------------------

    #[test]
    fn test_corrupted_blocks_returns_only_corrupted() {
        let mut v = make_verifier();
        let id0 = v.register("a".into(), b"original");
        let id1 = v.register("b".into(), b"good");
        let id2 = v.register("c".into(), b"also original");

        v.verify_block(id0, Some(b"changed"), 1); // corrupted
        v.verify_block(id1, Some(b"good"), 1); // ok
        v.verify_block(id2, Some(b"also changed"), 1); // corrupted

        let corrupted = v.corrupted_blocks();
        assert_eq!(corrupted, vec![id0, id2]);
    }

    #[test]
    fn test_corrupted_blocks_empty_initially() {
        let mut v = make_verifier();
        v.register("a".into(), b"x");
        assert!(v.corrupted_blocks().is_empty());
    }

    #[test]
    fn test_corrupted_blocks_not_including_missing() {
        let mut v = make_verifier();
        let id = v.register("a".into(), b"data");
        v.verify_block(id, None, 1); // missing, not corrupted
        assert!(v.corrupted_blocks().is_empty());
    }

    // -----------------------------------------------------------------------
    // batch with mixed results
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_mixed_results_all_categories() {
        let mut v = make_verifier();
        let id_ok = v.register("ok".into(), b"abc");
        let id_bad = v.register("bad".into(), b"xyz");
        let id_none = v.register("none".into(), b"zzz");

        let batch = vec![
            (id_ok, Some(b"abc".to_vec())),
            (id_bad, Some(b"XYZ-TAMPERED".to_vec())),
            (id_none, None),
            (42, Some(b"phantom".to_vec())), // unknown block id
        ];

        let report = v.verify_batch(batch, 100);
        assert_eq!(report.total_checked, 4);
        assert_eq!(report.ok_count, 1);
        assert_eq!(report.corrupted_count, 1);
        assert_eq!(report.missing_count, 2);
        assert!(!report.is_clean());

        // Stats should reflect totals
        let s = v.stats();
        assert_eq!(s.total_ok, 1);
        assert_eq!(s.total_corrupted, 1);
        assert_eq!(s.total_missing, 2);
        assert_eq!(s.total_verifications_run, 4);
    }

    #[test]
    fn test_empty_batch_produces_clean_report() {
        let mut v = make_verifier();
        let report = v.verify_batch(vec![], 1);
        assert_eq!(report.total_checked, 0);
        assert!(report.is_clean());
    }

    #[test]
    fn test_fnv1a_deterministic() {
        let data = b"determinism test";
        let a = fnv1a_checksum(data);
        let b = fnv1a_checksum(data);
        assert_eq!(a, b);
    }

    #[test]
    fn test_fnv1a_different_inputs_differ() {
        let a = fnv1a_checksum(b"alpha");
        let b = fnv1a_checksum(b"beta");
        assert_ne!(a, b);
    }

    #[test]
    fn test_get_record_returns_none_for_unknown() {
        let v = make_verifier();
        assert!(v.get_record(0).is_none());
    }
}
