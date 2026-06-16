//! Checkpoint pruning and validation for gradient checkpoints.
//!
//! This module provides [`CheckpointPruner`] for retention-policy-based pruning
//! and [`CheckpointValidator`] for CRC-32 integrity verification.

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// CRC-32 (IEEE 802.3 polynomial, table-driven, pure Rust)
// ---------------------------------------------------------------------------

/// Lookup table for CRC-32 (IEEE 802.3 / Ethernet, reversed polynomial 0xEDB88320).
const CRC32_TABLE: [u32; 256] = build_crc32_table();

const fn build_crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320u32;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// Compute the CRC-32 (IEEE polynomial) of `data`.
///
/// # Example
/// ```
/// use ipfrs_tensorlogic::checkpoint_manager::crc32;
/// assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
/// ```
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        let idx = ((crc ^ u32::from(byte)) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32_TABLE[idx];
    }
    crc ^ 0xFFFF_FFFF
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur during checkpoint validation.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ValidationError {
    /// The computed CRC-32 does not match the stored value.
    #[error("CRC-32 mismatch: expected 0x{expected:08X}, actual 0x{actual:08X}")]
    CrcMismatch { expected: u32, actual: u32 },

    /// The data length does not match the stored size.
    #[error("size mismatch: expected {expected} bytes, actual {actual} bytes")]
    SizeMismatch { expected: u64, actual: u64 },
}

// ---------------------------------------------------------------------------
// CheckpointRecord
// ---------------------------------------------------------------------------

/// Metadata about a single gradient checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointRecord {
    /// Human-readable checkpoint identifier (e.g. `"checkpoint_round_42"`).
    pub id: String,
    /// Content address (CID string) of the checkpoint block.
    pub cid: String,
    /// Training round at which this checkpoint was created.
    pub round: u64,
    /// Wall-clock creation time in Unix milliseconds.
    pub created_at_ms: u64,
    /// Serialised size of the checkpoint data in bytes.
    pub size_bytes: u64,
    /// CRC-32 checksum of the checkpoint data.
    pub crc32: u32,
    /// Pinned checkpoints survive all pruning rules (unless the policy
    /// explicitly ignores pins, which the default does not).
    pub is_pinned: bool,
}

// ---------------------------------------------------------------------------
// RetentionPolicy
// ---------------------------------------------------------------------------

/// Configures which checkpoints the [`CheckpointPruner`] should keep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Always keep the *N* most recent checkpoints (ordered by `round`).
    /// Default: 5.
    pub keep_last_n: usize,
    /// Pinned checkpoints are never deleted. Default: `true`.
    pub keep_pinned: bool,
    /// If set, prune oldest checkpoints until the total size is at or below
    /// this threshold (bytes). Pinned checkpoints count toward the budget but
    /// are never deleted because of it (they can only be deleted when
    /// `keep_pinned` is `false`).
    pub max_total_bytes: Option<u64>,
    /// Never prune checkpoints whose age (now_ms − created_at_ms) is less
    /// than this value. Default: 0 (no minimum age restriction).
    pub min_age_ms: u64,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            keep_last_n: 5,
            keep_pinned: true,
            max_total_bytes: None,
            min_age_ms: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// CheckpointPruner
// ---------------------------------------------------------------------------

/// Applies a [`RetentionPolicy`] to a collection of [`CheckpointRecord`]s and
/// determines which records should be deleted.
///
/// # Usage
/// ```
/// use ipfrs_tensorlogic::checkpoint_manager::{CheckpointRecord, CheckpointPruner, RetentionPolicy};
///
/// let records = vec![
///     CheckpointRecord { id: "cp_1".into(), cid: "Qm1".into(), round: 1,
///         created_at_ms: 0, size_bytes: 100, crc32: 0, is_pinned: false },
///     CheckpointRecord { id: "cp_2".into(), cid: "Qm2".into(), round: 2,
///         created_at_ms: 0, size_bytes: 100, crc32: 0, is_pinned: false },
/// ];
/// let policy = RetentionPolicy { keep_last_n: 1, ..Default::default() };
/// let mut pruner = CheckpointPruner::new(records, policy);
/// let to_delete = pruner.prune();
/// assert_eq!(to_delete.len(), 1);
/// assert_eq!(to_delete[0].id, "cp_1");
/// ```
pub struct CheckpointPruner {
    /// Checkpoint records, kept sorted by `round` ascending at all times.
    records: Vec<CheckpointRecord>,
    /// Retention policy to apply.
    policy: RetentionPolicy,
    /// Unix timestamp (ms) used as "now" for age-based calculations.
    /// Defaults to 0, meaning *all* checkpoints are considered arbitrarily old
    /// unless overridden via [`CheckpointPruner::with_now_ms`].
    now_ms: u64,
}

impl CheckpointPruner {
    /// Create a new pruner.  Records are sorted by `round` ascending.
    pub fn new(mut records: Vec<CheckpointRecord>, policy: RetentionPolicy) -> Self {
        records.sort_by_key(|r| r.round);
        Self {
            records,
            policy,
            now_ms: 0,
        }
    }

    /// Override the "current time" used for age-based pruning decisions.
    ///
    /// If not called, `now_ms` defaults to `0`, which means every checkpoint
    /// has an age ≥ 0 and the `min_age_ms` guard applies correctly.
    pub fn with_now_ms(mut self, now_ms: u64) -> Self {
        self.now_ms = now_ms;
        self
    }

    /// Apply the retention policy and return the list of records to **delete**.
    ///
    /// After the call the pruner's internal record list contains only the
    /// surviving records; repeated calls therefore return an empty list.
    pub fn prune(&mut self) -> Vec<CheckpointRecord> {
        // Mark each record with its index (ascending round order).
        // We build a boolean "keep" mask and then partition.
        let n = self.records.len();
        let mut keep = vec![false; n];

        // --- Step 1: keep_last_n ---------------------------------------------------
        // The records are sorted ascending by round, so the last `keep_last_n`
        // entries are the most-recent ones.
        let keep_last_n = self.policy.keep_last_n;
        if keep_last_n > 0 && n > 0 {
            let start = n.saturating_sub(keep_last_n);
            for k in &mut keep[start..] {
                *k = true;
            }
        }

        // --- Step 2: keep_pinned --------------------------------------------------
        if self.policy.keep_pinned {
            for (k, rec) in keep.iter_mut().zip(self.records.iter()) {
                if rec.is_pinned {
                    *k = true;
                }
            }
        }

        // --- Step 3: min_age_ms ---------------------------------------------------
        // Protect checkpoints that are too young to prune.
        for (k, rec) in keep.iter_mut().zip(self.records.iter()) {
            let age_ms = self.now_ms.saturating_sub(rec.created_at_ms);
            if age_ms < self.policy.min_age_ms {
                *k = true;
            }
        }

        // --- Step 4: max_total_bytes ----------------------------------------------
        // If the survivors still exceed the byte budget, prune oldest
        // non-pinned (or non-protected) survivors from oldest to newest.
        if let Some(budget) = self.policy.max_total_bytes {
            // Compute current total across *all* records (we haven't removed
            // anything yet; deletions happen at the end).
            let survivor_bytes: u64 = self
                .records
                .iter()
                .enumerate()
                .filter(|(i, _)| keep[*i])
                .map(|(_, r)| r.size_bytes)
                .sum();

            if survivor_bytes > budget {
                let mut excess = survivor_bytes - budget;
                // Walk from oldest to newest; try to evict survivors that are
                // not pinned (and not age-protected).  We need mutable access
                // to `keep[i]` while also reading `self.records[i]`, so we
                // collect the eviction decisions in a separate pass.
                let evict_flags: Vec<bool> = keep
                    .iter()
                    .zip(self.records.iter())
                    .map(|(k, rec)| {
                        if !k {
                            return false; // already scheduled for deletion
                        }
                        if self.policy.keep_pinned && rec.is_pinned {
                            return false;
                        }
                        let age_ms = self.now_ms.saturating_sub(rec.created_at_ms);
                        if age_ms < self.policy.min_age_ms {
                            return false;
                        }
                        true // candidate for eviction
                    })
                    .collect();

                for (flag, (k, rec)) in evict_flags
                    .iter()
                    .zip(keep.iter_mut().zip(self.records.iter()))
                {
                    if excess == 0 {
                        break;
                    }
                    if *flag {
                        *k = false;
                        excess = excess.saturating_sub(rec.size_bytes);
                    }
                }
            }
        }

        // --- Partition records ----------------------------------------------------
        // Collect records to delete (keep[i] == false) and update self.records.
        let mut to_delete = Vec::new();
        let mut survivors = Vec::new();
        for (i, rec) in self.records.drain(..).enumerate() {
            if keep[i] {
                survivors.push(rec);
            } else {
                to_delete.push(rec);
            }
        }
        self.records = survivors;
        to_delete
    }

    /// Number of records currently held by the pruner (i.e. after any pruning).
    pub fn surviving_count(&self) -> usize {
        self.records.len()
    }

    /// Sum of `size_bytes` across all records currently held by the pruner.
    pub fn total_bytes(&self) -> u64 {
        self.records.iter().map(|r| r.size_bytes).sum()
    }

    /// Number of pinned records currently held by the pruner.
    pub fn pinned_count(&self) -> usize {
        self.records.iter().filter(|r| r.is_pinned).count()
    }

    /// Read-only access to the surviving records.
    pub fn records(&self) -> &[CheckpointRecord] {
        &self.records
    }
}

// ---------------------------------------------------------------------------
// CheckpointValidator
// ---------------------------------------------------------------------------

/// Validates checkpoint data against stored CRC-32 and size metadata.
pub struct CheckpointValidator;

impl CheckpointValidator {
    /// Compute the CRC-32 of `data`.
    pub fn compute_crc32(data: &[u8]) -> u32 {
        crc32(data)
    }

    /// Return `Ok(())` if the CRC-32 of `data` matches `expected_crc32`.
    pub fn validate(data: &[u8], expected_crc32: u32) -> Result<(), ValidationError> {
        let actual = crc32(data);
        if actual != expected_crc32 {
            return Err(ValidationError::CrcMismatch {
                expected: expected_crc32,
                actual,
            });
        }
        Ok(())
    }

    /// Validate both the CRC-32 and the recorded size of a checkpoint.
    ///
    /// Returns the first error encountered (size is checked after CRC).
    pub fn validate_record(record: &CheckpointRecord, data: &[u8]) -> Result<(), ValidationError> {
        // CRC check first.
        let actual_crc = crc32(data);
        if actual_crc != record.crc32 {
            return Err(ValidationError::CrcMismatch {
                expected: record.crc32,
                actual: actual_crc,
            });
        }
        // Size check.
        let actual_size = data.len() as u64;
        if actual_size != record.size_bytes {
            return Err(ValidationError::SizeMismatch {
                expected: record.size_bytes,
                actual: actual_size,
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- helpers -----------------------------------------------------------

    fn make_record(id: &str, round: u64, size_bytes: u64, is_pinned: bool) -> CheckpointRecord {
        CheckpointRecord {
            id: id.to_string(),
            cid: format!("Qm{round}"),
            round,
            created_at_ms: round * 1000, // 1 s per round for simplicity
            size_bytes,
            crc32: 0,
            is_pinned,
        }
    }

    fn make_record_timed(
        id: &str,
        round: u64,
        size_bytes: u64,
        is_pinned: bool,
        created_at_ms: u64,
    ) -> CheckpointRecord {
        CheckpointRecord {
            id: id.to_string(),
            cid: format!("Qm{round}"),
            round,
            created_at_ms,
            size_bytes,
            crc32: 0,
            is_pinned,
        }
    }

    // -----------------------------------------------------------------------
    // CRC-32 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_crc32_known_vector() {
        // Standard IEEE 802.3 test vector.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn test_crc32_empty() {
        // CRC of empty slice is well-defined.
        assert_eq!(crc32(b""), 0x0000_0000);
    }

    #[test]
    fn test_crc32_single_byte() {
        // Regression: single-byte input.
        let v = crc32(b"a");
        assert_ne!(v, 0); // Must produce a non-zero value.
    }

    #[test]
    fn test_crc32_deterministic() {
        let data = b"hello, checkpoint!";
        assert_eq!(crc32(data), crc32(data));
    }

    #[test]
    fn test_crc32_sensitive_to_changes() {
        let a = crc32(b"hello");
        let b = crc32(b"hellp"); // single bit flip in last byte
        assert_ne!(a, b);
    }

    // -----------------------------------------------------------------------
    // CheckpointValidator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_passes_for_correct_data() {
        let data = b"gradient checkpoint payload";
        let checksum = crc32(data);
        assert!(CheckpointValidator::validate(data, checksum).is_ok());
    }

    #[test]
    fn test_validate_fails_for_corrupted_data() {
        let data = b"gradient checkpoint payload";
        let checksum = crc32(data);
        let mut corrupted = data.to_vec();
        corrupted[0] ^= 0xFF; // flip first byte
        let err = CheckpointValidator::validate(&corrupted, checksum).unwrap_err();
        assert!(matches!(err, ValidationError::CrcMismatch { .. }));
    }

    #[test]
    fn test_validate_record_passes() {
        let data = b"some checkpoint data";
        let record = CheckpointRecord {
            id: "cp_ok".into(),
            cid: "Qmabc".into(),
            round: 1,
            created_at_ms: 0,
            size_bytes: data.len() as u64,
            crc32: crc32(data),
            is_pinned: false,
        };
        assert!(CheckpointValidator::validate_record(&record, data).is_ok());
    }

    #[test]
    fn test_validate_record_crc_mismatch() {
        let data = b"some checkpoint data";
        let record = CheckpointRecord {
            id: "cp_bad_crc".into(),
            cid: "Qmabc".into(),
            round: 1,
            created_at_ms: 0,
            size_bytes: data.len() as u64,
            crc32: 0xDEAD_BEEF, // wrong checksum
            is_pinned: false,
        };
        let err = CheckpointValidator::validate_record(&record, data).unwrap_err();
        assert!(matches!(err, ValidationError::CrcMismatch { .. }));
    }

    #[test]
    fn test_validate_record_size_mismatch() {
        let data = b"some checkpoint data";
        let record = CheckpointRecord {
            id: "cp_bad_size".into(),
            cid: "Qmabc".into(),
            round: 1,
            created_at_ms: 0,
            size_bytes: 9999, // wrong size
            crc32: crc32(data),
            is_pinned: false,
        };
        let err = CheckpointValidator::validate_record(&record, data).unwrap_err();
        assert!(matches!(err, ValidationError::SizeMismatch { .. }));
    }

    #[test]
    fn test_compute_crc32_matches_standalone() {
        let data = b"cross-check";
        assert_eq!(CheckpointValidator::compute_crc32(data), crc32(data));
    }

    // -----------------------------------------------------------------------
    // RetentionPolicy / CheckpointPruner tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_keep_last_n_trims_oldest() {
        let records: Vec<CheckpointRecord> = (1..=5)
            .map(|r| make_record(&format!("cp_{r}"), r, 100, false))
            .collect();
        let policy = RetentionPolicy {
            keep_last_n: 3,
            ..Default::default()
        };
        let mut pruner = CheckpointPruner::new(records, policy);
        let deleted = pruner.prune();
        // Rounds 1 and 2 should be deleted.
        assert_eq!(deleted.len(), 2);
        let deleted_rounds: Vec<u64> = deleted.iter().map(|r| r.round).collect();
        assert!(deleted_rounds.contains(&1));
        assert!(deleted_rounds.contains(&2));
        assert_eq!(pruner.surviving_count(), 3);
    }

    #[test]
    fn test_keep_pinned_preserves_old_checkpoints() {
        let mut records: Vec<CheckpointRecord> = (1..=5)
            .map(|r| make_record(&format!("cp_{r}"), r, 100, false))
            .collect();
        // Pin round 1 (oldest).
        records[0].is_pinned = true;

        let policy = RetentionPolicy {
            keep_last_n: 3,
            keep_pinned: true,
            ..Default::default()
        };
        let mut pruner = CheckpointPruner::new(records, policy);
        let deleted = pruner.prune();
        // Round 2 should be deleted; round 1 (pinned) must survive.
        assert_eq!(deleted.len(), 1);
        assert_eq!(deleted[0].round, 2);
        assert_eq!(pruner.surviving_count(), 4);
        assert_eq!(pruner.pinned_count(), 1);
    }

    #[test]
    fn test_keep_pinned_false_allows_pruning_pinned() {
        let mut records: Vec<CheckpointRecord> = (1..=5)
            .map(|r| make_record(&format!("cp_{r}"), r, 100, false))
            .collect();
        records[0].is_pinned = true;

        let policy = RetentionPolicy {
            keep_last_n: 3,
            keep_pinned: false,
            ..Default::default()
        };
        let mut pruner = CheckpointPruner::new(records, policy);
        let deleted = pruner.prune();
        assert_eq!(deleted.len(), 2);
        assert_eq!(pruner.surviving_count(), 3);
    }

    #[test]
    fn test_max_total_bytes_budget_enforcement() {
        // 5 records of 100 bytes each → 500 bytes total.
        // Budget: 250 bytes → must prune at least 2 oldest.
        let records: Vec<CheckpointRecord> = (1..=5)
            .map(|r| make_record(&format!("cp_{r}"), r, 100, false))
            .collect();
        let policy = RetentionPolicy {
            keep_last_n: 5, // would keep all by default
            keep_pinned: true,
            max_total_bytes: Some(250),
            min_age_ms: 0,
        };
        let mut pruner = CheckpointPruner::new(records, policy);
        let deleted = pruner.prune();
        // 500 − 250 = 250 bytes to prune → 2 or 3 records (100 each).
        // The pruner removes from oldest first, so rounds 1 and 2 go first
        // (200 bytes), then round 3 if still over budget—but 500-200=300 > 250
        // so round 3 is also evicted: 3 records deleted, 200 bytes remaining.
        assert!(pruner.total_bytes() <= 250);
        assert!(!deleted.is_empty());
    }

    #[test]
    fn test_min_age_ms_protects_young_checkpoints() {
        // now_ms = 100_000.  min_age_ms = 50_000.
        // Records and their ages:
        //   round 1: created_at=0,      age=100_000 >= 50_000 → pruneable
        //   round 2: created_at=1_000,  age=99_000  >= 50_000 → pruneable
        //   round 3: created_at=2_000,  age=98_000  >= 50_000 → pruneable
        //   round 4: created_at=80_000, age=20_000  < 50_000  → age-protected
        //
        // Policy: keep_last_n=1 (would keep round 4 anyway), min_age_ms=50_000.
        // Rounds 1, 2, 3 are old enough and not in keep_last_n → deleted.
        // Round 4 survives (keep_last_n AND age-protected).
        let now_ms: u64 = 100_000;
        let records = vec![
            make_record_timed("cp_1", 1, 100, false, 0),
            make_record_timed("cp_2", 2, 100, false, 1_000),
            make_record_timed("cp_3", 3, 100, false, 2_000),
            make_record_timed("cp_4", 4, 100, false, 80_000),
        ];

        let policy = RetentionPolicy {
            keep_last_n: 1,
            keep_pinned: true,
            max_total_bytes: None,
            min_age_ms: 50_000,
        };
        let mut pruner = CheckpointPruner::new(records, policy).with_now_ms(now_ms);
        let deleted = pruner.prune();

        let surviving_rounds: Vec<u64> = pruner.records().iter().map(|r| r.round).collect();
        assert!(
            surviving_rounds.contains(&4),
            "round 4 must survive (keep_last_n)"
        );

        // Rounds 1, 2, 3 are old enough and not in keep_last_n → all deleted.
        assert_eq!(deleted.len(), 3, "rounds 1, 2, 3 should be deleted");
    }

    #[test]
    fn test_min_age_ms_protects_within_window() {
        let now_ms: u64 = 10_000;
        // All created at time 0 except the youngest, created at 8_000 ms.
        let records = vec![
            make_record_timed("cp_1", 1, 100, false, 0),
            make_record_timed("cp_2", 2, 100, false, 0),
            make_record_timed("cp_young", 3, 100, false, 8_000), // age 2_000 ms
        ];
        let policy = RetentionPolicy {
            keep_last_n: 1,
            keep_pinned: true,
            max_total_bytes: None,
            min_age_ms: 5_000, // protect checkpoints younger than 5 s
        };
        let mut pruner = CheckpointPruner::new(records, policy).with_now_ms(now_ms);
        let _deleted = pruner.prune();
        let surviving: Vec<u64> = pruner.records().iter().map(|r| r.round).collect();
        // cp_young has age 2_000 < 5_000 → protected; also kept by keep_last_n.
        assert!(surviving.contains(&3), "young checkpoint must survive");
    }

    #[test]
    fn test_combined_keep_last_n_and_keep_pinned() {
        // 6 records; rounds 1 and 3 are pinned; keep_last_n = 2.
        // Expected survivors: rounds 1 (pinned), 3 (pinned), 5, 6 (last 2).
        // Deleted: rounds 2, 4.
        let mut records: Vec<CheckpointRecord> = (1..=6)
            .map(|r| make_record(&format!("cp_{r}"), r, 100, false))
            .collect();
        records[0].is_pinned = true; // round 1
        records[2].is_pinned = true; // round 3

        let policy = RetentionPolicy {
            keep_last_n: 2,
            keep_pinned: true,
            max_total_bytes: None,
            min_age_ms: 0,
        };
        let mut pruner = CheckpointPruner::new(records, policy);
        let deleted = pruner.prune();

        let deleted_rounds: Vec<u64> = deleted.iter().map(|r| r.round).collect();
        assert!(deleted_rounds.contains(&2), "round 2 must be deleted");
        assert!(deleted_rounds.contains(&4), "round 4 must be deleted");
        assert_eq!(deleted.len(), 2, "exactly 2 records should be deleted");

        let surviving: Vec<u64> = pruner.records().iter().map(|r| r.round).collect();
        assert!(surviving.contains(&1), "pinned round 1 must survive");
        assert!(surviving.contains(&3), "pinned round 3 must survive");
        assert!(surviving.contains(&5), "keep_last_n round 5 must survive");
        assert!(surviving.contains(&6), "keep_last_n round 6 must survive");
    }

    #[test]
    fn test_prune_returns_correct_delete_set() {
        let records: Vec<CheckpointRecord> = (1..=4)
            .map(|r| make_record(&format!("cp_{r}"), r, 50, false))
            .collect();
        let policy = RetentionPolicy {
            keep_last_n: 2,
            ..Default::default()
        };
        let mut pruner = CheckpointPruner::new(records, policy);
        let deleted = pruner.prune();
        assert_eq!(deleted.len(), 2);
        let ids: Vec<&str> = deleted.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"cp_1"));
        assert!(ids.contains(&"cp_2"));
    }

    #[test]
    fn test_surviving_count_and_total_bytes() {
        let records: Vec<CheckpointRecord> = (1..=6)
            .map(|r| make_record(&format!("cp_{r}"), r, 200, false))
            .collect();
        let policy = RetentionPolicy {
            keep_last_n: 4,
            ..Default::default()
        };
        let mut pruner = CheckpointPruner::new(records, policy);
        pruner.prune();
        assert_eq!(pruner.surviving_count(), 4);
        assert_eq!(pruner.total_bytes(), 4 * 200);
    }

    #[test]
    fn test_prune_idempotent_after_first_call() {
        let records: Vec<CheckpointRecord> = (1..=5)
            .map(|r| make_record(&format!("cp_{r}"), r, 100, false))
            .collect();
        let policy = RetentionPolicy {
            keep_last_n: 3,
            ..Default::default()
        };
        let mut pruner = CheckpointPruner::new(records, policy);
        let first = pruner.prune();
        let second = pruner.prune();
        assert_eq!(first.len(), 2);
        assert!(second.is_empty(), "second prune should delete nothing");
    }

    #[test]
    fn test_pinned_count_accuracy() {
        let mut records: Vec<CheckpointRecord> = (1..=5)
            .map(|r| make_record(&format!("cp_{r}"), r, 100, false))
            .collect();
        records[1].is_pinned = true;
        records[3].is_pinned = true;
        let policy = RetentionPolicy::default();
        let pruner = CheckpointPruner::new(records, policy);
        assert_eq!(pruner.pinned_count(), 2);
    }
}
