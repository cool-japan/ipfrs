//! Block Integrity Checker
//!
//! Verifies stored blocks have valid CIDs matching their content,
//! detecting corruption and reporting detailed results.

/// Errors that can occur during integrity checking.
#[derive(Clone, Debug, PartialEq)]
pub enum IntegrityError {
    /// The stored CID does not match the computed CID for the block data.
    CidMismatch {
        stored_cid: String,
        computed_cid: String,
    },
    /// The block has no data (empty).
    EmptyBlock { cid: String },
    /// The CID string is not in a valid format.
    InvalidCidFormat { cid: String, reason: String },
    /// The hash function encoded in the CID is not supported.
    HashFunctionUnsupported { function_code: u64 },
}

/// The result of checking a single block.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedBlock {
    /// The CID of the checked block.
    pub cid: String,
    /// Size of the block data in bytes.
    pub size_bytes: usize,
    /// Outcome of the integrity check.
    pub status: CheckStatus,
}

/// The status of a block after integrity checking.
#[derive(Clone, Debug, PartialEq)]
pub enum CheckStatus {
    /// The block's CID matches the computed CID from its content.
    Valid,
    /// The block failed integrity checking.
    Invalid(IntegrityError),
    /// The block was skipped (e.g. unsupported codec, empty block with skip_empty=true).
    Skipped { reason: String },
}

/// Configuration for the `BlockIntegrityChecker`.
#[derive(Clone, Debug)]
pub struct CheckerConfig {
    /// If `true`, empty blocks are skipped rather than treated as errors.
    pub skip_empty: bool,
    /// Stop after checking this many blocks (if `Some`).
    pub max_blocks: Option<usize>,
    /// The hash function to use when computing expected CIDs.
    pub hash_fn: HashFunction,
}

impl Default for CheckerConfig {
    fn default() -> Self {
        Self {
            skip_empty: false,
            max_blocks: None,
            hash_fn: HashFunction::Sha256,
        }
    }
}

/// Hash function used to verify block CIDs.
#[derive(Clone, Debug, PartialEq)]
pub enum HashFunction {
    /// Standard CID hash (produces "bafy…" prefixed CIDs).
    Sha256,
    /// BLAKE3 hash (produces "bafk…" prefixed CIDs).
    Blake3,
    /// Identity: CID content equals the raw data.
    /// Used for test blocks — CID is `identity:<hex of first 8 bytes>`.
    Identity,
}

/// Aggregated report from checking a set of blocks.
#[derive(Debug)]
pub struct IntegrityReport {
    /// Total number of blocks that were checked (including skipped).
    pub total_checked: usize,
    /// Number of blocks whose CID matched the computed CID.
    pub valid: usize,
    /// Number of blocks whose CID did not match or were otherwise invalid.
    pub invalid: usize,
    /// Number of blocks that were skipped.
    pub skipped: usize,
    /// Per-block results.
    pub results: Vec<CheckedBlock>,
}

impl IntegrityReport {
    /// Returns the ratio of invalid blocks to total checked blocks.
    /// Returns `0.0` if no blocks were checked.
    pub fn error_rate(&self) -> f64 {
        if self.total_checked == 0 {
            0.0
        } else {
            self.invalid as f64 / self.total_checked as f64
        }
    }

    /// Returns the CIDs of all invalid blocks.
    pub fn invalid_cids(&self) -> Vec<&str> {
        self.results
            .iter()
            .filter(|b| matches!(b.status, CheckStatus::Invalid(_)))
            .map(|b| b.cid.as_str())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit hash — used for CID simulation.
fn fnv1a_64(data: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;
    let mut hash = OFFSET_BASIS;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

// ---------------------------------------------------------------------------
// BlockIntegrityChecker
// ---------------------------------------------------------------------------

/// Checks stored blocks by verifying that their CIDs match the computed CID
/// derived from the block content.
pub struct BlockIntegrityChecker {
    /// Configuration controlling check behaviour.
    pub config: CheckerConfig,
}

impl BlockIntegrityChecker {
    /// Creates a new checker with the given configuration.
    pub fn new(config: CheckerConfig) -> Self {
        Self { config }
    }

    /// Verifies that `cid` is a syntactically valid CID string.
    ///
    /// Currently this is a simplified check: an empty string is rejected,
    /// any non-empty string is considered valid.
    pub fn verify_cid_format(&self, cid: &str) -> Result<(), IntegrityError> {
        if cid.is_empty() {
            return Err(IntegrityError::InvalidCidFormat {
                cid: cid.to_string(),
                reason: "empty".to_string(),
            });
        }
        Ok(())
    }

    /// Computes the expected CID for `data` using the configured hash function.
    fn compute_cid(&self, data: &[u8]) -> String {
        match self.config.hash_fn {
            HashFunction::Identity => {
                // CID = "identity:" + hex of first 8 bytes (zero-padded)
                let len = data.len().min(8);
                let hex_str: String = data[..len].iter().map(|b| format!("{:02x}", b)).collect();
                format!("identity:{}", hex_str)
            }
            HashFunction::Sha256 => {
                // Simplified simulation: "bafy" + FNV-1a hash of data as hex
                let hash = fnv1a_64(data);
                format!("bafy{:016x}", hash)
            }
            HashFunction::Blake3 => {
                // Simplified simulation: "bafk" + FNV-1a of reversed data as hex
                let mut reversed = data.to_vec();
                reversed.reverse();
                let hash = fnv1a_64(&reversed);
                format!("bafk{:016x}", hash)
            }
        }
    }

    /// Checks the integrity of a single block.
    ///
    /// # Arguments
    /// * `cid`  – The CID stored alongside the block.
    /// * `data` – The raw block bytes.
    pub fn check_block(&self, cid: &str, data: &[u8]) -> CheckedBlock {
        // Handle empty data
        if data.is_empty() {
            if self.config.skip_empty {
                return CheckedBlock {
                    cid: cid.to_string(),
                    size_bytes: 0,
                    status: CheckStatus::Skipped {
                        reason: "empty block".to_string(),
                    },
                };
            } else {
                return CheckedBlock {
                    cid: cid.to_string(),
                    size_bytes: 0,
                    status: CheckStatus::Invalid(IntegrityError::EmptyBlock {
                        cid: cid.to_string(),
                    }),
                };
            }
        }

        // Validate CID format
        if let Err(e) = self.verify_cid_format(cid) {
            return CheckedBlock {
                cid: cid.to_string(),
                size_bytes: data.len(),
                status: CheckStatus::Invalid(e),
            };
        }

        // Compute expected CID and compare
        let computed = self.compute_cid(data);
        let status = if cid == computed {
            CheckStatus::Valid
        } else {
            CheckStatus::Invalid(IntegrityError::CidMismatch {
                stored_cid: cid.to_string(),
                computed_cid: computed,
            })
        };

        CheckedBlock {
            cid: cid.to_string(),
            size_bytes: data.len(),
            status,
        }
    }

    /// Checks a slice of `(cid, data)` pairs and returns an aggregated report.
    ///
    /// Respects `config.max_blocks`: stops after that many checks.
    pub fn check_blocks(&self, blocks: &[(&str, &[u8])]) -> IntegrityReport {
        let limit = self
            .config
            .max_blocks
            .unwrap_or(usize::MAX)
            .min(blocks.len());

        let mut results = Vec::with_capacity(limit);
        let mut valid = 0usize;
        let mut invalid = 0usize;
        let mut skipped = 0usize;

        for (cid, data) in blocks.iter().take(limit) {
            let checked = self.check_block(cid, data);
            match &checked.status {
                CheckStatus::Valid => valid += 1,
                CheckStatus::Invalid(_) => invalid += 1,
                CheckStatus::Skipped { .. } => skipped += 1,
            }
            results.push(checked);
        }

        IntegrityReport {
            total_checked: results.len(),
            valid,
            invalid,
            skipped,
            results,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_checker() -> BlockIntegrityChecker {
        BlockIntegrityChecker::new(CheckerConfig::default())
    }

    fn sha256_checker() -> BlockIntegrityChecker {
        BlockIntegrityChecker::new(CheckerConfig {
            hash_fn: HashFunction::Sha256,
            ..CheckerConfig::default()
        })
    }

    fn identity_checker() -> BlockIntegrityChecker {
        BlockIntegrityChecker::new(CheckerConfig {
            hash_fn: HashFunction::Identity,
            ..CheckerConfig::default()
        })
    }

    fn blake3_checker() -> BlockIntegrityChecker {
        BlockIntegrityChecker::new(CheckerConfig {
            hash_fn: HashFunction::Blake3,
            ..CheckerConfig::default()
        })
    }

    // 1. new() with default config
    #[test]
    fn test_new_default_config() {
        let checker = default_checker();
        assert!(!checker.config.skip_empty);
        assert!(checker.config.max_blocks.is_none());
        assert_eq!(checker.config.hash_fn, HashFunction::Sha256);
    }

    // 2. Empty data, skip_empty = false → Invalid(EmptyBlock)
    #[test]
    fn test_check_block_empty_skip_false() {
        let checker = default_checker();
        let result = checker.check_block("bafytest", &[]);
        assert_eq!(
            result.status,
            CheckStatus::Invalid(IntegrityError::EmptyBlock {
                cid: "bafytest".to_string()
            })
        );
    }

    // 3. Empty data, skip_empty = true → Skipped
    #[test]
    fn test_check_block_empty_skip_true() {
        let checker = BlockIntegrityChecker::new(CheckerConfig {
            skip_empty: true,
            ..CheckerConfig::default()
        });
        let result = checker.check_block("bafytest", &[]);
        assert!(matches!(result.status, CheckStatus::Skipped { .. }));
    }

    // 4. Identity hash matches computed CID
    #[test]
    fn test_check_block_identity_matches() {
        let checker = identity_checker();
        let data = b"hello world";
        let cid = checker.compute_cid(data);
        let result = checker.check_block(&cid, data);
        assert_eq!(result.status, CheckStatus::Valid);
    }

    // 5. Identity hash with wrong CID → CidMismatch
    #[test]
    fn test_check_block_identity_mismatch() {
        let checker = identity_checker();
        let data = b"hello world";
        let result = checker.check_block("identity:wrongvalue", data);
        assert!(matches!(
            result.status,
            CheckStatus::Invalid(IntegrityError::CidMismatch { .. })
        ));
    }

    // 6. Sha256 computed CID has "bafy" prefix
    #[test]
    fn test_check_block_sha256_prefix() {
        let checker = sha256_checker();
        let data = b"some block data";
        let cid = checker.compute_cid(data);
        assert!(
            cid.starts_with("bafy"),
            "CID should start with 'bafy', got: {}",
            cid
        );
    }

    // 7. Blake3 computed CID has "bafk" prefix
    #[test]
    fn test_check_block_blake3_prefix() {
        let checker = blake3_checker();
        let data = b"some block data";
        let cid = checker.compute_cid(data);
        assert!(
            cid.starts_with("bafk"),
            "CID should start with 'bafk', got: {}",
            cid
        );
    }

    // 8. check_blocks: multiple blocks, counts correct
    #[test]
    fn test_check_blocks_counts() {
        let checker = sha256_checker();
        let data1 = b"block one";
        let data2 = b"block two";
        let cid1 = checker.compute_cid(data1);
        let cid2 = checker.compute_cid(data2);
        let blocks: Vec<(&str, &[u8])> = vec![
            (cid1.as_str(), data1.as_ref()),
            (cid2.as_str(), data2.as_ref()),
        ];
        let report = checker.check_blocks(&blocks);
        assert_eq!(report.total_checked, 2);
        assert_eq!(report.valid, 2);
        assert_eq!(report.invalid, 0);
        assert_eq!(report.skipped, 0);
    }

    // 9. check_blocks: max_blocks limit respected
    #[test]
    fn test_check_blocks_max_blocks() {
        let checker = BlockIntegrityChecker::new(CheckerConfig {
            max_blocks: Some(2),
            hash_fn: HashFunction::Sha256,
            ..CheckerConfig::default()
        });
        let data: &[u8] = b"data";
        let cid = checker.compute_cid(data);
        let blocks: Vec<(&str, &[u8])> = vec![
            (cid.as_str(), data),
            (cid.as_str(), data),
            (cid.as_str(), data),
            (cid.as_str(), data),
        ];
        let report = checker.check_blocks(&blocks);
        assert_eq!(report.total_checked, 2);
    }

    // 10. check_blocks: mix of valid and invalid
    #[test]
    fn test_check_blocks_mixed() {
        let checker = sha256_checker();
        let data = b"real data";
        let good_cid = checker.compute_cid(data);
        let blocks: Vec<(&str, &[u8])> = vec![
            (good_cid.as_str(), data.as_ref()),
            ("bafybadcid", data.as_ref()),
        ];
        let report = checker.check_blocks(&blocks);
        assert_eq!(report.valid, 1);
        assert_eq!(report.invalid, 1);
    }

    // 11. IntegrityReport error_rate calculation
    #[test]
    fn test_error_rate() {
        let checker = sha256_checker();
        let data = b"some data";
        let good_cid = checker.compute_cid(data);
        let blocks: Vec<(&str, &[u8])> = vec![
            (good_cid.as_str(), data.as_ref()),
            ("bafybad1", data.as_ref()),
            ("bafybad2", data.as_ref()),
            ("bafybad3", data.as_ref()),
        ];
        let report = checker.check_blocks(&blocks);
        let rate = report.error_rate();
        assert!(
            (rate - 0.75).abs() < f64::EPSILON,
            "Expected 0.75, got {}",
            rate
        );
    }

    // 12. IntegrityReport error_rate when no blocks checked
    #[test]
    fn test_error_rate_zero_blocks() {
        let report = IntegrityReport {
            total_checked: 0,
            valid: 0,
            invalid: 0,
            skipped: 0,
            results: vec![],
        };
        assert_eq!(report.error_rate(), 0.0);
    }

    // 13. IntegrityReport invalid_cids returns correct CIDs
    #[test]
    fn test_invalid_cids() {
        let checker = sha256_checker();
        let data = b"some data";
        let good_cid = checker.compute_cid(data);
        let blocks: Vec<(&str, &[u8])> = vec![
            (good_cid.as_str(), data.as_ref()),
            ("bafybad_a", data.as_ref()),
            ("bafybad_b", data.as_ref()),
        ];
        let report = checker.check_blocks(&blocks);
        let mut bad = report.invalid_cids();
        bad.sort_unstable();
        assert_eq!(bad, vec!["bafybad_a", "bafybad_b"]);
    }

    // 14. verify_cid_format: empty string → Err
    #[test]
    fn test_verify_cid_format_empty() {
        let checker = default_checker();
        let result = checker.verify_cid_format("");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            IntegrityError::InvalidCidFormat { .. }
        ));
    }

    // 15. verify_cid_format: non-empty → Ok
    #[test]
    fn test_verify_cid_format_nonempty() {
        let checker = default_checker();
        assert!(checker
            .verify_cid_format("bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi")
            .is_ok());
        assert!(checker
            .verify_cid_format("QmPK1s3pNYLi9ERiq3BDxKa4XosgWwFRQUydHUtz4YgpqB")
            .is_ok());
    }

    // 16. Two identical data inputs produce the same CID
    #[test]
    fn test_identical_data_same_cid() {
        let checker = sha256_checker();
        let data = b"deterministic test data";
        let cid1 = checker.compute_cid(data);
        let cid2 = checker.compute_cid(data);
        assert_eq!(cid1, cid2);
    }

    // 17. Different data produce different CIDs (Sha256)
    #[test]
    fn test_different_data_different_cids() {
        let checker = sha256_checker();
        let cid_a = checker.compute_cid(b"data alpha");
        let cid_b = checker.compute_cid(b"data beta");
        assert_ne!(cid_a, cid_b);
    }

    // 18. valid + invalid + skipped == total_checked
    #[test]
    fn test_sum_equals_total() {
        let checker = BlockIntegrityChecker::new(CheckerConfig {
            skip_empty: true,
            hash_fn: HashFunction::Sha256,
            ..CheckerConfig::default()
        });
        let data = b"real data";
        let good_cid = checker.compute_cid(data);
        let blocks: Vec<(&str, &[u8])> = vec![
            (good_cid.as_str(), data.as_ref()), // valid
            ("bafybad", data.as_ref()),         // invalid (CID mismatch)
            ("bafyempty", b""),                 // skipped (empty + skip_empty=true)
        ];
        let report = checker.check_blocks(&blocks);
        assert_eq!(
            report.valid + report.invalid + report.skipped,
            report.total_checked
        );
        assert_eq!(report.total_checked, 3);
        assert_eq!(report.valid, 1);
        assert_eq!(report.invalid, 1);
        assert_eq!(report.skipped, 1);
    }
}
