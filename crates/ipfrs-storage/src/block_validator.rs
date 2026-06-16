//! Storage block validator with multiple hash and integrity checks.
//!
//! Provides [`StorageBlockValidator`] for validating block integrity
//! using configurable rules including size bounds, magic byte prefixes,
//! and FNV-1a hash verification.

/// Result of a block validation pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationResult {
    /// Block passed all validation rules.
    Valid,
    /// Block size fell outside the allowed range.
    InvalidSize,
    /// Block hash did not match the expected value.
    InvalidHash,
    /// Block format (e.g. magic bytes) was incorrect.
    InvalidFormat,
    /// Block appears corrupted (multiple failures).
    Corrupted,
}

/// A single validation rule applied to a block.
#[derive(Debug, Clone)]
pub struct ValidationRule {
    /// Human-readable name for this rule.
    pub name: String,
    /// Maximum allowed block size in bytes.
    pub max_size: Option<u64>,
    /// Minimum allowed block size in bytes.
    pub min_size: Option<u64>,
    /// Required prefix (magic bytes) that the block data must start with.
    pub required_prefix: Option<Vec<u8>>,
    /// Expected FNV-1a 64-bit hash of the block data.
    pub expected_hash: Option<u64>,
}

/// Detailed report produced after validating a single block.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// CID (content identifier) of the validated block.
    pub block_cid: String,
    /// Overall validation result.
    pub result: ValidationResult,
    /// Number of individual checks that passed.
    pub checks_passed: usize,
    /// Number of individual checks that failed.
    pub checks_failed: usize,
    /// Human-readable details for each check outcome.
    pub details: Vec<String>,
}

/// Aggregate statistics for the validator.
#[derive(Debug, Clone)]
pub struct ValidatorStats {
    /// Number of registered validation rules.
    pub rules_count: usize,
    /// Total number of blocks validated.
    pub blocks_validated: u64,
    /// Number of blocks that passed validation.
    pub blocks_valid: u64,
    /// Number of blocks that failed validation.
    pub blocks_invalid: u64,
    /// Ratio of valid blocks to total validated (0.0–1.0).
    pub validity_ratio: f64,
}

/// Block integrity validator with configurable rules.
///
/// Validates blocks against a set of [`ValidationRule`]s, tracking
/// cumulative statistics across invocations.
pub struct StorageBlockValidator {
    rules: Vec<ValidationRule>,
    blocks_validated: u64,
    blocks_valid: u64,
    blocks_invalid: u64,
}

impl StorageBlockValidator {
    /// Create a new validator with no rules.
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            blocks_validated: 0,
            blocks_valid: 0,
            blocks_invalid: 0,
        }
    }

    /// Register a validation rule.
    pub fn add_rule(&mut self, rule: ValidationRule) {
        self.rules.push(rule);
    }

    /// Remove a rule by name. Returns `true` if a rule was removed.
    pub fn remove_rule(&mut self, name: &str) -> bool {
        let before = self.rules.len();
        self.rules.retain(|r| r.name != name);
        self.rules.len() < before
    }

    /// Number of registered rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Validate a single block against all registered rules.
    ///
    /// Every rule is evaluated; a block is considered valid only if it
    /// passes **all** checks across **all** rules.  When more than one
    /// check fails the result is [`ValidationResult::Corrupted`].
    pub fn validate(&mut self, block_cid: &str, data: &[u8]) -> ValidationReport {
        let mut checks_passed: usize = 0;
        let mut checks_failed: usize = 0;
        let mut details: Vec<String> = Vec::new();
        let mut last_failure: Option<ValidationResult> = None;

        if self.rules.is_empty() {
            // No rules means the block is trivially valid.
            self.blocks_validated += 1;
            self.blocks_valid += 1;
            return ValidationReport {
                block_cid: block_cid.to_string(),
                result: ValidationResult::Valid,
                checks_passed: 0,
                checks_failed: 0,
                details: vec!["no rules configured; block accepted".to_string()],
            };
        }

        for rule in &self.rules {
            // Size check
            if rule.min_size.is_some() || rule.max_size.is_some() {
                if Self::validate_size(data, rule.min_size, rule.max_size) {
                    checks_passed += 1;
                    details.push(format!("[{}] size check passed", rule.name));
                } else {
                    checks_failed += 1;
                    details.push(format!(
                        "[{}] size check failed: len={} min={:?} max={:?}",
                        rule.name,
                        data.len(),
                        rule.min_size,
                        rule.max_size
                    ));
                    last_failure = Some(ValidationResult::InvalidSize);
                }
            }

            // Prefix check
            if let Some(ref prefix) = rule.required_prefix {
                if Self::validate_prefix(data, prefix) {
                    checks_passed += 1;
                    details.push(format!("[{}] prefix check passed", rule.name));
                } else {
                    checks_failed += 1;
                    details.push(format!("[{}] prefix check failed", rule.name));
                    last_failure = Some(ValidationResult::InvalidFormat);
                }
            }

            // Hash check
            if let Some(expected) = rule.expected_hash {
                if Self::validate_hash(data, expected) {
                    checks_passed += 1;
                    details.push(format!("[{}] hash check passed", rule.name));
                } else {
                    checks_failed += 1;
                    let actual = Self::fnv1a_hash(data);
                    details.push(format!(
                        "[{}] hash check failed: expected={} actual={}",
                        rule.name, expected, actual
                    ));
                    last_failure = Some(ValidationResult::InvalidHash);
                }
            }
        }

        let result = if checks_failed == 0 {
            ValidationResult::Valid
        } else if checks_failed > 1 {
            ValidationResult::Corrupted
        } else {
            // Exactly one failure – use the specific failure kind.
            last_failure.unwrap_or(ValidationResult::Corrupted)
        };

        self.blocks_validated += 1;
        if result == ValidationResult::Valid {
            self.blocks_valid += 1;
        } else {
            self.blocks_invalid += 1;
        }

        ValidationReport {
            block_cid: block_cid.to_string(),
            result,
            checks_passed,
            checks_failed,
            details,
        }
    }

    /// Check whether `data` length is within `[min, max]`.
    pub fn validate_size(data: &[u8], min: Option<u64>, max: Option<u64>) -> bool {
        let len = data.len() as u64;
        if let Some(lo) = min {
            if len < lo {
                return false;
            }
        }
        if let Some(hi) = max {
            if len > hi {
                return false;
            }
        }
        true
    }

    /// Check whether `data` starts with the given `prefix`.
    pub fn validate_prefix(data: &[u8], prefix: &[u8]) -> bool {
        data.starts_with(prefix)
    }

    /// Check whether the FNV-1a hash of `data` equals `expected`.
    pub fn validate_hash(data: &[u8], expected: u64) -> bool {
        Self::fnv1a_hash(data) == expected
    }

    /// Compute the FNV-1a 64-bit hash of `data`.
    pub fn fnv1a_hash(data: &[u8]) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0100_0000_01b3;
        let mut hash = FNV_OFFSET;
        for &byte in data {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    /// Validate multiple blocks, returning a report for each.
    pub fn batch_validate(&mut self, blocks: &[(&str, &[u8])]) -> Vec<ValidationReport> {
        blocks
            .iter()
            .map(|(cid, data)| self.validate(cid, data))
            .collect()
    }

    /// Return aggregate validation statistics.
    pub fn stats(&self) -> ValidatorStats {
        let validity_ratio = if self.blocks_validated == 0 {
            0.0
        } else {
            self.blocks_valid as f64 / self.blocks_validated as f64
        };
        ValidatorStats {
            rules_count: self.rules.len(),
            blocks_validated: self.blocks_validated,
            blocks_valid: self.blocks_valid,
            blocks_invalid: self.blocks_invalid,
            validity_ratio,
        }
    }
}

impl Default for StorageBlockValidator {
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

    // Helper: build a simple rule with only a name.
    fn rule(name: &str) -> ValidationRule {
        ValidationRule {
            name: name.to_string(),
            max_size: None,
            min_size: None,
            required_prefix: None,
            expected_hash: None,
        }
    }

    // ----- FNV-1a correctness -----

    #[test]
    fn test_fnv1a_empty() {
        // FNV-1a 64-bit of empty input is the offset basis.
        assert_eq!(
            StorageBlockValidator::fnv1a_hash(b""),
            0xcbf2_9ce4_8422_2325
        );
    }

    #[test]
    fn test_fnv1a_known_vector() {
        // "hello" – verify deterministic output.
        let hash = StorageBlockValidator::fnv1a_hash(b"hello");
        assert_eq!(hash, 0xa430_d846_80aa_bd0b);
    }

    #[test]
    fn test_fnv1a_different_inputs_differ() {
        let h1 = StorageBlockValidator::fnv1a_hash(b"abc");
        let h2 = StorageBlockValidator::fnv1a_hash(b"abd");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_fnv1a_deterministic() {
        let data = b"some block data for hashing";
        let h1 = StorageBlockValidator::fnv1a_hash(data);
        let h2 = StorageBlockValidator::fnv1a_hash(data);
        assert_eq!(h1, h2);
    }

    // ----- validate_size -----

    #[test]
    fn test_validate_size_within_bounds() {
        assert!(StorageBlockValidator::validate_size(
            &[0u8; 100],
            Some(50),
            Some(200)
        ));
    }

    #[test]
    fn test_validate_size_too_small() {
        assert!(!StorageBlockValidator::validate_size(
            &[0u8; 10],
            Some(50),
            None
        ));
    }

    #[test]
    fn test_validate_size_too_large() {
        assert!(!StorageBlockValidator::validate_size(
            &[0u8; 300],
            None,
            Some(200)
        ));
    }

    #[test]
    fn test_validate_size_no_bounds() {
        assert!(StorageBlockValidator::validate_size(
            &[0u8; 999],
            None,
            None
        ));
    }

    #[test]
    fn test_validate_size_exact_min() {
        assert!(StorageBlockValidator::validate_size(
            &[0u8; 50],
            Some(50),
            None
        ));
    }

    #[test]
    fn test_validate_size_exact_max() {
        assert!(StorageBlockValidator::validate_size(
            &[0u8; 200],
            None,
            Some(200)
        ));
    }

    // ----- validate_prefix -----

    #[test]
    fn test_validate_prefix_match() {
        assert!(StorageBlockValidator::validate_prefix(
            &[0x89, 0x50, 0x4E, 0x47, 0xFF],
            &[0x89, 0x50, 0x4E, 0x47]
        ));
    }

    #[test]
    fn test_validate_prefix_mismatch() {
        assert!(!StorageBlockValidator::validate_prefix(
            &[0x00, 0x01, 0x02],
            &[0x89, 0x50]
        ));
    }

    #[test]
    fn test_validate_prefix_data_shorter_than_prefix() {
        assert!(!StorageBlockValidator::validate_prefix(
            &[0x89],
            &[0x89, 0x50]
        ));
    }

    #[test]
    fn test_validate_prefix_empty_prefix() {
        assert!(StorageBlockValidator::validate_prefix(b"anything", b""));
    }

    // ----- validate_hash -----

    #[test]
    fn test_validate_hash_correct() {
        let data = b"test block";
        let expected = StorageBlockValidator::fnv1a_hash(data);
        assert!(StorageBlockValidator::validate_hash(data, expected));
    }

    #[test]
    fn test_validate_hash_wrong() {
        assert!(!StorageBlockValidator::validate_hash(b"data", 0));
    }

    // ----- validate (full pipeline) -----

    #[test]
    fn test_validate_valid_block() {
        let data = b"valid block content";
        let hash = StorageBlockValidator::fnv1a_hash(data);
        let mut v = StorageBlockValidator::new();
        v.add_rule(ValidationRule {
            name: "basic".to_string(),
            max_size: Some(1024),
            min_size: Some(1),
            required_prefix: Some(b"valid".to_vec()),
            expected_hash: Some(hash),
        });
        let report = v.validate("Qm123", data);
        assert_eq!(report.result, ValidationResult::Valid);
        assert_eq!(report.checks_failed, 0);
        assert!(report.checks_passed >= 3);
        assert_eq!(report.block_cid, "Qm123");
    }

    #[test]
    fn test_validate_size_violation() {
        let mut v = StorageBlockValidator::new();
        v.add_rule(ValidationRule {
            name: "size-only".to_string(),
            max_size: Some(5),
            min_size: None,
            required_prefix: None,
            expected_hash: None,
        });
        let report = v.validate("cid1", b"too long data here");
        assert_eq!(report.result, ValidationResult::InvalidSize);
        assert_eq!(report.checks_failed, 1);
    }

    #[test]
    fn test_validate_hash_mismatch() {
        let mut v = StorageBlockValidator::new();
        v.add_rule(ValidationRule {
            name: "hash-check".to_string(),
            max_size: None,
            min_size: None,
            required_prefix: None,
            expected_hash: Some(12345),
        });
        let report = v.validate("cid2", b"some data");
        assert_eq!(report.result, ValidationResult::InvalidHash);
    }

    #[test]
    fn test_validate_prefix_failure() {
        let mut v = StorageBlockValidator::new();
        v.add_rule(ValidationRule {
            name: "magic".to_string(),
            max_size: None,
            min_size: None,
            required_prefix: Some(vec![0xFF, 0xFE]),
            expected_hash: None,
        });
        let report = v.validate("cid3", b"\x00\x01rest");
        assert_eq!(report.result, ValidationResult::InvalidFormat);
    }

    #[test]
    fn test_validate_corrupted_multiple_failures() {
        let mut v = StorageBlockValidator::new();
        v.add_rule(ValidationRule {
            name: "strict".to_string(),
            max_size: Some(5),
            min_size: None,
            required_prefix: Some(vec![0xAA]),
            expected_hash: Some(0),
        });
        // Data fails size, prefix, and hash.
        let report = v.validate("cid4", b"this is clearly wrong data");
        assert_eq!(report.result, ValidationResult::Corrupted);
        assert!(report.checks_failed > 1);
    }

    #[test]
    fn test_validate_no_rules_all_valid() {
        let mut v = StorageBlockValidator::new();
        let report = v.validate("cid5", b"anything");
        assert_eq!(report.result, ValidationResult::Valid);
    }

    #[test]
    fn test_validate_empty_data() {
        let mut v = StorageBlockValidator::new();
        v.add_rule(ValidationRule {
            name: "needs-content".to_string(),
            max_size: None,
            min_size: Some(1),
            required_prefix: None,
            expected_hash: None,
        });
        let report = v.validate("cid6", b"");
        assert_eq!(report.result, ValidationResult::InvalidSize);
    }

    #[test]
    fn test_validate_empty_data_no_rules() {
        let mut v = StorageBlockValidator::new();
        let report = v.validate("cid7", b"");
        assert_eq!(report.result, ValidationResult::Valid);
    }

    #[test]
    fn test_validate_multiple_rules_and_logic() {
        let data = b"\x89PNG rest of image";
        let hash = StorageBlockValidator::fnv1a_hash(data);
        let mut v = StorageBlockValidator::new();
        // Rule 1: size
        v.add_rule(ValidationRule {
            name: "size".to_string(),
            max_size: Some(1024),
            min_size: Some(1),
            required_prefix: None,
            expected_hash: None,
        });
        // Rule 2: format
        v.add_rule(ValidationRule {
            name: "format".to_string(),
            max_size: None,
            min_size: None,
            required_prefix: Some(b"\x89PNG".to_vec()),
            expected_hash: None,
        });
        // Rule 3: integrity
        v.add_rule(ValidationRule {
            name: "integrity".to_string(),
            max_size: None,
            min_size: None,
            required_prefix: None,
            expected_hash: Some(hash),
        });
        let report = v.validate("img-cid", data);
        assert_eq!(report.result, ValidationResult::Valid);
        assert_eq!(report.checks_passed, 3);
        assert_eq!(report.checks_failed, 0);
    }

    // ----- batch_validate -----

    #[test]
    fn test_batch_validate() {
        let mut v = StorageBlockValidator::new();
        v.add_rule(ValidationRule {
            name: "min-size".to_string(),
            max_size: None,
            min_size: Some(3),
            required_prefix: None,
            expected_hash: None,
        });
        let blocks: Vec<(&str, &[u8])> =
            vec![("ok1", b"abcdef"), ("ok2", b"xyz"), ("fail1", b"ab")];
        let reports = v.batch_validate(&blocks);
        assert_eq!(reports.len(), 3);
        assert_eq!(reports[0].result, ValidationResult::Valid);
        assert_eq!(reports[1].result, ValidationResult::Valid);
        assert_eq!(reports[2].result, ValidationResult::InvalidSize);
    }

    #[test]
    fn test_batch_validate_empty() {
        let mut v = StorageBlockValidator::new();
        let reports = v.batch_validate(&[]);
        assert!(reports.is_empty());
    }

    // ----- remove_rule -----

    #[test]
    fn test_remove_rule_exists() {
        let mut v = StorageBlockValidator::new();
        v.add_rule(rule("alpha"));
        v.add_rule(rule("beta"));
        assert!(v.remove_rule("alpha"));
        assert_eq!(v.rule_count(), 1);
    }

    #[test]
    fn test_remove_rule_not_found() {
        let mut v = StorageBlockValidator::new();
        v.add_rule(rule("alpha"));
        assert!(!v.remove_rule("gamma"));
        assert_eq!(v.rule_count(), 1);
    }

    // ----- stats -----

    #[test]
    fn test_stats_initial() {
        let v = StorageBlockValidator::new();
        let s = v.stats();
        assert_eq!(s.blocks_validated, 0);
        assert_eq!(s.blocks_valid, 0);
        assert_eq!(s.blocks_invalid, 0);
        assert!((s.validity_ratio - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_after_validations() {
        let mut v = StorageBlockValidator::new();
        v.add_rule(ValidationRule {
            name: "size".to_string(),
            max_size: Some(10),
            min_size: None,
            required_prefix: None,
            expected_hash: None,
        });
        let _ = v.validate("a", b"short");
        let _ = v.validate("b", b"this is way too long for the rule");
        let _ = v.validate("c", b"ok");
        let s = v.stats();
        assert_eq!(s.blocks_validated, 3);
        assert_eq!(s.blocks_valid, 2);
        assert_eq!(s.blocks_invalid, 1);
        assert!((s.validity_ratio - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(s.rules_count, 1);
    }

    #[test]
    fn test_stats_all_valid() {
        let mut v = StorageBlockValidator::new();
        // No rules => everything valid.
        let _ = v.validate("x", b"data");
        let s = v.stats();
        assert!((s.validity_ratio - 1.0).abs() < f64::EPSILON);
    }

    // ----- rule_count -----

    #[test]
    fn test_rule_count() {
        let mut v = StorageBlockValidator::new();
        assert_eq!(v.rule_count(), 0);
        v.add_rule(rule("a"));
        v.add_rule(rule("b"));
        assert_eq!(v.rule_count(), 2);
        v.remove_rule("a");
        assert_eq!(v.rule_count(), 1);
    }

    // ----- Default impl -----

    #[test]
    fn test_default() {
        let v = StorageBlockValidator::default();
        assert_eq!(v.rule_count(), 0);
        assert_eq!(v.stats().blocks_validated, 0);
    }

    // ----- details content -----

    #[test]
    fn test_details_contain_rule_names() {
        let data = b"hello";
        let hash = StorageBlockValidator::fnv1a_hash(data);
        let mut v = StorageBlockValidator::new();
        v.add_rule(ValidationRule {
            name: "my-rule".to_string(),
            max_size: Some(1024),
            min_size: None,
            required_prefix: None,
            expected_hash: Some(hash),
        });
        let report = v.validate("cid", data);
        assert!(report.details.iter().all(|d| d.contains("my-rule")));
    }

    // ----- edge: rule with no checks -----

    #[test]
    fn test_rule_with_no_checks_passes() {
        let mut v = StorageBlockValidator::new();
        // Rule exists but defines no actual checks.
        v.add_rule(rule("empty-rule"));
        let report = v.validate("cid", b"data");
        // No checks run so nothing fails.
        assert_eq!(report.result, ValidationResult::Valid);
        assert_eq!(report.checks_passed, 0);
        assert_eq!(report.checks_failed, 0);
    }
}
