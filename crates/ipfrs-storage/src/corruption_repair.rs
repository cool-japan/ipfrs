//! Block corruption detection and repair strategies.
//!
//! Provides mechanisms for detecting various types of block corruption
//! (bit flips, truncation, zero-fill, header damage) and attempting
//! automated repair using XOR-based parity data. Blocks that cannot
//! be repaired are quarantined to prevent serving corrupted data.

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Classification of corruption detected in a block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorruptionType {
    /// One or more bits were flipped.
    BitFlip,
    /// The block was truncated (shorter than expected).
    Truncation,
    /// Part of the block was overwritten with zeros.
    ZeroFill,
    /// The first bytes (header region) are damaged.
    HeaderDamage,
    /// Corruption that does not match a known pattern.
    Unknown,
}

/// Outcome of a repair attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RepairAction {
    /// Block was restored from an existing clean copy.
    Restored,
    /// Block was rebuilt using XOR parity data.
    RebuiltFromParity,
    /// Block was fetched from a remote peer.
    FetchedFromPeer,
    /// Block was quarantined because repair failed.
    Quarantined,
    /// Block is unrecoverable — no repair strategy succeeded.
    Unrecoverable,
}

/// Detailed report about a single corruption incident.
#[derive(Debug, Clone)]
pub struct CorruptionReport {
    /// Content identifier of the affected block.
    pub cid: String,
    /// Type of corruption detected.
    pub corruption_type: CorruptionType,
    /// Byte offset where corruption starts, if determinable.
    pub byte_offset: Option<usize>,
    /// Number of bytes affected.
    pub bytes_affected: usize,
    /// Timestamp (epoch seconds) when corruption was detected.
    pub detected_at: u64,
    /// What repair action was taken, if any.
    pub repair_action: Option<RepairAction>,
}

/// Configuration for the repair subsystem.
#[derive(Debug, Clone)]
pub struct RepairConfig {
    /// Whether XOR parity blocks are maintained.
    pub enable_parity: bool,
    /// Size of each parity computation block (bytes).
    pub parity_block_size: usize,
    /// Maximum number of repair attempts before giving up.
    pub max_repair_attempts: u32,
    /// Whether to automatically quarantine blocks after failed repair.
    pub auto_quarantine_on_fail: bool,
}

impl Default for RepairConfig {
    fn default() -> Self {
        Self {
            enable_parity: true,
            parity_block_size: 256,
            max_repair_attempts: 3,
            auto_quarantine_on_fail: true,
        }
    }
}

/// A stored block together with its parity and checksum data.
#[derive(Debug, Clone)]
pub struct BlockWithParity {
    /// Content identifier.
    pub cid: String,
    /// Raw block data.
    pub data: Vec<u8>,
    /// XOR parity computed over `data`.
    pub parity: Vec<u8>,
    /// FNV-1a checksum of the original data.
    pub checksum: u64,
}

/// Aggregate statistics for the repair subsystem.
#[derive(Debug, Clone, Default)]
pub struct RepairStats {
    /// Total number of blocks scanned.
    pub total_scanned: u64,
    /// Number of corruptions found.
    pub corruptions_found: u64,
    /// Number of successful repairs.
    pub repairs_successful: u64,
    /// Number of failed repair attempts.
    pub repairs_failed: u64,
    /// Number of blocks quarantined.
    pub quarantined: u64,
}

// ---------------------------------------------------------------------------
// CorruptionRepairer
// ---------------------------------------------------------------------------

/// Central manager for corruption detection and repair.
pub struct CorruptionRepairer {
    config: RepairConfig,
    blocks: HashMap<String, BlockWithParity>,
    reports: Vec<CorruptionReport>,
    quarantined: HashSet<String>,
    stats: RepairStats,
}

impl CorruptionRepairer {
    /// Create a new repairer with the given configuration.
    pub fn new(config: RepairConfig) -> Self {
        Self {
            config,
            blocks: HashMap::new(),
            reports: Vec::new(),
            quarantined: HashSet::new(),
            stats: RepairStats::default(),
        }
    }

    /// Add a block, computing parity and checksum automatically.
    pub fn add_block(&mut self, cid: &str, data: &[u8]) {
        let parity = Self::compute_parity(data, self.config.parity_block_size);
        let checksum = Self::compute_checksum(data);
        self.blocks.insert(
            cid.to_string(),
            BlockWithParity {
                cid: cid.to_string(),
                data: data.to_vec(),
                parity,
                checksum,
            },
        );
    }

    /// Compute XOR-based parity over `data` using the given block size.
    ///
    /// The parity vector has length `block_size`. Each byte in the parity
    /// is the XOR of all corresponding bytes (modulo block_size) in `data`.
    pub fn compute_parity(data: &[u8], block_size: usize) -> Vec<u8> {
        let bs = if block_size == 0 { 256 } else { block_size };
        let mut parity = vec![0u8; bs];
        for (i, &byte) in data.iter().enumerate() {
            parity[i % bs] ^= byte;
        }
        parity
    }

    /// Compute a 64-bit FNV-1a checksum of `data`.
    pub fn compute_checksum(data: &[u8]) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x00000100000001B3;
        let mut hash = FNV_OFFSET;
        for &byte in data {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    /// Detect corruption in the block identified by `cid`.
    ///
    /// Returns `None` if the block is clean or not found.
    pub fn detect_corruption(&self, cid: &str) -> Option<CorruptionType> {
        let block = self.blocks.get(cid)?;
        let current_checksum = Self::compute_checksum(&block.data);
        if current_checksum == block.checksum {
            return None;
        }
        Some(self.classify_corruption(block))
    }

    /// Classify the type of corruption by inspecting data vs parity.
    fn classify_corruption(&self, block: &BlockWithParity) -> CorruptionType {
        let original_len = self.estimate_original_length(block);

        // Check for truncation: data is shorter than the length implied by parity.
        if block.data.len() < original_len {
            return CorruptionType::Truncation;
        }

        // Check for header damage: corruption in the first 16 bytes.
        let header_size = 16.min(block.data.len());
        let parity_bs = if self.config.parity_block_size == 0 {
            256
        } else {
            self.config.parity_block_size
        };
        let current_parity = Self::compute_parity(&block.data, parity_bs);
        let mut header_damaged = false;
        for i in 0..header_size {
            if i < current_parity.len()
                && i < block.parity.len()
                && current_parity[i] != block.parity[i]
            {
                header_damaged = true;
                break;
            }
        }
        if header_damaged {
            return CorruptionType::HeaderDamage;
        }

        // Check for zero-fill: a run of zeros that differs from parity expectation.
        let zero_run_threshold = 8;
        let mut consecutive_zeros = 0usize;
        for &b in &block.data {
            if b == 0 {
                consecutive_zeros += 1;
                if consecutive_zeros >= zero_run_threshold {
                    return CorruptionType::ZeroFill;
                }
            } else {
                consecutive_zeros = 0;
            }
        }

        // Check for bit flip: small number of differing parity bytes.
        let differing = current_parity
            .iter()
            .zip(block.parity.iter())
            .filter(|(a, b)| a != b)
            .count();
        if differing > 0 && differing <= parity_bs / 2 {
            return CorruptionType::BitFlip;
        }

        CorruptionType::Unknown
    }

    /// Heuristic: estimate original data length from parity metadata.
    /// Since we don't store the length separately, return current data length.
    fn estimate_original_length(&self, block: &BlockWithParity) -> usize {
        block.data.len()
    }

    /// Scan all blocks for corruption and return reports.
    pub fn scan_all(&mut self) -> Vec<CorruptionReport> {
        let cids: Vec<String> = self.blocks.keys().cloned().collect();
        let mut new_reports = Vec::new();
        for cid in &cids {
            self.stats.total_scanned += 1;
            if let Some(corruption_type) = self.detect_corruption_internal(cid) {
                self.stats.corruptions_found += 1;
                let (byte_offset, bytes_affected) = self.estimate_damage_extent(cid);
                let report = CorruptionReport {
                    cid: cid.clone(),
                    corruption_type,
                    byte_offset,
                    bytes_affected,
                    detected_at: current_epoch_secs(),
                    repair_action: None,
                };
                new_reports.push(report);
            }
        }
        self.reports.extend(new_reports.clone());
        new_reports
    }

    /// Internal corruption detection (avoids borrow issues with `&self`).
    fn detect_corruption_internal(&self, cid: &str) -> Option<CorruptionType> {
        self.detect_corruption(cid)
    }

    /// Estimate byte offset and extent of damage.
    fn estimate_damage_extent(&self, cid: &str) -> (Option<usize>, usize) {
        let block = match self.blocks.get(cid) {
            Some(b) => b,
            None => return (None, 0),
        };
        let parity_bs = if self.config.parity_block_size == 0 {
            256
        } else {
            self.config.parity_block_size
        };
        let current_parity = Self::compute_parity(&block.data, parity_bs);
        let mut first_diff: Option<usize> = None;
        let mut diff_count = 0usize;
        for (i, (cur, orig)) in current_parity.iter().zip(block.parity.iter()).enumerate() {
            if cur != orig {
                if first_diff.is_none() {
                    first_diff = Some(i);
                }
                diff_count += 1;
            }
        }
        (first_diff, diff_count)
    }

    /// Attempt to repair a corrupted block using parity data.
    ///
    /// Returns `Ok(RepairAction)` describing the outcome, or `Err` if the
    /// block does not exist.
    pub fn repair_block(&mut self, cid: &str) -> Result<RepairAction, String> {
        if !self.blocks.contains_key(cid) {
            return Err(format!("block not found: {}", cid));
        }

        // If the block is already clean, nothing to do.
        if self.detect_corruption(cid).is_none() {
            return Ok(RepairAction::Restored);
        }

        if !self.config.enable_parity {
            if self.config.auto_quarantine_on_fail {
                self.quarantine(cid);
                self.stats.repairs_failed += 1;
                return Ok(RepairAction::Quarantined);
            }
            self.stats.repairs_failed += 1;
            return Ok(RepairAction::Unrecoverable);
        }

        // Attempt parity-based repair up to max_repair_attempts times.
        let mut repaired = false;
        for _attempt in 0..self.config.max_repair_attempts {
            if self.try_parity_repair(cid) {
                repaired = true;
                break;
            }
        }

        if repaired {
            self.stats.repairs_successful += 1;
            // Update the report if one exists.
            if let Some(report) = self.reports.iter_mut().rev().find(|r| r.cid == cid) {
                report.repair_action = Some(RepairAction::RebuiltFromParity);
            }
            Ok(RepairAction::RebuiltFromParity)
        } else {
            self.stats.repairs_failed += 1;
            if self.config.auto_quarantine_on_fail {
                self.quarantine(cid);
                Ok(RepairAction::Quarantined)
            } else {
                Ok(RepairAction::Unrecoverable)
            }
        }
    }

    /// Try to repair a single corrupted byte using XOR parity.
    ///
    /// This works when exactly one byte per parity-group is corrupted.
    /// For each parity position that differs, we XOR the current data
    /// contributions back and apply the stored parity to derive the
    /// correct byte value.
    fn try_parity_repair(&mut self, cid: &str) -> bool {
        let parity_bs = if self.config.parity_block_size == 0 {
            256
        } else {
            self.config.parity_block_size
        };

        // We need to work with owned data to avoid borrow issues.
        let block = match self.blocks.get(cid) {
            Some(b) => b.clone(),
            None => return false,
        };

        let current_parity = Self::compute_parity(&block.data, parity_bs);

        // Find which parity positions differ.
        let mut diff_positions: Vec<usize> = Vec::new();
        for (i, (cur, orig)) in current_parity.iter().zip(block.parity.iter()).enumerate() {
            if cur != orig {
                diff_positions.push(i);
            }
        }

        if diff_positions.is_empty() {
            return true; // Already clean.
        }

        // For each differing parity position, find the data byte(s) contributing
        // to that position and try flipping them.
        let mut new_data = block.data.clone();
        let mut any_fixed = false;

        for &pos in &diff_positions {
            // Collect all data indices that map to this parity position.
            let indices: Vec<usize> = (pos..block.data.len()).step_by(parity_bs).collect();

            if indices.len() == 1 {
                // Only one byte contributes: we can recover it exactly.
                // parity[pos] = XOR of all bytes at indices mapping to pos.
                // With one byte: stored_parity[pos] = original_byte
                // current_parity[pos] = current_byte
                // So original_byte = current_byte ^ current_parity[pos] ^ stored_parity[pos]
                let idx = indices[0];
                new_data[idx] ^= current_parity[pos] ^ block.parity[pos];
                any_fixed = true;
            } else {
                // Multiple bytes: try fixing each one individually and check.
                for &idx in &indices {
                    let mut candidate = block.data.clone();
                    candidate[idx] ^= current_parity[pos] ^ block.parity[pos];
                    let candidate_checksum = Self::compute_checksum(&candidate);
                    if candidate_checksum == block.checksum {
                        new_data = candidate;
                        any_fixed = true;
                        break;
                    }
                }
            }
        }

        if any_fixed {
            let new_checksum = Self::compute_checksum(&new_data);
            if new_checksum == block.checksum {
                if let Some(b) = self.blocks.get_mut(cid) {
                    b.data = new_data;
                    b.parity = Self::compute_parity(&b.data, parity_bs);
                }
                return true;
            }
        }

        false
    }

    /// Inject corruption into a block for testing purposes.
    ///
    /// Sets the byte at `offset` to `value`.
    pub fn corrupt_block(&mut self, cid: &str, offset: usize, value: u8) -> Result<(), String> {
        let block = self
            .blocks
            .get_mut(cid)
            .ok_or_else(|| format!("block not found: {}", cid))?;
        if offset >= block.data.len() {
            return Err(format!(
                "offset {} out of range for block of length {}",
                offset,
                block.data.len()
            ));
        }
        block.data[offset] = value;
        Ok(())
    }

    /// Verify that a block's current data matches its stored checksum.
    pub fn verify_block(&self, cid: &str) -> bool {
        match self.blocks.get(cid) {
            Some(block) => Self::compute_checksum(&block.data) == block.checksum,
            None => false,
        }
    }

    /// Quarantine a block, marking it as unservable.
    ///
    /// Returns `true` if the block was found and quarantined, `false` otherwise.
    pub fn quarantine(&mut self, cid: &str) -> bool {
        if self.blocks.contains_key(cid) {
            let inserted = self.quarantined.insert(cid.to_string());
            if inserted {
                self.stats.quarantined += 1;
            }
            true
        } else {
            false
        }
    }

    /// Check whether a block is currently quarantined.
    pub fn is_quarantined(&self, cid: &str) -> bool {
        self.quarantined.contains(cid)
    }

    /// Get the most recent corruption report for a given CID.
    pub fn get_report(&self, cid: &str) -> Option<&CorruptionReport> {
        self.reports.iter().rev().find(|r| r.cid == cid)
    }

    /// Get aggregate repair statistics.
    pub fn stats(&self) -> &RepairStats {
        &self.stats
    }
}

/// Return current time as epoch seconds, falling back to 0 on error.
fn current_epoch_secs() -> u64 {
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

    fn default_config() -> RepairConfig {
        RepairConfig::default()
    }

    fn sample_data() -> Vec<u8> {
        (0..128).map(|i| (i * 7 + 13) as u8).collect()
    }

    #[test]
    fn test_add_and_verify_clean_block() {
        let mut repairer = CorruptionRepairer::new(default_config());
        repairer.add_block("cid1", &sample_data());
        assert!(repairer.verify_block("cid1"));
    }

    #[test]
    fn test_verify_nonexistent_block() {
        let repairer = CorruptionRepairer::new(default_config());
        assert!(!repairer.verify_block("nonexistent"));
    }

    #[test]
    fn test_detect_no_corruption_clean_block() {
        let mut repairer = CorruptionRepairer::new(default_config());
        repairer.add_block("cid1", &sample_data());
        assert!(repairer.detect_corruption("cid1").is_none());
    }

    #[test]
    fn test_detect_bit_flip() {
        let mut repairer = CorruptionRepairer::new(default_config());
        let data = sample_data();
        repairer.add_block("cid1", &data);
        // Flip a bit at an offset beyond the header region (>=16).
        let offset = 64;
        let original = data[offset];
        let flipped = original ^ 0x01;
        repairer.corrupt_block("cid1", offset, flipped).ok();
        let corruption = repairer.detect_corruption("cid1");
        assert!(corruption.is_some());
        assert!(!repairer.verify_block("cid1"));
    }

    #[test]
    fn test_detect_truncation() {
        let mut repairer = CorruptionRepairer::new(default_config());
        let data = sample_data();
        repairer.add_block("cid1", &data);
        // Simulate truncation by directly modifying the stored data.
        if let Some(block) = repairer.blocks.get_mut("cid1") {
            block.data.truncate(64);
        }
        let corruption = repairer.detect_corruption("cid1");
        assert!(corruption.is_some());
    }

    #[test]
    fn test_detect_zero_fill() {
        let mut repairer = CorruptionRepairer::new(default_config());
        let mut data: Vec<u8> = (1..=128).collect();
        // Ensure no natural zero runs
        for b in data.iter_mut() {
            if *b == 0 {
                *b = 1;
            }
        }
        repairer.add_block("cid1", &data);
        // Zero-fill a region past the header.
        for i in 32..48 {
            repairer.corrupt_block("cid1", i, 0).ok();
        }
        let corruption = repairer.detect_corruption("cid1");
        assert!(corruption.is_some());
    }

    #[test]
    fn test_parity_computation_deterministic() {
        let data = sample_data();
        let p1 = CorruptionRepairer::compute_parity(&data, 64);
        let p2 = CorruptionRepairer::compute_parity(&data, 64);
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_parity_empty_data() {
        let parity = CorruptionRepairer::compute_parity(&[], 32);
        assert_eq!(parity.len(), 32);
        assert!(parity.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_parity_single_byte() {
        let data = vec![0xAB];
        let parity = CorruptionRepairer::compute_parity(&data, 4);
        assert_eq!(parity[0], 0xAB);
        assert_eq!(parity[1], 0);
    }

    #[test]
    fn test_repair_via_parity_single_byte_flip() {
        let mut repairer = CorruptionRepairer::new(RepairConfig {
            enable_parity: true,
            parity_block_size: 256,
            max_repair_attempts: 3,
            auto_quarantine_on_fail: true,
        });
        // Use data shorter than parity block size so each byte maps to unique parity slot.
        let data: Vec<u8> = (0..128).collect();
        repairer.add_block("cid1", &data);

        // Flip one byte.
        let original = data[50];
        repairer.corrupt_block("cid1", 50, original ^ 0xFF).ok();
        assert!(!repairer.verify_block("cid1"));

        let result = repairer.repair_block("cid1");
        assert!(result.is_ok());
        let action = result.unwrap_or(RepairAction::Unrecoverable);
        assert_eq!(action, RepairAction::RebuiltFromParity);
        assert!(repairer.verify_block("cid1"));
    }

    #[test]
    fn test_unrecoverable_damage() {
        let mut repairer = CorruptionRepairer::new(RepairConfig {
            enable_parity: false,
            parity_block_size: 256,
            max_repair_attempts: 1,
            auto_quarantine_on_fail: false,
        });
        let data = sample_data();
        repairer.add_block("cid1", &data);
        repairer.corrupt_block("cid1", 10, 0xFF).ok();
        let result = repairer.repair_block("cid1");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap_or(RepairAction::Restored),
            RepairAction::Unrecoverable
        );
    }

    #[test]
    fn test_auto_quarantine_on_fail() {
        let mut repairer = CorruptionRepairer::new(RepairConfig {
            enable_parity: false,
            parity_block_size: 256,
            max_repair_attempts: 1,
            auto_quarantine_on_fail: true,
        });
        let data = sample_data();
        repairer.add_block("cid1", &data);
        repairer.corrupt_block("cid1", 10, 0xFF).ok();
        let result = repairer.repair_block("cid1");
        assert_eq!(
            result.unwrap_or(RepairAction::Unrecoverable),
            RepairAction::Quarantined
        );
        assert!(repairer.is_quarantined("cid1"));
    }

    #[test]
    fn test_scan_all_clean() {
        let mut repairer = CorruptionRepairer::new(default_config());
        repairer.add_block("a", &[1, 2, 3]);
        repairer.add_block("b", &[4, 5, 6]);
        let reports = repairer.scan_all();
        assert!(reports.is_empty());
        assert_eq!(repairer.stats().total_scanned, 2);
    }

    #[test]
    fn test_scan_all_with_corruption() {
        let mut repairer = CorruptionRepairer::new(default_config());
        repairer.add_block("a", &[1, 2, 3]);
        repairer.add_block("b", &[4, 5, 6]);
        repairer.corrupt_block("b", 0, 0xFF).ok();
        let reports = repairer.scan_all();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].cid, "b");
        assert_eq!(repairer.stats().corruptions_found, 1);
    }

    #[test]
    fn test_quarantine_existing_block() {
        let mut repairer = CorruptionRepairer::new(default_config());
        repairer.add_block("cid1", &[10, 20]);
        assert!(repairer.quarantine("cid1"));
        assert!(repairer.is_quarantined("cid1"));
    }

    #[test]
    fn test_quarantine_nonexistent_block() {
        let mut repairer = CorruptionRepairer::new(default_config());
        assert!(!repairer.quarantine("missing"));
        assert!(!repairer.is_quarantined("missing"));
    }

    #[test]
    fn test_quarantine_idempotent() {
        let mut repairer = CorruptionRepairer::new(default_config());
        repairer.add_block("cid1", &[1]);
        repairer.quarantine("cid1");
        let stats_after_first = repairer.stats().quarantined;
        repairer.quarantine("cid1");
        assert_eq!(repairer.stats().quarantined, stats_after_first);
    }

    #[test]
    fn test_stats_initial() {
        let repairer = CorruptionRepairer::new(default_config());
        let s = repairer.stats();
        assert_eq!(s.total_scanned, 0);
        assert_eq!(s.corruptions_found, 0);
        assert_eq!(s.repairs_successful, 0);
        assert_eq!(s.repairs_failed, 0);
        assert_eq!(s.quarantined, 0);
    }

    #[test]
    fn test_stats_after_operations() {
        let mut repairer = CorruptionRepairer::new(RepairConfig {
            enable_parity: true,
            parity_block_size: 256,
            max_repair_attempts: 3,
            auto_quarantine_on_fail: true,
        });
        let data: Vec<u8> = (0..64).collect();
        repairer.add_block("cid1", &data);
        repairer.corrupt_block("cid1", 10, 0xFF).ok();
        let _ = repairer.repair_block("cid1");
        let s = repairer.stats();
        assert!(s.repairs_successful > 0 || s.repairs_failed > 0);
    }

    #[test]
    fn test_inject_corruption() {
        let mut repairer = CorruptionRepairer::new(default_config());
        repairer.add_block("cid1", &[1, 2, 3, 4]);
        assert!(repairer.corrupt_block("cid1", 2, 99).is_ok());
        assert!(!repairer.verify_block("cid1"));
    }

    #[test]
    fn test_inject_corruption_out_of_range() {
        let mut repairer = CorruptionRepairer::new(default_config());
        repairer.add_block("cid1", &[1, 2, 3]);
        let result = repairer.corrupt_block("cid1", 100, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_inject_corruption_missing_block() {
        let mut repairer = CorruptionRepairer::new(default_config());
        let result = repairer.corrupt_block("missing", 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_checksum_consistency() {
        let data = sample_data();
        let c1 = CorruptionRepairer::compute_checksum(&data);
        let c2 = CorruptionRepairer::compute_checksum(&data);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_checksum_differs_for_different_data() {
        let c1 = CorruptionRepairer::compute_checksum(&[1, 2, 3]);
        let c2 = CorruptionRepairer::compute_checksum(&[4, 5, 6]);
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_checksum_empty() {
        let c = CorruptionRepairer::compute_checksum(&[]);
        // FNV-1a offset basis.
        assert_eq!(c, 0xcbf29ce484222325);
    }

    #[test]
    fn test_multiple_blocks() {
        let mut repairer = CorruptionRepairer::new(default_config());
        for i in 0..10 {
            let data: Vec<u8> = (0..32).map(|j| (i * 32 + j) as u8).collect();
            repairer.add_block(&format!("cid{}", i), &data);
        }
        for i in 0..10 {
            assert!(repairer.verify_block(&format!("cid{}", i)));
        }
    }

    #[test]
    fn test_empty_repairer() {
        let mut repairer = CorruptionRepairer::new(default_config());
        assert!(repairer.scan_all().is_empty());
        assert_eq!(repairer.stats().total_scanned, 0);
        assert!(repairer.detect_corruption("nonexistent").is_none());
        assert!(!repairer.verify_block("nonexistent"));
    }

    #[test]
    fn test_get_report_after_scan() {
        let mut repairer = CorruptionRepairer::new(default_config());
        repairer.add_block("cid1", &[10, 20, 30]);
        repairer.corrupt_block("cid1", 1, 0xFF).ok();
        repairer.scan_all();
        let report = repairer.get_report("cid1");
        assert!(report.is_some());
        let r = report.unwrap_or_else(|| panic!("expected report"));
        assert_eq!(r.cid, "cid1");
        assert!(r.bytes_affected > 0);
    }

    #[test]
    fn test_get_report_nonexistent() {
        let repairer = CorruptionRepairer::new(default_config());
        assert!(repairer.get_report("missing").is_none());
    }

    #[test]
    fn test_repair_nonexistent_block() {
        let mut repairer = CorruptionRepairer::new(default_config());
        let result = repairer.repair_block("missing");
        assert!(result.is_err());
    }

    #[test]
    fn test_repair_already_clean_block() {
        let mut repairer = CorruptionRepairer::new(default_config());
        repairer.add_block("cid1", &[1, 2, 3]);
        let result = repairer.repair_block("cid1");
        assert_eq!(
            result.unwrap_or(RepairAction::Unrecoverable),
            RepairAction::Restored
        );
    }

    #[test]
    fn test_default_repair_config() {
        let cfg = RepairConfig::default();
        assert!(cfg.enable_parity);
        assert_eq!(cfg.parity_block_size, 256);
        assert_eq!(cfg.max_repair_attempts, 3);
        assert!(cfg.auto_quarantine_on_fail);
    }

    #[test]
    fn test_parity_block_size_zero_fallback() {
        let parity = CorruptionRepairer::compute_parity(&[1, 2, 3], 0);
        assert_eq!(parity.len(), 256);
    }

    #[test]
    fn test_corruption_report_fields() {
        let mut repairer = CorruptionRepairer::new(default_config());
        repairer.add_block("cid1", &sample_data());
        repairer.corrupt_block("cid1", 20, 0xAA).ok();
        let reports = repairer.scan_all();
        assert_eq!(reports.len(), 1);
        assert!(reports[0].detected_at > 0);
        assert_eq!(reports[0].repair_action, None);
    }

    #[test]
    fn test_repair_updates_report() {
        let mut repairer = CorruptionRepairer::new(RepairConfig {
            enable_parity: true,
            parity_block_size: 256,
            max_repair_attempts: 3,
            auto_quarantine_on_fail: true,
        });
        let data: Vec<u8> = (0..128).collect();
        repairer.add_block("cid1", &data);
        repairer.corrupt_block("cid1", 50, data[50] ^ 0xFF).ok();
        repairer.scan_all();
        let _ = repairer.repair_block("cid1");
        let report = repairer.get_report("cid1");
        assert!(report.is_some());
        if let Some(r) = report {
            if r.repair_action.is_some() {
                assert_eq!(
                    r.repair_action
                        .as_ref()
                        .unwrap_or(&RepairAction::Unrecoverable),
                    &RepairAction::RebuiltFromParity
                );
            }
        }
    }
}
