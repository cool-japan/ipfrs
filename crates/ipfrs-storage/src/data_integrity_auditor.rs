//! Data Integrity Auditor — proactive continuous verification of stored blocks.
//!
//! Computes and re-computes checksums with multiple pure-Rust algorithms (CRC-32,
//! Adler-32, FNV-XOR-64, MultiCheck), detects silent corruption, and maintains a
//! bounded repair history.

use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Checksum algorithm selection
// ---------------------------------------------------------------------------

/// Algorithm used when computing a block checksum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChecksumAlgo {
    /// CRC-32 with polynomial 0xEDB88320 (Castagnoli / IEEE 802.3).
    Crc32,
    /// Adler-32 (RFC 1950).
    Adler32,
    /// FNV-1a 64-bit hash, result folded to 32 bits via XOR of the two halves.
    FnvXor64,
    /// CRC-32 XOR Adler-32 XOR FnvXor64 — paranoia mode.
    MultiCheck,
}

// ---------------------------------------------------------------------------
// Pure-Rust algorithm implementations
// ---------------------------------------------------------------------------

/// Pre-computed CRC-32 look-up table (polynomial 0xEDB88320).
const fn build_crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0usize;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
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

static CRC32_TABLE: [u32; 256] = build_crc32_table();

/// Compute CRC-32 (Castagnoli variant, polynomial 0xEDB88320).
pub fn compute_crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        let idx = ((crc ^ u32::from(byte)) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32_TABLE[idx];
    }
    crc ^ 0xFFFF_FFFF
}

/// Adler-32 modulus.
const ADLER32_MOD: u32 = 65521;

/// Compute Adler-32.
///
/// A = 1 + Σ bytes\[i\]  (mod 65521)
/// B = Σ A_i           (mod 65521)
/// result = (B << 16) | A
pub fn compute_adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + u32::from(byte)) % ADLER32_MOD;
        b = (b + a) % ADLER32_MOD;
    }
    (b << 16) | a
}

/// FNV-1a 64-bit offset basis and prime.
const FNV1A_64_OFFSET: u64 = 14_695_981_039_346_656_037;
const FNV1A_64_PRIME: u64 = 1_099_511_628_211;

/// Compute FNV-1a 64-bit hash, then fold to 32 bits via high32 XOR low32.
pub fn compute_fnv_xor64(data: &[u8]) -> u32 {
    let mut hash = FNV1A_64_OFFSET;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV1A_64_PRIME);
    }
    let high = (hash >> 32) as u32;
    let low = (hash & 0xFFFF_FFFF) as u32;
    high ^ low
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

/// Dispatch checksum computation to the correct algorithm.
pub fn compute_checksum(data: &[u8], algo: &ChecksumAlgo) -> u32 {
    match algo {
        ChecksumAlgo::Crc32 => compute_crc32(data),
        ChecksumAlgo::Adler32 => compute_adler32(data),
        ChecksumAlgo::FnvXor64 => compute_fnv_xor64(data),
        ChecksumAlgo::MultiCheck => {
            compute_crc32(data) ^ compute_adler32(data) ^ compute_fnv_xor64(data)
        }
    }
}

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Stored checksum record for a content block.
#[derive(Debug, Clone)]
pub struct BlockChecksum {
    /// Content identifier of the block.
    pub cid: String,
    /// Algorithm used to produce the checksum.
    pub algo: ChecksumAlgo,
    /// The checksum value.
    pub checksum: u32,
    /// Unix timestamp (seconds) when the checksum was recorded.
    pub computed_at: u64,
    /// Size of the block in bytes at registration time.
    pub block_size: usize,
}

/// Result of auditing a single block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditResult {
    /// Block data matches the stored checksum.
    Passed {
        /// Content identifier.
        cid: String,
        /// Verified checksum.
        checksum: u32,
    },
    /// Computed checksum differs from the stored one — corruption detected.
    Failed {
        /// Content identifier.
        cid: String,
        /// Checksum recorded at registration time.
        stored_checksum: u32,
        /// Checksum freshly computed from the supplied data.
        computed_checksum: u32,
    },
    /// No registered checksum for this CID — block was never registered.
    Missing {
        /// Content identifier.
        cid: String,
    },
}

/// Repair history entry.
#[derive(Debug, Clone)]
pub struct RepairRecord {
    /// Content identifier of the corrupted block.
    pub cid: String,
    /// Unix timestamp when corruption was detected.
    pub detected_at: u64,
    /// Unix timestamp when the repair was completed, `None` if still pending.
    pub repaired_at: Option<u64>,
    /// Name or URL of the repair source (e.g., peer ID, backup path).
    pub source: String,
    /// Whether the repair attempt succeeded.
    pub success: bool,
}

/// Configuration for a `DataIntegrityAuditor`.
#[derive(Debug, Clone)]
pub struct AuditConfig {
    /// Algorithm to use for checksum computation.
    pub algo: ChecksumAlgo,
    /// Maximum blocks to audit per `audit_batch` invocation.
    pub audit_batch_size: usize,
    /// Maximum number of repair records retained.
    pub max_repair_history: usize,
    /// When `true`, `audit_block` automatically schedules a repair on failure.
    pub auto_repair: bool,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            algo: ChecksumAlgo::MultiCheck,
            audit_batch_size: 100,
            max_repair_history: 1000,
            auto_repair: false,
        }
    }
}

/// Snapshot statistics exposed by the auditor.
#[derive(Debug, Clone)]
pub struct AuditorStats {
    /// Number of blocks currently registered.
    pub registered_blocks: usize,
    /// Cumulative number of audit checks performed.
    pub total_audited: u64,
    /// Cumulative number of checks that passed.
    pub total_passed: u64,
    /// Cumulative number of checks that failed.
    pub total_failed: u64,
    /// `total_passed / total_audited` (0.0 when nothing has been audited yet).
    pub integrity_rate: f64,
    /// Number of repair records with `repaired_at.is_none()`.
    pub pending_repairs: usize,
}

// ---------------------------------------------------------------------------
// DataIntegrityAuditor
// ---------------------------------------------------------------------------

/// Proactive data integrity auditing system.
///
/// Continuously verifies stored blocks using configurable checksum algorithms,
/// detects silent corruption, and tracks repair history.
#[derive(Debug)]
pub struct DataIntegrityAuditor {
    /// Configuration.
    pub config: AuditConfig,
    /// Map from CID → stored checksum record.
    pub checksums: HashMap<String, BlockChecksum>,
    /// Bounded ring of repair records (newest at the back).
    pub repair_history: VecDeque<RepairRecord>,
    /// Total audit operations performed.
    pub total_audited: u64,
    /// Total audits that produced a `Passed` result.
    pub total_passed: u64,
    /// Total audits that produced a `Failed` result.
    pub total_failed: u64,
}

impl DataIntegrityAuditor {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new auditor with the supplied configuration.
    pub fn new(config: AuditConfig) -> Self {
        Self {
            config,
            checksums: HashMap::new(),
            repair_history: VecDeque::new(),
            total_audited: 0,
            total_passed: 0,
            total_failed: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Checksum helpers (forwarding to free functions for testability)
    // -----------------------------------------------------------------------

    /// Compute a checksum over `data` using `algo`.
    pub fn compute_checksum(data: &[u8], algo: &ChecksumAlgo) -> u32 {
        compute_checksum(data, algo)
    }

    /// CRC-32 (Castagnoli, polynomial 0xEDB88320).
    pub fn compute_crc32(data: &[u8]) -> u32 {
        compute_crc32(data)
    }

    /// Adler-32.
    pub fn compute_adler32(data: &[u8]) -> u32 {
        compute_adler32(data)
    }

    /// FNV-1a 64-bit → 32-bit via XOR folding.
    pub fn compute_fnv_xor64(data: &[u8]) -> u32 {
        compute_fnv_xor64(data)
    }

    // -----------------------------------------------------------------------
    // Registration
    // -----------------------------------------------------------------------

    /// Compute the block's checksum and store it, overwriting any prior record.
    ///
    /// # Parameters
    /// - `cid`  — content identifier.
    /// - `data` — raw block bytes.
    /// - `now`  — current Unix timestamp (seconds).
    pub fn register_block(&mut self, cid: String, data: &[u8], now: u64) {
        let checksum = compute_checksum(data, &self.config.algo);
        let record = BlockChecksum {
            cid: cid.clone(),
            algo: self.config.algo,
            checksum,
            computed_at: now,
            block_size: data.len(),
        };
        self.checksums.insert(cid, record);
    }

    // -----------------------------------------------------------------------
    // Audit
    // -----------------------------------------------------------------------

    /// Audit a single block: recompute its checksum and compare with the stored value.
    ///
    /// Updates `total_audited`, `total_passed`, and `total_failed`.
    /// If `config.auto_repair` is `true` and the check fails, a repair is
    /// automatically scheduled (with `source = "auto"` and `detected_at = 0`).
    pub fn audit_block(&mut self, cid: &str, data: &[u8]) -> AuditResult {
        self.total_audited += 1;

        let stored = match self.checksums.get(cid) {
            Some(s) => s.clone(),
            None => {
                return AuditResult::Missing {
                    cid: cid.to_owned(),
                };
            }
        };

        let computed = compute_checksum(data, &stored.algo);

        if computed == stored.checksum {
            self.total_passed += 1;
            AuditResult::Passed {
                cid: cid.to_owned(),
                checksum: computed,
            }
        } else {
            self.total_failed += 1;
            if self.config.auto_repair {
                self.schedule_repair(cid.to_owned(), 0, "auto".to_owned());
            }
            AuditResult::Failed {
                cid: cid.to_owned(),
                stored_checksum: stored.checksum,
                computed_checksum: computed,
            }
        }
    }

    /// Audit a batch of blocks, respecting `config.audit_batch_size`.
    ///
    /// Processes at most `audit_batch_size` entries and returns one
    /// `AuditResult` per processed block.
    pub fn audit_batch(&mut self, blocks: &[(String, Vec<u8>)]) -> Vec<AuditResult> {
        let limit = self.config.audit_batch_size.min(blocks.len());
        let mut results = Vec::with_capacity(limit);
        for (cid, data) in &blocks[..limit] {
            let result = self.audit_block(cid, data);
            results.push(result);
        }
        results
    }

    // -----------------------------------------------------------------------
    // Repair tracking
    // -----------------------------------------------------------------------

    /// Record a repair request for `cid`.
    ///
    /// Pushes a new `RepairRecord` with `repaired_at = None` to the back of
    /// the history ring.  If the ring exceeds `max_repair_history` the oldest
    /// entry is evicted from the front.
    pub fn schedule_repair(&mut self, cid: String, now: u64, source: String) {
        let record = RepairRecord {
            cid,
            detected_at: now,
            repaired_at: None,
            source,
            success: false,
        };
        self.repair_history.push_back(record);
        // Evict oldest entries to stay within the configured limit.
        while self.repair_history.len() > self.config.max_repair_history {
            self.repair_history.pop_front();
        }
    }

    /// Mark the most-recently scheduled (still-pending) repair for `cid` as
    /// completed.
    ///
    /// Scans the history from newest to oldest.  Returns `true` if a pending
    /// record was found and updated, `false` otherwise.
    pub fn mark_repaired(&mut self, cid: &str, now: u64) -> bool {
        // Iterate from back (newest) to find the latest pending record.
        for record in self.repair_history.iter_mut().rev() {
            if record.cid == cid && record.repaired_at.is_none() {
                record.repaired_at = Some(now);
                record.success = true;
                return true;
            }
        }
        false
    }

    // -----------------------------------------------------------------------
    // Metrics
    // -----------------------------------------------------------------------

    /// Ratio of passed audits to total audits.  Returns `0.0` when nothing
    /// has been audited yet.
    pub fn integrity_rate(&self) -> f64 {
        let denom = self.total_audited.max(1) as f64;
        self.total_passed as f64 / denom
    }

    /// All repair records for which `repaired_at.is_none()`.
    pub fn pending_repairs(&self) -> Vec<&RepairRecord> {
        self.repair_history
            .iter()
            .filter(|r| r.repaired_at.is_none())
            .collect()
    }

    /// Snapshot of the current auditor statistics.
    pub fn auditor_stats(&self) -> AuditorStats {
        AuditorStats {
            registered_blocks: self.checksums.len(),
            total_audited: self.total_audited,
            total_passed: self.total_passed,
            total_failed: self.total_failed,
            integrity_rate: self.integrity_rate(),
            pending_repairs: self.pending_repairs().len(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::data_integrity_auditor::{
        compute_adler32, compute_checksum, compute_crc32, compute_fnv_xor64, AuditConfig,
        AuditResult, ChecksumAlgo, DataIntegrityAuditor,
    };

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn default_auditor() -> DataIntegrityAuditor {
        DataIntegrityAuditor::new(AuditConfig::default())
    }

    fn auditor_with_algo(algo: ChecksumAlgo) -> DataIntegrityAuditor {
        DataIntegrityAuditor::new(AuditConfig {
            algo,
            ..AuditConfig::default()
        })
    }

    // -----------------------------------------------------------------------
    // CRC-32
    // -----------------------------------------------------------------------

    #[test]
    fn crc32_empty() {
        // CRC-32 of empty slice is well-known: 0x00000000
        assert_eq!(compute_crc32(&[]), 0x0000_0000);
    }

    #[test]
    fn crc32_single_byte() {
        // Sanity: single byte must produce non-trivial output.
        let v = compute_crc32(&[0x61]); // b'a'
        assert_ne!(v, 0);
    }

    #[test]
    fn crc32_known_vector() {
        // CRC-32/ISO-HDLC ("123456789") = 0xCBF43926
        let data = b"123456789";
        assert_eq!(compute_crc32(data), 0xCBF4_3926);
    }

    #[test]
    fn crc32_deterministic() {
        let data = b"hello world";
        assert_eq!(compute_crc32(data), compute_crc32(data));
    }

    #[test]
    fn crc32_different_inputs_differ() {
        assert_ne!(compute_crc32(b"abc"), compute_crc32(b"abd"));
    }

    // -----------------------------------------------------------------------
    // Adler-32
    // -----------------------------------------------------------------------

    #[test]
    fn adler32_empty() {
        // Empty input: A=1, B=0  →  0x00000001
        assert_eq!(compute_adler32(&[]), 1);
    }

    #[test]
    fn adler32_known_vector() {
        // Adler-32("Wikipedia") = 0x11E60398
        let data = b"Wikipedia";
        assert_eq!(compute_adler32(data), 0x11E6_0398);
    }

    #[test]
    fn adler32_deterministic() {
        let data = b"data integrity";
        assert_eq!(compute_adler32(data), compute_adler32(data));
    }

    #[test]
    fn adler32_different_inputs_differ() {
        assert_ne!(compute_adler32(b"foo"), compute_adler32(b"bar"));
    }

    // -----------------------------------------------------------------------
    // FNV-XOR-64
    // -----------------------------------------------------------------------

    #[test]
    fn fnv_xor64_empty() {
        // FNV-1a of empty input equals the offset basis.
        // 14695981039346656037 → high = 0xCBF29CE4, low = 0x84222325
        // XOR  → 0x4FD0BFC1
        let v = compute_fnv_xor64(&[]);
        assert_eq!(v, 0x4FD0_BFC1);
    }

    #[test]
    fn fnv_xor64_deterministic() {
        let data = b"consistency check";
        assert_eq!(compute_fnv_xor64(data), compute_fnv_xor64(data));
    }

    #[test]
    fn fnv_xor64_different_inputs_differ() {
        assert_ne!(compute_fnv_xor64(b"alpha"), compute_fnv_xor64(b"beta"));
    }

    // -----------------------------------------------------------------------
    // MultiCheck
    // -----------------------------------------------------------------------

    #[test]
    fn multicheck_combines_three_algos() {
        let data = b"multi";
        let expected = compute_crc32(data) ^ compute_adler32(data) ^ compute_fnv_xor64(data);
        let got = compute_checksum(data, &ChecksumAlgo::MultiCheck);
        assert_eq!(got, expected);
    }

    #[test]
    fn multicheck_deterministic() {
        let data = b"reproducible";
        assert_eq!(
            compute_checksum(data, &ChecksumAlgo::MultiCheck),
            compute_checksum(data, &ChecksumAlgo::MultiCheck)
        );
    }

    // -----------------------------------------------------------------------
    // Checksum dispatcher
    // -----------------------------------------------------------------------

    #[test]
    fn dispatch_crc32() {
        let data = b"dispatch crc32";
        assert_eq!(
            compute_checksum(data, &ChecksumAlgo::Crc32),
            compute_crc32(data)
        );
    }

    #[test]
    fn dispatch_adler32() {
        let data = b"dispatch adler32";
        assert_eq!(
            compute_checksum(data, &ChecksumAlgo::Adler32),
            compute_adler32(data)
        );
    }

    #[test]
    fn dispatch_fnvxor64() {
        let data = b"dispatch fnv";
        assert_eq!(
            compute_checksum(data, &ChecksumAlgo::FnvXor64),
            compute_fnv_xor64(data)
        );
    }

    // -----------------------------------------------------------------------
    // AuditConfig defaults
    // -----------------------------------------------------------------------

    #[test]
    fn default_config_values() {
        let cfg = AuditConfig::default();
        assert_eq!(cfg.algo, ChecksumAlgo::MultiCheck);
        assert_eq!(cfg.audit_batch_size, 100);
        assert_eq!(cfg.max_repair_history, 1000);
        assert!(!cfg.auto_repair);
    }

    // -----------------------------------------------------------------------
    // register_block / audit_block
    // -----------------------------------------------------------------------

    #[test]
    fn register_and_audit_passes() {
        let mut aud = default_auditor();
        aud.register_block("cid1".to_owned(), b"hello", 1000);
        let result = aud.audit_block("cid1", b"hello");
        assert!(matches!(result, AuditResult::Passed { .. }));
    }

    #[test]
    fn audit_detects_corruption() {
        let mut aud = default_auditor();
        aud.register_block("cid2".to_owned(), b"original", 1000);
        let result = aud.audit_block("cid2", b"corrupted");
        assert!(matches!(result, AuditResult::Failed { .. }));
    }

    #[test]
    fn audit_missing_returns_missing() {
        let mut aud = default_auditor();
        let result = aud.audit_block("nonexistent", b"data");
        assert!(matches!(result, AuditResult::Missing { .. }));
    }

    #[test]
    fn register_block_overwrites_previous() {
        let mut aud = default_auditor();
        aud.register_block("cid3".to_owned(), b"v1", 1000);
        aud.register_block("cid3".to_owned(), b"v2", 2000);
        // Should pass with v2 data, fail with v1 data.
        assert!(matches!(
            aud.audit_block("cid3", b"v2"),
            AuditResult::Passed { .. }
        ));
        assert!(matches!(
            aud.audit_block("cid3", b"v1"),
            AuditResult::Failed { .. }
        ));
    }

    #[test]
    fn audit_increments_counters() {
        let mut aud = default_auditor();
        aud.register_block("cid4".to_owned(), b"data", 0);
        aud.audit_block("cid4", b"data");
        aud.audit_block("cid4", b"bad");
        assert_eq!(aud.total_audited, 2);
        assert_eq!(aud.total_passed, 1);
        assert_eq!(aud.total_failed, 1);
    }

    #[test]
    fn missing_audit_increments_total_only() {
        let mut aud = default_auditor();
        aud.audit_block("ghost", b"x");
        assert_eq!(aud.total_audited, 1);
        assert_eq!(aud.total_passed, 0);
        assert_eq!(aud.total_failed, 0);
    }

    // -----------------------------------------------------------------------
    // audit_batch
    // -----------------------------------------------------------------------

    #[test]
    fn audit_batch_respects_batch_size() {
        let mut aud = DataIntegrityAuditor::new(AuditConfig {
            audit_batch_size: 3,
            ..AuditConfig::default()
        });
        for i in 0u8..10 {
            let cid = format!("cid{i}");
            aud.register_block(cid, &[i], 0);
        }
        let blocks: Vec<(String, Vec<u8>)> =
            (0u8..10).map(|i| (format!("cid{i}"), vec![i])).collect();
        let results = aud.audit_batch(&blocks);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn audit_batch_all_pass() {
        let mut aud = default_auditor();
        for i in 0u8..5 {
            aud.register_block(format!("b{i}"), &[i, i + 1], 0);
        }
        let blocks: Vec<(String, Vec<u8>)> = (0u8..5)
            .map(|i| (format!("b{i}"), vec![i, i + 1]))
            .collect();
        let results = aud.audit_batch(&blocks);
        assert!(results
            .iter()
            .all(|r| matches!(r, AuditResult::Passed { .. })));
    }

    #[test]
    fn audit_batch_empty_input() {
        let mut aud = default_auditor();
        let results = aud.audit_batch(&[]);
        assert!(results.is_empty());
    }

    // -----------------------------------------------------------------------
    // schedule_repair / mark_repaired / pending_repairs
    // -----------------------------------------------------------------------

    #[test]
    fn schedule_repair_adds_record() {
        let mut aud = default_auditor();
        aud.schedule_repair("cid5".to_owned(), 100, "peer-A".to_owned());
        assert_eq!(aud.pending_repairs().len(), 1);
    }

    #[test]
    fn mark_repaired_succeeds() {
        let mut aud = default_auditor();
        aud.schedule_repair("cid6".to_owned(), 100, "peer-B".to_owned());
        let ok = aud.mark_repaired("cid6", 200);
        assert!(ok);
        assert_eq!(aud.pending_repairs().len(), 0);
    }

    #[test]
    fn mark_repaired_sets_timestamp() {
        let mut aud = default_auditor();
        aud.schedule_repair("cid7".to_owned(), 50, "src".to_owned());
        aud.mark_repaired("cid7", 999);
        let record = aud
            .repair_history
            .iter()
            .find(|r| r.cid == "cid7")
            .expect("record must exist");
        assert_eq!(record.repaired_at, Some(999));
        assert!(record.success);
    }

    #[test]
    fn mark_repaired_unknown_cid_returns_false() {
        let mut aud = default_auditor();
        assert!(!aud.mark_repaired("unknown", 0));
    }

    #[test]
    fn pending_repairs_filters_completed() {
        let mut aud = default_auditor();
        aud.schedule_repair("c1".to_owned(), 1, "s1".to_owned());
        aud.schedule_repair("c2".to_owned(), 2, "s2".to_owned());
        aud.mark_repaired("c1", 10);
        let pending = aud.pending_repairs();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].cid, "c2");
    }

    #[test]
    fn repair_history_evicts_oldest_when_full() {
        let mut aud = DataIntegrityAuditor::new(AuditConfig {
            max_repair_history: 3,
            ..AuditConfig::default()
        });
        for i in 0u64..5 {
            aud.schedule_repair(format!("c{i}"), i, "src".to_owned());
        }
        // Only the 3 most recent should remain.
        assert_eq!(aud.repair_history.len(), 3);
        // Oldest (c0, c1) must be evicted.
        let cids: Vec<&str> = aud.repair_history.iter().map(|r| r.cid.as_str()).collect();
        assert!(!cids.contains(&"c0"));
        assert!(!cids.contains(&"c1"));
        assert!(cids.contains(&"c4"));
    }

    #[test]
    fn mark_repaired_picks_most_recent_pending() {
        let mut aud = default_auditor();
        // Two records for same CID.
        aud.schedule_repair("dup".to_owned(), 1, "s1".to_owned());
        aud.schedule_repair("dup".to_owned(), 2, "s2".to_owned());
        // Mark repaired: should affect the second (newest) one.
        aud.mark_repaired("dup", 99);
        let pending = aud.pending_repairs();
        assert_eq!(pending.len(), 1);
        // The first (older) record should still be pending.
        assert_eq!(pending[0].detected_at, 1);
    }

    // -----------------------------------------------------------------------
    // integrity_rate
    // -----------------------------------------------------------------------

    #[test]
    fn integrity_rate_zero_when_nothing_audited() {
        let aud = default_auditor();
        // total_audited == 0 → denominator clamped to 1 → rate = 0/1 = 0.0
        assert_eq!(aud.integrity_rate(), 0.0);
    }

    #[test]
    fn integrity_rate_all_pass() {
        let mut aud = default_auditor();
        aud.register_block("cx".to_owned(), b"x", 0);
        aud.audit_block("cx", b"x");
        assert!((aud.integrity_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn integrity_rate_mixed() {
        let mut aud = default_auditor();
        aud.register_block("cy".to_owned(), b"y", 0);
        aud.audit_block("cy", b"y"); // pass
        aud.audit_block("cy", b"z"); // fail
        let rate = aud.integrity_rate();
        assert!((rate - 0.5).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // auditor_stats
    // -----------------------------------------------------------------------

    #[test]
    fn auditor_stats_reflects_state() {
        let mut aud = default_auditor();
        aud.register_block("s1".to_owned(), b"data", 0);
        aud.register_block("s2".to_owned(), b"more", 0);
        aud.audit_block("s1", b"data");
        aud.schedule_repair("s1".to_owned(), 0, "peer".to_owned());

        let stats = aud.auditor_stats();
        assert_eq!(stats.registered_blocks, 2);
        assert_eq!(stats.total_audited, 1);
        assert_eq!(stats.total_passed, 1);
        assert_eq!(stats.total_failed, 0);
        assert!((stats.integrity_rate - 1.0).abs() < f64::EPSILON);
        assert_eq!(stats.pending_repairs, 1);
    }

    // -----------------------------------------------------------------------
    // auto_repair
    // -----------------------------------------------------------------------

    #[test]
    fn auto_repair_schedules_on_failure() {
        let mut aud = DataIntegrityAuditor::new(AuditConfig {
            auto_repair: true,
            ..AuditConfig::default()
        });
        aud.register_block("ar1".to_owned(), b"good", 0);
        aud.audit_block("ar1", b"bad");
        assert_eq!(aud.pending_repairs().len(), 1);
    }

    #[test]
    fn auto_repair_does_not_schedule_on_pass() {
        let mut aud = DataIntegrityAuditor::new(AuditConfig {
            auto_repair: true,
            ..AuditConfig::default()
        });
        aud.register_block("ar2".to_owned(), b"ok", 0);
        aud.audit_block("ar2", b"ok");
        assert_eq!(aud.pending_repairs().len(), 0);
    }

    // -----------------------------------------------------------------------
    // Per-algorithm registration
    // -----------------------------------------------------------------------

    #[test]
    fn crc32_algo_round_trip() {
        let mut aud = auditor_with_algo(ChecksumAlgo::Crc32);
        aud.register_block("rc".to_owned(), b"crc check", 0);
        assert!(matches!(
            aud.audit_block("rc", b"crc check"),
            AuditResult::Passed { .. }
        ));
    }

    #[test]
    fn adler32_algo_round_trip() {
        let mut aud = auditor_with_algo(ChecksumAlgo::Adler32);
        aud.register_block("ra".to_owned(), b"adler check", 0);
        assert!(matches!(
            aud.audit_block("ra", b"adler check"),
            AuditResult::Passed { .. }
        ));
    }

    #[test]
    fn fnvxor64_algo_round_trip() {
        let mut aud = auditor_with_algo(ChecksumAlgo::FnvXor64);
        aud.register_block("rf".to_owned(), b"fnv check", 0);
        assert!(matches!(
            aud.audit_block("rf", b"fnv check"),
            AuditResult::Passed { .. }
        ));
    }

    #[test]
    fn multicheck_algo_round_trip() {
        let mut aud = auditor_with_algo(ChecksumAlgo::MultiCheck);
        aud.register_block("rm".to_owned(), b"multi check", 0);
        assert!(matches!(
            aud.audit_block("rm", b"multi check"),
            AuditResult::Passed { .. }
        ));
    }

    // -----------------------------------------------------------------------
    // Large block / binary data
    // -----------------------------------------------------------------------

    #[test]
    fn large_block_passes() {
        let mut aud = default_auditor();
        let data: Vec<u8> = (0u8..=255).cycle().take(65536).collect();
        aud.register_block("large".to_owned(), &data, 0);
        assert!(matches!(
            aud.audit_block("large", &data),
            AuditResult::Passed { .. }
        ));
    }

    #[test]
    fn single_bit_flip_detected() {
        let mut aud = default_auditor();
        let mut data = vec![0xABu8; 512];
        aud.register_block("flip".to_owned(), &data, 0);
        // Flip a single bit.
        data[256] ^= 0x01;
        assert!(matches!(
            aud.audit_block("flip", &data),
            AuditResult::Failed { .. }
        ));
    }

    #[test]
    fn zero_length_block_round_trip() {
        let mut aud = default_auditor();
        aud.register_block("empty_block".to_owned(), &[], 0);
        assert!(matches!(
            aud.audit_block("empty_block", &[]),
            AuditResult::Passed { .. }
        ));
    }

    // -----------------------------------------------------------------------
    // RepairRecord source field
    // -----------------------------------------------------------------------

    #[test]
    fn repair_record_stores_source() {
        let mut aud = default_auditor();
        aud.schedule_repair("src_test".to_owned(), 0, "peer-XYZ".to_owned());
        let pending = aud.pending_repairs();
        assert_eq!(pending[0].source, "peer-XYZ");
    }

    // -----------------------------------------------------------------------
    // AuditResult fields
    // -----------------------------------------------------------------------

    #[test]
    fn failed_result_contains_both_checksums() {
        let mut aud = default_auditor();
        aud.register_block("fc".to_owned(), b"original", 0);
        match aud.audit_block("fc", b"tampered") {
            AuditResult::Failed {
                stored_checksum,
                computed_checksum,
                ..
            } => {
                assert_ne!(stored_checksum, computed_checksum);
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn passed_result_contains_correct_checksum() {
        let mut aud = default_auditor();
        let data = b"pass me";
        aud.register_block("pm".to_owned(), data, 0);
        let expected = compute_checksum(data, &ChecksumAlgo::MultiCheck);
        match aud.audit_block("pm", data) {
            AuditResult::Passed { checksum, .. } => assert_eq!(checksum, expected),
            other => panic!("expected Passed, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Registered block count
    // -----------------------------------------------------------------------

    #[test]
    fn registered_block_count() {
        let mut aud = default_auditor();
        for i in 0u8..7 {
            aud.register_block(format!("blk{i}"), &[i], 0);
        }
        assert_eq!(aud.checksums.len(), 7);
    }

    // -----------------------------------------------------------------------
    // block_size stored in record
    // -----------------------------------------------------------------------

    #[test]
    fn block_checksum_records_size() {
        let mut aud = default_auditor();
        let data = b"size test";
        aud.register_block("sz".to_owned(), data, 42);
        let rec = aud.checksums.get("sz").expect("record must exist");
        assert_eq!(rec.block_size, data.len());
        assert_eq!(rec.computed_at, 42);
    }
}
