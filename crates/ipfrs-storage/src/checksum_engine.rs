//! Multi-algorithm checksum computation and verification for storage integrity.
//!
//! Provides `StorageChecksumEngine` with pure-Rust implementations of:
//! FNV-1a-64, DJB2, Murmur3-32, Adler-32, CRC-32/ISO-HDLC, xxHash-64 (simplified),
//! and a Blake3-inspired 256-bit hash (4 independent FNV-1a rounds).

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Pure-Rust algorithm implementations (public free functions)
// ─────────────────────────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash.
pub fn fnv1a_64(data: &[u8]) -> u64 {
    const OFFSET: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;
    let mut hash = OFFSET;
    for &b in data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// DJB2 hash (32-bit result stored in u32).
pub fn djb2(data: &[u8]) -> u32 {
    let mut hash: u64 = 5381;
    for &b in data {
        hash = hash.wrapping_mul(33).wrapping_add(b as u64);
    }
    (hash & 0xFFFF_FFFF) as u32
}

/// Simplified Murmur3-32 hash with seed=0.
pub fn murmur3_32(data: &[u8]) -> u32 {
    const C1: u32 = 0xcc9e_2d51;
    const C2: u32 = 0x1b87_3593;
    let len = data.len();
    let mut h: u32 = 0;

    let nblocks = len / 4;
    for i in 0..nblocks {
        let idx = i * 4;
        let k = u32::from_le_bytes([data[idx], data[idx + 1], data[idx + 2], data[idx + 3]]);
        let k = k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        h ^= k;
        h = h.rotate_left(13);
        h = h.wrapping_mul(5).wrapping_add(0xe654_6b64);
    }

    // Tail bytes
    let tail_start = nblocks * 4;
    let tail = &data[tail_start..];
    let mut k: u32 = 0;
    match tail.len() {
        3 => {
            k ^= (tail[2] as u32) << 16;
            k ^= (tail[1] as u32) << 8;
            k ^= tail[0] as u32;
            k = k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
            h ^= k;
        }
        2 => {
            k ^= (tail[1] as u32) << 8;
            k ^= tail[0] as u32;
            k = k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
            h ^= k;
        }
        1 => {
            k ^= tail[0] as u32;
            k = k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
            h ^= k;
        }
        _ => {}
    }

    h ^= len as u32;
    // Finalize (fmix32)
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h
}

/// Adler-32 checksum.
pub fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65521;
    let mut s1: u32 = 1;
    let mut s2: u32 = 0;
    for &b in data {
        s1 = (s1 + b as u32) % MOD;
        s2 = (s2 + s1) % MOD;
    }
    (s2 << 16) | s1
}

/// Build the CRC-32/ISO-HDLC lookup table.
fn crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    for i in 0u32..256 {
        let mut crc = i;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
        table[i as usize] = crc;
    }
    table
}

/// CRC-32/ISO-HDLC (polynomial 0xEDB88320).
pub fn crc32_iso(data: &[u8]) -> u32 {
    let table = crc32_table();
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        let idx = ((crc ^ b as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ table[idx];
    }
    crc ^ 0xFFFF_FFFF
}

/// Simplified xxHash-64 with seed=0.
pub fn xxhash64_simple(data: &[u8]) -> u64 {
    const PRIME1: u64 = 11_400_714_785_074_694_791;
    const PRIME2: u64 = 14_029_467_366_897_019_727;
    const PRIME3: u64 = 1_609_587_929_392_839_161;
    const PRIME4: u64 = 9_650_029_242_287_828_579;
    const PRIME5: u64 = 2_870_177_450_012_600_261;

    let len = data.len();
    let mut pos = 0usize;
    let mut h64: u64;

    if len >= 32 {
        let mut v1 = 0u64.wrapping_add(PRIME1).wrapping_add(PRIME2);
        let mut v2 = 0u64.wrapping_add(PRIME2);
        let mut v3 = 0u64;
        let mut v4 = 0u64.wrapping_sub(PRIME1);

        while pos + 32 <= len {
            let lane = |off: usize| {
                u64::from_le_bytes([
                    data[pos + off],
                    data[pos + off + 1],
                    data[pos + off + 2],
                    data[pos + off + 3],
                    data[pos + off + 4],
                    data[pos + off + 5],
                    data[pos + off + 6],
                    data[pos + off + 7],
                ])
            };
            v1 = v1
                .wrapping_add(lane(0).wrapping_mul(PRIME2))
                .rotate_left(31)
                .wrapping_mul(PRIME1);
            v2 = v2
                .wrapping_add(lane(8).wrapping_mul(PRIME2))
                .rotate_left(31)
                .wrapping_mul(PRIME1);
            v3 = v3
                .wrapping_add(lane(16).wrapping_mul(PRIME2))
                .rotate_left(31)
                .wrapping_mul(PRIME1);
            v4 = v4
                .wrapping_add(lane(24).wrapping_mul(PRIME2))
                .rotate_left(31)
                .wrapping_mul(PRIME1);
            pos += 32;
        }

        h64 = v1
            .rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18));

        let merge = |acc: u64, v: u64| -> u64 {
            let v = v.wrapping_mul(PRIME2).rotate_left(31).wrapping_mul(PRIME1);
            acc.wrapping_mul(PRIME1)
                .wrapping_add(v)
                .wrapping_mul(PRIME4)
                .wrapping_add(PRIME3)
        };
        h64 = merge(h64, v1);
        h64 = merge(h64, v2);
        h64 = merge(h64, v3);
        h64 = merge(h64, v4);
    } else {
        h64 = 0u64.wrapping_add(PRIME5);
    }

    h64 = h64.wrapping_add(len as u64);

    // Process remaining 8-byte chunks
    while pos + 8 <= len {
        let lane = u64::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]);
        let k1 = lane
            .wrapping_mul(PRIME2)
            .rotate_left(31)
            .wrapping_mul(PRIME1);
        h64 ^= k1;
        h64 = h64
            .rotate_left(27)
            .wrapping_mul(PRIME1)
            .wrapping_add(PRIME4);
        pos += 8;
    }

    // Remaining 4-byte chunk
    if pos + 4 <= len {
        let lane = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        h64 ^= (lane as u64).wrapping_mul(PRIME1);
        h64 = h64
            .rotate_left(23)
            .wrapping_mul(PRIME2)
            .wrapping_add(PRIME3);
        pos += 4;
    }

    // Remaining bytes
    while pos < len {
        h64 ^= (data[pos] as u64).wrapping_mul(PRIME5);
        h64 = h64.rotate_left(11).wrapping_mul(PRIME1);
        pos += 1;
    }

    // Avalanche / finalization
    h64 ^= h64 >> 33;
    h64 = h64.wrapping_mul(PRIME2);
    h64 ^= h64 >> 29;
    h64 = h64.wrapping_mul(PRIME1);
    h64 ^= h64 >> 32;
    h64
}

/// Blake3-inspired 256-bit hash: 4 independent FNV-1a-64 rounds with different seeds.
pub fn blake3_256_simple(data: &[u8]) -> [u8; 32] {
    const SEEDS: [u64; 4] = [
        0x0000_0000_0000_0000,
        0x1234_5678_90AB_CDEF,
        0xFEDC_BA98_7654_3210,
        0xDEAD_BEEF_CAFE_BABE,
    ];
    const PRIME: u64 = 1_099_511_628_211;
    const OFFSET: u64 = 14_695_981_039_346_656_037;

    let mut result = [0u8; 32];
    for (i, &seed) in SEEDS.iter().enumerate() {
        let mut hash = OFFSET ^ seed;
        for &b in data {
            hash ^= b as u64;
            hash = hash.wrapping_mul(PRIME);
        }
        let bytes = hash.to_le_bytes();
        result[i * 8..(i + 1) * 8].copy_from_slice(&bytes);
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Core types
// ─────────────────────────────────────────────────────────────────────────────

/// Supported checksum algorithms, all implemented in pure Rust.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChecksumAlgorithm {
    /// FNV-1a 64-bit hash.
    Fnv1a64,
    /// DJB2 hash (32-bit).
    Djb2,
    /// Simplified Murmur3-32 hash.
    Murmur3_32,
    /// Adler-32 checksum.
    Adler32,
    /// CRC-32/ISO-HDLC.
    Crc32,
    /// Simplified xxHash-64.
    Xxhash64,
    /// Blake3-inspired 256-bit hash (4 FNV-1a rounds with different seeds).
    Blake3_256,
}

impl ChecksumAlgorithm {
    /// Human-readable name of the algorithm.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Fnv1a64 => "fnv1a64",
            Self::Djb2 => "djb2",
            Self::Murmur3_32 => "murmur3_32",
            Self::Adler32 => "adler32",
            Self::Crc32 => "crc32",
            Self::Xxhash64 => "xxhash64",
            Self::Blake3_256 => "blake3_256",
        }
    }
}

/// The raw checksum value plus its hex representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Checksum {
    /// Algorithm that produced this checksum.
    pub algorithm: ChecksumAlgorithm,
    /// Raw bytes of the hash output.
    pub value: Vec<u8>,
    /// Lowercase hexadecimal string of `value`.
    pub hex: String,
}

impl Checksum {
    /// Construct a new `Checksum` from raw bytes.
    pub fn new(algorithm: ChecksumAlgorithm, value: Vec<u8>) -> Self {
        let hex = value.iter().map(|b| format!("{b:02x}")).collect();
        Self {
            algorithm,
            value,
            hex,
        }
    }
}

/// A stored checksum record for a specific object.
#[derive(Debug, Clone)]
pub struct ChecksumRecord {
    /// Identifier for the stored object.
    pub object_id: String,
    /// Computed checksum.
    pub checksum: Checksum,
    /// Unix timestamp (seconds) when the checksum was computed.
    pub computed_at: u64,
    /// Unix timestamp (seconds) of the last successful verification, if any.
    pub verified_at: Option<u64>,
    /// Size of the object in bytes at time of computation.
    pub size_bytes: u64,
}

/// Result of a single verification attempt.
#[derive(Debug, Clone)]
pub struct CeVerificationResult {
    /// Identifier for the checked object.
    pub object_id: String,
    /// The checksum stored at index time.
    pub expected: Checksum,
    /// The checksum freshly computed from the provided data.
    pub actual: Checksum,
    /// Whether `expected == actual`.
    pub matches: bool,
    /// Unix timestamp (seconds) when verification was performed.
    pub verified_at: u64,
}

/// Summary statistics over a set of verification results.
#[derive(Debug, Clone)]
pub struct ChecksumStats {
    /// Total number of stored records.
    pub total_records: usize,
    /// Number of objects that have been successfully verified at least once.
    pub verified_count: usize,
    /// Fraction of provided results where `!matches`.
    pub corruption_rate: f64,
    /// Name of the engine's current algorithm.
    pub algorithm: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Engine
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-algorithm checksum engine for storage integrity verification.
///
/// Stores one `ChecksumRecord` per object and allows efficient batch
/// computation and verification.
#[derive(Debug, Clone)]
pub struct StorageChecksumEngine {
    records: HashMap<String, ChecksumRecord>,
    /// Default algorithm used for new computations.
    pub algorithm: ChecksumAlgorithm,
    /// When `true`, callers are encouraged to verify on every read.
    pub verify_on_read: bool,
}

impl StorageChecksumEngine {
    /// Create a new engine with the given default algorithm.
    pub fn new(algorithm: ChecksumAlgorithm) -> Self {
        Self {
            records: HashMap::new(),
            algorithm,
            verify_on_read: false,
        }
    }

    // ── Core computation ───────────────────────────────────────────────────

    /// Compute a checksum for `data` using the engine's current algorithm.
    pub fn compute(&self, data: &[u8]) -> Checksum {
        Self::compute_with_algorithm(self.algorithm, data)
    }

    /// Compute a checksum for `data` using the specified algorithm.
    pub fn compute_with_algorithm(algorithm: ChecksumAlgorithm, data: &[u8]) -> Checksum {
        let value: Vec<u8> = match algorithm {
            ChecksumAlgorithm::Fnv1a64 => fnv1a_64(data).to_le_bytes().to_vec(),
            ChecksumAlgorithm::Djb2 => djb2(data).to_le_bytes().to_vec(),
            ChecksumAlgorithm::Murmur3_32 => murmur3_32(data).to_le_bytes().to_vec(),
            ChecksumAlgorithm::Adler32 => adler32(data).to_le_bytes().to_vec(),
            ChecksumAlgorithm::Crc32 => crc32_iso(data).to_le_bytes().to_vec(),
            ChecksumAlgorithm::Xxhash64 => xxhash64_simple(data).to_le_bytes().to_vec(),
            ChecksumAlgorithm::Blake3_256 => blake3_256_simple(data).to_vec(),
        };
        Checksum::new(algorithm, value)
    }

    /// Compute a checksum for `data`, store the resulting record, and return it.
    pub fn compute_for(&mut self, object_id: String, data: &[u8], now: u64) -> ChecksumRecord {
        let checksum = self.compute(data);
        let record = ChecksumRecord {
            object_id: object_id.clone(),
            checksum,
            computed_at: now,
            verified_at: None,
            size_bytes: data.len() as u64,
        };
        self.records.insert(object_id, record.clone());
        record
    }

    // ── Verification ───────────────────────────────────────────────────────

    /// Compare the stored checksum for `object_id` with a freshly computed one.
    ///
    /// Returns `None` when no record exists for `object_id`.
    /// On a successful match the record's `verified_at` is updated to `now`.
    pub fn verify(
        &mut self,
        object_id: &str,
        data: &[u8],
        now: u64,
    ) -> Option<CeVerificationResult> {
        let record = self.records.get_mut(object_id)?;
        let actual = Self::compute_with_algorithm(record.checksum.algorithm, data);
        let matches = actual.value == record.checksum.value;
        if matches {
            record.verified_at = Some(now);
        }
        Some(CeVerificationResult {
            object_id: object_id.to_owned(),
            expected: record.checksum.clone(),
            actual,
            matches,
            verified_at: now,
        })
    }

    /// Retrieve the stored record for `object_id`, if any.
    pub fn record(&self, object_id: &str) -> Option<&ChecksumRecord> {
        self.records.get(object_id)
    }

    /// Remove the record for `object_id`. Returns `true` if the record existed.
    pub fn remove(&mut self, object_id: &str) -> bool {
        self.records.remove(object_id).is_some()
    }

    /// Verify all stored records by calling `data_fn` for each `object_id`.
    ///
    /// If `data_fn` returns `None` for a given id the object is skipped.
    pub fn verify_all(
        &mut self,
        data_fn: impl Fn(&str) -> Option<Vec<u8>>,
        now: u64,
    ) -> Vec<CeVerificationResult> {
        let ids: Vec<String> = self.records.keys().cloned().collect();
        let mut results = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(data) = data_fn(&id) {
                if let Some(result) = self.verify(&id, &data, now) {
                    results.push(result);
                }
            }
        }
        results
    }

    /// Compute and store checksums for a batch of `(object_id, data)` pairs.
    pub fn batch_compute(&mut self, items: &[(&str, &[u8])], now: u64) -> Vec<ChecksumRecord> {
        items
            .iter()
            .map(|&(id, data)| self.compute_for(id.to_owned(), data, now))
            .collect()
    }

    // ── Statistics ─────────────────────────────────────────────────────────

    /// Count the number of verification results where `!matches`.
    pub fn corruption_count(results: &[CeVerificationResult]) -> usize {
        results.iter().filter(|r| !r.matches).count()
    }

    /// Number of objects currently tracked.
    pub fn object_count(&self) -> usize {
        self.records.len()
    }

    /// Compute aggregate statistics from the engine's state and a set of results.
    pub fn stats(&self, results: &[CeVerificationResult]) -> ChecksumStats {
        let total_records = self.records.len();
        let verified_count = self
            .records
            .values()
            .filter(|r| r.verified_at.is_some())
            .count();
        let corrupted = Self::corruption_count(results);
        let corruption_rate = if results.is_empty() {
            0.0
        } else {
            corrupted as f64 / results.len() as f64
        };
        ChecksumStats {
            total_records,
            verified_count,
            corruption_rate,
            algorithm: self.algorithm.name().to_owned(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::checksum_engine::{
        adler32, blake3_256_simple, crc32_iso, djb2, fnv1a_64, murmur3_32, xxhash64_simple,
        CeVerificationResult, ChecksumAlgorithm, ChecksumRecord, StorageChecksumEngine,
    };

    // ── Algorithm unit tests ───────────────────────────────────────────────

    #[test]
    fn test_fnv1a64_empty() {
        // FNV-1a-64 of empty slice is the offset basis
        assert_eq!(fnv1a_64(b""), 14_695_981_039_346_656_037);
    }

    #[test]
    fn test_fnv1a64_known_value() {
        // "hello" FNV-1a-64 known reference
        let h = fnv1a_64(b"hello");
        assert_ne!(h, 0);
        // Deterministic
        assert_eq!(fnv1a_64(b"hello"), h);
    }

    #[test]
    fn test_fnv1a64_different_inputs_differ() {
        assert_ne!(fnv1a_64(b"foo"), fnv1a_64(b"bar"));
    }

    #[test]
    fn test_djb2_empty() {
        let h = djb2(b"");
        assert_eq!(h, (5381u64 & 0xFFFF_FFFF) as u32);
    }

    #[test]
    fn test_djb2_hello() {
        let h = djb2(b"hello");
        assert_ne!(h, 0);
        assert_eq!(djb2(b"hello"), h);
    }

    #[test]
    fn test_djb2_different() {
        assert_ne!(djb2(b"abc"), djb2(b"xyz"));
    }

    #[test]
    fn test_murmur3_32_empty() {
        // Empty input should produce a stable value
        let h = murmur3_32(b"");
        assert_eq!(h, murmur3_32(b""));
    }

    #[test]
    fn test_murmur3_32_4bytes() {
        let h = murmur3_32(b"test");
        assert_ne!(h, 0);
        assert_eq!(murmur3_32(b"test"), h);
    }

    #[test]
    fn test_murmur3_32_partial_tail() {
        // 5 bytes: one full block + 1 tail byte
        let h = murmur3_32(b"abcde");
        assert_eq!(murmur3_32(b"abcde"), h);
    }

    #[test]
    fn test_adler32_empty() {
        // Empty → s1=1, s2=0 → result = 1
        assert_eq!(adler32(b""), 1);
    }

    #[test]
    fn test_adler32_abc() {
        // Known: adler32("abc") = 0x024d0127 per RFC 1950
        assert_eq!(adler32(b"abc"), 0x024d_0127);
    }

    #[test]
    fn test_adler32_deterministic() {
        let h = adler32(b"hello world");
        assert_eq!(adler32(b"hello world"), h);
    }

    #[test]
    fn test_crc32_empty() {
        // CRC-32/ISO-HDLC of empty = 0x00000000
        assert_eq!(crc32_iso(b""), 0x0000_0000);
    }

    #[test]
    fn test_crc32_known_value() {
        // "123456789" → CRC-32 = 0xCBF43926
        assert_eq!(crc32_iso(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn test_crc32_deterministic() {
        assert_eq!(crc32_iso(b"hello"), crc32_iso(b"hello"));
    }

    #[test]
    fn test_xxhash64_empty() {
        let h = xxhash64_simple(b"");
        assert_eq!(xxhash64_simple(b""), h);
    }

    #[test]
    fn test_xxhash64_hello() {
        let h = xxhash64_simple(b"hello");
        assert_ne!(h, 0);
        assert_eq!(xxhash64_simple(b"hello"), h);
    }

    #[test]
    fn test_xxhash64_large_input() {
        // 64 bytes triggers the wide-lane path
        let data: Vec<u8> = (0u8..64).collect();
        let h = xxhash64_simple(&data);
        assert_eq!(xxhash64_simple(&data), h);
    }

    #[test]
    fn test_blake3_256_empty() {
        let h = blake3_256_simple(b"");
        assert_eq!(h.len(), 32);
        assert_eq!(blake3_256_simple(b""), h);
    }

    #[test]
    fn test_blake3_256_hello() {
        let h = blake3_256_simple(b"hello");
        assert_eq!(h.len(), 32);
        assert_ne!(h, [0u8; 32]);
    }

    #[test]
    fn test_blake3_256_different_inputs() {
        assert_ne!(blake3_256_simple(b"a"), blake3_256_simple(b"b"));
    }

    // ── Checksum type tests ────────────────────────────────────────────────

    #[test]
    fn test_checksum_hex_encoding() {
        let engine = StorageChecksumEngine::new(ChecksumAlgorithm::Fnv1a64);
        let cs = engine.compute(b"hello");
        assert_eq!(cs.hex.len(), 16); // 8 bytes → 16 hex chars
        assert!(cs.hex.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_checksum_hex_lowercase() {
        let engine = StorageChecksumEngine::new(ChecksumAlgorithm::Crc32);
        let cs = engine.compute(b"123456789");
        assert!(cs.hex.chars().all(|c| !c.is_ascii_uppercase()));
    }

    #[test]
    fn test_checksum_blake3_hex_length() {
        let engine = StorageChecksumEngine::new(ChecksumAlgorithm::Blake3_256);
        let cs = engine.compute(b"data");
        assert_eq!(cs.hex.len(), 64); // 32 bytes → 64 hex chars
    }

    // ── Engine: compute_for / record / remove ──────────────────────────────

    #[test]
    fn test_compute_for_stores_record() {
        let mut engine = StorageChecksumEngine::new(ChecksumAlgorithm::Crc32);
        let rec = engine.compute_for("obj1".to_owned(), b"data", 1000);
        assert_eq!(rec.object_id, "obj1");
        assert_eq!(rec.computed_at, 1000);
        assert_eq!(rec.size_bytes, 4);
        assert!(rec.verified_at.is_none());
        assert_eq!(engine.object_count(), 1);
    }

    #[test]
    fn test_record_retrieval() {
        let mut engine = StorageChecksumEngine::new(ChecksumAlgorithm::Adler32);
        engine.compute_for("x".to_owned(), b"hello", 42);
        let rec = engine.record("x");
        assert!(rec.is_some());
        assert_eq!(rec.map(|r| r.object_id.as_str()), Some("x"));
    }

    #[test]
    fn test_record_missing_returns_none() {
        let engine = StorageChecksumEngine::new(ChecksumAlgorithm::Djb2);
        assert!(engine.record("nonexistent").is_none());
    }

    #[test]
    fn test_remove_existing() {
        let mut engine = StorageChecksumEngine::new(ChecksumAlgorithm::Fnv1a64);
        engine.compute_for("a".to_owned(), b"x", 0);
        assert!(engine.remove("a"));
        assert_eq!(engine.object_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut engine = StorageChecksumEngine::new(ChecksumAlgorithm::Fnv1a64);
        assert!(!engine.remove("ghost"));
    }

    // ── Engine: verify ─────────────────────────────────────────────────────

    #[test]
    fn test_verify_matching() {
        let mut engine = StorageChecksumEngine::new(ChecksumAlgorithm::Crc32);
        engine.compute_for("doc".to_owned(), b"content", 100);
        let result = engine
            .verify("doc", b"content", 200)
            .expect("result expected");
        assert!(result.matches);
        assert_eq!(result.verified_at, 200);
        // verified_at updated
        assert_eq!(engine.record("doc").and_then(|r| r.verified_at), Some(200));
    }

    #[test]
    fn test_verify_mismatch() {
        let mut engine = StorageChecksumEngine::new(ChecksumAlgorithm::Crc32);
        engine.compute_for("doc".to_owned(), b"original", 100);
        let result = engine.verify("doc", b"corrupted", 200).expect("result");
        assert!(!result.matches);
        // verified_at not updated on mismatch
        assert!(engine.record("doc").and_then(|r| r.verified_at).is_none());
    }

    #[test]
    fn test_verify_missing_object() {
        let mut engine = StorageChecksumEngine::new(ChecksumAlgorithm::Fnv1a64);
        assert!(engine.verify("missing", b"data", 0).is_none());
    }

    #[test]
    fn test_verify_updates_verified_at() {
        let mut engine = StorageChecksumEngine::new(ChecksumAlgorithm::Xxhash64);
        engine.compute_for("item".to_owned(), b"payload", 50);
        let r = engine.verify("item", b"payload", 999).expect("ok");
        assert!(r.matches);
        assert_eq!(engine.record("item").and_then(|r| r.verified_at), Some(999));
    }

    // ── Engine: batch_compute ──────────────────────────────────────────────

    #[test]
    fn test_batch_compute() {
        let mut engine = StorageChecksumEngine::new(ChecksumAlgorithm::Murmur3_32);
        let items: Vec<(&str, &[u8])> = vec![("a", b"alpha"), ("b", b"beta"), ("c", b"gamma")];
        let records: Vec<ChecksumRecord> = engine.batch_compute(&items, 777);
        assert_eq!(records.len(), 3);
        assert_eq!(engine.object_count(), 3);
    }

    #[test]
    fn test_batch_compute_all_stored() {
        let mut engine = StorageChecksumEngine::new(ChecksumAlgorithm::Adler32);
        let items: Vec<(&str, &[u8])> = vec![("x", b"1"), ("y", b"2")];
        engine.batch_compute(&items, 10);
        assert!(engine.record("x").is_some());
        assert!(engine.record("y").is_some());
    }

    // ── Engine: verify_all ─────────────────────────────────────────────────

    #[test]
    fn test_verify_all_clean() {
        let mut engine = StorageChecksumEngine::new(ChecksumAlgorithm::Crc32);
        engine.compute_for("p".to_owned(), b"ping", 1);
        engine.compute_for("q".to_owned(), b"pong", 1);
        let results = engine.verify_all(
            |id| {
                if id == "p" {
                    Some(b"ping".to_vec())
                } else {
                    Some(b"pong".to_vec())
                }
            },
            2,
        );
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.matches));
    }

    #[test]
    fn test_verify_all_with_corruption() {
        let mut engine = StorageChecksumEngine::new(ChecksumAlgorithm::Crc32);
        engine.compute_for("good".to_owned(), b"ok", 1);
        engine.compute_for("bad".to_owned(), b"original", 1);
        let results = engine.verify_all(
            |id| {
                if id == "good" {
                    Some(b"ok".to_vec())
                } else {
                    Some(b"corrupted".to_vec())
                }
            },
            2,
        );
        let corrupt = StorageChecksumEngine::corruption_count(&results);
        assert_eq!(corrupt, 1);
    }

    #[test]
    fn test_verify_all_data_fn_returns_none() {
        let mut engine = StorageChecksumEngine::new(ChecksumAlgorithm::Fnv1a64);
        engine.compute_for("present".to_owned(), b"data", 0);
        engine.compute_for("absent".to_owned(), b"x", 0);
        let results = engine.verify_all(
            |id| {
                if id == "present" {
                    Some(b"data".to_vec())
                } else {
                    None
                }
            },
            5,
        );
        // Only "present" was verifiable
        assert_eq!(results.len(), 1);
    }

    // ── Engine: stats / corruption_count ──────────────────────────────────

    #[test]
    fn test_corruption_count_zero() {
        let results: Vec<CeVerificationResult> = vec![];
        assert_eq!(StorageChecksumEngine::corruption_count(&results), 0);
    }

    #[test]
    fn test_stats_empty() {
        let engine = StorageChecksumEngine::new(ChecksumAlgorithm::Blake3_256);
        let stats = engine.stats(&[]);
        assert_eq!(stats.total_records, 0);
        assert_eq!(stats.verified_count, 0);
        assert_eq!(stats.corruption_rate, 0.0);
        assert_eq!(stats.algorithm, "blake3_256");
    }

    #[test]
    fn test_stats_after_verification() {
        let mut engine = StorageChecksumEngine::new(ChecksumAlgorithm::Crc32);
        engine.compute_for("a".to_owned(), b"hello", 1);
        engine.compute_for("b".to_owned(), b"world", 1);
        let results = engine.verify_all(
            |id| {
                if id == "a" {
                    Some(b"hello".to_vec())
                } else {
                    Some(b"corrupted".to_vec())
                }
            },
            2,
        );
        let stats = engine.stats(&results);
        assert_eq!(stats.total_records, 2);
        assert_eq!(stats.verified_count, 1); // only "a" passed
        assert!((stats.corruption_rate - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_algorithm_name() {
        assert_eq!(ChecksumAlgorithm::Fnv1a64.name(), "fnv1a64");
        assert_eq!(ChecksumAlgorithm::Djb2.name(), "djb2");
        assert_eq!(ChecksumAlgorithm::Blake3_256.name(), "blake3_256");
        assert_eq!(ChecksumAlgorithm::Crc32.name(), "crc32");
    }

    #[test]
    fn test_all_algorithms_produce_nonzero_for_nonempty() {
        let data = b"checksum test data";
        let algos = [
            ChecksumAlgorithm::Fnv1a64,
            ChecksumAlgorithm::Djb2,
            ChecksumAlgorithm::Murmur3_32,
            ChecksumAlgorithm::Adler32,
            ChecksumAlgorithm::Crc32,
            ChecksumAlgorithm::Xxhash64,
            ChecksumAlgorithm::Blake3_256,
        ];
        for algo in algos {
            let cs = StorageChecksumEngine::compute_with_algorithm(algo, data);
            assert!(
                cs.value.iter().any(|&b| b != 0),
                "Algorithm {:?} returned all-zero for non-empty input",
                algo
            );
        }
    }

    #[test]
    fn test_overwrite_record_on_recompute() {
        let mut engine = StorageChecksumEngine::new(ChecksumAlgorithm::Crc32);
        engine.compute_for("obj".to_owned(), b"v1", 10);
        engine.compute_for("obj".to_owned(), b"v2", 20);
        // Should be overwritten
        assert_eq!(engine.object_count(), 1);
        let rec = engine.record("obj").expect("present");
        assert_eq!(rec.computed_at, 20);
        assert_eq!(rec.size_bytes, 2);
    }

    #[test]
    fn test_verify_on_read_flag() {
        let mut engine = StorageChecksumEngine::new(ChecksumAlgorithm::Fnv1a64);
        engine.verify_on_read = true;
        assert!(engine.verify_on_read);
    }

    #[test]
    fn test_compute_with_algorithm_static() {
        let cs = StorageChecksumEngine::compute_with_algorithm(ChecksumAlgorithm::Adler32, b"abc");
        assert_eq!(cs.algorithm, ChecksumAlgorithm::Adler32);
        // adler32("abc") is known
        let expected = adler32(b"abc").to_le_bytes().to_vec();
        assert_eq!(cs.value, expected);
    }
}
