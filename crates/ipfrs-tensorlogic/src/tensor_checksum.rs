//! Tensor Checksum Engine
//!
//! Computes and verifies checksums for tensor data to detect corruption during
//! storage or transmission. Supports multiple pure-Rust checksum algorithms
//! with a stateful engine that tracks per-tensor records and verification stats.
//!
//! # Algorithms
//!
//! | Algorithm    | Width  | Speed  | Use-case                                 |
//! |-------------|--------|--------|------------------------------------------|
//! | `Fnv1a64`   | 64-bit | Fast   | General-purpose, non-cryptographic hash  |
//! | `Adler32`   | 32-bit | Fast   | Data integrity, used in zlib             |
//! | `Fletcher16`| 16-bit | Fast   | Lightweight embedded / small payloads    |
//! | `XorFold`   | 64-bit | Fastest| Ultra-fast, low-collision large tensors  |
//!
//! # Examples
//!
//! ```
//! use ipfrs_tensorlogic::tensor_checksum::{
//!     ChecksumAlgorithm, TensorChecksumEngine,
//! };
//!
//! let mut engine = TensorChecksumEngine::new();
//! let data = b"hello tensor world";
//! let record = engine.compute(1, "layer0".to_string(), data, ChecksumAlgorithm::Fnv1a64, 0);
//! assert!(record.is_valid(data));
//!
//! let ok = engine.verify(1, data).expect("example: should succeed in docs");
//! assert!(ok);
//! ```

use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// ChecksumAlgorithm
// ──────────────────────────────────────────────────────────────────────────────

/// Checksum algorithm selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChecksumAlgorithm {
    /// FNV-1a 64-bit non-cryptographic hash.
    Fnv1a64,
    /// Adler-32 checksum (mod 65521), result cast to u64.
    Adler32,
    /// Fletcher-16 checksum, result cast to u64.
    Fletcher16,
    /// XOR all 8-byte chunks (zero-pad last chunk), fold to u64.
    XorFold,
}

// ──────────────────────────────────────────────────────────────────────────────
// Pure-Rust implementations
// ──────────────────────────────────────────────────────────────────────────────

const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME: u64 = 1_099_511_628_211;

/// Compute the FNV-1a 64-bit hash over arbitrary bytes.
///
/// Uses the standard parameters: offset basis `14695981039346656037`,
/// prime `1099511628211`.
///
/// # Examples
///
/// ```
/// use ipfrs_tensorlogic::tensor_checksum::fnv1a64;
/// let h = fnv1a64(b"hello");
/// assert_ne!(h, 0);
/// assert_eq!(h, fnv1a64(b"hello")); // deterministic
/// ```
pub fn fnv1a64(data: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Adler-32 checksum (pure Rust).
///
/// Uses modulus 65521 (the largest prime less than 65536) as defined in
/// RFC 1950 / zlib spec. The 32-bit result is cast to u64.
///
/// # Examples
///
/// ```
/// use ipfrs_tensorlogic::tensor_checksum::adler32;
/// // Known value: "ABC" → 0x018D00C7
/// assert_eq!(adler32(b"ABC"), 0x018D00C7);
/// ```
pub fn adler32(data: &[u8]) -> u64 {
    const MOD_ADLER: u32 = 65521;
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + u32::from(byte)) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }
    u64::from((b << 16) | a)
}

/// Fletcher-16 checksum (pure Rust).
///
/// Processes bytes in pairs of sums, each taken modulo 255. The 16-bit
/// result is cast to u64.
///
/// # Examples
///
/// ```
/// use ipfrs_tensorlogic::tensor_checksum::fletcher16;
/// let h = fletcher16(b"hello");
/// assert_ne!(h, 0);
/// ```
pub fn fletcher16(data: &[u8]) -> u64 {
    let mut sum1: u16 = 0;
    let mut sum2: u16 = 0;
    for &byte in data {
        sum1 = (sum1 + u16::from(byte)) % 255;
        sum2 = (sum2 + sum1) % 255;
    }
    u64::from((sum2 << 8) | sum1)
}

/// XOR-fold checksum.
///
/// Splits `data` into 8-byte chunks (the final chunk is zero-padded if
/// shorter) and XORs all of them together to produce a u64.
///
/// # Examples
///
/// ```
/// use ipfrs_tensorlogic::tensor_checksum::xor_fold;
/// let h = xor_fold(b"12345678");
/// assert_ne!(h, 0);
/// assert_eq!(xor_fold(b""), 0);
/// ```
pub fn xor_fold(data: &[u8]) -> u64 {
    let mut result: u64 = 0;
    let mut idx = 0;
    while idx + 8 <= data.len() {
        let chunk = u64::from_le_bytes([
            data[idx],
            data[idx + 1],
            data[idx + 2],
            data[idx + 3],
            data[idx + 4],
            data[idx + 5],
            data[idx + 6],
            data[idx + 7],
        ]);
        result ^= chunk;
        idx += 8;
    }
    // Handle remaining bytes (zero-padded)
    let remainder = data.len() - idx;
    if remainder > 0 {
        let mut buf = [0u8; 8];
        buf[..remainder].copy_from_slice(&data[idx..]);
        result ^= u64::from_le_bytes(buf);
    }
    result
}

/// Dispatch to the correct algorithm implementation.
fn compute_checksum(data: &[u8], algorithm: ChecksumAlgorithm) -> u64 {
    match algorithm {
        ChecksumAlgorithm::Fnv1a64 => fnv1a64(data),
        ChecksumAlgorithm::Adler32 => adler32(data),
        ChecksumAlgorithm::Fletcher16 => fletcher16(data),
        ChecksumAlgorithm::XorFold => xor_fold(data),
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// TensorChecksum
// ──────────────────────────────────────────────────────────────────────────────

/// A checksum value together with the metadata needed to re-verify data later.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorChecksum {
    /// Algorithm used to produce `value`.
    pub algorithm: ChecksumAlgorithm,
    /// The checksum value.
    pub value: u64,
    /// Length (in bytes) of the data that was checksummed.
    pub data_len: usize,
    /// Unix timestamp (seconds) at which the checksum was computed.
    pub computed_at_secs: u64,
}

impl TensorChecksum {
    /// Recompute the checksum for `data` using the same algorithm and compare
    /// it against `self.value`.
    ///
    /// Returns `true` if the data is intact, `false` if it has been corrupted
    /// (or if `data.len() != self.data_len`).
    ///
    /// # Examples
    ///
    /// ```
    /// use ipfrs_tensorlogic::tensor_checksum::{ChecksumAlgorithm, TensorChecksum};
    ///
    /// let data = b"tensor payload";
    /// let cs = TensorChecksum {
    ///     algorithm: ChecksumAlgorithm::Fnv1a64,
    ///     value: ipfrs_tensorlogic::tensor_checksum::fnv1a64(data),
    ///     data_len: data.len(),
    ///     computed_at_secs: 0,
    /// };
    /// assert!(cs.verify(data));
    /// assert!(!cs.verify(b"corrupted payload"));
    /// ```
    pub fn verify(&self, data: &[u8]) -> bool {
        if data.len() != self.data_len {
            return false;
        }
        compute_checksum(data, self.algorithm) == self.value
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ChecksumRecord
// ──────────────────────────────────────────────────────────────────────────────

/// Associates a [`TensorChecksum`] with a specific tensor and layer name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChecksumRecord {
    /// Unique identifier of the tensor.
    pub tensor_id: u64,
    /// The checksum for this tensor's data.
    pub checksum: TensorChecksum,
    /// Human-readable name of the layer that owns this tensor.
    pub layer_name: String,
}

impl ChecksumRecord {
    /// Returns `true` if `data` matches the stored checksum.
    ///
    /// Delegates to [`TensorChecksum::verify`].
    pub fn is_valid(&self, data: &[u8]) -> bool {
        self.checksum.verify(data)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ChecksumEngineStats
// ──────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics for a [`TensorChecksumEngine`] instance.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChecksumEngineStats {
    /// Total number of checksums computed via [`TensorChecksumEngine::compute`].
    pub total_computed: u64,
    /// Total number of verifications attempted via [`TensorChecksumEngine::verify`].
    pub total_verified: u64,
    /// Total number of failed verifications (data mismatch or unknown id).
    pub total_failures: u64,
}

impl ChecksumEngineStats {
    /// Fraction of verifications that resulted in a failure.
    ///
    /// Returns `0.0` when `total_verified == 0` to avoid division by zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use ipfrs_tensorlogic::tensor_checksum::ChecksumEngineStats;
    ///
    /// let stats = ChecksumEngineStats {
    ///     total_computed: 10,
    ///     total_verified: 4,
    ///     total_failures: 1,
    /// };
    /// assert!((stats.failure_rate() - 0.25).abs() < 1e-9);
    /// ```
    pub fn failure_rate(&self) -> f64 {
        if self.total_verified == 0 {
            return 0.0;
        }
        self.total_failures as f64 / self.total_verified as f64
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// TensorChecksumEngine
// ──────────────────────────────────────────────────────────────────────────────

/// Stateful engine that manages per-tensor checksum records.
///
/// # Thread Safety
///
/// `TensorChecksumEngine` is **not** `Send`/`Sync` by default because it owns
/// a `HashMap`. Wrap in `Arc<Mutex<…>>` or `parking_lot::Mutex` for shared
/// access across threads.
///
/// # Examples
///
/// ```
/// use ipfrs_tensorlogic::tensor_checksum::{ChecksumAlgorithm, TensorChecksumEngine};
///
/// let mut engine = TensorChecksumEngine::new();
/// let data = b"layer weights";
///
/// engine.compute(42, "fc1".to_string(), data, ChecksumAlgorithm::Adler32, 1_000_000);
///
/// assert_eq!(engine.verify(42, data), Some(true));
/// assert_eq!(engine.verify(99, data), None); // unknown id
///
/// assert!(engine.remove(42));
/// assert!(!engine.remove(42)); // already gone
/// ```
pub struct TensorChecksumEngine {
    /// Per-tensor checksum records, keyed by `tensor_id`.
    pub records: HashMap<u64, ChecksumRecord>,
    /// Cumulative operational statistics.
    pub stats: ChecksumEngineStats,
}

impl TensorChecksumEngine {
    /// Create a new, empty engine with zeroed statistics.
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
            stats: ChecksumEngineStats::default(),
        }
    }

    /// Compute a checksum for `data`, store the resulting [`ChecksumRecord`],
    /// increment `stats.total_computed`, and return a reference to the record.
    ///
    /// If a record already exists for `tensor_id`, it is replaced.
    ///
    /// # Parameters
    ///
    /// - `tensor_id`  — Unique identifier for the tensor.
    /// - `layer_name` — Human-readable layer name (e.g. `"encoder.layer1"`).
    /// - `data`       — Raw bytes of the tensor payload.
    /// - `algorithm`  — Checksum algorithm to use.
    /// - `now_secs`   — Current time as Unix seconds (caller-supplied for
    ///   determinism in tests and embedded environments).
    pub fn compute(
        &mut self,
        tensor_id: u64,
        layer_name: String,
        data: &[u8],
        algorithm: ChecksumAlgorithm,
        now_secs: u64,
    ) -> &ChecksumRecord {
        let value = compute_checksum(data, algorithm);
        let record = ChecksumRecord {
            tensor_id,
            checksum: TensorChecksum {
                algorithm,
                value,
                data_len: data.len(),
                computed_at_secs: now_secs,
            },
            layer_name,
        };
        self.records.insert(tensor_id, record);
        self.stats.total_computed += 1;
        // SAFETY: we just inserted, so the key is guaranteed present.
        self.records.get(&tensor_id).expect("record just inserted")
    }

    /// Verify the stored checksum for `tensor_id` against `data`.
    ///
    /// - Returns `None` if `tensor_id` is not registered.
    /// - Returns `Some(true)` if the data matches.
    /// - Returns `Some(false)` if the data does not match.
    ///
    /// Both `total_verified` and (on failure) `total_failures` are updated.
    pub fn verify(&mut self, tensor_id: u64, data: &[u8]) -> Option<bool> {
        let record = self.records.get(&tensor_id)?;
        let ok = record.is_valid(data);
        self.stats.total_verified += 1;
        if !ok {
            self.stats.total_failures += 1;
        }
        Some(ok)
    }

    /// Remove the record for `tensor_id`.
    ///
    /// Returns `true` if the record existed and was removed, `false` if no
    /// record was found for the given id.
    pub fn remove(&mut self, tensor_id: u64) -> bool {
        self.records.remove(&tensor_id).is_some()
    }

    /// Return a reference to the current engine statistics.
    pub fn stats(&self) -> &ChecksumEngineStats {
        &self.stats
    }
}

impl Default for TensorChecksumEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── FNV-1a known value ────────────────────────────────────────────────────

    #[test]
    fn test_fnv1a64_empty() {
        // The FNV-1a hash of the empty byte sequence equals the offset basis.
        assert_eq!(fnv1a64(b""), FNV_OFFSET_BASIS);
    }

    #[test]
    fn test_fnv1a64_known_value() {
        // Verified against reference implementation:
        // echo -n "hello" | fnv64 --type fnv1a
        // FNV-1a("hello") = 0xa430d84680aabd0b
        let expected: u64 = 0xa430d84680aabd0b;
        assert_eq!(fnv1a64(b"hello"), expected);
    }

    #[test]
    fn test_fnv1a64_deterministic() {
        let h1 = fnv1a64(b"tensor data XYZ");
        let h2 = fnv1a64(b"tensor data XYZ");
        assert_eq!(h1, h2);
    }

    // ── Adler-32 known value ──────────────────────────────────────────────────

    #[test]
    fn test_adler32_abc_known_value() {
        // Adler-32("ABC"):
        //   a=1,b=0 → A(65): a=66,b=66 → B(66): a=132,b=198 → C(67): a=199,b=397
        //   result = (397 << 16) | 199 = 0x018D00C7
        assert_eq!(adler32(b"ABC"), 0x018D00C7);
    }

    #[test]
    fn test_adler32_empty() {
        // Adler-32 of the empty sequence: a=1, b=0 → (0 << 16) | 1 = 1
        assert_eq!(adler32(b""), 1);
    }

    #[test]
    fn test_adler32_deterministic() {
        assert_eq!(adler32(b"hello world"), adler32(b"hello world"));
    }

    // ── Fletcher-16 ───────────────────────────────────────────────────────────

    #[test]
    fn test_fletcher16_empty() {
        assert_eq!(fletcher16(b""), 0);
    }

    #[test]
    fn test_fletcher16_non_zero() {
        let h = fletcher16(b"abcde");
        assert_ne!(h, 0);
    }

    #[test]
    fn test_fletcher16_deterministic() {
        assert_eq!(fletcher16(b"tensor"), fletcher16(b"tensor"));
    }

    #[test]
    fn test_fletcher16_distinguishes_inputs() {
        // Different inputs should (in practice) give different checksums.
        assert_ne!(fletcher16(b"aaa"), fletcher16(b"bbb"));
    }

    // ── XorFold ───────────────────────────────────────────────────────────────

    #[test]
    fn test_xor_fold_empty() {
        assert_eq!(xor_fold(b""), 0);
    }

    #[test]
    fn test_xor_fold_exact_chunk() {
        // 8 bytes — one chunk, result equals that u64.
        let data = b"ABCDEFGH";
        let expected = u64::from_le_bytes(*data);
        assert_eq!(xor_fold(data), expected);
    }

    #[test]
    fn test_xor_fold_two_chunks() {
        let a = [0x01u8; 8];
        let b = [0x02u8; 8];
        let mut data = [0u8; 16];
        data[..8].copy_from_slice(&a);
        data[8..].copy_from_slice(&b);
        let expected = u64::from_le_bytes(a) ^ u64::from_le_bytes(b);
        assert_eq!(xor_fold(&data), expected);
    }

    #[test]
    fn test_xor_fold_partial_chunk_zero_padded() {
        // 9 bytes → one full chunk XOR one partial chunk (zero-padded).
        let data = b"ABCDEFGHI"; // 9 bytes
        let chunk1 = u64::from_le_bytes([b'A', b'B', b'C', b'D', b'E', b'F', b'G', b'H']);
        let chunk2 = u64::from_le_bytes([b'I', 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(xor_fold(data), chunk1 ^ chunk2);
    }

    // ── Compute + verify round-trip ───────────────────────────────────────────

    fn round_trip(algorithm: ChecksumAlgorithm) {
        let mut engine = TensorChecksumEngine::new();
        let data = b"round trip test data for tensor";
        engine.compute(1, "layer".to_string(), data, algorithm, 42);
        assert_eq!(engine.verify(1, data), Some(true));
    }

    #[test]
    fn test_round_trip_fnv1a64() {
        round_trip(ChecksumAlgorithm::Fnv1a64);
    }

    #[test]
    fn test_round_trip_adler32() {
        round_trip(ChecksumAlgorithm::Adler32);
    }

    #[test]
    fn test_round_trip_fletcher16() {
        round_trip(ChecksumAlgorithm::Fletcher16);
    }

    #[test]
    fn test_round_trip_xor_fold() {
        round_trip(ChecksumAlgorithm::XorFold);
    }

    // ── Corruption detection ──────────────────────────────────────────────────

    #[test]
    fn test_verify_detects_corruption() {
        let mut engine = TensorChecksumEngine::new();
        let original = b"important tensor weights";
        let corrupted = b"corrupted tensor weights";
        engine.compute(
            7,
            "output".to_string(),
            original,
            ChecksumAlgorithm::Fnv1a64,
            0,
        );
        assert_eq!(engine.verify(7, corrupted), Some(false));
    }

    #[test]
    fn test_verify_detects_length_change() {
        let mut engine = TensorChecksumEngine::new();
        let data = b"full data payload";
        engine.compute(8, "embed".to_string(), data, ChecksumAlgorithm::Adler32, 0);
        // Truncate the data
        assert_eq!(engine.verify(8, &data[..5]), Some(false));
    }

    // ── Unknown tensor_id returns None ────────────────────────────────────────

    #[test]
    fn test_verify_unknown_returns_none() {
        let mut engine = TensorChecksumEngine::new();
        assert_eq!(engine.verify(999, b"anything"), None);
    }

    // ── Remove ────────────────────────────────────────────────────────────────

    #[test]
    fn test_remove_existing() {
        let mut engine = TensorChecksumEngine::new();
        engine.compute(
            5,
            "fc".to_string(),
            b"data",
            ChecksumAlgorithm::Fletcher16,
            0,
        );
        assert!(engine.remove(5));
        // After removal, verify returns None
        assert_eq!(engine.verify(5, b"data"), None);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut engine = TensorChecksumEngine::new();
        assert!(!engine.remove(404));
    }

    // ── failure_rate ──────────────────────────────────────────────────────────

    #[test]
    fn test_failure_rate_no_verifications() {
        let stats = ChecksumEngineStats::default();
        assert_eq!(stats.failure_rate(), 0.0);
    }

    #[test]
    fn test_failure_rate_calculation() {
        let stats = ChecksumEngineStats {
            total_computed: 10,
            total_verified: 8,
            total_failures: 2,
        };
        assert!((stats.failure_rate() - 0.25).abs() < 1e-9);
    }

    // ── stats.total_failures increments ──────────────────────────────────────

    #[test]
    fn test_stats_total_failures_increments() {
        let mut engine = TensorChecksumEngine::new();
        let data = b"good data";
        engine.compute(10, "attn".to_string(), data, ChecksumAlgorithm::XorFold, 0);

        // Successful verification — no failure increment
        engine.verify(10, data);
        assert_eq!(engine.stats().total_failures, 0);

        // Corrupted data — failure increments
        engine.verify(10, b"bad data XX");
        assert_eq!(engine.stats().total_failures, 1);

        // Another failure
        engine.verify(10, b"also bad XX");
        assert_eq!(engine.stats().total_failures, 2);
    }

    #[test]
    fn test_stats_total_computed_and_verified() {
        let mut engine = TensorChecksumEngine::new();
        let data = b"weights";
        engine.compute(1, "l1".to_string(), data, ChecksumAlgorithm::Fnv1a64, 0);
        engine.compute(2, "l2".to_string(), data, ChecksumAlgorithm::Adler32, 0);
        assert_eq!(engine.stats().total_computed, 2);

        engine.verify(1, data);
        engine.verify(2, data);
        assert_eq!(engine.stats().total_verified, 2);
        assert_eq!(engine.stats().total_failures, 0);
    }

    // ── ChecksumRecord::is_valid delegates to TensorChecksum::verify ─────────

    #[test]
    fn test_checksum_record_is_valid() {
        let data = b"record test payload";
        let record = ChecksumRecord {
            tensor_id: 99,
            checksum: TensorChecksum {
                algorithm: ChecksumAlgorithm::Fletcher16,
                value: fletcher16(data),
                data_len: data.len(),
                computed_at_secs: 1_000,
            },
            layer_name: "norm".to_string(),
        };
        assert!(record.is_valid(data));
        assert!(!record.is_valid(b"wrong data!!"));
    }
}
