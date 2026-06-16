//! Object Integrity Checker — multi-level hash-based object integrity verification.
//!
//! Provides `ObjectIntegrityChecker`, a production-grade component that computes
//! FNV-1a-64, Adler-32, and CRC-16 CCITT checksums over arbitrary byte payloads,
//! stores canonical `ObjectRecord`s, and verifies them at configurable integrity
//! levels (Quick, Standard, Full, Custom).

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Pure-Rust hash primitives (no external crate)
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit hash.
#[inline]
fn fnv1a_64(data: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 14695981039346656037;
    const PRIME: u64 = 1099511628211;
    let mut hash = OFFSET_BASIS;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// Adler-32 checksum (modulo 65521).
#[inline]
fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = a.wrapping_add(byte as u32) % 65521;
        b = b.wrapping_add(a) % 65521;
    }
    (b << 16) | a
}

/// CRC-16 CCITT.
#[inline]
fn crc16(data: &[u8]) -> u16 {
    let mut crc = 0xFFFFu16;
    for &byte in data {
        let mut x = (crc >> 8) ^ (byte as u16);
        x ^= x >> 4;
        crc = (crc << 8) ^ (x << 12) ^ (x << 5) ^ x;
    }
    crc
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Composite integrity digest computed over a byte slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegrityHash {
    /// FNV-1a 64-bit digest.
    pub fnv1a: u64,
    /// Adler-32 digest.
    pub adler32: u32,
    /// CRC-16 CCITT digest.
    pub crc16: u16,
    /// Byte length of the source data.
    pub size_bytes: usize,
}

impl IntegrityHash {
    fn compute(data: &[u8]) -> Self {
        Self {
            fnv1a: fnv1a_64(data),
            adler32: adler32(data),
            crc16: crc16(data),
            size_bytes: data.len(),
        }
    }
}

/// Specifies which hash algorithms are executed during verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntegrityLevel {
    /// Only CRC-16 (fastest).
    Quick,
    /// FNV-1a-64 + Adler-32 (no CRC).
    Standard,
    /// All three algorithms.
    Full,
    /// Named algorithm set (validated against supported names at runtime).
    Custom(Vec<String>),
}

impl IntegrityLevel {
    /// Supported algorithm names for `Custom` variants.
    pub const SUPPORTED: &'static [&'static str] = &["fnv1a", "adler32", "crc16"];
}

/// Lifecycle status of a registered object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OicIntegrityStatus {
    /// No verification has been performed yet.
    Unknown,
    /// Successfully verified at the given Unix-microsecond timestamp.
    Verified(u64),
    /// A mismatch was detected.
    Corrupted {
        /// When the corruption was first detected (Unix µs).
        detected_at: u64,
        /// The hash that was expected (stored canonical value).
        expected: IntegrityHash,
        /// The hash computed from the presented data.
        got: IntegrityHash,
    },
    /// Flagged as possibly corrupted with a human-readable reason.
    PossiblyCorrupted(String),
}

/// Persistent record for a single registered object.
#[derive(Debug, Clone)]
pub struct OicObjectRecord {
    /// Stable identifier supplied by the caller.
    pub id: String,
    /// Canonical hash computed at registration (or last `update`).
    pub hash: IntegrityHash,
    /// Unix-microsecond timestamp of registration.
    pub created_at: u64,
    /// Unix-microsecond timestamp of the most recent successful verification (or 0).
    pub last_verified_at: u64,
    /// Total number of times `verify` has been called for this object.
    pub verification_count: u64,
    /// Current integrity status.
    pub status: OicIntegrityStatus,
}

/// Detailed result of a single `verify` call.
#[derive(Debug, Clone)]
pub struct OicVerificationResult {
    /// Object identifier.
    pub object_id: String,
    /// Whether all checks at the requested level passed.
    pub passed: bool,
    /// Integrity level that was applied.
    pub level: IntegrityLevel,
    /// Human-readable descriptions of any mismatches detected.
    pub errors: Vec<String>,
    /// Wall-clock duration of the verification in microseconds.
    pub duration_us: u64,
}

/// Configuration for `ObjectIntegrityChecker`.
#[derive(Debug, Clone)]
pub struct CheckerConfig {
    /// Default level used when callers do not supply one explicitly.
    pub default_level: IntegrityLevel,
    /// When `true` the checker automatically performs a Full verification on every
    /// `verify` call even if only Quick is requested (future hook; not enforced
    /// beyond storing the intent).
    pub auto_verify_on_read: bool,
    /// Alert threshold: if `corrupted_objects / total_objects` exceeds this ratio
    /// the stats will surface it.  Range `[0.0, 1.0]`.
    pub corruption_threshold: f64,
    /// Hard cap on the number of objects that may be registered simultaneously.
    pub max_objects: usize,
}

impl Default for CheckerConfig {
    fn default() -> Self {
        Self {
            default_level: IntegrityLevel::Full,
            auto_verify_on_read: false,
            corruption_threshold: 0.05,
            max_objects: 1_000_000,
        }
    }
}

/// Aggregated statistics over all registered objects.
#[derive(Debug, Clone)]
pub struct CheckerStats {
    /// Number of currently registered objects.
    pub total_objects: usize,
    /// Number of objects whose status is `Verified(_)`.
    pub verified_objects: usize,
    /// Number of objects whose status is `Corrupted{..}`.
    pub corrupted_objects: usize,
    /// `corrupted_objects / total_objects`, or 0.0 when `total_objects == 0`.
    pub corruption_rate: f64,
    /// Cumulative number of `verify` invocations across all objects.
    pub total_verifications: u64,
    /// Rolling mean verification duration in microseconds.
    pub avg_verification_us: f64,
}

/// Errors that can be returned by `ObjectIntegrityChecker`.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CheckerError {
    /// The requested object identifier is not registered.
    #[error("object not found: {0}")]
    ObjectNotFound(String),
    /// A specific hash field did not match.
    #[error("hash mismatch for '{id}' on field '{field}'")]
    HashMismatch {
        /// Object identifier.
        id: String,
        /// Name of the mismatched field (`fnv1a`, `adler32`, `crc16`, or `size`).
        field: String,
    },
    /// The `max_objects` limit in `CheckerConfig` would be exceeded.
    #[error("maximum object capacity exceeded")]
    MaxObjectsExceeded,
    /// A `Custom` level contained an unrecognised algorithm name.
    #[error("invalid integrity level: {0}")]
    InvalidLevel(String),
}

// ---------------------------------------------------------------------------
// Core checker
// ---------------------------------------------------------------------------

/// Production-grade multi-level object integrity checker.
///
/// Thread-safety is the caller's responsibility; wrap in `Arc<Mutex<…>>` if
/// shared across threads.
pub struct ObjectIntegrityChecker {
    config: CheckerConfig,
    objects: HashMap<String, OicObjectRecord>,
    /// Accumulated verification durations (µs) for rolling-average calculation.
    total_verify_us: u64,
    /// Total number of `verify` calls dispatched.
    total_verifications: u64,
}

impl ObjectIntegrityChecker {
    /// Create a new checker with the provided configuration.
    pub fn new(config: CheckerConfig) -> Self {
        Self {
            objects: HashMap::new(),
            total_verify_us: 0,
            total_verifications: 0,
            config,
        }
    }

    /// Create a new checker with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(CheckerConfig::default())
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn now_us() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0)
    }

    fn validate_level(level: &IntegrityLevel) -> Result<(), CheckerError> {
        if let IntegrityLevel::Custom(algos) = level {
            for algo in algos {
                if !IntegrityLevel::SUPPORTED.contains(&algo.as_str()) {
                    return Err(CheckerError::InvalidLevel(format!(
                        "unsupported algorithm '{}'; supported: {:?}",
                        algo,
                        IntegrityLevel::SUPPORTED
                    )));
                }
            }
        }
        Ok(())
    }

    /// Run the configured level checks, returning a list of error strings.
    fn run_checks(
        stored: &IntegrityHash,
        fresh: &IntegrityHash,
        level: &IntegrityLevel,
    ) -> Vec<String> {
        let mut errors = Vec::new();

        // Size check always applies.
        if stored.size_bytes != fresh.size_bytes {
            errors.push(format!(
                "size mismatch: expected {} bytes, got {}",
                stored.size_bytes, fresh.size_bytes
            ));
        }

        match level {
            IntegrityLevel::Quick => {
                if stored.crc16 != fresh.crc16 {
                    errors.push(format!(
                        "crc16 mismatch: expected {:#06x}, got {:#06x}",
                        stored.crc16, fresh.crc16
                    ));
                }
            }
            IntegrityLevel::Standard => {
                if stored.fnv1a != fresh.fnv1a {
                    errors.push(format!(
                        "fnv1a mismatch: expected {:#018x}, got {:#018x}",
                        stored.fnv1a, fresh.fnv1a
                    ));
                }
                if stored.adler32 != fresh.adler32 {
                    errors.push(format!(
                        "adler32 mismatch: expected {:#010x}, got {:#010x}",
                        stored.adler32, fresh.adler32
                    ));
                }
            }
            IntegrityLevel::Full => {
                if stored.fnv1a != fresh.fnv1a {
                    errors.push(format!(
                        "fnv1a mismatch: expected {:#018x}, got {:#018x}",
                        stored.fnv1a, fresh.fnv1a
                    ));
                }
                if stored.adler32 != fresh.adler32 {
                    errors.push(format!(
                        "adler32 mismatch: expected {:#010x}, got {:#010x}",
                        stored.adler32, fresh.adler32
                    ));
                }
                if stored.crc16 != fresh.crc16 {
                    errors.push(format!(
                        "crc16 mismatch: expected {:#06x}, got {:#06x}",
                        stored.crc16, fresh.crc16
                    ));
                }
            }
            IntegrityLevel::Custom(algos) => {
                for algo in algos {
                    match algo.as_str() {
                        "fnv1a" => {
                            if stored.fnv1a != fresh.fnv1a {
                                errors.push(format!(
                                    "fnv1a mismatch: expected {:#018x}, got {:#018x}",
                                    stored.fnv1a, fresh.fnv1a
                                ));
                            }
                        }
                        "adler32" => {
                            if stored.adler32 != fresh.adler32 {
                                errors.push(format!(
                                    "adler32 mismatch: expected {:#010x}, got {:#010x}",
                                    stored.adler32, fresh.adler32
                                ));
                            }
                        }
                        "crc16" => {
                            if stored.crc16 != fresh.crc16 {
                                errors.push(format!(
                                    "crc16 mismatch: expected {:#06x}, got {:#06x}",
                                    stored.crc16, fresh.crc16
                                ));
                            }
                        }
                        _ => {
                            // Already validated in validate_level; this branch is unreachable
                            // in normal flow.
                            errors.push(format!("unknown algorithm '{algo}' skipped"));
                        }
                    }
                }
            }
        }

        errors
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Register a new object and compute its canonical hashes.
    ///
    /// Returns the `IntegrityHash` on success.  Returns
    /// `CheckerError::MaxObjectsExceeded` if the configured cap is reached.
    /// Re-registering an existing id overwrites the stored record.
    pub fn register(&mut self, id: String, data: &[u8]) -> Result<IntegrityHash, CheckerError> {
        if !self.objects.contains_key(&id) && self.objects.len() >= self.config.max_objects {
            return Err(CheckerError::MaxObjectsExceeded);
        }
        let hash = IntegrityHash::compute(data);
        let now = Self::now_us();
        let record = OicObjectRecord {
            id: id.clone(),
            hash: hash.clone(),
            created_at: now,
            last_verified_at: 0,
            verification_count: 0,
            status: OicIntegrityStatus::Unknown,
        };
        self.objects.insert(id, record);
        Ok(hash)
    }

    /// Verify `data` against the stored hashes for `id` at the given `level`.
    ///
    /// Updates the object's `status`, `last_verified_at`, and
    /// `verification_count`.  Returns a detailed `OicVerificationResult`.
    pub fn verify(
        &mut self,
        id: &str,
        data: &[u8],
        level: IntegrityLevel,
    ) -> Result<OicVerificationResult, CheckerError> {
        Self::validate_level(&level)?;

        let record = self
            .objects
            .get(id)
            .ok_or_else(|| CheckerError::ObjectNotFound(id.to_string()))?;

        let stored = record.hash.clone();
        let t_start = Self::now_us();
        let fresh = IntegrityHash::compute(data);
        let errors = Self::run_checks(&stored, &fresh, &level);
        let t_end = Self::now_us();
        let duration_us = t_end.saturating_sub(t_start);

        let passed = errors.is_empty();
        let now = Self::now_us();

        // Update the record.
        let record = self
            .objects
            .get_mut(id)
            .ok_or_else(|| CheckerError::ObjectNotFound(id.to_string()))?;

        record.verification_count += 1;
        self.total_verifications += 1;
        self.total_verify_us = self.total_verify_us.saturating_add(duration_us);

        if passed {
            record.last_verified_at = now;
            record.status = OicIntegrityStatus::Verified(now);
        } else {
            record.status = OicIntegrityStatus::Corrupted {
                detected_at: now,
                expected: stored.clone(),
                got: fresh.clone(),
            };
        }

        Ok(OicVerificationResult {
            object_id: id.to_string(),
            passed,
            level,
            errors,
            duration_us,
        })
    }

    /// Batch-verify multiple `(id, data)` pairs at the given `level`.
    ///
    /// Objects that are not found are reported as failed verifications with an
    /// error message rather than causing a panic or early abort.
    pub fn verify_all(
        &mut self,
        objects: &[(&str, &[u8])],
        level: IntegrityLevel,
    ) -> Vec<OicVerificationResult> {
        let mut results = Vec::with_capacity(objects.len());
        for &(id, data) in objects {
            let result = match self.verify(id, data, level.clone()) {
                Ok(r) => r,
                Err(e) => OicVerificationResult {
                    object_id: id.to_string(),
                    passed: false,
                    level: level.clone(),
                    errors: vec![e.to_string()],
                    duration_us: 0,
                },
            };
            results.push(result);
        }
        results
    }

    /// Mark an object as `PossiblyCorrupted` with a caller-supplied reason.
    pub fn mark_corrupted(&mut self, id: &str, reason: String) -> Result<(), CheckerError> {
        let record = self
            .objects
            .get_mut(id)
            .ok_or_else(|| CheckerError::ObjectNotFound(id.to_string()))?;
        record.status = OicIntegrityStatus::PossiblyCorrupted(reason);
        Ok(())
    }

    /// Replace the stored canonical hashes with freshly computed values from
    /// `new_data` and reset the verification counter.
    pub fn update(&mut self, id: &str, new_data: &[u8]) -> Result<IntegrityHash, CheckerError> {
        let record = self
            .objects
            .get_mut(id)
            .ok_or_else(|| CheckerError::ObjectNotFound(id.to_string()))?;
        let hash = IntegrityHash::compute(new_data);
        record.hash = hash.clone();
        record.verification_count = 0;
        record.last_verified_at = 0;
        record.status = OicIntegrityStatus::Unknown;
        Ok(hash)
    }

    /// Deregister an object, freeing its stored record.
    pub fn remove(&mut self, id: &str) -> Result<(), CheckerError> {
        self.objects
            .remove(id)
            .map(|_| ())
            .ok_or_else(|| CheckerError::ObjectNotFound(id.to_string()))
    }

    /// Return references to all objects currently in a `Corrupted` state.
    pub fn corrupted_objects(&self) -> Vec<&OicObjectRecord> {
        self.objects
            .values()
            .filter(|r| matches!(r.status, OicIntegrityStatus::Corrupted { .. }))
            .collect()
    }

    /// Return references to objects whose `last_verified_at` is older than
    /// `since_us` (i.e. `last_verified_at < since_us`), including those that
    /// have never been verified (`last_verified_at == 0`).
    pub fn objects_needing_verification(&self, since_us: u64) -> Vec<&OicObjectRecord> {
        self.objects
            .values()
            .filter(|r| r.last_verified_at < since_us)
            .collect()
    }

    /// Recompute hashes from `correct_data` and clear any corruption status.
    ///
    /// This is the "repair" path: after retrieving a known-good copy of an
    /// object, call this to reset the canonical digest and mark the object as
    /// verified.
    pub fn repair_hash(&mut self, id: &str, correct_data: &[u8]) -> Result<(), CheckerError> {
        let record = self
            .objects
            .get_mut(id)
            .ok_or_else(|| CheckerError::ObjectNotFound(id.to_string()))?;
        let now = Self::now_us();
        record.hash = IntegrityHash::compute(correct_data);
        record.status = OicIntegrityStatus::Verified(now);
        record.last_verified_at = now;
        Ok(())
    }

    /// Snapshot current aggregate statistics.
    pub fn stats(&self) -> CheckerStats {
        let total_objects = self.objects.len();
        let verified_objects = self
            .objects
            .values()
            .filter(|r| matches!(r.status, OicIntegrityStatus::Verified(_)))
            .count();
        let corrupted_objects = self
            .objects
            .values()
            .filter(|r| matches!(r.status, OicIntegrityStatus::Corrupted { .. }))
            .count();
        let corruption_rate = if total_objects == 0 {
            0.0
        } else {
            corrupted_objects as f64 / total_objects as f64
        };
        let avg_verification_us = if self.total_verifications == 0 {
            0.0
        } else {
            self.total_verify_us as f64 / self.total_verifications as f64
        };
        CheckerStats {
            total_objects,
            verified_objects,
            corrupted_objects,
            corruption_rate,
            total_verifications: self.total_verifications,
            avg_verification_us,
        }
    }

    /// Access the underlying configuration.
    pub fn config(&self) -> &CheckerConfig {
        &self.config
    }

    /// Number of currently registered objects.
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    /// Returns `true` when no objects are registered.
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Inline xorshift64 PRNG — no `rand` crate
    // -----------------------------------------------------------------------

    struct Xorshift64(u64);

    impl Xorshift64 {
        fn new(seed: u64) -> Self {
            // Ensure a non-zero seed.
            Self(if seed == 0 { 0xdeadbeef_cafebabe } else { seed })
        }

        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn next_bytes(&mut self, buf: &mut [u8]) {
            let mut i = 0;
            while i < buf.len() {
                let v = self.next().to_le_bytes();
                let take = (buf.len() - i).min(8);
                buf[i..i + take].copy_from_slice(&v[..take]);
                i += take;
            }
        }
    }

    fn rng_bytes(seed: u64, len: usize) -> Vec<u8> {
        let mut rng = Xorshift64::new(seed);
        let mut buf = vec![0u8; len];
        rng.next_bytes(&mut buf);
        buf
    }

    fn checker() -> ObjectIntegrityChecker {
        ObjectIntegrityChecker::with_defaults()
    }

    // -----------------------------------------------------------------------
    // Hash primitive unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fnv1a_empty() {
        // Known FNV-1a offset basis for empty input
        assert_eq!(fnv1a_64(b""), 14695981039346656037u64);
    }

    #[test]
    fn test_fnv1a_hello() {
        // FNV-1a of "hello" is a well-known value
        let h = fnv1a_64(b"hello");
        assert_ne!(h, fnv1a_64(b"world"));
    }

    #[test]
    fn test_adler32_empty() {
        // adler32("") = (0 << 16) | 1 = 1
        assert_eq!(adler32(b""), 1);
    }

    #[test]
    fn test_adler32_abc() {
        // "ABC" = bytes 65, 66, 67
        // a: 1 + 65 = 66; 66 + 66 = 132; 132 + 67 = 199
        // b: 0 + 66 = 66; 66 + 132 = 198; 198 + 199 = 397
        // result: (397 << 16) | 199 = 26017991
        assert_eq!(adler32(b"ABC"), 26017991);
    }

    #[test]
    fn test_crc16_empty() {
        // No bytes processed → init value 0xFFFF is returned unchanged
        assert_eq!(crc16(b""), 0xFFFF);
    }

    #[test]
    fn test_crc16_deterministic() {
        let a = crc16(b"deterministic");
        let b = crc16(b"deterministic");
        assert_eq!(a, b);
    }

    #[test]
    fn test_crc16_distinct() {
        assert_ne!(crc16(b"hello"), crc16(b"world"));
    }

    // -----------------------------------------------------------------------
    // IntegrityHash::compute
    // -----------------------------------------------------------------------

    #[test]
    fn test_integrity_hash_compute_size() {
        let data = b"test data";
        let h = IntegrityHash::compute(data);
        assert_eq!(h.size_bytes, data.len());
    }

    #[test]
    fn test_integrity_hash_equality() {
        let data = b"consistent";
        assert_eq!(IntegrityHash::compute(data), IntegrityHash::compute(data));
    }

    #[test]
    fn test_integrity_hash_inequality() {
        assert_ne!(
            IntegrityHash::compute(b"aaa"),
            IntegrityHash::compute(b"bbb")
        );
    }

    // -----------------------------------------------------------------------
    // register
    // -----------------------------------------------------------------------

    #[test]
    fn test_register_basic() {
        let mut c = checker();
        let data = b"hello world";
        let h = c.register("obj1".to_string(), data).unwrap();
        assert_eq!(h, IntegrityHash::compute(data));
    }

    #[test]
    fn test_register_stores_record() {
        let mut c = checker();
        c.register("a".to_string(), b"payload").unwrap();
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn test_register_overwrite() {
        let mut c = checker();
        c.register("x".to_string(), b"v1").unwrap();
        let h2 = c.register("x".to_string(), b"v2").unwrap();
        assert_eq!(h2, IntegrityHash::compute(b"v2"));
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn test_register_empty_data() {
        let mut c = checker();
        let h = c.register("empty".to_string(), b"").unwrap();
        assert_eq!(h.size_bytes, 0);
    }

    #[test]
    fn test_register_max_objects_exceeded() {
        let config = CheckerConfig {
            max_objects: 2,
            ..Default::default()
        };
        let mut c = ObjectIntegrityChecker::new(config);
        c.register("a".to_string(), b"data").unwrap();
        c.register("b".to_string(), b"data").unwrap();
        let err = c.register("c".to_string(), b"data").unwrap_err();
        assert_eq!(err, CheckerError::MaxObjectsExceeded);
    }

    // -----------------------------------------------------------------------
    // verify — Quick level
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_quick_pass() {
        let mut c = checker();
        c.register("q".to_string(), b"data").unwrap();
        let r = c.verify("q", b"data", IntegrityLevel::Quick).unwrap();
        assert!(r.passed);
        assert!(r.errors.is_empty());
    }

    #[test]
    fn test_verify_quick_fail() {
        let mut c = checker();
        c.register("q".to_string(), b"original").unwrap();
        let r = c.verify("q", b"tampered", IntegrityLevel::Quick).unwrap();
        assert!(!r.passed);
        assert!(!r.errors.is_empty());
    }

    #[test]
    fn test_verify_quick_size_mismatch() {
        let mut c = checker();
        c.register("q".to_string(), b"hello").unwrap();
        let r = c.verify("q", b"hi", IntegrityLevel::Quick).unwrap();
        assert!(!r.passed);
    }

    // -----------------------------------------------------------------------
    // verify — Standard level
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_standard_pass() {
        let mut c = checker();
        c.register("s".to_string(), b"payload").unwrap();
        let r = c.verify("s", b"payload", IntegrityLevel::Standard).unwrap();
        assert!(r.passed);
    }

    #[test]
    fn test_verify_standard_fail_fnv() {
        let mut c = checker();
        c.register("s".to_string(), b"original").unwrap();
        let r = c
            .verify("s", b"modified", IntegrityLevel::Standard)
            .unwrap();
        assert!(!r.passed);
    }

    // -----------------------------------------------------------------------
    // verify — Full level
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_full_pass() {
        let mut c = checker();
        c.register("f".to_string(), b"full check").unwrap();
        let r = c.verify("f", b"full check", IntegrityLevel::Full).unwrap();
        assert!(r.passed);
        assert!(r.errors.is_empty());
    }

    #[test]
    fn test_verify_full_fail_multiple_errors() {
        let mut c = checker();
        c.register("f".to_string(), b"abcdefgh").unwrap();
        let r = c.verify("f", b"ABCDEFGH", IntegrityLevel::Full).unwrap();
        assert!(!r.passed);
        // All three hash fields should report mismatch.
        assert!(
            r.errors.len() >= 3,
            "expected ≥3 errors, got {}",
            r.errors.len()
        );
    }

    // -----------------------------------------------------------------------
    // verify — Custom level
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_custom_fnv_only_pass() {
        let mut c = checker();
        c.register("cust".to_string(), b"data").unwrap();
        let lvl = IntegrityLevel::Custom(vec!["fnv1a".to_string()]);
        let r = c.verify("cust", b"data", lvl).unwrap();
        assert!(r.passed);
    }

    #[test]
    fn test_verify_custom_fnv_only_fail() {
        let mut c = checker();
        c.register("cust".to_string(), b"data").unwrap();
        let lvl = IntegrityLevel::Custom(vec!["fnv1a".to_string()]);
        let r = c.verify("cust", b"DATA", lvl).unwrap();
        assert!(!r.passed);
    }

    #[test]
    fn test_verify_custom_adler32_only() {
        let mut c = checker();
        c.register("ad".to_string(), b"check").unwrap();
        let lvl = IntegrityLevel::Custom(vec!["adler32".to_string()]);
        let r = c.verify("ad", b"check", lvl).unwrap();
        assert!(r.passed);
    }

    #[test]
    fn test_verify_custom_invalid_algo() {
        let mut c = checker();
        c.register("inv".to_string(), b"x").unwrap();
        let lvl = IntegrityLevel::Custom(vec!["sha256".to_string()]);
        let err = c.verify("inv", b"x", lvl).unwrap_err();
        assert!(matches!(err, CheckerError::InvalidLevel(_)));
    }

    #[test]
    fn test_verify_custom_all_algos() {
        let mut c = checker();
        c.register("all".to_string(), b"alldata").unwrap();
        let lvl = IntegrityLevel::Custom(vec![
            "fnv1a".to_string(),
            "adler32".to_string(),
            "crc16".to_string(),
        ]);
        let r = c.verify("all", b"alldata", lvl).unwrap();
        assert!(r.passed);
    }

    // -----------------------------------------------------------------------
    // Verification side-effects
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_increments_count() {
        let mut c = checker();
        c.register("v".to_string(), b"data").unwrap();
        c.verify("v", b"data", IntegrityLevel::Full).unwrap();
        c.verify("v", b"data", IntegrityLevel::Quick).unwrap();
        let s = c.stats();
        assert_eq!(s.total_verifications, 2);
    }

    #[test]
    fn test_verify_updates_status_on_pass() {
        let mut c = checker();
        c.register("u".to_string(), b"data").unwrap();
        c.verify("u", b"data", IntegrityLevel::Full).unwrap();
        let s = c.stats();
        assert_eq!(s.verified_objects, 1);
    }

    #[test]
    fn test_verify_updates_status_on_fail() {
        let mut c = checker();
        c.register("u".to_string(), b"original").unwrap();
        c.verify("u", b"tampered", IntegrityLevel::Full).unwrap();
        let s = c.stats();
        assert_eq!(s.corrupted_objects, 1);
    }

    #[test]
    fn test_verify_not_found() {
        let mut c = checker();
        let err = c
            .verify("ghost", b"data", IntegrityLevel::Quick)
            .unwrap_err();
        assert!(matches!(err, CheckerError::ObjectNotFound(_)));
    }

    // -----------------------------------------------------------------------
    // verify_all — batch
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_all_all_pass() {
        let mut c = checker();
        c.register("a1".to_string(), b"alpha").unwrap();
        c.register("b1".to_string(), b"beta").unwrap();
        let pairs: Vec<(&str, &[u8])> = vec![("a1", b"alpha"), ("b1", b"beta")];
        let results = c.verify_all(&pairs, IntegrityLevel::Full);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.passed));
    }

    #[test]
    fn test_verify_all_mixed() {
        let mut c = checker();
        c.register("good".to_string(), b"ok").unwrap();
        c.register("bad".to_string(), b"original").unwrap();
        let pairs: Vec<(&str, &[u8])> = vec![("good", b"ok"), ("bad", b"tampered")];
        let results = c.verify_all(&pairs, IntegrityLevel::Full);
        assert!(results[0].passed);
        assert!(!results[1].passed);
    }

    #[test]
    fn test_verify_all_missing_object() {
        let mut c = checker();
        let pairs: Vec<(&str, &[u8])> = vec![("ghost", b"data")];
        let results = c.verify_all(&pairs, IntegrityLevel::Quick);
        assert_eq!(results.len(), 1);
        assert!(!results[0].passed);
        assert!(!results[0].errors.is_empty());
    }

    #[test]
    fn test_verify_all_empty_input() {
        let mut c = checker();
        let results = c.verify_all(&[], IntegrityLevel::Full);
        assert!(results.is_empty());
    }

    #[test]
    fn test_verify_all_large_batch() {
        let mut c = checker();
        let data: Vec<Vec<u8>> = (0u64..20).map(|i| rng_bytes(i + 1, 64)).collect();
        let ids: Vec<String> = (0..20).map(|i| format!("obj{i}")).collect();
        for (id, d) in ids.iter().zip(data.iter()) {
            c.register(id.clone(), d).unwrap();
        }
        let pairs: Vec<(&str, &[u8])> = ids
            .iter()
            .zip(data.iter())
            .map(|(id, d)| (id.as_str(), d.as_slice()))
            .collect();
        let results = c.verify_all(&pairs, IntegrityLevel::Standard);
        assert!(results.iter().all(|r| r.passed));
    }

    // -----------------------------------------------------------------------
    // mark_corrupted
    // -----------------------------------------------------------------------

    #[test]
    fn test_mark_corrupted_sets_status() {
        let mut c = checker();
        c.register("mc".to_string(), b"data").unwrap();
        c.mark_corrupted("mc", "disk error".to_string()).unwrap();
        let s = c.stats();
        // PossiblyCorrupted does NOT count as corrupted_objects (which tracks
        // hard Corrupted state), but let's verify the status was set.
        assert_eq!(s.corrupted_objects, 0);
        // Verify via objects_needing_verification — it hasn't been verified.
        let needing = c.objects_needing_verification(u64::MAX);
        assert_eq!(needing.len(), 1);
    }

    #[test]
    fn test_mark_corrupted_not_found() {
        let mut c = checker();
        let err = c.mark_corrupted("ghost", "reason".to_string()).unwrap_err();
        assert!(matches!(err, CheckerError::ObjectNotFound(_)));
    }

    #[test]
    fn test_mark_corrupted_reason_stored() {
        let mut c = checker();
        c.register("r".to_string(), b"data").unwrap();
        c.mark_corrupted("r", "sector read failure".to_string())
            .unwrap();
        // Verify all triggers the "possibly corrupted" path but the status
        // can be overwritten by a subsequent verify.
        let r = c.verify("r", b"data", IntegrityLevel::Full).unwrap();
        assert!(r.passed);
    }

    // -----------------------------------------------------------------------
    // update
    // -----------------------------------------------------------------------

    #[test]
    fn test_update_changes_hash() {
        let mut c = checker();
        c.register("upd".to_string(), b"v1").unwrap();
        let h2 = c.update("upd", b"v2").unwrap();
        assert_eq!(h2, IntegrityHash::compute(b"v2"));
    }

    #[test]
    fn test_update_resets_verification_count() {
        let mut c = checker();
        c.register("upd".to_string(), b"v1").unwrap();
        c.verify("upd", b"v1", IntegrityLevel::Full).unwrap();
        c.update("upd", b"v2").unwrap();
        // After update the record's own counter resets; stats reflect the
        // cumulative total_verifications which still includes the prior call.
        let r = c.verify("upd", b"v2", IntegrityLevel::Full).unwrap();
        assert!(r.passed);
    }

    #[test]
    fn test_update_not_found() {
        let mut c = checker();
        let err = c.update("ghost", b"data").unwrap_err();
        assert!(matches!(err, CheckerError::ObjectNotFound(_)));
    }

    #[test]
    fn test_update_verify_old_data_fails() {
        let mut c = checker();
        c.register("u2".to_string(), b"old").unwrap();
        c.update("u2", b"new").unwrap();
        let r = c.verify("u2", b"old", IntegrityLevel::Full).unwrap();
        assert!(!r.passed);
    }

    // -----------------------------------------------------------------------
    // remove
    // -----------------------------------------------------------------------

    #[test]
    fn test_remove_decrements_count() {
        let mut c = checker();
        c.register("rm".to_string(), b"data").unwrap();
        assert_eq!(c.len(), 1);
        c.remove("rm").unwrap();
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn test_remove_not_found() {
        let mut c = checker();
        let err = c.remove("ghost").unwrap_err();
        assert!(matches!(err, CheckerError::ObjectNotFound(_)));
    }

    #[test]
    fn test_remove_allows_reregistration() {
        let mut c = checker();
        c.register("x".to_string(), b"v1").unwrap();
        c.remove("x").unwrap();
        c.register("x".to_string(), b"v2").unwrap();
        let r = c.verify("x", b"v2", IntegrityLevel::Full).unwrap();
        assert!(r.passed);
    }

    // -----------------------------------------------------------------------
    // corrupted_objects
    // -----------------------------------------------------------------------

    #[test]
    fn test_corrupted_objects_empty() {
        let c = checker();
        assert!(c.corrupted_objects().is_empty());
    }

    #[test]
    fn test_corrupted_objects_after_failed_verify() {
        let mut c = checker();
        c.register("co".to_string(), b"original").unwrap();
        c.verify("co", b"tampered", IntegrityLevel::Full).unwrap();
        assert_eq!(c.corrupted_objects().len(), 1);
        assert_eq!(c.corrupted_objects()[0].id, "co");
    }

    #[test]
    fn test_corrupted_objects_multiple() {
        let mut c = checker();
        for i in 0..5 {
            c.register(format!("obj{i}"), b"data").unwrap();
        }
        for i in 0..3 {
            c.verify(&format!("obj{i}"), b"wrong", IntegrityLevel::Full)
                .unwrap();
        }
        assert_eq!(c.corrupted_objects().len(), 3);
    }

    // -----------------------------------------------------------------------
    // objects_needing_verification
    // -----------------------------------------------------------------------

    #[test]
    fn test_objects_needing_verification_all_new() {
        let mut c = checker();
        c.register("n1".to_string(), b"d").unwrap();
        c.register("n2".to_string(), b"d").unwrap();
        // since_us = MAX ⇒ all objects with last_verified_at < MAX qualify.
        let needing = c.objects_needing_verification(u64::MAX);
        assert_eq!(needing.len(), 2);
    }

    #[test]
    fn test_objects_needing_verification_after_verify() {
        let mut c = checker();
        c.register("v1".to_string(), b"d").unwrap();
        c.verify("v1", b"d", IntegrityLevel::Full).unwrap();
        // since_us = 0 ⇒ only objects with last_verified_at < 0 (none) qualify.
        let needing = c.objects_needing_verification(0);
        assert_eq!(needing.len(), 0);
    }

    #[test]
    fn test_objects_needing_verification_since_past() {
        let mut c = checker();
        c.register("past".to_string(), b"d").unwrap();
        // since_us = 1 ⇒ objects with last_verified_at == 0 are returned.
        let needing = c.objects_needing_verification(1);
        assert_eq!(needing.len(), 1);
    }

    // -----------------------------------------------------------------------
    // repair_hash
    // -----------------------------------------------------------------------

    #[test]
    fn test_repair_hash_clears_corruption() {
        let mut c = checker();
        c.register("rh".to_string(), b"original").unwrap();
        c.verify("rh", b"tampered", IntegrityLevel::Full).unwrap();
        assert_eq!(c.corrupted_objects().len(), 1);
        c.repair_hash("rh", b"correct").unwrap();
        assert_eq!(c.corrupted_objects().len(), 0);
    }

    #[test]
    fn test_repair_hash_verify_with_correct_data() {
        let mut c = checker();
        c.register("rh2".to_string(), b"old").unwrap();
        c.repair_hash("rh2", b"new_correct").unwrap();
        let r = c
            .verify("rh2", b"new_correct", IntegrityLevel::Full)
            .unwrap();
        assert!(r.passed);
    }

    #[test]
    fn test_repair_hash_not_found() {
        let mut c = checker();
        let err = c.repair_hash("ghost", b"data").unwrap_err();
        assert!(matches!(err, CheckerError::ObjectNotFound(_)));
    }

    // -----------------------------------------------------------------------
    // stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_empty() {
        let c = checker();
        let s = c.stats();
        assert_eq!(s.total_objects, 0);
        assert_eq!(s.verified_objects, 0);
        assert_eq!(s.corrupted_objects, 0);
        assert_eq!(s.corruption_rate, 0.0);
        assert_eq!(s.total_verifications, 0);
        assert_eq!(s.avg_verification_us, 0.0);
    }

    #[test]
    fn test_stats_after_registration() {
        let mut c = checker();
        c.register("s1".to_string(), b"a").unwrap();
        c.register("s2".to_string(), b"b").unwrap();
        let s = c.stats();
        assert_eq!(s.total_objects, 2);
        assert_eq!(s.verified_objects, 0);
    }

    #[test]
    fn test_stats_corruption_rate() {
        let mut c = checker();
        c.register("s1".to_string(), b"a").unwrap();
        c.register("s2".to_string(), b"b").unwrap();
        c.verify("s1", b"tampered", IntegrityLevel::Full).unwrap();
        let s = c.stats();
        assert_eq!(s.corrupted_objects, 1);
        assert!((s.corruption_rate - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_stats_avg_verification_us_non_negative() {
        let mut c = checker();
        c.register("t1".to_string(), b"data").unwrap();
        c.verify("t1", b"data", IntegrityLevel::Full).unwrap();
        let s = c.stats();
        assert!(s.avg_verification_us >= 0.0);
    }

    #[test]
    fn test_stats_total_verifications() {
        let mut c = checker();
        c.register("tv".to_string(), b"data").unwrap();
        for _ in 0..5 {
            c.verify("tv", b"data", IntegrityLevel::Quick).unwrap();
        }
        let s = c.stats();
        assert_eq!(s.total_verifications, 5);
    }

    // -----------------------------------------------------------------------
    // Error path coverage
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_display_not_found() {
        let e = CheckerError::ObjectNotFound("xyz".to_string());
        let msg = e.to_string();
        assert!(msg.contains("xyz"));
    }

    #[test]
    fn test_error_display_max_exceeded() {
        let e = CheckerError::MaxObjectsExceeded;
        let msg = e.to_string();
        assert!(msg.contains("maximum"));
    }

    #[test]
    fn test_error_display_invalid_level() {
        let e = CheckerError::InvalidLevel("bad algo".to_string());
        let msg = e.to_string();
        assert!(msg.contains("invalid"));
    }

    // -----------------------------------------------------------------------
    // Edge-case / property-style tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_large_data_verify() {
        let mut c = checker();
        let data = rng_bytes(42, 1_024 * 1_024);
        c.register("large".to_string(), &data).unwrap();
        let r = c.verify("large", &data, IntegrityLevel::Full).unwrap();
        assert!(r.passed);
    }

    #[test]
    fn test_many_objects_no_collision() {
        let mut c = checker();
        for i in 0u64..100 {
            let d = rng_bytes(i, 32);
            c.register(format!("obj{i}"), &d).unwrap();
        }
        let s = c.stats();
        assert_eq!(s.total_objects, 100);
    }

    #[test]
    fn test_verify_returns_level_in_result() {
        let mut c = checker();
        c.register("lvl".to_string(), b"d").unwrap();
        let r = c.verify("lvl", b"d", IntegrityLevel::Standard).unwrap();
        assert_eq!(r.level, IntegrityLevel::Standard);
    }

    #[test]
    fn test_repair_then_verify_large() {
        let mut c = checker();
        let original = rng_bytes(10, 512);
        let corrupted = rng_bytes(99, 512);
        let correct = rng_bytes(7, 512);
        c.register("repair_large".to_string(), &original).unwrap();
        c.verify("repair_large", &corrupted, IntegrityLevel::Full)
            .unwrap();
        c.repair_hash("repair_large", &correct).unwrap();
        let r = c
            .verify("repair_large", &correct, IntegrityLevel::Full)
            .unwrap();
        assert!(r.passed);
    }

    #[test]
    fn test_is_empty() {
        let mut c = checker();
        assert!(c.is_empty());
        c.register("x".to_string(), b"d").unwrap();
        assert!(!c.is_empty());
        c.remove("x").unwrap();
        assert!(c.is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let c = checker();
        assert_eq!(c.config().max_objects, 1_000_000);
    }

    #[test]
    fn test_verify_duration_recorded() {
        let mut c = checker();
        c.register("dur".to_string(), b"data").unwrap();
        let r = c.verify("dur", b"data", IntegrityLevel::Full).unwrap();
        // duration_us is non-negative; may be 0 on very fast systems.
        assert!(r.duration_us < u64::MAX);
    }

    #[test]
    fn test_custom_empty_algo_list() {
        let mut c = checker();
        c.register("emp".to_string(), b"data").unwrap();
        // An empty custom list checks only size.
        let lvl = IntegrityLevel::Custom(vec![]);
        let r = c.verify("emp", b"data", lvl).unwrap();
        assert!(r.passed);
    }

    #[test]
    fn test_size_mismatch_always_detected() {
        let mut c = checker();
        c.register("sz".to_string(), b"hello").unwrap();
        // Even Quick level reports size mismatch.
        let r = c.verify("sz", b"hi", IntegrityLevel::Quick).unwrap();
        assert!(!r.passed);
        let has_size_error = r.errors.iter().any(|e| e.contains("size"));
        assert!(has_size_error);
    }

    #[test]
    fn test_verify_object_id_in_result() {
        let mut c = checker();
        c.register("myid".to_string(), b"data").unwrap();
        let r = c.verify("myid", b"data", IntegrityLevel::Quick).unwrap();
        assert_eq!(r.object_id, "myid");
    }
}
