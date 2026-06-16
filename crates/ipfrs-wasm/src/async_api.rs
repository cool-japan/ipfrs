//! Async-friendly API types for IPFRS WASM bindings.
//!
//! This module provides structured result types for async add/get operations,
//! batch statistics, and lightweight FNV-1a-based CID helpers that are
//! suitable for quick content-identification without SHA-256 overhead.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// AddResult
// ---------------------------------------------------------------------------

/// Result of an async add operation.
///
/// Returned by batch-add helpers; carries both the computed CID and the
/// number of bytes that were stored.
#[wasm_bindgen]
pub struct AddResult {
    cid: String,
    bytes_written: usize,
}

#[wasm_bindgen]
impl AddResult {
    /// Construct a new [`AddResult`].
    #[wasm_bindgen(constructor)]
    pub fn new(cid: String, bytes_written: usize) -> Self {
        Self { cid, bytes_written }
    }

    /// The content identifier of the stored block.
    #[wasm_bindgen(getter)]
    pub fn cid(&self) -> String {
        self.cid.clone()
    }

    /// The number of bytes written to the block store.
    #[wasm_bindgen(getter)]
    pub fn bytes_written(&self) -> usize {
        self.bytes_written
    }
}

// ---------------------------------------------------------------------------
// GetResult
// ---------------------------------------------------------------------------

/// Result of an async get operation.
///
/// Wraps the data bytes and indicates whether the requested CID was found.
/// Use [`GetResult::not_found`] to construct a "miss" result without allocating
/// a data payload.
#[wasm_bindgen]
pub struct GetResult {
    is_found: bool,
    data: Vec<u8>,
    cid: String,
}

#[wasm_bindgen]
impl GetResult {
    /// Construct a "not found" result for the given `cid`.
    ///
    /// This mirrors the interface required by the task spec (`not_found` constructor).
    pub fn not_found(cid: String) -> Self {
        Self {
            is_found: false,
            data: vec![],
            cid,
        }
    }

    /// Whether the block was present in the store.
    #[wasm_bindgen(getter)]
    pub fn found(&self) -> bool {
        self.is_found
    }

    /// Raw block bytes.  Returns an empty `Vec<u8>` when `found` is `false`.
    pub fn data(&self) -> Vec<u8> {
        self.data.clone()
    }

    /// The content identifier that was requested.
    #[wasm_bindgen(getter)]
    pub fn cid(&self) -> String {
        self.cid.clone()
    }
}

impl GetResult {
    /// Construct a "found" result containing `data` and `cid` (Rust-only helper).
    pub fn new_found(cid: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            is_found: true,
            data,
            cid: cid.into(),
        }
    }

    /// Construct a "not found" result (Rust-only, accepts any `Into<String>`).
    pub fn new_not_found(cid: impl Into<String>) -> Self {
        Self {
            is_found: false,
            data: vec![],
            cid: cid.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// BatchStats
// ---------------------------------------------------------------------------

/// Statistics for a batch add or get operation.
///
/// Tracks total items processed, successful operations, and failures.
#[wasm_bindgen]
pub struct BatchStats {
    total: usize,
    succeeded: usize,
    failed: usize,
}

#[wasm_bindgen]
impl BatchStats {
    /// Construct new [`BatchStats`] from raw counts.
    #[wasm_bindgen(constructor)]
    pub fn new(total: usize, succeeded: usize, failed: usize) -> Self {
        Self {
            total,
            succeeded,
            failed,
        }
    }

    /// Total number of items in the batch.
    #[wasm_bindgen(getter)]
    pub fn total(&self) -> usize {
        self.total
    }

    /// Number of items that were processed successfully.
    #[wasm_bindgen(getter)]
    pub fn succeeded(&self) -> usize {
        self.succeeded
    }

    /// Number of items that failed to process.
    #[wasm_bindgen(getter)]
    pub fn failed(&self) -> usize {
        self.failed
    }

    /// Fraction of successful operations in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` when `total` is zero to avoid division by zero.
    pub fn success_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.succeeded as f64 / self.total as f64
        }
    }
}

// ---------------------------------------------------------------------------
// FNV-1a CID helpers
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit hash of `data`.
fn fnv1a_u64(data: &[u8]) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

/// Compute a lightweight CID for raw bytes using FNV-1a.
///
/// The result has a `"baf"` multibase-like prefix followed by a 16-character
/// lowercase hexadecimal encoding of the 64-bit FNV-1a digest, giving strings
/// of the form `"baf<016 hex digits>"`.
///
/// This is intentionally distinct from the SHA-256-based CIDv1 produced by
/// [`crate::compute_cid`]; it is suitable for fast content-tagging in tests
/// and batch helpers where cryptographic security is not required.
pub fn compute_cid_for_bytes(data: &[u8]) -> String {
    let hash = fnv1a_u64(data);
    format!("baf{hash:016x}")
}

/// Return `true` when `cid` was produced by [`compute_cid_for_bytes`] for `data`.
pub fn verify_cid_for_bytes(cid: &str, data: &[u8]) -> bool {
    compute_cid_for_bytes(data) == cid
}

// ---------------------------------------------------------------------------
// Tests – native target only
// ---------------------------------------------------------------------------

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // AddResult
    // ------------------------------------------------------------------

    #[test]
    fn test_add_result_getters() {
        let r = AddResult::new("bafabc123".to_string(), 42usize);
        assert_eq!(r.cid(), "bafabc123");
        assert_eq!(r.bytes_written(), 42);
    }

    // ------------------------------------------------------------------
    // GetResult
    // ------------------------------------------------------------------

    #[test]
    fn test_get_result_found() {
        let payload = vec![1u8, 2, 3];
        let r = GetResult::new_found("bafxyz", payload.clone());
        assert!(r.found());
        assert_eq!(r.data(), payload);
        assert_eq!(r.cid(), "bafxyz");
    }

    #[test]
    fn test_get_result_not_found() {
        let r = GetResult::new_not_found("bafmissing");
        assert!(!r.found());
        assert!(r.data().is_empty());
        assert_eq!(r.cid(), "bafmissing");
    }

    // ------------------------------------------------------------------
    // BatchStats
    // ------------------------------------------------------------------

    #[test]
    fn test_batch_stats_success_rate() {
        let s = BatchStats::new(10, 7, 3);
        assert_eq!(s.total(), 10);
        assert_eq!(s.succeeded(), 7);
        assert_eq!(s.failed(), 3);
        let rate = s.success_rate();
        assert!(
            (rate - 0.7f64).abs() < f64::EPSILON,
            "expected 0.7, got {rate}"
        );
    }

    #[test]
    fn test_batch_stats_zero_total() {
        let s = BatchStats::new(0, 0, 0);
        assert_eq!(s.success_rate(), 0.0);
    }

    // ------------------------------------------------------------------
    // compute_cid_for_bytes / verify_cid_for_bytes
    // ------------------------------------------------------------------

    #[test]
    fn test_compute_cid_deterministic() {
        let a = compute_cid_for_bytes(b"same input");
        let b = compute_cid_for_bytes(b"same input");
        assert_eq!(a, b, "same input must always produce the same CID");
    }

    #[test]
    fn test_compute_cid_different_inputs() {
        let a = compute_cid_for_bytes(b"alpha data");
        let b = compute_cid_for_bytes(b"beta data");
        assert_ne!(a, b, "different inputs must produce different CIDs");
    }

    #[test]
    fn test_compute_cid_prefix() {
        let cid = compute_cid_for_bytes(b"prefix check");
        assert!(
            cid.starts_with("baf"),
            "CID must start with 'baf', got: {cid}"
        );
        // "baf" + 16 hex chars = 19 chars total
        assert_eq!(cid.len(), 19, "unexpected CID length: {}", cid.len());
    }

    #[test]
    fn test_verify_cid_correct() {
        let data = b"verify correct";
        let cid = compute_cid_for_bytes(data);
        assert!(
            verify_cid_for_bytes(&cid, data),
            "verify_cid_for_bytes must return true for matching CID"
        );
    }

    #[test]
    fn test_verify_cid_wrong() {
        let data = b"verify wrong";
        let cid = compute_cid_for_bytes(data);
        assert!(
            !verify_cid_for_bytes(&cid, b"different payload"),
            "verify_cid_for_bytes must return false for non-matching data"
        );
        assert!(
            !verify_cid_for_bytes("baf0000000000000000", data),
            "verify_cid_for_bytes must return false for wrong CID"
        );
    }
}
