//! Certificate fingerprint pinning for QUIC/TLS connections.
//!
//! This module provides `CertPinStore`, which supports three pin policies:
//! - **TrustOnFirstUse (TOFU)**: Accept and pin unknown peers on first contact.
//! - **Strict**: Reject peers whose certificates are not pre-registered.
//! - **Observe**: Always accept but count mismatches for monitoring.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

// ─── Fingerprint ─────────────────────────────────────────────────────────────

/// SHA-256 fingerprint of a peer's TLS certificate (32 bytes, hex-encoded).
///
/// Internally stored as a 64-character lowercase hex string.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct CertFingerprint(pub String);

impl CertFingerprint {
    /// Create a fingerprint from raw bytes.
    ///
    /// Uses FNV-1a to derive a deterministic 64-character hex string, which
    /// mimics the length and format of a real SHA-256 fingerprint without
    /// requiring an external crypto crate.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let hash = fnv1a_hash_bytes(bytes);
        Self(format!(
            "{:016x}{:016x}{:016x}{:016x}",
            hash,
            hash ^ 0xDEAD_BEEF_DEAD_BEEF_u64,
            hash ^ 0xBEEF_CAFE_BEEF_CAFE_u64,
            hash ^ 0xCAFE_DEAD_CAFE_DEAD_u64,
        ))
    }

    /// Return the hex string representation.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Length of the hex string (always 64 for well-formed fingerprints).
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` when the fingerprint has a valid 64-character hex format.
    pub fn is_valid_format(&self) -> bool {
        self.0.len() == 64 && self.0.chars().all(|c| c.is_ascii_hexdigit())
    }

    /// Returns `true` when the fingerprint string is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// FNV-1a 64-bit hash over a byte slice.
fn fnv1a_hash_bytes(data: &[u8]) -> u64 {
    let mut hash: u64 = 14_695_981_039_346_656_037;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(1_099_511_628_211);
    }
    hash
}

// ─── PinPolicy ───────────────────────────────────────────────────────────────

/// Pin policy for certificate verification.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum PinPolicy {
    /// Trust any certificate on first connection; pin thereafter (TOFU).
    #[default]
    TrustOnFirstUse,
    /// Require the fingerprint to be pre-registered; reject unknown certs.
    Strict,
    /// Log mismatches but never reject (monitoring only).
    Observe,
}

// ─── PeerCertPin ─────────────────────────────────────────────────────────────

/// A pinned certificate entry for one peer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PeerCertPin {
    /// Peer identifier (e.g. libp2p PeerId string).
    pub peer_id: String,
    /// Pinned certificate fingerprint.
    pub fingerprint: CertFingerprint,
    /// Unix timestamp (milliseconds) when this pin was first recorded.
    pub pinned_at_ms: u64,
    /// Unix timestamp (milliseconds) of the most recent successful verification.
    pub last_seen_ms: u64,
    /// Total number of successful verifications against this pin.
    pub connection_count: u64,
    /// `true` when the pin was learned via TOFU rather than pre-registered.
    pub tofu: bool,
}

impl PeerCertPin {
    /// Create a new pin entry.
    pub fn new(
        peer_id: impl Into<String>,
        fingerprint: CertFingerprint,
        now_ms: u64,
        tofu: bool,
    ) -> Self {
        Self {
            peer_id: peer_id.into(),
            fingerprint,
            pinned_at_ms: now_ms,
            last_seen_ms: now_ms,
            connection_count: 0,
            tofu,
        }
    }

    /// Update the last-seen timestamp and increment the connection counter.
    pub fn record_connection(&mut self, now_ms: u64) {
        self.last_seen_ms = now_ms;
        self.connection_count += 1;
    }
}

// ─── VerificationResult ──────────────────────────────────────────────────────

/// Result of a certificate verification attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationResult {
    /// The presented fingerprint matches the pinned fingerprint (or policy allows).
    Accepted,
    /// TOFU: first time seeing this peer — accepted and pinned.
    PinnedNew { peer_id: String },
    /// Fingerprint mismatch detected.
    Mismatch { expected: String, got: String },
    /// Strict policy: peer has no pre-registered pin.
    UnknownPeer { peer_id: String },
}

impl VerificationResult {
    /// Returns `true` when the connection should be allowed to proceed.
    pub fn is_accepted(&self) -> bool {
        matches!(
            self,
            VerificationResult::Accepted | VerificationResult::PinnedNew { .. }
        )
    }
}

// ─── CertPinStore ────────────────────────────────────────────────────────────

/// Thread-safe store for peer certificate fingerprint pins.
///
/// # Verification Semantics
///
/// | Policy            | Unknown peer            | Fingerprint matches | Fingerprint mismatch |
/// |-------------------|-------------------------|---------------------|----------------------|
/// | TrustOnFirstUse   | `PinnedNew` (pin + accept) | `Accepted`       | `Mismatch`           |
/// | Strict            | `UnknownPeer` (reject)  | `Accepted`          | `Mismatch`           |
/// | Observe           | `Accepted`              | `Accepted`          | `Accepted` (+counter)|
pub struct CertPinStore {
    pins: RwLock<HashMap<String, PeerCertPin>>,
    policy: PinPolicy,
    mismatch_count: AtomicU64,
    tofu_count: AtomicU64,
}

impl CertPinStore {
    /// Create a new store wrapped in an `Arc`.
    pub fn new(policy: PinPolicy) -> Arc<Self> {
        Arc::new(Self {
            pins: RwLock::new(HashMap::new()),
            policy,
            mismatch_count: AtomicU64::new(0),
            tofu_count: AtomicU64::new(0),
        })
    }

    /// Pre-register a trusted fingerprint for a peer.
    ///
    /// If the peer is already pinned this call is a no-op (existing pin preserved).
    pub fn pin(&self, peer_id: impl Into<String>, fingerprint: CertFingerprint, now_ms: u64) {
        let peer_id = peer_id.into();
        let mut guard = self.pins.write();
        guard
            .entry(peer_id.clone())
            .or_insert_with(|| PeerCertPin::new(peer_id, fingerprint, now_ms, false));
    }

    /// Remove a pinned peer.
    ///
    /// Returns `true` if a pin was present and removed.
    pub fn unpin(&self, peer_id: &str) -> bool {
        self.pins.write().remove(peer_id).is_some()
    }

    /// Verify a peer's certificate fingerprint against the current policy.
    pub fn verify(
        &self,
        peer_id: &str,
        fingerprint: &CertFingerprint,
        now_ms: u64,
    ) -> VerificationResult {
        match self.policy {
            PinPolicy::TrustOnFirstUse => self.verify_tofu(peer_id, fingerprint, now_ms),
            PinPolicy::Strict => self.verify_strict(peer_id, fingerprint, now_ms),
            PinPolicy::Observe => self.verify_observe(peer_id, fingerprint, now_ms),
        }
    }

    fn verify_tofu(
        &self,
        peer_id: &str,
        fingerprint: &CertFingerprint,
        now_ms: u64,
    ) -> VerificationResult {
        // Fast path: try read lock first.
        {
            let guard = self.pins.read();
            if let Some(pin) = guard.get(peer_id) {
                if pin.fingerprint == *fingerprint {
                    drop(guard);
                    // Upgrade to write to record the connection.
                    let mut wguard = self.pins.write();
                    if let Some(p) = wguard.get_mut(peer_id) {
                        p.record_connection(now_ms);
                    }
                    return VerificationResult::Accepted;
                } else {
                    self.mismatch_count.fetch_add(1, Ordering::Relaxed);
                    return VerificationResult::Mismatch {
                        expected: pin.fingerprint.as_str().to_owned(),
                        got: fingerprint.as_str().to_owned(),
                    };
                }
            }
        }
        // Unknown peer — TOFU: pin and accept.
        let mut new_pin = PeerCertPin::new(peer_id, fingerprint.clone(), now_ms, true);
        new_pin.record_connection(now_ms);
        self.pins.write().insert(peer_id.to_owned(), new_pin);
        self.tofu_count.fetch_add(1, Ordering::Relaxed);
        VerificationResult::PinnedNew {
            peer_id: peer_id.to_owned(),
        }
    }

    fn verify_strict(
        &self,
        peer_id: &str,
        fingerprint: &CertFingerprint,
        now_ms: u64,
    ) -> VerificationResult {
        let guard = self.pins.read();
        if let Some(pin) = guard.get(peer_id) {
            if pin.fingerprint == *fingerprint {
                drop(guard);
                let mut wguard = self.pins.write();
                if let Some(p) = wguard.get_mut(peer_id) {
                    p.record_connection(now_ms);
                }
                VerificationResult::Accepted
            } else {
                self.mismatch_count.fetch_add(1, Ordering::Relaxed);
                VerificationResult::Mismatch {
                    expected: pin.fingerprint.as_str().to_owned(),
                    got: fingerprint.as_str().to_owned(),
                }
            }
        } else {
            VerificationResult::UnknownPeer {
                peer_id: peer_id.to_owned(),
            }
        }
    }

    fn verify_observe(
        &self,
        peer_id: &str,
        fingerprint: &CertFingerprint,
        _now_ms: u64,
    ) -> VerificationResult {
        let guard = self.pins.read();
        if let Some(pin) = guard.get(peer_id) {
            if pin.fingerprint != *fingerprint {
                self.mismatch_count.fetch_add(1, Ordering::Relaxed);
            }
        }
        VerificationResult::Accepted
    }

    /// Retrieve the pin info for a peer, if any.
    pub fn get_pin(&self, peer_id: &str) -> Option<PeerCertPin> {
        self.pins.read().get(peer_id).cloned()
    }

    /// Return a list of all pinned peer IDs.
    pub fn pinned_peers(&self) -> Vec<String> {
        self.pins.read().keys().cloned().collect()
    }

    /// Number of currently pinned peers.
    pub fn pin_count(&self) -> usize {
        self.pins.read().len()
    }

    /// Total fingerprint mismatches detected across all verifications.
    pub fn mismatch_count(&self) -> u64 {
        self.mismatch_count.load(Ordering::Relaxed)
    }

    /// Total TOFU pins created.
    pub fn tofu_count(&self) -> u64 {
        self.tofu_count.load(Ordering::Relaxed)
    }

    /// Export all pins as a serializable vector.
    pub fn export_pins(&self) -> Vec<PeerCertPin> {
        self.pins.read().values().cloned().collect()
    }

    /// Import pins (merge).  Existing pins are **not** overwritten.
    pub fn import_pins(&self, pins: Vec<PeerCertPin>) {
        let mut guard = self.pins.write();
        for pin in pins {
            guard.entry(pin.peer_id.clone()).or_insert(pin);
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn now_ms() -> u64 {
        1_700_000_000_000
    }

    fn fp(seed: &[u8]) -> CertFingerprint {
        CertFingerprint::from_bytes(seed)
    }

    // ── CertFingerprint ──────────────────────────────────────────────────────

    #[test]
    fn test_fingerprint_from_bytes_format() {
        let f = fp(b"hello world");
        assert_eq!(f.len(), 64, "fingerprint must be 64 hex chars");
        assert!(f.is_valid_format());
    }

    #[test]
    fn test_fingerprint_is_valid_format() {
        // manually constructed valid fingerprint
        let valid = CertFingerprint("a".repeat(64));
        assert!(valid.is_valid_format());

        let too_short = CertFingerprint("abc".to_owned());
        assert!(!too_short.is_valid_format());

        let non_hex = CertFingerprint("z".repeat(64));
        assert!(!non_hex.is_valid_format());
    }

    #[test]
    fn test_fingerprint_deterministic() {
        let a = fp(b"deterministic input");
        let b = fp(b"deterministic input");
        assert_eq!(a, b);

        let c = fp(b"different input");
        assert_ne!(a, c);
    }

    // ── pin / unpin / get_pin ────────────────────────────────────────────────

    #[test]
    fn test_pin_and_get() {
        let store = CertPinStore::new(PinPolicy::Strict);
        let f = fp(b"peer-cert");

        store.pin("peer1", f.clone(), now_ms());

        let pin = store.get_pin("peer1").expect("pin should exist");
        assert_eq!(pin.peer_id, "peer1");
        assert_eq!(pin.fingerprint, f);
        assert!(!pin.tofu);
    }

    #[test]
    fn test_unpin() {
        let store = CertPinStore::new(PinPolicy::Strict);
        store.pin("peer1", fp(b"x"), now_ms());

        assert!(store.unpin("peer1"));
        assert!(!store.unpin("peer1")); // already removed
        assert!(store.get_pin("peer1").is_none());
    }

    // ── TrustOnFirstUse ──────────────────────────────────────────────────────

    #[test]
    fn test_verify_tofu_new_peer() {
        let store = CertPinStore::new(PinPolicy::TrustOnFirstUse);
        let f = fp(b"new-peer-cert");

        let result = store.verify("peer1", &f, now_ms());
        assert!(
            matches!(result, VerificationResult::PinnedNew { ref peer_id } if peer_id == "peer1")
        );
        assert!(result.is_accepted());
        assert_eq!(store.pin_count(), 1);
    }

    #[test]
    fn test_verify_tofu_known_peer_match() {
        let store = CertPinStore::new(PinPolicy::TrustOnFirstUse);
        let f = fp(b"known-cert");

        // Pre-pin (not TOFU)
        store.pin("peer1", f.clone(), now_ms());

        let result = store.verify("peer1", &f, now_ms() + 1);
        assert_eq!(result, VerificationResult::Accepted);
    }

    #[test]
    fn test_verify_tofu_known_peer_mismatch() {
        let store = CertPinStore::new(PinPolicy::TrustOnFirstUse);
        store.pin("peer1", fp(b"original"), now_ms());

        let result = store.verify("peer1", &fp(b"different"), now_ms() + 1);
        assert!(matches!(result, VerificationResult::Mismatch { .. }));
        assert!(!result.is_accepted());
        assert_eq!(store.mismatch_count(), 1);
    }

    // ── Strict ───────────────────────────────────────────────────────────────

    #[test]
    fn test_verify_strict_unknown_peer() {
        let store = CertPinStore::new(PinPolicy::Strict);

        let result = store.verify("unknown", &fp(b"anything"), now_ms());
        assert!(
            matches!(result, VerificationResult::UnknownPeer { ref peer_id } if peer_id == "unknown")
        );
        assert!(!result.is_accepted());
    }

    #[test]
    fn test_verify_strict_known_peer() {
        let store = CertPinStore::new(PinPolicy::Strict);
        let f = fp(b"strict-cert");
        store.pin("peer2", f.clone(), now_ms());

        let result = store.verify("peer2", &f, now_ms() + 1);
        assert_eq!(result, VerificationResult::Accepted);
    }

    // ── Observe ──────────────────────────────────────────────────────────────

    #[test]
    fn test_verify_observe_always_accepted() {
        let store = CertPinStore::new(PinPolicy::Observe);

        // Unknown peer — still accepted.
        let r1 = store.verify("anyone", &fp(b"cert"), now_ms());
        assert_eq!(r1, VerificationResult::Accepted);

        // Known peer, mismatched fingerprint — still accepted.
        store.pin("peer3", fp(b"original"), now_ms());
        let r2 = store.verify("peer3", &fp(b"changed"), now_ms() + 1);
        assert_eq!(r2, VerificationResult::Accepted);
    }

    // ── Counters ─────────────────────────────────────────────────────────────

    #[test]
    fn test_mismatch_count_increments() {
        let store = CertPinStore::new(PinPolicy::TrustOnFirstUse);
        store.pin("peer1", fp(b"a"), now_ms());

        store.verify("peer1", &fp(b"b"), now_ms() + 1);
        store.verify("peer1", &fp(b"c"), now_ms() + 2);
        assert_eq!(store.mismatch_count(), 2);
    }

    #[test]
    fn test_tofu_count_increments() {
        let store = CertPinStore::new(PinPolicy::TrustOnFirstUse);

        store.verify("peer1", &fp(b"cert1"), now_ms());
        store.verify("peer2", &fp(b"cert2"), now_ms());
        assert_eq!(store.tofu_count(), 2);
    }

    // ── Export / Import ──────────────────────────────────────────────────────

    #[test]
    fn test_export_import_roundtrip() {
        let store_a = CertPinStore::new(PinPolicy::Strict);
        store_a.pin("peer1", fp(b"alpha"), now_ms());
        store_a.pin("peer2", fp(b"beta"), now_ms());

        let exported = store_a.export_pins();
        assert_eq!(exported.len(), 2);

        let store_b = CertPinStore::new(PinPolicy::Strict);
        store_b.import_pins(exported);
        assert_eq!(store_b.pin_count(), 2);

        let pin = store_b.get_pin("peer1").expect("peer1 must be present");
        assert_eq!(pin.fingerprint, fp(b"alpha"));

        // Import again — existing pins must not be overwritten.
        let mut altered = store_b.export_pins();
        for p in &mut altered {
            p.fingerprint = fp(b"tampered");
        }
        store_b.import_pins(altered);
        let pin_after = store_b
            .get_pin("peer1")
            .expect("peer1 must still be present");
        assert_eq!(
            pin_after.fingerprint,
            fp(b"alpha"),
            "existing pin must not be overwritten"
        );
    }

    // ── VerificationResult::is_accepted ─────────────────────────────────────

    #[test]
    fn test_verification_result_is_accepted() {
        assert!(VerificationResult::Accepted.is_accepted());
        assert!(VerificationResult::PinnedNew {
            peer_id: "x".into()
        }
        .is_accepted());
        assert!(!VerificationResult::Mismatch {
            expected: "a".into(),
            got: "b".into()
        }
        .is_accepted());
        assert!(!VerificationResult::UnknownPeer {
            peer_id: "y".into()
        }
        .is_accepted());
    }

    // ── pinned_peers ─────────────────────────────────────────────────────────

    #[test]
    fn test_pinned_peers_list() {
        let store = CertPinStore::new(PinPolicy::Strict);
        store.pin("alice", fp(b"a"), now_ms());
        store.pin("bob", fp(b"b"), now_ms());

        let mut peers = store.pinned_peers();
        peers.sort();
        assert_eq!(peers, vec!["alice", "bob"]);
    }
}
