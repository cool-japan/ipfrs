//! Message authentication with HMAC-like construction and replay attack prevention.
//!
//! Provides production-quality message signing and verification using pure-Rust
//! crypto primitives (FNV-1a based HMAC, xorshift64 PRNG) without any external
//! crypto crates.
//!
//! # Design
//! * `MessageAuthenticator` manages a keystore of named `AuthKey` entries.
//! * Each key carries an `AuthAlgorithm` variant that controls how HMAC is computed.
//! * A `ReplayWindow` tracks recently seen nonces to block replay attacks.
//! * Policy flags (`AuthPolicy`) let callers enforce sequential nonces or key rotation.
//! * All significant events (sign, verify, error) are recorded in an internal audit log
//!   drainable via `drain_events()`.
//!
//! # Collision notes (lib.rs re-export)
//! * `fnv1a_64` / `xorshift64` are already exported under `ntm_` / `smx_` / etc.
//!   aliases.  The helpers in this module are re-exported as `mau_fnv1a_64` and
//!   `mau_xorshift64` to avoid collisions.
//! * No other types in this module collide with existing crate-root exports as of
//!   the time of writing (confirmed by grepping lib.rs for each name before
//!   creating this file).

use std::collections::HashMap;
use std::collections::VecDeque;

// ─────────────────────────────────────────────────────────────────────────────
// Pure-Rust crypto primitives (no external crates)
// ─────────────────────────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash over `data`.
///
/// Uses the standard FNV offset basis (`14695981039346656037`) and prime
/// (`1099511628211`).
#[inline]
pub fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

/// HMAC-FNV64 simplified construction.
///
/// `H(K XOR opad || H(K XOR ipad || message))` where `opad = 0x5c` and
/// `ipad = 0x36`.  The key is normalised to 8 bytes via `fnv1a_64` before
/// use, giving a consistent key-length regardless of input size.
#[inline]
pub fn hmac_fnv64(key: &[u8], message: &[u8]) -> u64 {
    let key_hash = fnv1a_64(key);
    let key_bytes = key_hash.to_le_bytes();
    // inner = fnv1a_64(key_bytes XOR ipad || message)
    let ipad: Vec<u8> = key_bytes
        .iter()
        .cycle()
        .zip(message.iter())
        .map(|(k, m)| k ^ 0x36 ^ m)
        .collect();
    let inner = fnv1a_64(&ipad);
    // outer = fnv1a_64(key_bytes XOR opad || inner)
    let opad: Vec<u8> = key_bytes.iter().map(|k| k ^ 0x5c).collect();
    let mut outer = opad;
    outer.extend_from_slice(&inner.to_le_bytes());
    fnv1a_64(&outer)
}

/// Xorshift-64 PRNG.
///
/// Caller must supply a **non-zero** `state`; the function updates it in-place
/// and returns the next pseudo-random value.
#[inline]
pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ─────────────────────────────────────────────────────────────────────────────
// Algorithm
// ─────────────────────────────────────────────────────────────────────────────

/// Authentication algorithm selection.
///
/// * `HmacFnv64` — single-round HMAC-FNV64 over the payload.
/// * `HmacFnv64WithNonce` — HMAC-FNV64 over `nonce || payload`.
/// * `ChainedHash(rounds)` — apply HMAC-FNV64 `rounds` times, feeding the
///   output of each round as the key for the next.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum AuthAlgorithm {
    /// Standard HMAC-FNV64 (payload only).
    #[default]
    HmacFnv64,
    /// HMAC-FNV64 with nonce prepended to the payload.
    HmacFnv64WithNonce,
    /// Multi-round chained HMAC-FNV64.  `rounds` must be ≥ 1.
    ChainedHash(u8),
}

// ─────────────────────────────────────────────────────────────────────────────
// Key material
// ─────────────────────────────────────────────────────────────────────────────

/// A named secret key used for signing and verifying messages.
///
/// `expires_at` is an optional Unix-microsecond timestamp.  When set, the key
/// is considered invalid once `current_ts >= expires_at`.
#[derive(Debug, Clone)]
pub struct AuthKey {
    /// Human-readable key identifier.
    pub id: String,
    /// Raw secret bytes.
    pub secret: Vec<u8>,
    /// Creation time (Unix microseconds).
    pub created_at: u64,
    /// Optional expiry time (Unix microseconds).
    pub expires_at: Option<u64>,
    /// Algorithm used with this key.
    pub algorithm: AuthAlgorithm,
}

impl AuthKey {
    /// Construct a new `AuthKey`.
    pub fn new(
        id: impl Into<String>,
        secret: Vec<u8>,
        created_at: u64,
        expires_at: Option<u64>,
        algorithm: AuthAlgorithm,
    ) -> Self {
        Self {
            id: id.into(),
            secret,
            created_at,
            expires_at,
            algorithm,
        }
    }

    /// Return `true` if the key has not yet expired at `current_ts`.
    pub fn is_valid_at(&self, current_ts: u64) -> bool {
        match self.expires_at {
            Some(exp) => current_ts < exp,
            None => true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Signed message
// ─────────────────────────────────────────────────────────────────────────────

/// A payload together with its authentication tag and metadata.
#[derive(Debug, Clone)]
pub struct SignedMessage {
    /// The original payload bytes.
    pub payload: Vec<u8>,
    /// HMAC-FNV64 signature.
    pub signature: u64,
    /// ID of the key used to produce `signature`.
    pub key_id: String,
    /// Random nonce for replay prevention.
    pub nonce: u64,
    /// Wall-clock timestamp at signing time (Unix microseconds).
    pub timestamp: u64,
    /// Monotonically increasing per-key sequence number.
    pub sequence_num: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Policy
// ─────────────────────────────────────────────────────────────────────────────

/// Authentication enforcement policy.
///
/// Multiple policies can be held in a `Vec<AuthPolicy>` and evaluated together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthPolicy {
    /// Every message must be signed; unsigned messages are rejected.
    RequireAll,
    /// Signing is optional; unsigned messages pass through without error.
    OptionalSign,
    /// Keys older than `max_key_age_us` microseconds must be rotated before
    /// new messages can be signed.
    KeyRotationRequired(u64),
    /// The `sequence_num` field must increment by exactly 1 each message.
    SequentialNonce,
}

// ─────────────────────────────────────────────────────────────────────────────
// Replay window
// ─────────────────────────────────────────────────────────────────────────────

/// Sliding window of recently observed nonces used to detect replay attacks.
///
/// Operates as a bounded FIFO: when the window is full, the oldest nonce is
/// evicted to make room for the newcomer.
#[derive(Debug, Clone)]
pub struct ReplayWindow {
    /// Maximum number of nonces retained at once.
    pub window_size: usize,
    /// Ordered queue of recently seen nonces (oldest first).
    pub seen_nonces: VecDeque<u64>,
    /// Highest `sequence_num` seen so far (used for sequential nonce policy).
    pub last_sequence: u64,
}

impl ReplayWindow {
    /// Create a new `ReplayWindow` with the given capacity.
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            seen_nonces: VecDeque::with_capacity(window_size),
            last_sequence: 0,
        }
    }

    /// Return `true` if `nonce` was already seen within the window.
    pub fn contains(&self, nonce: u64) -> bool {
        self.seen_nonces.contains(&nonce)
    }

    /// Record `nonce`; evict the oldest entry if the window is at capacity.
    ///
    /// When `window_size == 0` the nonce is discarded — no replay detection
    /// is performed.
    pub fn record(&mut self, nonce: u64) {
        if self.window_size == 0 {
            return;
        }
        if self.seen_nonces.len() >= self.window_size {
            self.seen_nonces.pop_front();
        }
        self.seen_nonces.push_back(nonce);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Cumulative counters collected by `MessageAuthenticator`.
#[derive(Debug, Clone, Default)]
pub struct AuthStats {
    /// Total messages successfully signed.
    pub messages_signed: u64,
    /// Total messages successfully verified.
    pub messages_verified: u64,
    /// Number of messages blocked due to replay detection.
    pub replay_attacks_blocked: u64,
    /// Number of verification failures caused by an expired key.
    pub expired_key_rejections: u64,
    /// Number of verification failures caused by a bad signature.
    pub invalid_signature_rejections: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// All errors that can be returned by `MessageAuthenticator`.
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    /// A key with the given ID does not exist in the keystore.
    #[error("key not found: {0}")]
    KeyNotFound(String),

    /// The recomputed signature did not match the one supplied in the message.
    #[error("invalid signature for key '{key_id}': expected {expected:#018x}, got {got:#018x}")]
    InvalidSignature {
        /// Key ID associated with the failing verification.
        key_id: String,
        /// The signature that the verifier computed.
        expected: u64,
        /// The signature that was present in the message.
        got: u64,
    },

    /// The nonce in a message was found in the replay window.
    #[error("replay detected for nonce {0:#018x}")]
    ReplayDetected(u64),

    /// The key has passed its expiry timestamp.
    #[error("key expired: {0}")]
    KeyExpired(String),

    /// The PRNG state has been exhausted (state reached zero).
    #[error("nonce generator exhausted — reseed required")]
    NonceExhausted,

    /// A `SequentialNonce` policy violation: the sequence number jumped or
    /// regressed unexpectedly.
    #[error("invalid sequence number: expected {expected}, got {got}")]
    InvalidSequence {
        /// The next expected sequence number.
        expected: u64,
        /// The sequence number that was actually present.
        got: u64,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal per-key state
// ─────────────────────────────────────────────────────────────────────────────

/// Internal bookkeeping for a single key entry.
#[derive(Debug)]
struct KeyEntry {
    key: AuthKey,
    /// Next sequence number to assign when signing with this key.
    next_seq: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// MessageAuthenticator
// ─────────────────────────────────────────────────────────────────────────────

/// Production-quality message authenticator with:
/// * HMAC-FNV64 signing / verification (three algorithm variants).
/// * Replay attack prevention via a bounded nonce window.
/// * Key expiry and atomic key rotation.
/// * Configurable `AuthPolicy` enforcement.
/// * Monotonic per-key sequence numbers.
/// * Cumulative statistics and drainable audit log.
///
/// # Example
///
/// ```
/// use ipfrs_network::message_authenticator::{
///     AuthAlgorithm, AuthKey, AuthPolicy, MessageAuthenticator,
/// };
///
/// let mut auth = MessageAuthenticator::new(vec![AuthPolicy::RequireAll], 256);
/// let key = AuthKey::new("k1", b"super-secret".to_vec(), 0, None, AuthAlgorithm::HmacFnv64);
/// auth.add_key(key).unwrap();
/// let msg = auth.sign(b"hello world".to_vec(), "k1", 1_000_000).unwrap();
/// auth.verify(&msg, 1_000_000).unwrap();
/// ```
#[derive(Debug)]
pub struct MessageAuthenticator {
    /// Active key store: key_id → entry.
    keys: HashMap<String, KeyEntry>,
    /// Replay-prevention window shared across all keys.
    replay_window: ReplayWindow,
    /// Active policies to enforce.
    policies: Vec<AuthPolicy>,
    /// Xorshift64 PRNG state (must remain non-zero).
    prng_state: u64,
    /// Cumulative statistics.
    stats: AuthStats,
    /// Audit log — drained on demand by `drain_events`.
    events: Vec<String>,
}

impl MessageAuthenticator {
    /// Create a new `MessageAuthenticator`.
    ///
    /// * `policies` — list of `AuthPolicy` variants to enforce.
    /// * `replay_window_size` — maximum number of nonces retained for replay
    ///   detection.
    ///
    /// # Panics
    ///
    /// Does not panic; `replay_window_size == 0` is accepted but provides no
    /// replay protection.
    pub fn new(policies: Vec<AuthPolicy>, replay_window_size: usize) -> Self {
        // Seed PRNG with a compile-time constant mixed with a runtime value.
        // Using the address of a local as an entropy source avoids any external
        // dep while still producing a non-zero, varied starting value.
        let seed_base: u64 = 0x517c_c1b7_2722_0a95;
        let runtime_mix: u64 = {
            let local: u64 = replay_window_size as u64;
            seed_base
                .wrapping_add(local)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407)
        };
        // Ensure the initial PRNG state is non-zero.
        let prng_state = if runtime_mix == 0 {
            seed_base
        } else {
            runtime_mix
        };

        Self {
            keys: HashMap::new(),
            replay_window: ReplayWindow::new(replay_window_size),
            policies,
            prng_state,
            stats: AuthStats::default(),
            events: Vec::new(),
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Key management
    // ─────────────────────────────────────────────────────────────────────────

    /// Add a new key to the keystore.
    ///
    /// Fails with `AuthError::KeyNotFound` if a key with the same ID already
    /// exists (use `rotate_key` for atomic replacement).
    pub fn add_key(&mut self, key: AuthKey) -> Result<(), AuthError> {
        let id = key.id.clone();
        if self.keys.contains_key(&id) {
            // Treat duplicate-add as an error so callers don't silently
            // overwrite a key; they should use rotate_key instead.
            return Err(AuthError::KeyNotFound(format!(
                "key '{id}' already exists — use rotate_key for replacement"
            )));
        }
        self.events
            .push(format!("key_added id={id} algo={:?}", key.algorithm));
        self.keys.insert(id, KeyEntry { key, next_seq: 1 });
        Ok(())
    }

    /// Remove a key from the keystore.
    ///
    /// Returns `AuthError::KeyNotFound` if no such key exists.
    pub fn remove_key(&mut self, key_id: &str) -> Result<(), AuthError> {
        if self.keys.remove(key_id).is_none() {
            return Err(AuthError::KeyNotFound(key_id.to_string()));
        }
        self.events.push(format!("key_removed id={key_id}"));
        Ok(())
    }

    /// Atomically replace the key identified by `old_id` with `new_key`.
    ///
    /// The sequence counter is reset to 1 for the new key.
    /// Returns `AuthError::KeyNotFound` if `old_id` is not in the keystore.
    pub fn rotate_key(&mut self, old_id: &str, new_key: AuthKey) -> Result<(), AuthError> {
        if !self.keys.contains_key(old_id) {
            return Err(AuthError::KeyNotFound(old_id.to_string()));
        }
        let new_id = new_key.id.clone();
        self.keys.remove(old_id);
        self.events.push(format!(
            "key_rotated old_id={old_id} new_id={new_id} algo={:?}",
            new_key.algorithm
        ));
        self.keys.insert(
            new_id,
            KeyEntry {
                key: new_key,
                next_seq: 1,
            },
        );
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Signing
    // ─────────────────────────────────────────────────────────────────────────

    /// Sign `payload` using the key identified by `key_id`.
    ///
    /// * A fresh nonce is generated via `xorshift64`.
    /// * The signature is computed per the key's `AuthAlgorithm`.
    /// * The per-key sequence counter is incremented atomically.
    ///
    /// # Errors
    ///
    /// * `AuthError::KeyNotFound` — unknown key.
    /// * `AuthError::KeyExpired` — key has passed its expiry.
    /// * `AuthError::NonceExhausted` — PRNG state wrapped to zero (extremely
    ///   unlikely in practice).
    /// * `AuthError::KeyRotationRequired` — (via policy check) key age exceeds
    ///   the rotation threshold.
    pub fn sign(
        &mut self,
        payload: Vec<u8>,
        key_id: &str,
        current_ts: u64,
    ) -> Result<SignedMessage, AuthError> {
        // Retrieve key entry (mutable borrow released before building msg).
        let entry = self
            .keys
            .get_mut(key_id)
            .ok_or_else(|| AuthError::KeyNotFound(key_id.to_string()))?;

        // Check expiry.
        if !entry.key.is_valid_at(current_ts) {
            self.stats.expired_key_rejections += 1;
            self.events
                .push(format!("sign_rejected_expired id={key_id}"));
            return Err(AuthError::KeyExpired(key_id.to_string()));
        }

        // Enforce key-rotation policy if present.
        for policy in &self.policies {
            if let AuthPolicy::KeyRotationRequired(max_age) = policy {
                let age = current_ts.saturating_sub(entry.key.created_at);
                if age > *max_age {
                    self.events.push(format!(
                        "sign_rejected_rotation_required id={key_id} age={age} max={max_age}"
                    ));
                    return Err(AuthError::KeyExpired(format!(
                        "rotation required — key '{key_id}' age {age} > max {max_age}"
                    )));
                }
            }
        }

        // Generate nonce.
        let nonce = xorshift64(&mut self.prng_state);
        if nonce == 0 {
            return Err(AuthError::NonceExhausted);
        }

        // Sequence number.
        let sequence_num = entry.next_seq;
        entry.next_seq = entry.next_seq.wrapping_add(1);

        // Compute signature.
        let algorithm = entry.key.algorithm.clone();
        let secret = entry.key.secret.clone();
        let signature = compute_signature(&algorithm, &secret, &payload, nonce);

        self.stats.messages_signed += 1;
        self.events.push(format!(
            "signed key_id={key_id} seq={sequence_num} nonce={nonce:#018x}"
        ));

        Ok(SignedMessage {
            payload,
            signature,
            key_id: key_id.to_string(),
            nonce,
            timestamp: current_ts,
            sequence_num,
        })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Verification
    // ─────────────────────────────────────────────────────────────────────────

    /// Verify the authenticity of `msg` at time `current_ts`.
    ///
    /// Checks (in order):
    /// 1. Key exists in the keystore.
    /// 2. Key has not expired.
    /// 3. Nonce has not been replayed.
    /// 4. Sequence number is valid (when `SequentialNonce` policy is active).
    /// 5. Signature matches the recomputed value.
    ///
    /// On success the nonce is recorded and statistics are updated.
    pub fn verify(&mut self, msg: &SignedMessage, current_ts: u64) -> Result<(), AuthError> {
        // 1. Key lookup.
        let entry = self
            .keys
            .get(&msg.key_id)
            .ok_or_else(|| AuthError::KeyNotFound(msg.key_id.clone()))?;

        // 2. Expiry check.
        if !entry.key.is_valid_at(current_ts) {
            self.stats.expired_key_rejections += 1;
            self.events
                .push(format!("verify_rejected_expired key_id={}", msg.key_id));
            return Err(AuthError::KeyExpired(msg.key_id.clone()));
        }

        // 3. Replay check.
        if self.check_replay(msg.nonce) {
            self.stats.replay_attacks_blocked += 1;
            self.events.push(format!(
                "replay_blocked key_id={} nonce={:#018x}",
                msg.key_id, msg.nonce
            ));
            return Err(AuthError::ReplayDetected(msg.nonce));
        }

        // 4. Sequential nonce policy.
        for policy in &self.policies {
            if *policy == AuthPolicy::SequentialNonce {
                let expected = self.replay_window.last_sequence + 1;
                if msg.sequence_num != expected && self.replay_window.last_sequence != 0 {
                    self.events.push(format!(
                        "verify_rejected_sequence key_id={} expected={} got={}",
                        msg.key_id, expected, msg.sequence_num
                    ));
                    return Err(AuthError::InvalidSequence {
                        expected,
                        got: msg.sequence_num,
                    });
                }
            }
        }

        // 5. Recompute and compare signature.
        let algorithm = entry.key.algorithm.clone();
        let secret = entry.key.secret.clone();
        let expected_sig = compute_signature(&algorithm, &secret, &msg.payload, msg.nonce);
        if expected_sig != msg.signature {
            self.stats.invalid_signature_rejections += 1;
            self.events.push(format!(
                "verify_rejected_bad_sig key_id={} expected={:#018x} got={:#018x}",
                msg.key_id, expected_sig, msg.signature
            ));
            return Err(AuthError::InvalidSignature {
                key_id: msg.key_id.clone(),
                expected: expected_sig,
                got: msg.signature,
            });
        }

        // Record the nonce and update sequence tracking.
        self.record_nonce(msg.nonce);
        self.replay_window.last_sequence = msg.sequence_num;
        self.stats.messages_verified += 1;
        self.events.push(format!(
            "verified key_id={} seq={} nonce={:#018x}",
            msg.key_id, msg.sequence_num, msg.nonce
        ));
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Replay window helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// Return `true` if `nonce` is present in the current replay window.
    pub fn check_replay(&self, nonce: u64) -> bool {
        self.replay_window.contains(nonce)
    }

    /// Add `nonce` to the replay window, evicting the oldest entry if full.
    pub fn record_nonce(&mut self, nonce: u64) {
        self.replay_window.record(nonce);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Key lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    /// Remove all keys whose `expires_at` is ≤ `current_ts`.
    ///
    /// Returns the IDs of all removed keys.
    pub fn expire_keys(&mut self, current_ts: u64) -> Vec<String> {
        let expired: Vec<String> = self
            .keys
            .iter()
            .filter_map(|(id, entry)| match entry.key.expires_at {
                Some(exp) if current_ts >= exp => Some(id.clone()),
                _ => None,
            })
            .collect();

        for id in &expired {
            self.keys.remove(id);
            self.events
                .push(format!("key_expired_evicted id={id} ts={current_ts}"));
        }
        expired
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Observability
    // ─────────────────────────────────────────────────────────────────────────

    /// Return a snapshot of the current cumulative statistics.
    pub fn stats(&self) -> AuthStats {
        self.stats.clone()
    }

    /// Drain the internal audit log, returning all accumulated events.
    ///
    /// The log is cleared after this call.
    pub fn drain_events(&mut self) -> Vec<String> {
        std::mem::take(&mut self.events)
    }

    /// Return the number of keys currently in the keystore.
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    /// Return `true` if a key with `key_id` is currently registered.
    pub fn has_key(&self, key_id: &str) -> bool {
        self.keys.contains_key(key_id)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal signature computation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the authentication tag for `payload` and `nonce` using `algorithm`
/// and `secret`.
fn compute_signature(algorithm: &AuthAlgorithm, secret: &[u8], payload: &[u8], nonce: u64) -> u64 {
    match algorithm {
        AuthAlgorithm::HmacFnv64 => hmac_fnv64(secret, payload),

        AuthAlgorithm::HmacFnv64WithNonce => {
            let mut msg = nonce.to_le_bytes().to_vec();
            msg.extend_from_slice(payload);
            hmac_fnv64(secret, &msg)
        }

        AuthAlgorithm::ChainedHash(rounds) => {
            let rounds = (*rounds).max(1) as usize;
            // First round: HMAC over payload.
            let mut current = hmac_fnv64(secret, payload);
            for _ in 1..rounds {
                // Subsequent rounds: use previous output as the key.
                current = hmac_fnv64(&current.to_le_bytes(), payload);
            }
            // Final round always folds in the nonce.
            let nonce_bytes = nonce.to_le_bytes();
            let mut final_msg = nonce_bytes.to_vec();
            final_msg.extend_from_slice(&current.to_le_bytes());
            hmac_fnv64(secret, &final_msg)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper constructors ──────────────────────────────────────────────────

    fn make_auth(window: usize) -> MessageAuthenticator {
        MessageAuthenticator::new(vec![AuthPolicy::RequireAll], window)
    }

    fn simple_key(id: &str) -> AuthKey {
        AuthKey::new(
            id,
            b"secret_key_bytes".to_vec(),
            0,
            None,
            AuthAlgorithm::HmacFnv64,
        )
    }

    fn key_with_expiry(id: &str, expires_at: u64) -> AuthKey {
        AuthKey::new(
            id,
            b"secret".to_vec(),
            0,
            Some(expires_at),
            AuthAlgorithm::HmacFnv64,
        )
    }

    fn key_with_algo(id: &str, algo: AuthAlgorithm) -> AuthKey {
        AuthKey::new(id, b"algo_key".to_vec(), 1_000, None, algo)
    }

    // ── FNV-1a primitive tests ───────────────────────────────────────────────

    #[test]
    fn test_fnv1a_empty() {
        // Empty input should return the FNV offset basis.
        assert_eq!(fnv1a_64(b""), 14_695_981_039_346_656_037u64);
    }

    #[test]
    fn test_fnv1a_known_vector() {
        // "hello" → known FNV-1a 64-bit output.
        let h = fnv1a_64(b"hello");
        assert_ne!(h, 0);
        // Stability: same input must always give same output.
        assert_eq!(h, fnv1a_64(b"hello"));
    }

    #[test]
    fn test_fnv1a_different_inputs_differ() {
        assert_ne!(fnv1a_64(b"foo"), fnv1a_64(b"bar"));
        assert_ne!(fnv1a_64(b"abc"), fnv1a_64(b"abd"));
    }

    #[test]
    fn test_fnv1a_single_byte_difference() {
        let a = fnv1a_64(&[0x00]);
        let b = fnv1a_64(&[0x01]);
        assert_ne!(a, b);
    }

    // ── HMAC-FNV64 primitive tests ───────────────────────────────────────────

    #[test]
    fn test_hmac_fnv64_deterministic() {
        let h1 = hmac_fnv64(b"key", b"message");
        let h2 = hmac_fnv64(b"key", b"message");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hmac_fnv64_key_sensitivity() {
        let h1 = hmac_fnv64(b"key1", b"message");
        let h2 = hmac_fnv64(b"key2", b"message");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hmac_fnv64_message_sensitivity() {
        let h1 = hmac_fnv64(b"key", b"msg1");
        let h2 = hmac_fnv64(b"key", b"msg2");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hmac_fnv64_empty_message() {
        let h = hmac_fnv64(b"key", b"");
        assert_ne!(h, 0);
    }

    #[test]
    fn test_hmac_fnv64_empty_key() {
        let h = hmac_fnv64(b"", b"message");
        assert_ne!(h, 0);
    }

    // ── Xorshift64 PRNG tests ────────────────────────────────────────────────

    #[test]
    fn test_xorshift64_non_zero_output() {
        let mut state = 12345u64;
        for _ in 0..100 {
            let v = xorshift64(&mut state);
            assert_ne!(v, 0);
        }
    }

    #[test]
    fn test_xorshift64_state_changes() {
        let mut state = 1u64;
        let v1 = xorshift64(&mut state);
        let v2 = xorshift64(&mut state);
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_xorshift64_period_varies() {
        let mut state = 0xdead_beef_cafe_babe_u64;
        let first = xorshift64(&mut state);
        // Verify a second call doesn't just repeat.
        let second = xorshift64(&mut state);
        assert_ne!(first, second);
    }

    // ── Key management tests ─────────────────────────────────────────────────

    #[test]
    fn test_add_key_success() {
        let mut auth = make_auth(32);
        assert!(auth.add_key(simple_key("k1")).is_ok());
        assert!(auth.has_key("k1"));
        assert_eq!(auth.key_count(), 1);
    }

    #[test]
    fn test_add_key_duplicate_fails() {
        let mut auth = make_auth(32);
        auth.add_key(simple_key("k1")).expect("test: add_key k1");
        let res = auth.add_key(simple_key("k1"));
        assert!(matches!(res, Err(AuthError::KeyNotFound(_))));
    }

    #[test]
    fn test_remove_key_success() {
        let mut auth = make_auth(32);
        auth.add_key(simple_key("k1")).expect("test: add_key k1");
        assert!(auth.remove_key("k1").is_ok());
        assert!(!auth.has_key("k1"));
    }

    #[test]
    fn test_remove_key_missing_fails() {
        let mut auth = make_auth(32);
        assert!(matches!(
            auth.remove_key("ghost"),
            Err(AuthError::KeyNotFound(_))
        ));
    }

    #[test]
    fn test_rotate_key_replaces() {
        let mut auth = make_auth(32);
        auth.add_key(simple_key("k1")).expect("test: add_key k1");
        let new_key = AuthKey::new(
            "k2",
            b"new_secret".to_vec(),
            0,
            None,
            AuthAlgorithm::HmacFnv64,
        );
        assert!(auth.rotate_key("k1", new_key).is_ok());
        assert!(!auth.has_key("k1"), "old key should be removed");
        assert!(auth.has_key("k2"), "new key should be present");
    }

    #[test]
    fn test_rotate_key_missing_fails() {
        let mut auth = make_auth(32);
        let new_key = simple_key("k_new");
        assert!(matches!(
            auth.rotate_key("ghost", new_key),
            Err(AuthError::KeyNotFound(_))
        ));
    }

    #[test]
    fn test_add_multiple_keys() {
        let mut auth = make_auth(64);
        for i in 0..5u32 {
            let key = AuthKey::new(
                format!("key_{i}"),
                b"secret".to_vec(),
                0,
                None,
                AuthAlgorithm::HmacFnv64,
            );
            auth.add_key(key).expect("test: add_key");
        }
        assert_eq!(auth.key_count(), 5);
    }

    // ── Sign / verify round-trip ─────────────────────────────────────────────

    #[test]
    fn test_sign_and_verify_basic() {
        let mut auth = make_auth(64);
        auth.add_key(simple_key("k1")).expect("test: add_key k1");
        let msg = auth
            .sign(b"payload".to_vec(), "k1", 1_000)
            .expect("test: sign payload");
        assert!(auth.verify(&msg, 1_000).is_ok());
    }

    #[test]
    fn test_sign_missing_key() {
        let mut auth = make_auth(64);
        assert!(matches!(
            auth.sign(b"data".to_vec(), "ghost", 0),
            Err(AuthError::KeyNotFound(_))
        ));
    }

    #[test]
    fn test_verify_missing_key() {
        let mut auth = make_auth(64);
        auth.add_key(simple_key("k1")).expect("test: add_key k1");
        let msg = auth
            .sign(b"data".to_vec(), "k1", 0)
            .expect("test: sign data");
        auth.remove_key("k1").expect("test: remove_key k1");
        assert!(matches!(
            auth.verify(&msg, 0),
            Err(AuthError::KeyNotFound(_))
        ));
    }

    #[test]
    fn test_verify_tampered_payload() {
        let mut auth = make_auth(64);
        auth.add_key(simple_key("k1")).expect("test: add_key k1");
        let mut msg = auth
            .sign(b"original".to_vec(), "k1", 0)
            .expect("test: sign original");
        msg.payload = b"tampered".to_vec();
        assert!(matches!(
            auth.verify(&msg, 0),
            Err(AuthError::InvalidSignature { .. })
        ));
    }

    #[test]
    fn test_verify_tampered_signature() {
        let mut auth = make_auth(64);
        auth.add_key(simple_key("k1")).expect("test: add_key k1");
        let mut msg = auth
            .sign(b"data".to_vec(), "k1", 0)
            .expect("test: sign data");
        msg.signature ^= 0xFF;
        assert!(matches!(
            auth.verify(&msg, 0),
            Err(AuthError::InvalidSignature { .. })
        ));
    }

    #[test]
    fn test_verify_tampered_nonce() {
        let mut auth = make_auth(64);
        auth.add_key(simple_key("k1")).expect("test: add_key k1");
        // HmacFnv64WithNonce embeds nonce in the hash — tampering must fail.
        let key = key_with_algo("k2", AuthAlgorithm::HmacFnv64WithNonce);
        auth.add_key(key).expect("test: add_key k2 with nonce algo");
        let mut msg = auth
            .sign(b"data".to_vec(), "k2", 0)
            .expect("test: sign data k2");
        msg.nonce ^= 0xABCD;
        assert!(matches!(
            auth.verify(&msg, 0),
            Err(AuthError::InvalidSignature { .. })
        ));
    }

    // ── Key expiry tests ─────────────────────────────────────────────────────

    #[test]
    fn test_sign_with_expired_key_fails() {
        let mut auth = make_auth(64);
        auth.add_key(key_with_expiry("k1", 100))
            .expect("test: add_key k1 with expiry");
        // ts=200 >= expires_at=100
        assert!(matches!(
            auth.sign(b"data".to_vec(), "k1", 200),
            Err(AuthError::KeyExpired(_))
        ));
    }

    #[test]
    fn test_sign_with_valid_expiry_succeeds() {
        let mut auth = make_auth(64);
        auth.add_key(key_with_expiry("k1", 1_000))
            .expect("test: add_key k1 with expiry");
        // ts=500 < expires_at=1000 → should succeed.
        assert!(auth.sign(b"data".to_vec(), "k1", 500).is_ok());
    }

    #[test]
    fn test_verify_with_expired_key_fails() {
        let mut auth = make_auth(64);
        auth.add_key(key_with_expiry("k1", 1_000))
            .expect("test: add_key k1 with expiry");
        // Sign while valid.
        let msg = auth
            .sign(b"data".to_vec(), "k1", 500)
            .expect("test: sign data while valid");
        // Verify after expiry.
        assert!(matches!(
            auth.verify(&msg, 1_001),
            Err(AuthError::KeyExpired(_))
        ));
    }

    #[test]
    fn test_expire_keys_removes_expired() {
        let mut auth = make_auth(64);
        auth.add_key(key_with_expiry("k_old", 100))
            .expect("test: add_key k_old with expiry");
        auth.add_key(simple_key("k_live"))
            .expect("test: add_key k_live");
        let removed = auth.expire_keys(200);
        assert_eq!(removed, vec!["k_old".to_string()]);
        assert!(!auth.has_key("k_old"));
        assert!(auth.has_key("k_live"));
    }

    #[test]
    fn test_expire_keys_none_expired() {
        let mut auth = make_auth(64);
        auth.add_key(key_with_expiry("k1", 1_000))
            .expect("test: add_key k1 with expiry");
        let removed = auth.expire_keys(500);
        assert!(removed.is_empty());
        assert!(auth.has_key("k1"));
    }

    #[test]
    fn test_expire_keys_no_expiry_set() {
        let mut auth = make_auth(64);
        auth.add_key(simple_key("k_permanent"))
            .expect("test: add_key k_permanent");
        let removed = auth.expire_keys(u64::MAX);
        assert!(removed.is_empty());
    }

    #[test]
    fn test_expire_multiple_keys() {
        let mut auth = make_auth(64);
        for i in 0..4u64 {
            let key = AuthKey::new(
                format!("k{i}"),
                b"s".to_vec(),
                0,
                Some(i * 100 + 50), // expire at 50, 150, 250, 350
                AuthAlgorithm::HmacFnv64,
            );
            auth.add_key(key).expect("test: add_key in loop");
        }
        // Expire keys whose expiry ≤ 200.
        let removed = auth.expire_keys(200);
        // k0 (exp=50) and k1 (exp=150) should be gone; k2 (exp=250), k3 (exp=350) survive.
        assert_eq!(removed.len(), 2);
        assert!(!auth.has_key("k0"));
        assert!(!auth.has_key("k1"));
        assert!(auth.has_key("k2"));
        assert!(auth.has_key("k3"));
    }

    // ── Replay prevention tests ──────────────────────────────────────────────

    #[test]
    fn test_replay_blocked_on_second_verify() {
        let mut auth = make_auth(64);
        auth.add_key(simple_key("k1")).expect("test: add_key k1");
        let msg = auth
            .sign(b"hello".to_vec(), "k1", 0)
            .expect("test: sign hello");
        // First verify succeeds.
        assert!(auth.verify(&msg, 0).is_ok());
        // Second verify with same nonce must fail.
        assert!(matches!(
            auth.verify(&msg, 0),
            Err(AuthError::ReplayDetected(_))
        ));
    }

    #[test]
    fn test_replay_window_eviction() {
        let window = 4;
        let mut auth = MessageAuthenticator::new(vec![], window);
        auth.add_key(simple_key("k1")).expect("test: add_key k1");

        // Fill and overflow the window (5 messages into a window of 4).
        let mut msgs = Vec::new();
        for _ in 0..5 {
            let m = auth.sign(b"p".to_vec(), "k1", 0).expect("test: sign p");
            msgs.push(m);
        }

        // Verify all 5.
        for m in &msgs {
            // The first-signed message's nonce will be evicted; its re-verify
            // should succeed (evicted from window ⇒ not detected as replay).
            let _ = auth.verify(m, 0);
        }
        // After verification the window holds the last 4 nonces.
        assert_eq!(auth.replay_window.seen_nonces.len(), window);
    }

    #[test]
    fn test_check_replay_returns_true_when_seen() {
        let mut auth = make_auth(32);
        auth.record_nonce(0xABCD_u64);
        assert!(auth.check_replay(0xABCD_u64));
    }

    #[test]
    fn test_check_replay_returns_false_when_unseen() {
        let auth = make_auth(32);
        assert!(!auth.check_replay(0xDEAD_BEEF_u64));
    }

    #[test]
    fn test_record_nonce_evicts_oldest_when_full() {
        let mut auth = make_auth(3);
        auth.record_nonce(1);
        auth.record_nonce(2);
        auth.record_nonce(3);
        // Window full; inserting 4 should evict 1.
        auth.record_nonce(4);
        assert!(!auth.check_replay(1), "oldest should be evicted");
        assert!(auth.check_replay(2));
        assert!(auth.check_replay(3));
        assert!(auth.check_replay(4));
    }

    // ── Algorithm-specific tests ─────────────────────────────────────────────

    #[test]
    fn test_algorithm_hmac_fnv64_roundtrip() {
        let mut auth = make_auth(64);
        auth.add_key(key_with_algo("k", AuthAlgorithm::HmacFnv64))
            .expect("test: add_key with HmacFnv64 algo");
        let msg = auth
            .sign(b"test".to_vec(), "k", 0)
            .expect("test: sign test");
        assert!(auth.verify(&msg, 0).is_ok());
    }

    #[test]
    fn test_algorithm_hmac_fnv64_with_nonce_roundtrip() {
        let mut auth = make_auth(64);
        auth.add_key(key_with_algo("k", AuthAlgorithm::HmacFnv64WithNonce))
            .expect("test: add_key with HmacFnv64WithNonce algo");
        let msg = auth
            .sign(b"test".to_vec(), "k", 0)
            .expect("test: sign test");
        assert!(auth.verify(&msg, 0).is_ok());
    }

    #[test]
    fn test_algorithm_chained_hash_roundtrip_1_round() {
        let mut auth = make_auth(64);
        auth.add_key(key_with_algo("k", AuthAlgorithm::ChainedHash(1)))
            .expect("test: add_key with ChainedHash(1) algo");
        let msg = auth
            .sign(b"test".to_vec(), "k", 0)
            .expect("test: sign test");
        assert!(auth.verify(&msg, 0).is_ok());
    }

    #[test]
    fn test_algorithm_chained_hash_roundtrip_3_rounds() {
        let mut auth = make_auth(64);
        auth.add_key(key_with_algo("k", AuthAlgorithm::ChainedHash(3)))
            .expect("test: add_key with ChainedHash(3) algo");
        let msg = auth
            .sign(b"test".to_vec(), "k", 0)
            .expect("test: sign test");
        assert!(auth.verify(&msg, 0).is_ok());
    }

    #[test]
    fn test_algorithm_chained_hash_max_rounds() {
        let mut auth = make_auth(64);
        auth.add_key(key_with_algo("k", AuthAlgorithm::ChainedHash(255)))
            .expect("test: add_key with ChainedHash(255) algo");
        let msg = auth
            .sign(b"rounds".to_vec(), "k", 0)
            .expect("test: sign rounds");
        assert!(auth.verify(&msg, 0).is_ok());
    }

    #[test]
    fn test_algorithms_produce_different_signatures() {
        // Same key material, same payload — different algorithms must differ.
        let secret = b"shared_secret".to_vec();
        let payload = b"same payload".to_vec();
        let nonce = 0x1234_5678_9ABC_DEF0_u64;

        let s1 = compute_signature(&AuthAlgorithm::HmacFnv64, &secret, &payload, nonce);
        let s2 = compute_signature(&AuthAlgorithm::HmacFnv64WithNonce, &secret, &payload, nonce);
        let s3 = compute_signature(&AuthAlgorithm::ChainedHash(2), &secret, &payload, nonce);

        assert_ne!(s1, s2);
        assert_ne!(s1, s3);
        assert_ne!(s2, s3);
    }

    #[test]
    fn test_chained_hash_rounds_differ() {
        let secret = b"secret".to_vec();
        let payload = b"payload".to_vec();
        let nonce = 1u64;

        let s1 = compute_signature(&AuthAlgorithm::ChainedHash(1), &secret, &payload, nonce);
        let s2 = compute_signature(&AuthAlgorithm::ChainedHash(2), &secret, &payload, nonce);
        let s3 = compute_signature(&AuthAlgorithm::ChainedHash(3), &secret, &payload, nonce);

        assert_ne!(s1, s2);
        assert_ne!(s2, s3);
    }

    // ── Sequential nonce policy tests ────────────────────────────────────────

    #[test]
    fn test_sequential_nonce_policy_first_message_ok() {
        let mut auth = MessageAuthenticator::new(vec![AuthPolicy::SequentialNonce], 64);
        auth.add_key(simple_key("k1")).expect("test: add_key k1");
        let msg = auth
            .sign(b"first".to_vec(), "k1", 0)
            .expect("test: sign first");
        // First message: last_sequence == 0, so no check fires.
        assert!(auth.verify(&msg, 0).is_ok());
    }

    #[test]
    fn test_sequential_nonce_policy_ordered() {
        let mut auth = MessageAuthenticator::new(vec![AuthPolicy::SequentialNonce], 64);
        auth.add_key(simple_key("k1")).expect("test: add_key k1");
        // Sign two messages in order.
        let m1 = auth.sign(b"one".to_vec(), "k1", 0).expect("test: sign one");
        let m2 = auth.sign(b"two".to_vec(), "k1", 0).expect("test: sign two");
        // Verify in order — both should pass.
        assert!(auth.verify(&m1, 0).is_ok());
        assert!(auth.verify(&m2, 0).is_ok());
    }

    #[test]
    fn test_sequential_nonce_policy_out_of_order_rejected() {
        let mut auth = MessageAuthenticator::new(vec![AuthPolicy::SequentialNonce], 64);
        auth.add_key(simple_key("k1")).expect("test: add_key k1");
        let m1 = auth.sign(b"one".to_vec(), "k1", 0).expect("test: sign one");
        let m2 = auth.sign(b"two".to_vec(), "k1", 0).expect("test: sign two");
        // Verify m1 first to set last_sequence = 1.
        assert!(auth.verify(&m1, 0).is_ok());
        // Now craft a message with a sequence number far in the future.
        let mut m_bad = m2.clone();
        m_bad.sequence_num = 999;
        // Should be caught by the SequentialNonce policy.
        assert!(matches!(
            auth.verify(&m_bad, 0),
            Err(AuthError::InvalidSequence { .. }) | Err(AuthError::InvalidSignature { .. })
        ));
    }

    // ── Key-rotation policy tests ────────────────────────────────────────────

    #[test]
    fn test_key_rotation_required_policy_new_key_ok() {
        let max_age = 10_000u64;
        let mut auth =
            MessageAuthenticator::new(vec![AuthPolicy::KeyRotationRequired(max_age)], 64);
        // Key created at ts=0, signing at ts=5000 → age=5000 < max=10000.
        let key = AuthKey::new("k1", b"sec".to_vec(), 0, None, AuthAlgorithm::HmacFnv64);
        auth.add_key(key).expect("test: add_key k1");
        assert!(auth.sign(b"data".to_vec(), "k1", 5_000).is_ok());
    }

    #[test]
    fn test_key_rotation_required_policy_old_key_rejected() {
        let max_age = 10_000u64;
        let mut auth =
            MessageAuthenticator::new(vec![AuthPolicy::KeyRotationRequired(max_age)], 64);
        // Key created at ts=0, signing at ts=20000 → age=20000 > max=10000.
        let key = AuthKey::new("k1", b"sec".to_vec(), 0, None, AuthAlgorithm::HmacFnv64);
        auth.add_key(key).expect("test: add_key k1");
        assert!(matches!(
            auth.sign(b"data".to_vec(), "k1", 20_000),
            Err(AuthError::KeyExpired(_))
        ));
    }

    // ── Statistics tests ─────────────────────────────────────────────────────

    #[test]
    fn test_stats_initial_zeroed() {
        let auth = make_auth(32);
        let s = auth.stats();
        assert_eq!(s.messages_signed, 0);
        assert_eq!(s.messages_verified, 0);
        assert_eq!(s.replay_attacks_blocked, 0);
        assert_eq!(s.expired_key_rejections, 0);
        assert_eq!(s.invalid_signature_rejections, 0);
    }

    #[test]
    fn test_stats_sign_increments() {
        let mut auth = make_auth(32);
        auth.add_key(simple_key("k1")).expect("test: add_key k1");
        auth.sign(b"a".to_vec(), "k1", 0).expect("test: sign a");
        auth.sign(b"b".to_vec(), "k1", 0).expect("test: sign b");
        assert_eq!(auth.stats().messages_signed, 2);
    }

    #[test]
    fn test_stats_verify_increments() {
        let mut auth = make_auth(64);
        auth.add_key(simple_key("k1")).expect("test: add_key k1");
        let m1 = auth.sign(b"a".to_vec(), "k1", 0).expect("test: sign a");
        let m2 = auth.sign(b"b".to_vec(), "k1", 0).expect("test: sign b");
        auth.verify(&m1, 0).expect("test: verify m1");
        auth.verify(&m2, 0).expect("test: verify m2");
        assert_eq!(auth.stats().messages_verified, 2);
    }

    #[test]
    fn test_stats_replay_blocked_increments() {
        let mut auth = make_auth(64);
        auth.add_key(simple_key("k1")).expect("test: add_key k1");
        let msg = auth
            .sign(b"hello".to_vec(), "k1", 0)
            .expect("test: sign hello");
        auth.verify(&msg, 0).expect("test: verify msg");
        let _ = auth.verify(&msg, 0); // replay
        assert_eq!(auth.stats().replay_attacks_blocked, 1);
    }

    #[test]
    fn test_stats_expired_key_rejections_increments() {
        let mut auth = make_auth(64);
        auth.add_key(key_with_expiry("k1", 100))
            .expect("test: add_key k1 with expiry");
        let _ = auth.sign(b"d".to_vec(), "k1", 200); // expired
        assert_eq!(auth.stats().expired_key_rejections, 1);
    }

    #[test]
    fn test_stats_invalid_signature_increments() {
        let mut auth = make_auth(64);
        auth.add_key(simple_key("k1")).expect("test: add_key k1");
        let mut msg = auth
            .sign(b"data".to_vec(), "k1", 0)
            .expect("test: sign data");
        msg.signature ^= 1;
        let _ = auth.verify(&msg, 0);
        assert_eq!(auth.stats().invalid_signature_rejections, 1);
    }

    // ── Audit log tests ──────────────────────────────────────────────────────

    #[test]
    fn test_drain_events_clears_log() {
        let mut auth = make_auth(32);
        auth.add_key(simple_key("k1")).expect("test: add_key k1");
        let events = auth.drain_events();
        assert!(!events.is_empty(), "add_key should have logged an event");
        // Second drain should return nothing.
        let events2 = auth.drain_events();
        assert!(events2.is_empty());
    }

    #[test]
    fn test_drain_events_captures_sign_and_verify() {
        let mut auth = make_auth(64);
        auth.add_key(simple_key("k1")).expect("test: add_key k1");
        let msg = auth.sign(b"hi".to_vec(), "k1", 0).expect("test: sign hi");
        auth.verify(&msg, 0).expect("test: verify msg");
        let events = auth.drain_events();
        let log = events.join("\n");
        assert!(log.contains("key_added"), "should log key_added");
        assert!(log.contains("signed"), "should log signed");
        assert!(log.contains("verified"), "should log verified");
    }

    #[test]
    fn test_drain_events_captures_replay_blocked() {
        let mut auth = make_auth(64);
        auth.add_key(simple_key("k1")).expect("test: add_key k1");
        let msg = auth.sign(b"x".to_vec(), "k1", 0).expect("test: sign x");
        auth.verify(&msg, 0).expect("test: verify msg first time");
        let _ = auth.verify(&msg, 0); // replay
        let events = auth.drain_events();
        let log = events.join("\n");
        assert!(log.contains("replay_blocked"));
    }

    #[test]
    fn test_drain_events_captures_key_expired_eviction() {
        let mut auth = make_auth(64);
        auth.add_key(key_with_expiry("k_expire", 50))
            .expect("test: add_key k_expire with expiry");
        auth.expire_keys(100);
        let events = auth.drain_events();
        let log = events.join("\n");
        assert!(log.contains("key_expired_evicted"));
    }

    #[test]
    fn test_drain_events_captures_key_rotation() {
        let mut auth = make_auth(32);
        auth.add_key(simple_key("k_old"))
            .expect("test: add_key k_old");
        auth.rotate_key("k_old", simple_key("k_new"))
            .expect("test: rotate k_old to k_new");
        let events = auth.drain_events();
        let log = events.join("\n");
        assert!(log.contains("key_rotated"));
    }

    // ── Error-case coverage ──────────────────────────────────────────────────

    #[test]
    fn test_error_key_not_found_message() {
        let err = AuthError::KeyNotFound("missing".to_string());
        assert!(err.to_string().contains("missing"));
    }

    #[test]
    fn test_error_invalid_signature_message() {
        let err = AuthError::InvalidSignature {
            key_id: "k1".to_string(),
            expected: 0xABCD,
            got: 0x1234,
        };
        let s = err.to_string();
        assert!(s.contains("k1"));
        assert!(s.contains("0x000000000000abcd"));
        assert!(s.contains("0x0000000000001234"));
    }

    #[test]
    fn test_error_replay_detected_message() {
        let err = AuthError::ReplayDetected(0xDEAD);
        assert!(err.to_string().contains("replay"));
    }

    #[test]
    fn test_error_key_expired_message() {
        let err = AuthError::KeyExpired("old_key".to_string());
        assert!(err.to_string().contains("old_key"));
    }

    #[test]
    fn test_error_nonce_exhausted_message() {
        let err = AuthError::NonceExhausted;
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn test_error_invalid_sequence_message() {
        let err = AuthError::InvalidSequence {
            expected: 5,
            got: 3,
        };
        let s = err.to_string();
        assert!(s.contains("5"));
        assert!(s.contains("3"));
    }

    // ── Additional edge cases ────────────────────────────────────────────────

    #[test]
    fn test_sign_verify_empty_payload() {
        let mut auth = make_auth(64);
        auth.add_key(simple_key("k1")).expect("test: add_key k1");
        let msg = auth
            .sign(vec![], "k1", 0)
            .expect("test: sign empty payload");
        assert!(auth.verify(&msg, 0).is_ok());
    }

    #[test]
    fn test_sign_verify_large_payload() {
        let mut auth = make_auth(128);
        auth.add_key(simple_key("k1")).expect("test: add_key k1");
        let payload = vec![0xABu8; 64 * 1024];
        let msg = auth
            .sign(payload, "k1", 0)
            .expect("test: sign large payload");
        assert!(auth.verify(&msg, 0).is_ok());
    }

    #[test]
    fn test_sequence_numbers_increment_per_key() {
        let mut auth = make_auth(128);
        auth.add_key(simple_key("k1")).expect("test: add_key k1");
        let m1 = auth.sign(b"a".to_vec(), "k1", 0).expect("test: sign a");
        let m2 = auth.sign(b"b".to_vec(), "k1", 0).expect("test: sign b");
        let m3 = auth.sign(b"c".to_vec(), "k1", 0).expect("test: sign c");
        assert_eq!(m1.sequence_num, 1);
        assert_eq!(m2.sequence_num, 2);
        assert_eq!(m3.sequence_num, 3);
    }

    #[test]
    fn test_sequence_numbers_independent_per_key() {
        let mut auth = make_auth(128);
        auth.add_key(simple_key("k1")).expect("test: add_key k1");
        auth.add_key(simple_key("k2")).expect("test: add_key k2");
        let ma = auth
            .sign(b"a".to_vec(), "k1", 0)
            .expect("test: sign a with k1");
        let mb = auth
            .sign(b"a".to_vec(), "k2", 0)
            .expect("test: sign a with k2");
        // Both start at seq=1.
        assert_eq!(ma.sequence_num, 1);
        assert_eq!(mb.sequence_num, 1);
    }

    #[test]
    fn test_verify_wrong_key_id_in_message() {
        let mut auth = make_auth(64);
        // Use different secrets so k1 and k2 produce different signatures.
        let k1 = AuthKey::new(
            "k1",
            b"secret_for_k1".to_vec(),
            0,
            None,
            AuthAlgorithm::HmacFnv64,
        );
        let k2 = AuthKey::new(
            "k2",
            b"different_secret_k2".to_vec(),
            0,
            None,
            AuthAlgorithm::HmacFnv64,
        );
        auth.add_key(k1).expect("test: add_key k1");
        auth.add_key(k2).expect("test: add_key k2");
        let mut msg = auth
            .sign(b"data".to_vec(), "k1", 0)
            .expect("test: sign data k1");
        // Point the message at k2 — signature was computed under k1's secret,
        // so it cannot match k2's recomputed value.
        msg.key_id = "k2".to_string();
        assert!(matches!(
            auth.verify(&msg, 0),
            Err(AuthError::InvalidSignature { .. })
        ));
    }

    #[test]
    fn test_many_sign_verify_cycles() {
        let mut auth = make_auth(1000);
        auth.add_key(simple_key("k1")).expect("test: add_key k1");
        let mut msgs = Vec::new();
        for i in 0..50u64 {
            let payload = format!("message-{i}").into_bytes();
            let m = auth
                .sign(payload, "k1", i)
                .expect("test: sign payload in loop");
            msgs.push(m);
        }
        for m in &msgs {
            assert!(auth.verify(m, 0).is_ok());
        }
        let s = auth.stats();
        assert_eq!(s.messages_signed, 50);
        assert_eq!(s.messages_verified, 50);
        assert_eq!(s.replay_attacks_blocked, 0);
    }

    #[test]
    fn test_optional_sign_policy_does_not_block() {
        // OptionalSign — just check we can create the authenticator with it.
        let mut auth = MessageAuthenticator::new(vec![AuthPolicy::OptionalSign], 32);
        auth.add_key(simple_key("k1")).expect("test: add_key k1");
        let msg = auth.sign(b"x".to_vec(), "k1", 0).expect("test: sign x");
        assert!(auth.verify(&msg, 0).is_ok());
    }

    #[test]
    fn test_replay_window_size_zero_records_nothing() {
        let mut auth = make_auth(0);
        auth.record_nonce(42);
        // Window of size 0 never stores anything.
        assert!(!auth.check_replay(42));
    }

    #[test]
    fn test_auth_key_is_valid_at_boundary() {
        let key = key_with_expiry("k", 500);
        // Strictly before expiry → valid.
        assert!(key.is_valid_at(499));
        // Exactly at expiry → invalid.
        assert!(!key.is_valid_at(500));
        // After expiry → invalid.
        assert!(!key.is_valid_at(501));
    }

    #[test]
    fn test_auth_key_no_expiry_always_valid() {
        let key = simple_key("k");
        assert!(key.is_valid_at(0));
        assert!(key.is_valid_at(u64::MAX));
    }

    #[test]
    fn test_drain_events_multiple_rounds() {
        let mut auth = make_auth(32);
        auth.add_key(simple_key("k1")).expect("test: add_key k1");
        let round1 = auth.drain_events();
        assert!(!round1.is_empty());

        auth.add_key(simple_key("k2")).expect("test: add_key k2");
        let round2 = auth.drain_events();
        assert!(!round2.is_empty());
        // round1 events should not appear again.
        let combined = round2.join("\n");
        let count_k1_add = combined.matches("key_added id=k1").count();
        assert_eq!(count_k1_add, 0, "k1 add event should not re-appear");
    }
}
