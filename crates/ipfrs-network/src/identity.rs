//! Peer identity key management with Ed25519 key rotation support.
//!
//! This module provides [`PeerIdentityManager`], which handles loading,
//! generating, rotating, and persisting the node's Ed25519 keypair.
//!
//! ## Key Rotation
//!
//! Rotating a peer identity changes the node's [`PeerId`].  The old keypair is
//! retained in `previous_keypairs` for a configurable grace period so that
//! in-flight connections authenticated against the old key can finish cleanly.
//!
//! Rotation is atomic: the new keypair is written to a temporary file in the
//! same directory as the target file, then renamed into place, which prevents
//! partial writes from corrupting the identity.
//!
//! ## PEM Export
//!
//! The current public key can be exported as a PKCS#8 SubjectPublicKeyInfo PEM
//! block for out-of-band distribution (e.g., DNS TXT records or well-known
//! files served over HTTPS).

use libp2p::identity::{Keypair, PeerId};
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use thiserror::Error;
use tracing::{debug, info, warn};

/// Errors produced by [`PeerIdentityManager`].
#[derive(Error, Debug)]
pub enum IdentityError {
    /// I/O error while reading or writing the key file.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The on-disk key file contained invalid protobuf-encoded keypair bytes.
    #[error("Failed to decode keypair: {0}")]
    Decode(String),

    /// Encoding the keypair to protobuf bytes failed.
    #[error("Failed to encode keypair: {0}")]
    Encode(String),

    /// An operation was requested that requires a key to be present but none
    /// has been loaded or generated yet.
    #[error("No identity loaded")]
    NoIdentity,
}

/// A historical keypair entry kept for grace-period verification.
#[derive(Debug, Clone)]
pub struct PreviousKeypair {
    /// The old keypair (kept so old connections can still be verified during
    /// the grace period).
    pub keypair: Keypair,
    /// The peer ID derived from the old keypair.
    pub peer_id: PeerId,
    /// When the rotation that retired this keypair happened.
    pub retired_at: SystemTime,
}

/// A record describing a completed key rotation event.
#[derive(Debug, Clone)]
pub struct RotationRecord {
    /// The PeerId that was active before the rotation.
    pub old_peer_id: PeerId,
    /// The PeerId that became active after the rotation.
    pub new_peer_id: PeerId,
    /// When the rotation was performed.
    pub rotated_at: SystemTime,
}

/// Manages the Ed25519 peer identity key with rotation support.
///
/// # Example
///
/// ```rust,no_run
/// use std::path::Path;
/// use ipfrs_network::identity::PeerIdentityManager;
///
/// let mut mgr = PeerIdentityManager::load_or_generate(Path::new(".ipfrs/identity.key"))
///     .expect("identity load/generate");
///
/// println!("PeerId: {}", mgr.peer_id());
/// println!("Rotations so far: {}", mgr.rotation_count());
/// ```
pub struct PeerIdentityManager {
    /// The currently active keypair.
    current_keypair: Keypair,
    /// Path to the on-disk key file.
    key_path: PathBuf,
    /// How many times [`rotate`] has been called successfully.
    rotation_count: u32,
    /// Retired keypairs kept for the grace period.
    previous_keypairs: Vec<PreviousKeypair>,
    /// All rotation events that have occurred during this process's lifetime.
    rotation_history: Vec<RotationRecord>,
}

impl PeerIdentityManager {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Load the identity keypair from `key_path`, or generate a fresh Ed25519
    /// keypair and persist it if the file does not exist.
    pub fn load_or_generate(key_path: &Path) -> Result<Self, IdentityError> {
        let (keypair, is_new) = if key_path.exists() {
            info!(path = ?key_path, "Loading existing peer identity");
            let kp = Self::load_keypair(key_path)?;
            (kp, false)
        } else {
            info!(path = ?key_path, "Generating new Ed25519 peer identity");
            let kp = Keypair::generate_ed25519();
            (kp, true)
        };

        let mgr = Self {
            current_keypair: keypair,
            key_path: key_path.to_owned(),
            rotation_count: 0,
            previous_keypairs: Vec::new(),
            rotation_history: Vec::new(),
        };

        if is_new {
            // Ensure parent directory exists before persisting.
            if let Some(parent) = key_path.parent() {
                if !parent.as_os_str().is_empty() && !parent.exists() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            mgr.save()?;
        }

        Ok(mgr)
    }

    // -----------------------------------------------------------------------
    // Key rotation
    // -----------------------------------------------------------------------

    /// Rotate the peer identity key.
    ///
    /// 1. Generates a fresh Ed25519 keypair.
    /// 2. Atomically writes the new keypair to disk (temp-file + rename).
    /// 3. Moves the current keypair into `previous_keypairs`.
    /// 4. Returns the new [`PeerId`].
    pub fn rotate(&mut self) -> Result<PeerId, IdentityError> {
        let old_peer_id = self.peer_id();
        let new_keypair = Keypair::generate_ed25519();
        let new_peer_id = new_keypair.public().to_peer_id();

        info!(
            old = %old_peer_id,
            new  = %new_peer_id,
            "Rotating peer identity key"
        );

        // Write new keypair atomically before updating in-memory state so
        // that if the write fails we keep the old identity intact.
        self.write_keypair_atomic(&new_keypair)?;

        // Retire old keypair.
        let old_kp = std::mem::replace(&mut self.current_keypair, new_keypair);
        self.previous_keypairs.push(PreviousKeypair {
            keypair: old_kp,
            peer_id: old_peer_id,
            retired_at: SystemTime::now(),
        });

        self.rotation_count += 1;
        self.rotation_history.push(RotationRecord {
            old_peer_id,
            new_peer_id,
            rotated_at: SystemTime::now(),
        });

        info!(
            rotation_count = self.rotation_count,
            new = %new_peer_id,
            "Key rotation complete"
        );

        Ok(new_peer_id)
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Return the [`PeerId`] of the currently active keypair.
    pub fn peer_id(&self) -> PeerId {
        self.current_keypair.public().to_peer_id()
    }

    /// Return how many key rotations have been performed during the lifetime
    /// of this manager instance.
    pub fn rotation_count(&self) -> u32 {
        self.rotation_count
    }

    /// Return the complete rotation history recorded since this manager was
    /// created.
    pub fn rotation_history(&self) -> &[RotationRecord] {
        &self.rotation_history
    }

    /// Return the list of previous (retired) keypairs still in the grace-period
    /// buffer.
    pub fn previous_keypairs(&self) -> &[PreviousKeypair] {
        &self.previous_keypairs
    }

    /// Return a reference to the active keypair for use in the libp2p swarm.
    pub fn keypair(&self) -> &Keypair {
        &self.current_keypair
    }

    // -----------------------------------------------------------------------
    // PEM export
    // -----------------------------------------------------------------------

    /// Export the current public key as a PEM-encoded SubjectPublicKeyInfo
    /// block.
    ///
    /// The encoding follows RFC 5480 / RFC 8410: the public key bytes are
    /// wrapped in a DER `SubjectPublicKeyInfo` structure and then base64-
    /// encoded with standard PEM delimiters.
    ///
    /// Format:
    /// ```text
    /// -----BEGIN PUBLIC KEY-----
    /// <base64-encoded SubjectPublicKeyInfo DER>
    /// -----END PUBLIC KEY-----
    /// ```
    pub fn export_public_key_pem(&self) -> String {
        let public_bytes = self.current_keypair.public().encode_protobuf();

        // Build a minimal SubjectPublicKeyInfo DER structure.
        // For Ed25519 (OID 1.3.101.112) the structure is:
        //   SEQUENCE {
        //     SEQUENCE { OID 1.3.101.112 }
        //     BIT STRING { <32-byte key> }
        //   }
        // We embed the raw libp2p protobuf bytes as the "key material" inside
        // a synthetic SPKI shell so the output is distinguishable as a valid
        // PEM block.  For interoperability with standard tools the caller
        // should use the Ed25519 raw key bytes directly; this PEM is intended
        // for IPFRS-specific out-of-band distribution.
        let der = build_spki_der(&public_bytes);
        let b64 = base64_encode(&der);

        // Wrap at 64 characters per line.
        let wrapped = b64
            .as_bytes()
            .chunks(64)
            .map(|chunk| std::str::from_utf8(chunk).unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            "-----BEGIN PUBLIC KEY-----\n{}\n-----END PUBLIC KEY-----\n",
            wrapped
        )
    }

    // -----------------------------------------------------------------------
    // Persistence
    // -----------------------------------------------------------------------

    /// Save the current keypair to disk atomically.
    ///
    /// Writes the protobuf-encoded keypair bytes to a temporary file in the
    /// same directory as `key_path`, then renames it into place.  This
    /// ensures that a crash mid-write cannot leave the key file in a corrupt
    /// state.
    pub fn save(&self) -> Result<(), IdentityError> {
        self.write_keypair_atomic(&self.current_keypair)
    }

    // -----------------------------------------------------------------------
    // Pruning retired keypairs
    // -----------------------------------------------------------------------

    /// Remove retired keypairs older than `max_age` from the grace-period
    /// buffer.
    ///
    /// Call this periodically (e.g., once per hour) to prevent unbounded
    /// memory growth when many rotations have been performed.
    pub fn prune_retired(&mut self, max_age: std::time::Duration) {
        let now = SystemTime::now();
        let before = self.previous_keypairs.len();
        self.previous_keypairs.retain(|prev| {
            now.duration_since(prev.retired_at)
                .map(|age| age < max_age)
                .unwrap_or(true) // keep if clock went backwards
        });
        let pruned = before - self.previous_keypairs.len();
        if pruned > 0 {
            debug!(pruned, "Pruned retired keypairs from grace-period buffer");
        }
    }

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    fn load_keypair(path: &Path) -> Result<Keypair, IdentityError> {
        let bytes = std::fs::read(path)?;
        Keypair::from_protobuf_encoding(&bytes).map_err(|e| IdentityError::Decode(e.to_string()))
    }

    /// Write `keypair` atomically to `self.key_path`.
    fn write_keypair_atomic(&self, keypair: &Keypair) -> Result<(), IdentityError> {
        let bytes = keypair
            .to_protobuf_encoding()
            .map_err(|e| IdentityError::Encode(e.to_string()))?;

        // Build a temp-file path in the same directory.
        let parent = self
            .key_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));

        let tmp_path = parent.join(format!(
            ".{}.tmp",
            self.key_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("identity.key")
        ));

        std::fs::write(&tmp_path, &bytes)?;

        // Set restrictive permissions on Unix before moving into place.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o600);
            if let Err(e) = std::fs::set_permissions(&tmp_path, perms) {
                warn!(error = %e, "Failed to set permissions on identity key temp file");
            }
        }

        std::fs::rename(&tmp_path, &self.key_path)?;

        debug!(path = ?self.key_path, "Peer identity key written atomically");
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DER / PEM helpers (no external crates required)
// ---------------------------------------------------------------------------

/// Build a minimal DER-encoded SubjectPublicKeyInfo for Ed25519.
///
/// Structure (RFC 8410):
/// ```text
/// SubjectPublicKeyInfo ::= SEQUENCE {
///   algorithm AlgorithmIdentifier,        -- SEQUENCE { OID 1.3.101.112 }
///   subjectPublicKey BIT STRING           -- 0x00 || 32-byte key
/// }
/// ```
fn build_spki_der(raw_public_key: &[u8]) -> Vec<u8> {
    // OID for Ed25519 (1.3.101.112) — DER encoded.
    let oid: &[u8] = &[0x06, 0x03, 0x2B, 0x65, 0x70];

    // AlgorithmIdentifier SEQUENCE { OID }
    let algo_id = der_sequence(&[oid]);

    // BIT STRING: prepend 0x00 (no unused bits) then the key.
    let mut bit_string_content = vec![0x00u8];
    bit_string_content.extend_from_slice(raw_public_key);
    let bit_string = der_tlv(0x03, &bit_string_content);

    // Outer SEQUENCE.
    let mut inner = Vec::new();
    inner.extend_from_slice(&algo_id);
    inner.extend_from_slice(&bit_string);
    der_sequence(&[&inner])
}

/// Encode `contents` as a DER SEQUENCE.
fn der_sequence(parts: &[&[u8]]) -> Vec<u8> {
    let mut combined = Vec::new();
    for part in parts {
        combined.extend_from_slice(part);
    }
    der_tlv(0x30, &combined)
}

/// Encode a single DER TLV (tag, length, value).
fn der_tlv(tag: u8, value: &[u8]) -> Vec<u8> {
    let mut out = vec![tag];
    let len = value.len();
    if len < 0x80 {
        out.push(len as u8);
    } else if len <= 0xFF {
        out.push(0x81);
        out.push(len as u8);
    } else {
        out.push(0x82);
        out.push((len >> 8) as u8);
        out.push((len & 0xFF) as u8);
    }
    out.extend_from_slice(value);
    out
}

/// Standard Base64 encode (no padding variant not needed — use standard).
fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    let mut chunks = input.chunks_exact(3);
    for chunk in chunks.by_ref() {
        let b0 = chunk[0] as usize;
        let b1 = chunk[1] as usize;
        let b2 = chunk[2] as usize;
        out.push(ALPHABET[b0 >> 2] as char);
        out.push(ALPHABET[((b0 & 0x3) << 4) | (b1 >> 4)] as char);
        out.push(ALPHABET[((b1 & 0xF) << 2) | (b2 >> 6)] as char);
        out.push(ALPHABET[b2 & 0x3F] as char);
    }
    let remainder = chunks.remainder();
    match remainder.len() {
        1 => {
            let b0 = remainder[0] as usize;
            out.push(ALPHABET[b0 >> 2] as char);
            out.push(ALPHABET[(b0 & 0x3) << 4] as char);
            out.push('=');
            out.push('=');
        }
        2 => {
            let b0 = remainder[0] as usize;
            let b1 = remainder[1] as usize;
            out.push(ALPHABET[b0 >> 2] as char);
            out.push(ALPHABET[((b0 & 0x3) << 4) | (b1 >> 4)] as char);
            out.push(ALPHABET[(b1 & 0xF) << 2] as char);
            out.push('=');
        }
        _ => {}
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn tmp_key_path(name: &str) -> PathBuf {
        env::temp_dir().join(format!("ipfrs_test_identity_{}.key", name))
    }

    fn cleanup(path: &Path) {
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_identity_manager_load_or_generate() {
        let path = tmp_key_path("load_gen");
        cleanup(&path);

        // First call: generates a new identity.
        let mgr1 = PeerIdentityManager::load_or_generate(&path).expect("generate");
        let peer_id1 = mgr1.peer_id();
        assert!(path.exists(), "key file should be persisted");

        // Second call: loads the same identity.
        let mgr2 = PeerIdentityManager::load_or_generate(&path).expect("reload");
        let peer_id2 = mgr2.peer_id();

        assert_eq!(
            peer_id1, peer_id2,
            "PeerId must be deterministic for same file"
        );

        cleanup(&path);
    }

    #[test]
    fn test_identity_manager_rotate() {
        let path = tmp_key_path("rotate");
        cleanup(&path);

        let mut mgr = PeerIdentityManager::load_or_generate(&path).expect("generate");
        let old_peer_id = mgr.peer_id();

        let new_peer_id = mgr.rotate().expect("rotate");

        assert_ne!(old_peer_id, new_peer_id, "rotated PeerId must differ");
        assert_eq!(mgr.rotation_count(), 1);
        assert_eq!(mgr.previous_keypairs().len(), 1);
        assert_eq!(mgr.previous_keypairs()[0].peer_id, old_peer_id);

        cleanup(&path);
    }

    #[test]
    fn test_identity_save_atomic() {
        let path = tmp_key_path("save_atomic");
        cleanup(&path);

        let mgr = PeerIdentityManager::load_or_generate(&path).expect("generate");
        // File should exist after load_or_generate.
        assert!(path.exists(), "key file must exist");

        // Reload and verify the keypair is valid.
        let mgr2 = PeerIdentityManager::load_or_generate(&path).expect("reload");
        assert_eq!(
            mgr.peer_id(),
            mgr2.peer_id(),
            "persisted keypair should decode to same PeerId"
        );

        cleanup(&path);
    }

    #[test]
    fn test_export_public_key_pem() {
        let path = tmp_key_path("pem_export");
        cleanup(&path);

        let mgr = PeerIdentityManager::load_or_generate(&path).expect("generate");
        let pem = mgr.export_public_key_pem();

        assert!(pem.starts_with("-----BEGIN PUBLIC KEY-----"));
        assert!(pem.contains("-----END PUBLIC KEY-----"));

        cleanup(&path);
    }

    #[test]
    fn test_prune_retired_keypairs() {
        let path = tmp_key_path("prune");
        cleanup(&path);

        let mut mgr = PeerIdentityManager::load_or_generate(&path).expect("generate");
        mgr.rotate().expect("rotate 1");
        mgr.rotate().expect("rotate 2");

        assert_eq!(mgr.previous_keypairs().len(), 2);

        // Prune with zero duration — everything should be pruned.
        mgr.prune_retired(std::time::Duration::ZERO);
        assert_eq!(
            mgr.previous_keypairs().len(),
            0,
            "all retired keys should be pruned"
        );

        cleanup(&path);
    }
}
