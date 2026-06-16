//! Protocol handshake negotiation between peers.
//!
//! When two peers connect, they exchange `HandshakeOffer` messages to negotiate:
//! - A mutually compatible protocol version (same major version required)
//! - The intersection of supported feature flags
//! - The minimum supported frame size
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_network::protocol_handshake::{
//!     FeatureFlag, HandshakeOffer, ProtocolHandshaker, ProtocolVersion,
//! };
//!
//! let local_offer = HandshakeOffer {
//!     peer_id: "local-peer".to_string(),
//!     protocol_version: ProtocolVersion::new(1, 0, 0),
//!     supported_features: vec![FeatureFlag::Encryption, FeatureFlag::Compression],
//!     max_frame_size: HandshakeOffer::DEFAULT_MAX_FRAME_SIZE,
//!     timestamp_ms: 0,
//! };
//!
//! let handshaker = ProtocolHandshaker::new(local_offer);
//!
//! let remote_offer = HandshakeOffer {
//!     peer_id: "remote-peer".to_string(),
//!     protocol_version: ProtocolVersion::new(1, 2, 0),
//!     supported_features: vec![FeatureFlag::Encryption, FeatureFlag::VectorSearch],
//!     max_frame_size: HandshakeOffer::DEFAULT_MAX_FRAME_SIZE,
//!     timestamp_ms: 1000,
//! };
//!
//! let result = handshaker.negotiate(&remote_offer).expect("handshake failed");
//! assert_eq!(result.negotiated_features.len(), 1); // only Encryption
//! ```

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use thiserror::Error;

// ─────────────────────────────────────────────
//  Errors
// ─────────────────────────────────────────────

/// Errors that can occur during the protocol handshake.
#[derive(Debug, Error)]
pub enum HandshakeError {
    /// The local and remote peers do not share the same protocol major version.
    #[error("Incompatible protocol version: local={local}, remote={remote}")]
    IncompatibleVersion { local: String, remote: String },

    /// After intersecting feature flags, no common features remain.
    #[error("No common features between peers")]
    NoCommonFeatures,

    /// The remote offer is structurally or semantically invalid.
    #[error("Invalid handshake offer: {0}")]
    InvalidOffer(String),
}

// ─────────────────────────────────────────────
//  ProtocolVersion
// ─────────────────────────────────────────────

/// Semantic protocol version used during handshake.
///
/// Versions are compared lexicographically by `(major, minor, patch)`.
/// Compatibility requires matching major versions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProtocolVersion {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

impl ProtocolVersion {
    /// Create a new protocol version.
    pub fn new(major: u16, minor: u16, patch: u16) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Returns `true` when both versions share the same major version number.
    ///
    /// Minor/patch differences are acceptable — backward-compatibility is
    /// assumed within a major series.
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.major == other.major
    }
}

impl std::fmt::Display for ProtocolVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl PartialOrd for ProtocolVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ProtocolVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.major, self.minor, self.patch).cmp(&(other.major, other.minor, other.patch))
    }
}

// ─────────────────────────────────────────────
//  FeatureFlag
// ─────────────────────────────────────────────

/// Optional capabilities that a peer may support.
///
/// Each flag occupies a single bit in a `u32` bitmask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeatureFlag {
    Encryption,
    Compression,
    VectorSearch,
    TensorLogic,
    GradientSync,
    WebRtc,
}

impl FeatureFlag {
    /// All defined feature flags in canonical order.
    pub const ALL: &'static [Self] = &[
        Self::Encryption,
        Self::Compression,
        Self::VectorSearch,
        Self::TensorLogic,
        Self::GradientSync,
        Self::WebRtc,
    ];

    /// Returns the bit position (0-based) used to represent this flag in a bitmask.
    ///
    /// | Flag          | Bit |
    /// |---------------|-----|
    /// | Encryption    |  0  |
    /// | Compression   |  1  |
    /// | VectorSearch  |  2  |
    /// | TensorLogic   |  3  |
    /// | GradientSync  |  4  |
    /// | WebRtc        |  5  |
    pub fn flag_bit(&self) -> u32 {
        match self {
            Self::Encryption => 0,
            Self::Compression => 1,
            Self::VectorSearch => 2,
            Self::TensorLogic => 3,
            Self::GradientSync => 4,
            Self::WebRtc => 5,
        }
    }

    /// Decode a bitmask into the set of active feature flags.
    ///
    /// Unknown bits are silently ignored.
    pub fn from_bits(bits: u32) -> Vec<Self> {
        Self::ALL
            .iter()
            .filter(|flag| (bits >> flag.flag_bit()) & 1 == 1)
            .copied()
            .collect()
    }
}

// ─────────────────────────────────────────────
//  HandshakeOffer
// ─────────────────────────────────────────────

/// 4 MiB — the default maximum frame size used when no override is configured.
pub const DEFAULT_MAX_FRAME_SIZE: u32 = 4 * 1024 * 1024;

/// The packet a peer sends at the start of a connection to advertise its
/// capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeOffer {
    /// Peer identifier of the sender.
    pub peer_id: String,
    /// Protocol version spoken by the sender.
    pub protocol_version: ProtocolVersion,
    /// Feature flags supported by the sender.
    pub supported_features: Vec<FeatureFlag>,
    /// Maximum frame size (in bytes) the sender can handle.  Default: 4 MiB.
    pub max_frame_size: u32,
    /// Wall-clock timestamp at offer creation (milliseconds since UNIX epoch).
    pub timestamp_ms: u64,
}

impl HandshakeOffer {
    /// Default maximum frame size (4 MiB).
    pub const DEFAULT_MAX_FRAME_SIZE: u32 = DEFAULT_MAX_FRAME_SIZE;

    /// Encode `supported_features` as a `u32` bitmask.
    pub fn feature_bits(&self) -> u32 {
        self.supported_features
            .iter()
            .fold(0u32, |acc, flag| acc | (1 << flag.flag_bit()))
    }

    /// Validate that the offer is internally consistent.
    pub(crate) fn validate(&self) -> Result<(), HandshakeError> {
        if self.peer_id.is_empty() {
            return Err(HandshakeError::InvalidOffer(
                "peer_id must not be empty".to_string(),
            ));
        }
        if self.max_frame_size == 0 {
            return Err(HandshakeError::InvalidOffer(
                "max_frame_size must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────
//  HandshakeResult
// ─────────────────────────────────────────────

/// The outcome of a successful protocol negotiation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeResult {
    /// The protocol version both peers agreed to use.
    pub agreed_version: ProtocolVersion,
    /// Feature flags supported by *both* peers.
    pub negotiated_features: Vec<FeatureFlag>,
    /// Effective frame size cap — the minimum of both peers' advertised limits.
    pub max_frame_size: u32,
    /// Peer ID of the local side.
    pub local_peer_id: String,
    /// Peer ID of the remote side.
    pub remote_peer_id: String,
    /// Wall-clock timestamp when negotiation completed (ms since UNIX epoch).
    pub negotiated_at_ms: u64,
}

// ─────────────────────────────────────────────
//  HandshakeStats
// ─────────────────────────────────────────────

/// Live atomic counters tracking handshake outcomes.
#[derive(Debug, Default)]
pub struct HandshakeStats {
    /// Number of handshake attempts started (including those in progress).
    pub total_attempted: AtomicU64,
    /// Number of handshakes that completed successfully.
    pub total_succeeded: AtomicU64,
    /// Number of handshakes that ended in an error.
    pub total_failed: AtomicU64,
}

/// Point-in-time copy of [`HandshakeStats`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HandshakeStatsSnapshot {
    pub total_attempted: u64,
    pub total_succeeded: u64,
    pub total_failed: u64,
}

impl HandshakeStats {
    /// Create a new zeroed stats instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Take a consistent snapshot of the current counters.
    pub fn snapshot(&self) -> HandshakeStatsSnapshot {
        HandshakeStatsSnapshot {
            total_attempted: self.total_attempted.load(Ordering::Relaxed),
            total_succeeded: self.total_succeeded.load(Ordering::Relaxed),
            total_failed: self.total_failed.load(Ordering::Relaxed),
        }
    }
}

// ─────────────────────────────────────────────
//  ProtocolHandshaker
// ─────────────────────────────────────────────

/// Drives the protocol negotiation for a single local peer.
///
/// A `ProtocolHandshaker` is constructed once per local node and then
/// called for every incoming or outgoing connection to negotiate shared
/// parameters with the remote peer.
///
/// # Thread safety
///
/// The statistics field uses `Arc<HandshakeStats>` so the same stats object
/// can be shared across tasks/threads without requiring `&mut self`.
#[derive(Debug)]
pub struct ProtocolHandshaker {
    /// The offer that the local node will present to every remote peer.
    pub local_offer: HandshakeOffer,
    /// Accumulated handshake statistics.
    pub stats: Arc<HandshakeStats>,
}

impl ProtocolHandshaker {
    /// Create a new handshaker with the given local offer.
    pub fn new(local_offer: HandshakeOffer) -> Self {
        Self {
            local_offer,
            stats: Arc::new(HandshakeStats::new()),
        }
    }

    /// Return the feature flags advertised by the local peer.
    pub fn local_features(&self) -> Vec<FeatureFlag> {
        self.local_offer.supported_features.clone()
    }

    /// Negotiate protocol parameters with a remote peer.
    ///
    /// On success, returns a [`HandshakeResult`] describing the agreed
    /// parameters.  On failure, returns a [`HandshakeError`].
    ///
    /// # Steps
    ///
    /// 1. Validate the remote offer.
    /// 2. Check that major versions match.
    /// 3. Intersect feature-flag bitmasks.
    /// 4. Agree on the smaller of the two maximum frame sizes.
    /// 5. Choose the *lower* of the two protocol versions as the agreed
    ///    version so that the less-advanced peer is never asked to speak a
    ///    version it cannot fully implement.
    pub fn negotiate(&self, remote: &HandshakeOffer) -> Result<HandshakeResult, HandshakeError> {
        self.stats.total_attempted.fetch_add(1, Ordering::Relaxed);

        // Validate local + remote offers.
        if let Err(e) = self.local_offer.validate() {
            self.stats.total_failed.fetch_add(1, Ordering::Relaxed);
            return Err(e);
        }
        if let Err(e) = remote.validate() {
            self.stats.total_failed.fetch_add(1, Ordering::Relaxed);
            return Err(e);
        }

        // Version compatibility.
        let local_ver = &self.local_offer.protocol_version;
        let remote_ver = &remote.protocol_version;
        if !local_ver.is_compatible_with(remote_ver) {
            self.stats.total_failed.fetch_add(1, Ordering::Relaxed);
            return Err(HandshakeError::IncompatibleVersion {
                local: local_ver.to_string(),
                remote: remote_ver.to_string(),
            });
        }

        // Feature intersection via bitmask.
        let local_bits = self.local_offer.feature_bits();
        let remote_bits = remote.feature_bits();
        let common_bits = local_bits & remote_bits;

        // At least one shared feature is required for a meaningful session.
        // (A peer with zero features advertised on both sides is unusual but
        //  we allow it only when both sides explicitly sent zero features;
        //  see the `NoCommonFeatures` variant for the non-zero / empty-intersection case.)
        if local_bits != 0 && remote_bits != 0 && common_bits == 0 {
            self.stats.total_failed.fetch_add(1, Ordering::Relaxed);
            return Err(HandshakeError::NoCommonFeatures);
        }

        let negotiated_features = FeatureFlag::from_bits(common_bits);

        // Agreed version: the lower of the two (most conservative superset).
        let agreed_version = std::cmp::min(local_ver, remote_ver).clone();

        // Frame size: take the more restrictive limit.
        let max_frame_size = std::cmp::min(self.local_offer.max_frame_size, remote.max_frame_size);

        // Timestamp: use the current instant.  Since we are in a no-std-async
        // context we derive it from std::time rather than tokio.
        let negotiated_at_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        self.stats.total_succeeded.fetch_add(1, Ordering::Relaxed);

        Ok(HandshakeResult {
            agreed_version,
            negotiated_features,
            max_frame_size,
            local_peer_id: self.local_offer.peer_id.clone(),
            remote_peer_id: remote.peer_id.clone(),
            negotiated_at_ms,
        })
    }
}

// ─────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ProtocolVersion helpers ────────────────

    fn v(major: u16, minor: u16, patch: u16) -> ProtocolVersion {
        ProtocolVersion::new(major, minor, patch)
    }

    fn make_offer(
        peer_id: &str,
        version: ProtocolVersion,
        features: Vec<FeatureFlag>,
        max_frame_size: u32,
    ) -> HandshakeOffer {
        HandshakeOffer {
            peer_id: peer_id.to_string(),
            protocol_version: version,
            supported_features: features,
            max_frame_size,
            timestamp_ms: 0,
        }
    }

    // ── 1. ProtocolVersion::is_compatible_with ─

    #[test]
    fn compatible_same_major_same_minor() {
        let a = v(1, 0, 0);
        let b = v(1, 0, 0);
        assert!(a.is_compatible_with(&b));
    }

    #[test]
    fn compatible_same_major_different_minor() {
        let a = v(1, 0, 0);
        let b = v(1, 5, 3);
        assert!(a.is_compatible_with(&b));
    }

    #[test]
    fn incompatible_different_major() {
        let a = v(1, 0, 0);
        let b = v(2, 0, 0);
        assert!(!a.is_compatible_with(&b));
    }

    #[test]
    fn incompatible_major_zero_vs_one() {
        let a = v(0, 9, 9);
        let b = v(1, 0, 0);
        assert!(!a.is_compatible_with(&b));
    }

    // ── 2. ProtocolVersion ordering ───────────

    #[test]
    fn version_ordering_major_takes_precedence() {
        assert!(v(2, 0, 0) > v(1, 99, 99));
    }

    #[test]
    fn version_ordering_minor_within_same_major() {
        assert!(v(1, 3, 0) > v(1, 2, 99));
    }

    #[test]
    fn version_ordering_patch_within_same_major_minor() {
        assert!(v(1, 0, 5) > v(1, 0, 4));
    }

    #[test]
    fn version_ordering_equal() {
        assert_eq!(v(1, 2, 3), v(1, 2, 3));
        assert!((v(1, 2, 3) <= v(1, 2, 3)));
    }

    // ── 3. ProtocolVersion Display ────────────

    #[test]
    fn version_display() {
        assert_eq!(v(1, 2, 3).to_string(), "1.2.3");
        assert_eq!(v(0, 0, 0).to_string(), "0.0.0");
    }

    // ── 4. FeatureFlag::flag_bit ──────────────

    #[test]
    fn feature_flag_bit_values() {
        assert_eq!(FeatureFlag::Encryption.flag_bit(), 0);
        assert_eq!(FeatureFlag::Compression.flag_bit(), 1);
        assert_eq!(FeatureFlag::VectorSearch.flag_bit(), 2);
        assert_eq!(FeatureFlag::TensorLogic.flag_bit(), 3);
        assert_eq!(FeatureFlag::GradientSync.flag_bit(), 4);
        assert_eq!(FeatureFlag::WebRtc.flag_bit(), 5);
    }

    // ── 5. FeatureFlag::from_bits roundtrip ───

    #[test]
    fn feature_flag_from_bits_roundtrip_all() {
        let all_bits: u32 = FeatureFlag::ALL
            .iter()
            .fold(0, |acc, f| acc | (1 << f.flag_bit()));
        let decoded = FeatureFlag::from_bits(all_bits);
        // All flags should be present.
        for flag in FeatureFlag::ALL {
            assert!(
                decoded.contains(flag),
                "{:?} missing from decoded set",
                flag
            );
        }
    }

    #[test]
    fn feature_flag_from_bits_subset() {
        // Only Encryption (bit 0) + VectorSearch (bit 2) = 0b0000_0101 = 5
        let bits: u32 = (1 << 0) | (1 << 2);
        let flags = FeatureFlag::from_bits(bits);
        assert_eq!(flags.len(), 2);
        assert!(flags.contains(&FeatureFlag::Encryption));
        assert!(flags.contains(&FeatureFlag::VectorSearch));
    }

    #[test]
    fn feature_flag_from_bits_zero() {
        let flags = FeatureFlag::from_bits(0);
        assert!(flags.is_empty());
    }

    // ── 6. HandshakeOffer::feature_bits ───────

    #[test]
    fn offer_feature_bits_encoding() {
        let offer = make_offer(
            "peer",
            v(1, 0, 0),
            vec![FeatureFlag::Encryption, FeatureFlag::GradientSync],
            DEFAULT_MAX_FRAME_SIZE,
        );
        // Encryption = bit 0, GradientSync = bit 4  →  0b0001_0001 = 17
        assert_eq!(offer.feature_bits(), (1 << 0) | (1 << 4));
    }

    #[test]
    fn offer_feature_bits_empty() {
        let offer = make_offer("peer", v(1, 0, 0), vec![], DEFAULT_MAX_FRAME_SIZE);
        assert_eq!(offer.feature_bits(), 0);
    }

    // ── 7. negotiate succeeds with compatible versions ─

    #[test]
    fn negotiate_success_compatible_versions() {
        let local = make_offer(
            "local",
            v(1, 0, 0),
            vec![FeatureFlag::Encryption, FeatureFlag::Compression],
            DEFAULT_MAX_FRAME_SIZE,
        );
        let handshaker = ProtocolHandshaker::new(local);
        let remote = make_offer(
            "remote",
            v(1, 2, 0),
            vec![FeatureFlag::Encryption, FeatureFlag::VectorSearch],
            DEFAULT_MAX_FRAME_SIZE,
        );
        let result = handshaker
            .negotiate(&remote)
            .expect("negotiate should succeed");
        assert_eq!(result.local_peer_id, "local");
        assert_eq!(result.remote_peer_id, "remote");
        // agreed version = min(1.0.0, 1.2.0) = 1.0.0
        assert_eq!(result.agreed_version, v(1, 0, 0));
    }

    // ── 8. negotiate fails with incompatible major versions ─

    #[test]
    fn negotiate_fails_incompatible_major() {
        let local = make_offer(
            "local",
            v(1, 0, 0),
            vec![FeatureFlag::Encryption],
            DEFAULT_MAX_FRAME_SIZE,
        );
        let handshaker = ProtocolHandshaker::new(local);
        let remote = make_offer(
            "remote",
            v(2, 0, 0),
            vec![FeatureFlag::Encryption],
            DEFAULT_MAX_FRAME_SIZE,
        );
        match handshaker.negotiate(&remote) {
            Err(HandshakeError::IncompatibleVersion { local, remote }) => {
                assert_eq!(local, "1.0.0");
                assert_eq!(remote, "2.0.0");
            }
            other => panic!("expected IncompatibleVersion, got {:?}", other),
        }
    }

    // ── 9. Feature intersection correctness ───

    #[test]
    fn negotiate_feature_intersection() {
        let local = make_offer(
            "local",
            v(1, 0, 0),
            vec![
                FeatureFlag::Encryption,
                FeatureFlag::Compression,
                FeatureFlag::VectorSearch,
            ],
            DEFAULT_MAX_FRAME_SIZE,
        );
        let handshaker = ProtocolHandshaker::new(local);
        let remote = make_offer(
            "remote",
            v(1, 0, 0),
            vec![FeatureFlag::Encryption, FeatureFlag::TensorLogic],
            DEFAULT_MAX_FRAME_SIZE,
        );
        let result = handshaker
            .negotiate(&remote)
            .expect("negotiate should succeed");
        assert_eq!(result.negotiated_features.len(), 1);
        assert!(result
            .negotiated_features
            .contains(&FeatureFlag::Encryption));
    }

    // ── 10. NoCommonFeatures error ────────────

    #[test]
    fn negotiate_no_common_features() {
        let local = make_offer(
            "local",
            v(1, 0, 0),
            vec![FeatureFlag::Compression],
            DEFAULT_MAX_FRAME_SIZE,
        );
        let handshaker = ProtocolHandshaker::new(local);
        let remote = make_offer(
            "remote",
            v(1, 0, 0),
            vec![FeatureFlag::VectorSearch],
            DEFAULT_MAX_FRAME_SIZE,
        );
        assert!(matches!(
            handshaker.negotiate(&remote),
            Err(HandshakeError::NoCommonFeatures)
        ));
    }

    // ── 11. max_frame_size takes minimum ──────

    #[test]
    fn negotiate_frame_size_takes_minimum() {
        let local = make_offer(
            "local",
            v(1, 0, 0),
            vec![FeatureFlag::Encryption],
            8 * 1024 * 1024,
        );
        let handshaker = ProtocolHandshaker::new(local);
        let remote = make_offer(
            "remote",
            v(1, 0, 0),
            vec![FeatureFlag::Encryption],
            2 * 1024 * 1024,
        );
        let result = handshaker
            .negotiate(&remote)
            .expect("negotiate should succeed");
        assert_eq!(result.max_frame_size, 2 * 1024 * 1024);
    }

    #[test]
    fn negotiate_frame_size_local_smaller() {
        let local = make_offer("local", v(1, 0, 0), vec![FeatureFlag::Encryption], 1024);
        let handshaker = ProtocolHandshaker::new(local);
        let remote = make_offer(
            "remote",
            v(1, 0, 0),
            vec![FeatureFlag::Encryption],
            4 * 1024 * 1024,
        );
        let result = handshaker
            .negotiate(&remote)
            .expect("negotiate should succeed");
        assert_eq!(result.max_frame_size, 1024);
    }

    // ── 12. Stats accumulation ────────────────

    #[test]
    fn stats_accumulate_success() {
        let local = make_offer(
            "local",
            v(1, 0, 0),
            vec![FeatureFlag::Encryption],
            DEFAULT_MAX_FRAME_SIZE,
        );
        let handshaker = ProtocolHandshaker::new(local);
        let remote = make_offer(
            "remote",
            v(1, 0, 0),
            vec![FeatureFlag::Encryption],
            DEFAULT_MAX_FRAME_SIZE,
        );

        handshaker
            .negotiate(&remote)
            .expect("negotiate should succeed");
        handshaker
            .negotiate(&remote)
            .expect("negotiate should succeed");

        let snap = handshaker.stats.snapshot();
        assert_eq!(snap.total_attempted, 2);
        assert_eq!(snap.total_succeeded, 2);
        assert_eq!(snap.total_failed, 0);
    }

    #[test]
    fn stats_accumulate_failure() {
        let local = make_offer(
            "local",
            v(1, 0, 0),
            vec![FeatureFlag::Encryption],
            DEFAULT_MAX_FRAME_SIZE,
        );
        let handshaker = ProtocolHandshaker::new(local);
        // Will fail due to incompatible versions.
        let remote = make_offer(
            "remote",
            v(2, 0, 0),
            vec![FeatureFlag::Encryption],
            DEFAULT_MAX_FRAME_SIZE,
        );

        let _ = handshaker.negotiate(&remote);
        let _ = handshaker.negotiate(&remote);

        let snap = handshaker.stats.snapshot();
        assert_eq!(snap.total_attempted, 2);
        assert_eq!(snap.total_succeeded, 0);
        assert_eq!(snap.total_failed, 2);
    }

    // ── 13. InvalidOffer: empty peer_id ───────

    #[test]
    fn negotiate_invalid_offer_empty_peer_id() {
        let local = make_offer(
            "local",
            v(1, 0, 0),
            vec![FeatureFlag::Encryption],
            DEFAULT_MAX_FRAME_SIZE,
        );
        let handshaker = ProtocolHandshaker::new(local);
        let bad_remote = make_offer(
            "",
            v(1, 0, 0),
            vec![FeatureFlag::Encryption],
            DEFAULT_MAX_FRAME_SIZE,
        );
        assert!(matches!(
            handshaker.negotiate(&bad_remote),
            Err(HandshakeError::InvalidOffer(_))
        ));
    }

    // ── 14. Both sides zero features (no error) ─

    #[test]
    fn negotiate_both_zero_features_succeeds() {
        // When both sides send zero feature flags the intersection is also
        // zero, but `NoCommonFeatures` is only raised when *at least one*
        // side advertises features. Two bare peers may still negotiate a
        // frame size and version.
        let local = make_offer("local", v(1, 0, 0), vec![], DEFAULT_MAX_FRAME_SIZE);
        let handshaker = ProtocolHandshaker::new(local);
        let remote = make_offer("remote", v(1, 0, 0), vec![], DEFAULT_MAX_FRAME_SIZE);
        let result = handshaker
            .negotiate(&remote)
            .expect("negotiate should succeed");
        assert!(result.negotiated_features.is_empty());
    }

    // ── 15. local_features helper ─────────────

    #[test]
    fn local_features_returns_offer_features() {
        let local = make_offer(
            "local",
            v(1, 0, 0),
            vec![FeatureFlag::Encryption, FeatureFlag::WebRtc],
            DEFAULT_MAX_FRAME_SIZE,
        );
        let handshaker = ProtocolHandshaker::new(local);
        let features = handshaker.local_features();
        assert_eq!(features.len(), 2);
        assert!(features.contains(&FeatureFlag::Encryption));
        assert!(features.contains(&FeatureFlag::WebRtc));
    }

    // ── 16. agreed_version is the lower one ───

    #[test]
    fn negotiate_agreed_version_is_lower() {
        // local is 1.3.0, remote is 1.1.0 → agreed should be 1.1.0
        let local = make_offer(
            "local",
            v(1, 3, 0),
            vec![FeatureFlag::Encryption],
            DEFAULT_MAX_FRAME_SIZE,
        );
        let handshaker = ProtocolHandshaker::new(local);
        let remote = make_offer(
            "remote",
            v(1, 1, 0),
            vec![FeatureFlag::Encryption],
            DEFAULT_MAX_FRAME_SIZE,
        );
        let result = handshaker
            .negotiate(&remote)
            .expect("negotiate should succeed");
        assert_eq!(result.agreed_version, v(1, 1, 0));
    }
}
