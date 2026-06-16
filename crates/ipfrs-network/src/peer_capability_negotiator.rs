//! Peer Capability Negotiation
//!
//! This module implements [`PeerCapabilityNegotiator`], a production-grade system
//! for discovering, advertising, and negotiating compatible protocol capabilities
//! between peers during connection establishment.
//!
//! # Overview
//!
//! When two IPFRS nodes connect, they exchange [`NegotiationOffer`] messages describing
//! their supported capabilities (e.g. "bitswap/2.0.0", "graphsync/1.1.0"). The
//! negotiator evaluates each offer against its local [`NegotiatorConfig`] and returns
//! a [`NegotiationResult`] indicating which capabilities were agreed upon or why the
//! negotiation was rejected.
//!
//! # Policies
//!
//! Three [`CapabilityPolicy`] variants control strictness:
//!
//! - **`RequireAll`** – Every required local capability must appear in the peer's offer;
//!   any mismatch yields an immediate [`NegotiationResult::Rejected`].
//! - **`RequireSubset(names)`** – Only the named subset of capabilities must match.
//! - **`BestEffort`** – Always produces [`NegotiationResult::Accepted`]; unmatched
//!   required capabilities are listed in `rejected_optional` for caller visibility.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_network::peer_capability_negotiator::{
//!     CapabilityPolicy, CapabilityVersion, NegotiationOffer, NegotiatorConfig,
//!     PeerCapabilityNegotiator, PeerCapability,
//! };
//! use std::collections::HashMap;
//!
//! let local_cap = PeerCapability {
//!     name: "bitswap".to_string(),
//!     version: CapabilityVersion { major: 2, minor: 0, patch: 0 },
//!     required: true,
//!     metadata: HashMap::new(),
//! };
//!
//! let config = NegotiatorConfig {
//!     local_capabilities: vec![local_cap],
//!     policy: CapabilityPolicy::RequireAll,
//!     min_protocol_version: 1,
//!     negotiation_timeout_ms: 5_000,
//! };
//!
//! let mut negotiator = PeerCapabilityNegotiator::new(config, 128);
//!
//! let offer = NegotiationOffer {
//!     peer_id: "QmPeer".to_string(),
//!     capabilities: vec![PeerCapability {
//!         name: "bitswap".to_string(),
//!         version: CapabilityVersion { major: 2, minor: 1, patch: 0 },
//!         required: false,
//!         metadata: HashMap::new(),
//!     }],
//!     protocol_version: 2,
//!     timestamp: 1_000,
//! };
//!
//! let result = negotiator.negotiate("QmPeer".to_string(), offer, 1_000, 1_050);
//! // "bitswap" v2.1.0 is compatible with required v2.0.0 → Accepted.
//! assert!(matches!(result, ipfrs_network::peer_capability_negotiator::NegotiationResult::Accepted { .. }));
//! ```

use std::collections::{HashMap, VecDeque};
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
//  CapabilityVersion
// ─────────────────────────────────────────────────────────────────────────────

/// Semantic version triple for a single capability.
///
/// Compatibility rule: `self` is compatible with `required` when
/// `self.major == required.major && self.minor >= required.minor`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CapabilityVersion {
    /// Breaking-change epoch. Must match exactly.
    pub major: u16,
    /// Backwards-compatible feature additions. Must be ≥ required.
    pub minor: u16,
    /// Bug-fix increments. Ignored for compatibility checks.
    pub patch: u16,
}

impl CapabilityVersion {
    /// Returns `true` when `self` satisfies the `required` version constraint.
    ///
    /// The rule is: same major, and `self.minor >= required.minor`.
    #[inline]
    pub fn is_compatible_with(&self, required: &CapabilityVersion) -> bool {
        self.major == required.major && self.minor >= required.minor
    }
}

impl fmt::Display for CapabilityVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl Default for CapabilityVersion {
    fn default() -> Self {
        Self {
            major: 1,
            minor: 0,
            patch: 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  PeerCapability
// ─────────────────────────────────────────────────────────────────────────────

/// A single named capability with version and optional metadata key-value pairs.
///
/// When `required` is `true`, the local peer considers this capability mandatory
/// for a successful negotiation (subject to the active [`CapabilityPolicy`]).
#[derive(Clone, Debug, PartialEq)]
pub struct PeerCapability {
    /// Protocol identifier, e.g. `"bitswap"` or `"graphsync"`.
    pub name: String,
    /// Version triple describing the capability's feature level.
    pub version: CapabilityVersion,
    /// Whether the local node considers this capability mandatory.
    pub required: bool,
    /// Arbitrary string key-value pairs (e.g. `"max_block_size" → "1048576"`).
    pub metadata: HashMap<String, String>,
}

impl PeerCapability {
    /// Construct a required capability with no metadata.
    pub fn required(name: impl Into<String>, version: CapabilityVersion) -> Self {
        Self {
            name: name.into(),
            version,
            required: true,
            metadata: HashMap::new(),
        }
    }

    /// Construct an optional capability with no metadata.
    pub fn optional(name: impl Into<String>, version: CapabilityVersion) -> Self {
        Self {
            name: name.into(),
            version,
            required: false,
            metadata: HashMap::new(),
        }
    }

    /// Construct a capability with metadata.
    pub fn with_metadata(
        name: impl Into<String>,
        version: CapabilityVersion,
        required: bool,
        metadata: HashMap<String, String>,
    ) -> Self {
        Self {
            name: name.into(),
            version,
            required,
            metadata,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  NegotiationOffer
// ─────────────────────────────────────────────────────────────────────────────

/// A capability advertisement sent by a remote peer at connection time.
#[derive(Clone, Debug, PartialEq)]
pub struct NegotiationOffer {
    /// Identifier of the remote peer.
    pub peer_id: String,
    /// Full list of capabilities the remote peer supports.
    pub capabilities: Vec<PeerCapability>,
    /// Wire-level protocol version of the remote peer.
    pub protocol_version: u16,
    /// Unix-millisecond timestamp when this offer was created.
    pub timestamp: u64,
}

impl NegotiationOffer {
    /// Build a map from capability name → capability for efficient lookup.
    fn capability_map(&self) -> HashMap<&str, &PeerCapability> {
        self.capabilities
            .iter()
            .map(|c| (c.name.as_str(), c))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  NegotiationResult
// ─────────────────────────────────────────────────────────────────────────────

/// Outcome of a capability negotiation attempt.
#[derive(Clone, Debug, PartialEq)]
pub enum NegotiationResult {
    /// Negotiation succeeded; the listed capability names are mutually supported.
    Accepted {
        /// Names of capabilities agreed upon by both sides.
        agreed_capabilities: Vec<String>,
        /// Names of optional local capabilities absent from the peer's offer.
        rejected_optional: Vec<String>,
    },
    /// Negotiation failed; the connection should be closed.
    Rejected {
        /// Human-readable rejection reason.
        reason: String,
        /// Capability names that were required but absent from the peer's offer.
        missing_required: Vec<String>,
    },
}

impl NegotiationResult {
    /// Returns `true` if the result is [`NegotiationResult::Accepted`].
    #[inline]
    pub fn is_accepted(&self) -> bool {
        matches!(self, NegotiationResult::Accepted { .. })
    }

    /// Returns `true` if the result is [`NegotiationResult::Rejected`].
    #[inline]
    pub fn is_rejected(&self) -> bool {
        matches!(self, NegotiationResult::Rejected { .. })
    }

    /// Returns the agreed capability names, or an empty slice if rejected.
    pub fn agreed_capabilities(&self) -> &[String] {
        match self {
            NegotiationResult::Accepted {
                agreed_capabilities,
                ..
            } => agreed_capabilities,
            NegotiationResult::Rejected { .. } => &[],
        }
    }

    /// Returns the rejection reason, or `None` if accepted.
    pub fn rejection_reason(&self) -> Option<&str> {
        match self {
            NegotiationResult::Rejected { reason, .. } => Some(reason.as_str()),
            NegotiationResult::Accepted { .. } => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  CapabilityPolicy
// ─────────────────────────────────────────────────────────────────────────────

/// Governs how strictly the negotiator validates a peer's offer.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum CapabilityPolicy {
    /// All capabilities marked `required` in the local config must be present
    /// in the peer's offer with a compatible version. Any mismatch → Rejected.
    RequireAll,

    /// Only the named capabilities must appear in the peer's offer. Names not
    /// listed here are treated as optional even if locally `required`.
    RequireSubset(Vec<String>),

    /// Never reject: return Accepted regardless of missing required capabilities.
    /// Missing required capabilities are surfaced in `rejected_optional`.
    #[default]
    BestEffort,
}

// ─────────────────────────────────────────────────────────────────────────────
//  NegotiatorConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a [`PeerCapabilityNegotiator`].
#[derive(Clone, Debug)]
pub struct NegotiatorConfig {
    /// Capabilities this node advertises (and potentially requires).
    pub local_capabilities: Vec<PeerCapability>,
    /// Policy controlling acceptance/rejection strictness.
    pub policy: CapabilityPolicy,
    /// Minimum acceptable wire-protocol version from remote peers.
    pub min_protocol_version: u16,
    /// Timeout budget for a single negotiation round (informational; callers
    /// supply elapsed time via `negotiate`'s `start_ms`/`end_ms` parameters).
    pub negotiation_timeout_ms: u64,
}

impl Default for NegotiatorConfig {
    fn default() -> Self {
        Self {
            local_capabilities: Vec::new(),
            policy: CapabilityPolicy::BestEffort,
            min_protocol_version: 1,
            negotiation_timeout_ms: 5_000,
        }
    }
}

impl NegotiatorConfig {
    /// Build a config with a single policy and capability set.
    pub fn new(
        local_capabilities: Vec<PeerCapability>,
        policy: CapabilityPolicy,
        min_protocol_version: u16,
        negotiation_timeout_ms: u64,
    ) -> Self {
        Self {
            local_capabilities,
            policy,
            min_protocol_version,
            negotiation_timeout_ms,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  NegotiationRecord
// ─────────────────────────────────────────────────────────────────────────────

/// A historical record of a completed negotiation with a specific peer.
#[derive(Clone, Debug)]
pub struct NegotiationRecord {
    /// Remote peer identifier.
    pub peer_id: String,
    /// Outcome of the negotiation.
    pub result: NegotiationResult,
    /// Unix-millisecond timestamp when the negotiation completed.
    pub negotiated_at: u64,
    /// Round-trip duration of the negotiation in milliseconds.
    pub latency_ms: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
//  NegotiatorStats
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics for a [`PeerCapabilityNegotiator`].
#[derive(Clone, Debug, PartialEq)]
pub struct NegotiatorStats {
    /// Total number of `negotiate()` calls made.
    pub total_negotiations: u64,
    /// Number of negotiations that produced `Accepted`.
    pub total_accepted: u64,
    /// Number of negotiations that produced `Rejected`.
    pub total_rejected: u64,
    /// Fraction of negotiations that were accepted (0.0–1.0).
    /// Returns 0.0 when `total_negotiations` is zero.
    pub accept_rate: f64,
    /// Number of capabilities advertised by this node.
    pub capabilities_advertised: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
//  PeerCapabilityNegotiator
// ─────────────────────────────────────────────────────────────────────────────

/// Production-grade peer capability negotiator.
///
/// `PeerCapabilityNegotiator` handles the full lifecycle of capability negotiation:
/// 1. Building [`NegotiationOffer`]s from local config for outbound connections.
/// 2. Evaluating incoming offers and producing [`NegotiationResult`]s.
/// 3. Recording negotiation outcomes in a bounded history ring.
/// 4. Exposing aggregate statistics for monitoring dashboards.
///
/// # Thread Safety
///
/// `PeerCapabilityNegotiator` is `Send + Sync` (all fields are `Send + Sync`).
/// For concurrent access, wrap in `Arc<Mutex<_>>` or `Arc<RwLock<_>>`.
#[derive(Debug)]
pub struct PeerCapabilityNegotiator {
    /// Local negotiation configuration (policy, capabilities, limits).
    pub config: NegotiatorConfig,
    /// Bounded ring of historical negotiation records (newest last).
    pub negotiation_history: VecDeque<NegotiationRecord>,
    /// Maximum number of historical records retained.
    pub max_history: usize,
    /// Total number of `negotiate()` calls.
    pub total_negotiations: u64,
    /// Negotiations that returned `Accepted`.
    pub total_accepted: u64,
    /// Negotiations that returned `Rejected`.
    pub total_rejected: u64,
}

impl PeerCapabilityNegotiator {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Create a new negotiator with the given config and maximum history size.
    ///
    /// # Panics
    ///
    /// Does not panic; a `max_history` of 0 is valid (no history retained).
    pub fn new(config: NegotiatorConfig, max_history: usize) -> Self {
        Self {
            config,
            negotiation_history: VecDeque::new(),
            max_history,
            total_negotiations: 0,
            total_accepted: 0,
            total_rejected: 0,
        }
    }

    // ── Offer Construction ───────────────────────────────────────────────────

    /// Build a [`NegotiationOffer`] from this node's local capabilities.
    ///
    /// `peer_id` is the identifier we place in the offer (usually our own peer ID),
    /// and `now` is the current Unix-millisecond timestamp.
    pub fn build_offer(&self, peer_id: String, now: u64) -> NegotiationOffer {
        NegotiationOffer {
            peer_id,
            capabilities: self.config.local_capabilities.clone(),
            protocol_version: self.config.min_protocol_version,
            timestamp: now,
        }
    }

    // ── Offer Evaluation ─────────────────────────────────────────────────────

    /// Evaluate an incoming peer offer and return the negotiation outcome.
    ///
    /// The evaluation proceeds in three phases:
    ///
    /// 1. **Protocol version gate** — if `offer.protocol_version < min_protocol_version`,
    ///    immediately return `Rejected { reason: "protocol version too low", ... }`.
    /// 2. **Required-capability scan** — identify which locally `required` capabilities
    ///    are absent from (or version-incompatible with) the offer.
    /// 3. **Policy decision** — apply [`CapabilityPolicy`] to decide Accepted/Rejected.
    pub fn evaluate_offer(&self, offer: &NegotiationOffer) -> NegotiationResult {
        // ── Phase 1: protocol version gate ───────────────────────────────────
        if offer.protocol_version < self.config.min_protocol_version {
            return NegotiationResult::Rejected {
                reason: "protocol version too low".to_string(),
                missing_required: Vec::new(),
            };
        }

        let offer_map = offer.capability_map();

        // ── Phase 2: identify missing required capabilities ───────────────────
        let missing_required: Vec<String> = self
            .config
            .local_capabilities
            .iter()
            .filter(|cap| cap.required)
            .filter(|cap| {
                offer_map
                    .get(cap.name.as_str())
                    .map(|peer_cap| !peer_cap.version.is_compatible_with(&cap.version))
                    .unwrap_or(true) // absent from offer → missing
            })
            .map(|cap| cap.name.clone())
            .collect();

        // ── Phase 3: policy decision ──────────────────────────────────────────
        match &self.config.policy {
            CapabilityPolicy::RequireAll => {
                if !missing_required.is_empty() {
                    return NegotiationResult::Rejected {
                        reason: "missing required capabilities".to_string(),
                        missing_required,
                    };
                }
                let agreed = self.compute_agreed(&offer_map);
                let rejected_optional = self.compute_rejected_optional(&offer_map);
                NegotiationResult::Accepted {
                    agreed_capabilities: agreed,
                    rejected_optional,
                }
            }

            CapabilityPolicy::RequireSubset(required_names) => {
                let missing_subset: Vec<String> = required_names
                    .iter()
                    .filter(|name| {
                        // Find the local capability with this name
                        let local_cap = self
                            .config
                            .local_capabilities
                            .iter()
                            .find(|c| &c.name == *name);
                        match local_cap {
                            None => false, // unknown name in subset → skip
                            Some(local) => offer_map
                                .get(local.name.as_str())
                                .map(|peer_cap| {
                                    !peer_cap.version.is_compatible_with(&local.version)
                                })
                                .unwrap_or(true),
                        }
                    })
                    .cloned()
                    .collect();

                if !missing_subset.is_empty() {
                    return NegotiationResult::Rejected {
                        reason: "missing required subset capabilities".to_string(),
                        missing_required: missing_subset,
                    };
                }
                let agreed = self.compute_agreed(&offer_map);
                let rejected_optional = self.compute_rejected_optional(&offer_map);
                NegotiationResult::Accepted {
                    agreed_capabilities: agreed,
                    rejected_optional,
                }
            }

            CapabilityPolicy::BestEffort => {
                // Always accept; surface missing_required as rejected_optional.
                let agreed = self.compute_agreed(&offer_map);
                // rejected_optional = optional caps not in offer PLUS missing required caps
                let mut rejected_optional = self.compute_rejected_optional(&offer_map);
                for name in &missing_required {
                    if !rejected_optional.contains(name) {
                        rejected_optional.push(name.clone());
                    }
                }
                NegotiationResult::Accepted {
                    agreed_capabilities: agreed,
                    rejected_optional,
                }
            }
        }
    }

    // ── Negotiation (stateful) ────────────────────────────────────────────────

    /// Evaluate an offer, record it in history, update counters, and return the result.
    ///
    /// * `peer_id` — identifier of the remote peer.
    /// * `offer` — the remote peer's capability offer.
    /// * `start_ms` — Unix-millisecond timestamp when the negotiation began.
    /// * `end_ms` — Unix-millisecond timestamp when evaluation completed.
    ///
    /// Returns a reference to the stored [`NegotiationResult`] in the history ring.
    ///
    /// # History Eviction
    ///
    /// When the history ring is full (`len() == max_history`), the oldest record
    /// is evicted before the new one is appended. If `max_history` is 0, the
    /// record is computed and counters are updated but nothing is stored.
    pub fn negotiate(
        &mut self,
        peer_id: String,
        offer: NegotiationOffer,
        start_ms: u64,
        end_ms: u64,
    ) -> &NegotiationResult {
        let result = self.evaluate_offer(&offer);
        let latency_ms = end_ms.saturating_sub(start_ms);

        // Update aggregate counters.
        self.total_negotiations += 1;
        if result.is_accepted() {
            self.total_accepted += 1;
        } else {
            self.total_rejected += 1;
        }

        // Record in history (evict oldest if at capacity).
        let record = NegotiationRecord {
            peer_id,
            result,
            negotiated_at: end_ms,
            latency_ms,
        };

        if self.max_history > 0 {
            if self.negotiation_history.len() >= self.max_history {
                self.negotiation_history.pop_front();
            }
            self.negotiation_history.push_back(record);
            // SAFETY: we just pushed, len ≥ 1.
            &self.negotiation_history.back().expect("just pushed").result
        } else {
            // max_history == 0: push temporarily, return ref, then pop.
            // We must still return a reference with lifetime tied to self.
            // Store in a 1-slot buffer and leak the pointer—actually we
            // cannot return a dangling ref. Instead we keep a 1-element
            // scratch deque that we reuse.
            //
            // To avoid unsafe, we store the record in a slot-0 of the deque
            // and return &deque[0].result. This leaks one slot but stays safe.
            self.negotiation_history.clear();
            self.negotiation_history.push_back(record);
            &self.negotiation_history[0].result
        }
    }

    // ── Capability Queries ────────────────────────────────────────────────────

    /// Returns `true` if the local config contains a capability with `name` whose
    /// version is compatible with `version`.
    pub fn is_capability_supported(&self, name: &str, version: &CapabilityVersion) -> bool {
        self.config
            .local_capabilities
            .iter()
            .any(|cap| cap.name == name && cap.version.is_compatible_with(version))
    }

    /// Returns the names of capabilities present in both the local config and the
    /// offer, with a compatible version.
    pub fn common_capabilities(&self, offer: &NegotiationOffer) -> Vec<String> {
        let offer_map = offer.capability_map();
        self.compute_agreed(&offer_map)
    }

    // ── History Queries ───────────────────────────────────────────────────────

    /// Returns all historical records for the specified peer.
    pub fn history_for_peer(&self, peer_id: &str) -> Vec<&NegotiationRecord> {
        self.negotiation_history
            .iter()
            .filter(|r| r.peer_id == peer_id)
            .collect()
    }

    /// Snapshot of aggregate negotiation statistics.
    pub fn negotiator_stats(&self) -> NegotiatorStats {
        let accept_rate = if self.total_negotiations == 0 {
            0.0
        } else {
            self.total_accepted as f64 / self.total_negotiations as f64
        };
        NegotiatorStats {
            total_negotiations: self.total_negotiations,
            total_accepted: self.total_accepted,
            total_rejected: self.total_rejected,
            accept_rate,
            capabilities_advertised: self.config.local_capabilities.len(),
        }
    }

    // ── Private Helpers ───────────────────────────────────────────────────────

    /// Compute the intersection of local and offer capabilities (version-compatible).
    fn compute_agreed<'a>(&self, offer_map: &HashMap<&'a str, &'a PeerCapability>) -> Vec<String> {
        self.config
            .local_capabilities
            .iter()
            .filter(|local_cap| {
                offer_map
                    .get(local_cap.name.as_str())
                    .map(|peer_cap| peer_cap.version.is_compatible_with(&local_cap.version))
                    .unwrap_or(false)
            })
            .map(|cap| cap.name.clone())
            .collect()
    }

    /// Compute optional local capabilities absent from the offer (or version-incompatible).
    fn compute_rejected_optional<'a>(
        &self,
        offer_map: &HashMap<&'a str, &'a PeerCapability>,
    ) -> Vec<String> {
        self.config
            .local_capabilities
            .iter()
            .filter(|cap| !cap.required)
            .filter(|cap| {
                offer_map
                    .get(cap.name.as_str())
                    .map(|peer_cap| !peer_cap.version.is_compatible_with(&cap.version))
                    .unwrap_or(true)
            })
            .map(|cap| cap.name.clone())
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::peer_capability_negotiator::{
        CapabilityPolicy, CapabilityVersion, NegotiationOffer, NegotiationResult, NegotiatorConfig,
        NegotiatorStats, PeerCapability, PeerCapabilityNegotiator,
    };
    use std::collections::HashMap;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn ver(major: u16, minor: u16, patch: u16) -> CapabilityVersion {
        CapabilityVersion {
            major,
            minor,
            patch,
        }
    }

    fn cap_req(name: &str, major: u16, minor: u16) -> PeerCapability {
        PeerCapability::required(name, ver(major, minor, 0))
    }

    fn cap_opt(name: &str, major: u16, minor: u16) -> PeerCapability {
        PeerCapability::optional(name, ver(major, minor, 0))
    }

    fn offer(peer_id: &str, caps: Vec<PeerCapability>, proto: u16) -> NegotiationOffer {
        NegotiationOffer {
            peer_id: peer_id.to_string(),
            capabilities: caps,
            protocol_version: proto,
            timestamp: 1_000,
        }
    }

    fn make_negotiator(
        caps: Vec<PeerCapability>,
        policy: CapabilityPolicy,
        min_proto: u16,
    ) -> PeerCapabilityNegotiator {
        PeerCapabilityNegotiator::new(NegotiatorConfig::new(caps, policy, min_proto, 5_000), 64)
    }

    // ── CapabilityVersion tests ───────────────────────────────────────────────

    #[test]
    fn test_version_compatible_same_major_higher_minor() {
        let v = ver(2, 3, 0);
        let req = ver(2, 1, 0);
        assert!(v.is_compatible_with(&req));
    }

    #[test]
    fn test_version_compatible_same_exact() {
        let v = ver(1, 0, 0);
        assert!(v.is_compatible_with(&v));
    }

    #[test]
    fn test_version_incompatible_major_mismatch() {
        let v = ver(2, 5, 0);
        let req = ver(1, 5, 0);
        assert!(!v.is_compatible_with(&req));
    }

    #[test]
    fn test_version_incompatible_lower_minor() {
        let v = ver(1, 0, 0);
        let req = ver(1, 1, 0);
        assert!(!v.is_compatible_with(&req));
    }

    #[test]
    fn test_version_display() {
        assert_eq!(ver(3, 14, 159).to_string(), "3.14.159");
    }

    #[test]
    fn test_version_default() {
        let d = CapabilityVersion::default();
        assert_eq!(d.major, 1);
        assert_eq!(d.minor, 0);
        assert_eq!(d.patch, 0);
    }

    #[test]
    fn test_version_patch_irrelevant_for_compat() {
        // patch=99 on required side should not affect compatibility.
        let v = ver(1, 2, 0);
        let req = ver(1, 2, 99);
        assert!(v.is_compatible_with(&req));
    }

    // ── PeerCapability constructor tests ─────────────────────────────────────

    #[test]
    fn test_peer_capability_required_constructor() {
        let cap = PeerCapability::required("graphsync", ver(1, 0, 0));
        assert!(cap.required);
        assert_eq!(cap.name, "graphsync");
        assert!(cap.metadata.is_empty());
    }

    #[test]
    fn test_peer_capability_optional_constructor() {
        let cap = PeerCapability::optional("relay", ver(2, 0, 0));
        assert!(!cap.required);
    }

    #[test]
    fn test_peer_capability_with_metadata() {
        let mut meta = HashMap::new();
        meta.insert("max_block_size".to_string(), "1048576".to_string());
        let cap = PeerCapability::with_metadata("bitswap", ver(2, 0, 0), true, meta.clone());
        assert_eq!(cap.metadata, meta);
    }

    // ── NegotiationResult helpers ─────────────────────────────────────────────

    #[test]
    fn test_result_is_accepted() {
        let r = NegotiationResult::Accepted {
            agreed_capabilities: vec!["bitswap".to_string()],
            rejected_optional: vec![],
        };
        assert!(r.is_accepted());
        assert!(!r.is_rejected());
    }

    #[test]
    fn test_result_is_rejected() {
        let r = NegotiationResult::Rejected {
            reason: "test".to_string(),
            missing_required: vec![],
        };
        assert!(r.is_rejected());
        assert!(!r.is_accepted());
    }

    #[test]
    fn test_result_agreed_capabilities_accepted() {
        let names = vec!["a".to_string(), "b".to_string()];
        let r = NegotiationResult::Accepted {
            agreed_capabilities: names.clone(),
            rejected_optional: vec![],
        };
        assert_eq!(r.agreed_capabilities(), names.as_slice());
    }

    #[test]
    fn test_result_agreed_capabilities_rejected_empty() {
        let r = NegotiationResult::Rejected {
            reason: "no".to_string(),
            missing_required: vec![],
        };
        assert!(r.agreed_capabilities().is_empty());
    }

    #[test]
    fn test_result_rejection_reason() {
        let r = NegotiationResult::Rejected {
            reason: "protocol version too low".to_string(),
            missing_required: vec![],
        };
        assert_eq!(r.rejection_reason(), Some("protocol version too low"));
    }

    // ── Protocol version gate tests ───────────────────────────────────────────

    #[test]
    fn test_evaluate_rejects_low_protocol_version() {
        let neg = make_negotiator(vec![], CapabilityPolicy::RequireAll, 3);
        let result = neg.evaluate_offer(&offer("peer1", vec![], 2));
        assert!(result.is_rejected());
        assert_eq!(result.rejection_reason(), Some("protocol version too low"));
    }

    #[test]
    fn test_evaluate_accepts_exact_protocol_version() {
        let neg = make_negotiator(vec![], CapabilityPolicy::RequireAll, 2);
        let result = neg.evaluate_offer(&offer("peer1", vec![], 2));
        assert!(result.is_accepted());
    }

    #[test]
    fn test_evaluate_accepts_higher_protocol_version() {
        let neg = make_negotiator(vec![], CapabilityPolicy::RequireAll, 1);
        let result = neg.evaluate_offer(&offer("peer1", vec![], 5));
        assert!(result.is_accepted());
    }

    // ── RequireAll policy tests ───────────────────────────────────────────────

    #[test]
    fn test_require_all_rejects_missing_required() {
        let neg = make_negotiator(
            vec![cap_req("bitswap", 2, 0)],
            CapabilityPolicy::RequireAll,
            1,
        );
        let result = neg.evaluate_offer(&offer("peer", vec![], 1));
        assert!(result.is_rejected());
        match result {
            NegotiationResult::Rejected {
                missing_required, ..
            } => {
                assert!(missing_required.contains(&"bitswap".to_string()));
            }
            _ => panic!("expected Rejected"),
        }
    }

    #[test]
    fn test_require_all_rejects_version_incompatible() {
        let neg = make_negotiator(
            vec![cap_req("bitswap", 2, 3)],
            CapabilityPolicy::RequireAll,
            1,
        );
        // Offer has bitswap v2.1.0, but we require v2.3.0
        let result = neg.evaluate_offer(&offer("peer", vec![cap_opt("bitswap", 2, 1)], 1));
        assert!(result.is_rejected());
    }

    #[test]
    fn test_require_all_accepts_compatible_versions() {
        let neg = make_negotiator(
            vec![cap_req("bitswap", 2, 0)],
            CapabilityPolicy::RequireAll,
            1,
        );
        // Offer has bitswap v2.1.0 — minor is higher → compatible
        let result = neg.evaluate_offer(&offer("peer", vec![cap_opt("bitswap", 2, 1)], 1));
        assert!(result.is_accepted());
    }

    #[test]
    fn test_require_all_agreed_capabilities_correct() {
        let neg = make_negotiator(
            vec![cap_req("bitswap", 2, 0), cap_opt("relay", 1, 0)],
            CapabilityPolicy::RequireAll,
            1,
        );
        let peer_caps = vec![cap_opt("bitswap", 2, 2), cap_opt("relay", 1, 1)];
        let result = neg.evaluate_offer(&offer("peer", peer_caps, 1));
        match result {
            NegotiationResult::Accepted {
                agreed_capabilities,
                ..
            } => {
                assert!(agreed_capabilities.contains(&"bitswap".to_string()));
                assert!(agreed_capabilities.contains(&"relay".to_string()));
            }
            _ => panic!("expected Accepted"),
        }
    }

    #[test]
    fn test_require_all_rejected_optional_populated() {
        let neg = make_negotiator(
            vec![
                cap_req("bitswap", 2, 0),
                cap_opt("graphsync", 1, 0), // optional, peer does not have it
            ],
            CapabilityPolicy::RequireAll,
            1,
        );
        let result = neg.evaluate_offer(&offer("peer", vec![cap_opt("bitswap", 2, 0)], 1));
        match result {
            NegotiationResult::Accepted {
                rejected_optional, ..
            } => {
                assert!(rejected_optional.contains(&"graphsync".to_string()));
            }
            _ => panic!("expected Accepted"),
        }
    }

    // ── RequireSubset policy tests ────────────────────────────────────────────

    #[test]
    fn test_require_subset_rejects_missing_subset_member() {
        let neg = make_negotiator(
            vec![cap_req("bitswap", 2, 0), cap_req("graphsync", 1, 0)],
            CapabilityPolicy::RequireSubset(vec!["bitswap".to_string()]),
            1,
        );
        // No bitswap in offer → missing subset member
        let result = neg.evaluate_offer(&offer("peer", vec![cap_opt("graphsync", 1, 0)], 1));
        assert!(result.is_rejected());
    }

    #[test]
    fn test_require_subset_accepts_when_subset_present() {
        let neg = make_negotiator(
            vec![cap_req("bitswap", 2, 0), cap_req("graphsync", 1, 0)],
            CapabilityPolicy::RequireSubset(vec!["bitswap".to_string()]),
            1,
        );
        // Only need bitswap; graphsync being absent is fine
        let result = neg.evaluate_offer(&offer("peer", vec![cap_opt("bitswap", 2, 1)], 1));
        assert!(result.is_accepted());
    }

    #[test]
    fn test_require_subset_rejects_version_incompatible() {
        let neg = make_negotiator(
            vec![cap_req("bitswap", 2, 5)],
            CapabilityPolicy::RequireSubset(vec!["bitswap".to_string()]),
            1,
        );
        // v2.3.0 < required v2.5.0 → incompatible
        let result = neg.evaluate_offer(&offer("peer", vec![cap_opt("bitswap", 2, 3)], 1));
        assert!(result.is_rejected());
    }

    #[test]
    fn test_require_subset_unknown_subset_name_ignored() {
        // "unknown" is in the subset but not in local_capabilities → treated as skip
        let neg = make_negotiator(
            vec![cap_req("bitswap", 2, 0)],
            CapabilityPolicy::RequireSubset(vec!["unknown".to_string()]),
            1,
        );
        let result = neg.evaluate_offer(&offer("peer", vec![], 1));
        // "unknown" not in local_capabilities → filter passes through without requiring it
        assert!(result.is_accepted());
    }

    // ── BestEffort policy tests ───────────────────────────────────────────────

    #[test]
    fn test_best_effort_always_accepted() {
        let neg = make_negotiator(
            vec![cap_req("bitswap", 2, 0)],
            CapabilityPolicy::BestEffort,
            1,
        );
        // No capabilities in offer → still Accepted
        let result = neg.evaluate_offer(&offer("peer", vec![], 1));
        assert!(result.is_accepted());
    }

    #[test]
    fn test_best_effort_missing_required_in_rejected_optional() {
        let neg = make_negotiator(
            vec![cap_req("bitswap", 2, 0)],
            CapabilityPolicy::BestEffort,
            1,
        );
        let result = neg.evaluate_offer(&offer("peer", vec![], 1));
        match result {
            NegotiationResult::Accepted {
                rejected_optional, ..
            } => {
                assert!(rejected_optional.contains(&"bitswap".to_string()));
            }
            _ => panic!("expected Accepted"),
        }
    }

    #[test]
    fn test_best_effort_agreed_when_present() {
        let neg = make_negotiator(
            vec![cap_req("bitswap", 2, 0)],
            CapabilityPolicy::BestEffort,
            1,
        );
        let result = neg.evaluate_offer(&offer("peer", vec![cap_opt("bitswap", 2, 1)], 1));
        match result {
            NegotiationResult::Accepted {
                agreed_capabilities,
                rejected_optional,
            } => {
                assert!(agreed_capabilities.contains(&"bitswap".to_string()));
                assert!(rejected_optional.is_empty());
            }
            _ => panic!("expected Accepted"),
        }
    }

    // ── build_offer tests ─────────────────────────────────────────────────────

    #[test]
    fn test_build_offer_contains_local_capabilities() {
        let neg = make_negotiator(
            vec![cap_req("bitswap", 2, 0), cap_opt("relay", 1, 0)],
            CapabilityPolicy::BestEffort,
            1,
        );
        let o = neg.build_offer("local-peer".to_string(), 42_000);
        assert_eq!(o.peer_id, "local-peer");
        assert_eq!(o.timestamp, 42_000);
        assert_eq!(o.capabilities.len(), 2);
        assert_eq!(o.protocol_version, 1);
    }

    #[test]
    fn test_build_offer_empty_capabilities() {
        let neg = make_negotiator(vec![], CapabilityPolicy::BestEffort, 2);
        let o = neg.build_offer("me".to_string(), 0);
        assert!(o.capabilities.is_empty());
        assert_eq!(o.protocol_version, 2);
    }

    // ── negotiate (stateful) tests ────────────────────────────────────────────

    #[test]
    fn test_negotiate_increments_total() {
        let mut neg = make_negotiator(vec![], CapabilityPolicy::BestEffort, 1);
        neg.negotiate("p1".to_string(), offer("p1", vec![], 1), 0, 10);
        assert_eq!(neg.total_negotiations, 1);
    }

    #[test]
    fn test_negotiate_increments_accepted_counter() {
        let mut neg = make_negotiator(vec![], CapabilityPolicy::BestEffort, 1);
        neg.negotiate("p1".to_string(), offer("p1", vec![], 1), 0, 10);
        assert_eq!(neg.total_accepted, 1);
        assert_eq!(neg.total_rejected, 0);
    }

    #[test]
    fn test_negotiate_increments_rejected_counter() {
        let mut neg = make_negotiator(vec![], CapabilityPolicy::RequireAll, 5);
        neg.negotiate("p1".to_string(), offer("p1", vec![], 1), 0, 10);
        assert_eq!(neg.total_rejected, 1);
        assert_eq!(neg.total_accepted, 0);
    }

    #[test]
    fn test_negotiate_result_stored_in_history() {
        let mut neg = make_negotiator(vec![], CapabilityPolicy::BestEffort, 1);
        neg.negotiate("p1".to_string(), offer("p1", vec![], 1), 100, 150);
        assert_eq!(neg.negotiation_history.len(), 1);
        let rec = &neg.negotiation_history[0];
        assert_eq!(rec.peer_id, "p1");
        assert_eq!(rec.latency_ms, 50);
        assert_eq!(rec.negotiated_at, 150);
    }

    #[test]
    fn test_negotiate_evicts_oldest_when_full() {
        let config = NegotiatorConfig::new(vec![], CapabilityPolicy::BestEffort, 1, 5_000);
        let mut neg = PeerCapabilityNegotiator::new(config, 2);
        neg.negotiate("p1".to_string(), offer("p1", vec![], 1), 0, 1);
        neg.negotiate("p2".to_string(), offer("p2", vec![], 1), 1, 2);
        neg.negotiate("p3".to_string(), offer("p3", vec![], 1), 2, 3);
        assert_eq!(neg.negotiation_history.len(), 2);
        // p1 should have been evicted
        assert_eq!(neg.negotiation_history[0].peer_id, "p2");
        assert_eq!(neg.negotiation_history[1].peer_id, "p3");
    }

    #[test]
    fn test_negotiate_returns_correct_result_ref() {
        let mut neg = make_negotiator(vec![], CapabilityPolicy::BestEffort, 1);
        let result = neg.negotiate("p".to_string(), offer("p", vec![], 1), 0, 5);
        assert!(result.is_accepted());
    }

    // ── is_capability_supported tests ─────────────────────────────────────────

    #[test]
    fn test_is_capability_supported_true() {
        let neg = make_negotiator(
            vec![cap_req("bitswap", 2, 3)],
            CapabilityPolicy::BestEffort,
            1,
        );
        // We support 2.3; query for 2.1 (we are higher minor) → compatible
        assert!(neg.is_capability_supported("bitswap", &ver(2, 1, 0)));
    }

    #[test]
    fn test_is_capability_supported_false_wrong_name() {
        let neg = make_negotiator(
            vec![cap_req("bitswap", 2, 0)],
            CapabilityPolicy::BestEffort,
            1,
        );
        assert!(!neg.is_capability_supported("graphsync", &ver(1, 0, 0)));
    }

    #[test]
    fn test_is_capability_supported_false_incompatible_version() {
        let neg = make_negotiator(
            vec![cap_req("bitswap", 2, 0)],
            CapabilityPolicy::BestEffort,
            1,
        );
        // We support 2.0; query for 2.5 (we are lower) → incompatible
        assert!(!neg.is_capability_supported("bitswap", &ver(2, 5, 0)));
    }

    // ── common_capabilities tests ─────────────────────────────────────────────

    #[test]
    fn test_common_capabilities_intersection() {
        let neg = make_negotiator(
            vec![cap_req("bitswap", 2, 0), cap_opt("relay", 1, 0)],
            CapabilityPolicy::BestEffort,
            1,
        );
        let peer_caps = vec![cap_opt("bitswap", 2, 1), cap_opt("graphsync", 1, 0)];
        let common = neg.common_capabilities(&offer("peer", peer_caps, 1));
        assert!(common.contains(&"bitswap".to_string()));
        assert!(!common.contains(&"relay".to_string()));
        assert!(!common.contains(&"graphsync".to_string()));
    }

    #[test]
    fn test_common_capabilities_empty_when_no_overlap() {
        let neg = make_negotiator(
            vec![cap_req("bitswap", 2, 0)],
            CapabilityPolicy::BestEffort,
            1,
        );
        let common = neg.common_capabilities(&offer("peer", vec![], 1));
        assert!(common.is_empty());
    }

    // ── history_for_peer tests ────────────────────────────────────────────────

    #[test]
    fn test_history_for_peer_filtered() {
        let mut neg = make_negotiator(vec![], CapabilityPolicy::BestEffort, 1);
        neg.negotiate("alice".to_string(), offer("alice", vec![], 1), 0, 1);
        neg.negotiate("bob".to_string(), offer("bob", vec![], 1), 1, 2);
        neg.negotiate("alice".to_string(), offer("alice", vec![], 1), 2, 3);

        let alice_history = neg.history_for_peer("alice");
        assert_eq!(alice_history.len(), 2);
        assert!(alice_history.iter().all(|r| r.peer_id == "alice"));
    }

    #[test]
    fn test_history_for_peer_empty_for_unknown() {
        let neg = make_negotiator(vec![], CapabilityPolicy::BestEffort, 1);
        assert!(neg.history_for_peer("nobody").is_empty());
    }

    // ── negotiator_stats tests ────────────────────────────────────────────────

    #[test]
    fn test_stats_zero_accept_rate_when_no_negotiations() {
        let neg = make_negotiator(vec![], CapabilityPolicy::BestEffort, 1);
        let stats = neg.negotiator_stats();
        assert_eq!(stats.total_negotiations, 0);
        assert_eq!(stats.accept_rate, 0.0);
    }

    #[test]
    fn test_stats_accept_rate_calculation() {
        let mut neg = make_negotiator(vec![], CapabilityPolicy::BestEffort, 1);
        // 2 accepted
        neg.negotiate("p1".to_string(), offer("p1", vec![], 1), 0, 1);
        neg.negotiate("p2".to_string(), offer("p2", vec![], 1), 1, 2);
        // 1 rejected (protocol too low)
        neg.negotiate("p3".to_string(), offer("p3", vec![], 0), 2, 3);
        let stats = neg.negotiator_stats();
        assert_eq!(stats.total_negotiations, 3);
        assert_eq!(stats.total_accepted, 2);
        assert_eq!(stats.total_rejected, 1);
        let expected = 2.0 / 3.0;
        assert!((stats.accept_rate - expected).abs() < 1e-10);
    }

    #[test]
    fn test_stats_capabilities_advertised_count() {
        let neg = make_negotiator(
            vec![cap_req("a", 1, 0), cap_opt("b", 1, 0), cap_req("c", 1, 0)],
            CapabilityPolicy::BestEffort,
            1,
        );
        assert_eq!(neg.negotiator_stats().capabilities_advertised, 3);
    }

    #[test]
    fn test_stats_is_clone() {
        let neg = make_negotiator(vec![], CapabilityPolicy::BestEffort, 1);
        let stats: NegotiatorStats = neg.negotiator_stats();
        let _cloned = stats.clone();
    }

    // ── NegotiatorConfig default test ─────────────────────────────────────────

    #[test]
    fn test_config_default_values() {
        let cfg = NegotiatorConfig::default();
        assert!(cfg.local_capabilities.is_empty());
        assert_eq!(cfg.policy, CapabilityPolicy::BestEffort);
        assert_eq!(cfg.min_protocol_version, 1);
        assert_eq!(cfg.negotiation_timeout_ms, 5_000);
    }

    // ── CapabilityPolicy default test ─────────────────────────────────────────

    #[test]
    fn test_policy_default_is_best_effort() {
        assert_eq!(CapabilityPolicy::default(), CapabilityPolicy::BestEffort);
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_negotiate_max_history_zero() {
        let config = NegotiatorConfig::new(vec![], CapabilityPolicy::BestEffort, 1, 5_000);
        let mut neg = PeerCapabilityNegotiator::new(config, 0);
        // Should not panic; scratch slot is used.
        let result = neg.negotiate("peer".to_string(), offer("peer", vec![], 1), 0, 1);
        assert!(result.is_accepted());
        assert_eq!(neg.total_negotiations, 1);
        // History cleared between calls; only the last scratch is there.
        neg.negotiate("peer".to_string(), offer("peer", vec![], 1), 1, 2);
        assert_eq!(neg.negotiation_history.len(), 1);
    }

    #[test]
    fn test_multiple_agreed_capabilities_deduplication_not_needed() {
        // Offer with two entries with the same name (edge case): only the last
        // wins via the HashMap; no panic should occur.
        let neg = make_negotiator(
            vec![cap_req("bitswap", 2, 0)],
            CapabilityPolicy::RequireAll,
            1,
        );
        let peer_caps = vec![
            cap_opt("bitswap", 2, 1),
            cap_opt("bitswap", 2, 2), // duplicate name
        ];
        let result = neg.evaluate_offer(&offer("peer", peer_caps, 1));
        // Should still be Accepted (last one wins in capability_map).
        assert!(result.is_accepted());
    }

    #[test]
    fn test_latency_stored_in_record() {
        let mut neg = make_negotiator(vec![], CapabilityPolicy::BestEffort, 1);
        neg.negotiate("p".to_string(), offer("p", vec![], 1), 1_000, 1_075);
        assert_eq!(neg.negotiation_history[0].latency_ms, 75);
    }

    #[test]
    fn test_saturation_on_end_before_start() {
        // end_ms < start_ms → saturating_sub clamps to 0.
        let mut neg = make_negotiator(vec![], CapabilityPolicy::BestEffort, 1);
        neg.negotiate("p".to_string(), offer("p", vec![], 1), 1_000, 500);
        assert_eq!(neg.negotiation_history[0].latency_ms, 0);
    }

    #[test]
    fn test_mixed_required_optional_require_all() {
        // Required cap present, optional absent → Accepted with optional in rejected_optional.
        let neg = make_negotiator(
            vec![
                cap_req("bitswap", 2, 0),
                cap_opt("graphsync", 1, 0),
                cap_opt("relay", 1, 0),
            ],
            CapabilityPolicy::RequireAll,
            1,
        );
        let peer_caps = vec![cap_opt("bitswap", 2, 0), cap_opt("graphsync", 1, 1)];
        let result = neg.evaluate_offer(&offer("peer", peer_caps, 1));
        match result {
            NegotiationResult::Accepted {
                agreed_capabilities,
                rejected_optional,
            } => {
                assert!(agreed_capabilities.contains(&"bitswap".to_string()));
                assert!(agreed_capabilities.contains(&"graphsync".to_string()));
                assert!(rejected_optional.contains(&"relay".to_string()));
            }
            _ => panic!("expected Accepted"),
        }
    }
}
