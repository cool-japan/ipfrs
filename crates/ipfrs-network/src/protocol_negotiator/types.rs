//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, HashSet};

use super::constants::PN_HISTORY_CAP;
use super::functions::{all_known_features, fnv1a_64, xorshift64};

/// Negotiates a mutually acceptable protocol configuration between two peers.
pub struct ProtocolNegotiator {
    /// Configuration for the local side of each negotiation.
    pub config: NegotiatorConfig,
}
impl ProtocolNegotiator {
    /// Create a new negotiator from the given configuration.
    pub fn new(config: NegotiatorConfig) -> Self {
        Self { config }
    }
    /// Attempt to agree on a protocol configuration.
    ///
    /// The algorithm:
    /// 1. Check that `[local.min_version, local.max_version]` and
    ///    `[remote.min_version, remote.max_version]` overlap.
    /// 2. Compute `agreed_version = min(local.max_version, remote.max_version)` —
    ///    the highest version both sides know.
    /// 3. Intersect the supported feature sets.
    /// 4. Verify every `required_feature` is present in the intersection.
    /// 5. If the intersection is empty, return [`NegotiationResult::NoFeaturesInCommon`].
    /// 6. Otherwise return [`NegotiationResult::Agreed`] with the smallest chunk size.
    pub fn negotiate(&self, local: &ProtocolOffer, remote: &ProtocolOffer) -> NegotiationResult {
        if local.max_version < remote.min_version || remote.max_version < local.min_version {
            return NegotiationResult::VersionMismatch {
                local_max: local.max_version,
                remote_min: remote.min_version,
            };
        }
        let version = local.max_version.min(remote.max_version);
        let remote_set: HashSet<ProtocolFeature> =
            remote.supported_features.iter().copied().collect();
        let intersection: Vec<ProtocolFeature> = local
            .supported_features
            .iter()
            .copied()
            .filter(|f| remote_set.contains(f))
            .collect();
        let intersection_set: HashSet<ProtocolFeature> = intersection.iter().copied().collect();
        for required in &self.config.required_features {
            if !intersection_set.contains(required) {
                return NegotiationResult::Rejected {
                    reason: format!("required feature not supported by remote: {:?}", required),
                };
            }
        }
        if intersection.is_empty() {
            return NegotiationResult::NoFeaturesInCommon;
        }
        let chunk_size = local.preferred_chunk_size.min(remote.preferred_chunk_size);
        NegotiationResult::Agreed {
            version,
            features: intersection,
            chunk_size,
        }
    }
    /// Returns `true` when [`negotiate`](Self::negotiate) would produce
    /// [`NegotiationResult::Agreed`], *ignoring* the `required_features` check.
    ///
    /// Useful as a quick pre-flight check before actually committing to a
    /// connection with a peer whose offer has been received out-of-band.
    pub fn can_negotiate(&self, remote: &ProtocolOffer) -> bool {
        let versions_overlap = self.config.local_max_version >= remote.min_version
            && remote.max_version >= self.config.local_min_version;
        if !versions_overlap {
            return false;
        }
        let all_features = all_known_features();
        let remote_set: HashSet<ProtocolFeature> =
            remote.supported_features.iter().copied().collect();
        all_features.iter().any(|f| remote_set.contains(f))
    }
    /// All protocol versions supported by the local peer.
    pub fn supported_versions(&self) -> Vec<u32> {
        (self.config.local_min_version..=self.config.local_max_version).collect()
    }
}
/// A named protocol with a major/minor version pair, used by
/// [`PeerProtocolNegotiator`] for per-protocol version negotiation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PeerProtocolVersion {
    /// Human-readable protocol name (e.g. `"bitswap"`, `"dht"`).
    pub name: String,
    /// Major version – breaking changes increment this.
    pub major: u32,
    /// Minor version – backwards-compatible additions increment this.
    pub minor: u32,
}
impl PeerProtocolVersion {
    /// Create a new protocol version descriptor.
    pub fn new(name: impl Into<String>, major: u32, minor: u32) -> Self {
        Self {
            name: name.into(),
            major,
            minor,
        }
    }
}
/// Historical record of a negotiation attempt.
#[derive(Clone, Debug)]
pub struct PnNegotiationRecord {
    /// Unix-epoch milliseconds.
    pub ts: u64,
    /// Remote peer identity.
    pub peer_id: [u8; 32],
    /// Protocol for which negotiation was attempted.
    pub protocol_id: PnProtocolId,
    /// How the negotiation ended.
    pub outcome: PnNegotiationOutcome,
}
/// Semantic version for a protocol, used by [`PnProtocolNegotiator`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PnProtocolVersion {
    /// Incompatible changes increment this.
    pub major: u32,
    /// Backwards-compatible additions increment this.
    pub minor: u32,
    /// Bug fixes only.
    pub patch: u32,
}
impl PnProtocolVersion {
    /// Create a new version triple.
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }
    /// Returns `true` when `self` is compatible with `other`:
    /// major versions must match and `self.minor >= other.minor`.
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.major == other.major && self.minor >= other.minor
    }
}
/// Outcome of a [`ProtocolNegotiator::negotiate`] call.
#[derive(Clone, Debug, PartialEq)]
pub enum NegotiationResult {
    /// Both peers agreed on a compatible configuration.
    Agreed {
        /// Highest mutually supported protocol version.
        version: u32,
        /// Features present in both offers.
        features: Vec<ProtocolFeature>,
        /// Minimum of both preferred chunk sizes.
        chunk_size: u64,
    },
    /// The version ranges did not overlap.
    VersionMismatch {
        /// Highest version the local peer supports.
        local_max: u32,
        /// Lowest version the remote peer requires.
        remote_min: u32,
    },
    /// Version ranges overlap but no features are shared.
    NoFeaturesInCommon,
    /// A required feature was absent from the intersection.
    Rejected {
        /// Human-readable explanation, including the missing feature name.
        reason: String,
    },
}
/// An active, fully negotiated session between two peers.
#[derive(Clone, Debug)]
pub struct PnNegotiatedSession {
    /// Unique session identifier.
    pub id: PnSessionId,
    /// Remote peer identity (32-byte key).
    pub peer_id: [u8; 32],
    /// The protocol used for this session.
    pub protocol_id: PnProtocolId,
    /// The version both peers agreed on.
    pub agreed_version: PnProtocolVersion,
    /// Unix-epoch milliseconds when the session was established.
    pub established_at: u64,
    /// Unix-epoch milliseconds of the most recent activity.
    pub last_activity: u64,
    /// Total bytes exchanged over the session lifetime.
    pub bytes_exchanged: u64,
}
/// Configuration for the local side of a protocol negotiation.
#[derive(Clone, Debug)]
pub struct NegotiatorConfig {
    /// Lowest protocol version this peer will accept.
    pub local_min_version: u32,
    /// Highest protocol version this peer supports.
    pub local_max_version: u32,
    /// Features that *must* appear in the negotiated set; absence → [`NegotiationResult::Rejected`].
    pub required_features: Vec<ProtocolFeature>,
    /// Preferred transfer chunk size in bytes.
    pub local_chunk_size: u64,
}
impl NegotiatorConfig {
    /// Build a [`ProtocolOffer`] that represents the local peer's capabilities.
    ///
    /// `peer_id`  – identifier to embed in the offer.
    /// `features` – feature flags that the local peer supports.
    pub fn local_offer(&self, peer_id: String, features: Vec<ProtocolFeature>) -> ProtocolOffer {
        ProtocolOffer {
            peer_id,
            min_version: self.local_min_version,
            max_version: self.local_max_version,
            supported_features: features,
            preferred_chunk_size: self.local_chunk_size,
        }
    }
}
/// Cumulative statistics tracked by [`PeerProtocolNegotiator`].
#[derive(Debug, Clone)]
pub struct PeerNegotiatorStats {
    /// Number of currently registered protocol versions.
    pub supported_count: usize,
    /// Total calls to [`PeerProtocolNegotiator::negotiate`].
    pub negotiations: u64,
    /// How many resulted in [`PeerNegotiationResult::Accepted`].
    pub accepted: u64,
    /// How many resulted in [`PeerNegotiationResult::Rejected`].
    pub rejected: u64,
    /// How many resulted in [`PeerNegotiationResult::Downgraded`].
    pub downgraded: u64,
}
/// Outcome of a single negotiation attempt.
#[derive(Clone, Debug, PartialEq)]
pub enum PnNegotiationOutcome {
    /// Handshake succeeded; both peers agreed on `version`.
    Success {
        /// The agreed-upon protocol version.
        version: PnProtocolVersion,
    },
    /// The versions offered were not compatible.
    VersionMismatch {
        /// Versions the initiator offered.
        offered: Vec<PnProtocolVersion>,
        /// The version that the responder would have accepted (if any).
        accepted: Option<PnProtocolVersion>,
    },
    /// The requested protocol is not known to this node.
    ProtocolUnknown,
    /// The remote peer explicitly rejected the negotiation.
    Rejected,
    /// The handshake was not completed within the allowed time window.
    Timeout,
}
/// Outcome of a single [`PeerProtocolNegotiator::negotiate`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerNegotiationResult {
    /// The requested version is an exact match for a supported version.
    Accepted,
    /// The protocol name is unknown or the version is below the configured minimum.
    Rejected,
    /// The requested version has the same name and major version but a different
    /// minor; the negotiator proposes its own version instead.
    Downgraded,
}
/// Production-quality protocol-version negotiation with session lifecycle management.
pub struct PnProtocolNegotiator {
    /// Local protocol capabilities: id → acceptable versions.
    pub(super) supported: HashMap<PnProtocolId, Vec<PnProtocolVersion>>,
    /// Active sessions indexed by session id.
    pub(super) sessions: HashMap<PnSessionId, PnNegotiatedSession>,
    /// Bounded negotiation history (max 500 entries).
    pub(super) history: std::collections::VecDeque<PnNegotiationRecord>,
    /// Configuration.
    pub(super) config: PnNegotiatorConfig,
    /// PRNG state for session ID generation.
    pub(super) rng_state: u64,
    /// Monotonic millisecond clock (set externally or via `now_ms()`).
    pub(super) clock_offset: u64,
    /// Lifetime counters.
    pub(super) total_negotiations: u64,
    pub(super) total_successes: u64,
    /// Per-protocol success counters.
    pub(super) proto_successes: HashMap<PnProtocolId, u64>,
}
impl PnProtocolNegotiator {
    /// Create a new negotiator with the given configuration.
    pub fn new(config: PnNegotiatorConfig) -> Self {
        let seed = fnv1a_64(&config.max_sessions.to_le_bytes())
            ^ fnv1a_64(&config.session_ttl_secs.to_le_bytes())
            ^ 0x517cc1b727220a95;
        Self {
            supported: HashMap::new(),
            sessions: HashMap::new(),
            history: std::collections::VecDeque::with_capacity(PN_HISTORY_CAP),
            config,
            rng_state: seed | 1,
            clock_offset: 0,
            total_negotiations: 0,
            total_successes: 0,
            proto_successes: HashMap::new(),
        }
    }
    pub(super) fn now_ms(&mut self) -> u64 {
        self.clock_offset = self.clock_offset.wrapping_add(1);
        self.clock_offset
    }
    pub(super) fn next_session_id(
        &mut self,
        peer_id: &[u8; 32],
        protocol_id: &PnProtocolId,
    ) -> PnSessionId {
        let mut mix = fnv1a_64(peer_id);
        mix ^= fnv1a_64(&protocol_id.0);
        mix ^= xorshift64(&mut self.rng_state);
        PnSessionId(mix)
    }
    pub(super) fn push_history(&mut self, record: PnNegotiationRecord) {
        if self.history.len() >= PN_HISTORY_CAP {
            self.history.pop_front();
        }
        self.history.push_back(record);
    }
    pub(super) fn pick_version(
        &self,
        local_versions: &[PnProtocolVersion],
        peer_versions: &[PnProtocolVersion],
    ) -> Option<PnProtocolVersion> {
        let mut candidates: Vec<PnProtocolVersion> = local_versions
            .iter()
            .copied()
            .filter(|lv| {
                peer_versions.iter().any(|pv| {
                    if self.config.strict_compat {
                        lv.major == pv.major && lv.minor == pv.minor
                    } else {
                        lv.is_compatible_with(pv) || pv.is_compatible_with(lv)
                    }
                })
            })
            .collect();
        if candidates.is_empty() {
            return None;
        }
        if self.config.prefer_latest {
            candidates.sort_unstable_by(|a, b| b.cmp(a));
        } else {
            candidates.sort_unstable();
        }
        candidates.into_iter().next()
    }
    /// Register or replace the list of acceptable versions for a protocol.
    pub fn register_protocol(&mut self, id: PnProtocolId, versions: Vec<PnProtocolVersion>) {
        self.supported.insert(id, versions);
    }
    /// Remove a protocol from the local registry.  Returns `true` if it existed.
    pub fn deregister_protocol(&mut self, id: &PnProtocolId) -> bool {
        self.supported.remove(id).is_some()
    }
    /// List all registered protocol ids.
    pub fn supported_protocols(&self) -> Vec<PnProtocolId> {
        self.supported.keys().copied().collect()
    }
    /// Return the registered versions for a protocol, if any.
    pub fn versions_for(&self, id: &PnProtocolId) -> Option<&[PnProtocolVersion]> {
        self.supported.get(id).map(|v| v.as_slice())
    }
    /// Initiate a handshake: we offer `offered` versions and pick the best mutual match.
    pub fn initiate_handshake(
        &mut self,
        peer_id: [u8; 32],
        protocol_id: PnProtocolId,
        offered: Vec<PnProtocolVersion>,
    ) -> PnNegotiationOutcome {
        self.total_negotiations += 1;
        let ts = self.now_ms();
        let local_versions = match self.supported.get(&protocol_id) {
            Some(v) => v.clone(),
            None => {
                let outcome = PnNegotiationOutcome::ProtocolUnknown;
                self.push_history(PnNegotiationRecord {
                    ts,
                    peer_id,
                    protocol_id,
                    outcome: outcome.clone(),
                });
                return outcome;
            }
        };
        let outcome = match self.pick_version(&local_versions, &offered) {
            Some(version) => {
                if self.sessions.len() < self.config.max_sessions {
                    let sid = self.next_session_id(&peer_id, &protocol_id);
                    self.sessions.insert(
                        sid,
                        PnNegotiatedSession {
                            id: sid,
                            peer_id,
                            protocol_id,
                            agreed_version: version,
                            established_at: ts,
                            last_activity: ts,
                            bytes_exchanged: 0,
                        },
                    );
                }
                self.total_successes += 1;
                *self.proto_successes.entry(protocol_id).or_insert(0) += 1;
                PnNegotiationOutcome::Success { version }
            }
            None => PnNegotiationOutcome::VersionMismatch {
                offered,
                accepted: local_versions.first().copied(),
            },
        };
        self.push_history(PnNegotiationRecord {
            ts,
            peer_id,
            protocol_id,
            outcome: outcome.clone(),
        });
        outcome
    }
    /// Respond to an incoming handshake by selecting the best mutual match.
    pub fn respond_to_handshake(
        &mut self,
        peer_id: [u8; 32],
        protocol_id: PnProtocolId,
        peer_offers: Vec<PnProtocolVersion>,
    ) -> PnNegotiationOutcome {
        self.total_negotiations += 1;
        let ts = self.now_ms();
        let local_versions = match self.supported.get(&protocol_id) {
            Some(v) => v.clone(),
            None => {
                let outcome = PnNegotiationOutcome::ProtocolUnknown;
                self.push_history(PnNegotiationRecord {
                    ts,
                    peer_id,
                    protocol_id,
                    outcome: outcome.clone(),
                });
                return outcome;
            }
        };
        let outcome = match self.pick_version(&local_versions, &peer_offers) {
            Some(version) => {
                if self.sessions.len() < self.config.max_sessions {
                    let sid = self.next_session_id(&peer_id, &protocol_id);
                    self.sessions.insert(
                        sid,
                        PnNegotiatedSession {
                            id: sid,
                            peer_id,
                            protocol_id,
                            agreed_version: version,
                            established_at: ts,
                            last_activity: ts,
                            bytes_exchanged: 0,
                        },
                    );
                }
                self.total_successes += 1;
                *self.proto_successes.entry(protocol_id).or_insert(0) += 1;
                PnNegotiationOutcome::Success { version }
            }
            None => PnNegotiationOutcome::VersionMismatch {
                offered: peer_offers,
                accepted: local_versions.first().copied(),
            },
        };
        self.push_history(PnNegotiationRecord {
            ts,
            peer_id,
            protocol_id,
            outcome: outcome.clone(),
        });
        outcome
    }
    /// Record activity on a session; returns `false` if not found.
    pub fn update_session_activity(&mut self, session_id: PnSessionId, bytes: u64) -> bool {
        if let Some(s) = self.sessions.get_mut(&session_id) {
            let now = self.clock_offset.wrapping_add(1);
            self.clock_offset = now;
            s.last_activity = now;
            s.bytes_exchanged = s.bytes_exchanged.saturating_add(bytes);
            true
        } else {
            false
        }
    }
    /// Remove sessions inactive longer than the configured TTL.
    pub fn expire_sessions(&mut self) {
        let now = self.clock_offset;
        let ttl = self.config.session_ttl_secs;
        self.sessions
            .retain(|_, s| now.saturating_sub(s.last_activity) < ttl);
    }
    /// Return active sessions for a peer.
    pub fn sessions_for_peer(&self, peer_id: [u8; 32]) -> Vec<&PnNegotiatedSession> {
        self.sessions
            .values()
            .filter(|s| s.peer_id == peer_id)
            .collect()
    }
    /// Look up a session by id.
    pub fn get_session(&self, session_id: &PnSessionId) -> Option<&PnNegotiatedSession> {
        self.sessions.get(session_id)
    }
    /// Return all active session ids.
    pub fn active_session_ids(&self) -> Vec<PnSessionId> {
        self.sessions.keys().copied().collect()
    }
    /// Number of currently active sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }
    /// Snapshot of cumulative negotiation statistics.
    pub fn negotiation_stats(&self) -> PnNegotiationStats {
        let success_rate = if self.total_negotiations == 0 {
            0.0
        } else {
            self.total_successes as f64 / self.total_negotiations as f64
        };
        PnNegotiationStats {
            total: self.total_negotiations,
            success_rate,
            by_protocol: self.proto_successes.clone(),
        }
    }
    /// Negotiation history (oldest-first, bounded to 500 entries).
    pub fn history(&self) -> &std::collections::VecDeque<PnNegotiationRecord> {
        &self.history
    }
    /// Number of history records.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
    /// Access the current configuration.
    pub fn config(&self) -> &PnNegotiatorConfig {
        &self.config
    }
}
/// Unique protocol identifier (16-byte opaque key).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PnProtocolId(pub [u8; 16]);
/// Aggregate statistics returned by [`PnProtocolNegotiator::negotiation_stats`].
#[derive(Clone, Debug, Default)]
pub struct PnNegotiationStats {
    /// Total negotiation attempts (all outcomes).
    pub total: u64,
    /// Fraction of attempts that succeeded (0.0–1.0).
    pub success_rate: f64,
    /// Per-protocol breakdown: protocol id → success count.
    pub by_protocol: HashMap<PnProtocolId, u64>,
}
/// Feature flags that peers may support and negotiate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProtocolFeature {
    /// Block/stream compression (e.g. LZ4, Zstd).
    Compression,
    /// End-to-end encryption of the data channel.
    Encryption,
    /// Multiple logical streams over a single physical connection.
    Multiplexing,
    /// Priority-aware message scheduling.
    PriorityQueuing,
    /// Back-pressure / flow-control signalling.
    FlowControl,
    /// Apache Arrow IPC serialisation for structured data.
    ArrowIpc,
}
/// Capabilities advertised by one side of a connection.
#[derive(Clone, Debug, PartialEq)]
pub struct ProtocolOffer {
    /// Identifier of the peer making this offer.
    pub peer_id: String,
    /// Lowest protocol version the peer will accept.
    pub min_version: u32,
    /// Highest protocol version the peer supports.
    pub max_version: u32,
    /// Feature flags the peer has implemented.
    pub supported_features: Vec<ProtocolFeature>,
    /// Preferred transfer chunk size in bytes.
    pub preferred_chunk_size: u64,
}
/// Configuration for [`PnProtocolNegotiator`].
#[derive(Clone, Debug)]
pub struct PnNegotiatorConfig {
    /// Maximum number of concurrent negotiated sessions.
    pub max_sessions: usize,
    /// Seconds after last activity before a session is considered expired.
    pub session_ttl_secs: u64,
    /// When multiple versions are compatible, prefer the latest.
    pub prefer_latest: bool,
    /// Reject any offer that is not an exact major+minor match (no patch flexibility).
    pub strict_compat: bool,
}
/// Negotiates protocol versions between peers.
///
/// Each registered protocol is identified by its `name` string.  When a remote
/// peer requests a particular [`PeerProtocolVersion`] the negotiator checks:
///
/// 1. **Exact match** – same name, major, and minor → [`PeerNegotiationResult::Accepted`].
/// 2. **Downgrade** – same name, same major, but the requested minor differs;
///    the negotiator responds with its own version →
///    [`PeerNegotiationResult::Downgraded`].
/// 3. **Rejected** – the protocol name is unknown, or the major version differs,
///    or the version is below the configured minimum.
pub struct PeerProtocolNegotiator {
    /// Registered (supported) protocol versions keyed by protocol name.
    pub(super) supported: Vec<PeerProtocolVersion>,
    /// Per-protocol minimum acceptable version `(min_major, min_minor)`.
    pub(super) min_versions: HashMap<String, (u32, u32)>,
    /// Lifetime negotiation counters.
    pub(super) negotiations: u64,
    pub(super) accepted: u64,
    pub(super) rejected: u64,
    pub(super) downgraded: u64,
}
impl PeerProtocolNegotiator {
    /// Create a new, empty negotiator with no registered protocols.
    pub fn new() -> Self {
        Self {
            supported: Vec::new(),
            min_versions: HashMap::new(),
            negotiations: 0,
            accepted: 0,
            rejected: 0,
            downgraded: 0,
        }
    }
    /// Register a supported protocol version.
    ///
    /// If a protocol with the same `name` already exists it is replaced.
    pub fn register_protocol(&mut self, name: &str, major: u32, minor: u32) {
        self.supported.retain(|p| p.name != name);
        self.supported
            .push(PeerProtocolVersion::new(name, major, minor));
    }
    /// Set (or update) the minimum acceptable version for the given protocol.
    ///
    /// During [`negotiate`](Self::negotiate), any requested version whose major
    /// is below `min_major`, or whose major equals `min_major` but minor is below
    /// `min_minor`, will be rejected outright.
    pub fn set_minimum(&mut self, name: &str, min_major: u32, min_minor: u32) {
        self.min_versions
            .insert(name.to_string(), (min_major, min_minor));
    }
    /// Negotiate a requested protocol version.
    ///
    /// Returns `(result, option_version)`:
    /// - `Accepted`   → `Some(requested)` (echo back the exact version).
    /// - `Downgraded` → `Some(our_version)` (propose our version instead).
    /// - `Rejected`   → `None`.
    pub fn negotiate(
        &mut self,
        requested: &PeerProtocolVersion,
    ) -> (PeerNegotiationResult, Option<PeerProtocolVersion>) {
        self.negotiations += 1;
        let ours = match self.supported.iter().find(|p| p.name == requested.name) {
            Some(v) => v.clone(),
            None => {
                self.rejected += 1;
                return (PeerNegotiationResult::Rejected, None);
            }
        };
        if let Some(&(min_maj, min_min)) = self.min_versions.get(&requested.name) {
            if requested.major < min_maj
                || (requested.major == min_maj && requested.minor < min_min)
            {
                self.rejected += 1;
                return (PeerNegotiationResult::Rejected, None);
            }
        }
        if requested.major != ours.major {
            self.rejected += 1;
            return (PeerNegotiationResult::Rejected, None);
        }
        if requested.minor == ours.minor {
            self.accepted += 1;
            return (PeerNegotiationResult::Accepted, Some(requested.clone()));
        }
        self.downgraded += 1;
        (PeerNegotiationResult::Downgraded, Some(ours))
    }
    /// Check whether a protocol with the given name is registered.
    pub fn is_supported(&self, name: &str) -> bool {
        self.supported.iter().any(|p| p.name == name)
    }
    /// Return a reference to the registered version for the given protocol, if
    /// any.
    pub fn get_version(&self, name: &str) -> Option<&PeerProtocolVersion> {
        self.supported.iter().find(|p| p.name == name)
    }
    /// Return references to all currently registered protocol versions.
    pub fn supported_protocols(&self) -> Vec<&PeerProtocolVersion> {
        self.supported.iter().collect()
    }
    /// Remove the protocol with the given name.
    ///
    /// Returns `true` if a protocol was actually removed, `false` if the name
    /// was not registered.
    pub fn remove_protocol(&mut self, name: &str) -> bool {
        let before = self.supported.len();
        self.supported.retain(|p| p.name != name);
        self.min_versions.remove(name);
        self.supported.len() < before
    }
    /// Snapshot of cumulative negotiation statistics.
    pub fn stats(&self) -> PeerNegotiatorStats {
        PeerNegotiatorStats {
            supported_count: self.supported.len(),
            negotiations: self.negotiations,
            accepted: self.accepted,
            rejected: self.rejected,
            downgraded: self.downgraded,
        }
    }
}
/// Session identifier derived from peer + protocol + timestamp.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PnSessionId(pub u64);
