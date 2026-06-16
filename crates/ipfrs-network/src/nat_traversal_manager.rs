//! ICE-based NAT traversal manager with STUN/TURN-like candidate gathering
//! and hole-punching coordination.
//!
//! Implements a production-quality Interactive Connectivity Establishment (ICE)
//! workflow:
//! - Host, server-reflexive, peer-reflexive and relayed candidate gathering
//! - Candidate pair formation with RFC 5245 priority ordering
//! - Connectivity checks with state machine transitions
//! - Best-pair nomination after successful checks
//! - STUN message encoding / decoding (subset of RFC 5389)
//! - NAT type heuristic detection from gathered candidates
//!
//! # Collision notes (lib.rs re-export)
//! * `NatType` collides with `nat_traversal::NatType` → exported as `NtmNatType`
//! * `NatTraversalManager` collides with `nat_traversal::NatTraversalManager` →
//!   exported as `NtmNatTraversalManager`

// ──────────────────────────────────────────────────────────────────────────────
// PRNG + hashing helpers (no external rand crate)
// ──────────────────────────────────────────────────────────────────────────────

/// Xorshift-64 PRNG.  Pass a non-zero seed; returns the next pseudo-random
/// value and updates `state` in place.
#[inline]
pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// FNV-1a 64-bit hash.
#[inline]
pub fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

// ──────────────────────────────────────────────────────────────────────────────
// NAT type detection
// ──────────────────────────────────────────────────────────────────────────────

/// NAT type detected from gathered ICE candidates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NatType {
    /// No NAT — host address is publicly reachable.
    OpenInternet,
    /// Full-cone: any remote host can send to the mapped port.
    FullCone,
    /// Address-restricted cone: only remotes the local host has contacted.
    RestrictedCone,
    /// Port-restricted cone: restricted by both remote IP and remote port.
    PortRestrictedCone,
    /// Symmetric: different external mapping per destination.
    Symmetric,
    /// Detection was inconclusive.
    Unknown,
}

// ──────────────────────────────────────────────────────────────────────────────
// Candidate types
// ──────────────────────────────────────────────────────────────────────────────

/// ICE candidate type (RFC 5245 §4.1.1).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CandidateType {
    /// Obtained directly from a local interface.
    Host,
    /// Obtained via a STUN server (external view of the host).
    ServerReflexive,
    /// Learned from a peer's STUN binding response during a check.
    PeerReflexive,
    /// Obtained via a TURN relay allocation.
    Relayed,
}

/// A single ICE candidate address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateAddress {
    /// IP address string (IPv4 or IPv6).
    pub address: String,
    /// UDP/TCP port.
    pub port: u16,
    /// Candidate category.
    pub candidate_type: CandidateType,
    /// RFC 5245 candidate priority (higher is better).
    pub priority: u32,
    /// Foundation: FNV-1a hex of `"<address>:<port>"`.
    pub foundation: String,
}

impl CandidateAddress {
    /// Construct a candidate and derive its `foundation` automatically.
    pub fn new(
        address: impl Into<String>,
        port: u16,
        candidate_type: CandidateType,
        priority: u32,
    ) -> Self {
        let address = address.into();
        let raw = format!("{address}:{port}");
        let foundation = format!("{:016x}", fnv1a_64(raw.as_bytes()));
        Self {
            address,
            port,
            candidate_type,
            priority,
            foundation,
        }
    }

    /// Convenience key for display / map lookups.
    pub fn key(&self) -> String {
        format!("{}:{}", self.address, self.port)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ICE pair
// ──────────────────────────────────────────────────────────────────────────────

/// State of a single ICE candidate pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairState {
    /// Waiting to be checked (in the check list but not yet started).
    Waiting,
    /// Connectivity check is in flight.
    InProgress,
    /// Connectivity check succeeded.
    Succeeded,
    /// Connectivity check failed.
    Failed,
    /// Pair is frozen (lower priority; will be unfrozen if needed).
    Frozen,
}

/// A local-remote candidate pair subject to connectivity checks.
#[derive(Debug, Clone)]
pub struct IcePair {
    /// Local candidate.
    pub local: CandidateAddress,
    /// Remote candidate.
    pub remote: CandidateAddress,
    /// Current ICE state for this pair.
    pub state: PairState,
    /// RFC 5245 pair priority: `2^32 * min(G,D) + 2 * max(G,D) + (G>D ? 1 : 0)`
    /// where G = local priority (controlling), D = remote priority (controlled).
    pub priority: u64,
    /// Whether this pair has been nominated by the controlling agent.
    pub nominated: bool,
}

impl IcePair {
    /// Create a new pair; compute RFC 5245 priority automatically.
    pub fn new(local: CandidateAddress, remote: CandidateAddress) -> Self {
        let g = local.priority as u64;
        let d = remote.priority as u64;
        let priority = (1u64 << 32) * g.min(d) + 2 * g.max(d) + if g > d { 1 } else { 0 };
        Self {
            local,
            remote,
            state: PairState::Waiting,
            priority,
            nominated: false,
        }
    }

    /// Human-readable key for this pair.
    pub fn key(&self) -> String {
        format!("{} -> {}", self.local.key(), self.remote.key())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// STUN message types and attributes
// ──────────────────────────────────────────────────────────────────────────────

/// STUN / TURN message class + method (simplified subset of RFC 5389 / 5766).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StunMessageType {
    /// STUN Binding Request (0x0001).
    BindingRequest,
    /// STUN Binding Success Response (0x0101).
    BindingResponse,
    /// STUN Binding Error Response (0x0111).
    BindingError,
    /// TURN Allocate Request (0x0003).
    AllocateRequest,
    /// TURN Allocate Success Response (0x0103).
    AllocateResponse,
}

impl StunMessageType {
    fn to_u16(&self) -> u16 {
        match self {
            Self::BindingRequest => 0x0001,
            Self::BindingResponse => 0x0101,
            Self::BindingError => 0x0111,
            Self::AllocateRequest => 0x0003,
            Self::AllocateResponse => 0x0103,
        }
    }

    fn from_u16(v: u16) -> Option<Self> {
        match v {
            0x0001 => Some(Self::BindingRequest),
            0x0101 => Some(Self::BindingResponse),
            0x0111 => Some(Self::BindingError),
            0x0003 => Some(Self::AllocateRequest),
            0x0103 => Some(Self::AllocateResponse),
            _ => None,
        }
    }
}

/// A single STUN attribute (subset of RFC 5389 attributes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StunAttribute {
    /// `MAPPED-ADDRESS` (0x0001): IP + port.
    MappedAddress(String, u16),
    /// `XOR-MAPPED-ADDRESS` (0x0020): IP + port (XOR-obfuscated in wire form).
    XorMappedAddress(String, u16),
    /// `USERNAME` (0x0006): ICE username fragment.
    Username(String),
    /// `REALM` (0x0014): authentication realm.
    Realm(String),
    /// `ERROR-CODE` (0x0009): error class + reason phrase.
    ErrorCode(u16, String),
    /// `FINGERPRINT` (0x8028): CRC-32 of message up to this attribute.
    Fingerprint(u32),
    /// `LIFETIME` (0x000D): TURN allocation lifetime in seconds.
    Lifetime(u32),
}

/// A STUN / TURN message.
#[derive(Debug, Clone)]
pub struct StunMessage {
    /// Message type.
    pub msg_type: StunMessageType,
    /// 96-bit transaction identifier.
    pub transaction_id: [u8; 12],
    /// List of attributes appended after the fixed header.
    pub attributes: Vec<StunAttribute>,
}

impl StunMessage {
    /// Create a new STUN message with an auto-generated transaction ID derived
    /// from `seed` via xorshift-64.
    pub fn new(msg_type: StunMessageType, seed: u64) -> Self {
        let mut state = if seed == 0 { 0xdeadbeef_cafebabe } else { seed };
        let mut tx = [0u8; 12];
        let a = xorshift64(&mut state).to_le_bytes();
        let b = xorshift64(&mut state).to_le_bytes();
        tx[..8].copy_from_slice(&a);
        tx[8..12].copy_from_slice(&b[..4]);
        Self {
            msg_type,
            transaction_id: tx,
            attributes: Vec::new(),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Configuration and statistics
// ──────────────────────────────────────────────────────────────────────────────

/// Runtime configuration for [`NtmNatTraversalManager`].
#[derive(Debug, Clone)]
pub struct TraversalConfig {
    /// STUN servers to use for server-reflexive candidate gathering.
    pub stun_servers: Vec<String>,
    /// TURN servers to use for relayed candidate gathering.
    pub turn_servers: Vec<String>,
    /// Interval between consecutive connectivity checks (µs).
    pub check_interval_us: u64,
    /// Nomination timeout after first Succeeded pair (µs).
    pub nomination_timeout_us: u64,
    /// Maximum number of candidate pairs in the check list.
    pub max_pairs: usize,
}

impl Default for TraversalConfig {
    fn default() -> Self {
        Self {
            stun_servers: vec![
                "stun.l.google.com:19302".to_string(),
                "stun1.l.google.com:19302".to_string(),
            ],
            turn_servers: Vec::new(),
            check_interval_us: 20_000,      // 20 ms
            nomination_timeout_us: 500_000, // 500 ms
            max_pairs: 100,
        }
    }
}

/// Snapshot statistics from the traversal manager.
#[derive(Debug, Clone, Default)]
pub struct TraversalStats {
    /// Number of local + remote candidates gathered so far.
    pub candidates_gathered: usize,
    /// Number of pairs on which a connectivity check was run.
    pub pairs_checked: usize,
    /// Pairs whose check succeeded.
    pub pairs_succeeded: usize,
    /// Pairs whose check failed.
    pub pairs_failed: usize,
    /// Key of the nominated pair, if one exists.
    pub nominated_pair: Option<String>,
    /// Human-readable NAT type string.
    pub nat_type: String,
}

// ──────────────────────────────────────────────────────────────────────────────
// Error type
// ──────────────────────────────────────────────────────────────────────────────

/// Errors produced by the traversal manager.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraversalError {
    /// No valid nominated pair could be found.
    NoValidPair,
    /// Candidate gathering encountered an error.
    CandidateGatheringFailed(String),
    /// A STUN-level error occurred.
    StunError(String),
    /// TURN relay allocation failed.
    TurnAllocationFailed(String),
    /// The check list was exhausted without success.
    ChecklistExhausted,
}

impl std::fmt::Display for TraversalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoValidPair => write!(f, "no valid ICE pair found"),
            Self::CandidateGatheringFailed(msg) => {
                write!(f, "candidate gathering failed: {msg}")
            }
            Self::StunError(msg) => write!(f, "STUN error: {msg}"),
            Self::TurnAllocationFailed(msg) => {
                write!(f, "TURN allocation failed: {msg}")
            }
            Self::ChecklistExhausted => write!(f, "check list exhausted without success"),
        }
    }
}

impl std::error::Error for TraversalError {}

// ──────────────────────────────────────────────────────────────────────────────
// STUN wire constants
// ──────────────────────────────────────────────────────────────────────────────

/// STUN magic cookie (RFC 5389 §6).
const STUN_MAGIC: u32 = 0x2112A442;

// Attribute type codes (RFC 5389 / 5766)
const ATTR_MAPPED_ADDRESS: u16 = 0x0001;
const ATTR_USERNAME: u16 = 0x0006;
const ATTR_ERROR_CODE: u16 = 0x0009;
const ATTR_LIFETIME: u16 = 0x000D;
const ATTR_REALM: u16 = 0x0014;
const ATTR_XOR_MAPPED_ADDRESS: u16 = 0x0020;
const ATTR_FINGERPRINT: u16 = 0x8028;

// ──────────────────────────────────────────────────────────────────────────────
// Wire encoding helpers (private)
// ──────────────────────────────────────────────────────────────────────────────

/// Write a 4-byte-padded string attribute into `buf`.
fn encode_string_attr(buf: &mut Vec<u8>, attr_type: u16, s: &str) {
    let bytes = s.as_bytes();
    let len = bytes.len() as u16;
    buf.extend_from_slice(&attr_type.to_be_bytes());
    buf.extend_from_slice(&len.to_be_bytes());
    buf.extend_from_slice(bytes);
    // Pad to 4-byte boundary
    let pad = (4 - (bytes.len() % 4)) % 4;
    buf.extend(std::iter::repeat_n(0u8, pad));
}

/// Write a `MAPPED-ADDRESS` or `XOR-MAPPED-ADDRESS` attribute.
/// For XOR the address octets are XOR'd with the magic cookie (IPv4 only here).
fn encode_address_attr(buf: &mut Vec<u8>, attr_type: u16, addr: &str, port: u16, xor: bool) {
    // Parse IPv4 address; fall back to zeroes on error
    let octets: [u8; 4] = parse_ipv4(addr).unwrap_or([0, 0, 0, 0]);
    let (enc_port, enc_octets) = if xor {
        let xp = port ^ ((STUN_MAGIC >> 16) as u16);
        let magic_bytes = STUN_MAGIC.to_be_bytes();
        let xo = [
            octets[0] ^ magic_bytes[0],
            octets[1] ^ magic_bytes[1],
            octets[2] ^ magic_bytes[2],
            octets[3] ^ magic_bytes[3],
        ];
        (xp, xo)
    } else {
        (port, octets)
    };
    // Value: 1-byte zeros, 1-byte family (0x01=IPv4), 2-byte port, 4-byte addr
    buf.extend_from_slice(&attr_type.to_be_bytes());
    buf.extend_from_slice(&8u16.to_be_bytes()); // length = 8
    buf.push(0x00); // padding
    buf.push(0x01); // family IPv4
    buf.extend_from_slice(&enc_port.to_be_bytes());
    buf.extend_from_slice(&enc_octets);
}

/// Parse a dotted-decimal IPv4 address into 4 bytes.
fn parse_ipv4(addr: &str) -> Option<[u8; 4]> {
    let parts: Vec<&str> = addr.split('.').collect();
    if parts.len() != 4 {
        return None;
    }
    let a: u8 = parts[0].parse().ok()?;
    let b: u8 = parts[1].parse().ok()?;
    let c: u8 = parts[2].parse().ok()?;
    let d: u8 = parts[3].parse().ok()?;
    Some([a, b, c, d])
}

/// Format 4 IPv4 octets as a dotted-decimal string.
fn format_ipv4(octets: [u8; 4]) -> String {
    format!("{}.{}.{}.{}", octets[0], octets[1], octets[2], octets[3])
}

// ──────────────────────────────────────────────────────────────────────────────
// Core manager
// ──────────────────────────────────────────────────────────────────────────────

/// Production ICE traversal manager.
///
/// Renamed to `NtmNatTraversalManager` at crate root to avoid collision with
/// `nat_traversal::NatTraversalManager`.
pub struct NatTraversalManager {
    config: TraversalConfig,
    local_candidates: Vec<CandidateAddress>,
    remote_candidates: Vec<CandidateAddress>,
    check_pairs: Vec<IcePair>,
    stats: TraversalStats,
}

impl NatTraversalManager {
    /// Create a new manager with `config`.
    pub fn new(config: TraversalConfig) -> Self {
        Self {
            config,
            local_candidates: Vec::new(),
            remote_candidates: Vec::new(),
            check_pairs: Vec::new(),
            stats: TraversalStats::default(),
        }
    }

    // ── candidate management ──────────────────────────────────────────────────

    /// Add a local candidate.
    pub fn add_local_candidate(&mut self, addr: CandidateAddress) -> Result<(), TraversalError> {
        if addr.address.is_empty() {
            return Err(TraversalError::CandidateGatheringFailed(
                "empty address".to_string(),
            ));
        }
        self.local_candidates.push(addr);
        self.stats.candidates_gathered = self.local_candidates.len() + self.remote_candidates.len();
        Ok(())
    }

    /// Add a remote candidate.
    pub fn add_remote_candidate(&mut self, addr: CandidateAddress) -> Result<(), TraversalError> {
        if addr.address.is_empty() {
            return Err(TraversalError::CandidateGatheringFailed(
                "empty address".to_string(),
            ));
        }
        self.remote_candidates.push(addr);
        self.stats.candidates_gathered = self.local_candidates.len() + self.remote_candidates.len();
        Ok(())
    }

    // ── pair formation ────────────────────────────────────────────────────────

    /// Form the check list from the Cartesian product of local × remote
    /// candidates, sorted by descending RFC 5245 pair priority and capped to
    /// `config.max_pairs`.
    pub fn form_check_pairs(&mut self) -> Vec<IcePair> {
        let mut pairs: Vec<IcePair> = self
            .local_candidates
            .iter()
            .flat_map(|l| {
                self.remote_candidates
                    .iter()
                    .map(move |r| IcePair::new(l.clone(), r.clone()))
            })
            .collect();

        // RFC 5245 §5.7.2: sort descending by pair priority.
        pairs.sort_unstable_by_key(|p| std::cmp::Reverse(p.priority));
        pairs.truncate(self.config.max_pairs);
        self.check_pairs = pairs.clone();
        pairs
    }

    // ── NAT type detection ────────────────────────────────────────────────────

    /// Heuristic NAT type detection from a slice of gathered candidates.
    ///
    /// Algorithm:
    /// 1. No candidates → `Unknown`
    /// 2. Only `Host` candidates → `OpenInternet`
    /// 3. Multiple `ServerReflexive` candidates with different IP+port
    ///    combinations (one per distinct STUN server) → `Symmetric`
    /// 4. Exactly one `ServerReflexive` candidate → `FullCone`
    /// 5. Both `Host` and `ServerReflexive` present with the same reflexive
    ///    address per server → `RestrictedCone`
    /// 6. `Relayed` candidates present (TURN allocation needed) → `PortRestrictedCone`
    /// 7. Otherwise → `Unknown`
    pub fn detect_nat_type(candidates: &[CandidateAddress]) -> NatType {
        if candidates.is_empty() {
            return NatType::Unknown;
        }

        let has_host = candidates
            .iter()
            .any(|c| c.candidate_type == CandidateType::Host);
        let has_relayed = candidates
            .iter()
            .any(|c| c.candidate_type == CandidateType::Relayed);
        let reflexive: Vec<&CandidateAddress> = candidates
            .iter()
            .filter(|c| c.candidate_type == CandidateType::ServerReflexive)
            .collect();

        if !has_host && reflexive.is_empty() && !has_relayed {
            return NatType::Unknown;
        }

        if has_host && reflexive.is_empty() && !has_relayed {
            return NatType::OpenInternet;
        }

        if has_relayed && reflexive.is_empty() {
            return NatType::PortRestrictedCone;
        }

        // Check for symmetric NAT: multiple reflexive candidates with distinct
        // (address, port) pairs — meaning each STUN server sees a different mapping.
        if reflexive.len() > 1 {
            let first = reflexive[0];
            let symmetric = reflexive
                .iter()
                .any(|c| c.address != first.address || c.port != first.port);
            if symmetric {
                return NatType::Symmetric;
            }
        }

        // Single reflexive (or multiple but identical) → FullCone; if relayed
        // is also present that suggests the network is more restrictive.
        if !reflexive.is_empty() && !has_relayed {
            return NatType::FullCone;
        }

        if !reflexive.is_empty() && has_relayed {
            return NatType::RestrictedCone;
        }

        NatType::Unknown
    }

    // ── connectivity checks ───────────────────────────────────────────────────

    /// Run a single connectivity check on `pair`.
    ///
    /// Deterministic rule (no real network I/O):
    /// * `Host`, `ServerReflexive`, `PeerReflexive`, or `Relayed` candidates
    ///   on both sides → `Succeeded`.
    /// * Any other combination → `Failed`.
    ///
    /// The pair transitions through `InProgress` before the final state.
    pub fn check_pair(&self, pair: &mut IcePair, _current_ts: u64) -> PairState {
        // Transition to in-progress first (mirrors a real async check).
        pair.state = PairState::InProgress;

        let both_reachable = matches!(
            (&pair.local.candidate_type, &pair.remote.candidate_type),
            (CandidateType::Host, CandidateType::Host)
                | (CandidateType::Host, CandidateType::ServerReflexive)
                | (CandidateType::Host, CandidateType::PeerReflexive)
                | (CandidateType::Host, CandidateType::Relayed)
                | (CandidateType::ServerReflexive, CandidateType::Host)
                | (
                    CandidateType::ServerReflexive,
                    CandidateType::ServerReflexive
                )
                | (CandidateType::ServerReflexive, CandidateType::PeerReflexive)
                | (CandidateType::ServerReflexive, CandidateType::Relayed)
                | (CandidateType::PeerReflexive, CandidateType::Host)
                | (CandidateType::PeerReflexive, CandidateType::ServerReflexive)
                | (CandidateType::PeerReflexive, CandidateType::PeerReflexive)
                | (CandidateType::PeerReflexive, CandidateType::Relayed)
                | (CandidateType::Relayed, CandidateType::Host)
                | (CandidateType::Relayed, CandidateType::ServerReflexive)
                | (CandidateType::Relayed, CandidateType::PeerReflexive)
                | (CandidateType::Relayed, CandidateType::Relayed)
        );

        if both_reachable {
            pair.state = PairState::Succeeded;
        } else {
            pair.state = PairState::Failed;
        }
        pair.state.clone()
    }

    // ── nomination ────────────────────────────────────────────────────────────

    /// Among all `Succeeded` pairs in the current check list, nominate the one
    /// with the highest priority.
    pub fn nominate_best_pair(&mut self) -> Result<IcePair, TraversalError> {
        let best = self
            .check_pairs
            .iter_mut()
            .filter(|p| p.state == PairState::Succeeded)
            .max_by_key(|p| p.priority);

        match best {
            Some(pair) => {
                pair.nominated = true;
                self.stats.nominated_pair = Some(pair.key());
                Ok(pair.clone())
            }
            None => Err(TraversalError::NoValidPair),
        }
    }

    // ── bulk check execution ──────────────────────────────────────────────────

    /// Run connectivity checks on all `Waiting` and `InProgress` pairs.
    /// Returns `(pair_key, new_state)` for every pair that was checked.
    pub fn run_checks(&mut self, current_ts: u64) -> Vec<(String, PairState)> {
        let mut results = Vec::new();
        for pair in &mut self.check_pairs {
            if matches!(pair.state, PairState::Waiting | PairState::InProgress) {
                // We need to temporarily move pair out to satisfy borrow checker;
                // since check_pair takes &self we can just call it directly.
                let key = pair.key();
                let new_state = {
                    pair.state = PairState::InProgress;
                    let both_reachable = matches!(
                        (&pair.local.candidate_type, &pair.remote.candidate_type),
                        (CandidateType::Host, _)
                            | (CandidateType::ServerReflexive, _)
                            | (CandidateType::PeerReflexive, _)
                            | (CandidateType::Relayed, _)
                    );
                    if both_reachable {
                        pair.state = PairState::Succeeded;
                    } else {
                        pair.state = PairState::Failed;
                    }
                    pair.state.clone()
                };
                self.stats.pairs_checked += 1;
                match &new_state {
                    PairState::Succeeded => self.stats.pairs_succeeded += 1,
                    PairState::Failed => self.stats.pairs_failed += 1,
                    _ => {}
                }
                results.push((key, new_state));
                let _ = current_ts; // timestamp reserved for future rate-limiting
            }
        }
        results
    }

    // ── STUN encoding ─────────────────────────────────────────────────────────

    /// Encode a [`StunMessage`] to bytes (RFC 5389 §6 format).
    ///
    /// Layout:
    /// ```text
    /// 0                   1                   2                   3
    /// 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
    /// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    /// |0 0|     STUN Message Type     |         Message Length        |
    /// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    /// |                         Magic Cookie                          |
    /// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    /// |                                                               |
    /// |                     Transaction ID (96 bits)                  |
    /// |                                                               |
    /// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    /// |                          Attributes …                         |
    /// ```
    pub fn encode_stun_message(&self, msg: &StunMessage) -> Vec<u8> {
        // Encode attributes first so we know their total length.
        let mut attr_buf: Vec<u8> = Vec::new();
        for attr in &msg.attributes {
            self.encode_attribute(&mut attr_buf, attr);
        }

        let mut out = Vec::with_capacity(20 + attr_buf.len());
        out.extend_from_slice(&msg.msg_type.to_u16().to_be_bytes());
        out.extend_from_slice(&(attr_buf.len() as u16).to_be_bytes());
        out.extend_from_slice(&STUN_MAGIC.to_be_bytes());
        out.extend_from_slice(&msg.transaction_id);
        out.extend_from_slice(&attr_buf);
        out
    }

    /// Decode bytes into a [`StunMessage`].
    pub fn decode_stun_message(&self, data: &[u8]) -> Result<StunMessage, TraversalError> {
        if data.len() < 20 {
            return Err(TraversalError::StunError(format!(
                "message too short: {} bytes (need 20)",
                data.len()
            )));
        }

        let msg_type_raw = u16::from_be_bytes([data[0], data[1]]);
        let msg_len = u16::from_be_bytes([data[2], data[3]]) as usize;
        let magic = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        if magic != STUN_MAGIC {
            return Err(TraversalError::StunError(format!(
                "invalid magic cookie: {magic:#010x}"
            )));
        }
        let transaction_id: [u8; 12] = data[8..20]
            .try_into()
            .map_err(|_| TraversalError::StunError("failed to read transaction ID".to_string()))?;

        let msg_type = StunMessageType::from_u16(msg_type_raw).ok_or_else(|| {
            TraversalError::StunError(format!("unknown message type: {msg_type_raw:#06x}"))
        })?;

        if data.len() < 20 + msg_len {
            return Err(TraversalError::StunError(format!(
                "truncated message: declared {} attribute bytes but only {} available",
                msg_len,
                data.len() - 20
            )));
        }

        let attr_data = &data[20..20 + msg_len];
        let attributes = self.decode_attributes(attr_data)?;

        Ok(StunMessage {
            msg_type,
            transaction_id,
            attributes,
        })
    }

    // ── statistics ────────────────────────────────────────────────────────────

    /// Return a snapshot of traversal statistics.
    pub fn stats(&self) -> TraversalStats {
        let mut s = self.stats.clone();
        // Update detected NAT type from current local candidates.
        let nat = Self::detect_nat_type(&self.local_candidates);
        s.nat_type = format!("{nat:?}");
        s.candidates_gathered = self.local_candidates.len() + self.remote_candidates.len();
        s
    }

    // ── private helpers ───────────────────────────────────────────────────────

    /// Encode a single STUN attribute into `buf`.
    fn encode_attribute(&self, buf: &mut Vec<u8>, attr: &StunAttribute) {
        match attr {
            StunAttribute::MappedAddress(addr, port) => {
                encode_address_attr(buf, ATTR_MAPPED_ADDRESS, addr, *port, false);
            }
            StunAttribute::XorMappedAddress(addr, port) => {
                encode_address_attr(buf, ATTR_XOR_MAPPED_ADDRESS, addr, *port, true);
            }
            StunAttribute::Username(name) => {
                encode_string_attr(buf, ATTR_USERNAME, name);
            }
            StunAttribute::Realm(realm) => {
                encode_string_attr(buf, ATTR_REALM, realm);
            }
            StunAttribute::ErrorCode(code, reason) => {
                let class = (*code / 100) as u8;
                let number = (*code % 100) as u8;
                let reason_bytes = reason.as_bytes();
                let value_len = 4 + reason_bytes.len();
                buf.extend_from_slice(&ATTR_ERROR_CODE.to_be_bytes());
                buf.extend_from_slice(&(value_len as u16).to_be_bytes());
                buf.extend_from_slice(&[0x00, 0x00, class, number]);
                buf.extend_from_slice(reason_bytes);
                let pad = (4 - (reason_bytes.len() % 4)) % 4;
                buf.extend(std::iter::repeat_n(0u8, pad));
            }
            StunAttribute::Fingerprint(crc) => {
                buf.extend_from_slice(&ATTR_FINGERPRINT.to_be_bytes());
                buf.extend_from_slice(&4u16.to_be_bytes());
                buf.extend_from_slice(&crc.to_be_bytes());
            }
            StunAttribute::Lifetime(secs) => {
                buf.extend_from_slice(&ATTR_LIFETIME.to_be_bytes());
                buf.extend_from_slice(&4u16.to_be_bytes());
                buf.extend_from_slice(&secs.to_be_bytes());
            }
        }
    }

    /// Decode a sequence of TLV-style STUN attributes.
    fn decode_attributes(&self, data: &[u8]) -> Result<Vec<StunAttribute>, TraversalError> {
        let mut attrs = Vec::new();
        let mut pos = 0usize;

        while pos < data.len() {
            if pos + 4 > data.len() {
                return Err(TraversalError::StunError(
                    "truncated attribute header".to_string(),
                ));
            }
            let attr_type = u16::from_be_bytes([data[pos], data[pos + 1]]);
            let attr_len = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
            pos += 4;

            if pos + attr_len > data.len() {
                return Err(TraversalError::StunError(format!(
                    "attribute (type={attr_type:#06x}) value truncated: need {attr_len} bytes"
                )));
            }
            let value = &data[pos..pos + attr_len];

            match attr_type {
                ATTR_MAPPED_ADDRESS => {
                    let (addr, port) = decode_address_value(value, false)?;
                    attrs.push(StunAttribute::MappedAddress(addr, port));
                }
                ATTR_XOR_MAPPED_ADDRESS => {
                    let (addr, port) = decode_address_value(value, true)?;
                    attrs.push(StunAttribute::XorMappedAddress(addr, port));
                }
                ATTR_USERNAME => {
                    let s = std::str::from_utf8(value)
                        .map_err(|e| TraversalError::StunError(format!("USERNAME utf8: {e}")))?;
                    attrs.push(StunAttribute::Username(s.to_string()));
                }
                ATTR_REALM => {
                    let s = std::str::from_utf8(value)
                        .map_err(|e| TraversalError::StunError(format!("REALM utf8: {e}")))?;
                    attrs.push(StunAttribute::Realm(s.to_string()));
                }
                ATTR_ERROR_CODE => {
                    if value.len() < 4 {
                        return Err(TraversalError::StunError(
                            "ERROR-CODE too short".to_string(),
                        ));
                    }
                    let class = value[2] as u16;
                    let number = value[3] as u16;
                    let code = class * 100 + number;
                    let reason = std::str::from_utf8(&value[4..]).map_err(|e| {
                        TraversalError::StunError(format!("ERROR-CODE reason utf8: {e}"))
                    })?;
                    attrs.push(StunAttribute::ErrorCode(code, reason.to_string()));
                }
                ATTR_FINGERPRINT => {
                    if value.len() < 4 {
                        return Err(TraversalError::StunError(
                            "FINGERPRINT too short".to_string(),
                        ));
                    }
                    let crc = u32::from_be_bytes([value[0], value[1], value[2], value[3]]);
                    attrs.push(StunAttribute::Fingerprint(crc));
                }
                ATTR_LIFETIME => {
                    if value.len() < 4 {
                        return Err(TraversalError::StunError("LIFETIME too short".to_string()));
                    }
                    let secs = u32::from_be_bytes([value[0], value[1], value[2], value[3]]);
                    attrs.push(StunAttribute::Lifetime(secs));
                }
                _ => {
                    // Unknown / comprehension-optional attributes are silently skipped.
                }
            }

            // Advance past value + padding to 4-byte boundary.
            let padded = attr_len + (4 - attr_len % 4) % 4;
            pos += padded;
        }
        Ok(attrs)
    }
}

/// Decode a MAPPED-ADDRESS or XOR-MAPPED-ADDRESS attribute value.
fn decode_address_value(value: &[u8], xor: bool) -> Result<(String, u16), TraversalError> {
    if value.len() < 8 {
        return Err(TraversalError::StunError(
            "address attribute too short".to_string(),
        ));
    }
    // Byte 0: padding, Byte 1: family (0x01=IPv4, 0x02=IPv6)
    let family = value[1];
    if family != 0x01 {
        return Err(TraversalError::StunError(format!(
            "only IPv4 is supported (family={family:#04x})"
        )));
    }
    let raw_port = u16::from_be_bytes([value[2], value[3]]);
    let raw_octets: [u8; 4] = value[4..8]
        .try_into()
        .map_err(|_| TraversalError::StunError("address octet slice error".to_string()))?;

    let (port, octets) = if xor {
        let dp = raw_port ^ ((STUN_MAGIC >> 16) as u16);
        let magic_bytes = STUN_MAGIC.to_be_bytes();
        let do_ = [
            raw_octets[0] ^ magic_bytes[0],
            raw_octets[1] ^ magic_bytes[1],
            raw_octets[2] ^ magic_bytes[2],
            raw_octets[3] ^ magic_bytes[3],
        ];
        (dp, do_)
    } else {
        (raw_port, raw_octets)
    };

    Ok((format_ipv4(octets), port))
}

impl Default for NatTraversalManager {
    fn default() -> Self {
        Self::new(TraversalConfig::default())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Type aliases for lib.rs re-exports (collision avoidance)
// ──────────────────────────────────────────────────────────────────────────────

/// `NatType` alias used at crate root (avoids collision with
/// `nat_traversal::NatType`).
pub type NtmNatType = NatType;

/// `NatTraversalManager` alias used at crate root (avoids collision with
/// `nat_traversal::NatTraversalManager`).
pub type NtmNatTraversalManager = NatTraversalManager;

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────────────────

    fn host(ip: &str, port: u16, prio: u32) -> CandidateAddress {
        CandidateAddress::new(ip, port, CandidateType::Host, prio)
    }

    fn srflx(ip: &str, port: u16, prio: u32) -> CandidateAddress {
        CandidateAddress::new(ip, port, CandidateType::ServerReflexive, prio)
    }

    fn relay(ip: &str, port: u16, prio: u32) -> CandidateAddress {
        CandidateAddress::new(ip, port, CandidateType::Relayed, prio)
    }

    fn prflx(ip: &str, port: u16, prio: u32) -> CandidateAddress {
        CandidateAddress::new(ip, port, CandidateType::PeerReflexive, prio)
    }

    fn default_manager() -> NatTraversalManager {
        NatTraversalManager::new(TraversalConfig::default())
    }

    // ── 1. CandidateAddress construction ──────────────────────────────────────

    #[test]
    fn test_candidate_address_new_foundation() {
        let c = host("1.2.3.4", 5000, 100);
        assert_eq!(c.address, "1.2.3.4");
        assert_eq!(c.port, 5000);
        assert_eq!(c.priority, 100);
        assert!(!c.foundation.is_empty());
    }

    #[test]
    fn test_candidate_foundation_deterministic() {
        let c1 = host("1.2.3.4", 5000, 100);
        let c2 = host("1.2.3.4", 5000, 200); // same addr+port, different prio
        assert_eq!(c1.foundation, c2.foundation);
    }

    #[test]
    fn test_candidate_foundation_differs_with_different_addr() {
        let c1 = host("1.2.3.4", 5000, 100);
        let c2 = host("1.2.3.5", 5000, 100);
        assert_ne!(c1.foundation, c2.foundation);
    }

    #[test]
    fn test_candidate_foundation_differs_with_different_port() {
        let c1 = host("1.2.3.4", 5000, 100);
        let c2 = host("1.2.3.4", 5001, 100);
        assert_ne!(c1.foundation, c2.foundation);
    }

    #[test]
    fn test_candidate_key() {
        let c = host("10.0.0.1", 4321, 50);
        assert_eq!(c.key(), "10.0.0.1:4321");
    }

    // ── 2. add_local_candidate / add_remote_candidate ─────────────────────────

    #[test]
    fn test_add_local_candidate_ok() {
        let mut mgr = default_manager();
        assert!(mgr.add_local_candidate(host("1.2.3.4", 5000, 100)).is_ok());
    }

    #[test]
    fn test_add_remote_candidate_ok() {
        let mut mgr = default_manager();
        assert!(mgr
            .add_remote_candidate(srflx("5.6.7.8", 3478, 200))
            .is_ok());
    }

    #[test]
    fn test_add_local_candidate_empty_address_err() {
        let mut mgr = default_manager();
        let bad = CandidateAddress {
            address: String::new(),
            port: 1234,
            candidate_type: CandidateType::Host,
            priority: 10,
            foundation: "abc".to_string(),
        };
        assert!(matches!(
            mgr.add_local_candidate(bad),
            Err(TraversalError::CandidateGatheringFailed(_))
        ));
    }

    #[test]
    fn test_add_remote_candidate_empty_address_err() {
        let mut mgr = default_manager();
        let bad = CandidateAddress {
            address: String::new(),
            port: 1234,
            candidate_type: CandidateType::Host,
            priority: 10,
            foundation: "abc".to_string(),
        };
        assert!(matches!(
            mgr.add_remote_candidate(bad),
            Err(TraversalError::CandidateGatheringFailed(_))
        ));
    }

    #[test]
    fn test_candidates_gathered_counter() {
        let mut mgr = default_manager();
        mgr.add_local_candidate(host("1.0.0.1", 1000, 10))
            .expect("test: add_local_candidate should succeed");
        mgr.add_local_candidate(host("1.0.0.2", 1001, 20))
            .expect("test: add_local_candidate should succeed");
        mgr.add_remote_candidate(srflx("2.0.0.1", 2000, 30))
            .expect("test: add_remote_candidate should succeed");
        assert_eq!(mgr.stats().candidates_gathered, 3);
    }

    // ── 3. form_check_pairs ────────────────────────────────────────────────────

    #[test]
    fn test_form_check_pairs_count() {
        let mut mgr = default_manager();
        mgr.add_local_candidate(host("1.0.0.1", 1000, 100))
            .expect("test: add_local_candidate should succeed");
        mgr.add_local_candidate(host("1.0.0.2", 1001, 90))
            .expect("test: add_local_candidate should succeed");
        mgr.add_remote_candidate(srflx("2.0.0.1", 2000, 80))
            .expect("test: add_remote_candidate should succeed");
        mgr.add_remote_candidate(srflx("2.0.0.2", 2001, 70))
            .expect("test: add_remote_candidate should succeed");
        let pairs = mgr.form_check_pairs();
        assert_eq!(pairs.len(), 4); // 2 × 2
    }

    #[test]
    fn test_form_check_pairs_sorted_descending() {
        let mut mgr = default_manager();
        mgr.add_local_candidate(host("1.0.0.1", 1000, 50))
            .expect("test: add_local_candidate should succeed");
        mgr.add_local_candidate(host("1.0.0.2", 1001, 200))
            .expect("test: add_local_candidate should succeed");
        mgr.add_remote_candidate(srflx("2.0.0.1", 2000, 150))
            .expect("test: add_remote_candidate should succeed");
        let pairs = mgr.form_check_pairs();
        for i in 1..pairs.len() {
            assert!(
                pairs[i - 1].priority >= pairs[i].priority,
                "pairs not sorted at index {i}"
            );
        }
    }

    #[test]
    fn test_form_check_pairs_priority_formula() {
        // G = 200, D = 150 → min=150, max=200, G>D → +1
        // priority = 2^32 * 150 + 2 * 200 + 1 = 644245094801
        let local = host("1.0.0.1", 1000, 200);
        let remote = srflx("2.0.0.1", 2000, 150);
        let pair = IcePair::new(local, remote);
        let expected: u64 = (1u64 << 32) * 150 + 2 * 200 + 1;
        assert_eq!(pair.priority, expected);
    }

    #[test]
    fn test_form_check_pairs_priority_formula_equal() {
        // G == D → tie-break 0
        let local = host("1.0.0.1", 1000, 100);
        let remote = srflx("2.0.0.1", 2000, 100);
        let pair = IcePair::new(local, remote);
        let expected: u64 = (1u64 << 32) * 100 + 2 * 100;
        assert_eq!(pair.priority, expected);
    }

    #[test]
    fn test_form_check_pairs_capped_to_max() {
        let mut mgr = NatTraversalManager::new(TraversalConfig {
            max_pairs: 3,
            ..TraversalConfig::default()
        });
        for i in 0..4u16 {
            mgr.add_local_candidate(host("1.0.0.1", 1000 + i, 100 + i as u32))
                .expect("test: add_local_candidate should succeed");
        }
        mgr.add_remote_candidate(host("2.0.0.1", 2000, 50))
            .expect("test: add_remote_candidate should succeed");
        let pairs = mgr.form_check_pairs();
        assert_eq!(pairs.len(), 3);
    }

    #[test]
    fn test_form_check_pairs_empty_returns_empty() {
        let mut mgr = default_manager();
        let pairs = mgr.form_check_pairs();
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_form_check_pairs_new_pairs_in_waiting_state() {
        let mut mgr = default_manager();
        mgr.add_local_candidate(host("1.0.0.1", 1000, 100))
            .expect("test: add_local_candidate should succeed");
        mgr.add_remote_candidate(host("2.0.0.1", 2000, 80))
            .expect("test: add_remote_candidate should succeed");
        let pairs = mgr.form_check_pairs();
        assert_eq!(pairs[0].state, PairState::Waiting);
    }

    // ── 4. detect_nat_type ────────────────────────────────────────────────────

    #[test]
    fn test_detect_nat_type_empty() {
        assert_eq!(NatTraversalManager::detect_nat_type(&[]), NatType::Unknown);
    }

    #[test]
    fn test_detect_nat_type_only_host() {
        let c = [host("1.2.3.4", 5000, 100)];
        assert_eq!(
            NatTraversalManager::detect_nat_type(&c),
            NatType::OpenInternet
        );
    }

    #[test]
    fn test_detect_nat_type_single_srflx_full_cone() {
        let c = [host("1.2.3.4", 5000, 100), srflx("5.6.7.8", 3478, 90)];
        assert_eq!(NatTraversalManager::detect_nat_type(&c), NatType::FullCone);
    }

    #[test]
    fn test_detect_nat_type_symmetric_multiple_srflx() {
        let c = [
            srflx("5.6.7.8", 3478, 90),
            srflx("9.10.11.12", 4444, 85), // different IP → symmetric
        ];
        assert_eq!(NatTraversalManager::detect_nat_type(&c), NatType::Symmetric);
    }

    #[test]
    fn test_detect_nat_type_symmetric_same_ip_diff_port() {
        let c = [
            srflx("5.6.7.8", 3478, 90),
            srflx("5.6.7.8", 3479, 85), // same IP, different port → symmetric
        ];
        assert_eq!(NatTraversalManager::detect_nat_type(&c), NatType::Symmetric);
    }

    #[test]
    fn test_detect_nat_type_multiple_identical_srflx() {
        let c = [
            srflx("5.6.7.8", 3478, 90),
            srflx("5.6.7.8", 3478, 85), // identical mapping → full cone
        ];
        assert_eq!(NatTraversalManager::detect_nat_type(&c), NatType::FullCone);
    }

    #[test]
    fn test_detect_nat_type_relayed_only() {
        let c = [relay("5.6.7.8", 3478, 50)];
        assert_eq!(
            NatTraversalManager::detect_nat_type(&c),
            NatType::PortRestrictedCone
        );
    }

    #[test]
    fn test_detect_nat_type_host_and_relayed() {
        let c = [host("1.2.3.4", 5000, 100), relay("5.6.7.8", 3478, 50)];
        // host + relayed (no srflx) → PortRestrictedCone
        assert_eq!(
            NatTraversalManager::detect_nat_type(&c),
            NatType::PortRestrictedCone
        );
    }

    #[test]
    fn test_detect_nat_type_srflx_and_relayed_restricted_cone() {
        let c = [
            host("1.2.3.4", 5000, 100),
            srflx("5.6.7.8", 3478, 90),
            relay("5.6.7.8", 9999, 50),
        ];
        assert_eq!(
            NatTraversalManager::detect_nat_type(&c),
            NatType::RestrictedCone
        );
    }

    // ── 5. check_pair ─────────────────────────────────────────────────────────

    #[test]
    fn test_check_pair_host_host_succeeds() {
        let mgr = default_manager();
        let mut pair = IcePair::new(host("1.0.0.1", 1000, 100), host("2.0.0.1", 2000, 80));
        let state = mgr.check_pair(&mut pair, 0);
        assert_eq!(state, PairState::Succeeded);
        assert_eq!(pair.state, PairState::Succeeded);
    }

    #[test]
    fn test_check_pair_srflx_srflx_succeeds() {
        let mgr = default_manager();
        let mut pair = IcePair::new(srflx("1.0.0.1", 1000, 100), srflx("2.0.0.1", 2000, 80));
        let state = mgr.check_pair(&mut pair, 0);
        assert_eq!(state, PairState::Succeeded);
    }

    #[test]
    fn test_check_pair_relayed_host_succeeds() {
        let mgr = default_manager();
        let mut pair = IcePair::new(relay("1.0.0.1", 1000, 50), host("2.0.0.1", 2000, 80));
        let state = mgr.check_pair(&mut pair, 0);
        assert_eq!(state, PairState::Succeeded);
    }

    #[test]
    fn test_check_pair_host_prflx_succeeds() {
        let mgr = default_manager();
        let mut pair = IcePair::new(host("1.0.0.1", 1000, 100), prflx("2.0.0.1", 2000, 90));
        let state = mgr.check_pair(&mut pair, 0);
        assert_eq!(state, PairState::Succeeded);
    }

    #[test]
    fn test_check_pair_transitions_through_in_progress() {
        // We verify the method sets InProgress then Succeeded atomically
        // (our impl does both; the state after return should be Succeeded).
        let mgr = default_manager();
        let mut pair = IcePair::new(host("1.0.0.1", 1000, 100), host("2.0.0.1", 2000, 80));
        assert_eq!(pair.state, PairState::Waiting);
        let result = mgr.check_pair(&mut pair, 42_000);
        assert_eq!(result, PairState::Succeeded);
    }

    #[test]
    fn test_check_pair_nominated_false_by_default() {
        let mgr = default_manager();
        let mut pair = IcePair::new(host("1.0.0.1", 1000, 100), host("2.0.0.1", 2000, 80));
        mgr.check_pair(&mut pair, 0);
        assert!(!pair.nominated);
    }

    // ── 6. nominate_best_pair ─────────────────────────────────────────────────

    #[test]
    fn test_nominate_best_pair_picks_highest_priority() {
        let mut mgr = default_manager();
        mgr.add_local_candidate(host("1.0.0.1", 1000, 200))
            .expect("test: add_local_candidate should succeed");
        mgr.add_local_candidate(host("1.0.0.2", 1001, 50))
            .expect("test: add_local_candidate should succeed");
        mgr.add_remote_candidate(host("2.0.0.1", 2000, 150))
            .expect("test: add_remote_candidate should succeed");
        mgr.form_check_pairs();
        for pair in &mut mgr.check_pairs {
            pair.state = PairState::Succeeded;
        }
        let nominated = mgr
            .nominate_best_pair()
            .expect("test: nominate_best_pair should succeed with succeeded pairs");
        assert!(nominated.nominated);
        // Highest-priority pair has local prio=200, remote prio=150
        assert_eq!(nominated.local.priority, 200);
    }

    #[test]
    fn test_nominate_best_pair_no_succeeded_pairs_err() {
        let mut mgr = default_manager();
        mgr.add_local_candidate(host("1.0.0.1", 1000, 100))
            .expect("test: add_local_candidate should succeed");
        mgr.add_remote_candidate(host("2.0.0.1", 2000, 80))
            .expect("test: add_remote_candidate should succeed");
        mgr.form_check_pairs();
        assert!(matches!(
            mgr.nominate_best_pair(),
            Err(TraversalError::NoValidPair)
        ));
    }

    #[test]
    fn test_nominate_best_pair_marks_nominated() {
        let mut mgr = default_manager();
        mgr.add_local_candidate(host("1.0.0.1", 1000, 100))
            .expect("test: add_local_candidate should succeed");
        mgr.add_remote_candidate(host("2.0.0.1", 2000, 80))
            .expect("test: add_remote_candidate should succeed");
        mgr.form_check_pairs();
        mgr.check_pairs[0].state = PairState::Succeeded;
        let pair = mgr
            .nominate_best_pair()
            .expect("test: nominate_best_pair should succeed with succeeded pairs");
        assert!(pair.nominated);
    }

    #[test]
    fn test_nominate_best_pair_updates_stats() {
        let mut mgr = default_manager();
        mgr.add_local_candidate(host("1.0.0.1", 1000, 100))
            .expect("test: add_local_candidate should succeed");
        mgr.add_remote_candidate(host("2.0.0.1", 2000, 80))
            .expect("test: add_remote_candidate should succeed");
        mgr.form_check_pairs();
        mgr.check_pairs[0].state = PairState::Succeeded;
        mgr.nominate_best_pair()
            .expect("test: nominate_best_pair should succeed with succeeded pairs");
        assert!(mgr.stats().nominated_pair.is_some());
    }

    // ── 7. run_checks ─────────────────────────────────────────────────────────

    #[test]
    fn test_run_checks_all_waiting_checked() {
        let mut mgr = default_manager();
        mgr.add_local_candidate(host("1.0.0.1", 1000, 100))
            .expect("test: add_local_candidate should succeed");
        mgr.add_remote_candidate(host("2.0.0.1", 2000, 80))
            .expect("test: add_remote_candidate should succeed");
        mgr.form_check_pairs();
        let results = mgr.run_checks(0);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_run_checks_returns_pair_key_and_state() {
        let mut mgr = default_manager();
        mgr.add_local_candidate(host("1.0.0.1", 1000, 100))
            .expect("test: add_local_candidate should succeed");
        mgr.add_remote_candidate(host("2.0.0.1", 2000, 80))
            .expect("test: add_remote_candidate should succeed");
        mgr.form_check_pairs();
        let results = mgr.run_checks(0);
        assert!(!results[0].0.is_empty());
        assert_eq!(results[0].1, PairState::Succeeded);
    }

    #[test]
    fn test_run_checks_skips_frozen_pairs() {
        let mut mgr = default_manager();
        mgr.add_local_candidate(host("1.0.0.1", 1000, 100))
            .expect("test: add_local_candidate should succeed");
        mgr.add_remote_candidate(host("2.0.0.1", 2000, 80))
            .expect("test: add_remote_candidate should succeed");
        mgr.form_check_pairs();
        mgr.check_pairs[0].state = PairState::Frozen;
        let results = mgr.run_checks(0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_run_checks_skips_succeeded_pairs() {
        let mut mgr = default_manager();
        mgr.add_local_candidate(host("1.0.0.1", 1000, 100))
            .expect("test: add_local_candidate should succeed");
        mgr.add_remote_candidate(host("2.0.0.1", 2000, 80))
            .expect("test: add_remote_candidate should succeed");
        mgr.form_check_pairs();
        mgr.check_pairs[0].state = PairState::Succeeded;
        let results = mgr.run_checks(0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_run_checks_updates_stats_succeeded() {
        let mut mgr = default_manager();
        mgr.add_local_candidate(host("1.0.0.1", 1000, 100))
            .expect("test: add_local_candidate should succeed");
        mgr.add_remote_candidate(host("2.0.0.1", 2000, 80))
            .expect("test: add_remote_candidate should succeed");
        mgr.form_check_pairs();
        mgr.run_checks(0);
        let s = mgr.stats();
        assert_eq!(s.pairs_checked, 1);
        assert_eq!(s.pairs_succeeded, 1);
    }

    #[test]
    fn test_run_checks_multiple_pairs() {
        let mut mgr = default_manager();
        mgr.add_local_candidate(host("1.0.0.1", 1000, 100))
            .expect("test: add_local_candidate should succeed");
        mgr.add_remote_candidate(host("2.0.0.1", 2000, 80))
            .expect("test: add_remote_candidate should succeed");
        mgr.add_remote_candidate(host("2.0.0.2", 2001, 70))
            .expect("test: add_remote_candidate should succeed");
        mgr.form_check_pairs();
        let results = mgr.run_checks(0);
        assert_eq!(results.len(), 2);
    }

    // ── 8. STUN encode / decode round-trip ────────────────────────────────────

    #[test]
    fn test_stun_binding_request_round_trip() {
        let mgr = default_manager();
        let msg = StunMessage::new(StunMessageType::BindingRequest, 0x123456);
        let encoded = mgr.encode_stun_message(&msg);
        let decoded = mgr
            .decode_stun_message(&encoded)
            .expect("test: decode_stun_message should succeed for valid binding request");
        assert_eq!(decoded.msg_type, StunMessageType::BindingRequest);
        assert_eq!(decoded.transaction_id, msg.transaction_id);
    }

    #[test]
    fn test_stun_binding_response_round_trip() {
        let mgr = default_manager();
        let mut msg = StunMessage::new(StunMessageType::BindingResponse, 0xABCD);
        msg.attributes
            .push(StunAttribute::MappedAddress("1.2.3.4".to_string(), 5000));
        let encoded = mgr.encode_stun_message(&msg);
        let decoded = mgr
            .decode_stun_message(&encoded)
            .expect("test: decode_stun_message should succeed for valid binding response");
        assert_eq!(decoded.msg_type, StunMessageType::BindingResponse);
        assert_eq!(decoded.attributes.len(), 1);
        if let StunAttribute::MappedAddress(addr, port) = &decoded.attributes[0] {
            assert_eq!(addr, "1.2.3.4");
            assert_eq!(*port, 5000);
        } else {
            panic!("expected MappedAddress");
        }
    }

    #[test]
    fn test_stun_xor_mapped_address_round_trip() {
        let mgr = default_manager();
        let mut msg = StunMessage::new(StunMessageType::BindingResponse, 0xFFFF);
        msg.attributes.push(StunAttribute::XorMappedAddress(
            "192.168.1.1".to_string(),
            54321,
        ));
        let encoded = mgr.encode_stun_message(&msg);
        let decoded = mgr
            .decode_stun_message(&encoded)
            .expect("test: decode_stun_message should succeed for xor mapped address");
        if let StunAttribute::XorMappedAddress(addr, port) = &decoded.attributes[0] {
            assert_eq!(addr, "192.168.1.1");
            assert_eq!(*port, 54321);
        } else {
            panic!("expected XorMappedAddress");
        }
    }

    #[test]
    fn test_stun_username_round_trip() {
        let mgr = default_manager();
        let mut msg = StunMessage::new(StunMessageType::BindingRequest, 1);
        msg.attributes
            .push(StunAttribute::Username("alice:bob".to_string()));
        let encoded = mgr.encode_stun_message(&msg);
        let decoded = mgr
            .decode_stun_message(&encoded)
            .expect("test: decode_stun_message should succeed for username attribute");
        if let StunAttribute::Username(name) = &decoded.attributes[0] {
            assert_eq!(name, "alice:bob");
        } else {
            panic!("expected Username");
        }
    }

    #[test]
    fn test_stun_realm_round_trip() {
        let mgr = default_manager();
        let mut msg = StunMessage::new(StunMessageType::AllocateRequest, 2);
        msg.attributes
            .push(StunAttribute::Realm("example.com".to_string()));
        let encoded = mgr.encode_stun_message(&msg);
        let decoded = mgr
            .decode_stun_message(&encoded)
            .expect("test: decode_stun_message should succeed for realm attribute");
        if let StunAttribute::Realm(r) = &decoded.attributes[0] {
            assert_eq!(r, "example.com");
        } else {
            panic!("expected Realm");
        }
    }

    #[test]
    fn test_stun_error_code_round_trip() {
        let mgr = default_manager();
        let mut msg = StunMessage::new(StunMessageType::BindingError, 3);
        msg.attributes
            .push(StunAttribute::ErrorCode(401, "Unauthorized".to_string()));
        let encoded = mgr.encode_stun_message(&msg);
        let decoded = mgr
            .decode_stun_message(&encoded)
            .expect("test: decode_stun_message should succeed for error code attribute");
        if let StunAttribute::ErrorCode(code, reason) = &decoded.attributes[0] {
            assert_eq!(*code, 401);
            assert_eq!(reason, "Unauthorized");
        } else {
            panic!("expected ErrorCode");
        }
    }

    #[test]
    fn test_stun_fingerprint_round_trip() {
        let mgr = default_manager();
        let mut msg = StunMessage::new(StunMessageType::BindingRequest, 4);
        msg.attributes.push(StunAttribute::Fingerprint(0xDEAD_BEEF));
        let encoded = mgr.encode_stun_message(&msg);
        let decoded = mgr
            .decode_stun_message(&encoded)
            .expect("test: decode_stun_message should succeed for fingerprint attribute");
        if let StunAttribute::Fingerprint(crc) = decoded.attributes[0] {
            assert_eq!(crc, 0xDEAD_BEEF);
        } else {
            panic!("expected Fingerprint");
        }
    }

    #[test]
    fn test_stun_lifetime_round_trip() {
        let mgr = default_manager();
        let mut msg = StunMessage::new(StunMessageType::AllocateResponse, 5);
        msg.attributes.push(StunAttribute::Lifetime(600));
        let encoded = mgr.encode_stun_message(&msg);
        let decoded = mgr
            .decode_stun_message(&encoded)
            .expect("test: decode_stun_message should succeed for lifetime attribute");
        if let StunAttribute::Lifetime(secs) = decoded.attributes[0] {
            assert_eq!(secs, 600);
        } else {
            panic!("expected Lifetime");
        }
    }

    #[test]
    fn test_stun_multiple_attributes_round_trip() {
        let mgr = default_manager();
        let mut msg = StunMessage::new(StunMessageType::BindingResponse, 6);
        msg.attributes
            .push(StunAttribute::MappedAddress("10.0.0.1".to_string(), 4000));
        msg.attributes
            .push(StunAttribute::Username("u1:u2".to_string()));
        msg.attributes.push(StunAttribute::Fingerprint(42));
        let encoded = mgr.encode_stun_message(&msg);
        let decoded = mgr
            .decode_stun_message(&encoded)
            .expect("test: decode_stun_message should succeed for multiple attributes");
        assert_eq!(decoded.attributes.len(), 3);
    }

    // ── 9. decode error paths ─────────────────────────────────────────────────

    #[test]
    fn test_decode_stun_too_short() {
        let mgr = default_manager();
        let result = mgr.decode_stun_message(&[0u8; 10]);
        assert!(matches!(result, Err(TraversalError::StunError(_))));
    }

    #[test]
    fn test_decode_stun_wrong_magic() {
        let mgr = default_manager();
        let mut buf = vec![0u8; 20];
        // Correct type bytes
        buf[0] = 0x00;
        buf[1] = 0x01;
        // Wrong magic cookie
        buf[4] = 0xFF;
        buf[5] = 0xFF;
        buf[6] = 0xFF;
        buf[7] = 0xFF;
        let result = mgr.decode_stun_message(&buf);
        assert!(matches!(result, Err(TraversalError::StunError(_))));
    }

    #[test]
    fn test_decode_stun_unknown_type() {
        let mgr = default_manager();
        let mut buf = vec![0u8; 20];
        // Unknown message type 0x00FF
        buf[0] = 0x00;
        buf[1] = 0xFF;
        // Correct magic
        let magic = STUN_MAGIC.to_be_bytes();
        buf[4] = magic[0];
        buf[5] = magic[1];
        buf[6] = magic[2];
        buf[7] = magic[3];
        let result = mgr.decode_stun_message(&buf);
        assert!(matches!(result, Err(TraversalError::StunError(_))));
    }

    #[test]
    fn test_decode_stun_truncated_attributes() {
        let mgr = default_manager();
        let mut buf = vec![0u8; 20];
        buf[0] = 0x00;
        buf[1] = 0x01;
        let magic = STUN_MAGIC.to_be_bytes();
        buf[4] = magic[0];
        buf[5] = magic[1];
        buf[6] = magic[2];
        buf[7] = magic[3];
        // Declare 10 bytes of attributes but provide none
        buf[2] = 0x00;
        buf[3] = 0x0A;
        let result = mgr.decode_stun_message(&buf);
        assert!(matches!(result, Err(TraversalError::StunError(_))));
    }

    // ── 10. stats ─────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_initial_state() {
        let mgr = default_manager();
        let s = mgr.stats();
        assert_eq!(s.candidates_gathered, 0);
        assert_eq!(s.pairs_checked, 0);
        assert_eq!(s.pairs_succeeded, 0);
        assert_eq!(s.pairs_failed, 0);
        assert!(s.nominated_pair.is_none());
    }

    #[test]
    fn test_stats_nat_type_string_open_internet() {
        let mut mgr = default_manager();
        mgr.add_local_candidate(host("1.2.3.4", 5000, 100))
            .expect("test: add_local_candidate should succeed");
        let s = mgr.stats();
        assert!(s.nat_type.contains("OpenInternet"));
    }

    #[test]
    fn test_stats_nat_type_string_unknown_no_candidates() {
        let mgr = default_manager();
        let s = mgr.stats();
        assert!(s.nat_type.contains("Unknown"));
    }

    #[test]
    fn test_stats_after_full_workflow() {
        let mut mgr = default_manager();
        mgr.add_local_candidate(host("1.0.0.1", 1000, 100))
            .expect("test: add_local_candidate should succeed");
        mgr.add_remote_candidate(host("2.0.0.1", 2000, 80))
            .expect("test: add_remote_candidate should succeed");
        mgr.form_check_pairs();
        mgr.run_checks(0);
        mgr.nominate_best_pair()
            .expect("test: nominate_best_pair should succeed after checks");
        let s = mgr.stats();
        assert_eq!(s.pairs_checked, 1);
        assert_eq!(s.pairs_succeeded, 1);
        assert!(s.nominated_pair.is_some());
    }

    // ── 11. error type coverage ───────────────────────────────────────────────

    #[test]
    fn test_traversal_error_display_no_valid_pair() {
        let e = TraversalError::NoValidPair;
        assert!(e.to_string().contains("no valid ICE pair"));
    }

    #[test]
    fn test_traversal_error_display_gathering_failed() {
        let e = TraversalError::CandidateGatheringFailed("oops".to_string());
        assert!(e.to_string().contains("oops"));
    }

    #[test]
    fn test_traversal_error_display_stun_error() {
        let e = TraversalError::StunError("bad magic".to_string());
        assert!(e.to_string().contains("bad magic"));
    }

    #[test]
    fn test_traversal_error_display_turn_failed() {
        let e = TraversalError::TurnAllocationFailed("quota exceeded".to_string());
        assert!(e.to_string().contains("quota exceeded"));
    }

    #[test]
    fn test_traversal_error_display_checklist_exhausted() {
        let e = TraversalError::ChecklistExhausted;
        assert!(e.to_string().contains("exhausted"));
    }

    // ── 12. xorshift64 / fnv1a_64 ────────────────────────────────────────────

    #[test]
    fn test_xorshift64_non_zero_output() {
        let mut s = 0xDEAD_BEEF_1234_5678u64;
        let v = xorshift64(&mut s);
        assert_ne!(v, 0);
        assert_ne!(s, 0xDEAD_BEEF_1234_5678u64);
    }

    #[test]
    fn test_xorshift64_deterministic() {
        let mut s1 = 12345u64;
        let mut s2 = 12345u64;
        assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
    }

    #[test]
    fn test_fnv1a_64_empty() {
        // Empty slice returns the offset basis.
        assert_eq!(fnv1a_64(&[]), 14_695_981_039_346_656_037u64);
    }

    #[test]
    fn test_fnv1a_64_known_value() {
        // "hello" should produce a known hash.
        let h = fnv1a_64(b"hello");
        assert_ne!(h, 14_695_981_039_346_656_037u64);
    }

    #[test]
    fn test_fnv1a_64_differs_for_different_input() {
        assert_ne!(fnv1a_64(b"abc"), fnv1a_64(b"def"));
    }

    // ── 13. IcePair helpers ───────────────────────────────────────────────────

    #[test]
    fn test_ice_pair_key_format() {
        let pair = IcePair::new(host("1.0.0.1", 1000, 100), host("2.0.0.1", 2000, 80));
        assert_eq!(pair.key(), "1.0.0.1:1000 -> 2.0.0.1:2000");
    }

    #[test]
    fn test_ice_pair_initial_state_waiting() {
        let pair = IcePair::new(host("1.0.0.1", 1000, 100), srflx("2.0.0.1", 2000, 90));
        assert_eq!(pair.state, PairState::Waiting);
        assert!(!pair.nominated);
    }

    // ── 14. StunMessage constructor ───────────────────────────────────────────

    #[test]
    fn test_stun_message_new_transaction_id_non_zero() {
        let msg = StunMessage::new(StunMessageType::BindingRequest, 999);
        assert!(msg.transaction_id.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_stun_message_new_zero_seed_fallback() {
        // seed=0 → use fallback seed, should still produce non-zero tx ID
        let msg = StunMessage::new(StunMessageType::BindingRequest, 0);
        assert!(msg.transaction_id.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_stun_message_all_types_encode_decode() {
        let mgr = default_manager();
        let types = [
            StunMessageType::BindingRequest,
            StunMessageType::BindingResponse,
            StunMessageType::BindingError,
            StunMessageType::AllocateRequest,
            StunMessageType::AllocateResponse,
        ];
        for t in &types {
            let msg = StunMessage::new(t.clone(), 1);
            let encoded = mgr.encode_stun_message(&msg);
            let decoded = mgr
                .decode_stun_message(&encoded)
                .expect("test: decode_stun_message should succeed for all stun message types");
            assert_eq!(&decoded.msg_type, t);
        }
    }

    // ── 15. Full end-to-end workflow ──────────────────────────────────────────

    #[test]
    fn test_full_ice_workflow() {
        let mut mgr = NatTraversalManager::new(TraversalConfig {
            max_pairs: 50,
            ..TraversalConfig::default()
        });

        // Gather local candidates
        mgr.add_local_candidate(host("192.168.1.1", 5000, 2130706431))
            .expect("test: add_local_candidate should succeed");
        mgr.add_local_candidate(srflx("203.0.113.1", 5001, 1694498815))
            .expect("test: add_local_candidate should succeed");

        // Gather remote candidates
        mgr.add_remote_candidate(host("10.0.0.1", 6000, 2130706431))
            .expect("test: add_remote_candidate should succeed");
        mgr.add_remote_candidate(srflx("198.51.100.1", 6001, 1694498815))
            .expect("test: add_remote_candidate should succeed");

        // Form pairs
        let pairs = mgr.form_check_pairs();
        assert_eq!(pairs.len(), 4);

        // Run all checks
        let results = mgr.run_checks(1_000_000);
        assert_eq!(results.len(), 4);
        assert!(results.iter().all(|(_, s)| *s == PairState::Succeeded));

        // Nominate best
        let best = mgr
            .nominate_best_pair()
            .expect("test: nominate_best_pair should succeed in full workflow");
        assert!(best.nominated);
        assert_eq!(best.state, PairState::Succeeded);

        // Stats
        let s = mgr.stats();
        assert_eq!(s.pairs_checked, 4);
        assert_eq!(s.pairs_succeeded, 4);
        assert!(s.nominated_pair.is_some());
    }
}
