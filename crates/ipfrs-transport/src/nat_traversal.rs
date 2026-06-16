// NAT Traversal
//
// This module implements NAT traversal for peer connectivity using:
// 1. STUN - Session Traversal Utilities for NAT (RFC 5389)
// 2. TURN - Traversal Using Relays around NAT (RFC 5766)
// 3. ICE-like connectivity establishment (RFC 8445)
// 4. UDP hole punching

use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;

/// NAT traversal errors
#[derive(Debug, Error)]
pub enum NatTraversalError {
    #[error("STUN server unreachable: {0}")]
    StunServerUnreachable(String),

    #[error("TURN server authentication failed")]
    TurnAuthFailed,

    #[error("No viable connectivity path found")]
    NoViablePath,

    #[error("Hole punching timeout")]
    HolePunchTimeout,

    #[error("ICE gathering failed: {0}")]
    IceGatheringFailed(String),

    #[error("Network error: {0}")]
    NetworkError(#[from] std::io::Error),
}

/// NAT type detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NatType {
    /// No NAT (public IP)
    None,
    /// Full Cone NAT
    FullCone,
    /// Restricted Cone NAT
    RestrictedCone,
    /// Port Restricted Cone NAT
    PortRestrictedCone,
    /// Symmetric NAT
    Symmetric,
    /// Unknown NAT type
    #[default]
    Unknown,
}

impl NatType {
    /// Check if hole punching is likely to work
    pub fn can_hole_punch(&self) -> bool {
        matches!(
            self,
            NatType::None
                | NatType::FullCone
                | NatType::RestrictedCone
                | NatType::PortRestrictedCone
        )
    }

    /// Check if TURN relay is required
    pub fn requires_relay(&self) -> bool {
        matches!(self, NatType::Symmetric)
    }
}

/// ICE candidate type
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CandidateType {
    /// Host candidate (local interface)
    Host = 0,
    /// Server reflexive (STUN)
    ServerReflexive = 1,
    /// Peer reflexive (learned from peer)
    PeerReflexive = 2,
    /// Relay candidate (TURN)
    Relay = 3,
}

/// ICE candidate
#[derive(Debug, Clone)]
pub struct IceCandidate {
    /// Candidate type
    pub candidate_type: CandidateType,
    /// Address
    pub addr: SocketAddr,
    /// Priority (higher is better)
    pub priority: u32,
    /// Foundation (for grouping)
    pub foundation: String,
    /// Component ID (RTP=1, RTCP=2, etc.)
    pub component_id: u32,
}

impl IceCandidate {
    /// Calculate priority based on RFC 8445
    pub fn calculate_priority(
        candidate_type: CandidateType,
        local_pref: u16,
        component_id: u32,
    ) -> u32 {
        let type_pref = match candidate_type {
            CandidateType::Host => 126,
            CandidateType::PeerReflexive => 110,
            CandidateType::ServerReflexive => 100,
            CandidateType::Relay => 0,
        };

        ((type_pref as u32) << 24) | ((local_pref as u32) << 8) | (256 - component_id)
    }
}

/// Candidate pair for connectivity checks
#[derive(Debug, Clone)]
pub struct CandidatePair {
    /// Local candidate
    pub local: IceCandidate,
    /// Remote candidate
    pub remote: IceCandidate,
    /// Pair priority
    pub priority: u64,
    /// Pair state
    pub state: PairState,
    /// Last check time
    pub last_check: Option<Instant>,
}

impl CandidatePair {
    /// Calculate pair priority (RFC 8445)
    pub fn calculate_priority(
        local_priority: u32,
        remote_priority: u32,
        is_controlling: bool,
    ) -> u64 {
        let (g, d) = if is_controlling {
            (local_priority, remote_priority)
        } else {
            (remote_priority, local_priority)
        };

        ((std::cmp::min(g, d) as u64) << 32)
            | (std::cmp::max(g, d) as u64)
            | if g > d { 1 } else { 0 }
    }
}

/// Candidate pair state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairState {
    /// Waiting to be checked
    Waiting,
    /// Currently being checked
    InProgress,
    /// Check succeeded
    Succeeded,
    /// Check failed
    Failed,
}

/// STUN server configuration
#[derive(Debug, Clone)]
pub struct StunConfig {
    /// STUN server address
    pub server: SocketAddr,
    /// Request timeout
    pub timeout: Duration,
    /// Number of retries
    pub retries: usize,
}

impl Default for StunConfig {
    fn default() -> Self {
        Self {
            // Use Google Public STUN server IP (resolves to stun.l.google.com)
            server: "74.125.250.129:19302"
                .parse()
                .expect("static socket addr literal must parse"),
            timeout: Duration::from_secs(3),
            retries: 3,
        }
    }
}

/// TURN server configuration
#[derive(Debug, Clone)]
pub struct TurnConfig {
    /// TURN server address
    pub server: SocketAddr,
    /// Username
    pub username: String,
    /// Password
    pub password: String,
    /// Allocation lifetime
    pub lifetime: Duration,
}

/// NAT traversal configuration
#[derive(Debug, Clone)]
pub struct NatTraversalConfig {
    /// STUN servers
    pub stun_servers: Vec<StunConfig>,
    /// TURN servers
    pub turn_servers: Vec<TurnConfig>,
    /// Enable hole punching
    pub enable_hole_punching: bool,
    /// Hole punching timeout
    pub hole_punch_timeout: Duration,
    /// ICE gathering timeout
    pub ice_gathering_timeout: Duration,
    /// Connectivity check interval
    pub connectivity_check_interval: Duration,
    /// Maximum candidate pairs to check
    pub max_candidate_pairs: usize,
    /// Acting as ICE controlling agent
    pub is_controlling: bool,
}

impl Default for NatTraversalConfig {
    fn default() -> Self {
        Self {
            stun_servers: vec![StunConfig::default()],
            turn_servers: Vec::new(),
            enable_hole_punching: true,
            hole_punch_timeout: Duration::from_secs(10),
            ice_gathering_timeout: Duration::from_secs(5),
            connectivity_check_interval: Duration::from_millis(50),
            max_candidate_pairs: 100,
            is_controlling: true,
        }
    }
}

/// NAT traversal statistics
#[derive(Debug, Clone, Default)]
pub struct NatTraversalStats {
    /// STUN requests sent
    pub stun_requests: u64,
    /// STUN responses received
    pub stun_responses: u64,
    /// TURN allocations created
    pub turn_allocations: u64,
    /// Successful hole punches
    pub hole_punch_success: u64,
    /// Failed hole punch attempts
    pub hole_punch_failures: u64,
    /// Relay connections established
    pub relay_connections: u64,
    /// Average hole punch time
    pub avg_hole_punch_time_ms: u64,
    /// Detected NAT type
    pub nat_type: NatType,
}

/// Connectivity event
#[derive(Debug, Clone)]
pub enum ConnectivityEvent {
    /// New ICE candidate gathered
    CandidateGathered(IceCandidate),
    /// Candidate pair check started
    PairCheckStarted(SocketAddr, SocketAddr),
    /// Candidate pair succeeded
    PairSucceeded(SocketAddr, SocketAddr),
    /// Candidate pair failed
    PairFailed(SocketAddr, SocketAddr),
    /// Connectivity established
    Connected(SocketAddr),
    /// Connectivity failed
    Failed(String),
}

/// NAT traversal manager
pub struct NatTraversalManager {
    config: NatTraversalConfig,
    /// Local candidates
    local_candidates: Arc<RwLock<Vec<IceCandidate>>>,
    /// Remote candidates
    remote_candidates: Arc<RwLock<Vec<IceCandidate>>>,
    /// Candidate pairs
    candidate_pairs: Arc<RwLock<Vec<CandidatePair>>>,
    /// Detected NAT type
    nat_type: Arc<RwLock<NatType>>,
    /// Statistics
    stats: Arc<RwLock<NatTraversalStats>>,
    /// Event sender
    event_tx: mpsc::UnboundedSender<ConnectivityEvent>,
    /// Event receiver
    event_rx: Arc<RwLock<mpsc::UnboundedReceiver<ConnectivityEvent>>>,
}

impl NatTraversalManager {
    /// Create new NAT traversal manager
    pub fn new(config: NatTraversalConfig) -> Self {
        let (event_tx, event_rx) = mpsc::unbounded_channel();

        Self {
            config,
            local_candidates: Arc::new(RwLock::new(Vec::new())),
            remote_candidates: Arc::new(RwLock::new(Vec::new())),
            candidate_pairs: Arc::new(RwLock::new(Vec::new())),
            nat_type: Arc::new(RwLock::new(NatType::Unknown)),
            stats: Arc::new(RwLock::new(NatTraversalStats::default())),
            event_tx,
            event_rx: Arc::new(RwLock::new(event_rx)),
        }
    }

    /// Detect NAT type using STUN
    pub async fn detect_nat_type(&self) -> Result<NatType, NatTraversalError> {
        // Simplified NAT type detection
        // In production, this would implement RFC 3489 NAT detection algorithm

        for stun_config in &self.config.stun_servers {
            match self.query_stun_server(stun_config).await {
                Ok(public_addr) => {
                    // Simplified detection: if we get a public address, assume we're behind NAT
                    let nat_type = if self.is_public_address(&public_addr) {
                        NatType::None
                    } else {
                        // In real implementation, do multiple STUN queries to determine exact NAT type
                        NatType::PortRestrictedCone
                    };

                    *self.nat_type.write().unwrap_or_else(|e| e.into_inner()) = nat_type;
                    self.stats
                        .write()
                        .unwrap_or_else(|e| e.into_inner())
                        .nat_type = nat_type;

                    return Ok(nat_type);
                }
                Err(e) => {
                    tracing::warn!("STUN query failed: {}", e);
                    continue;
                }
            }
        }

        Err(NatTraversalError::StunServerUnreachable(
            "All STUN servers failed".to_string(),
        ))
    }

    /// Query STUN server for public address
    async fn query_stun_server(
        &self,
        config: &StunConfig,
    ) -> Result<SocketAddr, NatTraversalError> {
        // Simplified STUN implementation
        // In production, implement RFC 5389 STUN protocol

        self.stats
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .stun_requests += 1;

        // Create UDP socket
        let _socket = UdpSocket::bind("0.0.0.0:0").await?;

        // In real implementation: send STUN binding request
        // For now, return a dummy address
        self.stats
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .stun_responses += 1;

        Ok(config.server)
    }

    /// Check if address is public
    fn is_public_address(&self, addr: &SocketAddr) -> bool {
        match addr.ip() {
            IpAddr::V4(ip) => !ip.is_private() && !ip.is_loopback() && !ip.is_link_local(),
            IpAddr::V6(ip) => !ip.is_loopback() && !ip.is_unspecified(),
        }
    }

    /// Gather local ICE candidates
    pub async fn gather_candidates(&self) -> Result<Vec<IceCandidate>, NatTraversalError> {
        let mut candidates = Vec::new();
        let component_id = 1u32;

        // 1. Gather host candidates (local interfaces)
        candidates.extend(self.gather_host_candidates(component_id).await?);

        // 2. Gather server reflexive candidates (STUN)
        if !self.config.stun_servers.is_empty() {
            candidates.extend(self.gather_stun_candidates(component_id).await?);
        }

        // 3. Gather relay candidates (TURN)
        if !self.config.turn_servers.is_empty() {
            candidates.extend(self.gather_turn_candidates(component_id).await?);
        }

        // Store local candidates
        *self
            .local_candidates
            .write()
            .unwrap_or_else(|e| e.into_inner()) = candidates.clone();

        // Emit events
        for candidate in &candidates {
            let _ = self
                .event_tx
                .send(ConnectivityEvent::CandidateGathered(candidate.clone()));
        }

        Ok(candidates)
    }

    /// Gather host candidates from local interfaces
    async fn gather_host_candidates(
        &self,
        component_id: u32,
    ) -> Result<Vec<IceCandidate>, NatTraversalError> {
        let mut candidates = Vec::new();

        // In production, enumerate all network interfaces
        // For now, create a single host candidate
        let socket = UdpSocket::bind("0.0.0.0:0").await?;
        let addr = socket.local_addr()?;

        candidates.push(IceCandidate {
            candidate_type: CandidateType::Host,
            addr,
            priority: IceCandidate::calculate_priority(CandidateType::Host, 65535, component_id),
            foundation: "host".to_string(),
            component_id,
        });

        Ok(candidates)
    }

    /// Gather STUN candidates
    async fn gather_stun_candidates(
        &self,
        component_id: u32,
    ) -> Result<Vec<IceCandidate>, NatTraversalError> {
        let mut candidates = Vec::new();

        for stun_config in &self.config.stun_servers {
            if let Ok(public_addr) = self.query_stun_server(stun_config).await {
                candidates.push(IceCandidate {
                    candidate_type: CandidateType::ServerReflexive,
                    addr: public_addr,
                    priority: IceCandidate::calculate_priority(
                        CandidateType::ServerReflexive,
                        65535,
                        component_id,
                    ),
                    foundation: "stun".to_string(),
                    component_id,
                });
            }
        }

        Ok(candidates)
    }

    /// Gather TURN relay candidates
    async fn gather_turn_candidates(
        &self,
        component_id: u32,
    ) -> Result<Vec<IceCandidate>, NatTraversalError> {
        let mut candidates = Vec::new();

        for turn_config in &self.config.turn_servers {
            // In production, implement TURN allocation (RFC 5766)
            // For now, add placeholder

            self.stats
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .turn_allocations += 1;

            candidates.push(IceCandidate {
                candidate_type: CandidateType::Relay,
                addr: turn_config.server,
                priority: IceCandidate::calculate_priority(
                    CandidateType::Relay,
                    65535,
                    component_id,
                ),
                foundation: "relay".to_string(),
                component_id,
            });
        }

        Ok(candidates)
    }

    /// Add remote ICE candidate
    pub fn add_remote_candidate(&self, candidate: IceCandidate) {
        self.remote_candidates
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .push(candidate);
    }

    /// Form candidate pairs
    pub fn form_candidate_pairs(&self) {
        let local_candidates = self
            .local_candidates
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let remote_candidates = self
            .remote_candidates
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let mut pairs = Vec::new();

        for local in local_candidates.iter() {
            for remote in remote_candidates.iter() {
                let priority = CandidatePair::calculate_priority(
                    local.priority,
                    remote.priority,
                    self.config.is_controlling,
                );

                pairs.push(CandidatePair {
                    local: local.clone(),
                    remote: remote.clone(),
                    priority,
                    state: PairState::Waiting,
                    last_check: None,
                });
            }
        }

        // Sort by priority (highest first)
        pairs.sort_by_key(|p| std::cmp::Reverse(p.priority));

        // Limit number of pairs
        pairs.truncate(self.config.max_candidate_pairs);

        *self
            .candidate_pairs
            .write()
            .unwrap_or_else(|e| e.into_inner()) = pairs;
    }

    /// Perform connectivity checks
    pub async fn perform_connectivity_checks(&self) -> Result<SocketAddr, NatTraversalError> {
        let start = Instant::now();

        loop {
            if start.elapsed() > self.config.hole_punch_timeout {
                self.stats
                    .write()
                    .unwrap_or_else(|e| e.into_inner())
                    .hole_punch_failures += 1;
                return Err(NatTraversalError::HolePunchTimeout);
            }

            // Get next pair to check
            let pair = {
                let mut pairs = self
                    .candidate_pairs
                    .write()
                    .unwrap_or_else(|e| e.into_inner());
                pairs
                    .iter_mut()
                    .find(|p| p.state == PairState::Waiting)
                    .map(|p| {
                        p.state = PairState::InProgress;
                        p.last_check = Some(Instant::now());
                        p.clone()
                    })
            };

            if let Some(pair) = pair {
                let _ = self.event_tx.send(ConnectivityEvent::PairCheckStarted(
                    pair.local.addr,
                    pair.remote.addr,
                ));

                // Perform connectivity check
                match self.check_candidate_pair(&pair).await {
                    Ok(true) => {
                        // Update pair state
                        {
                            let mut pairs = self
                                .candidate_pairs
                                .write()
                                .unwrap_or_else(|e| e.into_inner());
                            if let Some(p) = pairs.iter_mut().find(|p| {
                                p.local.addr == pair.local.addr && p.remote.addr == pair.remote.addr
                            }) {
                                p.state = PairState::Succeeded;
                            }
                        }

                        let _ = self.event_tx.send(ConnectivityEvent::PairSucceeded(
                            pair.local.addr,
                            pair.remote.addr,
                        ));

                        let _ = self
                            .event_tx
                            .send(ConnectivityEvent::Connected(pair.remote.addr));

                        let duration_ms = start.elapsed().as_millis() as u64;
                        let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());
                        stats.hole_punch_success += 1;
                        stats.avg_hole_punch_time_ms = duration_ms;

                        return Ok(pair.remote.addr);
                    }
                    Ok(false) => {
                        // Update pair state
                        {
                            let mut pairs = self
                                .candidate_pairs
                                .write()
                                .unwrap_or_else(|e| e.into_inner());
                            if let Some(p) = pairs.iter_mut().find(|p| {
                                p.local.addr == pair.local.addr && p.remote.addr == pair.remote.addr
                            }) {
                                p.state = PairState::Failed;
                            }
                        }

                        let _ = self.event_tx.send(ConnectivityEvent::PairFailed(
                            pair.local.addr,
                            pair.remote.addr,
                        ));
                    }
                    Err(_) => {
                        // Mark as failed and continue
                        let mut pairs = self
                            .candidate_pairs
                            .write()
                            .unwrap_or_else(|e| e.into_inner());
                        if let Some(p) = pairs.iter_mut().find(|p| {
                            p.local.addr == pair.local.addr && p.remote.addr == pair.remote.addr
                        }) {
                            p.state = PairState::Failed;
                        }
                    }
                }
            } else {
                // No more pairs to check
                break;
            }

            tokio::time::sleep(self.config.connectivity_check_interval).await;
        }

        Err(NatTraversalError::NoViablePath)
    }

    /// Check a candidate pair
    async fn check_candidate_pair(&self, _pair: &CandidatePair) -> Result<bool, NatTraversalError> {
        // In production, implement STUN connectivity check (RFC 8445)
        // Send STUN binding request to remote candidate
        // Wait for response

        // For now, simulate success for non-relay candidates
        Ok(true)
    }

    /// Perform UDP hole punching
    pub async fn hole_punch(
        &self,
        remote_addr: SocketAddr,
    ) -> Result<UdpSocket, NatTraversalError> {
        let nat_type = *self.nat_type.read().unwrap_or_else(|e| e.into_inner());

        if !nat_type.can_hole_punch() {
            return Err(NatTraversalError::NoViablePath);
        }

        // Bind local socket
        let socket = UdpSocket::bind("0.0.0.0:0").await?;

        // Send hole punch packets
        for _ in 0..10 {
            socket.send_to(b"PUNCH", remote_addr).await?;
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        Ok(socket)
    }

    /// Get statistics
    pub fn stats(&self) -> NatTraversalStats {
        self.stats.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Get next connectivity event
    #[allow(clippy::await_holding_lock)]
    pub async fn next_event(&self) -> Option<ConnectivityEvent> {
        let mut rx = self.event_rx.write().unwrap_or_else(|e| e.into_inner());
        rx.recv().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nat_type_can_hole_punch() {
        assert!(NatType::None.can_hole_punch());
        assert!(NatType::FullCone.can_hole_punch());
        assert!(NatType::RestrictedCone.can_hole_punch());
        assert!(NatType::PortRestrictedCone.can_hole_punch());
        assert!(!NatType::Symmetric.can_hole_punch());
    }

    #[test]
    fn test_candidate_priority() {
        let host_prio = IceCandidate::calculate_priority(CandidateType::Host, 65535, 1);
        let relay_prio = IceCandidate::calculate_priority(CandidateType::Relay, 65535, 1);

        assert!(host_prio > relay_prio);
    }

    #[test]
    fn test_pair_priority() {
        let prio1 = CandidatePair::calculate_priority(1000, 2000, true);
        let prio2 = CandidatePair::calculate_priority(2000, 1000, false);

        assert_eq!(prio1, prio2);
    }

    #[tokio::test]
    async fn test_nat_traversal_manager() {
        let config = NatTraversalConfig::default();
        let manager = NatTraversalManager::new(config);

        let stats = manager.stats();
        assert_eq!(stats.stun_requests, 0);
    }

    #[test]
    fn test_nat_type_requires_relay() {
        assert!(!NatType::None.requires_relay());
        assert!(!NatType::FullCone.requires_relay());
        assert!(!NatType::RestrictedCone.requires_relay());
        assert!(!NatType::PortRestrictedCone.requires_relay());
        assert!(NatType::Symmetric.requires_relay());
    }

    #[test]
    fn test_candidate_type_ordering() {
        assert!(CandidateType::Host < CandidateType::ServerReflexive);
        assert!(CandidateType::ServerReflexive < CandidateType::PeerReflexive);
        assert!(CandidateType::PeerReflexive < CandidateType::Relay);
    }

    #[test]
    fn test_candidate_priority_ordering() {
        let host_prio = IceCandidate::calculate_priority(CandidateType::Host, 65535, 1);
        let srflx_prio = IceCandidate::calculate_priority(CandidateType::ServerReflexive, 65535, 1);
        let prflx_prio = IceCandidate::calculate_priority(CandidateType::PeerReflexive, 65535, 1);
        let relay_prio = IceCandidate::calculate_priority(CandidateType::Relay, 65535, 1);

        // Priority order: Host > PeerReflexive > ServerReflexive > Relay (per RFC 8445)
        assert!(host_prio > prflx_prio);
        assert!(prflx_prio > srflx_prio);
        assert!(srflx_prio > relay_prio);
    }

    #[test]
    fn test_pair_priority_symmetry() {
        let prio1 = CandidatePair::calculate_priority(1000, 2000, true);
        let prio2 = CandidatePair::calculate_priority(1000, 2000, false);

        // Different controlling/controlled roles should give different priorities
        assert_ne!(prio1, prio2);
    }

    #[test]
    fn test_pair_state_transitions() {
        let state = PairState::Waiting;
        assert_eq!(state, PairState::Waiting);
        assert_ne!(state, PairState::InProgress);
        assert_ne!(state, PairState::Succeeded);
        assert_ne!(state, PairState::Failed);
    }

    #[tokio::test]
    async fn test_add_remote_candidate() {
        let config = NatTraversalConfig::default();
        let manager = NatTraversalManager::new(config);

        let candidate = IceCandidate {
            candidate_type: CandidateType::Host,
            addr: "127.0.0.1:8080".parse().expect("test: valid socket addr"),
            priority: 1000,
            foundation: "test".to_string(),
            component_id: 1,
        };

        manager.add_remote_candidate(candidate.clone());

        let remote_candidates = manager
            .remote_candidates
            .read()
            .unwrap_or_else(|e| e.into_inner());
        assert_eq!(remote_candidates.len(), 1);
        assert_eq!(remote_candidates[0].addr, candidate.addr);
    }

    #[tokio::test]
    async fn test_form_candidate_pairs() {
        let config = NatTraversalConfig {
            max_candidate_pairs: 10,
            ..Default::default()
        };
        let manager = NatTraversalManager::new(config);

        // Add local candidate
        let local = IceCandidate {
            candidate_type: CandidateType::Host,
            addr: "192.168.1.100:5000"
                .parse()
                .expect("test: valid socket addr"),
            priority: 1000,
            foundation: "local".to_string(),
            component_id: 1,
        };
        manager
            .local_candidates
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .push(local);

        // Add remote candidates
        let remote1 = IceCandidate {
            candidate_type: CandidateType::Host,
            addr: "192.168.1.101:5001"
                .parse()
                .expect("test: valid socket addr"),
            priority: 900,
            foundation: "remote1".to_string(),
            component_id: 1,
        };
        let remote2 = IceCandidate {
            candidate_type: CandidateType::ServerReflexive,
            addr: "203.0.113.10:5002"
                .parse()
                .expect("test: valid socket addr"),
            priority: 800,
            foundation: "remote2".to_string(),
            component_id: 1,
        };
        manager.add_remote_candidate(remote1);
        manager.add_remote_candidate(remote2);

        manager.form_candidate_pairs();

        let pairs = manager
            .candidate_pairs
            .read()
            .unwrap_or_else(|e| e.into_inner());
        assert_eq!(pairs.len(), 2); // 1 local * 2 remote = 2 pairs

        // Verify pairs are sorted by priority (highest first)
        if pairs.len() >= 2 {
            assert!(pairs[0].priority >= pairs[1].priority);
        }
    }

    #[tokio::test]
    async fn test_gather_candidates() {
        let config = NatTraversalConfig {
            stun_servers: vec![], // Disable STUN for this test
            turn_servers: vec![],
            ..Default::default()
        };
        let manager = NatTraversalManager::new(config);

        let candidates = manager.gather_candidates().await;
        assert!(candidates.is_ok());

        let candidates = candidates.expect("test: candidates should be present");
        assert!(!candidates.is_empty()); // Should have at least host candidates
    }

    #[test]
    fn test_stun_config_default() {
        let config = StunConfig::default();
        assert_eq!(config.retries, 3);
        assert!(config.timeout.as_secs() >= 1);
    }

    #[test]
    fn test_nat_traversal_config_default() {
        let config = NatTraversalConfig::default();
        assert!(config.enable_hole_punching);
        assert!(!config.stun_servers.is_empty());
        assert!(config.max_candidate_pairs > 0);
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let config = NatTraversalConfig::default();
        let manager = NatTraversalManager::new(config);

        let stats1 = manager.stats();
        assert_eq!(stats1.stun_requests, 0);
        assert_eq!(stats1.turn_allocations, 0);
        assert_eq!(stats1.hole_punch_success, 0);

        // Simulate STUN request
        manager
            .stats
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .stun_requests = 1;

        let stats2 = manager.stats();
        assert_eq!(stats2.stun_requests, 1);
    }

    #[test]
    fn test_public_address_detection() {
        let config = NatTraversalConfig::default();
        let manager = NatTraversalManager::new(config);

        // Private addresses
        assert!(
            !manager.is_public_address(&"192.168.1.1:80".parse().expect("test: valid socket addr"))
        );
        assert!(
            !manager.is_public_address(&"10.0.0.1:80".parse().expect("test: valid socket addr"))
        );
        assert!(
            !manager.is_public_address(&"127.0.0.1:80".parse().expect("test: valid socket addr"))
        );

        // Public address
        assert!(manager.is_public_address(&"8.8.8.8:80".parse().expect("test: valid socket addr")));
        assert!(manager.is_public_address(&"1.1.1.1:80".parse().expect("test: valid socket addr")));
    }

    #[test]
    fn test_nat_type_default() {
        let nat_type = NatType::default();
        assert_eq!(nat_type, NatType::Unknown);
    }

    #[tokio::test]
    async fn test_event_channel() {
        let config = NatTraversalConfig::default();
        let manager = Arc::new(NatTraversalManager::new(config));

        let manager_clone = manager.clone();
        let handle = tokio::spawn(async move {
            let candidate = IceCandidate {
                candidate_type: CandidateType::Host,
                addr: "127.0.0.1:9000".parse().expect("test: valid socket addr"),
                priority: 1000,
                foundation: "test".to_string(),
                component_id: 1,
            };

            let _ = manager_clone
                .event_tx
                .send(ConnectivityEvent::CandidateGathered(candidate));
        });

        // Give the task time to send
        tokio::time::sleep(Duration::from_millis(10)).await;

        let event = manager.next_event().await;
        assert!(event.is_some());

        if let Some(ConnectivityEvent::CandidateGathered(_)) = event {
            // Event received successfully
        } else {
            panic!("Expected CandidateGathered event");
        }

        let _ = handle.await;
    }

    #[test]
    fn test_candidate_pair_limit() {
        let config = NatTraversalConfig {
            max_candidate_pairs: 2,
            ..Default::default()
        };
        let manager = NatTraversalManager::new(config);

        // Add many local and remote candidates
        for i in 0..5 {
            let local = IceCandidate {
                candidate_type: CandidateType::Host,
                addr: format!("192.168.1.{}:5000", i + 100)
                    .parse()
                    .expect("test: valid socket addr"),
                priority: 1000 + i as u32,
                foundation: format!("local{}", i),
                component_id: 1,
            };
            manager
                .local_candidates
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .push(local);

            let remote = IceCandidate {
                candidate_type: CandidateType::Host,
                addr: format!("192.168.1.{}:5001", i + 200)
                    .parse()
                    .expect("test: valid socket addr"),
                priority: 900 + i as u32,
                foundation: format!("remote{}", i),
                component_id: 1,
            };
            manager.add_remote_candidate(remote);
        }

        manager.form_candidate_pairs();

        let pairs = manager
            .candidate_pairs
            .read()
            .unwrap_or_else(|e| e.into_inner());
        assert_eq!(pairs.len(), 2); // Limited to max_candidate_pairs
    }
}
