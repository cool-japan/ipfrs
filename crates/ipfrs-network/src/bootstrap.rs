//! Bootstrap peer management with retry logic
//!
//! This module handles:
//! - Bootstrap peer dialing with exponential backoff
//! - Connection retry logic
//! - Circuit breaker pattern for failing peers

use libp2p::{Multiaddr, PeerId};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Default bootstrap peers for IPFS network
pub const DEFAULT_IPFS_BOOTSTRAP_PEERS: &[&str] = &[
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN",
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmQCU2EcMqAqQPR2i9bChDtGNJchTbq5TbXJJ16u19uLTa",
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmbLHAnMoJPWSCR5Zhtx6BHJX9KiKNN6tpvbUcqanj75Nb",
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmcZf59bWwK5XFi76CZX8cbJ4BhTzzA3gU1ZjYZcYW3dwt",
];

/// Bootstrap configuration
#[derive(Debug, Clone)]
pub struct BootstrapConfig {
    /// Maximum retry attempts per peer
    pub max_retries: u32,
    /// Initial backoff duration
    pub initial_backoff: Duration,
    /// Maximum backoff duration
    pub max_backoff: Duration,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    /// Circuit breaker failure threshold
    pub circuit_breaker_threshold: u32,
    /// Circuit breaker reset timeout
    pub circuit_breaker_timeout: Duration,
    /// Periodic re-bootstrap interval
    pub re_bootstrap_interval: Duration,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            max_retries: 5,
            initial_backoff: Duration::from_secs(1),
            max_backoff: Duration::from_secs(60),
            backoff_multiplier: 2.0,
            circuit_breaker_threshold: 3,
            circuit_breaker_timeout: Duration::from_secs(300),
            re_bootstrap_interval: Duration::from_secs(300),
        }
    }
}

/// State of a bootstrap peer
#[derive(Debug, Clone)]
struct BootstrapPeerState {
    /// Multiaddress of the peer
    addr: Multiaddr,
    /// Number of connection attempts
    attempts: u32,
    /// Number of consecutive failures
    consecutive_failures: u32,
    /// Last attempt time
    last_attempt: Option<Instant>,
    /// Last successful connection
    last_success: Option<Instant>,
    /// Circuit breaker state
    circuit_open: bool,
    /// Circuit breaker opened at
    circuit_opened_at: Option<Instant>,
    /// Current backoff duration
    current_backoff: Duration,
}

impl BootstrapPeerState {
    fn new(addr: Multiaddr, initial_backoff: Duration) -> Self {
        Self {
            addr,
            attempts: 0,
            consecutive_failures: 0,
            last_attempt: None,
            last_success: None,
            circuit_open: false,
            circuit_opened_at: None,
            current_backoff: initial_backoff,
        }
    }

    fn should_retry(&self, config: &BootstrapConfig) -> bool {
        // Check circuit breaker
        if self.circuit_open {
            if let Some(opened_at) = self.circuit_opened_at {
                if opened_at.elapsed() < config.circuit_breaker_timeout {
                    return false;
                }
            }
        }

        // Check if we've exceeded max retries
        if self.consecutive_failures >= config.max_retries {
            return false;
        }

        // Check backoff
        if let Some(last) = self.last_attempt {
            if last.elapsed() < self.current_backoff {
                return false;
            }
        }

        true
    }

    fn record_attempt(&mut self) {
        self.attempts += 1;
        self.last_attempt = Some(Instant::now());
    }

    fn record_success(&mut self, initial_backoff: Duration) {
        self.consecutive_failures = 0;
        self.last_success = Some(Instant::now());
        self.circuit_open = false;
        self.circuit_opened_at = None;
        self.current_backoff = initial_backoff;
    }

    fn record_failure(&mut self, config: &BootstrapConfig) {
        self.consecutive_failures += 1;

        // Increase backoff with exponential growth
        let new_backoff =
            Duration::from_secs_f64(self.current_backoff.as_secs_f64() * config.backoff_multiplier);
        self.current_backoff = new_backoff.min(config.max_backoff);

        // Check if we should open circuit breaker
        if self.consecutive_failures >= config.circuit_breaker_threshold {
            self.circuit_open = true;
            self.circuit_opened_at = Some(Instant::now());
            warn!(
                "Circuit breaker opened for peer {} after {} failures",
                self.addr, self.consecutive_failures
            );
        }
    }
}

/// Bootstrap peer manager
pub struct BootstrapManager {
    /// Configuration
    config: BootstrapConfig,
    /// Bootstrap peer states
    peers: Arc<RwLock<HashMap<String, BootstrapPeerState>>>,
    /// Successfully connected peers
    connected: Arc<RwLock<Vec<PeerId>>>,
}

impl BootstrapManager {
    /// Create a new bootstrap manager
    pub fn new(config: BootstrapConfig) -> Self {
        Self {
            config,
            peers: Arc::new(RwLock::new(HashMap::new())),
            connected: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Add a bootstrap peer
    pub fn add_peer(&self, addr: Multiaddr) {
        let key = addr.to_string();
        let mut peers = self.peers.write();
        peers
            .entry(key)
            .or_insert_with(|| BootstrapPeerState::new(addr, self.config.initial_backoff));
    }

    /// Add multiple bootstrap peers from strings
    pub fn add_peers_from_strings(&self, addrs: &[String]) {
        for addr_str in addrs {
            if let Ok(addr) = addr_str.parse::<Multiaddr>() {
                self.add_peer(addr);
            } else {
                warn!("Invalid bootstrap peer address: {}", addr_str);
            }
        }
    }

    /// Add default IPFS bootstrap peers
    pub fn add_default_peers(&self) {
        for addr_str in DEFAULT_IPFS_BOOTSTRAP_PEERS {
            if let Ok(addr) = addr_str.parse::<Multiaddr>() {
                self.add_peer(addr);
            }
        }
        info!(
            "Added {} default IPFS bootstrap peers",
            DEFAULT_IPFS_BOOTSTRAP_PEERS.len()
        );
    }

    /// Get peers that should be dialed
    pub fn get_peers_to_dial(&self) -> Vec<Multiaddr> {
        let peers = self.peers.read();
        peers
            .values()
            .filter(|state| state.should_retry(&self.config))
            .map(|state| state.addr.clone())
            .collect()
    }

    /// Record a dial attempt
    pub fn record_dial_attempt(&self, addr: &Multiaddr) {
        let key = addr.to_string();
        let mut peers = self.peers.write();
        if let Some(state) = peers.get_mut(&key) {
            state.record_attempt();
            debug!("Recorded dial attempt for {}", addr);
        }
    }

    /// Record a successful connection
    pub fn record_connection_success(&self, addr: &Multiaddr, peer_id: PeerId) {
        let key = addr.to_string();
        let mut peers = self.peers.write();
        if let Some(state) = peers.get_mut(&key) {
            state.record_success(self.config.initial_backoff);
            info!(
                "Successfully connected to bootstrap peer {} ({})",
                addr, peer_id
            );
        }

        // Track connected peer
        let mut connected = self.connected.write();
        if !connected.contains(&peer_id) {
            connected.push(peer_id);
        }
    }

    /// Record a connection failure
    pub fn record_connection_failure(&self, addr: &Multiaddr) {
        let key = addr.to_string();
        let mut peers = self.peers.write();
        if let Some(state) = peers.get_mut(&key) {
            state.record_failure(&self.config);
            warn!(
                "Failed to connect to bootstrap peer {}, backoff: {:?}",
                addr, state.current_backoff
            );
        }
    }

    /// Record peer disconnection
    pub fn record_disconnection(&self, peer_id: &PeerId) {
        let mut connected = self.connected.write();
        connected.retain(|p| p != peer_id);
    }

    /// Check if we have enough bootstrap connections
    pub fn has_sufficient_connections(&self, min_peers: usize) -> bool {
        self.connected.read().len() >= min_peers
    }

    /// Get number of connected bootstrap peers
    pub fn connected_count(&self) -> usize {
        self.connected.read().len()
    }

    /// Get bootstrap statistics
    pub fn stats(&self) -> BootstrapStats {
        let peers = self.peers.read();
        let connected = self.connected.read();

        let total_attempts: u32 = peers.values().map(|s| s.attempts).sum();
        let total_failures: u32 = peers.values().map(|s| s.consecutive_failures).sum();
        let open_circuits = peers.values().filter(|s| s.circuit_open).count();

        BootstrapStats {
            total_peers: peers.len(),
            connected_peers: connected.len(),
            total_attempts,
            total_failures,
            open_circuits,
        }
    }

    /// Reset a peer's circuit breaker (for manual intervention)
    pub fn reset_circuit_breaker(&self, addr: &Multiaddr) {
        let key = addr.to_string();
        let mut peers = self.peers.write();
        if let Some(state) = peers.get_mut(&key) {
            state.circuit_open = false;
            state.circuit_opened_at = None;
            state.consecutive_failures = 0;
            state.current_backoff = self.config.initial_backoff;
            info!("Reset circuit breaker for {}", addr);
        }
    }

    /// Reset all circuit breakers
    pub fn reset_all_circuit_breakers(&self) {
        let mut peers = self.peers.write();
        for state in peers.values_mut() {
            state.circuit_open = false;
            state.circuit_opened_at = None;
            state.consecutive_failures = 0;
            state.current_backoff = self.config.initial_backoff;
        }
        info!("Reset all circuit breakers");
    }
}

impl Default for BootstrapManager {
    fn default() -> Self {
        Self::new(BootstrapConfig::default())
    }
}

/// Bootstrap statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct BootstrapStats {
    /// Total number of bootstrap peers
    pub total_peers: usize,
    /// Number of connected bootstrap peers
    pub connected_peers: usize,
    /// Total connection attempts
    pub total_attempts: u32,
    /// Total failures
    pub total_failures: u32,
    /// Number of open circuit breakers
    pub open_circuits: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_config_default() {
        let config = BootstrapConfig::default();
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.initial_backoff, Duration::from_secs(1));
    }

    #[test]
    fn test_bootstrap_manager_add_peer() {
        let manager = BootstrapManager::default();
        let addr: Multiaddr = "/ip4/127.0.0.1/tcp/4001"
            .parse()
            .expect("test: valid multiaddr should parse");

        manager.add_peer(addr.clone());
        let peers = manager.get_peers_to_dial();
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0], addr);
    }

    #[test]
    fn test_bootstrap_manager_backoff() {
        let config = BootstrapConfig {
            initial_backoff: Duration::from_millis(10),
            max_backoff: Duration::from_secs(1),
            backoff_multiplier: 2.0,
            ..Default::default()
        };
        let manager = BootstrapManager::new(config);
        let addr: Multiaddr = "/ip4/127.0.0.1/tcp/4001"
            .parse()
            .expect("test: valid multiaddr should parse");

        manager.add_peer(addr.clone());
        manager.record_dial_attempt(&addr);
        manager.record_connection_failure(&addr);

        // Should not retry immediately due to backoff
        let peers = manager.get_peers_to_dial();
        assert!(peers.is_empty());

        // Wait for backoff
        std::thread::sleep(Duration::from_millis(25));
        let peers = manager.get_peers_to_dial();
        assert_eq!(peers.len(), 1);
    }

    #[test]
    fn test_bootstrap_manager_circuit_breaker() {
        let config = BootstrapConfig {
            initial_backoff: Duration::from_millis(1),
            circuit_breaker_threshold: 2,
            circuit_breaker_timeout: Duration::from_secs(1),
            ..Default::default()
        };
        let manager = BootstrapManager::new(config);
        let addr: Multiaddr = "/ip4/127.0.0.1/tcp/4001"
            .parse()
            .expect("test: valid multiaddr should parse");

        manager.add_peer(addr.clone());

        // First failure
        manager.record_dial_attempt(&addr);
        manager.record_connection_failure(&addr);
        std::thread::sleep(Duration::from_millis(5));

        // Second failure - circuit should open
        manager.record_dial_attempt(&addr);
        manager.record_connection_failure(&addr);

        // Should not retry with open circuit
        let peers = manager.get_peers_to_dial();
        assert!(peers.is_empty());

        // Check stats
        let stats = manager.stats();
        assert_eq!(stats.open_circuits, 1);
    }

    #[test]
    fn test_bootstrap_manager_success_resets() {
        let config = BootstrapConfig {
            initial_backoff: Duration::from_millis(1),
            ..Default::default()
        };
        let manager = BootstrapManager::new(config);
        let addr: Multiaddr = "/ip4/127.0.0.1/tcp/4001"
            .parse()
            .expect("test: valid multiaddr should parse");
        let peer_id = PeerId::random();

        manager.add_peer(addr.clone());
        manager.record_dial_attempt(&addr);
        manager.record_connection_failure(&addr);

        // Success should reset
        manager.record_connection_success(&addr, peer_id);

        assert!(manager.has_sufficient_connections(1));
        assert_eq!(manager.connected_count(), 1);
    }
}
