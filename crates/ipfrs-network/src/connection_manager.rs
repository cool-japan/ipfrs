//! Connection management with limits and pruning
//!
//! This module implements intelligent connection management:
//! - Connection limits (total, inbound, outbound)
//! - Priority-based connection scoring
//! - Automatic pruning of low-value connections
//! - Reserved slots for important peers

use libp2p::PeerId;
use parking_lot::RwLock;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Connection manager configuration
#[derive(Debug, Clone)]
pub struct ConnectionLimitsConfig {
    /// Maximum total connections
    pub max_connections: usize,
    /// Maximum inbound connections
    pub max_inbound: usize,
    /// Maximum outbound connections
    pub max_outbound: usize,
    /// Reserved slots for important peers
    pub reserved_slots: usize,
    /// Connection idle timeout
    pub idle_timeout: Duration,
    /// Minimum score to avoid pruning (0-100)
    pub min_score_threshold: u8,
}

impl Default for ConnectionLimitsConfig {
    fn default() -> Self {
        Self {
            max_connections: 256,
            max_inbound: 128,
            max_outbound: 128,
            reserved_slots: 8,
            idle_timeout: Duration::from_secs(300),
            min_score_threshold: 30,
        }
    }
}

/// Connection direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionDirection {
    Inbound,
    Outbound,
}

/// Information about a connection
#[derive(Debug, Clone)]
struct ConnectionInfo {
    /// Peer ID
    peer_id: PeerId,
    /// Connection direction
    direction: ConnectionDirection,
    /// Time when connection was established
    established_at: Instant,
    /// Last activity time
    last_activity: Instant,
    /// Connection score (0-100)
    score: u8,
    /// Whether this peer has reserved slot
    reserved: bool,
    /// Number of messages sent
    messages_sent: u64,
    /// Number of messages received
    messages_received: u64,
    /// Average latency if known
    avg_latency_ms: Option<u64>,
}

impl ConnectionInfo {
    fn new(peer_id: PeerId, direction: ConnectionDirection) -> Self {
        let now = Instant::now();
        Self {
            peer_id,
            direction,
            established_at: now,
            last_activity: now,
            score: 50, // Start neutral
            reserved: false,
            messages_sent: 0,
            messages_received: 0,
            avg_latency_ms: None,
        }
    }

    fn is_idle(&self, timeout: Duration) -> bool {
        self.last_activity.elapsed() > timeout
    }

    fn touch(&mut self) {
        self.last_activity = Instant::now();
    }

    /// Calculate connection value for pruning decisions
    fn calculate_value(&self) -> u64 {
        let age_secs = self.established_at.elapsed().as_secs();
        let activity = self.messages_sent + self.messages_received;

        // Value = score * 10 + activity_rate + latency_bonus
        let base_value = self.score as u64 * 10;
        let activity_rate = (activity * 60)
            .checked_div(age_secs)
            .unwrap_or(activity * 60); // messages per minute
        let latency_bonus = match self.avg_latency_ms {
            Some(lat) if lat < 50 => 20,
            Some(lat) if lat < 100 => 10,
            Some(lat) if lat < 200 => 5,
            _ => 0,
        };

        base_value + activity_rate + latency_bonus
    }
}

/// Connection manager
pub struct ConnectionManager {
    /// Configuration
    config: ConnectionLimitsConfig,
    /// Active connections
    connections: RwLock<HashMap<PeerId, ConnectionInfo>>,
    /// Reserved peers (always allowed to connect)
    reserved_peers: RwLock<HashSet<PeerId>>,
    /// Banned peers (never allowed to connect)
    banned_peers: RwLock<HashSet<PeerId>>,
}

impl ConnectionManager {
    /// Create a new connection manager
    pub fn new(config: ConnectionLimitsConfig) -> Self {
        Self {
            config,
            connections: RwLock::new(HashMap::new()),
            reserved_peers: RwLock::new(HashSet::new()),
            banned_peers: RwLock::new(HashSet::new()),
        }
    }

    /// Check if a new connection should be accepted
    pub fn should_accept(&self, peer_id: &PeerId, direction: ConnectionDirection) -> bool {
        // Always reject banned peers
        if self.banned_peers.read().contains(peer_id) {
            debug!("Rejecting banned peer: {}", peer_id);
            return false;
        }

        // Always accept reserved peers (up to reserved slots limit)
        if self.reserved_peers.read().contains(peer_id) {
            let reserved_count = self
                .connections
                .read()
                .values()
                .filter(|c| c.reserved)
                .count();
            if reserved_count < self.config.reserved_slots {
                return true;
            }
        }

        let connections = self.connections.read();

        // Check total limit
        if connections.len() >= self.config.max_connections {
            debug!(
                "At max connections ({}), rejecting {}",
                self.config.max_connections, peer_id
            );
            return false;
        }

        // Check direction-specific limits
        let (inbound, outbound) =
            connections
                .values()
                .fold((0, 0), |(i, o), c| match c.direction {
                    ConnectionDirection::Inbound => (i + 1, o),
                    ConnectionDirection::Outbound => (i, o + 1),
                });

        match direction {
            ConnectionDirection::Inbound => {
                if inbound >= self.config.max_inbound {
                    debug!(
                        "At max inbound ({}), rejecting {}",
                        self.config.max_inbound, peer_id
                    );
                    return false;
                }
            }
            ConnectionDirection::Outbound => {
                if outbound >= self.config.max_outbound {
                    debug!(
                        "At max outbound ({}), rejecting {}",
                        self.config.max_outbound, peer_id
                    );
                    return false;
                }
            }
        }

        true
    }

    /// Register a new connection
    pub fn connection_established(&self, peer_id: PeerId, direction: ConnectionDirection) {
        let is_reserved = self.reserved_peers.read().contains(&peer_id);

        let mut connections = self.connections.write();
        let mut info = ConnectionInfo::new(peer_id, direction);
        info.reserved = is_reserved;

        connections.insert(peer_id, info);
        info!("Connection established: {} ({:?})", peer_id, direction);
    }

    /// Unregister a connection
    pub fn connection_closed(&self, peer_id: &PeerId) {
        let mut connections = self.connections.write();
        if connections.remove(peer_id).is_some() {
            debug!("Connection closed: {}", peer_id);
        }
    }

    /// Record activity on a connection
    pub fn record_activity(&self, peer_id: &PeerId, sent: bool) {
        let mut connections = self.connections.write();
        if let Some(info) = connections.get_mut(peer_id) {
            info.touch();
            if sent {
                info.messages_sent += 1;
            } else {
                info.messages_received += 1;
            }
        }
    }

    /// Update connection score
    pub fn update_score(&self, peer_id: &PeerId, delta: i16) {
        let mut connections = self.connections.write();
        if let Some(info) = connections.get_mut(peer_id) {
            let new_score = (info.score as i16 + delta).clamp(0, 100) as u8;
            info.score = new_score;
        }
    }

    /// Update connection latency
    pub fn update_latency(&self, peer_id: &PeerId, latency_ms: u64) {
        let mut connections = self.connections.write();
        if let Some(info) = connections.get_mut(peer_id) {
            info.avg_latency_ms = Some(latency_ms);
            info.touch();
        }
    }

    /// Add a peer to reserved list
    pub fn add_reserved(&self, peer_id: PeerId) {
        self.reserved_peers.write().insert(peer_id);

        // Update connection if exists
        if let Some(info) = self.connections.write().get_mut(&peer_id) {
            info.reserved = true;
        }

        info!("Added reserved peer: {}", peer_id);
    }

    /// Remove a peer from reserved list
    pub fn remove_reserved(&self, peer_id: &PeerId) {
        self.reserved_peers.write().remove(peer_id);

        // Update connection if exists
        if let Some(info) = self.connections.write().get_mut(peer_id) {
            info.reserved = false;
        }

        debug!("Removed reserved peer: {}", peer_id);
    }

    /// Ban a peer
    pub fn ban_peer(&self, peer_id: PeerId) {
        self.banned_peers.write().insert(peer_id);
        self.reserved_peers.write().remove(&peer_id);
        warn!("Banned peer: {}", peer_id);
    }

    /// Unban a peer
    pub fn unban_peer(&self, peer_id: &PeerId) {
        self.banned_peers.write().remove(peer_id);
        info!("Unbanned peer: {}", peer_id);
    }

    /// Check if a peer is banned
    pub fn is_banned(&self, peer_id: &PeerId) -> bool {
        self.banned_peers.read().contains(peer_id)
    }

    /// Get peers that should be disconnected (pruning candidates)
    pub fn get_prune_candidates(&self, count: usize) -> Vec<PeerId> {
        let connections = self.connections.read();

        // Filter out reserved peers and those above threshold
        let mut candidates: Vec<_> = connections
            .values()
            .filter(|c| !c.reserved && c.score < self.config.min_score_threshold)
            .map(|c| (c.peer_id, c.calculate_value()))
            .collect();

        // Sort by value (lowest first)
        candidates.sort_by_key(|(_, value)| *value);

        candidates
            .into_iter()
            .take(count)
            .map(|(peer_id, _)| peer_id)
            .collect()
    }

    /// Get idle connections that should be closed
    pub fn get_idle_connections(&self) -> Vec<PeerId> {
        let connections = self.connections.read();
        let timeout = self.config.idle_timeout;

        connections
            .values()
            .filter(|c| !c.reserved && c.is_idle(timeout))
            .map(|c| c.peer_id)
            .collect()
    }

    /// Prune connections to make room for new ones
    ///
    /// Returns peer IDs that should be disconnected
    pub fn prune_to_limit(&self) -> Vec<PeerId> {
        let connections = self.connections.read();
        let current = connections.len();

        if current <= self.config.max_connections {
            return vec![];
        }

        let to_prune = current - self.config.max_connections;
        drop(connections);

        let candidates = self.get_prune_candidates(to_prune);
        info!(
            "Pruning {} connections to stay within limit",
            candidates.len()
        );
        candidates
    }

    /// Get all connected peer IDs
    pub fn connected_peers(&self) -> Vec<PeerId> {
        self.connections.read().keys().cloned().collect()
    }

    /// Get connection count
    pub fn connection_count(&self) -> usize {
        self.connections.read().len()
    }

    /// Check if connected to a peer
    pub fn is_connected(&self, peer_id: &PeerId) -> bool {
        self.connections.read().contains_key(peer_id)
    }

    /// Get connection statistics
    pub fn stats(&self) -> ConnectionManagerStats {
        let connections = self.connections.read();

        let (inbound, outbound) =
            connections
                .values()
                .fold((0, 0), |(i, o), c| match c.direction {
                    ConnectionDirection::Inbound => (i + 1, o),
                    ConnectionDirection::Outbound => (i, o + 1),
                });

        let reserved = connections.values().filter(|c| c.reserved).count();

        let avg_score = if connections.is_empty() {
            0
        } else {
            connections.values().map(|c| c.score as u64).sum::<u64>() / connections.len() as u64
        };

        ConnectionManagerStats {
            total_connections: connections.len(),
            max_connections: self.config.max_connections,
            inbound_connections: inbound,
            outbound_connections: outbound,
            reserved_connections: reserved,
            banned_peers: self.banned_peers.read().len(),
            average_score: avg_score as u8,
        }
    }
}

impl Default for ConnectionManager {
    fn default() -> Self {
        Self::new(ConnectionLimitsConfig::default())
    }
}

/// Connection manager statistics
#[derive(Debug, Clone, Serialize)]
pub struct ConnectionManagerStats {
    /// Total active connections
    pub total_connections: usize,
    /// Maximum connections allowed
    pub max_connections: usize,
    /// Inbound connection count
    pub inbound_connections: usize,
    /// Outbound connection count
    pub outbound_connections: usize,
    /// Reserved connection count
    pub reserved_connections: usize,
    /// Number of banned peers
    pub banned_peers: usize,
    /// Average connection score
    pub average_score: u8,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_peer() -> PeerId {
        PeerId::random()
    }

    #[test]
    fn test_connection_manager_basic() {
        let manager = ConnectionManager::default();
        let peer1 = random_peer();
        let peer2 = random_peer();

        assert!(manager.should_accept(&peer1, ConnectionDirection::Inbound));

        manager.connection_established(peer1, ConnectionDirection::Inbound);
        assert!(manager.is_connected(&peer1));
        assert_eq!(manager.connection_count(), 1);

        manager.connection_established(peer2, ConnectionDirection::Outbound);
        assert_eq!(manager.connection_count(), 2);

        manager.connection_closed(&peer1);
        assert!(!manager.is_connected(&peer1));
        assert_eq!(manager.connection_count(), 1);
    }

    #[test]
    fn test_connection_limits() {
        let config = ConnectionLimitsConfig {
            max_connections: 3,
            max_inbound: 2,
            max_outbound: 2,
            ..Default::default()
        };
        let manager = ConnectionManager::new(config);

        // Fill up inbound
        let peer1 = random_peer();
        let peer2 = random_peer();
        manager.connection_established(peer1, ConnectionDirection::Inbound);
        manager.connection_established(peer2, ConnectionDirection::Inbound);

        // Should reject new inbound
        let peer3 = random_peer();
        assert!(!manager.should_accept(&peer3, ConnectionDirection::Inbound));

        // But outbound should be ok
        assert!(manager.should_accept(&peer3, ConnectionDirection::Outbound));
        manager.connection_established(peer3, ConnectionDirection::Outbound);

        // Now at max total, should reject all
        let peer4 = random_peer();
        assert!(!manager.should_accept(&peer4, ConnectionDirection::Inbound));
        assert!(!manager.should_accept(&peer4, ConnectionDirection::Outbound));
    }

    #[test]
    fn test_reserved_peers() {
        let config = ConnectionLimitsConfig {
            max_connections: 2,
            reserved_slots: 1,
            ..Default::default()
        };
        let manager = ConnectionManager::new(config);

        let reserved_peer = random_peer();
        manager.add_reserved(reserved_peer);

        let peer1 = random_peer();
        let peer2 = random_peer();
        manager.connection_established(peer1, ConnectionDirection::Inbound);
        manager.connection_established(peer2, ConnectionDirection::Outbound);

        // At max, but reserved peer should be accepted
        assert!(manager.should_accept(&reserved_peer, ConnectionDirection::Inbound));
    }

    #[test]
    fn test_banned_peers() {
        let manager = ConnectionManager::default();
        let peer = random_peer();

        assert!(manager.should_accept(&peer, ConnectionDirection::Inbound));

        manager.ban_peer(peer);
        assert!(manager.is_banned(&peer));
        assert!(!manager.should_accept(&peer, ConnectionDirection::Inbound));

        manager.unban_peer(&peer);
        assert!(!manager.is_banned(&peer));
        assert!(manager.should_accept(&peer, ConnectionDirection::Inbound));
    }

    #[test]
    fn test_activity_tracking() {
        let manager = ConnectionManager::default();
        let peer = random_peer();

        manager.connection_established(peer, ConnectionDirection::Outbound);

        // Record some activity
        manager.record_activity(&peer, true); // sent
        manager.record_activity(&peer, false); // received
        manager.record_activity(&peer, true); // sent

        let stats = manager.stats();
        assert_eq!(stats.total_connections, 1);
    }

    #[test]
    fn test_score_update() {
        let manager = ConnectionManager::default();
        let peer = random_peer();

        manager.connection_established(peer, ConnectionDirection::Inbound);
        manager.update_score(&peer, 20); // 50 + 20 = 70
        manager.update_score(&peer, -40); // 70 - 40 = 30

        // Score should be clamped
        manager.update_score(&peer, -100); // Should clamp to 0
    }

    #[test]
    fn test_prune_candidates() {
        let config = ConnectionLimitsConfig {
            min_score_threshold: 50,
            ..Default::default()
        };
        let manager = ConnectionManager::new(config);

        // Add some peers
        let high_score = random_peer();
        let low_score1 = random_peer();
        let low_score2 = random_peer();
        let reserved = random_peer();

        manager.connection_established(high_score, ConnectionDirection::Inbound);
        manager.connection_established(low_score1, ConnectionDirection::Inbound);
        manager.connection_established(low_score2, ConnectionDirection::Outbound);
        manager.add_reserved(reserved);
        manager.connection_established(reserved, ConnectionDirection::Inbound);

        // Adjust scores
        manager.update_score(&high_score, 30); // 80
        manager.update_score(&low_score1, -30); // 20
        manager.update_score(&low_score2, -25); // 25

        // Get prune candidates
        let candidates = manager.get_prune_candidates(2);

        // Should include low score peers but not reserved
        assert!(!candidates.contains(&reserved));
        assert!(!candidates.contains(&high_score));
        assert!(candidates.len() <= 2);
    }

    #[test]
    fn test_idle_connections() {
        let config = ConnectionLimitsConfig {
            idle_timeout: Duration::from_millis(50),
            ..Default::default()
        };
        let manager = ConnectionManager::new(config);

        let peer = random_peer();
        manager.connection_established(peer, ConnectionDirection::Inbound);

        // Not idle yet
        assert!(manager.get_idle_connections().is_empty());

        // Wait for timeout
        std::thread::sleep(Duration::from_millis(100));

        // Should be idle now
        let idle = manager.get_idle_connections();
        assert_eq!(idle.len(), 1);
        assert_eq!(idle[0], peer);
    }

    #[test]
    fn test_stats() {
        let manager = ConnectionManager::default();

        let peer1 = random_peer();
        let peer2 = random_peer();
        let reserved = random_peer();

        manager.connection_established(peer1, ConnectionDirection::Inbound);
        manager.connection_established(peer2, ConnectionDirection::Outbound);
        manager.add_reserved(reserved);
        manager.connection_established(reserved, ConnectionDirection::Inbound);

        let banned = random_peer();
        manager.ban_peer(banned);

        let stats = manager.stats();
        assert_eq!(stats.total_connections, 3);
        assert_eq!(stats.inbound_connections, 2);
        assert_eq!(stats.outbound_connections, 1);
        assert_eq!(stats.reserved_connections, 1);
        assert_eq!(stats.banned_peers, 1);
    }
}
