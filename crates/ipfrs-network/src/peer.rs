//! Peer information and management
//!
//! This module provides peer tracking, storage, and management functionality:
//! - PeerInfo: Information about a single peer
//! - PeerStore: In-memory database of known peers
//! - Connection tracking and history
//! - Peer persistence (save/load to disk)

use dashmap::DashMap;
use libp2p::{Multiaddr, PeerId};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Information about a peer in the network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    /// Peer ID
    pub peer_id: String,
    /// Multiaddresses
    pub addrs: Vec<String>,
    /// Protocol versions supported
    pub protocols: Vec<String>,
    /// Agent version (from identify)
    pub agent_version: Option<String>,
    /// Protocol version (from identify)
    pub protocol_version: Option<String>,
    /// Last seen timestamp (unix timestamp)
    pub last_seen: u64,
    /// Connection count
    pub connection_count: u64,
    /// Average latency in milliseconds
    pub avg_latency_ms: Option<u64>,
    /// Peer reputation score (0-100)
    pub reputation: u8,
}

impl PeerInfo {
    pub fn new(peer_id: String) -> Self {
        Self {
            peer_id,
            addrs: vec![],
            protocols: vec![],
            agent_version: None,
            protocol_version: None,
            last_seen: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            connection_count: 0,
            avg_latency_ms: None,
            reputation: 50, // Start neutral
        }
    }
}

/// Internal peer record with runtime information
#[derive(Debug)]
struct PeerRecord {
    /// Basic peer info (serializable)
    info: PeerInfo,
    /// Multiaddresses (runtime)
    addrs: HashSet<Multiaddr>,
    /// Currently connected
    connected: bool,
    /// Connection established time
    connected_at: Option<Instant>,
    /// Latency samples for averaging
    latency_samples: Vec<Duration>,
}

impl PeerRecord {
    fn new(peer_id: PeerId) -> Self {
        Self {
            info: PeerInfo::new(peer_id.to_string()),
            addrs: HashSet::new(),
            connected: false,
            connected_at: None,
            latency_samples: Vec::new(), // Don't pre-allocate, let it grow as needed
        }
    }

    fn update_latency(&mut self, rtt: Duration, max_samples: usize) {
        // Keep last N samples (configurable for memory optimization)
        if self.latency_samples.len() >= max_samples {
            self.latency_samples.remove(0);
        }
        self.latency_samples.push(rtt);

        // Calculate average
        let total: Duration = self.latency_samples.iter().sum();
        let avg = total.as_millis() as u64 / self.latency_samples.len() as u64;
        self.info.avg_latency_ms = Some(avg);
    }

    fn touch(&mut self) {
        self.info.last_seen = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }
}

/// Peer store configuration
#[derive(Debug, Clone)]
pub struct PeerStoreConfig {
    /// Maximum number of peers to store
    pub max_peers: usize,
    /// Maximum addresses to store per peer
    pub max_addrs_per_peer: usize,
    /// Maximum latency samples to keep
    pub max_latency_samples: usize,
    /// Maximum protocols to store per peer
    pub max_protocols_per_peer: usize,
}

impl Default for PeerStoreConfig {
    fn default() -> Self {
        Self {
            max_peers: 1000,
            max_addrs_per_peer: 10,
            max_latency_samples: 10,
            max_protocols_per_peer: 20,
        }
    }
}

impl PeerStoreConfig {
    /// Low-memory configuration for constrained devices
    pub fn low_memory() -> Self {
        Self {
            max_peers: 100,            // Very limited peer count
            max_addrs_per_peer: 2,     // Only keep best addresses
            max_latency_samples: 3,    // Minimal history
            max_protocols_per_peer: 5, // Limit protocol list
        }
    }

    /// IoT device configuration
    pub fn iot() -> Self {
        Self {
            max_peers: 200,
            max_addrs_per_peer: 3,
            max_latency_samples: 5,
            max_protocols_per_peer: 10,
        }
    }

    /// Mobile device configuration
    pub fn mobile() -> Self {
        Self {
            max_peers: 500,
            max_addrs_per_peer: 5,
            max_latency_samples: 8,
            max_protocols_per_peer: 15,
        }
    }

    /// Server configuration with larger limits
    pub fn server() -> Self {
        Self {
            max_peers: 5000,
            max_addrs_per_peer: 20,
            max_latency_samples: 20,
            max_protocols_per_peer: 50,
        }
    }
}

/// Peer store for managing known peers
pub struct PeerStore {
    /// Known peers indexed by PeerId
    peers: DashMap<PeerId, PeerRecord>,
    /// Connected peers (subset of known peers)
    connected_peers: Arc<RwLock<HashSet<PeerId>>>,
    /// Configuration
    config: PeerStoreConfig,
}

impl PeerStore {
    /// Create a new peer store
    pub fn new(max_peers: usize) -> Self {
        Self::with_config(PeerStoreConfig {
            max_peers,
            ..Default::default()
        })
    }

    /// Create a new peer store with configuration
    pub fn with_config(config: PeerStoreConfig) -> Self {
        Self {
            peers: DashMap::new(),
            connected_peers: Arc::new(RwLock::new(HashSet::new())),
            config,
        }
    }

    /// Get peer store configuration
    pub fn config(&self) -> &PeerStoreConfig {
        &self.config
    }

    /// Add or update a peer with addresses
    pub fn add_peer(&self, peer_id: PeerId, addrs: Vec<Multiaddr>) {
        // Use a block to release the entry guard before calling maybe_prune
        {
            let mut entry = self
                .peers
                .entry(peer_id)
                .or_insert_with(|| PeerRecord::new(peer_id));

            // Enforce address limit
            for addr in addrs {
                if entry.addrs.len() >= self.config.max_addrs_per_peer {
                    break; // Don't add more addresses than configured
                }
                entry.addrs.insert(addr.clone());
                let addr_str = addr.to_string();
                if !entry.info.addrs.contains(&addr_str)
                    && entry.info.addrs.len() < self.config.max_addrs_per_peer
                {
                    entry.info.addrs.push(addr_str);
                }
            }
            entry.touch();
        } // Entry guard dropped here

        // Prune if over limit (safe now that we don't hold any locks)
        self.maybe_prune();
    }

    /// Record peer connection
    pub fn peer_connected(&self, peer_id: PeerId) {
        if let Some(mut entry) = self.peers.get_mut(&peer_id) {
            entry.connected = true;
            entry.connected_at = Some(Instant::now());
            entry.info.connection_count += 1;
            entry.touch();
            debug!("Peer connected: {}", peer_id);
        } else {
            // New peer we haven't seen before
            let mut record = PeerRecord::new(peer_id);
            record.connected = true;
            record.connected_at = Some(Instant::now());
            record.info.connection_count = 1;
            self.peers.insert(peer_id, record);
        }

        self.connected_peers.write().insert(peer_id);
    }

    /// Record peer disconnection
    pub fn peer_disconnected(&self, peer_id: &PeerId) {
        if let Some(mut entry) = self.peers.get_mut(peer_id) {
            entry.connected = false;
            entry.connected_at = None;
            entry.touch();
            debug!("Peer disconnected: {}", peer_id);
        }

        self.connected_peers.write().remove(peer_id);
    }

    /// Update peer latency from ping
    pub fn update_latency(&self, peer_id: &PeerId, rtt: Duration) {
        if let Some(mut entry) = self.peers.get_mut(peer_id) {
            entry.update_latency(rtt, self.config.max_latency_samples);
        }
    }

    /// Update peer info from identify
    pub fn update_identify_info(
        &self,
        peer_id: &PeerId,
        protocols: Vec<String>,
        agent_version: Option<String>,
        protocol_version: Option<String>,
        addrs: Vec<Multiaddr>,
    ) {
        if let Some(mut entry) = self.peers.get_mut(peer_id) {
            // Enforce protocol limit
            entry.info.protocols = protocols
                .into_iter()
                .take(self.config.max_protocols_per_peer)
                .collect();
            entry.info.agent_version = agent_version;
            entry.info.protocol_version = protocol_version;

            // Enforce address limit
            for addr in addrs {
                if entry.addrs.len() >= self.config.max_addrs_per_peer {
                    break;
                }
                entry.addrs.insert(addr.clone());
                let addr_str = addr.to_string();
                if !entry.info.addrs.contains(&addr_str)
                    && entry.info.addrs.len() < self.config.max_addrs_per_peer
                {
                    entry.info.addrs.push(addr_str);
                }
            }
            entry.touch();
        }
    }

    /// Increase peer reputation
    pub fn increase_reputation(&self, peer_id: &PeerId, amount: u8) {
        if let Some(mut entry) = self.peers.get_mut(peer_id) {
            entry.info.reputation = entry.info.reputation.saturating_add(amount).min(100);
        }
    }

    /// Decrease peer reputation
    pub fn decrease_reputation(&self, peer_id: &PeerId, amount: u8) {
        if let Some(mut entry) = self.peers.get_mut(peer_id) {
            entry.info.reputation = entry.info.reputation.saturating_sub(amount);
        }
    }

    /// Get peer info
    pub fn get_peer(&self, peer_id: &PeerId) -> Option<PeerInfo> {
        self.peers.get(peer_id).map(|entry| entry.info.clone())
    }

    /// Get addresses for a peer
    pub fn get_addrs(&self, peer_id: &PeerId) -> Vec<Multiaddr> {
        self.peers
            .get(peer_id)
            .map(|entry| entry.addrs.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Check if peer is connected
    pub fn is_connected(&self, peer_id: &PeerId) -> bool {
        self.connected_peers.read().contains(peer_id)
    }

    /// Get all connected peer IDs
    pub fn connected_peers(&self) -> Vec<PeerId> {
        self.connected_peers.read().iter().cloned().collect()
    }

    /// Get number of connected peers
    pub fn connected_count(&self) -> usize {
        self.connected_peers.read().len()
    }

    /// Get all known peer IDs
    pub fn known_peers(&self) -> Vec<PeerId> {
        self.peers.iter().map(|entry| *entry.key()).collect()
    }

    /// Get number of known peers
    pub fn known_count(&self) -> usize {
        self.peers.len()
    }

    /// Get peers sorted by reputation (highest first)
    pub fn peers_by_reputation(&self) -> Vec<PeerInfo> {
        let mut peers: Vec<_> = self.peers.iter().map(|e| e.info.clone()).collect();
        peers.sort_by_key(|p| std::cmp::Reverse(p.reputation));
        peers
    }

    /// Get peers sorted by latency (lowest first)
    pub fn peers_by_latency(&self) -> Vec<PeerInfo> {
        let mut peers: Vec<_> = self.peers.iter().map(|e| e.info.clone()).collect();
        peers.sort_by(|a, b| match (a.avg_latency_ms, b.avg_latency_ms) {
            (Some(a_lat), Some(b_lat)) => a_lat.cmp(&b_lat),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        });
        peers
    }

    /// Remove a peer
    pub fn remove_peer(&self, peer_id: &PeerId) {
        self.peers.remove(peer_id);
        self.connected_peers.write().remove(peer_id);
    }

    /// Prune least valuable peers if over limit
    fn maybe_prune(&self) {
        if self.peers.len() <= self.config.max_peers {
            return;
        }

        // Get disconnected peers sorted by reputation (lowest first)
        let mut candidates: Vec<_> = self
            .peers
            .iter()
            .filter(|e| !e.connected)
            .map(|e| (*e.key(), e.info.reputation, e.info.last_seen))
            .collect();

        // Sort by reputation (lowest first), then by last_seen (oldest first)
        candidates.sort_by(|a, b| a.1.cmp(&b.1).then(a.2.cmp(&b.2)));

        // Remove excess peers
        let to_remove = self.peers.len() - self.config.max_peers;
        for (peer_id, _, _) in candidates.into_iter().take(to_remove) {
            self.peers.remove(&peer_id);
            info!("Pruned peer: {}", peer_id);
        }
    }

    /// Get peer store statistics
    pub fn stats(&self) -> PeerStoreStats {
        let connected = self.connected_count();
        let known = self.known_count();

        let avg_reputation = if known > 0 {
            let total: u64 = self.peers.iter().map(|e| e.info.reputation as u64).sum();
            (total / known as u64) as u8
        } else {
            0
        };

        PeerStoreStats {
            connected_peers: connected,
            known_peers: known,
            max_peers: self.config.max_peers,
            average_reputation: avg_reputation,
        }
    }

    // ============== Persistence Methods ==============

    /// Save peer store to file
    pub fn save_to_file(&self, path: &Path) -> std::io::Result<()> {
        let data = PeerStorePersistence {
            peers: self.get_all_peer_info(),
        };

        let json = serde_json::to_string_pretty(&data).map_err(std::io::Error::other)?;

        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent)?;
            }
        }

        fs::write(path, json)?;
        info!("Saved {} peers to {:?}", data.peers.len(), path);
        Ok(())
    }

    /// Load peer store from file
    pub fn load_from_file(&self, path: &Path) -> std::io::Result<usize> {
        if !path.exists() {
            debug!("Peer store file does not exist: {:?}", path);
            return Ok(0);
        }

        let json = fs::read_to_string(path)?;
        let data: PeerStorePersistence = serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let mut loaded = 0;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        for peer_info in data.peers {
            // Skip peers not seen in the last 7 days
            if now.saturating_sub(peer_info.last_seen) > 7 * 24 * 60 * 60 {
                debug!("Skipping stale peer: {}", peer_info.peer_id);
                continue;
            }

            // Parse peer ID
            let peer_id = match peer_info.peer_id.parse::<PeerId>() {
                Ok(id) => id,
                Err(e) => {
                    warn!("Invalid peer ID in store: {}: {}", peer_info.peer_id, e);
                    continue;
                }
            };

            // Parse addresses
            let addrs: Vec<Multiaddr> = peer_info
                .addrs
                .iter()
                .filter_map(|s| s.parse().ok())
                .collect();

            // Add peer with addresses
            self.add_peer(peer_id, addrs);

            // Restore reputation
            if let Some(mut entry) = self.peers.get_mut(&peer_id) {
                entry.info.reputation = peer_info.reputation;
                entry.info.connection_count = peer_info.connection_count;
                entry.info.agent_version = peer_info.agent_version.clone();
                entry.info.protocol_version = peer_info.protocol_version.clone();
                entry.info.protocols = peer_info.protocols.clone();
            }

            loaded += 1;
        }

        info!("Loaded {} peers from {:?}", loaded, path);
        Ok(loaded)
    }

    /// Get all peer info for persistence
    fn get_all_peer_info(&self) -> Vec<PeerInfo> {
        self.peers.iter().map(|e| e.info.clone()).collect()
    }

    /// Export peers with high reputation (for sharing)
    pub fn export_good_peers(&self, min_reputation: u8) -> Vec<PeerInfo> {
        self.peers
            .iter()
            .filter(|e| e.info.reputation >= min_reputation)
            .map(|e| e.info.clone())
            .collect()
    }

    /// Import peers from another source
    pub fn import_peers(&self, peers: &[PeerInfo]) -> usize {
        let mut imported = 0;
        for peer_info in peers {
            let peer_id = match peer_info.peer_id.parse::<PeerId>() {
                Ok(id) => id,
                Err(_) => continue,
            };

            let addrs: Vec<Multiaddr> = peer_info
                .addrs
                .iter()
                .filter_map(|s| s.parse().ok())
                .collect();

            self.add_peer(peer_id, addrs);
            imported += 1;
        }
        imported
    }
}

impl Default for PeerStore {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Peer store persistence format
#[derive(Debug, Serialize, Deserialize)]
struct PeerStorePersistence {
    peers: Vec<PeerInfo>,
}

/// Peer store statistics
#[derive(Debug, Clone, Serialize)]
pub struct PeerStoreStats {
    /// Number of currently connected peers
    pub connected_peers: usize,
    /// Number of known peers
    pub known_peers: usize,
    /// Maximum number of peers to store
    pub max_peers: usize,
    /// Average peer reputation
    pub average_reputation: u8,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_peer_id() -> PeerId {
        PeerId::random()
    }

    #[test]
    fn test_peer_store_add_peer() {
        let store = PeerStore::new(100);
        let peer_id = random_peer_id();

        store.add_peer(peer_id, vec![]);
        assert!(store.get_peer(&peer_id).is_some());
        assert_eq!(store.known_count(), 1);
    }

    #[test]
    fn test_peer_store_connection() {
        let store = PeerStore::new(100);
        let peer_id = random_peer_id();

        store.peer_connected(peer_id);
        assert!(store.is_connected(&peer_id));
        assert_eq!(store.connected_count(), 1);

        store.peer_disconnected(&peer_id);
        assert!(!store.is_connected(&peer_id));
        assert_eq!(store.connected_count(), 0);
    }

    #[test]
    fn test_peer_store_latency() {
        let store = PeerStore::new(100);
        let peer_id = random_peer_id();

        store.peer_connected(peer_id);
        store.update_latency(&peer_id, Duration::from_millis(50));
        store.update_latency(&peer_id, Duration::from_millis(100));

        let info = store
            .get_peer(&peer_id)
            .expect("test: peer should exist after update_latency");
        assert!(info.avg_latency_ms.is_some());
        assert_eq!(
            info.avg_latency_ms
                .expect("test: avg_latency_ms should be set"),
            75
        ); // average of 50 and 100
    }

    #[test]
    fn test_peer_store_reputation() {
        let store = PeerStore::new(100);
        let peer_id = random_peer_id();

        store.peer_connected(peer_id);

        // Initial reputation is 50
        let info = store
            .get_peer(&peer_id)
            .expect("test: peer should exist after connect");
        assert_eq!(info.reputation, 50);

        // Increase reputation
        store.increase_reputation(&peer_id, 10);
        let info = store
            .get_peer(&peer_id)
            .expect("test: peer should exist after increase_reputation");
        assert_eq!(info.reputation, 60);

        // Decrease reputation
        store.decrease_reputation(&peer_id, 20);
        let info = store
            .get_peer(&peer_id)
            .expect("test: peer should exist after decrease_reputation");
        assert_eq!(info.reputation, 40);
    }

    #[test]
    fn test_peer_store_prune() {
        let store = PeerStore::new(5);

        // Add 10 peers
        for _ in 0..10 {
            let peer_id = random_peer_id();
            store.add_peer(peer_id, vec![]);
        }

        // Should have pruned to max
        assert!(store.known_count() <= 5);
    }

    #[test]
    fn test_peer_store_sorting() {
        let store = PeerStore::new(100);

        // Add peers with different reputations
        let peer1 = random_peer_id();
        let peer2 = random_peer_id();
        let peer3 = random_peer_id();

        store.peer_connected(peer1);
        store.peer_connected(peer2);
        store.peer_connected(peer3);

        store.increase_reputation(&peer1, 30); // 80
        store.decrease_reputation(&peer2, 20); // 30
                                               // peer3 stays at 50

        let by_rep = store.peers_by_reputation();
        assert_eq!(by_rep[0].reputation, 80);
        assert_eq!(by_rep[1].reputation, 50);
        assert_eq!(by_rep[2].reputation, 30);
    }

    #[test]
    fn test_peer_store_persistence() {
        let store = PeerStore::new(100);
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_peer_store.json");

        // Add some peers
        let peer1 = random_peer_id();
        let peer2 = random_peer_id();

        let addr1: Multiaddr = "/ip4/127.0.0.1/tcp/4001"
            .parse()
            .expect("test: valid multiaddr should parse");
        let addr2: Multiaddr = "/ip4/192.168.1.1/tcp/4001"
            .parse()
            .expect("test: valid multiaddr should parse");

        store.add_peer(peer1, vec![addr1.clone()]);
        store.add_peer(peer2, vec![addr2.clone()]);
        store.increase_reputation(&peer1, 30);

        // Save to file
        store
            .save_to_file(&file_path)
            .expect("test: save_to_file should succeed");

        // Create new store and load
        let store2 = PeerStore::new(100);
        let loaded = store2
            .load_from_file(&file_path)
            .expect("test: load_from_file should succeed");

        assert_eq!(loaded, 2);
        assert_eq!(store2.known_count(), 2);

        // Verify peer1 reputation was preserved
        let info1 = store2
            .get_peer(&peer1)
            .expect("test: peer1 should exist after load");
        assert_eq!(info1.reputation, 80);

        // Clean up
        let _ = std::fs::remove_file(&file_path);
    }

    #[test]
    fn test_peer_store_export_import() {
        let store1 = PeerStore::new(100);

        // Add peers with different reputations
        let peer1 = random_peer_id();
        let peer2 = random_peer_id();
        let peer3 = random_peer_id();

        store1.peer_connected(peer1);
        store1.peer_connected(peer2);
        store1.peer_connected(peer3);

        store1.increase_reputation(&peer1, 40); // 90
        store1.increase_reputation(&peer2, 20); // 70
                                                // peer3 stays at 50

        // Export good peers (reputation >= 70)
        let good_peers = store1.export_good_peers(70);
        assert_eq!(good_peers.len(), 2);

        // Import into new store
        let store2 = PeerStore::new(100);
        let imported = store2.import_peers(&good_peers);
        assert_eq!(imported, 2);
        assert_eq!(store2.known_count(), 2);
    }

    #[test]
    fn test_peer_store_load_nonexistent() {
        let store = PeerStore::new(100);
        let result = store.load_from_file(Path::new("/nonexistent/path/peers.json"));
        assert!(result.is_ok());
        assert_eq!(
            result.expect("test: load from nonexistent path should return Ok"),
            0
        );
    }
}
