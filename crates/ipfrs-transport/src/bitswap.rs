//! Bitswap-compatible block exchange protocol
//!
//! Implements the Bitswap protocol for requesting and exchanging blocks
//! with other peers in the IPFS network.
//!
//! # Example
//!
//! ```
//! use ipfrs_transport::bitswap::{BitswapExchange, BitswapConfig};
//! use ipfrs_storage::MemoryBlockStore;
//! use ipfrs_core::{Block, Cid};
//! use multihash::Multihash;
//! use bytes::Bytes;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create an in-memory block store
//! let store = std::sync::Arc::new(MemoryBlockStore::new());
//!
//! // Create a Bitswap exchange instance
//! let config = BitswapConfig::default();
//! let exchange = BitswapExchange::new(store, config)?;
//!
//! // Add a peer
//! exchange.add_peer("peer1".to_string());
//!
//! // The exchange is now ready to request and send blocks
//! # Ok(())
//! # }
//! ```

use crate::messages::{Message, WantEntry as MessageWantEntry};
use crate::peer_manager::{ConcurrentPeerManager, PeerId, PeerScoringConfig, SelectionStrategy};
use crate::want_list::{ConcurrentWantList, WantEntry, WantListConfig};
use ipfrs_core::{Block, Cid, Result};
use ipfrs_storage::traits::BlockStore;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Bitswap exchange configuration
#[derive(Debug, Clone)]
pub struct BitswapConfig {
    /// Want list configuration
    pub want_list: WantListConfig,
    /// Peer scoring configuration
    pub peer_scoring: PeerScoringConfig,
    /// Maximum peers to send wants to
    pub max_peers_per_request: usize,
    /// Default peer selection strategy
    pub selection_strategy: SelectionStrategy,
    /// Enable automatic cleanup of stale wants
    pub auto_cleanup: bool,
    /// Cleanup interval
    pub cleanup_interval: Duration,
}

impl Default for BitswapConfig {
    fn default() -> Self {
        Self {
            want_list: WantListConfig::default(),
            peer_scoring: PeerScoringConfig::default(),
            max_peers_per_request: 3,
            selection_strategy: SelectionStrategy::BestScore,
            auto_cleanup: true,
            cleanup_interval: Duration::from_secs(10),
        }
    }
}

/// Type alias for pending request tracking
type PendingRequestMap = Arc<RwLock<HashMap<Cid, (Vec<PeerId>, Instant)>>>;

/// Bitswap exchange handler with enhanced want list and peer management
pub struct BitswapExchange<S: BlockStore> {
    /// Local block store
    store: Arc<S>,
    /// Enhanced want list with priority queue
    want_list: ConcurrentWantList,
    /// Enhanced peer manager with scoring and selection
    peer_manager: ConcurrentPeerManager,
    /// Request tracking (CID -> peers sent to, start time)
    pending_requests: PendingRequestMap,
    /// Configuration
    config: BitswapConfig,
}

impl<S: BlockStore> BitswapExchange<S> {
    /// Create a new Bitswap exchange handler
    pub fn new(store: Arc<S>, config: BitswapConfig) -> Result<Self> {
        Ok(Self {
            store,
            want_list: ConcurrentWantList::new(config.want_list.clone()),
            peer_manager: ConcurrentPeerManager::new(config.peer_scoring.clone()),
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
            config,
        })
    }

    /// Create with default configuration
    pub fn with_defaults(store: Arc<S>) -> Result<Self> {
        Self::new(store, BitswapConfig::default())
    }

    /// Add a CID to the want list with default priority
    pub fn want(&self, cid: Cid, priority: i32) -> Result<()> {
        if !self.want_list.add_simple(cid, priority) {
            // Already wanted or list full - check which
            if self.want_list.contains(&cid) {
                // Already wanted, update priority if higher
                self.want_list.update_priority(&cid, priority);
                return Ok(());
            }
            return Err(ipfrs_core::Error::Internal("Want list full".to_string()));
        }
        Ok(())
    }

    /// Add a CID with full entry configuration
    pub fn want_with_entry(&self, entry: WantEntry) -> Result<()> {
        if !self.want_list.add(entry) {
            return Err(ipfrs_core::Error::Internal(
                "Want list full or duplicate".to_string(),
            ));
        }
        Ok(())
    }

    /// Cancel a want
    pub fn cancel_want(&self, cid: &Cid) -> Result<()> {
        self.want_list.remove(cid);
        self.pending_requests
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove(cid);
        Ok(())
    }

    /// Check if we're currently wanting a CID
    pub fn is_wanted(&self, cid: &Cid) -> bool {
        self.want_list.contains(cid)
    }

    /// Get the next wanted CID (highest priority)
    pub fn next_want(&self) -> Option<Cid> {
        self.want_list.pop().map(|e| e.cid)
    }

    /// Get current want list as message entries
    pub fn get_want_list(&self) -> Vec<MessageWantEntry> {
        self.want_list
            .cids()
            .into_iter()
            .map(MessageWantEntry::new)
            .collect()
    }

    /// Get all CIDs in want list
    pub fn get_wanted_cids(&self) -> Vec<Cid> {
        self.want_list.cids()
    }

    /// Update priority for a CID
    pub fn update_priority(&self, cid: &Cid, priority: i32) -> bool {
        self.want_list.update_priority(cid, priority)
    }

    /// Boost priorities for deadline-approaching entries
    pub fn boost_deadline_priorities(&self) {
        self.want_list.boost_deadline_priorities()
    }

    /// Add a peer to the manager
    pub fn add_peer(&self, peer_id: PeerId) {
        self.peer_manager.add_peer(peer_id);
    }

    /// Remove a peer
    pub fn remove_peer(&self, peer_id: &PeerId) {
        self.peer_manager.remove_peer(peer_id);
    }

    /// Select peers for a CID request
    pub fn select_peers_for_request(&self, cid: &Cid) -> Vec<PeerId> {
        // First try peers that definitely have the block
        let providers = self
            .peer_manager
            .select_providers(cid, self.config.max_peers_per_request);

        if !providers.is_empty() {
            return providers;
        }

        // Fall back to general selection
        self.peer_manager.select_peers(
            cid,
            self.config.max_peers_per_request,
            self.config.selection_strategy,
        )
    }

    /// Process incoming block from peer
    pub async fn receive_block(&self, peer_id: &PeerId, block: Block) -> Result<()> {
        let cid = *block.cid();
        let size = block.size();

        // Calculate latency if we tracked the request
        let latency = {
            let pending = self
                .pending_requests
                .read()
                .unwrap_or_else(|e| e.into_inner());
            pending.get(&cid).map(|(_, start)| start.elapsed())
        };

        // Store the block
        self.store.put(&block).await?;

        // Update peer manager with success
        if let Some(latency) = latency {
            self.peer_manager.record_success(peer_id, size, latency);
        } else {
            self.peer_manager
                .record_success(peer_id, size, Duration::from_millis(100));
        }

        // Record that this peer has the block
        self.peer_manager.record_has(peer_id, cid);

        // Remove from want list and pending
        self.cancel_want(&cid)?;

        Ok(())
    }

    /// Record a failed request
    pub fn record_failure(&self, peer_id: &PeerId, cid: &Cid) {
        self.peer_manager.record_failure(peer_id);
        self.want_list.mark_attempted(cid);
    }

    /// Record HAVE message from peer
    pub fn record_have(&self, peer_id: &PeerId, cid: Cid) {
        self.peer_manager.record_has(peer_id, cid);
    }

    /// Record DONT_HAVE message from peer
    pub fn record_dont_have(&self, peer_id: &PeerId, cid: Cid) {
        self.peer_manager.record_doesnt_have(peer_id, cid);
    }

    /// Send block to peer
    pub async fn send_block(&self, _peer_id: &PeerId, cid: &Cid) -> Result<Option<Message>> {
        // Check if we have the block
        let block = match self.store.get(cid).await? {
            Some(b) => b,
            None => {
                return Ok(Some(Message::dont_have(*cid)));
            }
        };

        // Create block message
        Ok(Some(Message::block(*cid, block.data().to_vec())))
    }

    /// Mark that we sent a request to peers
    pub fn mark_request_sent(&self, cid: Cid, peers: Vec<PeerId>) {
        for peer in &peers {
            self.peer_manager.mark_request_sent(peer);
        }
        self.pending_requests
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(cid, (peers, Instant::now()));
    }

    /// Clean up stale wants and expired requests
    pub fn cleanup(&self) {
        let expired = self.want_list.cleanup_expired();

        // Also clean up pending requests for expired wants
        let mut pending = self
            .pending_requests
            .write()
            .unwrap_or_else(|e| e.into_inner());
        for entry in expired {
            pending.remove(&entry.cid);
        }
    }

    /// Get peer scores
    pub fn get_peer_scores(&self) -> HashMap<PeerId, f64> {
        self.peer_manager.get_scores()
    }

    /// Check if a peer is blacklisted
    pub fn is_peer_blacklisted(&self, peer_id: &PeerId) -> bool {
        self.peer_manager.is_blacklisted(peer_id)
    }

    /// Get the peer manager for direct access
    pub fn peer_manager(&self) -> &ConcurrentPeerManager {
        &self.peer_manager
    }

    /// Get the want list for direct access
    pub fn want_list_manager(&self) -> &ConcurrentWantList {
        &self.want_list
    }

    /// Get statistics
    pub fn stats(&self) -> BitswapStats {
        let peer_stats = self.peer_manager.stats();
        BitswapStats {
            want_list_size: self.want_list.len(),
            pending_requests: self
                .pending_requests
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .len(),
            num_peers: peer_stats.total_peers,
            connected_peers: peer_stats.connected_peers,
            blacklisted_peers: peer_stats.blacklisted_peers,
            total_bytes_sent: 0, // Would need additional tracking
            total_bytes_recv: 0, // Would need additional tracking
            total_requests: peer_stats.total_requests,
            completed_requests: peer_stats.total_completed,
            failed_requests: peer_stats.total_failed,
            avg_peer_score: peer_stats.avg_score,
            avg_latency_ms: peer_stats.avg_latency_ms,
        }
    }
}

/// Bitswap statistics
#[derive(Debug, Clone)]
pub struct BitswapStats {
    /// Number of CIDs in want list
    pub want_list_size: usize,
    /// Number of pending requests
    pub pending_requests: usize,
    /// Number of known peers
    pub num_peers: usize,
    /// Number of connected peers
    pub connected_peers: usize,
    /// Number of blacklisted peers
    pub blacklisted_peers: usize,
    /// Total bytes sent
    pub total_bytes_sent: u64,
    /// Total bytes received
    pub total_bytes_recv: u64,
    /// Total requests sent
    pub total_requests: u64,
    /// Completed requests
    pub completed_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Average peer score
    pub avg_peer_score: f64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::peer_manager::BlacklistReason;
    use ipfrs_storage::{BlockStoreConfig, SledBlockStore};

    #[tokio::test]
    async fn test_bitswap_want_list() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-bitswap-want"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("test: create block store"));
        let bitswap = BitswapExchange::with_defaults(store).expect("test: create bitswap");

        let cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse::<Cid>()
            .expect("test: parse cid");

        // Add to want list
        bitswap.want(cid, 10).expect("test: add want");
        assert!(bitswap.is_wanted(&cid));

        // Cancel want
        bitswap.cancel_want(&cid).expect("test: cancel want");
        assert!(!bitswap.is_wanted(&cid));
    }

    #[tokio::test]
    async fn test_peer_management() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-bitswap-peer"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("test: create block store"));
        let bitswap = BitswapExchange::with_defaults(store).expect("test: create bitswap");

        let peer_id = "peer1".to_string();

        // Add peer
        bitswap.add_peer(peer_id.clone());

        // Record HAVE message
        let cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse::<Cid>()
            .expect("test: parse cid");
        bitswap.record_have(&peer_id, cid);

        // Peer should be selected as provider
        let providers = bitswap.select_peers_for_request(&cid);
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0], peer_id);

        // Check stats
        let stats = bitswap.stats();
        assert_eq!(stats.num_peers, 1);
    }

    #[tokio::test]
    async fn test_receive_block_updates_peer() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-bitswap-recv"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("test: create block store"));
        let bitswap = BitswapExchange::with_defaults(store).expect("test: create bitswap");

        let peer_id = "peer1".to_string();
        bitswap.add_peer(peer_id.clone());

        let block = Block::new(vec![1, 2, 3, 4].into()).expect("test: create block");
        let cid = *block.cid();

        // Want the block first
        bitswap.want(cid, 10).expect("test: add want");
        assert!(bitswap.is_wanted(&cid));

        // Receive block
        bitswap
            .receive_block(&peer_id, block)
            .await
            .expect("test: receive block");

        // Block should no longer be wanted
        assert!(!bitswap.is_wanted(&cid));

        // Stats should show completed request
        let stats = bitswap.stats();
        assert_eq!(stats.completed_requests, 1);
    }

    #[tokio::test]
    async fn test_blacklist() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-bitswap-blacklist"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("test: create block store"));
        let bitswap = BitswapExchange::with_defaults(store).expect("test: create bitswap");

        let peer_id = "bad_peer".to_string();
        bitswap.add_peer(peer_id.clone());

        // Blacklist the peer
        bitswap
            .peer_manager()
            .blacklist_peer(peer_id.clone(), BlacklistReason::Manual, None);

        assert!(bitswap.is_peer_blacklisted(&peer_id));

        // Stats should reflect blacklist
        let stats = bitswap.stats();
        assert_eq!(stats.blacklisted_peers, 1);
    }

    #[tokio::test]
    async fn test_priority_update() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-bitswap-priority"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("test: create block store"));
        let bitswap = BitswapExchange::with_defaults(store).expect("test: create bitswap");

        let cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse::<Cid>()
            .expect("test: parse cid");

        // Add with low priority
        bitswap.want(cid, 10).expect("test: add want");

        // Update to high priority
        assert!(bitswap.update_priority(&cid, 100));
    }

    #[tokio::test]
    async fn test_multiple_concurrent_wants() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-bitswap-multi-want"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("test: create block store"));
        let bitswap = BitswapExchange::with_defaults(store).expect("test: create bitswap");

        let cid1 = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse::<Cid>()
            .expect("test: parse cid");
        let cid2 = "bafybeihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku"
            .parse::<Cid>()
            .expect("test: parse cid");

        // Add multiple wants with different priorities
        bitswap.want(cid1, 100).expect("test: add want");
        bitswap.want(cid2, 50).expect("test: add want");

        let wanted_cids = bitswap.get_wanted_cids();
        assert_eq!(wanted_cids.len(), 2);
        assert!(wanted_cids.contains(&cid1));
        assert!(wanted_cids.contains(&cid2));

        // Cancel one
        bitswap.cancel_want(&cid1).expect("test: cancel want");
        assert!(!bitswap.is_wanted(&cid1));
        assert!(bitswap.is_wanted(&cid2));
    }

    #[tokio::test]
    async fn test_send_block_exists() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-bitswap-send-exists"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("test: create block store"));

        // Store a block first
        let block = Block::new(vec![1, 2, 3, 4].into()).expect("test: create block");
        let cid = *block.cid();
        store.put(&block).await.expect("test: put block");

        let bitswap = BitswapExchange::with_defaults(store).expect("test: create bitswap");
        let peer_id = "peer1".to_string();

        // Send block should succeed
        let msg = bitswap
            .send_block(&peer_id, &cid)
            .await
            .expect("test: send block");
        assert!(msg.is_some());

        match msg.expect("test: message should be Some") {
            Message::Block(block_msg) => {
                assert_eq!(block_msg.cid, cid);
                assert_eq!(block_msg.data.len(), 4);
            }
            _ => panic!("Expected Block message"),
        }
    }

    #[tokio::test]
    async fn test_send_block_not_found() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-bitswap-send-notfound"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("test: create block store"));
        let bitswap = BitswapExchange::with_defaults(store).expect("test: create bitswap");

        let cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse::<Cid>()
            .expect("test: parse cid");
        let peer_id = "peer1".to_string();

        // Send block should return DONT_HAVE
        let msg = bitswap
            .send_block(&peer_id, &cid)
            .await
            .expect("test: send block");
        assert!(msg.is_some());

        match msg.expect("test: message should be Some") {
            Message::DontHave(dont_have) => {
                assert_eq!(dont_have.cid, cid);
            }
            _ => panic!("Expected DontHave message"),
        }
    }

    #[tokio::test]
    async fn test_cleanup_stale_wants() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-bitswap-cleanup"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("test: create block store"));

        // Create with very short timeout
        let mut bitswap_config = BitswapConfig::default();
        bitswap_config.want_list.default_timeout = Duration::from_millis(1);

        let bitswap =
            BitswapExchange::new(store, bitswap_config).expect("test: create bitswap with config");

        let cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse::<Cid>()
            .expect("test: parse cid");

        bitswap.want(cid, 10).expect("test: add want");
        assert!(bitswap.is_wanted(&cid));

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Cleanup should remove expired want
        bitswap.cleanup();
        assert!(!bitswap.is_wanted(&cid));
    }

    #[tokio::test]
    async fn test_mark_request_sent() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-bitswap-mark-sent"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("test: create block store"));
        let bitswap = BitswapExchange::with_defaults(store).expect("test: create bitswap");

        let cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse::<Cid>()
            .expect("test: parse cid");

        let peers = vec!["peer1".to_string(), "peer2".to_string()];

        // Mark request as sent
        bitswap.mark_request_sent(cid, peers.clone());

        // Stats should show pending request
        let stats = bitswap.stats();
        assert_eq!(stats.pending_requests, 1);
    }

    #[tokio::test]
    async fn test_record_failure() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-bitswap-failure"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("test: create block store"));
        let bitswap = BitswapExchange::with_defaults(store).expect("test: create bitswap");

        let peer_id = "peer1".to_string();
        bitswap.add_peer(peer_id.clone());

        let cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse::<Cid>()
            .expect("test: parse cid");

        bitswap.want(cid, 10).expect("test: add want");

        // Record failure
        bitswap.record_failure(&peer_id, &cid);

        // Stats should show failed request
        let stats = bitswap.stats();
        assert!(stats.failed_requests >= 1);
    }

    #[tokio::test]
    async fn test_multiple_peers_provider_selection() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-bitswap-multi-peers"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("test: create block store"));
        let bitswap = BitswapExchange::with_defaults(store).expect("test: create bitswap");

        let cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse::<Cid>()
            .expect("test: parse cid");

        // Add multiple peers
        bitswap.add_peer("peer1".to_string());
        bitswap.add_peer("peer2".to_string());
        bitswap.add_peer("peer3".to_string());

        // Only peer2 has the block
        bitswap.record_have(&"peer2".to_string(), cid);

        // Should select peer2 as provider
        let providers = bitswap.select_peers_for_request(&cid);
        assert!(providers.contains(&"peer2".to_string()));
    }

    #[tokio::test]
    async fn test_record_dont_have() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-bitswap-dont-have"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("test: create block store"));
        let bitswap = BitswapExchange::with_defaults(store).expect("test: create bitswap");

        let peer_id = "peer1".to_string();
        bitswap.add_peer(peer_id.clone());

        let cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse::<Cid>()
            .expect("test: parse cid");

        // Record DONT_HAVE
        bitswap.record_dont_have(&peer_id, cid);

        // Peer should not be selected as provider
        let providers = bitswap.select_peers_for_request(&cid);
        assert!(!providers.contains(&peer_id) || providers.is_empty());
    }

    #[tokio::test]
    async fn test_get_want_list_message() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-bitswap-want-msg"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("test: create block store"));
        let bitswap = BitswapExchange::with_defaults(store).expect("test: create bitswap");

        let cid1 = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse::<Cid>()
            .expect("test: parse cid");
        let cid2 = "bafybeihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku"
            .parse::<Cid>()
            .expect("test: parse cid");

        bitswap.want(cid1, 100).expect("test: add want");
        bitswap.want(cid2, 50).expect("test: add want");

        let want_list = bitswap.get_want_list();
        assert_eq!(want_list.len(), 2);
        assert!(want_list.iter().any(|e| e.cid == cid1));
        assert!(want_list.iter().any(|e| e.cid == cid2));
    }

    #[tokio::test]
    async fn test_duplicate_want() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-bitswap-dup-want"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("test: create block store"));
        let bitswap = BitswapExchange::with_defaults(store).expect("test: create bitswap");

        let cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse::<Cid>()
            .expect("test: parse cid");

        // Add same CID twice
        bitswap.want(cid, 10).expect("test: add want");
        bitswap.want(cid, 20).expect("test: add want"); // Should update priority, not error

        assert!(bitswap.is_wanted(&cid));
        assert_eq!(bitswap.get_wanted_cids().len(), 1);
    }

    #[tokio::test]
    async fn test_peer_removal() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-bitswap-peer-remove"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("test: create block store"));
        let bitswap = BitswapExchange::with_defaults(store).expect("test: create bitswap");

        let peer_id = "peer1".to_string();
        bitswap.add_peer(peer_id.clone());

        let stats = bitswap.stats();
        assert_eq!(stats.num_peers, 1);

        // Remove peer
        bitswap.remove_peer(&peer_id);

        let stats = bitswap.stats();
        assert_eq!(stats.num_peers, 0);
    }

    #[tokio::test]
    async fn test_peer_scores() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-bitswap-scores"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("test: create block store"));
        let bitswap = BitswapExchange::with_defaults(store).expect("test: create bitswap");

        let peer_id = "peer1".to_string();
        bitswap.add_peer(peer_id.clone());

        // Get initial scores
        let scores = bitswap.get_peer_scores();
        assert!(scores.contains_key(&peer_id));
    }

    #[tokio::test]
    async fn test_empty_want_list() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-bitswap-empty"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("test: create block store"));
        let bitswap = BitswapExchange::with_defaults(store).expect("test: create bitswap");

        let wanted_cids = bitswap.get_wanted_cids();
        assert_eq!(wanted_cids.len(), 0);

        let next = bitswap.next_want();
        assert!(next.is_none());

        let stats = bitswap.stats();
        assert_eq!(stats.want_list_size, 0);
    }

    #[tokio::test]
    async fn test_boost_deadline_priorities() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-bitswap-deadline"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("test: create block store"));
        let bitswap = BitswapExchange::with_defaults(store).expect("test: create bitswap");

        let cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse::<Cid>()
            .expect("test: parse cid");

        bitswap.want(cid, 10).expect("test: add want");

        // Boost deadline priorities should not panic
        bitswap.boost_deadline_priorities();

        assert!(bitswap.is_wanted(&cid));
    }
}
