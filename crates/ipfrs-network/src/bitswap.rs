//! Bitswap protocol implementation for block exchange
//!
//! Bitswap is a data exchange protocol for sharing blocks of content-addressed data.
//! It allows peers to request blocks they need and provide blocks they have.

use cid::Cid;
use ipfrs_core::error::Result;
use libp2p::PeerId;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info};

/// Bitswap protocol handler
pub struct Bitswap {
    /// Blocks we want (want list)
    want_list: Arc<RwLock<HashSet<Cid>>>,
    /// Blocks we have (have list)
    have_list: Arc<RwLock<HashSet<Cid>>>,
    /// Pending block requests per peer
    pending_requests: Arc<RwLock<HashMap<PeerId, HashSet<Cid>>>>,
    /// Block exchange events
    event_tx: mpsc::Sender<BitswapEvent>,
    event_rx: Option<mpsc::Receiver<BitswapEvent>>,
}

/// Bitswap events
#[derive(Debug, Clone)]
pub enum BitswapEvent {
    /// Block received from peer
    BlockReceived {
        cid: Cid,
        data: Vec<u8>,
        from: PeerId,
    },
    /// Block sent to peer
    BlockSent { cid: Cid, to: PeerId },
    /// Block request received
    BlockRequested { cid: Cid, from: PeerId },
    /// Peer doesn't have requested block
    BlockNotFound { cid: Cid, peer: PeerId },
}

/// Bitswap message types
#[derive(Debug, Clone)]
pub enum BitswapMessage {
    /// Want a block
    Want(Cid),
    /// Have a block
    Have(Cid),
    /// Provide block data
    Block { cid: Cid, data: Vec<u8> },
    /// Don't have requested block
    DontHave(Cid),
}

impl Bitswap {
    /// Create a new Bitswap instance
    pub fn new() -> Self {
        let (event_tx, event_rx) = mpsc::channel(1024);

        Self {
            want_list: Arc::new(RwLock::new(HashSet::new())),
            have_list: Arc::new(RwLock::new(HashSet::new())),
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
            event_tx,
            event_rx: Some(event_rx),
        }
    }

    /// Add a CID to our want list
    pub async fn want(&self, cid: Cid) -> Result<()> {
        let mut want_list = self.want_list.write().await;
        want_list.insert(cid);
        debug!("Added to want list: {}", cid);
        Ok(())
    }

    /// Remove a CID from our want list
    pub async fn cancel_want(&self, cid: &Cid) -> Result<()> {
        let mut want_list = self.want_list.write().await;
        want_list.remove(cid);
        debug!("Removed from want list: {}", cid);
        Ok(())
    }

    /// Add a CID to our have list
    pub async fn have(&self, cid: Cid) -> Result<()> {
        let mut have_list = self.have_list.write().await;
        have_list.insert(cid);
        debug!("Added to have list: {}", cid);
        Ok(())
    }

    /// Check if we want a CID
    pub async fn wants(&self, cid: &Cid) -> bool {
        let want_list = self.want_list.read().await;
        want_list.contains(cid)
    }

    /// Check if we have a CID
    pub async fn has(&self, cid: &Cid) -> bool {
        let have_list = self.have_list.read().await;
        have_list.contains(cid)
    }

    /// Get all CIDs we want
    pub async fn get_want_list(&self) -> HashSet<Cid> {
        let want_list = self.want_list.read().await;
        want_list.clone()
    }

    /// Get all CIDs we have
    pub async fn get_have_list(&self) -> HashSet<Cid> {
        let have_list = self.have_list.read().await;
        have_list.clone()
    }

    /// Request a block from a specific peer
    pub async fn request_block(&self, cid: Cid, peer: PeerId) -> Result<()> {
        let mut pending = self.pending_requests.write().await;
        pending.entry(peer).or_insert_with(HashSet::new).insert(cid);
        debug!("Requesting block {} from peer {}", cid, peer);
        Ok(())
    }

    /// Handle incoming Bitswap message
    pub async fn handle_message(
        &self,
        message: BitswapMessage,
        from: PeerId,
    ) -> Result<Option<BitswapMessage>> {
        match message {
            BitswapMessage::Want(cid) => {
                // Peer wants a block
                debug!("Peer {} wants block {}", from, cid);
                let _ = self
                    .event_tx
                    .send(BitswapEvent::BlockRequested { cid, from })
                    .await;

                // Check if we have it
                if self.has(&cid).await {
                    // We'll send the block data through the event system
                    // The actual block retrieval happens outside Bitswap
                    Ok(Some(BitswapMessage::Have(cid)))
                } else {
                    Ok(Some(BitswapMessage::DontHave(cid)))
                }
            }
            BitswapMessage::Have(cid) => {
                // Peer has a block we might want
                debug!("Peer {} has block {}", from, cid);

                // If we want it, request it
                if self.wants(&cid).await {
                    self.request_block(cid, from).await?;
                }
                Ok(None)
            }
            BitswapMessage::Block { cid, data } => {
                // Received block data
                info!(
                    "Received block {} ({} bytes) from peer {}",
                    cid,
                    data.len(),
                    from
                );

                // Remove from want list
                self.cancel_want(&cid).await?;

                // Remove from pending requests
                let mut pending = self.pending_requests.write().await;
                if let Some(peer_requests) = pending.get_mut(&from) {
                    peer_requests.remove(&cid);
                }

                // Emit event
                let _ = self
                    .event_tx
                    .send(BitswapEvent::BlockReceived { cid, data, from })
                    .await;

                Ok(None)
            }
            BitswapMessage::DontHave(cid) => {
                // Peer doesn't have the block
                debug!("Peer {} doesn't have block {}", from, cid);

                let _ = self
                    .event_tx
                    .send(BitswapEvent::BlockNotFound { cid, peer: from })
                    .await;

                Ok(None)
            }
        }
    }

    /// Send a block to a peer
    pub async fn send_block(&self, cid: Cid, data: Vec<u8>, to: PeerId) -> Result<BitswapMessage> {
        debug!(
            "Sending block {} ({} bytes) to peer {}",
            cid,
            data.len(),
            to
        );

        let _ = self
            .event_tx
            .send(BitswapEvent::BlockSent { cid, to })
            .await;

        Ok(BitswapMessage::Block { cid, data })
    }

    /// Get pending requests for a peer
    pub async fn get_pending_requests(&self, peer: &PeerId) -> HashSet<Cid> {
        let pending = self.pending_requests.read().await;
        pending.get(peer).cloned().unwrap_or_default()
    }

    /// Get Bitswap statistics
    pub async fn stats(&self) -> BitswapStats {
        let want_list = self.want_list.read().await;
        let have_list = self.have_list.read().await;
        let pending = self.pending_requests.read().await;

        BitswapStats {
            want_list_size: want_list.len(),
            have_list_size: have_list.len(),
            pending_requests: pending.values().map(|s| s.len()).sum(),
            peers_with_pending_requests: pending.len(),
        }
    }

    /// Take the event receiver
    pub fn take_event_receiver(&mut self) -> Option<mpsc::Receiver<BitswapEvent>> {
        self.event_rx.take()
    }
}

impl Default for Bitswap {
    fn default() -> Self {
        Self::new()
    }
}

/// Bitswap statistics
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct BitswapStats {
    /// Number of blocks we want
    pub want_list_size: usize,
    /// Number of blocks we have
    pub have_list_size: usize,
    /// Total pending block requests
    pub pending_requests: usize,
    /// Number of peers with pending requests
    pub peers_with_pending_requests: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use multihash_codetable::{Code, MultihashDigest};

    fn test_cid() -> Cid {
        let hash = Code::Sha2_256.digest(b"test data");
        Cid::new_v1(0x55, hash)
    }

    fn test_peer_id() -> PeerId {
        PeerId::random()
    }

    #[tokio::test]
    async fn test_bitswap_creation() {
        let bitswap = Bitswap::new();
        let stats = bitswap.stats().await;

        assert_eq!(stats.want_list_size, 0);
        assert_eq!(stats.have_list_size, 0);
        assert_eq!(stats.pending_requests, 0);
    }

    #[tokio::test]
    async fn test_want_list() {
        let bitswap = Bitswap::new();
        let cid = test_cid();

        // Add to want list
        bitswap
            .want(cid)
            .await
            .expect("test: want() should succeed");
        assert!(bitswap.wants(&cid).await);

        let want_list = bitswap.get_want_list().await;
        assert_eq!(want_list.len(), 1);
        assert!(want_list.contains(&cid));

        // Cancel want
        bitswap
            .cancel_want(&cid)
            .await
            .expect("test: cancel_want() should succeed");
        assert!(!bitswap.wants(&cid).await);
    }

    #[tokio::test]
    async fn test_have_list() {
        let bitswap = Bitswap::new();
        let cid = test_cid();

        // Add to have list
        bitswap
            .have(cid)
            .await
            .expect("test: have() should succeed");
        assert!(bitswap.has(&cid).await);

        let have_list = bitswap.get_have_list().await;
        assert_eq!(have_list.len(), 1);
        assert!(have_list.contains(&cid));
    }

    #[tokio::test]
    async fn test_request_block() {
        let bitswap = Bitswap::new();
        let cid = test_cid();
        let peer = test_peer_id();

        bitswap
            .request_block(cid, peer)
            .await
            .expect("test: request_block() should succeed");

        let pending = bitswap.get_pending_requests(&peer).await;
        assert_eq!(pending.len(), 1);
        assert!(pending.contains(&cid));
    }

    #[tokio::test]
    async fn test_handle_want_message_have_block() {
        let bitswap = Bitswap::new();
        let cid = test_cid();
        let peer = test_peer_id();

        // Add to have list
        bitswap
            .have(cid)
            .await
            .expect("test: have() should succeed");

        // Handle want message
        let response = bitswap
            .handle_message(BitswapMessage::Want(cid), peer)
            .await
            .expect("test: handle_message(Want) should succeed when block is in have list");

        assert!(response.is_some());
        match response
            .expect("test: handle_message should return Some(Have) when block is in have list")
        {
            BitswapMessage::Have(received_cid) => assert_eq!(received_cid, cid),
            _ => panic!("Expected Have message"),
        }
    }

    #[tokio::test]
    async fn test_handle_want_message_dont_have() {
        let bitswap = Bitswap::new();
        let cid = test_cid();
        let peer = test_peer_id();

        // Don't add to have list
        let response = bitswap
            .handle_message(BitswapMessage::Want(cid), peer)
            .await
            .expect("test: handle_message(Want) should succeed when block is not in have list");

        assert!(response.is_some());
        match response
            .expect("test: handle_message should return Some(DontHave) when block is absent")
        {
            BitswapMessage::DontHave(received_cid) => assert_eq!(received_cid, cid),
            _ => panic!("Expected DontHave message"),
        }
    }

    #[tokio::test]
    async fn test_handle_have_message() {
        let bitswap = Bitswap::new();
        let cid = test_cid();
        let peer = test_peer_id();

        // Add to want list
        bitswap
            .want(cid)
            .await
            .expect("test: want() should succeed");

        // Handle have message
        let response = bitswap
            .handle_message(BitswapMessage::Have(cid), peer)
            .await
            .expect("test: handle_message(Have) should succeed");

        assert!(response.is_none());

        // Should have created a pending request
        let pending = bitswap.get_pending_requests(&peer).await;
        assert_eq!(pending.len(), 1);
        assert!(pending.contains(&cid));
    }

    #[tokio::test]
    async fn test_handle_block_message() {
        let bitswap = Bitswap::new();
        let cid = test_cid();
        let peer = test_peer_id();
        let data = b"test block data".to_vec();

        // Add to want list and pending requests
        bitswap
            .want(cid)
            .await
            .expect("test: want() should succeed");
        bitswap
            .request_block(cid, peer)
            .await
            .expect("test: request_block() should succeed");

        // Handle block message
        let response = bitswap
            .handle_message(
                BitswapMessage::Block {
                    cid,
                    data: data.clone(),
                },
                peer,
            )
            .await
            .expect("test: handle_message(Block) should succeed");

        assert!(response.is_none());

        // Should have removed from want list
        assert!(!bitswap.wants(&cid).await);

        // Should have removed from pending requests
        let pending = bitswap.get_pending_requests(&peer).await;
        assert_eq!(pending.len(), 0);
    }

    #[tokio::test]
    async fn test_send_block() {
        let bitswap = Bitswap::new();
        let cid = test_cid();
        let peer = test_peer_id();
        let data = b"test block data".to_vec();

        let message = bitswap
            .send_block(cid, data.clone(), peer)
            .await
            .expect("test: send_block() should succeed");

        match message {
            BitswapMessage::Block {
                cid: received_cid,
                data: received_data,
            } => {
                assert_eq!(received_cid, cid);
                assert_eq!(received_data, data);
            }
            _ => panic!("Expected Block message"),
        }
    }

    #[tokio::test]
    async fn test_bitswap_stats() {
        let bitswap = Bitswap::new();
        let cid1 = test_cid();
        let cid2 = test_cid();
        let peer = test_peer_id();

        bitswap
            .want(cid1)
            .await
            .expect("test: want() should succeed");
        bitswap
            .have(cid2)
            .await
            .expect("test: have() should succeed");
        bitswap
            .request_block(cid1, peer)
            .await
            .expect("test: request_block() should succeed");

        let stats = bitswap.stats().await;

        assert_eq!(stats.want_list_size, 1);
        assert_eq!(stats.have_list_size, 1);
        assert_eq!(stats.pending_requests, 1);
        assert_eq!(stats.peers_with_pending_requests, 1);
    }
}
