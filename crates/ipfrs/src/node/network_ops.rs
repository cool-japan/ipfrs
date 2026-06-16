//! Network and peer-to-peer operations for Node

use ipfrs_core::{Cid, Error, Result};
use std::time::Duration;

use super::Node;

impl Node {
    /// Get local peer ID
    pub fn peer_id(&self) -> Result<String> {
        let network = self.network()?;
        Ok(network.peer_id().to_string())
    }

    /// Get connected peers
    pub async fn peers(&self) -> Result<Vec<String>> {
        let network = self.network()?;
        let peers = network.connected_peers();
        Ok(peers.into_iter().map(|p| p.to_string()).collect())
    }

    /// Connect to a peer
    pub async fn connect(&mut self, addr: &str) -> Result<()> {
        let addr: ipfrs_network::libp2p::Multiaddr = addr
            .parse()
            .map_err(|e| Error::Network(format!("Invalid multiaddr: {}", e)))?;

        if let Some(network) = &mut self.network {
            network.connect(addr).await?;
        }
        Ok(())
    }

    /// Disconnect from a peer
    pub async fn disconnect(&mut self, peer_id: &str) -> Result<()> {
        use std::str::FromStr;
        let peer_id: ipfrs_network::libp2p::PeerId =
            ipfrs_network::libp2p::PeerId::from_str(peer_id)
                .map_err(|e| Error::Network(format!("Invalid peer ID: {}", e)))?;

        if let Some(network) = &mut self.network {
            network.disconnect(peer_id).await?;
        }
        Ok(())
    }

    /// Announce content to DHT (provide)
    pub async fn provide(&mut self, cid: &Cid) -> Result<()> {
        if let Some(network) = &mut self.network {
            network.provide(cid).await?;
        }
        Ok(())
    }

    /// Find providers for content in DHT.
    ///
    /// Queries the Kademlia DHT and waits up to 30 seconds for provider results.
    /// Returns the peer IDs of nodes that have announced themselves as providers
    /// for the given CID.
    pub async fn find_providers(
        &mut self,
        cid: &Cid,
    ) -> Result<Vec<ipfrs_network::libp2p::PeerId>> {
        if let Some(network) = &mut self.network {
            let providers = network
                .find_providers_await(cid, Duration::from_secs(30))
                .await?;
            Ok(providers)
        } else {
            Ok(Vec::new())
        }
    }

    /// Find providers for content with a custom timeout.
    ///
    /// Like `find_providers` but lets the caller specify how long to wait.
    pub async fn find_providers_timeout(
        &mut self,
        cid: &Cid,
        timeout: Duration,
    ) -> Result<Vec<ipfrs_network::libp2p::PeerId>> {
        if let Some(network) = &mut self.network {
            let providers = network.find_providers_await(cid, timeout).await?;
            Ok(providers)
        } else {
            Ok(Vec::new())
        }
    }

    /// Get network statistics
    pub fn network_stats(&self) -> Result<ipfrs_network::NetworkStats> {
        let network = self.network()?;
        Ok(network.stats())
    }

    /// Get bitswap statistics
    pub fn bitswap_stats(&self) -> Result<ipfrs_network::BitswapStats> {
        // Placeholder implementation - returns default stats
        Ok(ipfrs_network::BitswapStats::default())
    }

    /// Ping a peer (placeholder implementation)
    pub async fn ping(&mut self, _peer_id: &str) -> Result<()> {
        // Placeholder - actual ping would use libp2p ping protocol
        // For now, just verify we're connected
        let _network = self.network()?;
        Ok(())
    }

    /// Find a peer's addresses in the DHT (placeholder implementation)
    pub async fn find_peer(&mut self, peer_id: &str) -> Result<Vec<String>> {
        let _network = self.network()?;
        // Placeholder - actual implementation would query DHT
        Ok(vec![format!("/p2p/{}", peer_id)])
    }

    /// Get bootstrap peers
    pub fn bootstrap_peers(&self) -> Result<Vec<String>> {
        let stats = self.network_stats()?;
        Ok(stats.bootstrap_peers)
    }

    /// Add a bootstrap peer
    pub async fn add_bootstrap_peer(&mut self, addr: &str) -> Result<()> {
        let multiaddr: ipfrs_network::libp2p::Multiaddr = addr
            .parse()
            .map_err(|e| Error::Network(format!("Invalid multiaddr: {}", e)))?;
        let network = self.network_mut()?;
        network.connect(multiaddr).await?;
        Ok(())
    }

    /// Remove a bootstrap peer
    pub async fn remove_bootstrap_peer(&mut self, _addr: &str) -> Result<()> {
        // Placeholder - would update config and disconnect
        Ok(())
    }
}
