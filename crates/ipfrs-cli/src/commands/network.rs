//! Network commands (swarm, dht, bootstrap)
//!
//! This module provides network-related operations:
//! - Swarm: peers, connect, disconnect, addrs
//! - DHT: findprovs, findpeer, provide
//! - Bootstrap: list, add, rm

use anyhow::Result;

use crate::output::{self, print_header, success};
use crate::progress;

// ============================================================================
// Swarm Commands
// ============================================================================

/// Show connected peers
pub async fn show_peers(format: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};

    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    let peers = node.peers().await?;

    match format {
        "json" => {
            println!("[");
            for (i, peer) in peers.iter().enumerate() {
                if i > 0 {
                    println!(",");
                }
                print!("  \"{}\"", peer);
            }
            println!();
            println!("]");
        }
        _ => {
            if peers.is_empty() {
                println!("No connected peers");
            } else {
                println!("Connected peers ({}):", peers.len());
                for peer in &peers {
                    println!("  {}", peer);
                }
            }
        }
    }

    node.stop().await?;
    Ok(())
}

/// Connect to a peer
pub async fn swarm_connect(addr: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};

    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    println!("Connecting to {}...", addr);
    node.connect(addr).await?;
    println!("Connection initiated");

    node.stop().await?;
    Ok(())
}

/// Disconnect from a peer
pub async fn swarm_disconnect(peer_id: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};

    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    println!("Disconnecting from {}...", peer_id);
    node.disconnect(peer_id).await?;
    println!("Disconnected");

    node.stop().await?;
    Ok(())
}

/// List listening addresses
pub async fn swarm_addrs(format: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};

    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    let stats = node.network_stats()?;

    match format {
        "json" => {
            println!("[");
            for (i, addr) in stats.listen_addrs.iter().enumerate() {
                if i > 0 {
                    println!(",");
                }
                print!("  \"{}\"", addr);
            }
            println!();
            println!("]");
        }
        _ => {
            if stats.listen_addrs.is_empty() {
                println!("No listening addresses");
            } else {
                println!("Listening addresses:");
                for addr in &stats.listen_addrs {
                    println!("  {}", addr);
                }
            }
        }
    }

    node.stop().await?;
    Ok(())
}

// ============================================================================
// DHT Commands
// ============================================================================

/// Find providers for a CID
pub async fn dht_findprovs(cid: &str, _format: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};

    let cid_parsed: ipfrs_core::Cid = cid
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid CID: {}", e))?;

    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    println!("Searching for providers of {}...", cid);
    node.find_providers(&cid_parsed).await?;

    // In a real implementation, we'd wait for provider events
    // For now, just indicate the query was sent
    println!("Provider query sent to DHT");
    println!("(Provider discovery events would be received asynchronously)");

    node.stop().await?;
    Ok(())
}

/// Announce content to DHT
pub async fn dht_provide(cid: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};

    let cid_parsed: ipfrs_core::Cid = cid
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid CID: {}", e))?;

    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    println!("Announcing {} to DHT...", cid);
    node.provide(&cid_parsed).await?;
    println!("Content announced to DHT");

    node.stop().await?;
    Ok(())
}

/// Find peer addresses in DHT
pub async fn dht_findpeer(peer_id: &str, format: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};

    let pb = progress::spinner(&format!("Looking up peer {}", peer_id));
    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    // In a full implementation, this would query the DHT
    // For now, we'll show a placeholder response
    progress::finish_spinner_success(&pb, "Lookup complete");

    match format {
        "json" => {
            println!("{{");
            println!("  \"peer_id\": \"{}\",", peer_id);
            println!("  \"addresses\": [],");
            println!("  \"note\": \"DHT lookup placeholder - actual implementation pending\"");
            println!("}}");
        }
        _ => {
            print_header(&format!("Peer: {}", peer_id));
            output::info("DHT peer lookup is a placeholder");
            output::info("Full implementation requires peer routing protocol");
        }
    }

    node.stop().await?;
    Ok(())
}

// ============================================================================
// Bootstrap Commands
// ============================================================================

/// List bootstrap peers
pub async fn bootstrap_list(format: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};

    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    let bootstrap_peers = node.bootstrap_peers()?;

    match format {
        "json" => {
            println!("[");
            for (i, addr) in bootstrap_peers.iter().enumerate() {
                if i > 0 {
                    println!(",");
                }
                print!("  \"{}\"", addr);
            }
            println!();
            println!("]");
        }
        _ => {
            if bootstrap_peers.is_empty() {
                println!("No bootstrap peers configured");
            } else {
                println!("Bootstrap peers ({}):", bootstrap_peers.len());
                for addr in &bootstrap_peers {
                    println!("  {}", addr);
                }
            }
        }
    }

    node.stop().await?;
    Ok(())
}

/// Add a bootstrap peer
pub async fn bootstrap_add(addr: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};

    let pb = progress::spinner(&format!("Adding bootstrap peer {}", addr));
    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    node.add_bootstrap_peer(addr).await?;
    progress::finish_spinner_success(&pb, "Peer added");

    success(&format!("Added bootstrap peer: {}", addr));

    node.stop().await?;
    Ok(())
}

/// Remove a bootstrap peer
pub async fn bootstrap_rm(addr: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};

    let pb = progress::spinner(&format!("Removing bootstrap peer {}", addr));
    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    node.remove_bootstrap_peer(addr).await?;
    progress::finish_spinner_success(&pb, "Peer removed");

    success(&format!("Removed bootstrap peer: {}", addr));

    node.stop().await?;
    Ok(())
}
