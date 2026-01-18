//! Connection management example
//!
//! This example demonstrates how to:
//! - Track connected peers
//! - Monitor bandwidth usage
//! - Manage connections
//! - Handle connection events

use ipfrs_network::{NetworkConfig, NetworkNode};
use libp2p::Multiaddr;
use std::str::FromStr;
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== Connection Management Example ===\n");

    // Create network configuration with connection limits
    let config = NetworkConfig {
        max_connections: Some(100),
        max_inbound_connections: Some(50),
        max_outbound_connections: Some(50),
        ..Default::default()
    };

    println!("Creating network node with connection limits:");
    println!("  Max total connections: 100");
    println!("  Max inbound: 50");
    println!("  Max outbound: 50\n");

    // Create and start the network node
    let mut node = NetworkNode::new(config)?;
    node.start().await?;

    println!("Node peer ID: {}", node.peer_id());
    println!("Listening addresses:");
    for addr in node.listeners() {
        println!("  {}", addr);
    }

    // Wait for initialization
    println!("\nWaiting for node to initialize...");
    sleep(Duration::from_secs(2)).await;

    // Scenario 1: Check initial connection status
    println!("\n--- Scenario 1: Initial Connection Status ---");

    let peer_count = node.get_peer_count();
    println!("Currently connected to {} peers", peer_count);

    let stats = node.stats();
    println!("Connected peers: {}", stats.connected_peers);
    println!("QUIC enabled: {}", stats.quic_enabled);

    // Scenario 2: Try to connect to bootstrap nodes
    println!("\n--- Scenario 2: Connect to Bootstrap Nodes ---");

    // Try connecting to public IPFS bootstrap nodes
    let bootstrap_addrs = vec![
        "/dnsaddr/bootstrap.libp2p.io/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN",
        "/dnsaddr/bootstrap.libp2p.io/p2p/QmQCU2EcMqAqQPR2i9bChDtGNJchTbq5TbXJJ16u19uLTa",
    ];

    println!(
        "Attempting to connect to {} bootstrap nodes...",
        bootstrap_addrs.len()
    );

    let mut connected = 0;
    for addr_str in &bootstrap_addrs {
        if let Ok(addr) = Multiaddr::from_str(addr_str) {
            println!("  Connecting to: {}...", addr);
            match node.connect(addr).await {
                Ok(_) => {
                    println!("    Connected successfully!");
                    connected += 1;
                }
                Err(e) => {
                    println!("    Connection failed: {}", e);
                }
            }
            sleep(Duration::from_millis(500)).await;
        }
    }

    println!(
        "Successfully connected to {} out of {} nodes",
        connected,
        bootstrap_addrs.len()
    );

    // Scenario 3: Monitor peer connections
    println!("\n--- Scenario 3: Monitor Peer Connections ---");

    sleep(Duration::from_secs(2)).await;

    let peer_count = node.get_peer_count();
    println!("Currently connected peers: {}", peer_count);

    // Get list of connected peers
    let connected_peers = node.connected_peers();
    println!("Connected peer IDs:");
    for (i, peer) in connected_peers.iter().take(5).enumerate() {
        println!("  {}. {}", i + 1, peer);
    }

    // Scenario 4: Bandwidth tracking
    println!("\n--- Scenario 4: Bandwidth Tracking ---");

    // Simulate some network activity by making DHT queries
    println!("Performing some network operations...");

    let test_peer = node.peer_id();
    for i in 1..=3 {
        println!("  DHT query {}...", i);
        let _ = node.find_node(test_peer).await;
        sleep(Duration::from_secs(1)).await;
    }

    // Get bandwidth statistics
    let bytes_sent = node.get_bytes_sent();
    let bytes_received = node.get_bytes_received();

    println!("\nBandwidth statistics:");
    println!(
        "  Bytes sent: {} bytes ({:.2} KB)",
        bytes_sent,
        bytes_sent as f64 / 1024.0
    );
    println!(
        "  Bytes received: {} bytes ({:.2} KB)",
        bytes_received,
        bytes_received as f64 / 1024.0
    );

    // Update bandwidth (simulating manual tracking)
    println!("\nManually updating bandwidth stats (+1000 sent, +2000 received)...");
    node.update_bandwidth(1000, 2000);

    let bytes_sent = node.get_bytes_sent();
    let bytes_received = node.get_bytes_received();
    println!("Updated bandwidth:");
    println!("  Bytes sent: {}", bytes_sent);
    println!("  Bytes received: {}", bytes_received);

    // Scenario 5: Connection health
    println!("\n--- Scenario 5: Connection Health ---");

    let health = node.get_network_health();
    println!("Network health status: {:?}", health.status);
    println!("Connected peers: {}", health.connected_peers);
    println!("Publicly reachable: {}", health.is_publicly_reachable);
    println!("External addresses: {}", health.external_addresses);

    let is_healthy = node.is_healthy();
    println!("Is healthy: {}", is_healthy);

    // Scenario 6: Check specific peer connection
    println!("\n--- Scenario 6: Check Specific Peer Connections ---");

    let connected_peers = node.connected_peers();
    if let Some(peer) = connected_peers.first() {
        let is_connected = node.is_connected_to(peer);
        println!("Checking if connected to {}...", peer);
        println!("  Connected: {}", is_connected);
    }

    // Scenario 7: Disconnect from all peers (cleanup)
    println!("\n--- Scenario 7: Disconnect from Peers ---");

    println!("Current peer count: {}", node.get_peer_count());

    println!("Disconnecting from all peers...");
    node.disconnect_all().await?;

    sleep(Duration::from_secs(1)).await;

    let peer_count_after = node.get_peer_count();
    println!("Peer count after disconnect: {}", peer_count_after);

    // Final statistics
    println!("\n--- Final Statistics ---");
    let stats = node.stats();
    println!("Peer ID: {}", stats.peer_id);
    println!("Connected peers: {}", stats.connected_peers);
    println!("Bytes sent: {}", stats.bytes_sent);
    println!("Bytes received: {}", stats.bytes_received);
    println!("Bootstrap peers: {}", stats.bootstrap_peers.len());

    println!("\nConnection management example complete!");

    // Cleanup
    node.stop().await?;

    Ok(())
}
