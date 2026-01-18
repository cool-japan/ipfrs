//! Basic network node example
//!
//! This example demonstrates how to:
//! - Create a basic IPFRS network node
//! - Configure network settings
//! - Start the node and listen for connections
//! - Check network health
//! - Get node statistics

use ipfrs_network::{NetworkConfig, NetworkNode};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== Basic IPFRS Network Node Example ===\n");

    // Create network configuration
    let config = NetworkConfig {
        listen_addrs: vec![
            "/ip4/0.0.0.0/udp/0/quic-v1".to_string(),
            "/ip4/0.0.0.0/tcp/0".to_string(),
        ],
        enable_quic: true,
        enable_mdns: true,
        enable_nat_traversal: true,
        ..Default::default()
    };

    println!("Creating network node with configuration:");
    println!("  - QUIC transport: enabled");
    println!("  - TCP fallback: enabled");
    println!("  - mDNS discovery: enabled");
    println!("  - NAT traversal: enabled\n");

    // Create the network node
    let mut node = NetworkNode::new(config)?;

    // Get the peer ID
    let peer_id = node.peer_id();
    println!("Node peer ID: {}\n", peer_id);

    // Start the node
    println!("Starting network node...");
    node.start().await?;

    // Get listening addresses
    let addrs = node.listeners();
    println!("\nListening on {} addresses:", addrs.len());
    for addr in &addrs {
        println!("  {}", addr);
    }

    // Wait a moment for the node to initialize
    println!("\nWaiting for node to initialize...");
    sleep(Duration::from_secs(2)).await;

    // Check network health
    let health = node.get_network_health();
    println!("\nNetwork Health:");
    println!("  Status: {:?}", health.status);
    println!("  Connected peers: {}", health.connected_peers);
    println!("  Publicly reachable: {}", health.is_publicly_reachable);
    println!("  External addresses: {}", health.external_addresses);

    // Get network statistics
    let stats = node.stats();
    println!("\nNetwork Statistics:");
    println!("  Peer ID: {}", stats.peer_id);
    println!("  Connected peers: {}", stats.connected_peers);
    println!("  QUIC enabled: {}", stats.quic_enabled);
    println!("  Bytes sent: {}", stats.bytes_sent);
    println!("  Bytes received: {}", stats.bytes_received);
    println!("  Bootstrap peers: {}", stats.bootstrap_peers.len());

    // Check if publicly reachable
    let is_public = node.is_publicly_reachable();
    println!("\nPublicly reachable: {}", is_public);

    if is_public {
        let external_addrs = node.get_external_addresses();
        println!("External addresses:");
        for addr in &external_addrs {
            println!("  {}", addr);
        }
    }

    // Check if node is healthy
    let is_healthy = node.is_healthy();
    println!("\nNode is healthy: {}", is_healthy);

    // Keep the node running for a while
    println!("\nNode is running. Press Ctrl+C to stop.");
    println!("Keeping node alive for 30 seconds...\n");

    for i in 1..=6 {
        sleep(Duration::from_secs(5)).await;

        // Check stats periodically
        let peer_count = node.get_peer_count();
        println!("After {} seconds: {} peers connected", i * 5, peer_count);
    }

    println!("\nShutting down node...");
    node.stop().await?;
    println!("Node stopped successfully!");

    Ok(())
}
