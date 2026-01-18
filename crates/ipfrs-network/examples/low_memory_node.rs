//! Example: Low-Memory Network Node
//!
//! This example demonstrates how to configure a network node for
//! memory-constrained devices (< 128 MB RAM) such as embedded systems.

use ipfrs_network::{NetworkConfig, NetworkNode, PeerStoreConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== IPFRS Low-Memory Network Node Example ===\n");

    // Create low-memory configuration
    let config = NetworkConfig::low_memory();

    println!("Low-Memory Configuration:");
    println!("  Max connections: {:?}", config.max_connections);
    println!("  Max inbound: {:?}", config.max_inbound_connections);
    println!("  Max outbound: {:?}", config.max_outbound_connections);
    println!("  Buffer size: {} bytes", config.connection_buffer_size);
    println!("  Low-memory mode: {}", config.low_memory_mode);
    println!(
        "  DHT replication factor: {}",
        config.kademlia.replication_factor
    );
    println!("  DHT k-bucket size: {}", config.kademlia.kbucket_size);
    println!();

    // Create peer store with low-memory configuration
    let peer_config = PeerStoreConfig::low_memory();
    println!("Peer Store Configuration:");
    println!("  Max peers: {}", peer_config.max_peers);
    println!(
        "  Max addresses per peer: {}",
        peer_config.max_addrs_per_peer
    );
    println!("  Max latency samples: {}", peer_config.max_latency_samples);
    println!(
        "  Max protocols per peer: {}",
        peer_config.max_protocols_per_peer
    );
    println!();

    // Create and start network node
    println!("Starting network node...");
    let mut node = NetworkNode::new(config)?;
    node.start().await?;

    println!("Node started successfully!");
    println!("Peer ID: {}", node.peer_id());
    println!();

    // Display network health
    let health = node.get_network_health();
    println!("Network Health:");
    println!("  Status: {:?}", health.status);
    println!("  Connected peers: {}", health.connected_peers);
    println!("  Publicly reachable: {}", health.is_publicly_reachable);
    println!();

    // Get network statistics
    let stats = node.stats();
    println!("Network Statistics:");
    println!("  Connected peers: {}", stats.connected_peers);
    println!("  Bytes sent: {}", stats.bytes_sent);
    println!("  Bytes received: {}", stats.bytes_received);
    println!();

    println!("Node is running with minimal memory footprint!");
    println!("Press Ctrl+C to stop...");

    // Keep the node running
    tokio::signal::ctrl_c().await?;

    println!("\nShutting down...");
    Ok(())
}
