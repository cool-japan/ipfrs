//! DHT operations example
//!
//! This example demonstrates how to:
//! - Announce content to the DHT (provider records)
//! - Find providers for content
//! - Perform DHT queries
//! - Monitor routing table

use cid::Cid;
use ipfrs_network::{NetworkConfig, NetworkNode};
use std::str::FromStr;
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== DHT Operations Example ===\n");

    // Create network configuration
    let config = NetworkConfig::default();

    // Create and start the network node
    println!("Creating and starting network node...");
    let mut node = NetworkNode::new(config)?;
    node.start().await?;

    println!("Node peer ID: {}", node.peer_id());
    println!("Listening addresses:");
    for addr in node.listeners() {
        println!("  {}", addr);
    }

    // Wait for node to initialize
    println!("\nWaiting for node to initialize and bootstrap...");
    sleep(Duration::from_secs(3)).await;

    // Bootstrap the DHT
    println!("\nBootstrapping DHT...");
    node.bootstrap_dht().await?;
    sleep(Duration::from_secs(2)).await;

    // Scenario 1: Announce content to DHT
    println!("\n--- Scenario 1: Announce Content to DHT ---");

    // Create a test CID (this would normally be the CID of actual content)
    let test_cid_str = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi";
    let cid = Cid::from_str(test_cid_str)?;

    println!("Announcing content with CID: {}", cid);
    node.provide(&cid).await?;

    println!("Content announced! This node is now a provider for this CID.");
    println!("Other nodes can find this content by querying the DHT.");

    // Wait a moment for the announcement to propagate
    sleep(Duration::from_secs(2)).await;

    // Scenario 2: Find providers for content
    println!("\n--- Scenario 2: Find Providers for Content ---");

    println!("Searching for providers of CID: {}", cid);
    println!("(Note: Results will come through events)");
    node.find_providers(&cid).await?;

    // Wait for results
    sleep(Duration::from_secs(2)).await;

    // Scenario 3: Find a node in the DHT
    println!("\n--- Scenario 3: DHT Node Lookup ---");

    // Try to find ourselves (this demonstrates the DHT lookup mechanism)
    let target_peer = node.peer_id();
    println!("Looking up peer: {}", target_peer);
    println!("(Note: Results will come through events)");

    node.find_node(target_peer).await?;

    // Wait for results
    sleep(Duration::from_secs(2)).await;

    // Scenario 4: Get routing table info
    println!("\n--- Scenario 4: Routing Table Information ---");

    let rt_info = node.get_routing_table_info()?;
    println!("Routing table:");
    println!("  Total peers: {}", rt_info.total_peers);
    println!("  Number of buckets: {}", rt_info.num_buckets);

    if !rt_info.buckets.is_empty() {
        println!("\n  Bucket details (first 5):");
        for bucket in rt_info.buckets.iter().take(5) {
            println!("    Bucket {}: {} peers", bucket.index, bucket.num_entries);
        }
    }

    // Scenario 5: Get closest local peers
    println!("\n--- Scenario 5: Get Closest Local Peers ---");

    println!("Querying for closest peers to our local peer ID...");
    match node.get_closest_local_peers().await {
        Ok(peers) => {
            println!("Found {} closest peers:", peers.len());
            for (i, peer) in peers.iter().take(5).enumerate() {
                println!("  {}. {}", i + 1, peer);
            }
        }
        Err(e) => {
            println!("Error getting closest peers: {}", e);
        }
    }

    // Scenario 6: Announce multiple content items
    println!("\n--- Scenario 6: Announce Multiple Content Items ---");

    let cids = [
        "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi",
        "bafybeihkoviema7g3gxyt6la7vd5ho32ictqbilu3wnlo3rs7ewhnp7lly",
        "bafybeihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku",
    ];

    println!("Announcing {} content items...", cids.len());
    for (i, cid_str) in cids.iter().enumerate() {
        if let Ok(cid) = Cid::from_str(cid_str) {
            println!("  {}: {}", i + 1, cid);
            if let Err(e) = node.provide(&cid).await {
                println!("    Error: {}", e);
            } else {
                println!("    Announced successfully");
            }
            sleep(Duration::from_millis(500)).await;
        }
    }

    // Final statistics
    println!("\n--- Final Statistics ---");
    let stats = node.stats();
    println!("Connected peers: {}", stats.connected_peers);
    println!("Bytes sent: {}", stats.bytes_sent);
    println!("Bytes received: {}", stats.bytes_received);

    // Get updated routing table
    let rt_info = node.get_routing_table_info()?;
    println!("Routing table peers: {}", rt_info.total_peers);

    println!("\nDHT operations complete!");

    // Cleanup
    node.stop().await?;

    Ok(())
}
