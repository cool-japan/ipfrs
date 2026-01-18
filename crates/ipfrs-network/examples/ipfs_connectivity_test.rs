//! Example: Testing connectivity to the public IPFS network
//!
//! This example demonstrates how to test IPFS compatibility and connectivity.

use ipfrs_network::{ipfs_test_config, test_ipfs_connectivity, NetworkNode};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== IPFS Connectivity Test ===\n");

    // Create network configuration optimized for IPFS testing
    let config = ipfs_test_config();
    println!(
        "Bootstrap nodes configured: {}",
        config.bootstrap_peers.len()
    );

    // Create network node
    let mut node = NetworkNode::new(config)?;
    println!("Network node created");

    // Start the node
    node.start().await?;
    println!("Network node started\n");

    // Wait a moment for the node to initialize
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Run IPFS compatibility test
    println!("Starting IPFS compatibility test...\n");
    let timeout = Duration::from_secs(30);

    match test_ipfs_connectivity(&mut node, timeout).await {
        Ok(results) => {
            println!("\n{}", "=".repeat(60));
            println!("{}", results.summary());
            println!("{}", "=".repeat(60));

            if results.all_passed() {
                println!("\n✅ All IPFS compatibility tests passed!");
            } else {
                println!("\n⚠️  Some tests failed or had errors:");
                for error in &results.errors {
                    println!("  - {}", error);
                }
            }

            // Show detailed results
            println!("\nDetailed Results:");
            println!(
                "  Bootstrap: {}",
                if results.bootstrap_connected {
                    "✅"
                } else {
                    "❌"
                }
            );
            println!("  Connected nodes: {}", results.connected_ipfs_nodes);
            println!(
                "  DHT queries: {}",
                if results.dht_queries_work {
                    "✅"
                } else {
                    "❌"
                }
            );
            println!(
                "  Identify protocol: {}",
                if results.identify_protocol_works {
                    "✅"
                } else {
                    "❌"
                }
            );
            println!(
                "  Ping protocol: {}",
                if results.ping_protocol_works {
                    "✅"
                } else {
                    "❌"
                }
            );
            println!(
                "  Provider records: {}",
                if results.provider_records_work {
                    "✅"
                } else {
                    "❌"
                }
            );
            println!("  Test duration: {:?}", results.test_duration);
        }
        Err(e) => {
            eprintln!("Error running IPFS compatibility test: {}", e);
            return Err(e.into());
        }
    }

    // Show network statistics
    println!("\nNetwork Statistics:");
    let stats = node.stats();
    println!("  Connected peers: {}", stats.connected_peers);
    println!("  Bytes sent: {}", stats.bytes_sent);
    println!("  Bytes received: {}", stats.bytes_received);

    // Check network health
    let health = node.get_network_health();
    println!("\nNetwork Health: {:?}", health.status);

    println!("\nTest complete!");

    Ok(())
}
