//! Basic Storage Example
//!
//! This example demonstrates fundamental IPFRS operations:
//! - Creating a node
//! - Adding content
//! - Retrieving content
//! - Working with files

use ipfrs::{Node, NodeConfig};

#[tokio::main]
async fn main() -> ipfrs::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== IPFRS Basic Storage Example ===\n");

    // Create a new node with default configuration
    let mut node = Node::new(NodeConfig::default())?;
    println!("✓ Node created");

    // Start the node
    node.start().await?;
    println!("✓ Node started");

    // Example 1: Add bytes directly
    println!("\n--- Example 1: Adding Bytes ---");
    let content = b"Hello, IPFRS! This is my first content.";
    let cid = node.add_bytes(&content[..]).await?;
    println!("Added content with CID: {}", cid);

    // Retrieve the content
    if let Some(data) = node.get(&cid).await? {
        let retrieved = String::from_utf8_lossy(&data);
        println!("Retrieved: {}", retrieved);
        assert_eq!(retrieved.as_bytes(), content);
        println!("✓ Content verified!");
    }

    // Example 2: Check if block exists
    println!("\n--- Example 2: Checking Block Existence ---");
    let exists = node.has_block(&cid).await?;
    println!("Block {} exists: {}", cid, exists);

    // Example 3: Get block statistics
    println!("\n--- Example 3: Block Statistics ---");
    if let Some(stat) = node.block_stat(&cid).await? {
        println!("Block CID: {}", stat.cid);
        println!("Block size: {} bytes", stat.size);
    }

    // Example 4: Batch operations
    println!("\n--- Example 4: Batch Operations ---");
    let content1 = b"First batch item";
    let content2 = b"Second batch item";
    let content3 = b"Third batch item";

    let cid1 = node.add_bytes(&content1[..]).await?;
    let cid2 = node.add_bytes(&content2[..]).await?;
    let cid3 = node.add_bytes(&content3[..]).await?;

    // Check and retrieve blocks individually
    let cids = [cid1, cid2, cid3];

    for cid in cids.iter() {
        let exists = node.has_block(cid).await?;
        println!("Block {}: exists={}", cid, exists);

        if let Some(data) = node.get(cid).await? {
            let content = String::from_utf8_lossy(&data);
            println!("  Content: \"{}\"", content);
        }
    }

    // Example 5: Storage statistics
    println!("\n--- Example 5: Storage Statistics ---");
    let stats = node.storage_stats()?;
    println!("Total blocks: {}", stats.num_blocks);
    println!("Storage is empty: {}", stats.is_empty);

    // Example 6: Node status
    println!("\n--- Example 6: Node Status ---");
    let status = node.status();
    println!("Node running: {}", status.running);
    println!("Storage enabled: {}", status.storage_enabled);
    println!("Network enabled: {}", status.network_enabled);

    // Clean shutdown
    println!("\n--- Shutting Down ---");
    node.stop().await?;
    println!("✓ Node stopped");

    println!("\n=== Example Complete ===");
    Ok(())
}
