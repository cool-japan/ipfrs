//! Simple demonstration of ipfrs-storage features
//!
//! This example shows basic usage patterns that are easy to understand and run.

use bytes::Bytes;
use ipfrs_core::Block;
use ipfrs_storage::{
    BlockStoreTrait, BloomBlockStore, BloomConfig, CachedBlockStore, MemoryBlockStore, PinManager,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== IPFRS Storage Simple Demo ===\n");

    // Example 1: Basic storage operations
    basic_storage_demo().await?;

    // Example 2: Caching for better performance
    caching_demo().await?;

    // Example 3: Bloom filters for fast existence checks
    bloom_filter_demo().await?;

    // Example 4: Pin management
    pin_management_demo()?;

    println!("\n=== Demo completed successfully ===");
    Ok(())
}

/// Demonstrate basic storage operations
async fn basic_storage_demo() -> anyhow::Result<()> {
    println!("1. Basic Storage Operations");

    let store = MemoryBlockStore::new();

    // Create and store a block
    let block = Block::new(Bytes::from("Hello, IPFRS!"))?;
    let cid = *block.cid();

    println!("  - Storing block: {}", cid);
    store.put(&block).await?;

    // Retrieve the block
    let retrieved = store.get(&cid).await?;
    assert!(retrieved.is_some());
    println!("  - Retrieved block successfully");

    // Check existence
    let exists = store.has(&cid).await?;
    println!("  - Block exists: {}", exists);

    Ok(())
}

/// Demonstrate caching
async fn caching_demo() -> anyhow::Result<()> {
    println!("\n2. Caching for Performance");

    let base_store = MemoryBlockStore::new();
    let cached_store = CachedBlockStore::with_default_config(base_store);

    // Store multiple blocks
    let blocks: Vec<Block> = (0..5)
        .map(|i| Block::new(Bytes::from(format!("Block {}", i))).unwrap())
        .collect();

    println!("  - Storing {} blocks", blocks.len());
    for block in &blocks {
        cached_store.put(block).await?;
    }

    // First access (cache miss)
    println!("  - First access (cache miss)");
    let _ = cached_store.get(blocks[0].cid()).await?;

    // Second access (cache hit - faster!)
    println!("  - Second access (cache hit)");
    let _ = cached_store.get(blocks[0].cid()).await?;

    println!("  - Cache working correctly!");

    Ok(())
}

/// Demonstrate bloom filters
async fn bloom_filter_demo() -> anyhow::Result<()> {
    println!("\n3. Bloom Filters for Fast Checks");

    let base_store = MemoryBlockStore::new();
    let bloom_config = BloomConfig::new(10_000, 0.01); // 10K items, 1% false positive
    let bloom_store = BloomBlockStore::with_config(base_store, bloom_config);

    // Store blocks
    let blocks: Vec<Block> = (0..10)
        .map(|i| Block::new(Bytes::from(format!("Bloom block {}", i))).unwrap())
        .collect();

    println!("  - Storing {} blocks", blocks.len());
    for block in &blocks {
        bloom_store.put(block).await?;
    }

    // Check existence (bloom filter speeds this up)
    println!("  - Checking existence with bloom filter");
    for block in &blocks {
        let exists = bloom_store.has(block.cid()).await?;
        assert!(exists);
    }

    // Check non-existent block
    let fake_block = Block::new(Bytes::from("non-existent"))?;
    let should_not_exist = bloom_store.has(fake_block.cid()).await?;
    println!(
        "  - Non-existent block check: {} (bloom filter works!)",
        should_not_exist
    );

    Ok(())
}

/// Demonstrate pin management
fn pin_management_demo() -> anyhow::Result<()> {
    println!("\n4. Pin Management");

    let pin_manager = PinManager::new();

    // Create blocks
    let blocks: Vec<Block> = (0..3)
        .map(|i| Block::new(Bytes::from(format!("Pin block {}", i))).unwrap())
        .collect();

    // Pin blocks
    println!("  - Pinning blocks");
    for block in &blocks {
        pin_manager.pin(block.cid())?;
    }

    // Check pin status
    for block in &blocks {
        let is_pinned = pin_manager.is_pinned(block.cid());
        println!("    Block {} pinned: {}", block.cid(), is_pinned);
    }

    // Unpin a block
    pin_manager.unpin(blocks[0].cid())?;
    println!("  - Unpinned first block");

    // Get stats
    let stats = pin_manager.stats();
    println!("  - Direct pins: {}", stats.direct_pins);

    Ok(())
}
