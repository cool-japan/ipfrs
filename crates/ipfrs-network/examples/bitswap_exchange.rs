//! Bitswap Protocol Usage Example
//!
//! This example demonstrates how to use the Bitswap protocol for block exchange:
//! 1. Creating Bitswap instances
//! 2. Managing want/have lists
//! 3. Handling block requests and exchanges between peers
//! 4. Processing Bitswap events
//! 5. Tracking exchange statistics

use ipfrs_network::{Bitswap, BitswapEvent, BitswapMessage};
use libp2p::PeerId;
use multihash_codetable::{Code, MultihashDigest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("=== Bitswap Protocol Example ===\n");

    // Scenario: Two peers exchanging blocks

    println!("1. Creating Bitswap instances for two peers");
    let mut peer1_bitswap = Bitswap::new();
    let mut peer2_bitswap = Bitswap::new();

    let peer1_id = PeerId::random();
    let peer2_id = PeerId::random();

    println!("   Peer 1: {}", peer1_id);
    println!("   Peer 2: {}\n", peer2_id);

    // Create test CIDs
    let block1_hash = Code::Sha2_256.digest(b"Hello, IPFS!");
    let block1_cid = cid::Cid::new_v1(0x55, block1_hash);
    let block1_data = b"Hello, IPFS!".to_vec();

    let block2_hash = Code::Sha2_256.digest(b"Distributed storage is awesome");
    let block2_cid = cid::Cid::new_v1(0x55, block2_hash);
    let block2_data = b"Distributed storage is awesome".to_vec();

    println!("2. Initial setup");
    println!("   Peer 1 has block: {}", block1_cid);
    println!("   Peer 2 has block: {}\n", block2_cid);

    // Peer 1 has block1, wants block2
    peer1_bitswap.have(block1_cid).await?;
    peer1_bitswap.want(block2_cid).await?;

    // Peer 2 has block2, wants block1
    peer2_bitswap.have(block2_cid).await?;
    peer2_bitswap.want(block1_cid).await?;

    // Get event receivers
    let mut peer1_events = peer1_bitswap
        .take_event_receiver()
        .expect("Failed to get event receiver");
    let mut peer2_events = peer2_bitswap
        .take_event_receiver()
        .expect("Failed to get event receiver");

    println!("3. Peer 1 sends Want message for block2 to Peer 2");
    let response = peer2_bitswap
        .handle_message(BitswapMessage::Want(block2_cid), peer1_id)
        .await?;

    match response {
        Some(BitswapMessage::Have(cid)) => {
            println!("   ✓ Peer 2 responded: Has block {}\n", cid);
        }
        Some(BitswapMessage::DontHave(cid)) => {
            println!("   ✗ Peer 2 responded: Doesn't have block {}\n", cid);
        }
        _ => println!("   No response\n"),
    }

    println!("4. Peer 2 sends Want message for block1 to Peer 1");
    let response = peer1_bitswap
        .handle_message(BitswapMessage::Want(block1_cid), peer2_id)
        .await?;

    match response {
        Some(BitswapMessage::Have(cid)) => {
            println!("   ✓ Peer 1 responded: Has block {}\n", cid);
        }
        Some(BitswapMessage::DontHave(cid)) => {
            println!("   ✗ Peer 1 responded: Doesn't have block {}\n", cid);
        }
        _ => println!("   No response\n"),
    }

    println!("5. Peer 2 sends block2 to Peer 1");
    let block_message = peer2_bitswap
        .send_block(block2_cid, block2_data.clone(), peer1_id)
        .await?;

    peer1_bitswap
        .handle_message(block_message, peer2_id)
        .await?;

    println!("   ✓ Block sent and received\n");

    println!("6. Peer 1 sends block1 to Peer 2");
    let block_message = peer1_bitswap
        .send_block(block1_cid, block1_data.clone(), peer2_id)
        .await?;

    peer2_bitswap
        .handle_message(block_message, peer1_id)
        .await?;

    println!("   ✓ Block sent and received\n");

    println!("7. Processing events");

    // Process Peer 1's events
    println!("   Peer 1 events:");
    while let Ok(event) = peer1_events.try_recv() {
        match event {
            BitswapEvent::BlockReceived { cid, data, from } => {
                println!(
                    "     - Received block {} ({} bytes) from {}",
                    cid,
                    data.len(),
                    from
                );
            }
            BitswapEvent::BlockSent { cid, to } => {
                println!("     - Sent block {} to {}", cid, to);
            }
            BitswapEvent::BlockRequested { cid, from } => {
                println!("     - Block {} requested by {}", cid, from);
            }
            BitswapEvent::BlockNotFound { cid, peer } => {
                println!("     - Block {} not found at peer {}", cid, peer);
            }
        }
    }

    // Process Peer 2's events
    println!("   Peer 2 events:");
    while let Ok(event) = peer2_events.try_recv() {
        match event {
            BitswapEvent::BlockReceived { cid, data, from } => {
                println!(
                    "     - Received block {} ({} bytes) from {}",
                    cid,
                    data.len(),
                    from
                );
            }
            BitswapEvent::BlockSent { cid, to } => {
                println!("     - Sent block {} to {}", cid, to);
            }
            BitswapEvent::BlockRequested { cid, from } => {
                println!("     - Block {} requested by {}", cid, from);
            }
            BitswapEvent::BlockNotFound { cid, peer } => {
                println!("     - Block {} not found at peer {}", cid, peer);
            }
        }
    }

    println!();

    println!("8. Final statistics");

    let stats1 = peer1_bitswap.stats().await;
    println!("   Peer 1:");
    println!("     - Want list size: {}", stats1.want_list_size);
    println!("     - Have list size: {}", stats1.have_list_size);
    println!("     - Pending requests: {}", stats1.pending_requests);
    println!(
        "     - Peers with pending: {}",
        stats1.peers_with_pending_requests
    );

    let stats2 = peer2_bitswap.stats().await;
    println!("   Peer 2:");
    println!("     - Want list size: {}", stats2.want_list_size);
    println!("     - Have list size: {}", stats2.have_list_size);
    println!("     - Pending requests: {}", stats2.pending_requests);
    println!(
        "     - Peers with pending: {}",
        stats2.peers_with_pending_requests
    );

    println!("\n=== Example completed successfully ===");

    Ok(())
}
