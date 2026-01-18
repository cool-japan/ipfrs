//! Advanced Workflow Example
//!
//! This example demonstrates an advanced end-to-end workflow including:
//! - Peer management with scoring and selection
//! - Session lifecycle management
//! - Circuit breaker pattern for fault tolerance
//!
//! Run with: cargo run --example advanced_workflow

use bytes::Bytes;
use ipfrs_core::Cid;
use ipfrs_transport::{
    CircuitBreaker, CircuitBreakerConfig, ConcurrentPeerManager, PeerId, PeerScoringConfig,
    Priority, Session, SessionConfig,
};
use multihash::Multihash;
use std::time::Duration;

/// Create a dummy CID for demonstration
fn create_cid(seed: u64) -> Cid {
    let data = seed.to_le_bytes();
    let hash = Multihash::wrap(0x12, &data).unwrap();
    Cid::new_v1(0x55, hash)
}

/// Create a dummy PeerId for demonstration
fn create_peer_id(seed: u8) -> PeerId {
    format!("peer-{}", seed)
}

fn main() {
    println!("=== Advanced Workflow Example ===\n");

    // 1. Peer Management with Scoring
    println!("--- Peer Management ---\n");

    let scoring_config = PeerScoringConfig {
        latency_weight: 0.4,
        bandwidth_weight: 0.3,
        reliability_weight: 0.3,
        ewma_alpha: 0.2,
        inactivity_decay: 0.01,
        min_score: 0.1,
        max_failures: 5,
    };

    let peer_manager = ConcurrentPeerManager::new(scoring_config);

    // Add some peers
    let peer1 = create_peer_id(1);
    let peer2 = create_peer_id(2);
    let peer3 = create_peer_id(3);

    peer_manager.add_peer(peer1.clone());
    peer_manager.add_peer(peer2.clone());
    peer_manager.add_peer(peer3.clone());

    // Simulate peer activity with combined metrics
    peer_manager.record_success(&peer1, 10_000_000, Duration::from_millis(10)); // 10 MB, 10ms
    peer_manager.record_success(&peer2, 5_000_000, Duration::from_millis(50)); // 5 MB, 50ms
    peer_manager.record_success(&peer3, 8_000_000, Duration::from_millis(25)); // 8 MB, 25ms

    println!("Added 3 peers with different characteristics:");
    println!("  Peer 1: 10ms latency, 10 MB/s bandwidth");
    println!("  Peer 2: 50ms latency, 5 MB/s bandwidth");
    println!("  Peer 3: 25ms latency, 8 MB/s bandwidth");

    // Display stats
    let stats = peer_manager.stats();
    println!("\nPeer manager statistics:");
    println!("  Total peers: {}", stats.total_peers);
    println!("  Connected peers: {}", stats.connected_peers);

    // 2. Circuit Breaker for Fault Tolerance
    println!("\n--- Circuit Breaker ---\n");

    let cb_config = CircuitBreakerConfig {
        failure_threshold: 3,
        success_threshold: 2,
        timeout: Duration::from_secs(5),
        window_duration: Duration::from_secs(60),
    };

    let circuit_breaker = CircuitBreaker::new(cb_config);

    println!("Circuit breaker initialized:");
    println!("  Failure threshold: 3");
    println!("  Success threshold: 2");
    println!("  Timeout: 5s");
    println!("  Window duration: 60s");

    // Simulate some failures
    circuit_breaker.record_failure();
    circuit_breaker.record_failure();
    println!("\nRecorded 2 failures - circuit still CLOSED");

    circuit_breaker.record_failure();
    println!("Recorded 3rd failure - circuit now OPEN");

    // Try to execute (will fail because circuit is open)
    if circuit_breaker.is_request_allowed() {
        println!("Request allowed");
    } else {
        println!("Request rejected - circuit is OPEN");
    }

    // 3. Session Management
    println!("\n--- Session Management ---\n");

    let session_config = SessionConfig {
        max_concurrent_blocks: 100,
        timeout: Duration::from_secs(60),
        default_priority: Priority::Normal,
        progress_notifications: true,
    };

    let session_id = 1u64;
    let blocks = vec![create_cid(2001), create_cid(2002), create_cid(2003)];

    let session = Session::new(session_id, session_config, None);

    // Add blocks to the session
    session
        .add_blocks(&blocks, None)
        .expect("Failed to add blocks");

    println!("Created session {} with 3 blocks", session_id);

    // Simulate receiving blocks
    let block_data = Bytes::from(vec![0u8; 1024]); // 1 KB block

    for (i, cid) in blocks.iter().enumerate() {
        session
            .mark_received(cid, &block_data)
            .expect("Failed to mark received");

        let stats = session.stats();
        let progress = if stats.total_blocks > 0 {
            (stats.blocks_received as f64 / stats.total_blocks as f64) * 100.0
        } else {
            0.0
        };

        println!(
            "  Block {} received ({} bytes) - Progress: {:.1}%",
            i + 1,
            block_data.len(),
            progress
        );
    }

    println!("\n✓ Advanced workflow completed successfully!");
    println!("\nThis example demonstrated:");
    println!("  • Peer management with scoring and selection");
    println!("  • Circuit breaker pattern for fault tolerance");
    println!("  • Session lifecycle management with progress tracking");
}
