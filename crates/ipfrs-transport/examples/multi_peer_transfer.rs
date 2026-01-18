//! Multi-Peer Transfer Example
//!
//! This example demonstrates a realistic multi-peer block transfer scenario including:
//! - Multi-peer setup with different characteristics
//! - Peer selection strategies (fastest-first, best-score)
//! - Want list management with priorities
//! - Session-based transfer with progress tracking
//! - Network partition detection and recovery
//! - Circuit breaker for fault tolerance
//! - Performance metrics tracking
//!
//! Run with: cargo run --example multi_peer_transfer

use bytes::Bytes;
use ipfrs_core::Cid;
use ipfrs_transport::{
    CircuitBreaker, CircuitBreakerConfig, ConcurrentPeerManager, ConcurrentWantList,
    LatencyTracker, PartitionConfig, PartitionDetector, PeerId, PeerScoringConfig, Priority,
    SelectionStrategy, Session, SessionConfig, SessionEvent, Timer, WantListConfig,
};
use multihash::Multihash;
use std::net::SocketAddr;
use std::time::Duration;
use tokio::sync::mpsc;

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
    println!("=== Multi-Peer Transfer Example ===\n");

    // 1. Setup Peer Manager with Multiple Peers
    println!("--- Setting up Peer Manager ---\n");

    let scoring_config = PeerScoringConfig {
        latency_weight: 0.4,
        bandwidth_weight: 0.4,
        reliability_weight: 0.2,
        ewma_alpha: 0.2,
        inactivity_decay: 0.01,
        min_score: 0.1,
        max_failures: 5,
    };

    let peer_manager = ConcurrentPeerManager::new(scoring_config);

    // Add peers with different characteristics
    let fast_peer = create_peer_id(1);
    let medium_peer = create_peer_id(2);
    let slow_peer = create_peer_id(3);
    let unreliable_peer = create_peer_id(4);

    peer_manager.add_peer(fast_peer.clone());
    peer_manager.add_peer(medium_peer.clone());
    peer_manager.add_peer(slow_peer.clone());
    peer_manager.add_peer(unreliable_peer.clone());

    // Simulate peer performance characteristics
    // Fast peer: low latency, high bandwidth
    peer_manager.record_success(&fast_peer, 20_000_000, Duration::from_millis(5)); // 20 MB, 5ms
    peer_manager.record_success(&fast_peer, 20_000_000, Duration::from_millis(5));

    // Medium peer: medium latency, medium bandwidth
    peer_manager.record_success(&medium_peer, 10_000_000, Duration::from_millis(20)); // 10 MB, 20ms
    peer_manager.record_success(&medium_peer, 10_000_000, Duration::from_millis(20));

    // Slow peer: high latency, low bandwidth
    peer_manager.record_success(&slow_peer, 5_000_000, Duration::from_millis(50)); // 5 MB, 50ms

    // Unreliable peer: some failures
    peer_manager.record_success(&unreliable_peer, 15_000_000, Duration::from_millis(10));
    peer_manager.record_failure(&unreliable_peer);
    peer_manager.record_failure(&unreliable_peer);

    println!("Added 4 peers:");
    println!("  • Fast peer: 5ms latency, 20 MB/transfer");
    println!("  • Medium peer: 20ms latency, 10 MB/transfer");
    println!("  • Slow peer: 50ms latency, 5 MB/transfer");
    println!("  • Unreliable peer: 10ms latency but 2 failures");

    let stats = peer_manager.stats();
    println!("\nPeer manager stats:");
    println!("  Total peers: {}", stats.total_peers);
    println!("  Connected peers: {}", stats.connected_peers);
    println!("  Blacklisted peers: {}", stats.blacklisted_peers);

    // 2. Test Peer Selection Strategies
    println!("\n--- Testing Peer Selection Strategies ---\n");

    let dummy_cid = create_cid(9999);

    let best_peers = peer_manager.select_peers(&dummy_cid, 1, SelectionStrategy::BestScore);
    if !best_peers.is_empty() {
        println!("Best score peer: {}", best_peers[0]);
    }

    let fastest_peers = peer_manager.select_peers(&dummy_cid, 1, SelectionStrategy::FastestFirst);
    if !fastest_peers.is_empty() {
        println!("Fastest peer: {}", fastest_peers[0]);
    }

    let bandwidth_peers =
        peer_manager.select_peers(&dummy_cid, 1, SelectionStrategy::HighestBandwidth);
    if !bandwidth_peers.is_empty() {
        println!("Highest bandwidth peer: {}", bandwidth_peers[0]);
    }

    // 3. Setup Want List for Block Transfer
    println!("\n--- Setting up Want List ---\n");

    let want_config = WantListConfig {
        max_wants: 1000,
        default_timeout: Duration::from_secs(60),
        max_retries: 3,
        base_retry_delay: Duration::from_millis(100),
        max_retry_delay: Duration::from_secs(10),
    };

    let want_list = ConcurrentWantList::new(want_config);

    // Add blocks with different priorities
    let critical_blocks: Vec<Cid> = (0..5).map(|i| create_cid(1000 + i)).collect();
    let high_priority_blocks: Vec<Cid> = (0..10).map(|i| create_cid(2000 + i)).collect();
    let normal_blocks: Vec<Cid> = (0..20).map(|i| create_cid(3000 + i)).collect();

    for cid in &critical_blocks {
        want_list.add_simple(*cid, 1000); // Critical priority
    }
    for cid in &high_priority_blocks {
        want_list.add_simple(*cid, 750); // High priority
    }
    for cid in &normal_blocks {
        want_list.add_simple(*cid, 500); // Normal priority
    }

    println!("Added blocks to want list:");
    println!("  • {} critical priority blocks", critical_blocks.len());
    println!("  • {} high priority blocks", high_priority_blocks.len());
    println!("  • {} normal priority blocks", normal_blocks.len());
    println!("  Total wants: {}", want_list.len());

    // 4. Create Session for Organized Transfer
    println!("\n--- Creating Transfer Session ---\n");

    let session_config = SessionConfig {
        max_concurrent_blocks: 100,
        timeout: Duration::from_secs(120),
        default_priority: Priority::Normal,
        progress_notifications: true,
    };

    let session_id = 42u64;
    let (tx, mut rx) = mpsc::unbounded_channel();
    let session = Session::new(session_id, session_config, Some(tx));

    // Add all blocks to session
    let all_blocks: Vec<Cid> = critical_blocks
        .iter()
        .chain(high_priority_blocks.iter())
        .chain(normal_blocks.iter())
        .copied()
        .collect();

    session
        .add_blocks(&all_blocks, Some(Priority::High))
        .expect("Failed to add blocks");

    println!("Created session {}", session_id);
    println!(
        "  Total blocks in session: {}",
        session.stats().total_blocks
    );

    // 5. Setup Circuit Breaker for Fault Tolerance
    println!("\n--- Setting up Circuit Breaker ---\n");

    let cb_config = CircuitBreakerConfig {
        failure_threshold: 3,
        success_threshold: 2,
        timeout: Duration::from_secs(5),
        window_duration: Duration::from_secs(60),
    };

    println!("Circuit breaker configured:");
    println!("  Failure threshold: {}", cb_config.failure_threshold);
    println!("  Success threshold: {}", cb_config.success_threshold);
    println!("  Timeout: {:?}", cb_config.timeout);

    let circuit_breaker = CircuitBreaker::new(cb_config);

    // 6. Setup Partition Detector
    println!("\n--- Setting up Partition Detector ---\n");

    let partition_config = PartitionConfig {
        failure_threshold: 3,
        failure_window: Duration::from_secs(60),
        probe_interval: Duration::from_secs(10),
        max_queued_requests: 1000,
        recovery_probe_count: 2,
        peer_timeout: Duration::from_secs(30),
    };

    println!("Partition detector configured:");
    println!(
        "  Failure threshold: {}",
        partition_config.failure_threshold
    );
    println!(
        "  Recovery probe count: {}",
        partition_config.recovery_probe_count
    );

    let partition_detector = PartitionDetector::new(partition_config);

    // 7. Setup Performance Metrics
    println!("\n--- Setting up Performance Metrics ---\n");

    let latency_tracker = LatencyTracker::new();

    // 8. Simulate Block Transfer
    println!("\n--- Simulating Block Transfer ---\n");

    let mut blocks_received = 0;
    let total_blocks = 35;
    let block_data = Bytes::from(vec![0u8; 1024 * 100]); // 100 KB blocks

    // Process first 10 blocks successfully
    for cid in all_blocks.iter().take(10) {
        let timer = Timer::start();

        // Simulate network delay
        std::thread::sleep(Duration::from_micros(100));

        timer.stop_and_record(&latency_tracker);

        session
            .mark_received(cid, &block_data)
            .expect("Failed to mark received");

        blocks_received += 1;
        circuit_breaker.record_success();

        // Check for session events
        while let Ok(event) = rx.try_recv() {
            match event {
                SessionEvent::BlockReceived { cid, .. } => {
                    println!("  ✓ Block received: {}", cid);
                }
                SessionEvent::Progress { stats, .. } => {
                    let progress = if stats.total_blocks > 0 {
                        (stats.blocks_received as f64 / stats.total_blocks as f64) * 100.0
                    } else {
                        0.0
                    };
                    println!(
                        "  Progress: {}/{} blocks ({:.1}%)",
                        stats.blocks_received, stats.total_blocks, progress
                    );
                }
                _ => {}
            }
        }
    }

    println!("\nTransfer progress:");
    println!("  Blocks received: {}/{}", blocks_received, total_blocks);

    // 9. Simulate Network Issue
    println!("\n--- Simulating Network Issue ---\n");

    // Create a peer socket address for partition detector
    let peer_addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();

    // Record some failures
    for _ in 0..3 {
        circuit_breaker.record_failure();
        partition_detector.record_failure(&peer_addr);
    }

    println!("Recorded 3 consecutive failures");
    println!(
        "  Circuit breaker open: {}",
        !circuit_breaker.is_request_allowed()
    );

    let partition_stats = partition_detector.stats();
    println!("  Partition detector stats:");
    println!(
        "    Partitions detected: {}",
        partition_stats.partitions_detected
    );
    println!("    Queued requests: {}", partition_stats.queued_requests);

    // 10. Simulate Recovery
    println!("\n--- Simulating Recovery ---\n");

    // Wait for circuit breaker timeout
    println!("Waiting for circuit breaker to transition to half-open...");
    std::thread::sleep(Duration::from_millis(100)); // Shortened for demo

    // Record successes for recovery
    for _ in 0..2 {
        circuit_breaker.record_success();
        partition_detector.record_success(&peer_addr);
    }

    println!("Recorded 2 successful operations");
    println!(
        "  Circuit breaker closed: {}",
        circuit_breaker.is_request_allowed()
    );

    // 11. Display Final Statistics
    println!("\n--- Final Statistics ---\n");

    let session_stats = session.stats();
    println!("Session statistics:");
    println!("  Total blocks: {}", session_stats.total_blocks);
    println!("  Blocks received: {}", session_stats.blocks_received);
    println!(
        "  Total bytes: {} KB",
        session_stats.bytes_transferred / 1024
    );

    let progress = if session_stats.total_blocks > 0 {
        (session_stats.blocks_received as f64 / session_stats.total_blocks as f64) * 100.0
    } else {
        0.0
    };
    println!("  Progress: {:.1}%", progress);

    let latency_stats = latency_tracker.stats();
    println!("\nLatency statistics:");
    println!("  Mean: {:?}", latency_stats.mean);
    println!("  P50: {:?}", latency_stats.p50);
    println!("  P95: {:?}", latency_stats.p95);
    println!("  P99: {:?}", latency_stats.p99);

    let peer_stats = peer_manager.stats();
    println!("\nPeer manager statistics:");
    println!("  Total peers: {}", peer_stats.total_peers);
    println!("  Connected peers: {}", peer_stats.connected_peers);
    println!("  Blacklisted peers: {}", peer_stats.blacklisted_peers);

    let cb_stats = circuit_breaker.stats();
    println!("\nCircuit breaker statistics:");
    println!("  Failure count: {}", cb_stats.failure_count);
    println!("  Success count: {}", cb_stats.success_count);
    println!("  Window failures: {}", cb_stats.window_failures);

    println!("\n✓ Multi-peer transfer example completed successfully!");
    println!("\nThis example demonstrated:");
    println!("  • Multi-peer setup with different performance characteristics");
    println!("  • Peer selection strategies (BestScore, FastestFirst, HighestBandwidth)");
    println!("  • Want list management with multiple priority levels");
    println!("  • Session-based transfer with progress tracking");
    println!("  • Circuit breaker pattern for fault tolerance");
    println!("  • Network partition detection and recovery");
    println!("  • Performance metrics tracking (latency percentiles)");
}
