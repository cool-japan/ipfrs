//! Demonstration of utility helper functions
//!
//! This example shows how to use the various utility functions provided by ipfrs-transport
//! for common tasks like batch operations, configuration validation, and performance tuning.

use ipfrs_core::Cid;
use ipfrs_transport::{
    all_wants_present, any_want_present, bulk_add_wants, bulk_remove_wants, bulk_update_priorities,
    calculate_optimal_chunk_size, calculate_optimal_concurrency, create_balanced_peer_scoring,
    create_bandwidth_optimized_peer_manager, create_bulk_transfer_session,
    create_high_throughput_want_list, create_interactive_session,
    create_latency_optimized_peer_manager, create_low_latency_want_list,
    create_reliability_focused_scoring, estimate_transfer_time, format_bandwidth, format_bytes,
    validate_peer_scoring_config, validate_session_config, validate_want_list_config,
    SessionConfig, WantListConfig,
};
use multihash::Multihash;
use std::time::Duration;

/// Create a dummy CID for demonstration
fn demo_cid(seed: u64) -> Cid {
    let data = seed.to_le_bytes();
    let hash = Multihash::wrap(0x12, &data).unwrap();
    Cid::new_v1(0x55, hash)
}

fn main() {
    println!("=== IPFRS Transport Utility Functions Demo ===\n");

    // 1. Quick setup helpers
    println!("1. Quick Setup Helpers");
    println!("   Creating optimized want lists...");
    let _high_throughput_list = create_high_throughput_want_list();
    println!("   ✓ High-throughput want list created (max: 10000 wants, 120s timeout)");

    let _low_latency_list = create_low_latency_want_list();
    println!("   ✓ Low-latency want list created (max: 1000 wants, 30s timeout)");

    println!("\n   Creating optimized peer managers...");
    let _latency_optimized = create_latency_optimized_peer_manager();
    println!("   ✓ Latency-optimized peer manager (latency weight: 60%)");

    let _bandwidth_optimized = create_bandwidth_optimized_peer_manager();
    println!("   ✓ Bandwidth-optimized peer manager (bandwidth weight: 60%)");

    println!("\n   Creating session configurations...");
    let _bulk_session = create_bulk_transfer_session(1);
    println!("   ✓ Bulk transfer session (5 min timeout, 500 concurrent blocks)");

    let _interactive_session = create_interactive_session(2);
    println!("   ✓ Interactive session (1 min timeout, 100 concurrent blocks)");

    // 2. Batch operations
    println!("\n2. Batch Operations");
    let want_list = create_low_latency_want_list();

    // Create 100 test CIDs
    let cids: Vec<Cid> = (0..100).map(demo_cid).collect();
    println!("   Created 100 test CIDs");

    // Bulk add with same priority
    bulk_add_wants(&want_list, &cids, 100);
    println!("   ✓ Added 100 CIDs to want list with priority 100");
    assert_eq!(want_list.len(), 100);

    // Check presence
    assert!(all_wants_present(&want_list, &cids));
    println!("   ✓ Confirmed all CIDs are present");

    assert!(any_want_present(&want_list, &cids[0..10]));
    println!("   ✓ Confirmed any subset is present");

    // Bulk update priorities
    let updates: Vec<(Cid, i32)> = cids[0..50].iter().map(|c| (*c, 200)).collect();
    bulk_update_priorities(&want_list, &updates);
    println!("   ✓ Updated priorities for first 50 CIDs to 200");

    // Bulk remove
    bulk_remove_wants(&want_list, &cids[0..50]);
    println!("   ✓ Removed first 50 CIDs");
    assert_eq!(want_list.len(), 50);

    // 3. Configuration validation
    println!("\n3. Configuration Validation");

    // Valid configurations
    let valid_want_config = WantListConfig::default();
    match validate_want_list_config(&valid_want_config) {
        Ok(_) => println!("   ✓ Default WantListConfig is valid"),
        Err(e) => println!("   ✗ Validation error: {}", e),
    }

    let valid_session_config = SessionConfig {
        timeout: Duration::from_secs(60),
        default_priority: ipfrs_transport::Priority::Normal,
        max_concurrent_blocks: 100,
        progress_notifications: true,
    };
    match validate_session_config(&valid_session_config) {
        Ok(_) => println!("   ✓ SessionConfig is valid"),
        Err(e) => println!("   ✗ Validation error: {}", e),
    }

    let balanced_scoring = create_balanced_peer_scoring();
    match validate_peer_scoring_config(&balanced_scoring) {
        Ok(_) => println!("   ✓ Balanced PeerScoringConfig is valid"),
        Err(e) => println!("   ✗ Validation error: {}", e),
    }

    // Invalid configuration example
    let invalid_want_config = WantListConfig {
        max_wants: 0, // Invalid!
        ..Default::default()
    };
    match validate_want_list_config(&invalid_want_config) {
        Ok(_) => println!("   ✗ Should have been invalid!"),
        Err(e) => println!("   ✓ Caught invalid config: {}", e),
    }

    // 4. Performance calculation helpers
    println!("\n4. Performance Calculation Helpers");

    // Estimate transfer time
    let size = 100_000_000; // 100 MB
    let bandwidth = 10_000_000; // 10 Mbps
    let time = estimate_transfer_time(size, bandwidth);
    println!(
        "   Transfer time for {} at {}: {:.2}s",
        format_bytes(size),
        format_bandwidth(bandwidth),
        time.as_secs_f64()
    );

    // Calculate optimal chunk size
    let chunk_size = calculate_optimal_chunk_size(bandwidth, Duration::from_millis(100));
    println!(
        "   Optimal chunk size for {} bandwidth and 100ms latency: {}",
        format_bandwidth(bandwidth),
        format_bytes(chunk_size as u64)
    );

    // Calculate optimal concurrency
    let concurrency = calculate_optimal_concurrency(
        bandwidth,
        Duration::from_millis(100),
        256 * 1024, // 256 KB blocks
    );
    println!(
        "   Optimal concurrency for {} bandwidth, 100ms latency, 256KB blocks: {} parallel requests",
        format_bandwidth(bandwidth),
        concurrency
    );

    // 5. Preset configurations
    println!("\n5. Preset Configurations");

    let balanced = create_balanced_peer_scoring();
    println!("   Balanced peer scoring:");
    println!(
        "     - Latency weight: {:.0}%",
        balanced.latency_weight * 100.0
    );
    println!(
        "     - Bandwidth weight: {:.0}%",
        balanced.bandwidth_weight * 100.0
    );
    println!(
        "     - Reliability weight: {:.0}%",
        balanced.reliability_weight * 100.0
    );

    let reliability_focused = create_reliability_focused_scoring();
    println!("\n   Reliability-focused peer scoring:");
    println!(
        "     - Latency weight: {:.0}%",
        reliability_focused.latency_weight * 100.0
    );
    println!(
        "     - Bandwidth weight: {:.0}%",
        reliability_focused.bandwidth_weight * 100.0
    );
    println!(
        "     - Reliability weight: {:.0}%",
        reliability_focused.reliability_weight * 100.0
    );
    println!("     - Max failures: {}", reliability_focused.max_failures);

    // 6. Formatting helpers
    println!("\n6. Formatting Helpers");
    let sizes = vec![512, 1024, 1024 * 1024, 1024 * 1024 * 1024];
    for size in sizes {
        println!("   {} bytes = {}", size, format_bytes(size));
    }

    println!();
    let bandwidths = vec![1000, 1_000_000, 10_000_000, 1_000_000_000];
    for bw in bandwidths {
        println!("   {} bps = {}", bw, format_bandwidth(bw));
    }

    // 7. Real-world scenario
    println!("\n7. Real-World Scenario: High-Throughput Bulk Transfer");

    // Setup
    let transfer_size: u64 = 10 * 1024 * 1024 * 1024; // 10 GB
    let network_bandwidth: u64 = 100_000_000; // 100 Mbps
    let network_latency = Duration::from_millis(50); // 50ms
    let block_size = 1024 * 1024; // 1 MB blocks

    println!("   Scenario parameters:");
    println!("     - Total data: {}", format_bytes(transfer_size));
    println!(
        "     - Network bandwidth: {}",
        format_bandwidth(network_bandwidth)
    );
    println!("     - Network latency: {}ms", network_latency.as_millis());
    println!("     - Block size: {}", format_bytes(block_size as u64));

    // Calculate optimal parameters
    let chunk_size = calculate_optimal_chunk_size(network_bandwidth, network_latency);
    let concurrency = calculate_optimal_concurrency(network_bandwidth, network_latency, block_size);
    let estimated_time = estimate_transfer_time(transfer_size, network_bandwidth);

    println!("\n   Recommended configuration:");
    println!("     - Chunk size: {}", format_bytes(chunk_size as u64));
    println!("     - Parallel requests: {}", concurrency);
    println!(
        "     - Estimated transfer time: {:.1} minutes",
        estimated_time.as_secs_f64() / 60.0
    );

    println!("\n   Creating optimized configuration...");
    let _want_list = create_high_throughput_want_list();
    let _peer_manager = create_bandwidth_optimized_peer_manager();
    let _session = create_bulk_transfer_session(1);

    println!("   ✓ Want list configured for high throughput");
    println!("   ✓ Peer manager optimized for bandwidth");
    println!("   ✓ Session configured for bulk transfer");

    println!("\n=== Demo Complete ===");
    println!("\nAll utility functions demonstrated successfully!");
    println!("These helpers simplify common tasks and provide sensible defaults.");
    println!("See the utils.rs module for the full API documentation.");
}
