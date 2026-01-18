//! Example demonstrating the new configuration presets and utility helpers
//!
//! This example shows:
//! - Edge device, datacenter, and specialized configuration presets
//! - Performance calculation utilities
//! - Debugging and diagnostic helpers
//! - Configuration analysis utilities

use ipfrs_transport::*;
use std::time::Duration;

fn main() {
    println!("=== Configuration Presets Example ===\n");

    // 1. Edge Device Configuration (Mobile/IoT devices)
    println!("1. Edge Device Configuration (Resource-Constrained)");
    let edge_want_list = create_edge_device_want_list();
    let edge_peer_manager = create_edge_device_peer_manager();
    println!(
        "   - Want list: {} entries, optimized for low memory",
        edge_want_list.len()
    );
    println!(
        "   - Peer manager: {} peers, aggressive decay\n",
        edge_peer_manager.stats().total_peers
    );

    // 2. Data Center Configuration (High-Resource deployments)
    println!("2. Data Center Configuration (High-Performance)");
    let datacenter_want_list = create_datacenter_want_list();
    println!(
        "   - Want list: {} entries, optimized for maximum throughput\n",
        datacenter_want_list.len()
    );

    // 3. Specialized Session Configurations
    println!("3. Specialized Session Configurations");

    let realtime_session = create_realtime_session(1);
    let realtime_stats = realtime_session.stats();
    println!(
        "   - Real-time session: {} blocks, {:?}",
        realtime_stats.total_blocks,
        debug_session_config(&SessionConfig {
            timeout: Duration::from_secs(30),
            default_priority: Priority::Urgent,
            max_concurrent_blocks: 50,
            progress_notifications: true,
        })
    );

    let scientific_session = create_scientific_session(2);
    let scientific_stats = scientific_session.stats();
    println!(
        "   - Scientific session: {} blocks, {:?}\n",
        scientific_stats.total_blocks,
        debug_session_config(&SessionConfig {
            timeout: Duration::from_secs(600),
            default_priority: Priority::High,
            max_concurrent_blocks: 1000,
            progress_notifications: true,
        })
    );

    // 4. Performance Calculation Utilities
    println!("4. Performance Calculation Utilities");

    // Calculate recommended buffer size
    let bandwidth = 100_000_000; // 100 Mbps
    let latency = Duration::from_millis(50);
    let buffer_size = calculate_recommended_buffer_size(bandwidth, latency);
    println!(
        "   - Recommended buffer size for {} at {}: {}",
        format_bandwidth(bandwidth),
        format_duration(latency),
        format_bytes(buffer_size as u64)
    );

    // Estimate required peers for target bandwidth
    let target_bandwidth = 1_000_000_000; // 1 Gbps
    let per_peer_bandwidth = 100_000_000; // 100 Mbps per peer
    let required_peers = estimate_required_peers(target_bandwidth, per_peer_bandwidth);
    println!(
        "   - Peers needed for {}: {} peers at {} each",
        format_bandwidth(target_bandwidth),
        required_peers,
        format_bandwidth(per_peer_bandwidth)
    );

    // Calculate expected throughput
    let concurrent_blocks = 100;
    let block_size = 256 * 1024; // 256 KB
    let latency = Duration::from_millis(100);
    let expected_throughput = calculate_expected_throughput(concurrent_blocks, block_size, latency);
    println!(
        "   - Expected throughput with {} blocks ({} each): {}/s\n",
        concurrent_blocks,
        format_bytes(block_size as u64),
        format_bytes(expected_throughput)
    );

    // 5. Debugging and Diagnostic Utilities
    println!("5. Debugging and Diagnostic Utilities");

    let want_config = WantListConfig {
        max_wants: 5000,
        default_timeout: Duration::from_secs(90),
        max_retries: 5,
        base_retry_delay: Duration::from_millis(20),
        max_retry_delay: Duration::from_secs(15),
    };
    println!("   - {}", debug_want_list_config(&want_config));

    let peer_config = create_balanced_peer_scoring();
    println!("   - {}", debug_peer_scoring_config(&peer_config));

    // 6. Configuration Analysis Utilities
    println!("\n6. Configuration Analysis Utilities");

    let high_throughput_config = WantListConfig {
        max_wants: 10000,
        default_timeout: Duration::from_secs(120),
        max_retries: 5,
        base_retry_delay: Duration::from_millis(50),
        max_retry_delay: Duration::from_secs(10),
    };
    println!(
        "   - Is high-throughput config: {}",
        is_high_throughput_config(&high_throughput_config)
    );

    let low_latency_config = WantListConfig {
        max_wants: 1000,
        default_timeout: Duration::from_secs(30),
        max_retries: 3,
        base_retry_delay: Duration::from_millis(10),
        max_retry_delay: Duration::from_secs(5),
    };
    println!(
        "   - Is low-latency config: {}",
        is_low_latency_config(&low_latency_config)
    );

    // Memory estimation
    println!(
        "   - Memory overhead (high-throughput): {}",
        format_bytes(estimate_want_list_memory(&high_throughput_config) as u64)
    );
    println!(
        "   - Memory overhead (low-latency): {}",
        format_bytes(estimate_want_list_memory(&low_latency_config) as u64)
    );

    // 7. Comparison of Different Presets
    println!("\n7. Comparison of Configuration Presets");

    println!("\n   Edge Device (Mobile/IoT):");
    println!("   - Max wants: 500");
    println!("   - Timeout: 60s");
    println!("   - Retries: 2");
    println!("   - Memory: ~50 KB");
    println!("   - Use case: Resource-constrained devices");

    println!("\n   High Throughput (Default):");
    println!("   - Max wants: 10,000");
    println!("   - Timeout: 120s");
    println!("   - Retries: 5");
    println!("   - Memory: ~1 MB");
    println!("   - Use case: Bulk data transfers");

    println!("\n   Low Latency (Interactive):");
    println!("   - Max wants: 1,000");
    println!("   - Timeout: 30s");
    println!("   - Retries: 3");
    println!("   - Memory: ~100 KB");
    println!("   - Use case: Real-time applications");

    println!("\n   Data Center (High-Performance):");
    println!("   - Max wants: 50,000");
    println!("   - Timeout: 180s");
    println!("   - Retries: 10");
    println!("   - Memory: ~5 MB");
    println!("   - Use case: Maximum throughput, ample resources");

    // 8. Duration Formatting Examples
    println!("\n8. Duration Formatting Examples");
    println!("   - 45s: {}", format_duration(Duration::from_secs(45)));
    println!("   - 90s: {}", format_duration(Duration::from_secs(90)));
    println!("   - 3661s: {}", format_duration(Duration::from_secs(3661)));
    println!("   - 7200s: {}", format_duration(Duration::from_secs(7200)));

    println!("\n=== Example Complete ===");
}
