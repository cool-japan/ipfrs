//! Basic Usage Example
//!
//! This example demonstrates basic functionality of ipfrs-transport.
//! It shows how to:
//! - Use the want list for block prioritization
//! - Use metrics tracking
//! - Use retry policies
//!
//! Run with: cargo run --example basic_usage

use ipfrs_core::Cid;
use ipfrs_transport::{ConcurrentWantList, LatencyTracker, Timer, WantListConfig};
use multihash::Multihash;
use std::time::Duration;

/// Create a dummy CID for demonstration
fn create_cid(seed: u64) -> Cid {
    let data = seed.to_le_bytes();
    let hash = Multihash::wrap(0x12, &data).unwrap();
    Cid::new_v1(0x55, hash)
}

fn main() {
    println!("=== Basic Usage Example ===\n");

    // 1. Want List Management
    println!("--- Want List Management ---\n");

    let config = WantListConfig {
        max_wants: 1000,
        default_timeout: Duration::from_secs(30),
        max_retries: 3,
        base_retry_delay: Duration::from_millis(100),
        max_retry_delay: Duration::from_secs(5),
    };

    let want_list = ConcurrentWantList::new(config);

    // Add some blocks with different priorities
    let cids: Vec<Cid> = (0..5).map(create_cid).collect();

    want_list.add_simple(cids[0], 1000); // Critical priority
    want_list.add_simple(cids[1], 750); // High priority
    want_list.add_simple(cids[2], 500); // Normal priority
    want_list.add_simple(cids[3], 250); // Low priority

    println!("Added 4 blocks to want list");
    println!("  CID {}: Critical priority", 0);
    println!("  CID {}: High priority", 1);
    println!("  CID {}: Normal priority", 2);
    println!("  CID {}: Low priority", 3);

    // Get highest priority item
    if let Some(entry) = want_list.pop() {
        println!("\nHighest priority block:");
        println!("  CID: {}", entry.cid);
        println!("  Priority: {}", entry.priority);
    }

    let size = want_list.len();
    println!("\nRemaining wants: {}", size);

    // 2. Latency Tracking
    println!("\n--- Latency Tracking ---\n");

    let tracker = LatencyTracker::new();

    // Simulate some operations with timing
    for i in 0..10 {
        let timer = Timer::start();
        // Simulate some work
        std::thread::sleep(Duration::from_micros(100 + i * 10));
        timer.stop_and_record(&tracker);
    }

    let stats = tracker.stats();
    println!("Latency Statistics:");
    println!("  Min: {:?}", stats.min);
    println!("  Max: {:?}", stats.max);
    println!("  Mean: {:?}", stats.mean);
    println!("  P50: {:?}", stats.p50);
    println!("  P90: {:?}", stats.p90);
    println!("  P95: {:?}", stats.p95);
    println!("  P99: {:?}", stats.p99);
    println!("  P99.9: {:?}", stats.p99_9);
    println!("  Sample count: {}", stats.count);

    println!("\n✓ Example completed successfully!");
}
