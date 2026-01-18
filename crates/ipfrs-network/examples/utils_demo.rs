//! Network Utilities Example
//!
//! This example demonstrates the utility functions available in ipfrs-network
//! for formatting, parsing, and common network operations.

use ipfrs_network::utils::{
    exponential_backoff, format_bandwidth, format_bytes, format_duration, is_local_addr,
    is_public_addr, jittered_backoff, moving_average, parse_multiaddr, parse_multiaddrs,
    percentage, truncate_peer_id,
};
use libp2p::PeerId;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Network Utilities Demo");
    println!("======================\n");

    // 1. Formatting Functions
    println!("1. Formatting Functions");
    println!("-----------------------");

    // Format bytes
    println!("Bytes formatting:");
    println!("  1024 bytes = {}", format_bytes(1024));
    println!("  1048576 bytes = {}", format_bytes(1_048_576));
    println!("  1073741824 bytes = {}", format_bytes(1_073_741_824));
    println!("  500 bytes = {}", format_bytes(500));
    println!();

    // Format bandwidth
    println!("Bandwidth formatting:");
    println!("  1024 bytes/sec = {}", format_bandwidth(1024));
    println!("  10485760 bytes/sec = {}", format_bandwidth(10_485_760));
    println!();

    // Format duration
    println!("Duration formatting:");
    println!(
        "  90 seconds = {}",
        format_duration(Duration::from_secs(90))
    );
    println!(
        "  3665 seconds = {}",
        format_duration(Duration::from_secs(3665))
    );
    println!(
        "  500 milliseconds = {}",
        format_duration(Duration::from_millis(500))
    );
    println!();

    // 2. Multiaddress Parsing
    println!("2. Multiaddress Parsing");
    println!("------------------------");

    let addr = parse_multiaddr("/ip4/127.0.0.1/tcp/4001")?;
    println!("Parsed address: {}", addr);

    let addrs = parse_multiaddrs(&[
        "/ip4/127.0.0.1/tcp/4001".to_string(),
        "/ip6/::1/tcp/4001".to_string(),
        "/ip4/192.168.1.1/tcp/4001".to_string(),
    ])?;
    println!("Parsed {} addresses:", addrs.len());
    for (i, addr) in addrs.iter().enumerate() {
        println!("  {}. {}", i + 1, addr);
    }
    println!();

    // 3. Address Classification
    println!("3. Address Classification");
    println!("-------------------------");

    let test_addrs = vec![
        "/ip4/127.0.0.1/tcp/4001",
        "/ip4/192.168.1.1/tcp/4001",
        "/ip4/8.8.8.8/tcp/4001",
        "/ip6/::1/tcp/4001",
    ];

    for addr_str in test_addrs {
        let addr = parse_multiaddr(addr_str)?;
        println!("{}", addr);
        println!("  Is local: {}", is_local_addr(&addr));
        println!("  Is public: {}", is_public_addr(&addr));
    }
    println!();

    // 4. Exponential Backoff
    println!("4. Exponential Backoff");
    println!("----------------------");

    let base = Duration::from_secs(1);
    let max = Duration::from_secs(60);

    println!("Base: {:?}, Max: {:?}", base, max);
    for attempt in 0..8 {
        let backoff = exponential_backoff(attempt, base, max);
        let jittered = jittered_backoff(attempt, base, max);
        println!(
            "  Attempt {}: {} (jittered: {})",
            attempt,
            format_duration(backoff),
            format_duration(jittered)
        );
    }
    println!();

    // 5. Peer ID Operations
    println!("5. Peer ID Operations");
    println!("---------------------");

    let peer_id = PeerId::random();
    println!("Full peer ID: {}", peer_id);
    println!("Truncated (8 chars): {}", truncate_peer_id(&peer_id, 8));
    println!("Truncated (16 chars): {}", truncate_peer_id(&peer_id, 16));
    println!();

    // 6. Statistical Functions
    println!("6. Statistical Functions");
    println!("------------------------");

    // Percentage calculation
    println!("Percentages:");
    println!("  25 of 100 = {}%", percentage(25, 100));
    println!("  1 of 3 = {}%", percentage(1, 3));
    println!(
        "  0 of 0 = {}% (handles division by zero)",
        percentage(0, 0)
    );
    println!();

    // Moving average
    println!("Moving average (alpha = 0.5):");
    let mut avg = 10.0;
    println!("  Current: {}", avg);
    for new_value in &[20.0, 15.0, 25.0, 18.0] {
        avg = moving_average(avg, *new_value, 0.5);
        println!("  After {}: {:.2}", new_value, avg);
    }
    println!();

    // 7. Practical Example: Connection Retry Logic
    println!("7. Practical Example: Connection Retry");
    println!("---------------------------------------");

    println!("Simulating connection retry with exponential backoff:");
    let max_attempts = 5;
    for attempt in 0..max_attempts {
        let backoff = jittered_backoff(attempt, Duration::from_secs(1), Duration::from_secs(30));
        println!(
            "  Attempt {}/{}: Waiting {} before retry...",
            attempt + 1,
            max_attempts,
            format_duration(backoff)
        );
    }
    println!();

    // 8. Practical Example: Bandwidth Statistics
    println!("8. Practical Example: Bandwidth Statistics");
    println!("-------------------------------------------");

    let bytes_sent = 10_485_760; // 10 MB
    let bytes_received = 52_428_800; // 50 MB
    let total = bytes_sent + bytes_received;

    println!("Bytes sent: {}", format_bytes(bytes_sent));
    println!("Bytes received: {}", format_bytes(bytes_received));
    println!("Total: {}", format_bytes(total));
    println!();

    println!("Upload ratio: {}%", percentage(bytes_sent, total));
    println!("Download ratio: {}%", percentage(bytes_received, total));
    println!();

    // Assuming 60 second duration
    let duration_secs = 60;
    println!(
        "Average upload speed: {}",
        format_bandwidth(bytes_sent / duration_secs)
    );
    println!(
        "Average download speed: {}",
        format_bandwidth(bytes_received / duration_secs)
    );
    println!();

    println!("All utilities demonstrated successfully!");

    Ok(())
}
