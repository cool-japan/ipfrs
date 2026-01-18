//! Metrics and Prometheus Export Example
//!
//! This example demonstrates the network metrics system:
//! 1. Tracking connection metrics
//! 2. Recording bandwidth usage
//! 3. Monitoring DHT operations
//! 4. Protocol-specific metrics
//! 5. Exporting metrics in Prometheus format
//! 6. Creating custom metric snapshots

use ipfrs_network::NetworkMetrics;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Network Metrics and Prometheus Export Example ===\n");

    // Scenario 1: Creating Metrics Tracker
    println!("1. Creating metrics tracker");
    let metrics = NetworkMetrics::new();
    println!("   ✓ Metrics tracker created\n");

    // Scenario 2: Connection Metrics
    println!("2. Recording connection metrics");

    // Establish some connections
    for i in 0..10 {
        let inbound = i % 2 == 0;
        metrics.connections().connection_established(inbound);
        println!(
            "   Connection {}: {} established",
            i + 1,
            if inbound { "Inbound" } else { "Outbound" }
        );
    }

    // Record some failures
    for _ in 0..2 {
        metrics.connections().connection_failed();
    }
    println!("   Recorded 2 failed connection attempts");

    // Close some connections
    for i in 0..3 {
        metrics
            .connections()
            .connection_closed(Duration::from_secs(60 + i * 30));
    }
    println!("   Closed 3 connections\n");

    // View connection metrics
    let conn_snapshot = metrics.connections().snapshot();
    println!("   Connection Metrics:");
    println!(
        "     - Total established: {}",
        conn_snapshot.total_established
    );
    println!("     - Total failed: {}", conn_snapshot.total_failed);
    println!("     - Active connections: {}", conn_snapshot.active);
    println!("     - Inbound: {}", conn_snapshot.total_inbound);
    println!("     - Outbound: {}", conn_snapshot.total_outbound);
    if let Some(avg_duration_ms) = conn_snapshot.avg_duration_ms {
        println!("     - Avg connection duration: {} ms", avg_duration_ms);
    }
    println!();

    // Scenario 3: Bandwidth Metrics
    println!("3. Recording bandwidth metrics");

    // Simulate data transfer
    metrics.bandwidth().record_sent(1024 * 1024); // 1 MB sent
    metrics.bandwidth().record_received(2 * 1024 * 1024); // 2 MB received
    metrics.bandwidth().record_sent(512 * 1024); // 512 KB sent
    metrics.bandwidth().record_received(3 * 1024 * 1024); // 3 MB received

    println!("   Data transfer recorded:");
    println!("     - Sent: 1.5 MB");
    println!("     - Received: 5 MB\n");

    let bw_snapshot = metrics.bandwidth().snapshot();
    println!("   Bandwidth Metrics:");
    println!(
        "     - Total sent: {} bytes ({:.2} MB)",
        bw_snapshot.total_sent,
        bw_snapshot.total_sent as f64 / (1024.0 * 1024.0)
    );
    println!(
        "     - Total received: {} bytes ({:.2} MB)",
        bw_snapshot.total_received,
        bw_snapshot.total_received as f64 / (1024.0 * 1024.0)
    );
    println!();

    // Scenario 4: DHT Metrics
    println!("4. Recording DHT metrics");

    // Record DHT queries
    for i in 0..15 {
        if i < 12 {
            metrics.dht().query_successful();
        } else {
            metrics.dht().query_failed();
        }
    }

    println!("   DHT queries recorded:");
    println!("     - Successful: 12");
    println!("     - Failed: 3");

    // Provider records
    metrics.dht().provider_published();
    metrics.dht().provider_published();
    metrics.dht().provider_published();
    metrics.dht().providers_found(5);
    metrics.dht().providers_found(3);

    println!("   Provider records:");
    println!("     - Published: 3");
    println!("     - Found: 8\n");

    let dht_snapshot = metrics.dht().snapshot();
    println!("   DHT Metrics:");
    println!(
        "     - Queries successful: {}",
        dht_snapshot.queries_successful
    );
    println!("     - Queries failed: {}", dht_snapshot.queries_failed);
    println!(
        "     - Success rate: {:.1}%",
        if dht_snapshot.queries_successful + dht_snapshot.queries_failed > 0 {
            (dht_snapshot.queries_successful as f64
                / (dht_snapshot.queries_successful + dht_snapshot.queries_failed) as f64)
                * 100.0
        } else {
            0.0
        }
    );
    println!(
        "     - Providers published: {}",
        dht_snapshot.providers_published
    );
    println!("     - Providers found: {}", dht_snapshot.providers_found);
    println!();

    // Scenario 5: Protocol Metrics
    println!("5. Recording protocol-specific metrics");

    metrics
        .protocols()
        .message_sent("/ipfrs/bitswap/1.0.0", 1024);
    metrics
        .protocols()
        .message_sent("/ipfrs/bitswap/1.0.0", 2048);
    metrics
        .protocols()
        .message_sent("/ipfrs/tensorswap/1.0.0", 512);
    metrics
        .protocols()
        .message_received("/ipfrs/bitswap/1.0.0", 4096);

    println!("   Protocol messages:");
    println!("     - bitswap sent: 2 messages (3 KB)");
    println!("     - tensorswap sent: 1 message (512 B)");
    println!("     - bitswap received: 1 message (4 KB)");

    if let Some((sent_msgs, sent_bytes, recv_msgs, recv_bytes)) =
        metrics.protocols().get_stats("/ipfrs/bitswap/1.0.0")
    {
        println!("   Bitswap stats:");
        println!("     - Sent: {} messages, {} bytes", sent_msgs, sent_bytes);
        println!(
            "     - Received: {} messages, {} bytes",
            recv_msgs, recv_bytes
        );
    }
    println!();

    // Scenario 6: Complete Metrics Snapshot
    println!("6. Complete metrics snapshot");

    let snapshot = metrics.snapshot();
    println!("   Overall Statistics:");
    println!("     Connections:");
    println!("       - Active: {}", snapshot.connections.active);
    println!(
        "       - Total established: {}",
        snapshot.connections.total_established
    );
    println!(
        "       - Total failed: {}",
        snapshot.connections.total_failed
    );
    println!("     Bandwidth:");
    println!(
        "       - Sent: {:.2} MB",
        snapshot.bandwidth.total_sent as f64 / (1024.0 * 1024.0)
    );
    println!(
        "       - Received: {:.2} MB",
        snapshot.bandwidth.total_received as f64 / (1024.0 * 1024.0)
    );
    println!("     DHT:");
    println!(
        "       - Queries: {} successful, {} failed",
        snapshot.dht.queries_successful, snapshot.dht.queries_failed
    );
    println!(
        "       - Providers: {} published, {} found",
        snapshot.dht.providers_published, snapshot.dht.providers_found
    );
    println!("     Uptime: {} seconds", snapshot.uptime_secs);
    println!();

    // Scenario 7: Prometheus Export
    println!("7. Exporting metrics in Prometheus format");

    match metrics.export_prometheus() {
        Ok(prometheus_text) => {
            println!("   ✓ Metrics exported successfully\n");
            println!("   Prometheus Metrics Output:");
            println!("   {}", "─".repeat(70));
            for line in prometheus_text.lines().take(30) {
                println!("   {}", line);
            }
            let line_count = prometheus_text.lines().count();
            if line_count > 30 {
                println!("   ... ({} more lines)", line_count - 30);
            }
            println!("   {}", "─".repeat(70));
        }
        Err(e) => {
            eprintln!("   ✗ Failed to export metrics: {}", e);
        }
    }
    println!();

    // Scenario 8: Uptime Tracking
    println!("8. Uptime tracking");

    let uptime = metrics.uptime();
    println!("   Node uptime: {:?}", uptime);
    println!("   Uptime in seconds: {}", uptime.as_secs());
    println!();

    // Scenario 9: Usage Example - HTTP Endpoint
    println!("9. Usage example: HTTP metrics endpoint");
    println!("   You can serve these metrics via HTTP:");
    println!();
    println!("   Example code:");
    println!("   ```rust");
    println!("   // In your HTTP server:");
    println!("   async fn metrics_endpoint(metrics: &NetworkMetrics) -> String {{");
    println!("       metrics.export_prometheus().unwrap_or_else(|e| {{");
    println!("           format!(\"Error exporting metrics: {{}}\", e)");
    println!("       }})");
    println!("   }}");
    println!("   ```");
    println!();
    println!("   Then configure Prometheus to scrape:");
    println!("   ```yaml");
    println!("   scrape_configs:");
    println!("     - job_name: 'ipfrs-network'");
    println!("       static_configs:");
    println!("         - targets: ['localhost:9090']");
    println!("   ```");
    println!();

    println!("=== Example completed successfully ===");

    Ok(())
}
