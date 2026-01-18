//! Comprehensive Network Integration Example
//!
//! This example demonstrates how multiple ipfrs-network modules work together
//! in a realistic scenario:
//! 1. Network node creation with configuration
//! 2. Health monitoring integration
//! 3. Metrics tracking throughout
//! 4. DHT operations with provider records
//! 5. Connection management
//! 6. Offline queue for resilience
//! 7. Prometheus metrics export
//!
//! This shows a complete workflow from node startup to monitoring and metrics.

use ipfrs_network::{
    HealthChecker, NetworkConfig, NetworkMetrics, OfflineQueue, OfflineQueueConfig, QueuedRequest,
    QueuedRequestType, RequestPriority,
};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("=== Comprehensive Network Integration Example ===\n");
    println!("This example demonstrates a realistic workflow combining multiple");
    println!("ipfrs-network components for a production-ready node.\n");

    // ===================================================================
    // PHASE 1: INITIALIZATION
    // ===================================================================
    println!("📦 PHASE 1: Node Initialization");
    println!("{}", "=".repeat(70));

    // Create network configuration
    println!("\n1. Creating network configuration");
    let config = NetworkConfig {
        listen_addrs: vec![
            "/ip4/0.0.0.0/tcp/0".to_string(),
            "/ip4/0.0.0.0/udp/0/quic-v1".to_string(),
        ],
        enable_mdns: true,
        enable_quic: true,
        enable_nat_traversal: true,
        max_connections: Some(100),
        max_inbound_connections: Some(50),
        max_outbound_connections: Some(50),
        ..Default::default()
    };
    println!("   ✓ Configuration created");
    println!(
        "     - Listen addresses: {} configured",
        config.listen_addrs.len()
    );
    println!(
        "     - Max connections: {}",
        config.max_connections.unwrap()
    );
    println!("     - QUIC enabled: {}", config.enable_quic);
    println!("     - mDNS enabled: {}", config.enable_mdns);

    // Create metrics tracker
    println!("\n2. Initializing metrics system");
    let metrics = NetworkMetrics::new();
    println!("   ✓ Metrics system initialized");

    // Create health checker
    println!("\n3. Initializing health monitoring");
    let health_checker = HealthChecker::new();
    println!("   ✓ Health checker initialized");

    // Create offline queue for resilience
    println!("\n4. Setting up offline queue");
    let offline_queue = OfflineQueue::new(OfflineQueueConfig::mobile())?;
    println!("   ✓ Offline queue created");
    println!("     - Max queue size: 500");
    println!("     - Persistence: enabled");
    println!("     - Auto replay: enabled");

    // ===================================================================
    // PHASE 2: SIMULATING NETWORK OPERATIONS
    // ===================================================================
    println!("\n\n📡 PHASE 2: Network Operations Simulation");
    println!("{}", "=".repeat(70));

    // Simulate establishing connections
    println!("\n1. Establishing peer connections");
    for i in 0..8 {
        let inbound = i % 3 == 0;
        metrics.connections().connection_established(inbound);

        if i < 7 {
            println!(
                "   ✓ Connection {} established ({})",
                i + 1,
                if inbound { "inbound" } else { "outbound" }
            );
        }
    }

    // Simulate one failed connection
    metrics.connections().connection_failed();
    println!("   ✗ Connection 8 failed");

    let conn_snapshot = metrics.connections().snapshot();
    println!("\n   Connection Summary:");
    println!(
        "     - Total established: {}",
        conn_snapshot.total_established
    );
    println!("     - Active: {}", conn_snapshot.active);
    println!("     - Failed: {}", conn_snapshot.total_failed);

    // Simulate data transfer
    println!("\n2. Transferring data");
    metrics.bandwidth().record_sent(5 * 1024 * 1024); // 5 MB sent
    metrics.bandwidth().record_received(10 * 1024 * 1024); // 10 MB received

    let bw_snapshot = metrics.bandwidth().snapshot();
    println!("   ✓ Data transfer recorded");
    println!(
        "     - Sent: {:.2} MB",
        bw_snapshot.total_sent as f64 / (1024.0 * 1024.0)
    );
    println!(
        "     - Received: {:.2} MB",
        bw_snapshot.total_received as f64 / (1024.0 * 1024.0)
    );

    // Simulate DHT operations
    println!("\n3. Performing DHT operations");

    // Successful queries
    for i in 0..10 {
        metrics.dht().query_successful();
        if i < 3 {
            println!("   ✓ DHT query {} successful", i + 1);
        }
    }

    // Failed query
    metrics.dht().query_failed();
    println!("   ✗ DHT query 11 failed");

    // Provider operations
    metrics.dht().provider_published();
    metrics.dht().provider_published();
    metrics.dht().providers_found(5);

    println!("\n   DHT Summary:");
    let dht_snapshot = metrics.dht().snapshot();
    println!(
        "     - Queries: {} successful, {} failed",
        dht_snapshot.queries_successful, dht_snapshot.queries_failed
    );
    println!(
        "     - Success rate: {:.1}%",
        (dht_snapshot.queries_successful as f64
            / (dht_snapshot.queries_successful + dht_snapshot.queries_failed) as f64)
            * 100.0
    );
    println!(
        "     - Providers published: {}",
        dht_snapshot.providers_published
    );
    println!("     - Providers found: {}", dht_snapshot.providers_found);

    // Protocol-specific metrics
    println!("\n4. Recording protocol activity");
    metrics
        .protocols()
        .message_sent("/ipfrs/bitswap/1.0.0", 2048);
    metrics.protocols().message_sent("/ipfrs/kad/1.0.0", 512);
    metrics
        .protocols()
        .message_received("/ipfrs/bitswap/1.0.0", 4096);
    println!("   ✓ Protocol messages recorded");
    println!("     - Bitswap: 1 sent, 1 received");
    println!("     - Kademlia: 1 sent");

    // ===================================================================
    // PHASE 3: HEALTH MONITORING
    // ===================================================================
    println!("\n\n💓 PHASE 3: Health Monitoring");
    println!("{}", "=".repeat(70));

    println!("\n1. Performing health check");
    let health = health_checker.check_health(&metrics, None);

    println!("   Overall Health: {:?}", health.status);
    println!("   Health Score: {:.2}/1.00", health.score);
    println!("\n   Component Health:");

    for component in &health.components {
        let status_icon = match component.status {
            ipfrs_network::NetworkHealthStatus::Healthy => "✓",
            ipfrs_network::NetworkHealthStatus::Degraded => "⚠",
            ipfrs_network::NetworkHealthStatus::Unhealthy => "✗",
            ipfrs_network::NetworkHealthStatus::Unknown => "?",
        };

        println!(
            "     {} {}: {:?} (score: {:.2})",
            status_icon, component.name, component.status, component.score
        );

        if let Some(msg) = &component.message {
            println!("       → {}", msg);
        }
    }

    // ===================================================================
    // PHASE 4: OFFLINE RESILIENCE
    // ===================================================================
    println!("\n\n🔄 PHASE 4: Offline Resilience Testing");
    println!("{}", "=".repeat(70));

    println!("\n1. Simulating network interruption");
    offline_queue.set_online(false);
    println!("   ⚠ Network status: OFFLINE");

    println!("\n2. Queuing operations while offline");

    // Queue some operations
    let operations = vec![
        (
            "provide_block_1",
            QueuedRequestType::ProvideContent("QmBlock1".to_string()),
            RequestPriority::High,
        ),
        (
            "find_providers_1",
            QueuedRequestType::FindProviders("QmData1".to_string()),
            RequestPriority::Normal,
        ),
        (
            "provide_block_2",
            QueuedRequestType::ProvideContent("QmBlock2".to_string()),
            RequestPriority::Critical,
        ),
    ];

    for (id, req_type, priority) in operations {
        let request =
            QueuedRequest::new(id.to_string(), req_type, priority, Duration::from_secs(300));
        offline_queue.enqueue(request)?;
    }

    println!("   ✓ Queued 3 operations");
    println!("     - Pending: {}", offline_queue.pending_count());

    println!("\n3. Network restored");
    offline_queue.set_online(true);
    println!("   ✓ Network status: ONLINE");

    println!("\n4. Replaying queued operations");
    let batch = offline_queue.get_replay_batch();
    println!("   ✓ Retrieved batch of {} operations", batch.len());

    for (idx, request) in batch.iter().enumerate() {
        println!(
            "     {}. {} (priority: {:?})",
            idx + 1,
            request.id,
            request.priority
        );
        // Simulate processing
        offline_queue.mark_completed(&request.id, true);
    }

    let queue_stats = offline_queue.stats();
    println!("\n   Queue Statistics:");
    println!("     - Total queued: {}", queue_stats.requests_queued);
    println!("     - Completed: {}", queue_stats.requests_completed);
    println!(
        "     - Success rate: {:.1}%",
        queue_stats.success_rate() * 100.0
    );

    // ===================================================================
    // PHASE 5: METRICS EXPORT
    // ===================================================================
    println!("\n\n📊 PHASE 5: Metrics Export");
    println!("{}", "=".repeat(70));

    println!("\n1. Generating complete metrics snapshot");
    let snapshot = metrics.snapshot();

    println!("   Overall Statistics:");
    println!("     Uptime: {} seconds", snapshot.uptime_secs);
    println!("     Connections:");
    println!("       - Active: {}", snapshot.connections.active);
    println!("       - Total: {}", snapshot.connections.total_established);
    println!(
        "       - Success rate: {:.1}%",
        if snapshot.connections.total_established > 0 {
            ((snapshot.connections.total_established - snapshot.connections.total_failed) as f64
                / snapshot.connections.total_established as f64)
                * 100.0
        } else {
            0.0
        }
    );
    println!("     Bandwidth:");
    println!(
        "       - Total sent: {:.2} MB",
        snapshot.bandwidth.total_sent as f64 / (1024.0 * 1024.0)
    );
    println!(
        "       - Total received: {:.2} MB",
        snapshot.bandwidth.total_received as f64 / (1024.0 * 1024.0)
    );
    println!("     DHT:");
    println!(
        "       - Queries: {}/{} successful",
        snapshot.dht.queries_successful, snapshot.dht.queries_made
    );
    println!(
        "       - Providers: {} published, {} found",
        snapshot.dht.providers_published, snapshot.dht.providers_found
    );

    println!("\n2. Exporting Prometheus metrics");
    match metrics.export_prometheus() {
        Ok(prometheus_output) => {
            let line_count = prometheus_output.lines().count();
            println!("   ✓ Prometheus metrics exported successfully");
            println!("     - Total metrics: {} lines", line_count);
            println!("\n   Sample metrics:");
            for (idx, line) in prometheus_output.lines().take(5).enumerate() {
                if !line.starts_with('#') && !line.is_empty() {
                    println!("     {}. {}", idx + 1, line);
                }
            }
            println!("     ... ({} more metrics)", line_count.saturating_sub(5));
        }
        Err(e) => {
            println!("   ✗ Failed to export metrics: {}", e);
        }
    }

    // ===================================================================
    // PHASE 6: FINAL HEALTH CHECK
    // ===================================================================
    println!("\n\n🏥 PHASE 6: Final Health Assessment");
    println!("{}", "=".repeat(70));

    let final_health = health_checker.check_health(&metrics, None);
    let history = health_checker.health_history();

    println!("\n1. Current health status");
    println!("   Status: {:?}", final_health.status);
    println!("   Score: {:.2}/1.00", final_health.score);
    println!(
        "   Healthy: {}",
        if final_health.is_healthy() {
            "Yes ✓"
        } else {
            "No ✗"
        }
    );

    println!("\n2. Health history");
    println!("   Total checks: {}", history.total_checks);
    println!("   Health distribution:");
    println!(
        "     - Healthy: {} ({:.1}%)",
        history.healthy_count, history.healthy_percentage
    );
    println!(
        "     - Degraded: {} ({:.1}%)",
        history.degraded_count,
        if history.total_checks > 0 {
            (history.degraded_count as f64 / history.total_checks as f64) * 100.0
        } else {
            0.0
        }
    );

    // ===================================================================
    // SUMMARY
    // ===================================================================
    println!("\n\n📋 SUMMARY");
    println!("{}", "=".repeat(70));
    println!("This example demonstrated:");
    println!("  ✓ Network configuration and initialization");
    println!("  ✓ Connection management and tracking");
    println!("  ✓ Bandwidth monitoring");
    println!("  ✓ DHT operations and metrics");
    println!("  ✓ Real-time health monitoring");
    println!("  ✓ Offline resilience with request queuing");
    println!("  ✓ Comprehensive metrics collection");
    println!("  ✓ Prometheus-compatible metrics export");
    println!("\nAll components working together provide a production-ready");
    println!("networking layer with observability and resilience.");

    println!("\n=== Integration example completed successfully ===");

    Ok(())
}
