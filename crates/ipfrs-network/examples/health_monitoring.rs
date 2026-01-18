//! Health Monitoring Example
//!
//! This example demonstrates the network health monitoring system:
//! 1. Creating a health checker
//! 2. Performing health checks on network components
//! 3. Monitoring connection health
//! 4. Tracking DHT health
//! 5. Analyzing bandwidth health
//! 6. Health history and trending

use ipfrs_network::{DhtHealth, DhtHealthStatus, HealthChecker, NetworkMetrics};
use std::time::Duration;

fn main() {
    println!("=== Network Health Monitoring Example ===\n");

    // Scenario 1: Creating Health Checker
    println!("1. Creating health checker");
    let checker = HealthChecker::new();
    println!("   ✓ Health checker created\n");

    // Scenario 2: Initial Health Check (No Connections)
    println!("2. Initial health check (startup)");
    let metrics = NetworkMetrics::new();

    let health = checker.check_health(&metrics, None);
    println!("   Overall status: {:?}", health.status);
    println!("   Overall score: {:.2}/1.00", health.score);
    println!("   Components:");
    for component in &health.components {
        println!(
            "     - {}: {:?} (score: {:.2})",
            component.name, component.status, component.score
        );
        if let Some(msg) = &component.message {
            println!("       Message: {}", msg);
        }
    }
    println!();

    // Scenario 3: Simulating Network Activity
    println!("3. Simulating network activity");

    // Add some successful connections
    println!("   Adding connections...");
    for i in 0..5 {
        metrics.connections().connection_established(i < 4); // 4 successful, 1 failed
    }
    println!("     - 4 successful connections");
    println!("     - 1 failed connection");

    // Add bandwidth
    println!("   Recording bandwidth...");
    metrics.bandwidth().record_sent(1024 * 1024); // 1 MB sent
    metrics.bandwidth().record_received(2 * 1024 * 1024); // 2 MB received
    println!("     - Sent: 1 MB");
    println!("     - Received: 2 MB\n");

    // Scenario 4: Health Check With Active Network
    println!("4. Health check with active network");
    let health = checker.check_health(&metrics, None);
    println!("   Overall status: {:?}", health.status);
    println!("   Overall score: {:.2}/1.00", health.score);
    println!("   Uptime: {} seconds", health.uptime_secs);
    println!("   Components:");
    for component in &health.components {
        println!(
            "     - {}: {:?} (score: {:.2})",
            component.name, component.status, component.score
        );
        if let Some(msg) = &component.message {
            println!("       Message: {}", msg);
        }
    }
    println!();

    // Scenario 5: DHT Health Monitoring
    println!("5. DHT health monitoring");

    let dht_health_good = DhtHealth {
        health_score: 0.9,
        query_success_rate: 0.9,
        cache_hit_rate: 0.8,
        peer_count: 50,
        cached_query_count: 100,
        provider_count: 25,
        status: DhtHealthStatus::Healthy,
    };

    let health = checker.check_health(&metrics, Some(&dht_health_good));
    println!("   DHT Status: {:?}", dht_health_good.status);
    println!("   Peer count: {}", dht_health_good.peer_count);
    println!(
        "   Query success rate: {:.1}%",
        dht_health_good.query_success_rate * 100.0
    );
    println!(
        "   Cache hit rate: {:.1}%",
        dht_health_good.cache_hit_rate * 100.0
    );
    println!(
        "   Overall health: {:?} (score: {:.2})",
        health.status, health.score
    );
    println!();

    // Scenario 6: Degraded DHT Health
    println!("6. Degraded DHT health scenario");

    let dht_health_degraded = DhtHealth {
        health_score: 0.6,
        query_success_rate: 0.6,
        cache_hit_rate: 0.4,
        peer_count: 10, // Low peer count
        cached_query_count: 30,
        provider_count: 5,
        status: DhtHealthStatus::Degraded,
    };

    let health = checker.check_health(&metrics, Some(&dht_health_degraded));
    println!("   DHT Status: {:?}", dht_health_degraded.status);
    println!("   Peer count: {} (low)", dht_health_degraded.peer_count);
    println!(
        "   Query success rate: {:.1}%",
        dht_health_degraded.query_success_rate * 100.0
    );
    println!(
        "   Overall health: {:?} (score: {:.2})",
        health.status, health.score
    );

    println!("   Components:");
    for component in &health.components {
        if component.name == "dht" {
            println!(
                "     - DHT: {:?} (score: {:.2})",
                component.status, component.score
            );
            if let Some(msg) = &component.message {
                println!("       ⚠ {}", msg);
            }
        }
    }
    println!();

    // Scenario 7: Connection Failures
    println!("7. High connection failure scenario");

    // Add more failed connections
    for _ in 0..10 {
        metrics.connections().connection_failed();
    }

    let health = checker.check_health(&metrics, None);
    println!(
        "   Total connections: {}",
        metrics.connections().total_established()
    );
    println!(
        "   Failed connections: {}",
        metrics.connections().total_failed()
    );
    println!(
        "   Overall health: {:?} (score: {:.2})",
        health.status, health.score
    );

    for component in &health.components {
        if component.name == "connections" {
            println!("   Connection component:");
            println!("     - Status: {:?}", component.status);
            println!("     - Score: {:.2}", component.score);
            if let Some(msg) = &component.message {
                println!("     - ⚠ {}", msg);
            }
        }
    }
    println!();

    // Scenario 8: Health History
    println!("8. Health history tracking");

    // Perform several more health checks to build history
    for i in 0..10 {
        if i % 3 == 0 {
            // Occasionally add a connection failure
            metrics.connections().connection_failed();
        } else {
            // Mostly successful
            metrics.connections().connection_established(true);
            metrics.bandwidth().record_sent(1024 * 100);
        }
        checker.check_health(&metrics, Some(&dht_health_good));
    }

    let history = checker.health_history();
    println!("   Health checks performed: {}", history.total_checks);
    println!(
        "   Healthy: {} ({:.1}%)",
        history.healthy_count,
        (history.healthy_count as f64 / history.total_checks as f64) * 100.0
    );
    println!(
        "   Degraded: {} ({:.1}%)",
        history.degraded_count,
        (history.degraded_count as f64 / history.total_checks as f64) * 100.0
    );
    println!(
        "   Unhealthy: {} ({:.1}%)",
        history.unhealthy_count,
        (history.unhealthy_count as f64 / history.total_checks as f64) * 100.0
    );
    println!(
        "   Unknown: {} ({:.1}%)",
        history.unknown_count,
        (history.unknown_count as f64 / history.total_checks as f64) * 100.0
    );
    println!(
        "   Overall health percentage: {:.1}%",
        history.healthy_percentage
    );
    println!();

    // Scenario 9: Last Health Check Retrieval
    println!("9. Retrieving last health check");

    if let Some(last_health) = checker.last_health() {
        println!("   Last check timestamp: {}", last_health.timestamp);
        println!("   Status: {:?}", last_health.status);
        println!("   Score: {:.2}", last_health.score);
        println!(
            "   Is healthy? {}",
            if last_health.is_healthy() {
                "Yes"
            } else {
                "No"
            }
        );
        println!(
            "   Is degraded? {}",
            if last_health.is_degraded() {
                "Yes"
            } else {
                "No"
            }
        );
        println!(
            "   Is unhealthy? {}",
            if last_health.is_unhealthy() {
                "Yes"
            } else {
                "No"
            }
        );
    }
    println!();

    // Scenario 10: Critical Health Situation
    println!("10. Critical health scenario (all connections lost)");

    // Close all connections
    let active = metrics.connections().active();
    for _ in 0..active {
        metrics
            .connections()
            .connection_closed(Duration::from_secs(60));
    }

    let health = checker.check_health(&metrics, None);
    println!("   Active connections: {}", metrics.connections().active());
    println!(
        "   Overall health: {:?} (score: {:.2})",
        health.status, health.score
    );

    if health.is_unhealthy() {
        println!("   ⚠⚠⚠ CRITICAL: Network is unhealthy!");
    }
    println!();

    println!("=== Example completed successfully ===");
}
