//! Comprehensive example demonstrating the monitoring and diagnostics features
//!
//! This example showcases:
//! - Health monitoring with component tracking
//! - Diagnostic engine for issue detection
//! - Statistics aggregation and trend analysis
//! - Auto-tuning based on network conditions
//!
//! Run with: `cargo run --example monitoring_diagnostics`

use ipfrs_core::Cid;
use ipfrs_transport::{
    AutoTuner, AutoTunerConfig, ComponentHealth, ComponentType, ConcurrentPeerManager,
    ConcurrentWantList, DiagnosticConfig, DiagnosticEngine, HealthCheckBuilder, HealthMonitor,
    HealthMonitorConfig, NetworkCondition, NetworkMetrics, PeerScoringConfig, Priority, Session,
    SessionConfig, StatsCollector, WantListConfig,
};
use std::thread;
use std::time::Duration;

fn main() {
    println!("=== IPFRS Transport Monitoring & Diagnostics Demo ===\n");

    // Setup basic components
    let want_list = setup_want_list();
    let peer_manager = setup_peer_manager();
    let session = setup_session();

    // Part 1: Health Monitoring
    demonstrate_health_monitoring(&want_list, &peer_manager, &session);

    // Part 2: Diagnostics Engine
    demonstrate_diagnostics(&want_list, &peer_manager);

    // Part 3: Statistics Aggregation
    demonstrate_stats_aggregation(&want_list, &peer_manager, &session);

    // Part 4: Auto-tuning
    demonstrate_auto_tuning();

    println!("\n=== Demo Complete ===");
}

fn setup_want_list() -> ConcurrentWantList {
    let config = WantListConfig {
        max_wants: 5000,
        default_timeout: Duration::from_secs(60),
        max_retries: 3,
        base_retry_delay: Duration::from_millis(100),
        max_retry_delay: Duration::from_secs(5),
    };
    ConcurrentWantList::new(config)
}

fn setup_peer_manager() -> ConcurrentPeerManager {
    let config = PeerScoringConfig {
        latency_weight: 0.4,
        bandwidth_weight: 0.4,
        reliability_weight: 0.2,
        ewma_alpha: 0.2,
        inactivity_decay: 0.01,
        min_score: 0.1,
        max_failures: 5,
    };
    ConcurrentPeerManager::new(config)
}

fn setup_session() -> Session {
    let config = SessionConfig {
        timeout: Duration::from_secs(120),
        default_priority: Priority::Normal,
        max_concurrent_blocks: 200,
        progress_notifications: true,
    };
    Session::new(1, config, None)
}

fn demonstrate_health_monitoring(
    want_list: &ConcurrentWantList,
    peer_manager: &ConcurrentPeerManager,
    session: &Session,
) {
    println!("--- Part 1: Health Monitoring ---\n");

    let config = HealthMonitorConfig {
        check_interval: Duration::from_secs(5),
        failure_threshold: 3,
        recovery_threshold: 2,
        auto_degradation: true,
        latency_threshold_ms: 1000,
        error_rate_threshold: 0.1,
    };

    let monitor = HealthMonitor::new(config);

    // Register components
    monitor.register_component(ComponentType::WantList, 100);
    monitor.register_component(ComponentType::PeerManager, 100);
    monitor.register_component(ComponentType::SessionManager, 100);

    // Set up alert callback
    monitor.on_alert(|alert| {
        println!(
            "  [ALERT] Component '{:?}' changed from {:?} to {:?}",
            alert.component, alert.old_status, alert.new_status
        );
    });

    // Simulate health checks over time
    println!("Running health checks...");

    // Check 1: All healthy
    let check1 = HealthCheckBuilder::new(ComponentType::WantList)
        .status(ComponentHealth::Healthy)
        .message("All operations normal")
        .metric("active_operations", want_list.len() as f64)
        .metric("error_count", 0.0)
        .build();
    monitor.record_health_check(check1);

    let check2 = HealthCheckBuilder::new(ComponentType::PeerManager)
        .status(ComponentHealth::Healthy)
        .message("Peer pool healthy")
        .metric("active_operations", peer_manager.stats().total_peers as f64)
        .metric("error_count", 0.0)
        .build();
    monitor.record_health_check(check2);

    let check3 = HealthCheckBuilder::new(ComponentType::SessionManager)
        .status(ComponentHealth::Healthy)
        .message("Session active")
        .metric("active_operations", session.stats().total_blocks as f64)
        .metric("error_count", 0.0)
        .build();
    monitor.record_health_check(check3);

    thread::sleep(Duration::from_millis(100));

    // Check 2: Session becomes degraded
    println!("\nSimulating degraded session performance...");
    let degraded_check = HealthCheckBuilder::new(ComponentType::SessionManager)
        .status(ComponentHealth::Degraded)
        .message("High latency detected")
        .metric("active_operations", 100.0)
        .metric("error_count", 5.0)
        .build();
    monitor.record_health_check(degraded_check);

    thread::sleep(Duration::from_millis(100));

    // Check 3: Peer manager becomes unhealthy
    println!("Simulating unhealthy peer manager...");
    let unhealthy_check = HealthCheckBuilder::new(ComponentType::PeerManager)
        .status(ComponentHealth::Unhealthy)
        .message("Multiple peer failures")
        .metric("active_operations", 10.0)
        .metric("error_count", 15.0)
        .build();
    monitor.record_health_check(unhealthy_check);

    // Display overall health
    println!("\n  Overall system health: {:?}", monitor.overall_health());

    // Display component statistics
    println!("\n  Component Statistics:");
    for component in &[
        ComponentType::WantList,
        ComponentType::PeerManager,
        ComponentType::SessionManager,
    ] {
        if let Some(stats) = monitor.get_stats(*component) {
            println!("    {:?}:", component);
            println!("      Current health: {:?}", stats.current_health);
            println!("      Total checks: {}", stats.total_checks);
            println!("      Consecutive failures: {}", stats.consecutive_failures);
            println!(
                "      Consecutive successes: {}",
                stats.consecutive_successes
            );
            println!("      Uptime ratio: {:.1}%", stats.uptime_ratio * 100.0);
        }
    }
}

fn demonstrate_diagnostics(want_list: &ConcurrentWantList, peer_manager: &ConcurrentPeerManager) {
    println!("\n--- Part 2: Diagnostic Engine ---\n");

    let config = DiagnosticConfig {
        max_queue_time: Duration::from_secs(60),
        min_active_peers: 3,
        min_avg_score: 0.5,
        max_expired_ratio: 0.1,
        min_progress_rate: 1.0,
    };

    let engine = DiagnosticEngine::with_config(config);

    // Add some test data to make diagnostics interesting
    let test_cid = Cid::default();

    // Add many wants to trigger issues
    for i in 0..100 {
        want_list.add_simple(test_cid, if i % 10 == 0 { 900 } else { 100 });
    }

    println!("Running comprehensive diagnostics...\n");

    let report = engine.generate_report(
        want_list,
        peer_manager,
        &[], // No sessions in this demo
    );

    // Display diagnostic report
    println!("{}", report);

    println!("\n  Summary:");
    println!("    Health Status: {:?}", report.health_status);
    println!("    Total Issues: {}", report.issues.len());
    println!(
        "    Critical: {}",
        report
            .issues
            .iter()
            .filter(|i| matches!(i.severity, ipfrs_transport::IssueSeverity::Critical))
            .count()
    );
    println!(
        "    Errors: {}",
        report
            .issues
            .iter()
            .filter(|i| matches!(i.severity, ipfrs_transport::IssueSeverity::Error))
            .count()
    );
    println!(
        "    Warnings: {}",
        report
            .issues
            .iter()
            .filter(|i| matches!(i.severity, ipfrs_transport::IssueSeverity::Warning))
            .count()
    );

    if !report.recommendations.is_empty() {
        println!("\n  Recommendations:");
        for rec in &report.recommendations {
            println!("    - {}", rec);
        }
    }

    // Clean up
    for _ in 0..100 {
        want_list.pop();
    }
}

fn demonstrate_stats_aggregation(
    _want_list: &ConcurrentWantList,
    peer_manager: &ConcurrentPeerManager,
    session: &Session,
) {
    println!("\n--- Part 3: Statistics Aggregation ---\n");

    let mut collector = StatsCollector::new(100); // Max history

    // Collect multiple data points over time
    println!("Collecting statistics over time...");

    for i in 0..5 {
        thread::sleep(Duration::from_millis(200));

        // Build aggregated stats using the builder
        let aggregated = ipfrs_transport::AggregatedStatsBuilder::new()
            .period(Duration::from_secs(1))
            .peer_stats(peer_manager.stats())
            .add_session_stats(session.stats())
            .build();

        collector.record(aggregated);

        println!("  Collected data point {}", i + 1);
    }

    println!("\n  Aggregated Statistics:");
    println!("    Total snapshots: {}", collector.len());

    if let Some(latest) = collector.latest() {
        println!("    Latest snapshot:");
        println!("      Period: {:?}", latest.period);

        if let Some(peer_stats) = &latest.peer_stats {
            println!("\n    Peer Manager:");
            println!("      Total peers: {}", peer_stats.total_peers);
            println!("      Connected peers: {}", peer_stats.connected_peers);
            println!("      Blacklisted peers: {}", peer_stats.blacklisted_peers);
            println!("      Avg latency: {} ms", peer_stats.avg_latency_ms);
        }

        if let Some(session_stats) = &latest.session_stats {
            println!("\n    Sessions (Aggregated):");
            println!("      Total sessions: {}", session_stats.total_sessions);
            println!("      Active sessions: {}", session_stats.active_sessions);
            println!(
                "      Completed sessions: {}",
                session_stats.completed_sessions
            );
            println!("      Total blocks: {}", session_stats.total_blocks);
            println!("      Total received: {}", session_stats.total_received);
            println!("      Total bytes: {}", session_stats.total_bytes);
            println!(
                "      Avg throughput: {} bytes/sec",
                session_stats.avg_throughput
            );
        }

        // Display performance metrics
        println!("\n    Performance Metrics:");
        println!(
            "      Total throughput: {} bytes/sec",
            latest.performance.total_throughput
        );
        println!(
            "      Success rate: {:.1}%",
            latest.performance.success_rate * 100.0
        );
        println!(
            "      Cache hit rate: {:.1}%",
            latest.performance.cache_hit_rate * 100.0
        );
        println!(
            "      Peer utilization: {:.1}%",
            latest.performance.peer_utilization * 100.0
        );
        println!(
            "      Efficiency score: {:.2}",
            latest.performance.efficiency_score
        );
    }

    // Show trends
    println!("\n  Throughput Trend:");
    let trend = collector.throughput_trend();
    for (idx, point) in trend.iter().enumerate() {
        println!("    Point {}: {:.0} bytes/sec", idx + 1, point.value);
    }

    println!(
        "\n  Average throughput over history: {} bytes/sec",
        collector.avg_throughput()
    );
}

fn demonstrate_auto_tuning() {
    println!("\n--- Part 4: Auto-tuning ---\n");

    let config = AutoTunerConfig {
        enabled: true,
        min_profile_change_interval: Duration::from_secs(30),
        excellent_latency_ms: 20,
        good_latency_ms: 50,
        fair_latency_ms: 150,
        poor_latency_ms: 500,
        excellent_bandwidth: 10_000_000,
        good_bandwidth: 5_000_000,
        max_acceptable_loss: 0.01,
    };

    let mut tuner = AutoTuner::with_config(config);

    // Test different network conditions
    // Based on default thresholds: excellent_latency_ms: 20, good_latency_ms: 50,
    // fair_latency_ms: 150, poor_latency_ms: 500
    let scenarios = vec![
        ("Excellent Network", 10, 100_000_000, 0.0, 0.99), // 10ms, 100 Mbps, 0% loss
        ("Good Network", 40, 8_000_000, 0.005, 0.98),      // 40ms, 8 Mbps, 0.5% loss
        ("Fair Network", 120, 3_000_000, 0.025, 0.95),     // 120ms, 3 Mbps, 2.5% loss
        ("Poor Network", 600, 800_000, 0.08, 0.82),        // 600ms, 800 Kbps, 8% loss
        ("Very Poor Network", 1500, 100_000, 0.15, 0.60),  // 1.5s, 100 Kbps, 15% loss
    ];

    for (name, latency_ms, bandwidth_bps, loss_rate, success_rate) in scenarios {
        println!("  Scenario: {}", name);

        let metrics = NetworkMetrics {
            avg_latency: Duration::from_millis(latency_ms),
            latency_stddev: Duration::from_millis(latency_ms / 10),
            avg_bandwidth: bandwidth_bps,
            packet_loss_rate: loss_rate,
            success_rate,
            active_peers: 5,
        };

        tuner.update_metrics(metrics);

        let condition = tuner.current_condition();
        println!("    Detected condition: {:?}", condition);

        let profile = tuner.current_profile();
        println!("    Profile recommendations:");
        println!("      Concurrent blocks: {}", profile.max_concurrent_blocks);
        println!("      Want timeout: {:?}", profile.want_timeout);
        println!("      Max retries: {}", profile.max_retries);
        println!("      Batch size: {}", profile.batch_size);
        println!(
            "      Pipelining: {}",
            if profile.enable_pipelining {
                "enabled"
            } else {
                "disabled"
            }
        );

        let recommendations = tuner.get_recommendations();
        if recommendations.len() > 6 {
            println!("    Additional recommendations:");
            for rec in &recommendations[6..] {
                println!("      - {}", rec);
            }
        }
        println!();

        // Verify profile matches expected condition
        let expected_condition = match name {
            "Excellent Network" => NetworkCondition::Excellent,
            "Good Network" => NetworkCondition::Good,
            "Fair Network" => NetworkCondition::Fair,
            "Poor Network" => NetworkCondition::Poor,
            "Very Poor Network" => NetworkCondition::VeryPoor,
            _ => NetworkCondition::Good,
        };

        assert_eq!(
            condition, expected_condition,
            "Network condition detection mismatch for {}",
            name
        );
    }

    println!("  Auto-tuning demonstration complete!");
}
