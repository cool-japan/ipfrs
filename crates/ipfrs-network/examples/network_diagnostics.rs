//! Network Diagnostics Example
//!
//! This example demonstrates how to use the diagnostics module to troubleshoot
//! and monitor network health.

use ipfrs_network::diagnostics::{
    DiagnosticTest, NetworkDiagnostics, PerformanceMetrics, TroubleshootingGuide,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Network Diagnostics Example");
    println!("===========================\n");

    // Create diagnostics instance
    let mut diagnostics = NetworkDiagnostics::new();

    // 1. Run all diagnostic tests
    println!("Running all diagnostic tests...\n");
    let results = diagnostics.run_all_tests();

    for result in &results {
        let status = if result.passed {
            "✓ PASS"
        } else {
            "✗ FAIL"
        };
        println!("{} - {} ({:?})", status, result.test_name, result.duration);

        if !result.passed {
            println!("  Issue: {}", result.message);
            if let Some(fix) = &result.suggested_fix {
                println!("  Fix: {}", fix);
            }
        }
    }

    println!(
        "\n{} / {} tests passed\n",
        results.iter().filter(|r| r.passed).count(),
        results.len()
    );

    // 2. Run specific diagnostic test
    println!("Running specific test: DHT Health");
    let dht_result = diagnostics.run_test(DiagnosticTest::DhtHealth);
    println!(
        "Result: {}",
        if dht_result.passed { "PASS" } else { "FAIL" }
    );
    println!("Message: {}\n", dht_result.message);

    // 3. Check test history
    println!("Test History:");
    println!("-------------");
    for (i, result) in diagnostics.results_history().iter().enumerate() {
        println!(
            "{}. {} - {}",
            i + 1,
            result.test_name,
            if result.passed { "PASS" } else { "FAIL" }
        );
    }
    println!();

    // 4. Get latest result for a specific test
    if let Some(latest) = diagnostics.latest_result(DiagnosticTest::BasicConnectivity) {
        println!("Latest Basic Connectivity Test:");
        println!("  Status: {}", if latest.passed { "PASS" } else { "FAIL" });
        println!("  Duration: {:?}\n", latest.duration);
    }

    // 5. Record performance metrics
    println!("Recording Performance Metrics...");
    let metrics = PerformanceMetrics {
        avg_latency_ms: 45.5,
        median_latency_ms: 42.0,
        p95_latency_ms: 95.0,
        avg_bandwidth_bps: 1_500_000,
        dht_success_rate: 0.98,
        avg_dht_query_ms: 180.0,
        connected_peers: 15,
        routing_table_size: 75,
    };

    diagnostics.record_metrics(metrics);

    if let Some(latest_metrics) = diagnostics.latest_metrics() {
        println!("Latest Metrics:");
        println!("  Average Latency: {:.1} ms", latest_metrics.avg_latency_ms);
        println!(
            "  Median Latency: {:.1} ms",
            latest_metrics.median_latency_ms
        );
        println!("  P95 Latency: {:.1} ms", latest_metrics.p95_latency_ms);
        println!(
            "  Bandwidth: {} bytes/sec",
            latest_metrics.avg_bandwidth_bps
        );
        println!(
            "  DHT Success Rate: {:.1}%",
            latest_metrics.dht_success_rate * 100.0
        );
        println!(
            "  DHT Query Time: {:.1} ms",
            latest_metrics.avg_dht_query_ms
        );
        println!("  Connected Peers: {}", latest_metrics.connected_peers);
        println!(
            "  Routing Table Size: {}\n",
            latest_metrics.routing_table_size
        );
    }

    // 6. Generate comprehensive diagnostic report
    println!("Diagnostic Report:");
    println!("==================");
    let report = diagnostics.generate_report();
    println!("{}\n", report);

    // 7. Use troubleshooting guide
    println!("Troubleshooting Guide:");
    println!("=====================\n");

    let topics = TroubleshootingGuide::list_topics();
    println!("Available topics:");
    for topic in &topics {
        println!("  - {}", topic);
    }
    println!();

    // Get specific troubleshooting advice
    println!("Example: Getting advice for 'no_peers' issue:");
    if let Some(advice) = TroubleshootingGuide::get_advice("no_peers") {
        println!("{}\n", advice);
    }

    println!("Example: Getting advice for 'slow_dht' issue:");
    if let Some(advice) = TroubleshootingGuide::get_advice("slow_dht") {
        println!("{}\n", advice);
    }

    println!("Example: Getting advice for 'nat_issues':");
    if let Some(advice) = TroubleshootingGuide::get_advice("nat_issues") {
        println!("{}\n", advice);
    }

    // 8. Demonstrate diagnostic test types
    println!("Available Diagnostic Tests:");
    println!("---------------------------");
    let test_types = vec![
        DiagnosticTest::BasicConnectivity,
        DiagnosticTest::DhtHealth,
        DiagnosticTest::NatTraversal,
        DiagnosticTest::PeerDiscovery,
        DiagnosticTest::BootstrapConnectivity,
        DiagnosticTest::ConfigValidation,
        DiagnosticTest::ResourceCheck,
        DiagnosticTest::ProtocolCompatibility,
    ];

    for test in test_types {
        println!("\n{}", test.name());
        println!("  {}", test.description());
    }

    println!("\n\nDiagnostics example completed successfully!");

    Ok(())
}
