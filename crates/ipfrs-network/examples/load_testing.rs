//! Load Testing Example
//!
//! This example demonstrates how to use the load testing module to stress-test
//! network components and validate performance under various load conditions.

use ipfrs_network::{LoadTestConfig, LoadTestType, LoadTester};
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== IPFRS Network Load Testing Demo ===\n");

    // Scenario 1: Light Load Testing
    println!("Scenario 1: Light Load Testing");
    println!("--------------------------------");
    run_light_load_test()?;

    println!();

    // Scenario 2: Connection Stress Testing
    println!("Scenario 2: Connection Stress Testing");
    println!("--------------------------------------");
    run_connection_stress_test()?;

    println!();

    // Scenario 3: DHT Query Storm Testing
    println!("Scenario 3: DHT Query Storm Testing");
    println!("-----------------------------------");
    run_dht_query_storm_test()?;

    println!();

    // Scenario 4: Bandwidth Saturation Testing
    println!("Scenario 4: Bandwidth Saturation Testing");
    println!("-----------------------------------------");
    run_bandwidth_test()?;

    println!();

    // Scenario 5: Memory Pressure Testing
    println!("Scenario 5: Memory Pressure Testing");
    println!("-----------------------------------");
    run_memory_pressure_test()?;

    println!();

    // Scenario 6: Comprehensive Test Suite
    println!("Scenario 6: Comprehensive Test Suite");
    println!("------------------------------------");
    run_comprehensive_suite()?;

    println!();

    // Scenario 7: Custom Configuration
    println!("Scenario 7: Custom Configuration");
    println!("--------------------------------");
    run_custom_config_test()?;

    println!("\n=== Load Testing Complete ===");

    Ok(())
}

/// Run a light load test
fn run_light_load_test() -> Result<(), Box<dyn std::error::Error>> {
    let config = LoadTestConfig::light();
    println!("Running light load test...");
    println!("Duration: {:?}", config.duration);
    println!("Target connections: {}", config.connection_target);
    println!("Query rate: {} q/s", config.query_rate);

    let mut tester = LoadTester::new(config);

    let results = tester.run_test(LoadTestType::ConnectionStress)?;

    println!("\nResults:");
    println!(
        "  Status: {}",
        if results.passed {
            "PASSED ✓"
        } else {
            "FAILED ✗"
        }
    );
    println!("  Peak connections: {}", results.peak_connections);
    println!("  Duration: {:?}", results.duration);

    Ok(())
}

/// Run connection stress test
fn run_connection_stress_test() -> Result<(), Box<dyn std::error::Error>> {
    let config = LoadTestConfig {
        duration: Duration::from_millis(200),
        connection_target: 50,
        ..LoadTestConfig::moderate()
    };

    let mut tester = LoadTester::new(config);

    println!("Stress testing connection handling...");
    let results = tester.run_test(LoadTestType::ConnectionStress)?;

    println!("\nConnection Stress Results:");
    println!("{}", results.summary());
    println!("  Average latency: {:?}", results.average_latency);
    println!("  P95 latency: {:?}", results.p95_latency);
    println!("  P99 latency: {:?}", results.p99_latency);

    // Check metrics snapshot
    let metrics = tester.get_metrics_snapshot();
    println!("\nMetrics Snapshot:");
    println!("  Current connections: {}", metrics.connections);
    println!("  Peak connections: {}", metrics.peak_connections);

    Ok(())
}

/// Run DHT query storm test
fn run_dht_query_storm_test() -> Result<(), Box<dyn std::error::Error>> {
    let config = LoadTestConfig {
        duration: Duration::from_millis(200),
        query_rate: 20,
        ..LoadTestConfig::moderate()
    };

    let mut tester = LoadTester::new(config);

    println!("Generating DHT query storm...");
    let results = tester.run_test(LoadTestType::DhtQueryStorm)?;

    println!("\nDHT Query Storm Results:");
    println!("  Total queries: {}", results.total_queries);
    println!("  Successful: {}", results.successful_queries);
    println!("  Failed: {}", results.failed_queries);
    println!("  Success rate: {:.2}%", results.success_rate());
    println!(
        "  Query rate achieved: {:.2} q/s",
        results.query_rate_achieved
    );
    println!("  Average latency: {:?}", results.average_latency);
    println!("  P95 latency: {:?}", results.p95_latency);

    Ok(())
}

/// Run bandwidth saturation test
fn run_bandwidth_test() -> Result<(), Box<dyn std::error::Error>> {
    let config = LoadTestConfig {
        duration: Duration::from_millis(200),
        bandwidth_target: 5_000_000, // 5 MB/s
        ..LoadTestConfig::moderate()
    };

    let mut tester = LoadTester::new(config);

    println!("Testing bandwidth saturation...");
    let results = tester.run_test(LoadTestType::BandwidthSaturation)?;

    println!("\nBandwidth Test Results:");
    println!(
        "  Total bytes sent: {}",
        ipfrs_network::format_bytes(results.total_bytes_sent as usize)
    );
    println!(
        "  Total bytes received: {}",
        ipfrs_network::format_bytes(results.total_bytes_received as usize)
    );
    println!("  Throughput: {}", results.throughput_human());
    println!("  Duration: {:?}", results.duration);

    Ok(())
}

/// Run memory pressure test
fn run_memory_pressure_test() -> Result<(), Box<dyn std::error::Error>> {
    let memory_limit = 200 * 1024 * 1024; // 200 MB
    let config = LoadTestConfig {
        duration: Duration::from_millis(200),
        memory_limit,
        ..LoadTestConfig::moderate()
    };

    let mut tester = LoadTester::new(config);

    println!("Testing memory pressure handling...");
    let results = tester.run_test(LoadTestType::MemoryPressure)?;

    println!("\nMemory Pressure Results:");
    println!(
        "  Peak memory: {}",
        ipfrs_network::format_bytes(results.peak_memory_usage as usize)
    );
    println!(
        "  Average memory: {}",
        ipfrs_network::format_bytes(results.average_memory_usage as usize)
    );
    println!(
        "  Memory limit: {}",
        ipfrs_network::format_bytes(memory_limit as usize)
    );
    println!(
        "  Within limits: {}",
        if results.passed { "Yes ✓" } else { "No ✗" }
    );

    Ok(())
}

/// Run comprehensive test suite
fn run_comprehensive_suite() -> Result<(), Box<dyn std::error::Error>> {
    let config = LoadTestConfig {
        duration: Duration::from_millis(100),
        connection_target: 20,
        query_rate: 10,
        ..LoadTestConfig::light()
    };

    let mut tester = LoadTester::new(config);

    println!("Running comprehensive test suite...");
    println!("This will run all test types sequentially.");

    let results = tester.run_test(LoadTestType::ComprehensiveSuite)?;

    println!("\nComprehensive Suite Results:");
    println!(
        "  Overall status: {}",
        if results.passed {
            "PASSED ✓"
        } else {
            "FAILED ✗"
        }
    );
    println!("  Total queries: {}", results.total_queries);
    println!("  Peak connections: {}", results.peak_connections);
    println!(
        "  Peak memory: {}",
        ipfrs_network::format_bytes(results.peak_memory_usage as usize)
    );

    if !results.errors.is_empty() {
        println!("\nErrors encountered:");
        for error in &results.errors {
            println!("  - {}", error);
        }
    }

    Ok(())
}

/// Run test with custom configuration
fn run_custom_config_test() -> Result<(), Box<dyn std::error::Error>> {
    // Create a fully custom configuration
    let config = LoadTestConfig {
        duration: Duration::from_millis(150),
        connection_target: 30,
        query_rate: 15,
        bandwidth_target: 3_000_000, // 3 MB/s
        provider_publish_rate: 5,
        concurrent_operations: 25,
        memory_limit: 150 * 1024 * 1024, // 150 MB
        warmup_duration: Duration::from_millis(10),
        rampup_duration: Duration::from_millis(20),
    };

    println!("Running custom configured test...");
    println!("Configuration:");
    println!("  Duration: {:?}", config.duration);
    println!("  Connection target: {}", config.connection_target);
    println!("  Query rate: {} q/s", config.query_rate);
    println!(
        "  Bandwidth target: {}",
        ipfrs_network::format_bandwidth(config.bandwidth_target as usize)
    );
    println!("  Warmup: {:?}", config.warmup_duration);
    println!("  Rampup: {:?}", config.rampup_duration);

    let mut tester = LoadTester::new(config);

    let results = tester.run_test(LoadTestType::ConcurrentOps)?;

    println!("\nCustom Test Results:");
    println!("{}", results.summary());

    Ok(())
}
