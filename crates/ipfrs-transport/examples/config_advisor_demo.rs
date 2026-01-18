//! Example demonstrating the configuration advisor
//!
//! This example shows how to use the ConfigAdvisor to get intelligent
//! configuration recommendations based on your specific requirements.

use ipfrs_transport::*;
use std::time::Duration;

fn main() {
    println!("=== Configuration Advisor Demo ===\n");

    // Scenario 1: Real-time streaming application
    println!("Scenario 1: Real-time Streaming Application");
    println!("--------------------------------------------");
    let req = ConfigRequirements {
        use_case: UseCase::RealTime,
        resource_level: ResourceLevel::Moderate,
        network_quality: NetworkQuality::Good,
        expected_peers: 10,
        avg_block_size: 64 * 1024, // 64 KB blocks
        target_latency: Some(Duration::from_millis(100)),
        target_throughput: None,
    };

    let config = ConfigAdvisor::recommend(&req);
    print_recommendation(&config);

    // Scenario 2: Bulk data transfer
    println!("\nScenario 2: Bulk Data Transfer");
    println!("-------------------------------");
    let req = ConfigRequirements {
        use_case: UseCase::BulkTransfer,
        resource_level: ResourceLevel::High,
        network_quality: NetworkQuality::Excellent,
        expected_peers: 50,
        avg_block_size: 1024 * 1024, // 1 MB blocks
        target_latency: None,
        target_throughput: Some(1_000_000_000), // 1 Gbps
    };

    let config = ConfigAdvisor::recommend(&req);
    print_recommendation(&config);

    // Scenario 3: Edge/IoT device
    println!("\nScenario 3: Edge/IoT Device");
    println!("---------------------------");
    let req = ConfigRequirements {
        use_case: UseCase::EdgeComputing,
        resource_level: ResourceLevel::Minimal,
        network_quality: NetworkQuality::Fair,
        expected_peers: 3,
        avg_block_size: 128 * 1024, // 128 KB blocks
        target_latency: None,
        target_throughput: None,
    };

    let config = ConfigAdvisor::recommend(&req);
    print_recommendation(&config);

    // Scenario 4: ML model distribution
    println!("\nScenario 4: Machine Learning Distribution");
    println!("-----------------------------------------");
    let req = ConfigRequirements {
        use_case: UseCase::MLDistribution,
        resource_level: ResourceLevel::High,
        network_quality: NetworkQuality::Good,
        expected_peers: 30,
        avg_block_size: 10 * 1024 * 1024, // 10 MB blocks (large models)
        target_latency: None,
        target_throughput: Some(500_000_000), // 500 Mbps
    };

    let config = ConfigAdvisor::recommend(&req);
    print_recommendation(&config);

    // Scenario 5: Scientific computing
    println!("\nScenario 5: Scientific Computing");
    println!("--------------------------------");
    let req = ConfigRequirements {
        use_case: UseCase::ScientificComputing,
        resource_level: ResourceLevel::High,
        network_quality: NetworkQuality::Excellent,
        expected_peers: 100,
        avg_block_size: 5 * 1024 * 1024, // 5 MB blocks
        target_latency: None,
        target_throughput: Some(10_000_000_000), // 10 Gbps
    };

    let config = ConfigAdvisor::recommend(&req);
    print_recommendation(&config);

    // Performance analysis example
    println!("\n\n=== Performance Analysis Example ===\n");

    let want_list = WantListConfig {
        max_wants: 5000,
        default_timeout: Duration::from_secs(90),
        max_retries: 5,
        base_retry_delay: Duration::from_millis(20),
        max_retry_delay: Duration::from_secs(15),
    };

    let session = SessionConfig {
        timeout: Duration::from_secs(120),
        default_priority: Priority::Normal,
        max_concurrent_blocks: 100,
        progress_notifications: true,
    };

    println!("Analyzing configuration:");
    println!("  - Want list: {} max entries", want_list.max_wants);
    println!(
        "  - Session: {} concurrent blocks",
        session.max_concurrent_blocks
    );
    println!();

    let profile = ConfigAdvisor::analyze_performance(
        &want_list,
        &session,
        256 * 1024,  // 256 KB blocks
        50.0,        // 50ms network latency
        100_000_000, // 100 Mbps bandwidth
    );

    print_performance_profile(&profile);

    // Comparison of different network conditions
    println!("\n\n=== Network Quality Comparison ===\n");

    for (quality, (latency_ms, bandwidth)) in [
        (NetworkQuality::Excellent, (5.0, 1_000_000_000u64)),
        (NetworkQuality::Good, (30.0, 100_000_000u64)),
        (NetworkQuality::Fair, (100.0, 10_000_000u64)),
        (NetworkQuality::Poor, (300.0, 1_000_000u64)),
    ] {
        let profile = ConfigAdvisor::analyze_performance(
            &want_list,
            &session,
            256 * 1024,
            latency_ms,
            bandwidth,
        );

        println!(
            "{:?} network ({:.0}ms, {}):",
            quality,
            latency_ms,
            format_bandwidth(bandwidth)
        );
        println!(
            "  - Estimated throughput: {}/s",
            format_bytes(profile.estimated_throughput_bps)
        );
        println!(
            "  - Network utilization: {:.1}%",
            profile.network_utilization * 100.0
        );
        println!("  - Overall score: {:.2}", profile.overall_score);
        if !profile.bottlenecks.is_empty() {
            println!("  - Bottlenecks: {:?}", profile.bottlenecks);
        }
        println!();
    }

    println!("=== Demo Complete ===");
}

fn print_recommendation(config: &RecommendedConfig) {
    println!("Recommendations:");
    println!("  Confidence: {:.1}%", config.confidence * 100.0);
    println!(
        "  Estimated memory: {}",
        format_bytes(config.estimated_memory as u64)
    );
    println!();
    println!("  Want List:");
    println!("    - Max entries: {}", config.want_list.max_wants);
    println!(
        "    - Timeout: {}",
        format_duration(config.want_list.default_timeout)
    );
    println!("    - Max retries: {}", config.want_list.max_retries);
    println!();
    println!("  Peer Scoring:");
    println!(
        "    - Latency weight: {:.0}%",
        config.peer_scoring.latency_weight * 100.0
    );
    println!(
        "    - Bandwidth weight: {:.0}%",
        config.peer_scoring.bandwidth_weight * 100.0
    );
    println!(
        "    - Reliability weight: {:.0}%",
        config.peer_scoring.reliability_weight * 100.0
    );
    println!("    - Max failures: {}", config.peer_scoring.max_failures);
    println!();
    println!("  Session:");
    println!("    - Timeout: {}", format_duration(config.session.timeout));
    println!("    - Priority: {:?}", config.session.default_priority);
    println!(
        "    - Max concurrent blocks: {}",
        config.session.max_concurrent_blocks
    );
    println!();
    println!("  {}", config.explanation);
}

fn print_performance_profile(profile: &PerformanceProfile) {
    println!("Performance Profile:");
    println!(
        "  - Estimated latency: {:.2}ms",
        profile.estimated_latency_ms
    );
    println!(
        "  - Estimated throughput: {}/s",
        format_bytes(profile.estimated_throughput_bps)
    );
    println!(
        "  - Memory efficiency: {:.1}%",
        profile.memory_efficiency * 100.0
    );
    println!(
        "  - Network utilization: {:.1}%",
        profile.network_utilization * 100.0
    );
    println!("  - Overall score: {:.2}", profile.overall_score);

    if !profile.bottlenecks.is_empty() {
        println!("  - Bottlenecks detected:");
        for bottleneck in &profile.bottlenecks {
            println!("    * {}", bottleneck);
        }
    } else {
        println!("  - No bottlenecks detected");
    }
}
