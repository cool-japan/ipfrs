//! Example demonstrating the simplified TransportFacade API
//!
//! This example shows how easy it is to set up a complete transport system
//! with monitoring, diagnostics, and auto-tuning using the facade pattern.
//!
//! Run with: `cargo run --example facade_demo`

use ipfrs_transport::{NetworkMetrics, TransportFacade, TransportPreset};
use std::time::Duration;

fn main() {
    println!("=== IPFRS Transport Facade Demo ===\n");

    // Part 1: Simple setup with default configuration
    demonstrate_default_setup();

    // Part 2: Preset configurations
    demonstrate_presets();

    // Part 3: Custom configuration
    demonstrate_custom_setup();

    // Part 4: Using the facade features
    demonstrate_features();

    println!("\n=== Demo Complete ===");
}

fn demonstrate_default_setup() {
    println!("--- Part 1: Default Setup ---\n");

    // Create a transport system with all features enabled using defaults
    let transport = TransportFacade::builder().build();

    println!("  Created transport with default configuration");
    println!("  Overall health: {:?}", transport.overall_health());
    println!("  Want list enabled: Yes");
    println!("  Peer manager enabled: Yes");
    println!(
        "  Monitoring enabled: {:?}",
        transport.health_monitor().is_some()
    );
    println!(
        "  Diagnostics enabled: {:?}",
        transport.run_diagnostics().is_some()
    );
    println!(
        "  Auto-tuning enabled: {:?}",
        transport.auto_tuner().is_some()
    );
    println!(
        "  Stats collection enabled: {:?}",
        transport.stats_collector().is_some()
    );
}

fn demonstrate_presets() {
    println!("\n--- Part 2: Preset Configurations ---\n");

    // Low-latency preset
    println!("  Low-latency preset:");
    let low_latency =
        ipfrs_transport::TransportFacadeBuilder::from_preset(TransportPreset::LowLatency).build();
    println!(
        "    Max wants: {}",
        low_latency.config().want_list_config.max_wants
    );
    println!(
        "    Latency weight: {}",
        low_latency.config().peer_scoring_config.latency_weight
    );
    println!(
        "    Timeout: {:?}",
        low_latency.config().want_list_config.default_timeout
    );

    // High-throughput preset
    println!("\n  High-throughput preset:");
    let high_throughput =
        ipfrs_transport::TransportFacadeBuilder::from_preset(TransportPreset::HighThroughput)
            .build();
    println!(
        "    Max wants: {}",
        high_throughput.config().want_list_config.max_wants
    );
    println!(
        "    Bandwidth weight: {}",
        high_throughput
            .config()
            .peer_scoring_config
            .bandwidth_weight
    );
    println!(
        "    Timeout: {:?}",
        high_throughput.config().want_list_config.default_timeout
    );

    // Edge device preset
    println!("\n  Edge device preset:");
    let edge =
        ipfrs_transport::TransportFacadeBuilder::from_preset(TransportPreset::EdgeDevice).build();
    println!(
        "    Max wants: {}",
        edge.config().want_list_config.max_wants
    );
    println!(
        "    Max failures: {}",
        edge.config().peer_scoring_config.max_failures
    );
    println!(
        "    Reliability weight: {}",
        edge.config().peer_scoring_config.reliability_weight
    );

    // Federated learning preset
    println!("\n  Federated learning preset:");
    let federated =
        ipfrs_transport::TransportFacadeBuilder::from_preset(TransportPreset::FederatedLearning)
            .build();
    println!(
        "    Max wants: {}",
        federated.config().want_list_config.max_wants
    );
    println!(
        "    Timeout: {:?}",
        federated.config().want_list_config.default_timeout
    );
    println!(
        "    Max retries: {}",
        federated.config().want_list_config.max_retries
    );
}

fn demonstrate_custom_setup() {
    println!("\n--- Part 3: Custom Configuration ---\n");

    // Build with selective features
    let transport = TransportFacade::builder()
        .with_monitoring()
        .with_auto_tuning()
        .without_diagnostics() // Disable diagnostics
        .with_stats(500) // Enable stats with custom history size
        .build();

    println!("  Created transport with custom feature selection:");
    println!("    Monitoring: {:?}", transport.health_monitor().is_some());
    println!(
        "    Diagnostics: {:?}",
        transport.run_diagnostics().is_some()
    );
    println!("    Auto-tuning: {:?}", transport.auto_tuner().is_some());
    println!("    Stats: {:?}", transport.stats_collector().is_some());
}

fn demonstrate_features() {
    println!("\n--- Part 4: Using Facade Features ---\n");

    // Create a fully-featured transport
    let transport = TransportFacade::builder()
        .with_monitoring()
        .with_diagnostics()
        .with_auto_tuning()
        .with_stats(100)
        .build();

    // 1. Health Monitoring
    println!("  Health Monitoring:");
    println!("    Overall health: {:?}", transport.overall_health());
    println!(
        "    Want list health: {:?}",
        transport.component_health(ipfrs_transport::ComponentType::WantList)
    );
    println!(
        "    Peer manager health: {:?}",
        transport.component_health(ipfrs_transport::ComponentType::PeerManager)
    );

    // 2. Diagnostics
    println!("\n  Diagnostics:");
    if let Some(report) = transport.run_diagnostics() {
        println!("    Health status: {:?}", report.health_status);
        println!("    Total issues: {}", report.issues.len());
        println!("    Recommendations: {}", report.recommendations.len());
        if !report.recommendations.is_empty() {
            println!("    First recommendation: {}", report.recommendations[0]);
        }
    }

    // 3. Auto-tuning
    println!("\n  Auto-tuning:");
    let metrics = NetworkMetrics {
        avg_latency: Duration::from_millis(50),
        latency_stddev: Duration::from_millis(10),
        avg_bandwidth: 5_000_000, // 5 MB/s
        packet_loss_rate: 0.01,
        success_rate: 0.95,
        active_peers: 5,
    };
    transport.update_network_metrics(metrics);

    if let Some(recommendations) = transport.get_tuning_recommendations() {
        println!("    Tuning recommendations:");
        for (i, rec) in recommendations.iter().take(3).enumerate() {
            println!("      {}. {}", i + 1, rec);
        }
    }

    // 4. Statistics
    println!("\n  Statistics:");
    transport.record_stats();
    if let Some(stats) = transport.latest_stats() {
        println!("    Period: {:?}", stats.period);
        println!(
            "    Total throughput: {} bytes/sec",
            stats.performance.total_throughput
        );
        println!(
            "    Success rate: {:.1}%",
            stats.performance.success_rate * 100.0
        );
    }
    println!(
        "    Average throughput: {} bytes/sec",
        transport.avg_throughput()
    );

    // 5. Direct component access
    println!("\n  Direct Component Access:");
    println!("    Want list length: {}", transport.want_list().len());
    println!(
        "    Total peers: {}",
        transport.peer_manager().stats().total_peers
    );
}
