//! Automatic Network Configuration Tuning Example
//!
//! This example demonstrates how to use the AutoTuner to automatically
//! optimize network configuration based on system resources and workload.

use ipfrs_network::{AutoTuner, AutoTunerConfig};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== IPFRS Network Auto-Tuning Example ===\n");

    // Scenario 1: Basic Auto-Tuning
    println!("--- Scenario 1: Basic Auto-Tuning ---");
    {
        let mut tuner = AutoTuner::new();

        // Analyze system resources
        let resources = tuner.analyze_system().await?;
        println!("Detected System Resources:");
        println!(
            "  Total Memory: {} MB",
            resources.total_memory / (1024 * 1024)
        );
        println!(
            "  Available Memory: {} MB",
            resources.available_memory / (1024 * 1024)
        );
        println!("  CPU Cores: {}", resources.cpu_cores);
        println!("  Memory Category: {}", resources.memory_category());
        println!("  Battery Powered: {}\n", resources.is_battery_powered);

        // Generate optimized configuration
        let config = tuner.generate_config().await?;
        println!("Generated Configuration:");
        println!("  Max Connections: {:?}", config.max_connections);
        println!(
            "  Connection Buffer: {:?} bytes",
            config.connection_buffer_size
        );
        println!("  NAT Traversal: {}", config.enable_nat_traversal);
        println!("  Low Memory Mode: {:?}\n", config.low_memory_mode);

        let stats = tuner.stats();
        println!("Tuner Stats:");
        println!("  Adjustments Made: {}", stats.adjustments_made);
        println!("  Optimization Score: {:.2}\n", stats.optimization_score);
    }

    // Scenario 2: Conservative Tuning
    println!("--- Scenario 2: Conservative Tuning ---");
    {
        let config = AutoTunerConfig::conservative();
        let mut tuner = AutoTuner::with_config(config);

        println!("Using conservative tuning configuration:");
        println!("  Safety Margin: 30%");
        println!("  Adjustment Interval: 10 minutes");
        println!("  Aggressive Mode: Disabled\n");

        let net_config = tuner.generate_config().await?;
        println!(
            "Generated conservative configuration with {} max connections\n",
            net_config.max_connections.unwrap_or(0)
        );
    }

    // Scenario 3: Aggressive Tuning
    println!("--- Scenario 3: Aggressive Tuning ---");
    {
        let config = AutoTunerConfig::aggressive();
        let mut tuner = AutoTuner::with_config(config);

        println!("Using aggressive tuning configuration:");
        println!("  Safety Margin: 10%");
        println!("  Adjustment Interval: 1 minute");
        println!("  Aggressive Mode: Enabled\n");

        let net_config = tuner.generate_config().await?;
        println!(
            "Generated aggressive configuration with {} max connections\n",
            net_config.max_connections.unwrap_or(0)
        );
    }

    // Scenario 4: Workload Monitoring
    println!("--- Scenario 4: Workload Monitoring ---");
    {
        let mut tuner = AutoTuner::new();
        tuner.analyze_system().await?;

        println!("Simulating workload updates...");

        // Simulate low load
        tuner.update_workload(5, 2.0, 50_000.0, 100_000_000);
        println!("Low load workload:");
        let profile = tuner.workload_profile();
        println!("  Avg Connections: {:.1}", profile.avg_connections);
        println!("  Avg Query Rate: {:.1} qps", profile.avg_query_rate);
        println!(
            "  Avg Bandwidth: {:.1} bytes/s",
            profile.avg_bandwidth_usage
        );

        // Simulate medium load
        tuner.update_workload(25, 10.0, 500_000.0, 500_000_000);
        println!("\nMedium load workload:");
        let profile = tuner.workload_profile();
        println!("  Avg Connections: {:.1}", profile.avg_connections);
        println!("  Avg Query Rate: {:.1} qps", profile.avg_query_rate);
        println!(
            "  Avg Bandwidth: {:.1} bytes/s",
            profile.avg_bandwidth_usage
        );

        // Simulate high load
        tuner.update_workload(100, 50.0, 5_000_000.0, 2_000_000_000);
        println!("\nHigh load workload:");
        let profile = tuner.workload_profile();
        println!("  Avg Connections: {:.1}", profile.avg_connections);
        println!("  Avg Query Rate: {:.1} qps", profile.avg_query_rate);
        println!(
            "  Avg Bandwidth: {:.1} bytes/s",
            profile.avg_bandwidth_usage
        );
        println!("  CPU Bound: {}", profile.cpu_bound);
        println!("  Memory Bound: {}", profile.memory_bound);
        println!("  Bandwidth Bound: {}\n", profile.bandwidth_bound);
    }

    // Scenario 5: Getting Recommendations
    println!("--- Scenario 5: Getting Recommendations ---");
    {
        let mut tuner = AutoTuner::new();
        tuner.analyze_system().await?;

        // Simulate memory pressure
        tuner.update_workload(50, 20.0, 1_000_000.0, 3_500_000_000);

        let recommendations = tuner.recommendations();
        println!("Tuning Recommendations:");
        for (i, rec) in recommendations.iter().enumerate() {
            println!("  {}. {}", i + 1, rec);
        }
        println!();
    }

    // Scenario 6: Monitoring Lifecycle
    println!("--- Scenario 6: Monitoring Lifecycle ---");
    {
        let mut tuner = AutoTuner::new();
        tuner.analyze_system().await?;

        println!("Starting monitoring...");
        tuner.start_monitoring().await?;
        println!("  Monitoring active: {}", tuner.is_monitoring());

        // Simulate some time passing
        sleep(Duration::from_millis(100)).await;

        println!("Stopping monitoring...");
        tuner.stop_monitoring();
        println!("  Monitoring active: {}\n", tuner.is_monitoring());
    }

    // Scenario 7: Real-World Integration
    println!("--- Scenario 7: Real-World Integration ---");
    {
        let mut tuner = AutoTuner::new();

        // Analyze system and generate config
        let resources = tuner.analyze_system().await?;
        let config = tuner.generate_config().await?;

        println!("Creating network node with auto-tuned configuration...");
        println!(
            "  System: {} MB RAM, {} cores",
            resources.total_memory / (1024 * 1024),
            resources.cpu_cores
        );
        println!(
            "  Config: {} max connections",
            config.max_connections.unwrap_or(0)
        );

        // In a real application, you would use this config to create a NetworkNode:
        // let mut node = NetworkNode::new(config)?;
        // node.start().await?;

        println!("\nNode would be started with optimized configuration.");
        println!("Auto-tuner would continue monitoring and adjusting as needed.\n");
    }

    // Scenario 8: Multiple Adjustment Cycles
    println!("--- Scenario 8: Multiple Adjustment Cycles ---");
    {
        let mut tuner = AutoTuner::new();
        tuner.analyze_system().await?;

        println!("Simulating multiple adjustment cycles...");
        for i in 1..=5 {
            tuner.generate_config().await?;
            let stats = tuner.stats();
            println!(
                "  Cycle {}: {} adjustments, score: {:.2}",
                i, stats.adjustments_made, stats.optimization_score
            );
        }
        println!();
    }

    println!("=== Auto-Tuning Example Complete ===");
    Ok(())
}
