//! QUIC Multipath Example
//!
//! This example demonstrates the use of multipath QUIC for managing multiple
//! network paths simultaneously, providing improved reliability, throughput,
//! and seamless network transitions.
//!
//! # Features Demonstrated
//!
//! - Creating a multipath QUIC manager
//! - Adding multiple network paths
//! - Path quality monitoring and updates
//! - Different path selection strategies (Round Robin, Quality Based, Lowest Latency)
//! - Automatic path migration based on quality
//! - Traffic distribution across paths
//! - Path statistics and health monitoring
//!
//! # Run Example
//!
//! ```bash
//! cargo run --example multipath_quic
//! ```

use ipfrs_network::multipath_quic::{
    MultipathConfig, MultipathQuicManager, PathSelectionStrategy, PathState,
};
use std::net::SocketAddr;
use tracing::{info, Level};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("=== QUIC Multipath Example ===\n");

    // Example 1: Basic multipath usage with quality-based selection
    example_quality_based().await?;

    // Example 2: Mobile scenario (WiFi + Cellular)
    example_mobile_scenario().await?;

    // Example 3: High-bandwidth aggregation
    example_high_bandwidth().await?;

    // Example 4: Redundant transmission for reliability
    example_redundant_transmission().await?;

    // Example 5: Automatic path migration
    example_path_migration().await?;

    Ok(())
}

/// Example 1: Quality-based path selection
async fn example_quality_based() -> Result<(), Box<dyn std::error::Error>> {
    info!("--- Example 1: Quality-Based Path Selection ---");

    // Create multipath manager with quality-based strategy
    let config = MultipathConfig {
        max_paths: 4,
        strategy: PathSelectionStrategy::QualityBased,
        enable_redundancy: false,
        min_quality_threshold: 0.3,
        ..Default::default()
    };

    let manager = MultipathQuicManager::new(config);

    // Simulate adding multiple paths (e.g., different network interfaces)
    let paths = vec![
        ("192.168.1.100:4000".parse()?, "203.0.113.1:5000".parse()?), // WiFi
        ("10.0.0.100:4001".parse()?, "203.0.113.1:5000".parse()?),    // Ethernet
        ("172.16.0.100:4002".parse()?, "203.0.113.1:5000".parse()?),  // Cellular
    ];

    for (local, remote) in paths {
        let path_id = manager.add_path(local, remote)?;
        manager.update_path_state(path_id, PathState::Active)?;
        info!("Added path {}: {} -> {}", path_id, local, remote);
    }

    // Simulate quality updates for each path
    // Path 0 (WiFi): Good quality
    manager.update_path_quality(0, 15.0, 50_000_000, 0.01, 3.0)?;

    // Path 1 (Ethernet): Excellent quality
    manager.update_path_quality(1, 5.0, 100_000_000, 0.001, 1.0)?;

    // Path 2 (Cellular): Moderate quality
    manager.update_path_quality(2, 50.0, 10_000_000, 0.05, 10.0)?;

    // Select best path (should be Ethernet)
    let selected_path_id = manager.select_path()?;
    let selected_path = manager.get_path(selected_path_id).unwrap();
    info!(
        "Selected path {} with quality score: {:.2}",
        selected_path_id, selected_path.quality.score
    );

    // Display all path qualities
    info!("\nPath Quality Summary:");
    for path in manager.get_active_paths() {
        info!(
            "  Path {}: Quality={:.2}, RTT={:.1}ms, Bandwidth={}Mbps, Loss={:.2}%",
            path.id,
            path.quality.score,
            path.quality.rtt_ms,
            path.quality.bandwidth_bps / 125_000,
            path.quality.loss_rate * 100.0
        );
    }

    // Display statistics
    let stats = manager.stats();
    info!("\nStatistics:");
    info!("  Active paths: {}", stats.active_paths);
    info!("  Paths created: {}", stats.paths_created);
    info!("  Average quality: {:.2}", stats.avg_quality_score);
    info!("  Best path quality: {:.2}\n", stats.best_path_quality);

    Ok(())
}

/// Example 2: Mobile scenario with WiFi and Cellular
async fn example_mobile_scenario() -> Result<(), Box<dyn std::error::Error>> {
    info!("--- Example 2: Mobile Scenario (WiFi + Cellular) ---");

    // Use mobile-optimized configuration
    let config = MultipathConfig::mobile();
    let manager = MultipathQuicManager::new(config);

    // Add WiFi path
    let wifi_local: SocketAddr = "192.168.1.100:4000".parse()?;
    let remote: SocketAddr = "203.0.113.1:5000".parse()?;
    let wifi_path = manager.add_path(wifi_local, remote)?;
    manager.update_path_state(wifi_path, PathState::Active)?;
    info!("WiFi path added: {}", wifi_path);

    // Add Cellular path
    let cellular_local: SocketAddr = "10.0.0.100:4001".parse()?;
    let cellular_path = manager.add_path(cellular_local, remote)?;
    manager.update_path_state(cellular_path, PathState::Active)?;
    info!("Cellular path added: {}", cellular_path);

    // Simulate WiFi with good quality
    manager.update_path_quality(wifi_path, 10.0, 50_000_000, 0.01, 2.0)?;

    // Simulate Cellular with moderate quality
    manager.update_path_quality(cellular_path, 40.0, 5_000_000, 0.03, 8.0)?;

    // Initially WiFi is selected
    let selected = manager.select_path()?;
    info!("Initially selected: Path {} (WiFi)", selected);

    // Simulate WiFi becoming poor (e.g., signal degradation)
    info!("\nSimulating WiFi degradation...");
    manager.update_path_quality(wifi_path, 200.0, 1_000_000, 0.2, 50.0)?;

    // Now Cellular should be selected
    let selected = manager.select_path()?;
    info!("After degradation, selected: Path {} (Cellular)", selected);

    // Check if migration is recommended
    if let Some(new_path) = manager.should_migrate(wifi_path) {
        info!(
            "Migration recommended from WiFi (path {}) to Cellular (path {})\n",
            wifi_path, new_path
        );
    }

    Ok(())
}

/// Example 3: High-bandwidth aggregation
async fn example_high_bandwidth() -> Result<(), Box<dyn std::error::Error>> {
    info!("--- Example 3: High-Bandwidth Aggregation ---");

    // Use high-bandwidth configuration
    let config = MultipathConfig::high_bandwidth();
    let manager = MultipathQuicManager::new(config);

    // Add multiple high-bandwidth paths
    let remote: SocketAddr = "203.0.113.1:5000".parse()?;

    for i in 0..3 {
        let local: SocketAddr = format!("192.168.1.{}:4000", 100 + i).parse()?;
        let path_id = manager.add_path(local, remote)?;
        manager.update_path_state(path_id, PathState::Active)?;

        // Simulate varying bandwidth on each path
        let bandwidth = (50 + i * 25) * 1_000_000; // 50, 75, 100 Mbps
        manager.update_path_quality(path_id, 10.0, bandwidth, 0.01, 2.0)?;

        info!(
            "Added path {}: Bandwidth = {}Mbps",
            path_id,
            bandwidth / 125_000
        );
    }

    // Select highest bandwidth path
    let selected = manager.select_path()?;
    let path = manager.get_path(selected).unwrap();
    info!(
        "\nSelected path {} with highest bandwidth: {}Mbps",
        selected,
        path.quality.bandwidth_bps / 125_000
    );

    // Simulate sending data on the selected path
    manager.record_sent(selected, 10_000_000); // 10 MB sent

    let stats = manager.stats();
    info!(
        "Total bytes sent: {} MB\n",
        stats.total_bytes_sent / 1_000_000
    );

    Ok(())
}

/// Example 4: Redundant transmission for reliability
async fn example_redundant_transmission() -> Result<(), Box<dyn std::error::Error>> {
    info!("--- Example 4: Redundant Transmission ---");

    // Use high-reliability configuration with redundant strategy
    let config = MultipathConfig::high_reliability();
    let manager = MultipathQuicManager::new(config);

    // Add multiple paths
    let remote: SocketAddr = "203.0.113.1:5000".parse()?;

    for i in 0..3 {
        let local: SocketAddr = format!("192.168.1.{}:4000", 100 + i).parse()?;
        let path_id = manager.add_path(local, remote)?;
        manager.update_path_state(path_id, PathState::Active)?;

        // All paths have reasonable quality
        manager.update_path_quality(path_id, 20.0 + (i as f64) * 10.0, 10_000_000, 0.02, 5.0)?;

        info!("Added redundant path {}", path_id);
    }

    // In redundant mode, get all paths for transmission
    let all_paths = manager.select_all_paths();
    info!(
        "\nRedundant mode: Sending data on {} paths simultaneously",
        all_paths.len()
    );

    // Simulate sending the same packet on all paths
    let packet_size = 1500; // bytes
    for path_id in &all_paths {
        manager.record_sent(*path_id, packet_size);
        info!("  Sent {} bytes on path {}", packet_size, path_id);
    }

    let stats = manager.stats();
    info!(
        "\nTotal bytes sent (redundant): {} bytes across {} paths\n",
        stats.total_bytes_sent,
        all_paths.len()
    );

    Ok(())
}

/// Example 5: Automatic path migration
async fn example_path_migration() -> Result<(), Box<dyn std::error::Error>> {
    info!("--- Example 5: Automatic Path Migration ---");

    // Enable automatic migration
    let config = MultipathConfig {
        enable_auto_migration: true,
        migration_quality_threshold: 0.2,
        ..Default::default()
    };

    let manager = MultipathQuicManager::new(config);

    // Add two paths
    let remote: SocketAddr = "203.0.113.1:5000".parse()?;

    let path1 = manager.add_path("192.168.1.100:4000".parse()?, remote)?;
    let path2 = manager.add_path("192.168.1.101:4001".parse()?, remote)?;

    manager.update_path_state(path1, PathState::Active)?;
    manager.update_path_state(path2, PathState::Active)?;

    info!("Created paths {} and {}", path1, path2);

    // Initially both paths have similar quality
    manager.update_path_quality(path1, 15.0, 50_000_000, 0.01, 3.0)?;
    manager.update_path_quality(path2, 20.0, 45_000_000, 0.015, 4.0)?;

    info!("\nInitial quality:");
    for path in manager.get_active_paths() {
        info!("  Path {}: Quality = {:.2}", path.id, path.quality.score);
    }

    // Simulate quality degradation on path1
    info!("\nSimulating quality degradation on path {}...", path1);
    manager.update_path_quality(path1, 150.0, 5_000_000, 0.15, 30.0)?;

    info!("Updated quality:");
    for path in manager.get_active_paths() {
        info!("  Path {}: Quality = {:.2}", path.id, path.quality.score);
    }

    // Check if migration should occur
    if let Some(new_path) = manager.should_migrate(path1) {
        info!(
            "\n✓ Automatic migration triggered: Path {} -> Path {}",
            path1, new_path
        );

        let old_path = manager.get_path(path1).unwrap();
        let new_path_info = manager.get_path(new_path).unwrap();

        info!(
            "  Old path quality: {:.2} (RTT: {:.1}ms)",
            old_path.quality.score, old_path.quality.rtt_ms
        );
        info!(
            "  New path quality: {:.2} (RTT: {:.1}ms)",
            new_path_info.quality.score, new_path_info.quality.rtt_ms
        );

        let stats = manager.stats();
        info!("  Total migrations: {}", stats.migrations_count);
    }

    info!("");

    Ok(())
}
