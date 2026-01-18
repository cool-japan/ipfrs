//! Tor Privacy Example
//!
//! This example demonstrates the use of Tor integration for privacy-preserving
//! networking including onion routing, hidden services, and circuit management.
//!
//! # Features Demonstrated
//!
//! - Creating a Tor manager
//! - Starting and stopping Tor
//! - Creating Tor circuits for onion routing
//! - Connecting to .onion addresses through Tor
//! - Stream isolation for privacy
//! - Creating and hosting hidden services
//! - Circuit management and cleanup
//! - Configuration presets for different use cases
//!
//! # Run Example
//!
//! ```bash
//! cargo run --example tor_privacy
//! ```

use ipfrs_network::tor::{HiddenServiceConfig, TorConfig, TorManager};
use std::path::PathBuf;
use std::time::Duration;
use tracing::{info, Level};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("=== Tor Privacy Integration Example ===\n");

    // Example 1: Basic Tor usage
    example_basic_usage().await?;

    // Example 2: Hidden service hosting
    example_hidden_service().await?;

    // Example 3: Stream isolation for privacy
    example_stream_isolation().await?;

    // Example 4: Circuit management
    example_circuit_management().await?;

    // Example 5: Configuration presets
    example_config_presets().await?;

    Ok(())
}

/// Example 1: Basic Tor usage
async fn example_basic_usage() -> Result<(), Box<dyn std::error::Error>> {
    info!("--- Example 1: Basic Tor Usage ---");

    // Create Tor manager with default configuration
    let config = TorConfig::default();
    let mut manager = TorManager::new(config).await?;

    // Start Tor
    info!("Starting Tor...");
    manager.start().await?;
    info!("Tor started successfully");

    // Create a circuit for onion routing
    let circuit_id = manager.create_circuit().await?;
    info!("Created circuit {}", circuit_id);

    // Get circuit information
    if let Some(circuit) = manager.get_circuit(circuit_id) {
        info!("Circuit details:");
        info!("  State: {:?}", circuit.state);
        info!("  Hops: {:?}", circuit.hops);
        info!("  Created: {:?} ago", circuit.created_at.elapsed());
    }

    // Connect to an onion address through Tor
    info!("\nConnecting to example.onion...");
    let stream_id = manager.connect("example.onion:8080").await?;
    info!("Connected! Stream ID: {}", stream_id);

    // Get statistics
    let stats = manager.stats();
    info!("\nStatistics:");
    info!("  Circuits created: {}", stats.circuits_created);
    info!("  Active circuits: {}", stats.active_circuits);
    info!("  Streams created: {}", stats.streams_created);
    info!("  Active streams: {}", stats.active_streams);

    // Close the stream
    manager.close_stream(stream_id)?;
    info!("Stream closed");

    // Stop Tor
    manager.stop().await?;
    info!("Tor stopped\n");

    Ok(())
}

/// Example 2: Hidden service hosting
async fn example_hidden_service() -> Result<(), Box<dyn std::error::Error>> {
    info!("--- Example 2: Hidden Service Hosting ---");

    let config = TorConfig::default();
    let mut manager = TorManager::new(config).await?;

    manager.start().await?;

    // Configure hidden service
    let hs_config = HiddenServiceConfig {
        local_port: 8080,
        virtual_port: 80,
        data_dir: PathBuf::from("/tmp/tor-hidden-service"),
        max_connections: 100,
        use_v3: true, // Use v3 onion addresses (recommended)
    };

    // Create hidden service
    info!("Creating hidden service...");
    let onion_addr = manager.create_hidden_service(hs_config).await?;
    info!("✓ Hidden service created!");
    info!("  Onion address: {}", onion_addr);
    info!("  Local port: 8080");
    info!("  Virtual port: 80");
    info!("\nYour service is now accessible at: http://{}", onion_addr);

    // Validate onion address
    let is_valid = TorManager::validate_onion_address(&onion_addr);
    info!(
        "Address validation: {}",
        if is_valid { "✓ Valid" } else { "✗ Invalid" }
    );

    // List all hidden services
    let services = manager.get_hidden_services();
    info!("\nHosted hidden services: {}", services.len());
    for (addr, config) in services {
        info!("  {} -> localhost:{}", addr, config.local_port);
    }

    // Remove hidden service
    manager.remove_hidden_service(&onion_addr).await?;
    info!("\nHidden service removed");

    manager.stop().await?;
    info!("");

    Ok(())
}

/// Example 3: Stream isolation for privacy
async fn example_stream_isolation() -> Result<(), Box<dyn std::error::Error>> {
    info!("--- Example 3: Stream Isolation ---");

    // Enable stream isolation for maximum privacy
    let config = TorConfig {
        stream_isolation: true,
        ..Default::default()
    };

    let mut manager = TorManager::new(config).await?;
    manager.start().await?;

    info!("Stream isolation: enabled");
    info!("Each connection uses a separate circuit\n");

    // Connect to multiple addresses
    let addresses = [
        "example1.onion:8080",
        "example2.onion:9090",
        "example3.onion:7070",
    ];

    for (i, addr) in addresses.iter().enumerate() {
        let stream_id = manager.connect(addr).await?;
        info!("Stream {} -> {} (dedicated circuit)", i + 1, addr);
        info!("  Stream ID: {}", stream_id);
    }

    let stats = manager.stats();
    info!(
        "\nCircuits created: {} (one per stream)",
        stats.circuits_created
    );
    info!("This prevents correlation between connections\n");

    manager.stop().await?;

    Ok(())
}

/// Example 4: Circuit management
async fn example_circuit_management() -> Result<(), Box<dyn std::error::Error>> {
    info!("--- Example 4: Circuit Management ---");

    let config = TorConfig {
        max_circuits: 5,
        circuit_timeout: Duration::from_secs(60),
        ..Default::default()
    };

    let mut manager = TorManager::new(config).await?;
    manager.start().await?;

    info!("Creating multiple circuits...");

    // Create several circuits
    for _ in 0..3 {
        let circuit_id = manager.create_circuit().await?;
        info!("Circuit {} created", circuit_id);

        if let Some(circuit) = manager.get_circuit(circuit_id) {
            info!("  Hops: {}", circuit.hops.join(" -> "));
        }
    }

    // List all circuits
    let circuits = manager.get_circuits();
    info!("\nAll circuits ({}): ", circuits.len());
    for circuit in &circuits {
        info!(
            "  Circuit {}: {:?}, {} streams",
            circuit.id, circuit.state, circuit.stream_count
        );
    }

    let stats = manager.stats();
    info!("Active circuits: {}", stats.active_circuits);
    info!(
        "Average circuit build time: {:.1}ms\n",
        stats.avg_circuit_build_time_ms
    );

    manager.stop().await?;

    Ok(())
}

/// Example 5: Configuration presets
async fn example_config_presets() -> Result<(), Box<dyn std::error::Error>> {
    info!("--- Example 5: Configuration Presets ---");

    // High-privacy configuration
    info!("\n1. High Privacy Mode:");
    let high_privacy = TorConfig::high_privacy();
    info!("  Stream isolation: {}", high_privacy.stream_isolation);
    info!("  Max circuits: {}", high_privacy.max_circuits);
    info!(
        "  Bandwidth limit: {} bytes/sec",
        high_privacy.max_bandwidth_bps
    );

    let mut manager1 = TorManager::new(high_privacy).await?;
    manager1.start().await?;
    info!("  ✓ High privacy mode active");
    manager1.stop().await?;

    // High-performance configuration
    info!("\n2. High Performance Mode:");
    let high_perf = TorConfig::high_performance();
    info!("  Stream isolation: {}", high_perf.stream_isolation);
    info!("  Max circuits: {}", high_perf.max_circuits);
    info!("  Bandwidth limit: unlimited");

    let mut manager2 = TorManager::new(high_perf).await?;
    manager2.start().await?;
    info!("  ✓ High performance mode active");
    manager2.stop().await?;

    // Censorship-resistant configuration
    info!("\n3. Censorship Resistant Mode:");
    let censorship = TorConfig::censorship_resistant();
    info!("  Use bridges: {}", censorship.use_bridges);
    info!("  Bridges configured: {}", censorship.bridges.len());
    info!("  Circuit timeout: {:?}", censorship.circuit_timeout);

    let mut manager3 = TorManager::new(censorship).await?;
    manager3.start().await?;
    info!("  ✓ Censorship resistant mode active");
    info!("  This mode helps bypass network censorship");
    manager3.stop().await?;

    info!("\n=== Example Complete ===");

    Ok(())
}
