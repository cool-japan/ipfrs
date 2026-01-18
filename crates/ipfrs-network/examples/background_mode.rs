//! Example: Background Mode Support
//!
//! This example demonstrates pause/resume functionality for mobile applications
//! that need to conserve battery when the app goes to the background.

use ipfrs_network::{BackgroundModeConfig, BackgroundModeManager};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== IPFRS Background Mode Example ===\n");

    // Create background mode manager with mobile configuration
    let config = BackgroundModeConfig::mobile();
    println!("Mobile Background Mode Configuration:");
    println!("  Pause DHT queries: {}", config.pause_dht_queries);
    println!(
        "  Pause announcements: {}",
        config.pause_provider_announcements
    );
    println!(
        "  Close idle connections: {}",
        config.close_idle_connections
    );
    println!("  Idle threshold: {:?}", config.idle_connection_threshold);
    println!(
        "  Keep minimal connections: {}",
        config.keep_minimal_connections
    );
    println!(
        "  Minimal connection count: {}",
        config.minimal_connection_count
    );
    println!("  Reduce DHT frequency: {}", config.reduce_dht_frequency);
    println!(
        "  Background DHT interval: {:?}",
        config.background_dht_interval
    );
    println!();

    let manager = BackgroundModeManager::new(config);

    // Simulate app lifecycle
    println!("📱 Simulating mobile app lifecycle...\n");

    // Active state
    println!("1️⃣  App in FOREGROUND (Active)");
    println!("   State: {:?}", manager.state());
    println!("   Allow DHT queries: {}", manager.should_allow_dht_query());
    println!(
        "   Allow announcements: {}",
        manager.should_allow_provider_announcements()
    );
    println!();

    tokio::time::sleep(Duration::from_secs(2)).await;

    // Pause (app goes to background)
    println!("2️⃣  App going to BACKGROUND...");
    manager.pause()?;
    println!("   State: {:?}", manager.state());
    println!("   Allow DHT queries: {}", manager.should_allow_dht_query());
    println!(
        "   Allow announcements: {}",
        manager.should_allow_provider_announcements()
    );

    // Simulate DHT query attempt in background
    println!("\n   🔍 Attempting DHT query...");
    if manager.should_allow_dht_query() {
        println!("      ✅ Query allowed");
    } else {
        println!("      ❌ Query blocked (power saving)");
        manager.record_dht_query_skipped();
    }
    println!();

    tokio::time::sleep(Duration::from_secs(3)).await;

    // Resume (app comes to foreground)
    println!("3️⃣  App returning to FOREGROUND...");
    manager.resume()?;
    println!("   State: {:?}", manager.state());
    println!("   Allow DHT queries: {}", manager.should_allow_dht_query());
    println!(
        "   Allow announcements: {}",
        manager.should_allow_provider_announcements()
    );
    println!();

    tokio::time::sleep(Duration::from_secs(2)).await;

    // Multiple pause/resume cycles
    println!("4️⃣  Simulating multiple background cycles...\n");

    for i in 1..=3 {
        println!("   Cycle {} - Going to background", i);
        manager.pause()?;
        manager.record_connections_closed(2); // Simulate closing idle connections
        tokio::time::sleep(Duration::from_secs(1)).await;

        println!("   Cycle {} - Returning to foreground", i);
        manager.resume()?;
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    println!();

    // Display statistics
    let stats = manager.stats();
    println!("📊 Background Mode Statistics:");
    println!("   Pause count: {}", stats.pause_count);
    println!("   Resume count: {}", stats.resume_count);
    println!(
        "   Total background time: {:?}",
        stats.total_background_time
    );
    println!(
        "   Total foreground time: {:?}",
        stats.total_foreground_time
    );
    println!(
        "   Connections closed on pause: {}",
        stats.connections_closed_on_pause
    );
    println!("   DHT queries skipped: {}", stats.dht_queries_skipped);
    println!();

    // Test different configurations
    println!("5️⃣  Testing different configurations...\n");

    // Balanced configuration
    println!("   📊 Balanced Configuration:");
    let balanced_config = BackgroundModeConfig::balanced();
    let balanced_manager = BackgroundModeManager::new(balanced_config);
    balanced_manager.pause()?;
    println!(
        "      Pause DHT queries: {}",
        balanced_manager.config().pause_dht_queries
    );
    println!(
        "      Allow DHT in background: {}",
        balanced_manager.should_allow_dht_query()
    );
    println!();

    // Server configuration (minimal impact)
    println!("   🖥️  Server Configuration:");
    let server_config = BackgroundModeConfig::server();
    let server_manager = BackgroundModeManager::new(server_config);
    server_manager.pause()?;
    println!(
        "      Pause DHT queries: {}",
        server_manager.config().pause_dht_queries
    );
    println!(
        "      Allow DHT in background: {}",
        server_manager.should_allow_dht_query()
    );
    println!();

    // Invalid state transition example
    println!("6️⃣  Testing state management...\n");
    let test_manager = BackgroundModeManager::new(BackgroundModeConfig::mobile());

    // Try to pause when already paused
    test_manager.pause()?;
    println!("   First pause: Success");

    let result = test_manager.pause();
    match result {
        Ok(_) => println!("   Second pause: Success (no-op)"),
        Err(e) => println!("   Second pause: Error - {}", e),
    }
    println!();

    println!("✅ Background mode example completed!");
    println!("\nKey Takeaways:");
    println!("  • Use BackgroundModeConfig::mobile() for aggressive power saving");
    println!("  • Use BackgroundModeConfig::balanced() for moderate savings");
    println!("  • Use BackgroundModeConfig::server() for minimal impact");
    println!("  • Always check should_allow_*() before operations in background");
    println!("  • Track statistics to optimize power consumption");

    Ok(())
}
