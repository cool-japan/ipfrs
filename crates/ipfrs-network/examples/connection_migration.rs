//! Example: QUIC Connection Migration for Mobile Support
//!
//! This example demonstrates how to use the connection migration manager
//! to handle network changes seamlessly, particularly useful for mobile devices.

use ipfrs_network::{ConnectionMigrationManager, MigrationState};
use libp2p::{Multiaddr, PeerId};
use std::str::FromStr;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== QUIC Connection Migration Example ===\n");

    // Create a mobile-optimized migration manager
    let manager = ConnectionMigrationManager::mobile();
    let config = manager.config();

    println!("Migration Configuration:");
    println!("  Auto-migrate: {}", config.auto_migrate);
    println!("  Migration timeout: {:?}", config.migration_timeout);
    println!("  Max retry attempts: {}", config.max_retry_attempts);
    println!("  Retry backoff: {:?}", config.retry_backoff);
    println!("  Migration cooldown: {:?}", config.migration_cooldown);
    println!("  Keep old path: {}", config.keep_old_path);
    println!("  Validate new path: {}\n", config.validate_new_path);

    // Simulate a peer connection
    let peer_id = PeerId::random();
    let wifi_address = Multiaddr::from_str("/ip4/192.168.1.100/tcp/4001")?;
    let cellular_address = Multiaddr::from_str("/ip4/10.0.0.1/tcp/4001")?;

    println!("Simulating network switch for peer: {}", peer_id);
    println!("  Old address (WiFi): {}", wifi_address);
    println!("  New address (Cellular): {}\n", cellular_address);

    // Scenario 1: Successful migration
    println!("=== Scenario 1: Successful Migration ===");

    // Initiate migration
    match manager.initiate_migration(peer_id, wifi_address.clone(), cellular_address.clone()) {
        Ok(_) => println!("✅ Migration initiated"),
        Err(e) => println!("❌ Failed to initiate migration: {}", e),
    }

    // Check migration status
    if manager.is_migrating(&peer_id) {
        println!("✅ Migration in progress");
        if let Some(state) = manager.get_migration_state(&peer_id) {
            println!("   State: {:?}", state);
        }
    }

    // Update migration state through the process
    manager.update_migration_state(&peer_id, MigrationState::Validating)?;
    println!("   Validating new path...");

    tokio::time::sleep(Duration::from_millis(100)).await;

    manager.update_migration_state(&peer_id, MigrationState::Migrating)?;
    println!("   Migrating connection...");

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Complete migration
    match manager.complete_migration(&peer_id) {
        Ok(_) => println!("✅ Migration completed successfully"),
        Err(e) => println!("❌ Failed to complete migration: {}", e),
    }

    // Show statistics
    let stats = manager.stats();
    println!("\nMigration Statistics:");
    println!("  Total attempts: {}", stats.total_attempts);
    println!("  Successful: {}", stats.successful_migrations);
    println!("  Failed: {}", stats.failed_migrations);
    println!("  In progress: {}", stats.in_progress);
    println!("  Average duration: {} ms\n", stats.avg_duration_ms);

    // Scenario 2: Failed migration with retry
    println!("=== Scenario 2: Failed Migration with Retry ===");

    let peer_id2 = PeerId::random();
    let old_addr = Multiaddr::from_str("/ip4/192.168.1.101/tcp/4001")?;
    let new_addr = Multiaddr::from_str("/ip4/10.0.0.2/tcp/4001")?;

    println!("Attempting migration for peer: {}", peer_id2);

    manager.initiate_migration(peer_id2, old_addr.clone(), new_addr.clone())?;
    println!("✅ Migration initiated");

    // Simulate failure
    manager.fail_migration(&peer_id2, "Network unreachable".to_string())?;
    println!("❌ Migration failed: Network unreachable");

    // Retry migration
    println!("\nAttempting retry...");
    tokio::time::sleep(Duration::from_millis(500)).await;

    manager.initiate_migration(peer_id2, old_addr, new_addr)?;
    match manager.retry_migration(&peer_id2) {
        Ok(_) => println!("✅ Retry initiated (attempt #2)"),
        Err(e) => println!("❌ Retry failed: {}", e),
    }

    // Complete the retry
    manager.update_migration_state(&peer_id2, MigrationState::Migrating)?;
    tokio::time::sleep(Duration::from_millis(100)).await;
    manager.complete_migration(&peer_id2)?;
    println!("✅ Migration completed on retry");

    // Show updated statistics
    let stats = manager.stats();
    println!("\nUpdated Statistics:");
    println!("  Total attempts: {}", stats.total_attempts);
    println!("  Successful: {}", stats.successful_migrations);
    println!("  Failed: {}", stats.failed_migrations);
    println!("  Total retries: {}\n", stats.total_retries);

    // Scenario 3: Multiple concurrent migrations
    println!("=== Scenario 3: Multiple Concurrent Migrations ===");

    let peer_id3 = PeerId::random();
    let peer_id4 = PeerId::random();

    manager.initiate_migration(
        peer_id3,
        Multiaddr::from_str("/ip4/192.168.1.102/tcp/4001")?,
        Multiaddr::from_str("/ip4/10.0.0.3/tcp/4001")?,
    )?;

    manager.initiate_migration(
        peer_id4,
        Multiaddr::from_str("/ip4/192.168.1.103/tcp/4001")?,
        Multiaddr::from_str("/ip4/10.0.0.4/tcp/4001")?,
    )?;

    println!("Started {} concurrent migrations", 2);

    let active = manager.get_active_migrations();
    println!("Active migrations:");
    for attempt in &active {
        println!("  - Peer: {}", attempt.peer_id);
        println!("    From: {}", attempt.old_address);
        println!("    To: {}", attempt.new_address);
        println!("    State: {:?}", attempt.state);
        println!("    Retry count: {}", attempt.retry_count);
    }

    // Complete all migrations
    for attempt in active {
        manager.update_migration_state(&attempt.peer_id, MigrationState::Migrating)?;
        tokio::time::sleep(Duration::from_millis(50)).await;
        manager.complete_migration(&attempt.peer_id)?;
    }

    println!("\n✅ All concurrent migrations completed");

    // Scenario 4: Migration cooldown
    println!("\n=== Scenario 4: Migration Cooldown ===");

    let peer_id5 = PeerId::random();
    manager.initiate_migration(
        peer_id5,
        Multiaddr::from_str("/ip4/192.168.1.104/tcp/4001")?,
        Multiaddr::from_str("/ip4/10.0.0.5/tcp/4001")?,
    )?;
    manager.complete_migration(&peer_id5)?;

    println!("First migration completed");

    // Try to migrate again immediately
    if manager.can_migrate(&peer_id5) {
        println!("✅ Can migrate immediately");
    } else {
        println!("⏳ Migration cooldown active");
        println!("   (Cooldown period: {:?})", config.migration_cooldown);
    }

    // Final statistics
    let stats = manager.stats();
    println!("\n=== Final Statistics ===");
    println!("Total attempts: {}", stats.total_attempts);
    println!("Successful migrations: {}", stats.successful_migrations);
    println!("Failed migrations: {}", stats.failed_migrations);
    println!("Total retries: {}", stats.total_retries);
    println!("Average duration: {} ms", stats.avg_duration_ms);

    println!("\n=== Example Complete ===");

    Ok(())
}
