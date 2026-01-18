//! Example: Network Interface Monitoring
//!
//! This example demonstrates how to monitor network interface changes,
//! which is essential for mobile devices that switch between WiFi and cellular.

use ipfrs_network::{NetworkChange, NetworkMonitor, NetworkMonitorConfig};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== IPFRS Network Interface Monitoring Example ===\n");

    // Create network monitor with mobile configuration
    let config = NetworkMonitorConfig::mobile();
    println!("Network Monitor Configuration:");
    println!("  Poll interval: {:?}", config.poll_interval);
    println!("  Debounce duration: {:?}", config.debounce_duration);
    println!("  Monitor loopback: {}", config.monitor_loopback);
    println!();

    let monitor = NetworkMonitor::new(config);

    println!("Starting network interface monitoring...");
    println!("This example uses mock interface detection.");
    println!("In production, implement platform-specific interface querying.\n");

    // Monitor for changes
    let mut check_count = 0;
    loop {
        // Check for network changes
        match monitor.check_for_changes() {
            Ok(changes) => {
                if !changes.is_empty() {
                    println!("\n🔄 Network changes detected:");
                    for change in changes {
                        match change {
                            NetworkChange::InterfaceAdded(iface) => {
                                println!(
                                    "  ➕ Interface added: {} ({:?})",
                                    iface.name, iface.interface_type
                                );
                            }
                            NetworkChange::InterfaceRemoved(iface) => {
                                println!(
                                    "  ➖ Interface removed: {} ({:?})",
                                    iface.name, iface.interface_type
                                );
                            }
                            NetworkChange::PrimaryInterfaceChanged { old, new } => {
                                println!("  🔀 Primary interface changed:");
                                if let Some(old_iface) = old {
                                    println!(
                                        "     From: {} ({:?})",
                                        old_iface.name, old_iface.interface_type
                                    );
                                }
                                println!("     To: {} ({:?})", new.name, new.interface_type);
                            }
                            NetworkChange::AddressChanged {
                                interface,
                                old_addresses,
                                new_addresses,
                            } => {
                                println!("  📍 Address changed on {}:", interface);
                                println!("     Old: {:?}", old_addresses);
                                println!("     New: {:?}", new_addresses);
                            }
                            NetworkChange::InterfaceUp(iface) => {
                                println!(
                                    "  ⬆️  Interface up: {} ({:?})",
                                    iface.name, iface.interface_type
                                );
                            }
                            NetworkChange::InterfaceDown(iface) => {
                                println!(
                                    "  ⬇️  Interface down: {} ({:?})",
                                    iface.name, iface.interface_type
                                );
                            }
                        }
                    }
                    println!();
                }
            }
            Err(e) => {
                eprintln!("Error checking for changes: {}", e);
            }
        }

        // Display current state
        check_count += 1;
        if check_count % 10 == 0 {
            let interfaces = monitor.get_interfaces();
            let primary = monitor.get_primary_interface();
            let stats = monitor.stats();

            println!("\n📊 Network Status:");
            println!("  Active interfaces: {}", interfaces.len());
            if let Some(primary_iface) = primary {
                println!(
                    "  Primary interface: {} ({:?})",
                    primary_iface.name, primary_iface.interface_type
                );
            } else {
                println!("  Primary interface: None");
            }
            println!("\n  Statistics:");
            println!("    Interfaces added: {}", stats.interfaces_added);
            println!("    Interfaces removed: {}", stats.interfaces_removed);
            println!("    Primary changes: {}", stats.primary_changes);
            println!("    Address changes: {}", stats.address_changes);
            println!("    Total changes: {}", stats.total_changes);
            println!();
        }

        // Wait before next check
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Exit after 60 seconds for demo
        if check_count >= 30 {
            println!(
                "\nDemo completed. In a real application, monitoring would continue indefinitely."
            );
            break;
        }
    }

    println!("\n✅ Network monitoring example completed!");
    Ok(())
}
