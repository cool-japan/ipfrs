//! Configuration presets example
//!
//! This example demonstrates how to use pre-configured network presets
//! for different scenarios instead of manually configuring each component.

use ipfrs_network::presets::NetworkPreset;

fn main() {
    println!("=== Configuration Presets Example ===\n");

    // Scenario 1: Default preset
    println!("--- Scenario 1: Default Preset ---");
    let preset = NetworkPreset::default_preset();
    print_preset_info(&preset);

    // Scenario 2: Low memory devices (< 128 MB RAM)
    println!("\n--- Scenario 2: Low Memory Preset ---");
    println!("For: Raspberry Pi Zero, embedded devices, microcontrollers\n");
    let preset = NetworkPreset::low_memory();
    print_preset_info(&preset);

    // Scenario 3: IoT devices (128-512 MB RAM)
    println!("\n--- Scenario 3: IoT Preset ---");
    println!("For: ESP32, Raspberry Pi 3, IoT gateways\n");
    let preset = NetworkPreset::iot();
    print_preset_info(&preset);

    // Scenario 4: Mobile devices
    println!("\n--- Scenario 4: Mobile Preset ---");
    println!("For: iOS, Android, tablets with battery optimization\n");
    let preset = NetworkPreset::mobile();
    print_preset_info(&preset);

    // Scenario 5: High performance servers
    println!("\n--- Scenario 5: High Performance Preset ---");
    println!("For: Servers, desktops with ample resources (> 2 GB RAM)\n");
    let preset = NetworkPreset::high_performance();
    print_preset_info(&preset);

    // Scenario 6: Low latency applications
    println!("\n--- Scenario 6: Low Latency Preset ---");
    println!("For: Gaming, VoIP, real-time communications\n");
    let preset = NetworkPreset::low_latency();
    print_preset_info(&preset);

    // Scenario 7: High throughput applications
    println!("\n--- Scenario 7: High Throughput Preset ---");
    println!("For: CDN, video streaming, bulk data transfers\n");
    let preset = NetworkPreset::high_throughput();
    print_preset_info(&preset);

    // Scenario 8: Privacy-focused applications
    println!("\n--- Scenario 8: Privacy Preset ---");
    println!("For: Anonymous networking, whistleblowing platforms\n");
    let preset = NetworkPreset::privacy();
    print_preset_info(&preset);

    // Scenario 9: Development and testing
    println!("\n--- Scenario 9: Development Preset ---");
    println!("For: Local development, testing, debugging\n");
    let preset = NetworkPreset::development();
    print_preset_info(&preset);

    // Scenario 10: Compare presets
    println!("\n--- Scenario 10: Preset Comparison ---");
    compare_presets();

    // Scenario 11: Using a preset to create a node
    println!("\n--- Scenario 11: Creating a Node with Preset ---");
    println!("Example code:");
    println!("  let preset = NetworkPreset::mobile();");
    println!("  let node = NetworkNode::new(preset.network)?;");
    println!("  // Configure other components with preset configs:");
    println!("  let throttle = BandwidthThrottle::new(preset.throttle.unwrap());");
    println!("  let polling = AdaptivePolling::new(preset.adaptive_polling.unwrap());");
}

fn print_preset_info(preset: &NetworkPreset) {
    println!("Preset: {}", preset.name());
    println!("Description: {}", preset.description);
    println!();

    // Network configuration
    println!("Network Configuration:");
    println!("  QUIC enabled: {}", preset.network.enable_quic);
    println!("  mDNS enabled: {}", preset.network.enable_mdns);
    println!("  NAT traversal: {}", preset.network.enable_nat_traversal);
    println!();

    // Connection limits
    println!("Connection Limits:");
    println!(
        "  Max connections: {}",
        preset.connection_limits.max_connections
    );
    println!("  Max inbound: {}", preset.connection_limits.max_inbound);
    println!("  Max outbound: {}", preset.connection_limits.max_outbound);
    println!(
        "  Reserved slots: {}",
        preset.connection_limits.reserved_slots
    );
    println!(
        "  Idle timeout: {:?}",
        preset.connection_limits.idle_timeout
    );
    println!();

    // QUIC configuration
    println!("QUIC Configuration:");
    println!("  Congestion control: {:?}", preset.quic.congestion_control);
    println!("  Max idle timeout: {} ms", preset.quic.max_idle_timeout_ms);
    println!("  Keep-alive: {} ms", preset.quic.keep_alive_interval_ms);
    println!("  0-RTT enabled: {}", preset.quic.enable_0rtt);
    println!("  Datagrams enabled: {}", preset.quic.enable_datagrams);
    println!();

    // Peer store configuration
    println!("Peer Store:");
    println!("  Max peers: {}", preset.peer_store.max_peers);
    println!(
        "  Max addresses per peer: {}",
        preset.peer_store.max_addrs_per_peer
    );
    println!();

    // Enabled features
    let features = preset.features_summary();
    if !features.is_empty() {
        println!("Enabled Features:");
        for feature in &features {
            println!("  ✓ {}", feature);
        }
        println!();
    }
}

fn compare_presets() {
    let presets = [
        ("Low Memory", NetworkPreset::low_memory()),
        ("IoT", NetworkPreset::iot()),
        ("Mobile", NetworkPreset::mobile()),
        ("High Performance", NetworkPreset::high_performance()),
    ];

    println!("Comparison Table:");
    println!(
        "{:<20} {:>15} {:>15} {:>15}",
        "Preset", "Max Connections", "Features", "Memory Use"
    );
    println!("{:-<20} {:-<15} {:-<15} {:-<15}", "", "", "", "");

    for (name, preset) in &presets {
        let feature_count = preset.features_summary().len();
        let memory_use = match preset.connection_limits.max_connections {
            n if n <= 16 => "Very Low",
            n if n <= 64 => "Low",
            n if n <= 256 => "Medium",
            _ => "High",
        };

        println!(
            "{:<20} {:>15} {:>15} {:>15}",
            name, preset.connection_limits.max_connections, feature_count, memory_use
        );
    }

    println!();
    println!("Key Insights:");
    println!("  • Low Memory: Minimal footprint, essential features only");
    println!("  • IoT: Balanced for moderate resources");
    println!("  • Mobile: Battery-optimized with full feature set");
    println!("  • High Performance: No limits, maximum throughput");
}
