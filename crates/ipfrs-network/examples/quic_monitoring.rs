//! QUIC monitoring example
//!
//! This example demonstrates how to use the QUIC utilities for:
//! - Creating QUIC configurations
//! - Monitoring QUIC connections
//! - Tracking QUIC statistics
//! - Configuring congestion control

use ipfrs_network::quic::{CongestionControl, QuicConfig, QuicMonitor};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

fn main() {
    println!("=== QUIC Monitoring Example ===\n");

    // Scenario 1: Default QUIC Configuration
    scenario_1_default_config();

    // Scenario 2: Low-Latency Configuration
    scenario_2_low_latency();

    // Scenario 3: High-Throughput Configuration
    scenario_3_high_throughput();

    // Scenario 4: Mobile Configuration
    scenario_4_mobile();

    // Scenario 5: Connection Monitoring
    scenario_5_connection_monitoring();

    // Scenario 6: Custom Configuration Builder
    scenario_6_custom_config();
}

fn scenario_1_default_config() {
    println!("--- Scenario 1: Default QUIC Configuration ---");

    let config = QuicConfig::default();
    println!("Max idle timeout: {} ms", config.max_idle_timeout_ms);
    println!("Keep-alive interval: {} ms", config.keep_alive_interval_ms);
    println!(
        "Max concurrent bidirectional streams: {}",
        config.max_concurrent_bidi_streams
    );
    println!(
        "Max concurrent unidirectional streams: {}",
        config.max_concurrent_uni_streams
    );
    println!("Initial max data: {} bytes", config.initial_max_data);
    println!("Max stream data: {} bytes", config.max_stream_data);
    println!(
        "Max UDP payload size: {} bytes",
        config.max_udp_payload_size
    );
    println!("Congestion control: {:?}", config.congestion_control);
    println!("0-RTT enabled: {}", config.enable_0rtt);
    println!("Datagrams enabled: {}", config.enable_datagrams);
    println!();
}

fn scenario_2_low_latency() {
    println!("--- Scenario 2: Low-Latency Configuration ---");
    println!("Optimized for gaming, real-time communications, and interactive applications\n");

    let config = QuicConfig::low_latency();
    println!(
        "Congestion control: {:?} (BBR for better adaptation)",
        config.congestion_control
    );
    println!(
        "Max idle timeout: {} ms (shorter)",
        config.max_idle_timeout_ms
    );
    println!(
        "Keep-alive interval: {} ms (more frequent)",
        config.keep_alive_interval_ms
    );
    println!(
        "Max concurrent streams: {} (fewer for lower overhead)",
        config.max_concurrent_bidi_streams
    );
    println!(
        "Max UDP payload: {} bytes (smaller for faster transmission)",
        config.max_udp_payload_size
    );
    println!();
}

fn scenario_3_high_throughput() {
    println!("--- Scenario 3: High-Throughput Configuration ---");
    println!("Optimized for file transfers, video streaming, and bulk data\n");

    let config = QuicConfig::high_throughput();
    println!(
        "Congestion control: {:?} (CUBIC for high bandwidth)",
        config.congestion_control
    );
    println!(
        "Max idle timeout: {} ms (longer)",
        config.max_idle_timeout_ms
    );
    println!(
        "Max concurrent streams: {} (many for parallelism)",
        config.max_concurrent_bidi_streams
    );
    println!(
        "Initial max data: {} MB (large window)",
        config.initial_max_data / 1_000_000
    );
    println!(
        "Max stream data: {} MB (large buffers)",
        config.max_stream_data / 1_000_000
    );
    println!(
        "Datagram buffers: {} KB",
        config.datagram_recv_buffer_size / 1024
    );
    println!();
}

fn scenario_4_mobile() {
    println!("--- Scenario 4: Mobile Configuration ---");
    println!("Optimized for battery life and unreliable networks\n");

    let config = QuicConfig::mobile();
    println!(
        "Congestion control: {:?} (BBR for varying conditions)",
        config.congestion_control
    );
    println!(
        "Max concurrent streams: {} (fewer for efficiency)",
        config.max_concurrent_bidi_streams
    );
    println!(
        "Initial max data: {} MB (smaller)",
        config.initial_max_data / 1_000_000
    );
    println!(
        "Max UDP payload: {} bytes (smaller for packet loss)",
        config.max_udp_payload_size
    );
    println!(
        "Datagrams enabled: {} (disabled to save battery)",
        config.enable_datagrams
    );
    println!();
}

fn scenario_5_connection_monitoring() {
    println!("--- Scenario 5: Connection Monitoring ---");
    println!("Track QUIC connections and statistics\n");

    // Create monitor with default config
    let monitor = QuicMonitor::default();

    // Simulate establishing connections
    let peer1 = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100)), 4433);
    let peer2 = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 101)), 4433);
    let peer3 = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 102)), 4433);

    println!("Establishing connections...");
    monitor.record_connection_established(peer1, false);
    monitor.record_connection_established(peer2, true); // 0-RTT connection
    monitor.record_connection_established(peer3, false);

    println!("Active connections: {}", monitor.active_connection_count());
    println!();

    // Update connection metrics
    println!("Updating connection metrics...");
    monitor.update_rtt(&peer1, Duration::from_millis(25));
    monitor.update_rtt(&peer2, Duration::from_millis(50));
    monitor.update_rtt(&peer3, Duration::from_millis(75));

    monitor.update_bytes(&peer1, 1_000_000, 500_000);
    monitor.update_bytes(&peer2, 2_000_000, 1_000_000);
    monitor.update_bytes(&peer3, 500_000, 250_000);

    monitor.update_streams(&peer1, 5, 2);
    monitor.update_streams(&peer2, 10, 5);
    monitor.update_streams(&peer3, 3, 1);

    // Simulate connection migration
    println!("Simulating connection migration for peer1...");
    monitor.record_migration(&peer1);
    monitor.record_migration(&peer1);

    // Get connection info
    println!("\nConnection details:");
    if let Some(info) = monitor.get_connection(&peer1) {
        println!(
            "  Peer 1 ({}:{}):",
            info.remote_addr.ip(),
            info.remote_addr.port()
        );
        println!("    State: {:?}", info.state);
        println!("    RTT: {:?}", info.rtt);
        println!("    Bytes sent: {}", info.bytes_sent);
        println!("    Bytes received: {}", info.bytes_received);
        println!(
            "    Active bidirectional streams: {}",
            info.active_bidi_streams
        );
        println!(
            "    Active unidirectional streams: {}",
            info.active_uni_streams
        );
        println!("    Migrations: {}", info.migration_count);
    }

    // Get overall statistics
    let stats = monitor.stats();
    println!("\nOverall statistics:");
    println!(
        "  Total connections established: {}",
        stats.connections_established
    );
    println!("  Active connections: {}", stats.active_connections);
    println!("  0-RTT connections: {}", stats.zero_rtt_connections);
    println!("  Average RTT: {:.2} ms", stats.avg_rtt_ms);

    // Close one connection
    println!("\nClosing connection to peer2...");
    monitor.record_connection_closed(&peer2);

    let stats = monitor.stats();
    println!(
        "Active connections after close: {}",
        stats.active_connections
    );
    println!("Total connections closed: {}", stats.connections_closed);
    println!("Total bytes sent: {}", stats.total_bytes_sent);
    println!("Total bytes received: {}", stats.total_bytes_received);
    println!();

    // Simulate failed connection
    println!("Simulating failed connection to peer3...");
    monitor.record_connection_failed(&peer3);

    let stats = monitor.stats();
    println!(
        "Active connections after failure: {}",
        stats.active_connections
    );
    println!("Total connections failed: {}", stats.connections_failed);
    println!();
}

fn scenario_6_custom_config() {
    println!("--- Scenario 6: Custom Configuration Builder ---");
    println!("Build a custom configuration using the builder pattern\n");

    let config = QuicConfig::default()
        .with_max_idle_timeout(45_000)
        .with_keep_alive(12_000)
        .with_congestion_control(CongestionControl::Bbr)
        .with_0rtt(false)
        .with_datagrams(true);

    println!("Custom configuration:");
    println!("  Max idle timeout: {} ms", config.max_idle_timeout_ms);
    println!(
        "  Keep-alive interval: {} ms",
        config.keep_alive_interval_ms
    );
    println!("  Congestion control: {:?}", config.congestion_control);
    println!("  0-RTT enabled: {}", config.enable_0rtt);
    println!("  Datagrams enabled: {}", config.enable_datagrams);
    println!();

    println!("Use cases for custom configuration:");
    println!("  - Fine-tuning for specific network conditions");
    println!("  - Balancing latency vs throughput");
    println!("  - Adapting to application requirements");
    println!("  - Testing different congestion control algorithms");
    println!();
}
