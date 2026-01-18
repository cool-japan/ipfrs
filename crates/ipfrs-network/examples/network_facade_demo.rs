//! Network Facade demonstration
//!
//! This example shows how to use the NetworkFacade to easily create
//! a fully-featured network node with all advanced capabilities.

use ipfrs_network::{NetworkConfig, NetworkFacadeBuilder};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== Network Facade Demo ===\n");

    // Example 1: Mobile preset with advanced features
    println!("1. Creating mobile-optimized network node...");
    let mut facade = NetworkFacadeBuilder::new()
        .with_preset_mobile()
        .with_semantic_dht()
        .with_gossipsub()
        .with_geo_routing()
        .build()?;

    println!("   Peer ID: {}", facade.peer_id());
    println!("   Features enabled:");
    println!(
        "   - Bandwidth throttling: {}",
        facade.bandwidth_throttle.is_some()
    );
    println!(
        "   - Adaptive polling: {}",
        facade.adaptive_polling.is_some()
    );
    println!("   - Semantic DHT: {}", facade.semantic_dht.is_some());
    println!("   - GossipSub: {}", facade.gossipsub.is_some());
    println!("   - Geographic routing: {}", facade.geo_router.is_some());
    println!("   - Memory monitor: {}", facade.memory_monitor.is_some());
    println!("   - Offline queue: {}", facade.offline_queue.is_some());
    println!("   - Background mode: {}", facade.background_mode.is_some());

    // Example 2: IoT preset
    println!("\n2. Creating IoT-optimized network node...");
    let iot_facade = NetworkFacadeBuilder::new().with_preset_iot().build()?;

    println!("   Peer ID: {}", iot_facade.peer_id());
    println!("   Features enabled:");
    println!(
        "   - Bandwidth throttling: {}",
        iot_facade.bandwidth_throttle.is_some()
    );
    println!(
        "   - Query batching: {}",
        iot_facade.query_batcher.is_some()
    );
    println!(
        "   - Memory monitor: {}",
        iot_facade.memory_monitor.is_some()
    );

    // Example 3: High-performance preset
    println!("\n3. Creating high-performance network node...");
    let hp_facade = NetworkFacadeBuilder::new()
        .with_preset_high_performance()
        .build()?;

    println!("   Peer ID: {}", hp_facade.peer_id());
    let health = hp_facade.get_health();
    println!("   Network health: {:?}", health.status);
    println!("   Connected peers: {}", hp_facade.peer_count());

    // Example 4: Custom configuration
    println!("\n4. Creating custom network node...");
    let custom_config = NetworkConfig {
        listen_addrs: vec!["/ip4/0.0.0.0/udp/9000/quic-v1".to_string()],
        enable_mdns: true,
        ..Default::default()
    };

    let custom_facade = NetworkFacadeBuilder::new()
        .with_config(custom_config)
        .with_quality_predictor()
        .with_peer_selector()
        .with_multipath_quic()
        .build()?;

    println!("   Peer ID: {}", custom_facade.peer_id());
    println!("   Features enabled:");
    println!(
        "   - Quality predictor: {}",
        custom_facade.quality_predictor.is_some()
    );
    println!(
        "   - Peer selector: {}",
        custom_facade.peer_selector.is_some()
    );
    println!(
        "   - Multipath QUIC: {}",
        custom_facade.multipath_quic.is_some()
    );

    // Example 5: Privacy-focused preset
    println!("\n5. Creating privacy-focused network node...");
    let privacy_facade = NetworkFacadeBuilder::new().with_preset_privacy().build()?;

    println!("   Peer ID: {}", privacy_facade.peer_id());
    println!("   Tor integration: Enabled (use with_tor_manager() to initialize)");

    // Example 6: Full-featured node
    println!("\n6. Creating full-featured network node...");
    let full_facade = NetworkFacadeBuilder::new()
        .with_semantic_dht()
        .with_gossipsub()
        .with_geo_routing()
        .with_quality_predictor()
        .with_peer_selector()
        .with_multipath_quic()
        .with_bandwidth_throttle()
        .with_adaptive_polling()
        .with_background_mode()
        .with_offline_queue()
        .with_memory_monitor()
        .with_network_monitor()
        .with_query_batcher()
        .build()?;

    println!("   Peer ID: {}", full_facade.peer_id());
    println!("   All features enabled!");
    println!("   - Semantic DHT: {}", full_facade.semantic_dht.is_some());
    println!("   - GossipSub: {}", full_facade.gossipsub.is_some());
    println!(
        "   - Geographic routing: {}",
        full_facade.geo_router.is_some()
    );
    println!(
        "   - Quality predictor: {}",
        full_facade.quality_predictor.is_some()
    );
    println!(
        "   - Peer selector: {}",
        full_facade.peer_selector.is_some()
    );
    println!(
        "   - Multipath QUIC: {}",
        full_facade.multipath_quic.is_some()
    );
    println!(
        "   - Bandwidth throttle: {}",
        full_facade.bandwidth_throttle.is_some()
    );
    println!(
        "   - Adaptive polling: {}",
        full_facade.adaptive_polling.is_some()
    );
    println!(
        "   - Background mode: {}",
        full_facade.background_mode.is_some()
    );
    println!(
        "   - Offline queue: {}",
        full_facade.offline_queue.is_some()
    );
    println!(
        "   - Memory monitor: {}",
        full_facade.memory_monitor.is_some()
    );
    println!(
        "   - Network monitor: {}",
        full_facade.network_monitor.is_some()
    );
    println!(
        "   - Query batcher: {}",
        full_facade.query_batcher.is_some()
    );

    // Start the node (for demonstration)
    println!("\n7. Starting mobile network node...");
    facade.start().await?;
    println!("   Node started successfully!");

    // Check health
    let health = facade.get_health();
    println!("   Network health: {:?}", health.status);
    println!("   Peer count: {}", facade.peer_count());
    println!("   Is healthy: {}", facade.is_healthy());

    // Get network statistics
    println!("\n8. Network statistics:");
    println!("   Bytes sent: {}", facade.bytes_sent());
    println!("   Bytes received: {}", facade.bytes_received());
    println!(
        "   External addresses: {}",
        facade.node.get_external_addresses().len()
    );
    println!(
        "   Publicly reachable: {}",
        facade.node.is_publicly_reachable()
    );

    // Stop the node
    println!("\n9. Stopping network node...");
    facade.stop().await?;
    println!("   Node stopped successfully!");

    println!("\n=== Demo Complete ===");

    Ok(())
}
