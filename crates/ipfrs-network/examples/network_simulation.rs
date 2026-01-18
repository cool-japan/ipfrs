//! Network Simulation Example
//!
//! This example demonstrates the network simulator module which allows testing
//! application behavior under various network conditions.
//!
//! Run with: `cargo run --example network_simulation`

use ipfrs_network::{NetworkCondition, NetworkSimulator, SimulatorConfig};
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    println!("=== Network Simulator Demo ===\n");

    // Scenario 1: Test predefined network conditions
    println!("Scenario 1: Testing Predefined Network Conditions");
    println!("---------------------------------------------------");

    let conditions = vec![
        ("Perfect Network", NetworkCondition::Perfect),
        ("Good Network", NetworkCondition::Good),
        ("Fair Network", NetworkCondition::Fair),
        ("Poor Network", NetworkCondition::Poor),
        ("Very Poor Network", NetworkCondition::VeryPoor),
        ("Mobile 3G", NetworkCondition::Mobile3G),
        ("Mobile 4G", NetworkCondition::Mobile4G),
        ("Mobile 5G", NetworkCondition::Mobile5G),
        ("Satellite", NetworkCondition::Satellite),
    ];

    for (name, condition) in conditions {
        let simulator = NetworkSimulator::from_condition(condition);
        let config = simulator.config();

        println!("\n{}:", name);
        println!("  Base latency: {} ms", config.base_latency_ms);
        println!("  Latency variance: {} ms", config.latency_variance_ms);
        println!(
            "  Packet loss rate: {:.1}%",
            config.packet_loss_rate * 100.0
        );
        if config.bandwidth_limit_bps > 0 {
            println!(
                "  Bandwidth limit: {:.2} Mbps",
                config.bandwidth_limit_bps as f64 / 1_000_000.0
            );
        }
    }

    // Scenario 2: Simulate packet delivery with latency
    println!("\n\nScenario 2: Packet Delivery Simulation");
    println!("----------------------------------------");

    let simulator = NetworkSimulator::from_condition(NetworkCondition::Poor);
    simulator.start().await?;

    println!("Sending 10 packets through poor network...");

    let mut delivered = 0;
    let mut total_latency = 0u128;

    for i in 1..=10 {
        let start = Instant::now();
        let packet_delivered = simulator.delay_packet(1024).await?;
        let latency = start.elapsed().as_millis();

        if packet_delivered {
            delivered += 1;
            total_latency += latency;
            println!("  Packet {}: Delivered (latency: {} ms)", i, latency);
        } else {
            println!("  Packet {}: DROPPED", i);
        }
    }

    let stats = simulator.stats();
    println!("\nPacket Delivery Statistics:");
    println!(
        "  Packets delivered: {}/{}",
        delivered, stats.packets_processed
    );
    println!(
        "  Packet loss rate: {:.1}%",
        stats.packet_loss_rate() * 100.0
    );
    if delivered > 0 {
        println!(
            "  Average latency: {} ms",
            total_latency / delivered as u128
        );
    }

    simulator.stop().await?;

    // Scenario 3: Custom network condition
    println!("\n\nScenario 3: Custom Network Condition");
    println!("--------------------------------------");

    let custom_config = SimulatorConfig {
        base_latency_ms: 150,
        latency_variance_ms: 75,
        packet_loss_rate: 0.15,  // 15% loss
        spike_probability: 0.25, // 25% chance of spike
        spike_multiplier: 8,
        ..Default::default()
    };

    let custom_simulator = NetworkSimulator::new(custom_config);
    custom_simulator.start().await?;

    println!("Testing custom unreliable network...");
    println!("(150ms base latency, 75ms variance, 15% loss, 25% spike chance)\n");

    let mut spikes = 0;
    for i in 1..=20 {
        let start = Instant::now();
        if custom_simulator.delay_packet(512).await? {
            let latency = start.elapsed().as_millis();
            if latency > 500 {
                spikes += 1;
                println!("  Packet {}: SPIKE (latency: {} ms)", i, latency);
            } else if i % 5 == 0 {
                println!("  Packet {}: Delivered (latency: {} ms)", i, latency);
            }
        }
    }

    let custom_stats = custom_simulator.stats();
    println!("\nCustom Network Statistics:");
    println!("  Packets processed: {}", custom_stats.packets_processed);
    println!(
        "  Packets dropped: {} ({:.1}%)",
        custom_stats.packets_dropped,
        custom_stats.packet_loss_rate() * 100.0
    );
    println!("  Latency spikes: {}", spikes);
    println!("  Average latency: {:.1} ms", custom_stats.avg_latency_ms);
    println!("  Max latency: {} ms", custom_stats.max_latency_ms);

    custom_simulator.stop().await?;

    // Scenario 4: Network partitions
    println!("\n\nScenario 4: Network Partitions");
    println!("--------------------------------");

    let partition_simulator = NetworkSimulator::from_condition(NetworkCondition::Good);

    let group1 = vec!["peer1".to_string(), "peer2".to_string()];
    let group2 = vec!["peer3".to_string(), "peer4".to_string()];

    partition_simulator.create_partition(group1.clone(), group2.clone());

    println!("Created network partition:");
    println!("  Group 1: {:?}", group1);
    println!("  Group 2: {:?}", group2);

    println!("\nPartition status:");
    println!(
        "  peer1 <-> peer3: {}",
        if partition_simulator.is_partitioned("peer1", "peer3") {
            "PARTITIONED"
        } else {
            "Connected"
        }
    );
    println!(
        "  peer1 <-> peer2: {}",
        if partition_simulator.is_partitioned("peer1", "peer2") {
            "PARTITIONED"
        } else {
            "Connected"
        }
    );
    println!(
        "  peer2 <-> peer4: {}",
        if partition_simulator.is_partitioned("peer2", "peer4") {
            "PARTITIONED"
        } else {
            "Connected"
        }
    );

    partition_simulator.clear_partitions();
    println!("\nPartitions cleared.");

    // Scenario 5: Throughput measurement
    println!("\n\nScenario 5: Throughput Measurement");
    println!("------------------------------------");

    let throughput_simulator = NetworkSimulator::from_condition(NetworkCondition::Mobile4G);
    throughput_simulator.start().await?;

    println!("Simulating Mobile 4G network...");
    println!("Sending 1 MB of data (1024 packets of 1KB each)...");

    let packet_count = 1024;
    let packet_size = 1024;

    for _ in 0..packet_count {
        throughput_simulator.delay_packet(packet_size).await?;
    }

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let throughput_stats = throughput_simulator.stats();
    println!("\nThroughput Statistics:");
    println!(
        "  Data transferred: {:.2} MB",
        throughput_stats.bytes_processed as f64 / 1_048_576.0
    );
    println!(
        "  Duration: {:.2} s",
        throughput_stats.duration.as_secs_f64()
    );
    println!(
        "  Throughput: {:.2} Mbps",
        (throughput_stats.throughput_bps() * 8.0) / 1_000_000.0
    );
    println!(
        "  Average latency: {:.1} ms",
        throughput_stats.avg_latency_ms
    );

    throughput_simulator.stop().await?;

    // Scenario 6: Reset statistics
    println!("\n\nScenario 6: Statistics Reset");
    println!("-----------------------------");

    let reset_simulator = NetworkSimulator::from_condition(NetworkCondition::Fair);
    reset_simulator.start().await?;

    for _ in 0..5 {
        reset_simulator.delay_packet(512).await?;
    }

    println!("Before reset:");
    println!(
        "  Packets processed: {}",
        reset_simulator.stats().packets_processed
    );

    reset_simulator.reset_stats();

    println!("After reset:");
    println!(
        "  Packets processed: {}",
        reset_simulator.stats().packets_processed
    );

    for _ in 0..3 {
        reset_simulator.delay_packet(512).await?;
    }

    println!("After sending 3 more packets:");
    println!(
        "  Packets processed: {}",
        reset_simulator.stats().packets_processed
    );

    reset_simulator.stop().await?;

    // Scenario 7: Comparing network conditions
    println!("\n\nScenario 7: Comparing Network Conditions");
    println!("------------------------------------------");

    let comparison_conditions = vec![
        NetworkCondition::Mobile5G,
        NetworkCondition::Mobile4G,
        NetworkCondition::Mobile3G,
    ];

    println!("Sending 50 packets under each condition...\n");

    for condition in comparison_conditions {
        let simulator = NetworkSimulator::from_condition(condition);
        simulator.start().await?;

        for _ in 0..50 {
            simulator.delay_packet(1024).await?;
        }

        let stats = simulator.stats();
        println!("{:?}:", condition);
        println!(
            "  Delivery rate: {:.1}%",
            100.0 - (stats.packet_loss_rate() * 100.0)
        );
        println!("  Average latency: {:.1} ms", stats.avg_latency_ms);
        println!("  Max latency: {} ms", stats.max_latency_ms);

        simulator.stop().await?;
    }

    println!("\n=== Demo Complete ===");
    Ok(())
}
