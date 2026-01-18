//! Example: ARM Performance Profiling
//!
//! This example demonstrates how to use the ARM profiler to monitor
//! network performance on ARM devices like Raspberry Pi and Jetson.

use ipfrs_network::{ArmDevice, ArmProfiler, NetworkConfig, NetworkNode};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ARM Performance Profiling Example ===\n");

    // Detect ARM device
    let device = ArmDevice::detect();
    println!("Detected device: {:?}", device);
    println!("Architecture: {}", std::env::consts::ARCH);
    println!();

    // Create profiler with auto-detected configuration
    let mut profiler = ArmProfiler::auto_detect();
    println!("Profiler configuration:");
    let config = profiler.config();
    println!("  Track CPU: {}", config.track_cpu);
    println!("  Track memory: {}", config.track_memory);
    println!("  Track throughput: {}", config.track_throughput);
    println!("  Track latency: {}", config.track_latency);
    println!("  Sample interval: {:?}", config.sample_interval);
    println!("  Max samples: {}", config.max_samples);
    println!();

    // Start profiling
    profiler.start();
    println!("Profiling started...\n");

    // Create network node with device-appropriate configuration
    let network_config = match device {
        ArmDevice::RaspberryPi => NetworkConfig::iot(),
        ArmDevice::Jetson => NetworkConfig::mobile(),
        _ => NetworkConfig::iot(),
    };

    println!("Creating network node...");
    let start = std::time::Instant::now();
    let mut node = NetworkNode::new(network_config)?;
    let creation_time = start.elapsed();
    println!("Node created in {:?}", creation_time);

    // Record node creation latency
    profiler.record_latency(creation_time);

    // Start the node
    println!("\nStarting network node...");
    let start = std::time::Instant::now();
    node.start().await?;
    let startup_time = start.elapsed();
    println!("Node started in {:?}", startup_time);

    profiler.record_latency(startup_time);

    tokio::time::sleep(Duration::from_secs(1)).await;

    // Simulate network operations and profile them
    println!("\n=== Profiling Network Operations ===\n");

    // Profile stats queries
    println!("1. Profiling stats queries (100 iterations)...");
    for i in 0..100 {
        let start = std::time::Instant::now();
        let _ = node.stats();
        profiler.record_latency(start.elapsed());

        // Simulate some CPU usage (varying between 30% and 70%)
        let cpu_usage = 30.0 + ((i % 40) as f64);
        profiler.record_cpu(cpu_usage);
    }

    // Profile health checks
    println!("2. Profiling health checks (100 iterations)...");
    for i in 0..100 {
        let start = std::time::Instant::now();
        let _ = node.get_network_health();
        profiler.record_latency(start.elapsed());

        // Simulate memory usage (varying between 10MB and 15MB)
        let memory_usage = 10 * 1024 * 1024 + ((i % 5) * 1024 * 1024) as u64;
        profiler.record_memory(memory_usage);
    }

    // Profile bandwidth updates
    println!("3. Profiling bandwidth updates (100 iterations)...");
    for _ in 0..100 {
        let start = std::time::Instant::now();
        node.update_bandwidth(1024, 2048);
        profiler.record_latency(start.elapsed());

        // Record throughput
        profiler.record_throughput(1024 * 1024); // 1 MB/s
    }

    println!("\nProfiling complete. Analyzing results...\n");

    // Get profiling statistics
    match profiler.stats() {
        Ok(stats) => {
            println!("=== Performance Statistics ===");
            println!();
            println!("CPU Usage:");
            println!("  Average: {:.2}%", stats.avg_cpu);
            println!("  Peak: {:.2}%", stats.peak_cpu);
            println!();

            println!("Memory Usage:");
            println!(
                "  Average: {:.2} MB",
                stats.avg_memory as f64 / 1024.0 / 1024.0
            );
            println!(
                "  Peak: {:.2} MB",
                stats.peak_memory as f64 / 1024.0 / 1024.0
            );
            println!();

            println!("Network Throughput:");
            println!(
                "  Average: {:.2} MB/s",
                stats.avg_throughput as f64 / 1024.0 / 1024.0
            );
            println!(
                "  Peak: {:.2} MB/s",
                stats.peak_throughput as f64 / 1024.0 / 1024.0
            );
            println!();

            println!("Latency:");
            println!("  Average: {} μs", stats.avg_latency);
            println!("  P95: {} μs", stats.p95_latency);
            println!("  P99: {} μs", stats.p99_latency);
            println!();

            if let Some(avg_temp) = stats.avg_temperature {
                println!("Temperature:");
                println!("  Average: {:.1}°C", avg_temp);
                if let Some(peak_temp) = stats.peak_temperature {
                    println!("  Peak: {:.1}°C", peak_temp);
                }
                println!();
            }

            println!("Profiling Summary:");
            println!("  Total samples: {}", stats.sample_count);
            println!("  Duration: {:?}", stats.duration);
            println!();

            // Recommendations based on results
            println!("=== Performance Recommendations ===");
            println!();

            if stats.avg_cpu > 80.0 {
                println!("⚠️  High CPU usage detected ({}%)", stats.avg_cpu);
                println!("   Consider using low-power mode or reducing connection limits");
            } else if stats.avg_cpu > 60.0 {
                println!("ℹ️  Moderate CPU usage ({}%)", stats.avg_cpu);
            } else {
                println!("✅ CPU usage is healthy ({}%)", stats.avg_cpu);
            }

            let avg_memory_mb = stats.avg_memory as f64 / 1024.0 / 1024.0;
            if avg_memory_mb > 200.0 {
                println!("⚠️  High memory usage detected ({:.2} MB)", avg_memory_mb);
                println!("   Consider using low-memory configuration");
            } else if avg_memory_mb > 100.0 {
                println!("ℹ️  Moderate memory usage ({:.2} MB)", avg_memory_mb);
            } else {
                println!("✅ Memory usage is healthy ({:.2} MB)", avg_memory_mb);
            }

            if stats.p95_latency > 10000 {
                // > 10ms
                println!("⚠️  High latency detected (P95: {} μs)", stats.p95_latency);
                println!("   Consider optimizing query operations or reducing concurrency");
            } else {
                println!("✅ Latency is healthy (P95: {} μs)", stats.p95_latency);
            }

            if let Some(peak_temp) = stats.peak_temperature {
                if peak_temp > 80.0 {
                    println!("🌡️  High temperature detected ({:.1}°C)", peak_temp);
                    println!("   Consider thermal throttling or active cooling");
                }
            }
        }
        Err(e) => {
            eprintln!("Error getting profiling stats: {}", e);
        }
    }

    println!("\n=== Profiling Complete ===");

    Ok(())
}
