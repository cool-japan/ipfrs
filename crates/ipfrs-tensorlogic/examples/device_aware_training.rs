//! Example: Device-aware training with adaptive batch sizing
//!
//! This example demonstrates:
//! - Device capability detection
//! - Adaptive batch size calculation
//! - Memory pressure monitoring
//! - Device profiling

use ipfrs_tensorlogic::{AdaptiveBatchSizer, DeviceCapabilities, DeviceProfiler};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Device-Aware Training Example ===\n");

    // Detect device capabilities
    println!("Detecting device capabilities...");
    let capabilities = DeviceCapabilities::detect()?;

    println!("\n--- Device Information ---");
    println!("Device Type: {:?}", capabilities.device_type);
    println!(
        "CPU: {} logical cores, {} physical cores",
        capabilities.cpu.logical_cores, capabilities.cpu.physical_cores
    );
    println!("Architecture: {:?}", capabilities.cpu.arch);
    println!(
        "Memory: {:.2} GB total, {:.2} GB available",
        capabilities.memory.total_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
        capabilities.memory.available_bytes as f64 / 1024.0 / 1024.0 / 1024.0
    );
    println!(
        "Memory Pressure: {:.1}%",
        capabilities.memory.pressure * 100.0
    );
    println!("GPU Available: {}", capabilities.has_gpu);
    println!("Fast Storage: {}", capabilities.has_fast_storage);

    // Get recommendations
    println!("\n--- Recommendations ---");
    println!(
        "Recommended threads: {}",
        capabilities.cpu.recommended_threads()
    );
    println!(
        "Recommended workers: {}",
        capabilities.recommended_workers()
    );

    // Profile device performance
    println!("\n--- Performance Profile ---");
    let profiler = DeviceProfiler::new(Arc::new(capabilities.clone()));
    let performance_tier = profiler.performance_tier();
    println!("Performance Tier: {:?}", performance_tier);

    println!("Profiling memory bandwidth...");
    let memory_bandwidth = profiler.profile_memory_bandwidth();
    println!("Memory Bandwidth: {:.2} GB/s", memory_bandwidth);

    println!("Profiling compute throughput...");
    let compute_throughput = profiler.profile_compute_throughput();
    println!("Compute Throughput: {:.2} GFLOPS", compute_throughput / 1e9);

    // Simulate training scenario
    println!("\n--- Training Scenario Simulation ---");

    // Model parameters
    let model_size_mb = 500; // 500 MB model
    let model_size_bytes = model_size_mb * 1024 * 1024;

    let batch_item_size_kb = 256; // 256 KB per item
    let batch_item_size_bytes = batch_item_size_kb * 1024;

    println!("Model size: {} MB", model_size_mb);
    println!("Batch item size: {} KB", batch_item_size_kb);

    // Calculate optimal batch size
    let optimal_batch =
        capabilities.optimal_batch_size(model_size_bytes as u64, batch_item_size_bytes as u64);
    println!("Optimal batch size: {}", optimal_batch);

    // Use adaptive batch sizer
    let caps_arc = Arc::new(capabilities);
    let sizer = AdaptiveBatchSizer::new(caps_arc.clone())
        .with_min_batch_size(1)
        .with_max_batch_size(256)
        .with_target_utilization(0.7);

    let adaptive_batch = sizer.calculate(batch_item_size_bytes as u64, model_size_bytes as u64);
    println!("Adaptive batch size: {}", adaptive_batch);

    // Simulate different memory pressure scenarios
    println!("\n--- Memory Pressure Adaptation ---");
    let scenarios = vec![
        ("Low pressure", 0.2),
        ("Medium pressure", 0.5),
        ("High pressure", 0.75),
        ("Critical pressure", 0.95),
    ];

    let mut current_batch = adaptive_batch;
    for (scenario, pressure) in scenarios {
        // Update memory pressure (in real scenario, this would be detected)
        let mut caps_modified = (*caps_arc).clone();
        caps_modified.memory.pressure = pressure;

        let sizer_modified = AdaptiveBatchSizer::new(Arc::new(caps_modified));
        let adjusted = sizer_modified.adjust_for_pressure(current_batch);

        println!(
            "{}: pressure={:.1}%, batch_size={}",
            scenario,
            pressure * 100.0,
            adjusted
        );
        current_batch = adjusted;
    }

    // Training recommendations
    println!("\n--- Training Recommendations ---");
    println!(
        "• Use {} worker threads for data loading",
        caps_arc.recommended_workers()
    );
    println!("• Start with batch size of {}", adaptive_batch);
    println!("• Monitor memory pressure and adjust batch size dynamically");
    if caps_arc.has_fast_storage {
        println!("• Fast storage detected: prefetching recommended");
    }
    if caps_arc.has_gpu {
        println!("• GPU detected: consider GPU acceleration");
    }

    println!("\n✓ Example completed successfully!");
    Ok(())
}
