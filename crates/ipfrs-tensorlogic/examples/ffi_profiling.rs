//! Example: FFI overhead profiling
//!
//! This example demonstrates:
//! - Profiling FFI call overhead
//! - Identifying hotspots
//! - Generating profiling reports
//! - Using the global profiler

use ipfrs_tensorlogic::{global_profiler, FfiProfiler};
use std::thread;
use std::time::Duration;

// Simulate FFI functions
fn simulate_ffi_call_fast() {
    thread::sleep(Duration::from_micros(10));
}

fn simulate_ffi_call_medium() {
    thread::sleep(Duration::from_micros(100));
}

fn simulate_ffi_call_slow() {
    thread::sleep(Duration::from_micros(1000));
}

fn simulate_data_transfer(size: usize) {
    let _data = vec![0u8; size];
    thread::sleep(Duration::from_micros(size as u64 / 100));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== FFI Profiling Example ===\n");

    // Create a local profiler
    let profiler = FfiProfiler::new();

    println!("--- Simulating FFI Calls ---");

    // Profile various FFI operations
    for _ in 0..10 {
        let _guard = profiler.start("ffi_fast");
        simulate_ffi_call_fast();
    }
    println!("Profiled 10 fast calls");

    for _ in 0..20 {
        let _guard = profiler.start("ffi_medium");
        simulate_ffi_call_medium();
    }
    println!("Profiled 20 medium calls");

    for _ in 0..5 {
        let _guard = profiler.start("ffi_slow");
        simulate_ffi_call_slow();
    }
    println!("Profiled 5 slow calls");

    // Profile data transfers
    for size in [64, 256, 1024, 4096].iter() {
        let _guard = profiler.start(&format!("data_transfer_{}", size));
        simulate_data_transfer(*size);
    }
    println!("Profiled data transfers\n");

    // Get individual function statistics
    println!("--- Individual Function Stats ---");
    if let Some(stats) = profiler.get_stats("ffi_slow") {
        println!("Function: {}", stats.name);
        println!("  Call count: {}", stats.call_count);
        println!("  Average: {:.2} μs", stats.avg_duration.as_micros());
        println!("  Min: {:.2} μs", stats.min_duration.as_micros());
        println!("  Max: {:.2} μs", stats.max_duration.as_micros());
        println!("  Total: {:.2} μs", stats.total_duration.as_micros());

        // Check against target
        let target_micros = 1000;
        if stats.exceeds_target(target_micros) {
            println!(
                "  ⚠ WARNING: Exceeds target of {} μs by {:.1}%",
                target_micros,
                stats.overhead_percentage(target_micros)
            );
        } else {
            println!("  ✓ Meets target of {} μs", target_micros);
        }
    }

    // Get hotspots (sorted by average duration)
    println!("\n--- Hotspots (sorted by avg duration) ---");
    let hotspots = profiler.get_hotspots();
    for (i, stats) in hotspots.iter().take(5).enumerate() {
        println!(
            "{}. {} - {:.2} μs avg ({} calls)",
            i + 1,
            stats.name,
            stats.avg_duration.as_micros(),
            stats.call_count
        );
    }

    // Generate comprehensive report
    println!("\n--- Profiling Report ---");
    let report = profiler.report();
    println!("Total calls: {}", report.total_calls);
    println!(
        "Total duration: {:.2} ms",
        report.total_duration.as_micros() as f64 / 1000.0
    );

    let summary = report.summary();
    println!(
        "Average call duration: {:.2} μs",
        summary.avg_call_duration.as_micros()
    );
    println!(
        "Max call duration: {:.2} μs",
        summary.max_call_duration.as_micros()
    );
    println!("Functions profiled: {}", summary.functions_profiled);

    // Check if average meets target
    let target = 1000; // 1ms target
    if summary.meets_target(target) {
        println!("✓ Average overhead meets target of {} μs", target);
    } else {
        println!("⚠ Average overhead exceeds target of {} μs", target);
    }

    // Identify bottlenecks
    println!("\n--- Bottleneck Analysis ---");
    let target_threshold = 500; // 500 μs
    let bottlenecks = report.identify_bottlenecks(target_threshold);
    if bottlenecks.is_empty() {
        println!("✓ No functions exceed {} μs threshold", target_threshold);
    } else {
        println!("⚠ Functions exceeding {} μs threshold:", target_threshold);
        for func in bottlenecks {
            println!("  - {}", func);
        }
    }

    // Print detailed report
    println!("\n--- Detailed Report ---");
    report.print();

    // Using global profiler
    println!("\n--- Global Profiler Example ---");
    let global = global_profiler();

    // Reset any previous stats
    global.reset();

    // Profile some operations
    for i in 0..5 {
        let _guard = global.start("global_operation");
        thread::sleep(Duration::from_millis(i * 2));
    }

    // Get stats from global profiler
    if let Some(stats) = global.get_stats("global_operation") {
        println!("Global profiler tracked {} calls", stats.call_count);
        println!("Average duration: {:.2} ms", stats.avg_duration.as_millis());
    }

    // Demonstrate enable/disable
    println!("\n--- Enable/Disable ---");
    println!("Profiler enabled: {}", profiler.is_enabled());

    profiler.disable();
    println!("Profiler disabled: {}", !profiler.is_enabled());

    // This won't be tracked
    {
        let _guard = profiler.start("disabled_call");
        simulate_ffi_call_fast();
    }

    println!(
        "Stats for disabled_call: {:?}",
        profiler.get_stats("disabled_call").map(|s| s.call_count)
    );

    profiler.enable();
    println!("Profiler re-enabled: {}", profiler.is_enabled());

    // Recommendations
    println!("\n--- Recommendations ---");
    println!("• Profile critical paths in your FFI boundary");
    println!("• Identify functions exceeding 1μs overhead");
    println!("• Consider batching small FFI calls");
    println!("• Use zero-copy techniques for data transfer");
    println!("• Cache frequently accessed data on the Rust side");

    println!("\n✓ Example completed successfully!");
    Ok(())
}
