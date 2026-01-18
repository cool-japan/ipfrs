//! Performance Benchmarking Example
//!
//! This example demonstrates the benchmarking module for measuring
//! network component performance.
//!
//! Run with: `cargo run --example performance_benchmarking`

use ipfrs_network::{BenchmarkConfig, BenchmarkType, PerformanceBenchmark};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    println!("=== Performance Benchmarking Demo ===\n");

    // Scenario 1: Quick benchmark configurations
    println!("Scenario 1: Quick Benchmarks");
    println!("------------------------------");

    let quick_bench = PerformanceBenchmark::new(BenchmarkConfig::quick());
    println!("Running quick connection benchmark (50 iterations)...");

    let result = quick_bench.bench_connection_establishment(50).await?;
    println!("\nConnection Establishment Results:");
    println!("  Operations: {}", result.operations);
    println!("  Success Rate: {:.1}%", result.success_rate());
    println!("  Average: {:.3} ms", result.avg_duration_ms);
    println!("  Median: {:.3} ms", result.median_duration_ms);
    println!("  Min: {:.3} ms", result.min_duration_ms);
    println!("  Max: {:.3} ms", result.max_duration_ms);
    println!("  P95: {:.3} ms", result.p95_latency_ms);
    println!("  P99: {:.3} ms", result.p99_latency_ms);
    println!("  Std Dev: {:.3} ms", result.std_deviation_ms);
    println!("  Throughput: {:.2} ops/s", result.throughput_ops);

    // Scenario 2: DHT query benchmarks
    println!("\n\nScenario 2: DHT Query Benchmarks");
    println!("----------------------------------");

    println!("Running DHT query benchmark (50 iterations)...");
    let dht_result = quick_bench.bench_dht_query(50).await?;

    println!("\nDHT Query Results:");
    println!("  Operations: {}", dht_result.operations);
    println!("  Successful: {}", dht_result.successful_operations);
    println!("  Success Rate: {:.1}%", dht_result.success_rate());
    println!("  Average: {:.2} ms", dht_result.avg_duration_ms);
    println!("  P95: {:.2} ms", dht_result.p95_latency_ms);
    println!("  P99: {:.2} ms", dht_result.p99_latency_ms);
    println!("  Throughput: {:.2} queries/s", dht_result.throughput_ops);

    if let Some(mem) = dht_result.memory_bytes {
        println!("  Avg Memory: {} bytes", mem);
    }
    if let Some(peak_mem) = dht_result.peak_memory_bytes {
        println!("  Peak Memory: {} bytes", peak_mem);
    }

    // Scenario 3: Throughput benchmarks
    println!("\n\nScenario 3: Throughput Benchmarks");
    println!("-----------------------------------");

    let message_sizes = vec![
        (512, "512 B"),
        (1024, "1 KB"),
        (10240, "10 KB"),
        (102400, "100 KB"),
    ];

    println!("Testing message throughput at different sizes...\n");

    for (size, label) in message_sizes {
        let throughput_result = quick_bench.bench_throughput(50, size).await?;

        println!("Message Size: {}", label);
        println!("  Messages Processed: {}", throughput_result.operations);
        println!(
            "  Average Latency: {:.3} ms",
            throughput_result.avg_duration_ms
        );
        println!(
            "  Throughput: {:.2} msg/s",
            throughput_result.throughput_ops
        );

        if let Some(mem) = throughput_result.memory_bytes {
            println!("  Avg Memory/Message: {} bytes", mem);
        }
        println!();
    }

    // Scenario 4: Custom benchmarks
    println!("\nScenario 4: Custom Benchmarks");
    println!("-------------------------------");

    println!("Running custom operation benchmark...");

    let custom_result = quick_bench
        .bench_custom(BenchmarkType::Custom(1), || async {
            // Simulate some custom network operation
            tokio::time::sleep(Duration::from_micros(500)).await;
            // Simulate 98% success rate
            rand::random::<f64>() > 0.02
        })
        .await?;

    println!("\nCustom Operation Results:");
    println!("  Operations: {}", custom_result.operations);
    println!("  Success Rate: {:.1}%", custom_result.success_rate());
    println!("  Average: {:.3} ms", custom_result.avg_duration_ms);
    println!("  P95: {:.3} ms", custom_result.p95_latency_ms);
    println!("  Throughput: {:.2} ops/s", custom_result.throughput_ops);

    // Scenario 5: Thorough benchmarks
    println!("\n\nScenario 5: Thorough Benchmarks");
    println!("---------------------------------");

    let thorough_bench = PerformanceBenchmark::new(BenchmarkConfig::thorough());
    println!("Running thorough connection benchmark (500 iterations)...");
    println!("This may take a moment...");

    let thorough_result = thorough_bench.bench_connection_establishment(500).await?;

    println!("\nThorough Connection Benchmark:");
    println!("  Operations: {}", thorough_result.operations);
    println!("  Average: {:.3} ms", thorough_result.avg_duration_ms);
    println!("  Median: {:.3} ms", thorough_result.median_duration_ms);
    println!("  P95: {:.3} ms", thorough_result.p95_latency_ms);
    println!("  P99: {:.3} ms", thorough_result.p99_latency_ms);
    println!("  Std Dev: {:.3} ms", thorough_result.std_deviation_ms);
    println!("  Total Time: {:.2} ms", thorough_result.total_time_ms);

    // Scenario 6: Performance criteria checking
    println!("\n\nScenario 6: Performance Criteria");
    println!("----------------------------------");

    println!("Checking if benchmarks meet performance criteria...\n");

    let criteria_tests = vec![
        (
            "Connection < 1ms, 95% success",
            thorough_result.meets_criteria(1.0, 95.0),
        ),
        (
            "Connection < 5ms, 95% success",
            thorough_result.meets_criteria(5.0, 95.0),
        ),
        (
            "DHT Query < 20ms, 90% success",
            dht_result.meets_criteria(20.0, 90.0),
        ),
        (
            "DHT Query < 10ms, 90% success",
            dht_result.meets_criteria(10.0, 90.0),
        ),
    ];

    for (test, passed) in criteria_tests {
        let status = if passed { "✓ PASS" } else { "✗ FAIL" };
        println!("  {} - {}", status, test);
    }

    // Scenario 7: Multiple benchmark runs
    println!("\n\nScenario 7: Multiple Benchmark Runs");
    println!("-------------------------------------");

    let multi_bench = PerformanceBenchmark::new(BenchmarkConfig::quick());

    println!("Running 3 rounds of connection benchmarks...\n");

    for i in 1..=3 {
        println!("Round {}:", i);
        let round_result = multi_bench.bench_connection_establishment(30).await?;
        println!("  Average: {:.3} ms", round_result.avg_duration_ms);
        println!("  P95: {:.3} ms", round_result.p95_latency_ms);
        println!("  Throughput: {:.2} ops/s", round_result.throughput_ops);
    }

    // Check all stored results
    if let Some(conn_results) = multi_bench.results_for(BenchmarkType::ConnectionEstablishment) {
        println!("\nStored {} benchmark results", conn_results.len());

        let avg_of_avgs: f64 =
            conn_results.iter().map(|r| r.avg_duration_ms).sum::<f64>() / conn_results.len() as f64;

        println!("Average across all runs: {:.3} ms", avg_of_avgs);
    }

    // Scenario 8: Summary report
    println!("\n\nScenario 8: Summary Report");
    println!("---------------------------");

    let report_bench = PerformanceBenchmark::new(BenchmarkConfig::quick());

    // Run various benchmarks
    report_bench.bench_connection_establishment(20).await?;
    report_bench.bench_dht_query(20).await?;
    report_bench.bench_throughput(20, 1024).await?;

    println!("{}", report_bench.summary_report());

    // Scenario 9: Production monitoring configuration
    println!("\nScenario 9: Production Monitoring");
    println!("-----------------------------------");

    let prod_config = BenchmarkConfig::production();
    println!("Production monitoring configuration:");
    println!("  Warmup iterations: {}", prod_config.warmup_iterations);
    println!("  Iterations: {}", prod_config.iterations);
    println!("  Sample rate: 1/{}", prod_config.sample_rate);
    println!("  Track memory: {}", prod_config.track_memory);
    println!("  Track CPU: {}", prod_config.track_cpu);

    let prod_bench = PerformanceBenchmark::new(prod_config);
    let prod_result = prod_bench.bench_connection_establishment(10).await?;

    println!("\nProduction benchmark result:");
    println!("  Operations: {}", prod_result.operations);
    println!("  Average: {:.3} ms", prod_result.avg_duration_ms);
    println!("  P95: {:.3} ms", prod_result.p95_latency_ms);

    // Scenario 10: Comparison of configurations
    println!("\n\nScenario 10: Configuration Comparison");
    println!("---------------------------------------");

    let configs = vec![
        ("Quick", BenchmarkConfig::quick()),
        ("Default", BenchmarkConfig::default()),
        ("Thorough", BenchmarkConfig::thorough()),
    ];

    println!("Comparing different benchmark configurations:\n");

    for (name, config) in configs {
        println!("{}:", name);
        println!("  Warmup: {} iterations", config.warmup_iterations);
        println!("  Benchmark: {} iterations", config.iterations);
        println!("  Timeout: {:?}", config.operation_timeout);
        println!("  Track memory: {}", config.track_memory);
        println!();
    }

    println!("=== Demo Complete ===");
    Ok(())
}
