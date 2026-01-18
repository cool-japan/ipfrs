//! Metrics time-series aggregation example
//!
//! This example demonstrates how to use the metrics aggregator for
//! tracking and analyzing network metrics over time.

use ipfrs_network::metrics_aggregator::{AggregatorConfig, MetricsAggregator, TimeWindow};
use std::thread;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Metrics Time-Series Aggregation Demo ===\n");

    // Scenario 1: Basic metrics recording
    println!("Scenario 1: Basic Metrics Recording");
    println!("------------------------------------");
    basic_metrics_recording()?;
    println!();

    // Scenario 2: Statistical analysis
    println!("Scenario 2: Statistical Analysis");
    println!("--------------------------------");
    statistical_analysis()?;
    println!();

    // Scenario 3: Percentile calculations
    println!("Scenario 3: Percentile Calculations");
    println!("------------------------------------");
    percentile_calculations()?;
    println!();

    // Scenario 4: Trend analysis
    println!("Scenario 4: Trend Analysis");
    println!("--------------------------");
    trend_analysis()?;
    println!();

    // Scenario 5: Different time windows
    println!("Scenario 5: Different Time Windows");
    println!("-----------------------------------");
    time_windows()?;
    println!();

    // Scenario 6: Configuration presets
    println!("Scenario 6: Configuration Presets");
    println!("----------------------------------");
    configuration_presets()?;
    println!();

    // Scenario 7: Sampling and retention
    println!("Scenario 7: Sampling and Retention");
    println!("-----------------------------------");
    sampling_and_retention()?;
    println!();

    println!("=== Demo Complete ===");
    Ok(())
}

fn basic_metrics_recording() -> Result<(), Box<dyn std::error::Error>> {
    let config = AggregatorConfig::default();
    let aggregator = MetricsAggregator::new(config);

    // Record some bandwidth measurements
    aggregator.record_bandwidth(1024);
    aggregator.record_bandwidth(2048);
    aggregator.record_bandwidth(1536);

    // Record some latency measurements
    aggregator.record_latency(50);
    aggregator.record_latency(75);
    aggregator.record_latency(60);

    // Record events
    aggregator.record_connection_event();
    aggregator.record_connection_event();
    aggregator.record_query_event();

    // Get statistics
    let stats = aggregator.get_statistics(TimeWindow::Minute);

    println!("Bandwidth:");
    println!("  Count: {}", stats.bandwidth.count);
    println!("  Min: {} bytes", stats.bandwidth.min);
    println!("  Max: {} bytes", stats.bandwidth.max);
    println!("  Avg: {:.2} bytes", stats.bandwidth.avg);

    println!("\nLatency:");
    println!("  Count: {}", stats.latency.count);
    println!("  Min: {} ms", stats.latency.min);
    println!("  Max: {} ms", stats.latency.max);
    println!("  Avg: {:.2} ms", stats.latency.avg);

    println!("\nConnection Events: {}", stats.connection_rate.count);
    println!("Query Events: {}", stats.query_rate.count);

    Ok(())
}

fn statistical_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let config = AggregatorConfig::default();
    let aggregator = MetricsAggregator::new(config);

    // Record variable latencies
    let latencies = vec![10, 15, 20, 50, 100, 200, 150, 75, 30, 25];
    for latency in latencies {
        aggregator.record_latency(latency);
    }

    let stats = aggregator.get_statistics(TimeWindow::Minute);

    println!("Latency Statistics:");
    println!("  Count: {}", stats.latency.count);
    println!("  Min: {:.2} ms", stats.latency.min);
    println!("  Max: {:.2} ms", stats.latency.max);
    println!("  Average: {:.2} ms", stats.latency.avg);
    println!("  Std Dev: {:.2} ms", stats.latency.stddev);
    println!("  Median (P50): {:.2} ms", stats.latency.p50);
    println!("  P95: {:.2} ms", stats.latency.p95);
    println!("  P99: {:.2} ms", stats.latency.p99);

    Ok(())
}

fn percentile_calculations() -> Result<(), Box<dyn std::error::Error>> {
    let config = AggregatorConfig::default();
    let aggregator = MetricsAggregator::new(config);

    // Simulate 100 bandwidth measurements
    for i in 1..=100 {
        aggregator.record_bandwidth((i * 100) as u64);
    }

    let stats = aggregator.get_statistics(TimeWindow::Minute);

    println!("Bandwidth Percentiles (100 samples):");
    println!("  P50 (Median): {:.2} bytes", stats.bandwidth.p50);
    println!("  P95: {:.2} bytes", stats.bandwidth.p95);
    println!("  P99: {:.2} bytes", stats.bandwidth.p99);
    println!("  Max: {:.2} bytes", stats.bandwidth.max);

    // The P95 should be around the 95th value
    println!("\nInterpretation:");
    println!(
        "  95% of measurements were below {:.2} bytes",
        stats.bandwidth.p95
    );
    println!(
        "  99% of measurements were below {:.2} bytes",
        stats.bandwidth.p99
    );

    Ok(())
}

fn trend_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let config = AggregatorConfig::default();
    let aggregator = MetricsAggregator::new(config);

    println!("Increasing Latency Pattern:");
    // Simulate increasing latency
    for i in 1..=20 {
        aggregator.record_latency((i * 5) as u64);
        thread::sleep(Duration::from_millis(10));
    }

    let stats = aggregator.get_statistics(TimeWindow::Minute);
    println!(
        "  Trend: {:.2} (positive = increasing)",
        stats.latency.trend
    );
    println!("  Average: {:.2} ms", stats.latency.avg);

    // Clear and test decreasing pattern
    aggregator.clear();

    println!("\nDecreasing Latency Pattern:");
    for i in (1..=20).rev() {
        aggregator.record_latency((i * 5) as u64);
        thread::sleep(Duration::from_millis(10));
    }

    let stats = aggregator.get_statistics(TimeWindow::Minute);
    println!(
        "  Trend: {:.2} (negative = decreasing)",
        stats.latency.trend
    );
    println!("  Average: {:.2} ms", stats.latency.avg);

    // Clear and test stable pattern
    aggregator.clear();

    println!("\nStable Latency Pattern:");
    for _ in 1..=20 {
        aggregator.record_latency(50);
        thread::sleep(Duration::from_millis(10));
    }

    let stats = aggregator.get_statistics(TimeWindow::Minute);
    println!("  Trend: {:.2} (near zero = stable)", stats.latency.trend);
    println!("  Average: {:.2} ms", stats.latency.avg);

    Ok(())
}

fn time_windows() -> Result<(), Box<dyn std::error::Error>> {
    let config = AggregatorConfig::default();
    let aggregator = MetricsAggregator::new(config);

    // Record some metrics
    for _ in 0..10 {
        aggregator.record_bandwidth(1024);
        aggregator.record_latency(50);
    }

    println!("Statistics across different time windows:");

    let windows = vec![
        TimeWindow::Second,
        TimeWindow::Minute,
        TimeWindow::Hour,
        TimeWindow::Day,
    ];

    for window in windows {
        let stats = aggregator.get_statistics(window);
        println!("\n{:?} Window:", window);
        println!("  Bandwidth count: {}", stats.bandwidth.count);
        println!("  Latency count: {}", stats.latency.count);
        println!("  (All recent data falls within all windows in this example)");
    }

    Ok(())
}

fn configuration_presets() -> Result<(), Box<dyn std::error::Error>> {
    println!("Realtime Configuration:");
    let realtime = AggregatorConfig::realtime();
    println!("  Max data points: {}", realtime.max_data_points);
    println!("  Retention: {:?}", realtime.retention_period);
    println!("  Percentiles: {}", realtime.enable_percentiles);
    println!("  Trends: {}", realtime.enable_trends);

    println!("\nLong-term Configuration:");
    let longterm = AggregatorConfig::longterm();
    println!("  Max data points: {}", longterm.max_data_points);
    println!("  Retention: {:?}", longterm.retention_period);
    println!("  Percentiles: {}", longterm.enable_percentiles);
    println!("  Trends: {}", longterm.enable_trends);
    println!("  Sample rate: 1 in {}", longterm.sample_rate);

    println!("\nBalanced Configuration:");
    let balanced = AggregatorConfig::balanced();
    println!("  Max data points: {}", balanced.max_data_points);
    println!("  Retention: {:?}", balanced.retention_period);
    println!("  Percentiles: {}", balanced.enable_percentiles);
    println!("  Trends: {}", balanced.enable_trends);
    println!("  Sample rate: 1 in {}", balanced.sample_rate);

    Ok(())
}

fn sampling_and_retention() -> Result<(), Box<dyn std::error::Error>> {
    let config = AggregatorConfig {
        sample_rate: 3, // Sample 1 in 3
        max_data_points: 10,
        ..Default::default()
    };

    let max_points = config.max_data_points; // Save for later use
    let aggregator = MetricsAggregator::new(config);

    println!("Recording 15 bandwidth measurements (sample rate: 1 in 3):");
    for i in 1..=15 {
        aggregator.record_bandwidth((i * 100) as u64);
    }

    let stats = aggregator.get_statistics(TimeWindow::Minute);
    println!("  Recorded: 15");
    println!("  Sampled: {}", stats.bandwidth.count);
    println!("  Expected: ~5 (15 / 3)");

    println!("\nRecording 20 more measurements (max points: 10):");
    for i in 16..=35 {
        aggregator.record_bandwidth((i * 100) as u64);
    }

    println!("  Total data points: {}", aggregator.data_point_count());
    println!("  Max allowed per series: {}", max_points);
    println!("  (Oldest points are automatically removed)");

    Ok(())
}
