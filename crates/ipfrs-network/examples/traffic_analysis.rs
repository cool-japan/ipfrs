//! Traffic Analysis Example
//!
//! This example demonstrates how to use the traffic analyzer to analyze network
//! traffic patterns, detect anomalies, and profile peer behavior.

use ipfrs_network::{TrafficAnalyzer, TrafficAnalyzerConfig, TrendDirection};
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== IPFRS Network Traffic Analysis Demo ===\n");

    // Scenario 1: Basic Traffic Recording and Analysis
    println!("Scenario 1: Basic Traffic Recording and Analysis");
    println!("------------------------------------------------");
    run_basic_analysis()?;

    println!();

    // Scenario 2: Real-time Monitoring
    println!("Scenario 2: Real-time Monitoring");
    println!("--------------------------------");
    run_realtime_monitoring()?;

    println!();

    // Scenario 3: Peer Behavior Profiling
    println!("Scenario 3: Peer Behavior Profiling");
    println!("-----------------------------------");
    run_peer_profiling()?;

    println!();

    // Scenario 4: Anomaly Detection
    println!("Scenario 4: Anomaly Detection");
    println!("-----------------------------");
    run_anomaly_detection()?;

    println!();

    // Scenario 5: Traffic Pattern Detection
    println!("Scenario 5: Traffic Pattern Detection");
    println!("-------------------------------------");
    run_pattern_detection()?;

    println!();

    // Scenario 6: Trend Analysis
    println!("Scenario 6: Trend Analysis");
    println!("-------------------------");
    run_trend_analysis()?;

    println!();

    // Scenario 7: Long-term Analysis
    println!("Scenario 7: Long-term Analysis");
    println!("------------------------------");
    run_longterm_analysis()?;

    println!("\n=== Traffic Analysis Complete ===");

    Ok(())
}

/// Run basic traffic analysis
fn run_basic_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let config = TrafficAnalyzerConfig::default();
    let mut analyzer = TrafficAnalyzer::new(config);

    println!("Recording basic traffic events...");

    // Simulate some network activity
    analyzer.record_connection("peer1".to_string(), 1024);
    analyzer.record_connection("peer2".to_string(), 2048);
    analyzer.record_connection("peer3".to_string(), 512);

    analyzer.record_query("peer1".to_string(), Duration::from_millis(50), true);
    analyzer.record_query("peer2".to_string(), Duration::from_millis(75), true);
    analyzer.record_query("peer3".to_string(), Duration::from_millis(100), false);

    analyzer.record_bandwidth(5000, 3000);
    analyzer.record_bandwidth(6000, 4000);

    // Get statistics
    let stats = analyzer.get_stats();
    println!("\nAnalyzer Statistics:");
    println!("  Total events: {}", stats.total_events);
    println!("  Total peers: {}", stats.total_peers);
    println!("  Bandwidth samples: {}", stats.bandwidth_samples);
    println!("  Query samples: {}", stats.query_samples);
    println!("  Uptime: {:?}", stats.uptime);

    // Analyze traffic
    let analysis = analyzer.analyze()?;
    println!("\nTraffic Analysis:");
    println!("  Total bandwidth: {} bytes", analysis.total_bandwidth);
    println!("  Total connections: {}", analysis.total_connections);
    println!("  Total queries: {}", analysis.total_queries);
    println!("  Query success rate: {:.2}%", analysis.query_success_rate);
    println!("  Average latency: {:?}", analysis.average_latency);

    Ok(())
}

/// Run real-time monitoring
fn run_realtime_monitoring() -> Result<(), Box<dyn std::error::Error>> {
    let config = TrafficAnalyzerConfig::realtime();

    println!("Configuration:");
    println!("  Window size: {:?}", config.window_size);
    println!("  History size: {}", config.history_size);
    println!("  Anomaly threshold: {:.1}σ", config.anomaly_threshold);

    let mut analyzer = TrafficAnalyzer::new(config);

    println!("\nSimulating real-time traffic...");

    // Simulate rapid traffic events
    for i in 0..20 {
        let peer_id = format!("peer{}", i % 5);
        analyzer.record_connection(peer_id.clone(), (i + 1) * 100);

        let latency = Duration::from_millis(30 + (i * 5));
        let success = i % 10 != 0; // Occasional failure
        analyzer.record_query(peer_id, latency, success);

        analyzer.record_bandwidth(1000 * (i + 1), 800 * (i + 1));
    }

    let analysis = analyzer.analyze()?;
    println!("\nReal-time Analysis:");
    println!("  Peers tracked: {}", analysis.peer_profiles.len());
    println!("  Bandwidth trend: {:?}", analysis.bandwidth_trend);
    println!("  Connection trend: {:?}", analysis.connection_trend);
    println!("  Patterns detected: {}", analysis.patterns.len());
    println!("  Anomalies detected: {}", analysis.anomalies.len());

    Ok(())
}

/// Run peer behavior profiling
fn run_peer_profiling() -> Result<(), Box<dyn std::error::Error>> {
    let config = TrafficAnalyzerConfig {
        enable_peer_profiling: true,
        ..TrafficAnalyzerConfig::default()
    };
    let mut analyzer = TrafficAnalyzer::new(config);

    println!("Profiling peer behavior...");

    // Simulate different peer behaviors

    // Good peer: high success rate, low latency
    for _ in 0..10 {
        analyzer.record_connection("good_peer".to_string(), 1024);
        analyzer.record_query("good_peer".to_string(), Duration::from_millis(30), true);
    }

    // Average peer: moderate success rate, moderate latency
    for i in 0..10 {
        analyzer.record_connection("avg_peer".to_string(), 2048);
        analyzer.record_query(
            "avg_peer".to_string(),
            Duration::from_millis(60),
            i % 4 != 0,
        );
    }

    // Poor peer: low success rate, high latency
    for i in 0..10 {
        analyzer.record_connection("poor_peer".to_string(), 512);
        analyzer.record_query(
            "poor_peer".to_string(),
            Duration::from_millis(150),
            i % 3 == 0,
        );
    }

    let analysis = analyzer.analyze()?;
    println!("\nPeer Profiles:");

    for (peer_id, profile) in &analysis.peer_profiles {
        println!("\n  Peer: {}", peer_id);
        println!("    Connections: {}", profile.total_connections);
        println!(
            "    Queries: {}/{}",
            profile.successful_queries, profile.total_queries
        );
        println!(
            "    Success rate: {:.2}%",
            (profile.successful_queries as f64 / profile.total_queries as f64) * 100.0
        );
        println!("    Average latency: {:?}", profile.average_latency);
        println!("    Total bytes: {}", profile.total_bytes);
        println!("    Behavior score: {:.3}", profile.behavior_score);
    }

    Ok(())
}

/// Run anomaly detection
fn run_anomaly_detection() -> Result<(), Box<dyn std::error::Error>> {
    let config = TrafficAnalyzerConfig {
        anomaly_threshold: 2.5, // More sensitive
        min_samples: 5,
        ..TrafficAnalyzerConfig::default()
    };
    let mut analyzer = TrafficAnalyzer::new(config);

    println!("Detecting traffic anomalies...");

    // Normal traffic
    for _ in 0..10 {
        analyzer.record_bandwidth(5000, 5000);
    }

    // Bandwidth spike (anomaly)
    analyzer.record_bandwidth(50000, 50000);
    analyzer.record_bandwidth(60000, 60000);

    // Back to normal
    for _ in 0..5 {
        analyzer.record_bandwidth(5000, 5000);
    }

    // Simulate query failures
    let peer = "test_peer".to_string();
    analyzer.record_connection(peer.clone(), 1024);

    for i in 0..15 {
        let success = i < 5; // First 5 succeed, then failures
        analyzer.record_query(peer.clone(), Duration::from_millis(50), success);
    }

    let analysis = analyzer.analyze()?;
    println!("\nAnomaly Detection Results:");
    println!("  Anomalies detected: {}", analysis.anomalies.len());

    for (i, anomaly) in analysis.anomalies.iter().enumerate() {
        println!("\n  Anomaly {}: {:?}", i + 1, anomaly.anomaly_type);
        println!("    Description: {}", anomaly.description);
        println!("    Severity: {:.2}", anomaly.severity);
        if let Some(peer) = &anomaly.peer_id {
            println!("    Affected peer: {}", peer);
        }
    }

    Ok(())
}

/// Run pattern detection
fn run_pattern_detection() -> Result<(), Box<dyn std::error::Error>> {
    let config = TrafficAnalyzerConfig {
        window_size: Duration::from_secs(10),
        min_samples: 5,
        ..TrafficAnalyzerConfig::default()
    };
    let mut analyzer = TrafficAnalyzer::new(config);

    println!("Detecting traffic patterns...");

    // Simulate bursty traffic pattern
    for i in 0..20 {
        let bytes = if i % 5 == 0 {
            10000 // Burst
        } else {
            1000 // Normal
        };
        analyzer.record_bandwidth(bytes, bytes);
    }

    let analysis = analyzer.analyze()?;
    println!("\nPattern Detection Results:");
    println!("  Patterns detected: {}", analysis.patterns.len());

    for (i, pattern) in analysis.patterns.iter().enumerate() {
        println!("\n  Pattern {}: {:?}", i + 1, pattern.pattern_type);
        println!("    Description: {}", pattern.description);
        println!("    Confidence: {:.2}", pattern.confidence);
        println!("    Duration: {:?}", pattern.duration);
    }

    Ok(())
}

/// Run trend analysis
fn run_trend_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let config = TrafficAnalyzerConfig {
        min_samples: 10,
        ..TrafficAnalyzerConfig::default()
    };
    let mut analyzer = TrafficAnalyzer::new(config);

    println!("Analyzing traffic trends...");

    // Simulate increasing bandwidth trend
    for i in 0..15 {
        let bytes = 1000 + (i * 500);
        analyzer.record_bandwidth(bytes, bytes);
    }

    // Simulate increasing connection trend
    for i in 0..15 {
        let peer = format!("peer{}", i);
        analyzer.record_connection(peer, 1024);
    }

    let analysis = analyzer.analyze()?;
    println!("\nTrend Analysis:");
    println!("  Bandwidth trend: {:?}", analysis.bandwidth_trend);
    println!("  Connection trend: {:?}", analysis.connection_trend);

    match analysis.bandwidth_trend {
        TrendDirection::Increasing => println!("  → Bandwidth is increasing over time"),
        TrendDirection::Decreasing => println!("  → Bandwidth is decreasing over time"),
        TrendDirection::Steady => println!("  → Bandwidth is stable"),
        TrendDirection::Unknown => println!("  → Insufficient data for trend analysis"),
    }

    Ok(())
}

/// Run long-term analysis
fn run_longterm_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let config = TrafficAnalyzerConfig::long_term();

    println!("Configuration:");
    println!("  Window size: {:?} (1 hour)", config.window_size);
    println!("  History size: {} windows (24 hours)", config.history_size);

    let mut analyzer = TrafficAnalyzer::new(config);

    println!("\nSimulating long-term traffic collection...");

    // Simulate a day's worth of traffic patterns
    for hour in 0..24 {
        for _ in 0..10 {
            let peer = format!("peer{}", hour % 5);

            // Peak hours (9-17) have more traffic
            let multiplier = if (9..=17).contains(&hour) { 3 } else { 1 };

            analyzer.record_connection(peer.clone(), 1024 * multiplier);
            analyzer.record_query(peer, Duration::from_millis(50), true);
            analyzer.record_bandwidth(5000 * multiplier, 3000 * multiplier);
        }
    }

    let analysis = analyzer.analyze()?;
    println!("\nLong-term Analysis Summary:");
    println!("  Total bandwidth: {} bytes", analysis.total_bandwidth);
    println!("  Total connections: {}", analysis.total_connections);
    println!("  Total queries: {}", analysis.total_queries);
    println!("  Success rate: {:.2}%", analysis.query_success_rate);
    println!("  Unique peers: {}", analysis.peer_profiles.len());
    println!("  Bandwidth trend: {:?}", analysis.bandwidth_trend);
    println!("  Patterns detected: {}", analysis.patterns.len());
    println!("  Anomalies detected: {}", analysis.anomalies.len());

    // Clear analyzer to start fresh
    println!("\nClearing analyzer data...");
    analyzer.clear();
    let stats = analyzer.get_stats();
    println!("  Events after clear: {}", stats.total_events);
    println!("  Peers after clear: {}", stats.total_peers);

    Ok(())
}
