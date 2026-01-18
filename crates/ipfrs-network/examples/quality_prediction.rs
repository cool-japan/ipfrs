//! Example: Connection Quality Prediction
//!
//! This example demonstrates how to use the quality predictor to monitor
//! connection quality and make intelligent switching decisions.

use ipfrs_network::quality_predictor::{QualityPredictor, QualityPredictorConfig};
use libp2p::PeerId;
use std::thread;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== Connection Quality Prediction Example ===\n");

    // 1. Create quality predictor with default config
    println!("1. Creating Quality Predictor:");
    let config = QualityPredictorConfig::default();
    let predictor = QualityPredictor::new(config)?;
    println!("   ✓ Predictor created with default configuration\n");

    // 2. Simulate multiple peers with different quality profiles
    println!("2. Simulating Peer Connections:");

    let peer_excellent = PeerId::random();
    let peer_good = PeerId::random();
    let peer_poor = PeerId::random();

    // Excellent peer: low latency, high bandwidth, reliable
    println!("   Peer A (Excellent):");
    for i in 0..10 {
        predictor.record_latency(peer_excellent, 10 + i);
        predictor.record_bandwidth(peer_excellent, 10_000_000 + i * 100_000);
        predictor.record_success(peer_excellent);
        thread::sleep(Duration::from_millis(10));
    }
    println!("      - Latency: ~10-20ms");
    println!("      - Bandwidth: ~10 MB/s");
    println!("      - Reliability: 100%\n");

    // Good peer: moderate latency, good bandwidth
    println!("   Peer B (Good):");
    for i in 0..10 {
        predictor.record_latency(peer_good, 50 + i * 2);
        predictor.record_bandwidth(peer_good, 5_000_000 + i * 50_000);
        if i % 10 != 0 {
            predictor.record_success(peer_good);
        } else {
            predictor.record_failure(peer_good);
        }
        thread::sleep(Duration::from_millis(10));
    }
    println!("      - Latency: ~50-70ms");
    println!("      - Bandwidth: ~5 MB/s");
    println!("      - Reliability: 90%\n");

    // Poor peer: high latency, low bandwidth, unreliable
    println!("   Peer C (Poor):");
    for i in 0..10 {
        predictor.record_latency(peer_poor, 300 + i * 10);
        predictor.record_bandwidth(peer_poor, 500_000 + i * 10_000);
        if i % 2 == 0 {
            predictor.record_success(peer_poor);
        } else {
            predictor.record_failure(peer_poor);
        }
        thread::sleep(Duration::from_millis(10));
    }
    println!("      - Latency: ~300-400ms");
    println!("      - Bandwidth: ~500 KB/s");
    println!("      - Reliability: 50%\n");

    // 3. Predict quality for each peer
    println!("3. Quality Predictions:");
    let peers = vec![peer_excellent, peer_good, peer_poor];

    for (idx, peer) in peers.iter().enumerate() {
        if let Some(prediction) = predictor.predict_quality(peer) {
            let label = match idx {
                0 => "Excellent",
                1 => "Good",
                2 => "Poor",
                _ => "Unknown",
            };

            println!("   Peer {} ({}):", (b'A' + idx as u8) as char, label);
            println!("      Overall Score: {:.3}", prediction.overall_score);
            println!("      Latency Score: {:.3}", prediction.latency_score);
            println!("      Bandwidth Score: {:.3}", prediction.bandwidth_score);
            println!(
                "      Reliability Score: {:.3}",
                prediction.reliability_score
            );
            println!("      Uptime Score: {:.3}", prediction.uptime_score);
            println!(
                "      Avg Latency: {:.1}ms",
                prediction.avg_latency_ms.unwrap_or(0.0)
            );
            println!(
                "      Avg Bandwidth: {:.1} MB/s",
                prediction.avg_bandwidth_bps.unwrap_or(0.0) / 1_000_000.0
            );
            println!("      Acceptable: {}", prediction.is_acceptable);
            println!("      Recommend Switch: {}", prediction.should_switch);
            println!();
        }
    }

    // 4. Find the best peer
    println!("4. Finding Best Peer:");
    if let Some((best_peer, prediction)) = predictor.get_best_peer(&peers) {
        let label = if best_peer == peer_excellent {
            "A (Excellent)"
        } else if best_peer == peer_good {
            "B (Good)"
        } else {
            "C (Poor)"
        };

        println!("   ✓ Best peer: Peer {}", label);
        println!("   Overall quality score: {:.3}", prediction.overall_score);
        println!();
    }

    // 5. Rank all peers
    println!("5. Peer Ranking (Best to Worst):");
    let ranked = predictor.rank_peers(&peers);
    for (idx, (peer, prediction)) in ranked.iter().enumerate() {
        let label = if *peer == peer_excellent {
            "A (Excellent)"
        } else if *peer == peer_good {
            "B (Good)"
        } else {
            "C (Poor)"
        };

        println!(
            "   {}. Peer {} - Score: {:.3}",
            idx + 1,
            label,
            prediction.overall_score
        );
    }
    println!();

    // 6. Check for switch recommendations
    println!("6. Switch Recommendations:");
    for peer in &peers {
        if predictor.should_switch_connection(peer) {
            let label = if *peer == peer_excellent {
                "A (Excellent)"
            } else if *peer == peer_good {
                "B (Good)"
            } else {
                "C (Poor)"
            };
            println!("   ⚠ Recommend switching from Peer {}", label);
        }
    }
    println!();

    // 7. Demonstrate configuration presets
    println!("7. Configuration Presets:");

    // Low latency configuration
    println!("   Low Latency Config:");
    let ll_config = QualityPredictorConfig::low_latency();
    let ll_predictor = QualityPredictor::new(ll_config)?;
    ll_predictor.record_latency(peer_excellent, 10);
    ll_predictor.record_bandwidth(peer_excellent, 1_000_000);
    if let Some(pred) = ll_predictor.predict_quality(&peer_excellent) {
        println!(
            "      Overall Score: {:.3} (latency-focused)",
            pred.overall_score
        );
    }

    // High bandwidth configuration
    println!("   High Bandwidth Config:");
    let hb_config = QualityPredictorConfig::high_bandwidth();
    let hb_predictor = QualityPredictor::new(hb_config)?;
    hb_predictor.record_latency(peer_excellent, 10);
    hb_predictor.record_bandwidth(peer_excellent, 10_000_000);
    if let Some(pred) = hb_predictor.predict_quality(&peer_excellent) {
        println!(
            "      Overall Score: {:.3} (bandwidth-focused)",
            pred.overall_score
        );
    }

    // High reliability configuration
    println!("   High Reliability Config:");
    let hr_config = QualityPredictorConfig::high_reliability();
    let hr_predictor = QualityPredictor::new(hr_config)?;
    hr_predictor.record_success(peer_excellent);
    hr_predictor.record_success(peer_excellent);
    hr_predictor.record_success(peer_excellent);
    if let Some(pred) = hr_predictor.predict_quality(&peer_excellent) {
        println!(
            "      Overall Score: {:.3} (reliability-focused)",
            pred.overall_score
        );
    }
    println!();

    // 8. Display statistics
    println!("8. Predictor Statistics:");
    let stats = predictor.stats();
    println!("   Tracked Peers: {}", stats.tracked_peers);
    println!("   Predictions Made: {}", stats.predictions_made);
    println!(
        "   Switch Recommendations: {}",
        stats.switch_recommendations
    );
    println!("   Average Quality: {:.3}", stats.avg_quality);
    println!();

    // 9. Simulate quality degradation and recovery
    println!("9. Simulating Quality Degradation:");
    println!("   Recording degraded performance for Peer A...");

    for i in 0..5 {
        predictor.record_latency(peer_excellent, 500 + i * 50);
        predictor.record_bandwidth(peer_excellent, 100_000);
        predictor.record_failure(peer_excellent);
        thread::sleep(Duration::from_millis(10));
    }

    if let Some(prediction) = predictor.predict_quality(&peer_excellent) {
        println!("   New Overall Score: {:.3}", prediction.overall_score);
        println!("   Recommend Switch: {}", prediction.should_switch);
    }

    println!("\n   Simulating recovery...");
    for i in 0..10 {
        predictor.record_latency(peer_excellent, 15 + i);
        predictor.record_bandwidth(peer_excellent, 10_000_000);
        predictor.record_success(peer_excellent);
        thread::sleep(Duration::from_millis(10));
    }

    if let Some(prediction) = predictor.predict_quality(&peer_excellent) {
        println!("   Recovered Score: {:.3}", prediction.overall_score);
        println!("   Recommend Switch: {}", prediction.should_switch);
    }
    println!();

    println!("=== Quality Prediction Example Complete ===");

    Ok(())
}
