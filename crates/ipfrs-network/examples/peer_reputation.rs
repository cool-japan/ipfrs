//! Peer reputation system example
//!
//! This example demonstrates how to use the peer reputation system to track
//! and score peer behavior over time.

use ipfrs_network::reputation::{ReputationConfig, ReputationEvent, ReputationManager};
use libp2p::PeerId;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Peer Reputation System Demo ===\n");

    // Scenario 1: Basic reputation tracking
    println!("Scenario 1: Basic Reputation Tracking");
    println!("----------------------------------------");
    basic_reputation_tracking()?;
    println!();

    // Scenario 2: Identifying trusted and bad peers
    println!("Scenario 2: Identifying Trusted and Bad Peers");
    println!("----------------------------------------------");
    identify_peer_categories()?;
    println!();

    // Scenario 3: Different configuration presets
    println!("Scenario 3: Configuration Presets");
    println!("----------------------------------");
    configuration_presets()?;
    println!();

    // Scenario 4: Latency-based scoring
    println!("Scenario 4: Latency-Based Scoring");
    println!("----------------------------------");
    latency_scoring()?;
    println!();

    // Scenario 5: Protocol violation tracking
    println!("Scenario 5: Protocol Violation Tracking");
    println!("----------------------------------------");
    protocol_violations()?;
    println!();

    // Scenario 6: Uptime tracking
    println!("Scenario 6: Uptime Tracking");
    println!("----------------------------");
    uptime_tracking()?;
    println!();

    // Scenario 7: Using reputation events
    println!("Scenario 7: Reputation Events");
    println!("------------------------------");
    reputation_events()?;
    println!();

    println!("=== Demo Complete ===");
    Ok(())
}

fn basic_reputation_tracking() -> Result<(), Box<dyn std::error::Error>> {
    let config = ReputationConfig::default();
    let mut manager = ReputationManager::new(config.clone());

    let peer1 = PeerId::random();
    let peer2 = PeerId::random();

    // Record successful transfers for peer1
    for i in 0..5 {
        manager.record_successful_transfer(&peer1, 1024 * (i + 1));
    }

    // Record mixed results for peer2
    for i in 0..3 {
        manager.record_successful_transfer(&peer2, 512 * (i + 1));
    }
    for _ in 0..2 {
        manager.record_failed_transfer(&peer2);
    }

    // Display reputations
    if let Some(score) = manager.get_reputation(&peer1) {
        println!("Peer 1 (Good):");
        println!("  Overall Score: {:.3}", score.overall_score(&config));
        println!("  Successful Transfers: {}", score.successful_transfers);
        println!("  Failed Transfers: {}", score.failed_transfers);
        println!(
            "  Transfer Success Rate: {:.3}",
            score.transfer_success_rate
        );
    }

    if let Some(score) = manager.get_reputation(&peer2) {
        println!("\nPeer 2 (Mixed):");
        println!("  Overall Score: {:.3}", score.overall_score(&config));
        println!("  Successful Transfers: {}", score.successful_transfers);
        println!("  Failed Transfers: {}", score.failed_transfers);
        println!(
            "  Transfer Success Rate: {:.3}",
            score.transfer_success_rate
        );
    }

    let stats = manager.stats();
    println!("\nStatistics:");
    println!("  Total Successful Events: {}", stats.successful_events);
    println!("  Total Failed Events: {}", stats.failed_events);

    Ok(())
}

fn identify_peer_categories() -> Result<(), Box<dyn std::error::Error>> {
    let config = ReputationConfig::default();
    let mut manager = ReputationManager::new(config.clone());

    // Create a trusted peer (many successes)
    let trusted_peer = PeerId::random();
    for _ in 0..10 {
        manager.record_successful_transfer(&trusted_peer, 2048);
        manager.record_low_latency(&trusted_peer, 30);
    }

    // Create a bad peer (many failures)
    let bad_peer = PeerId::random();
    for _ in 0..15 {
        manager.record_failed_transfer(&bad_peer);
    }

    // Create a neutral peer
    let neutral_peer = PeerId::random();
    for _ in 0..3 {
        manager.record_successful_transfer(&neutral_peer, 1024);
        manager.record_failed_transfer(&neutral_peer);
    }

    println!("Peer Categories:");
    println!("  Trusted Peers: {}", manager.get_trusted_peers().len());
    println!("  Bad Peers: {}", manager.get_bad_peers().len());
    println!("  Total Tracked: {}", manager.tracked_peer_count());

    if let Some(score) = manager.get_reputation(&trusted_peer) {
        println!("\nTrusted Peer:");
        println!("  Overall Score: {:.3}", score.overall_score(&config));
        println!("  Is Trusted: {}", manager.is_trusted(&trusted_peer));
    }

    if let Some(score) = manager.get_reputation(&bad_peer) {
        println!("\nBad Peer:");
        println!("  Overall Score: {:.3}", score.overall_score(&config));
        println!("  Is Bad: {}", manager.is_bad_peer(&bad_peer));
    }

    Ok(())
}

fn configuration_presets() -> Result<(), Box<dyn std::error::Error>> {
    let peer = PeerId::random();

    // Test with strict configuration
    let strict_config = ReputationConfig::strict();
    let mut strict_manager = ReputationManager::new(strict_config.clone());

    for _ in 0..5 {
        strict_manager.record_successful_transfer(&peer, 1024);
    }
    strict_manager.record_failed_transfer(&peer);

    if let Some(score) = strict_manager.get_reputation(&peer) {
        println!("Strict Configuration:");
        println!("  Trust Threshold: {:.2}", strict_config.trust_threshold);
        println!(
            "  Overall Score: {:.3}",
            score.overall_score(&strict_config)
        );
        println!("  Is Trusted: {}", strict_manager.is_trusted(&peer));
    }

    // Test with lenient configuration
    let lenient_config = ReputationConfig::lenient();
    let mut lenient_manager = ReputationManager::new(lenient_config.clone());

    for _ in 0..5 {
        lenient_manager.record_successful_transfer(&peer, 1024);
    }
    lenient_manager.record_failed_transfer(&peer);

    if let Some(score) = lenient_manager.get_reputation(&peer) {
        println!("\nLenient Configuration:");
        println!("  Trust Threshold: {:.2}", lenient_config.trust_threshold);
        println!(
            "  Overall Score: {:.3}",
            score.overall_score(&lenient_config)
        );
        println!("  Is Trusted: {}", lenient_manager.is_trusted(&peer));
    }

    // Test with performance-focused configuration
    let perf_config = ReputationConfig::performance_focused();
    let mut perf_manager = ReputationManager::new(perf_config.clone());

    for _ in 0..3 {
        perf_manager.record_successful_transfer(&peer, 1024);
        perf_manager.record_low_latency(&peer, 20); // Very low latency
    }

    if let Some(score) = perf_manager.get_reputation(&peer) {
        println!("\nPerformance-Focused Configuration:");
        println!("  Latency Weight: {:.2}", perf_config.latency_weight);
        println!("  Overall Score: {:.3}", score.overall_score(&perf_config));
        println!("  Latency Score: {:.3}", score.latency_score);
    }

    Ok(())
}

fn latency_scoring() -> Result<(), Box<dyn std::error::Error>> {
    let config = ReputationConfig::default();
    let mut manager = ReputationManager::new(config.clone());

    let low_latency_peer = PeerId::random();
    let high_latency_peer = PeerId::random();

    // Peer with consistently low latency
    for _ in 0..10 {
        manager.record_low_latency(&low_latency_peer, 25);
    }

    // Peer with consistently high latency
    for _ in 0..10 {
        manager.record_low_latency(&high_latency_peer, 800);
    }

    if let Some(score) = manager.get_reputation(&low_latency_peer) {
        println!("Low Latency Peer:");
        println!("  Average Latency: {} ms", score.average_latency_ms);
        println!("  Latency Score: {:.3}", score.latency_score);
        println!("  Overall Score: {:.3}", score.overall_score(&config));
    }

    if let Some(score) = manager.get_reputation(&high_latency_peer) {
        println!("\nHigh Latency Peer:");
        println!("  Average Latency: {} ms", score.average_latency_ms);
        println!("  Latency Score: {:.3}", score.latency_score);
        println!("  Overall Score: {:.3}", score.overall_score(&config));
    }

    Ok(())
}

fn protocol_violations() -> Result<(), Box<dyn std::error::Error>> {
    let config = ReputationConfig::default();
    let mut manager = ReputationManager::new(config.clone());

    let compliant_peer = PeerId::random();
    let violating_peer = PeerId::random();

    // Compliant peer
    for _ in 0..10 {
        manager.record_successful_transfer(&compliant_peer, 1024);
    }

    // Violating peer
    for _ in 0..5 {
        manager.record_successful_transfer(&violating_peer, 1024);
    }
    for _ in 0..3 {
        manager.record_protocol_violation(&violating_peer);
    }

    if let Some(score) = manager.get_reputation(&compliant_peer) {
        println!("Compliant Peer:");
        println!("  Protocol Violations: {}", score.protocol_violations);
        println!(
            "  Protocol Compliance Score: {:.3}",
            score.protocol_compliance_score
        );
        println!("  Overall Score: {:.3}", score.overall_score(&config));
    }

    if let Some(score) = manager.get_reputation(&violating_peer) {
        println!("\nViolating Peer:");
        println!("  Protocol Violations: {}", score.protocol_violations);
        println!(
            "  Protocol Compliance Score: {:.3}",
            score.protocol_compliance_score
        );
        println!("  Overall Score: {:.3}", score.overall_score(&config));
    }

    let stats = manager.stats();
    println!("\nTotal Protocol Violations: {}", stats.protocol_violations);

    Ok(())
}

fn uptime_tracking() -> Result<(), Box<dyn std::error::Error>> {
    let config = ReputationConfig::default();
    let mut manager = ReputationManager::new(config.clone());

    let reliable_peer = PeerId::random();
    let unreliable_peer = PeerId::random();

    // Reliable peer with good uptime
    manager.record_successful_transfer(&reliable_peer, 1024);
    std::thread::sleep(Duration::from_millis(100));
    manager.update_uptime(&reliable_peer, Duration::from_secs(7200)); // 2 hours

    // Unreliable peer with poor uptime
    manager.record_successful_transfer(&unreliable_peer, 1024);
    std::thread::sleep(Duration::from_millis(100));
    manager.update_uptime(&unreliable_peer, Duration::from_secs(300)); // 5 minutes

    if let Some(score) = manager.get_reputation(&reliable_peer) {
        println!("Reliable Peer:");
        println!("  Total Uptime: {:?}", score.total_uptime);
        println!("  Uptime Score: {:.3}", score.uptime_score);
        println!("  Overall Score: {:.3}", score.overall_score(&config));
    }

    if let Some(score) = manager.get_reputation(&unreliable_peer) {
        println!("\nUnreliable Peer:");
        println!("  Total Uptime: {:?}", score.total_uptime);
        println!("  Uptime Score: {:.3}", score.uptime_score);
        println!("  Overall Score: {:.3}", score.overall_score(&config));
    }

    Ok(())
}

fn reputation_events() -> Result<(), Box<dyn std::error::Error>> {
    let config = ReputationConfig::default();
    let mut manager = ReputationManager::new(config.clone());

    let peer = PeerId::random();

    // Record various events
    manager.record_event(&peer, ReputationEvent::SuccessfulTransfer);
    manager.record_event(&peer, ReputationEvent::LowLatency);
    manager.record_event(&peer, ReputationEvent::SuccessfulTransfer);
    manager.record_event(&peer, ReputationEvent::GracefulDisconnect);
    manager.record_event(&peer, ReputationEvent::SuccessfulTransfer);

    if let Some(score) = manager.get_reputation(&peer) {
        println!("Events Recorded:");
        println!("  SuccessfulTransfer (3x)");
        println!("  LowLatency (1x)");
        println!("  GracefulDisconnect (1x)");
        println!("\nPeer Reputation:");
        println!("  Overall Score: {:.3}", score.overall_score(&config));
        println!("  Successful Transfers: {}", score.successful_transfers);
        println!("  Is Trusted: {}", manager.is_trusted(&peer));
    }

    // Simulate some bad events
    let bad_peer = PeerId::random();
    manager.record_event(&bad_peer, ReputationEvent::FailedTransfer);
    manager.record_event(&bad_peer, ReputationEvent::HighLatency);
    manager.record_event(&bad_peer, ReputationEvent::ProtocolViolation);
    manager.record_event(&bad_peer, ReputationEvent::UnexpectedDisconnect);

    if let Some(score) = manager.get_reputation(&bad_peer) {
        println!("\nBad Events Recorded:");
        println!("  FailedTransfer (1x)");
        println!("  HighLatency (1x)");
        println!("  ProtocolViolation (1x)");
        println!("  UnexpectedDisconnect (1x)");
        println!("\nBad Peer Reputation:");
        println!("  Overall Score: {:.3}", score.overall_score(&config));
        println!("  Is Bad: {}", manager.is_bad_peer(&bad_peer));
    }

    Ok(())
}
