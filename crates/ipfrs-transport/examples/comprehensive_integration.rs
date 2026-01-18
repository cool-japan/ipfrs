//! Comprehensive Integration Example
//!
//! This example demonstrates multiple transport features working together:
//! - Session management with priority scheduling
//! - Circuit breaker pattern for fault tolerance
//! - Auto-tuning based on network conditions
//! - Backpressure control for flow management
//!
//! Run with: cargo run --example comprehensive_integration

use bytes::Bytes;
use ipfrs_core::Cid;
use ipfrs_transport::{
    AutoTuner, BackpressureConfig, BackpressureController, CircuitBreaker, CircuitBreakerConfig,
    Priority, Session, SessionConfig, TuningProfile,
};
use multihash::Multihash;
use std::time::{Duration, Instant};

/// Create a test CID for demonstration
fn create_cid(seed: u64) -> Cid {
    let data = seed.to_le_bytes();
    let hash = Multihash::wrap(0x12, &data).unwrap();
    Cid::new_v1(0x55, hash)
}

/// Simulated data block
fn create_block_data(size: usize) -> Bytes {
    Bytes::from(vec![0u8; size])
}

fn main() {
    println!("=== Comprehensive Integration Example ===\n");
    println!("Demonstrating multiple transport features working together\n");

    // Scenario 1: Session-based Transfer with Prioritization
    println!("--- Scenario 1: Session-Based Transfer ---");
    demonstrate_session_transfer();

    // Scenario 2: Circuit Breaker Pattern
    println!("\n--- Scenario 2: Circuit Breaker ---");
    demonstrate_circuit_breaker();

    // Scenario 3: Auto-Tuning
    println!("\n--- Scenario 3: Auto-Tuning ---");
    demonstrate_auto_tuning();

    // Scenario 4: Backpressure Control
    println!("\n--- Scenario 4: Backpressure Control ---");
    demonstrate_backpressure();

    println!("\n=== Example Completed Successfully ===");
    println!("\nThis example demonstrated:");
    println!("  • Session-based coordinated block transfers");
    println!("  • Progress tracking and statistics");
    println!("  • Circuit breaker pattern for fault tolerance");
    println!("  • Auto-tuning profiles for network conditions");
    println!("  • Backpressure control for flow management");
}

fn demonstrate_session_transfer() {
    // Create session for coordinated transfer
    let session_config = SessionConfig {
        timeout: Duration::from_secs(300),
        default_priority: Priority::High,
        max_concurrent_blocks: 50,
        progress_notifications: true,
    };

    let session = Session::new(1, session_config, None);
    println!("Session created (ID: {})", session.id());

    // Simulate transferring blocks with different priorities
    let num_blocks = 100;
    let start = Instant::now();

    for i in 0..num_blocks {
        let cid = create_cid(i);
        let data = create_block_data(1024);

        // Simulate receiving blocks
        let _ = session.mark_received(&cid, &data);

        // Progress update every 25 blocks
        if (i + 1) % 25 == 0 {
            let stats = session.stats();
            println!(
                "  Progress: {:.1}% ({}/{})",
                stats.progress(),
                stats.blocks_received,
                num_blocks
            );
        }
    }

    let duration = start.elapsed();
    let stats = session.stats();

    println!("✓ Session transfer completed in {:?}", duration);
    println!("  Blocks received: {}", stats.blocks_received);
    println!("  Bytes transferred: {}", stats.bytes_transferred);
    println!("  Success rate: {:.1}%", stats.progress());
}

fn demonstrate_circuit_breaker() {
    println!("Initializing circuit breaker...");

    let config = CircuitBreakerConfig {
        failure_threshold: 5,
        timeout: Duration::from_secs(30),
        success_threshold: 2,
        window_duration: Duration::from_secs(60),
    };

    let circuit_breaker = CircuitBreaker::new(config);

    // Simulate failures
    println!("  Simulating failures...");
    for i in 1..=3 {
        circuit_breaker.record_failure();
        println!("    Failure {}: State = {:?}", i, circuit_breaker.state());
    }

    // Simulate successes
    println!("  Simulating successes...");
    for i in 1..=3 {
        circuit_breaker.record_success();
        println!("    Success {}: State = {:?}", i, circuit_breaker.state());
    }

    // Check if circuit allows requests
    let is_allowed = circuit_breaker.is_request_allowed();
    println!("✓ Circuit breaker operational");
    println!("  Request allowed: {}", is_allowed);
    println!("  Current state: {:?}", circuit_breaker.state());
}

fn demonstrate_auto_tuning() {
    let _auto_tuner = AutoTuner::new();
    println!("Auto-tuner initialized");

    // Demonstrate tuning profiles for different network conditions
    let profiles = vec![
        TuningProfile::excellent(),
        TuningProfile::good(),
        TuningProfile::fair(),
        TuningProfile::poor(),
        TuningProfile::very_poor(),
    ];

    for profile in profiles {
        println!("\nNetwork condition: {:?}", profile.condition);
        println!("  Profile: {}", profile.name);
        println!(
            "    Max concurrent blocks: {}",
            profile.max_concurrent_blocks
        );
        println!("    Want timeout: {:?}", profile.want_timeout);
        println!("    Max retries: {}", profile.max_retries);
        println!("    Batch size: {}", profile.batch_size);
    }

    println!("\n✓ Auto-tuning demonstration completed");
}

fn demonstrate_backpressure() {
    println!("Initializing backpressure controller...");

    let config = BackpressureConfig {
        max_pending: 100,
        high_watermark: 80,
        low_watermark: 40,
        max_buffer_bytes: 1024 * 1024, // 1 MB
    };

    let mut backpressure = BackpressureController::new(config);

    // Simulate incoming requests
    println!("Simulating incoming data chunks...");
    let mut accepted_count = 0;
    let mut rejected_count = 0;

    for i in 1..=150 {
        if backpressure.should_accept() {
            backpressure.on_send(1024); // 1 KB per chunk
            accepted_count += 1;
            if i % 50 == 0 {
                println!(
                    "  Chunk {}: Accepted (pending: {})",
                    i,
                    backpressure.pending_count()
                );
            }
        } else {
            rejected_count += 1;
            if i % 50 == 0 {
                println!(
                    "  Chunk {}: Rejected (pending: {}, paused: {})",
                    i,
                    backpressure.pending_count(),
                    backpressure.is_paused()
                );
            }
        }

        // Simulate some acknowledgements
        if i % 3 == 0 {
            backpressure.on_ack(1024);
        }
    }

    println!("\n✓ Backpressure control demonstration completed");
    println!("  Total accepted: {}", accepted_count);
    println!("  Total rejected: {}", rejected_count);
    println!("  Final pending: {}", backpressure.pending_count());
}
