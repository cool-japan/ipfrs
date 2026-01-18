//! Connection Rate Limiting Example
//!
//! This example demonstrates how to use the ConnectionRateLimiter to prevent
//! connection storms and implement sophisticated rate limiting.

use ipfrs_network::rate_limiter::{ConnectionPriority, ConnectionRateLimiter, RateLimiterConfig};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== IPFRS Network Connection Rate Limiting Example ===\n");

    // Scenario 1: Basic Rate Limiting
    println!("--- Scenario 1: Basic Rate Limiting ---");
    {
        let config = RateLimiterConfig {
            max_rate: 10.0, // 10 connections per second
            burst_size: 5,  // Allow burst of 5
            ..Default::default()
        };
        let limiter = ConnectionRateLimiter::new(config);

        println!("Configuration: 10 conn/sec, burst size 5");

        // Try rapid connections
        let mut allowed_count = 0;
        let mut denied_count = 0;

        for i in 1..=10 {
            let peer_id = format!("peer{}", i);
            if limiter.allow_connection(&peer_id).await {
                allowed_count += 1;
                println!("  Connection {}: ALLOWED", i);
            } else {
                denied_count += 1;
                println!("  Connection {}: RATE LIMITED", i);
            }
        }

        println!(
            "\nResults: {} allowed, {} rate limited",
            allowed_count, denied_count
        );

        let stats = limiter.stats();
        println!("Stats:");
        println!("  Total attempts: {}", stats.total_attempts);
        println!("  Allowed: {}", stats.allowed);
        println!("  Rate limited: {}", stats.rate_limited);
        println!("  Tokens available: {:.2}\n", stats.tokens_available);
    }

    // Scenario 2: Per-Peer Rate Limiting
    println!("--- Scenario 2: Per-Peer Rate Limiting ---");
    {
        let config = RateLimiterConfig {
            max_rate: 100.0, // High global limit
            burst_size: 100,
            enable_per_peer_limits: true,
            max_per_peer_rate: 2.0, // But only 2 conn/sec per peer
            ..Default::default()
        };
        let limiter = ConnectionRateLimiter::new(config);

        println!("Configuration: Per-peer limit of 2 conn/sec");

        // Try multiple connections from same peer
        let peer_id = "busy_peer";
        for i in 1..=5 {
            if limiter.allow_connection(peer_id).await {
                println!("  Attempt {}: ALLOWED", i);
            } else {
                println!("  Attempt {}: RATE LIMITED (per-peer limit)", i);
            }
            sleep(Duration::from_millis(100)).await;
        }

        // Check peer stats
        if let Some((successes, failures, rate)) = limiter.peer_stats(peer_id) {
            println!("\nPeer stats for '{}':", peer_id);
            println!("  Successes: {}", successes);
            println!("  Failures: {}", failures);
            println!("  Current rate: {:.2} conn/sec\n", rate);
        }
    }

    // Scenario 3: Priority-Based Rate Limiting
    println!("--- Scenario 3: Priority-Based Rate Limiting ---");
    {
        let config = RateLimiterConfig {
            max_rate: 5.0,
            burst_size: 3,
            ..Default::default()
        };
        let limiter = ConnectionRateLimiter::new(config);

        println!("Configuration: 5 conn/sec with priority support");
        println!("Priority multipliers:");
        println!(
            "  Critical: {}x",
            ConnectionPriority::Critical.rate_multiplier()
        );
        println!("  High: {}x", ConnectionPriority::High.rate_multiplier());
        println!(
            "  Normal: {}x",
            ConnectionPriority::Normal.rate_multiplier()
        );
        println!("  Low: {}x\n", ConnectionPriority::Low.rate_multiplier());

        // Critical connections should be allowed more
        let priorities = vec![
            ("critical_peer", ConnectionPriority::Critical),
            ("high_peer", ConnectionPriority::High),
            ("normal_peer", ConnectionPriority::Normal),
            ("low_peer", ConnectionPriority::Low),
        ];

        for (peer, priority) in priorities {
            if limiter.allow_connection_with_priority(peer, priority).await {
                println!("  {:?} priority ({}): ALLOWED", priority, peer);
            } else {
                println!("  {:?} priority ({}): RATE LIMITED", priority, peer);
            }
        }

        let stats = limiter.stats();
        println!("\nTokens remaining: {:.2}\n", stats.tokens_available);
    }

    // Scenario 4: Connection Queuing
    println!("--- Scenario 4: Connection Queuing ---");
    {
        let config = RateLimiterConfig {
            max_rate: 2.0,
            burst_size: 2,
            enable_queuing: true,
            max_queue_size: 10,
            ..Default::default()
        };
        let limiter = ConnectionRateLimiter::new(config);

        println!("Configuration: Queuing enabled, max queue size 10");

        // Fill up capacity and queue
        for i in 1..=8 {
            let peer_id = format!("peer{}", i);
            limiter.allow_connection(&peer_id).await;
        }

        let stats = limiter.stats();
        println!("\nAfter burst:");
        println!("  Allowed: {}", stats.allowed);
        println!("  Queued: {}", stats.queued);
        println!("  Queue size: {}", stats.current_queue_size);

        // Wait and process queue
        println!("\nWaiting for token refill...");
        sleep(Duration::from_secs(1)).await;

        let processed = limiter.process_queue().await;
        println!("Processed {} connections from queue", processed.len());

        let stats = limiter.stats();
        println!(
            "Queue size after processing: {}\n",
            stats.current_queue_size
        );
    }

    // Scenario 5: Adaptive Rate Limiting
    println!("--- Scenario 5: Adaptive Rate Limiting ---");
    {
        let config = RateLimiterConfig::adaptive();
        let limiter = ConnectionRateLimiter::new(config);

        println!("Configuration: Adaptive rate limiting enabled");
        println!(
            "Initial rate: {:.1} conn/sec",
            limiter.stats().current_limit
        );

        // Simulate successful connections
        println!("\nSimulating successful connections...");
        for i in 1..=5 {
            if limiter.allow_connection(&format!("peer{}", i)).await {
                limiter.record_success(&format!("peer{}", i));
            }
            sleep(Duration::from_millis(100)).await;
        }

        println!(
            "Rate after successes: {:.1} conn/sec",
            limiter.stats().current_limit
        );

        // Simulate failures
        println!("\nSimulating failed connections...");
        for i in 1..=3 {
            limiter.record_failure(&format!("peer{}", i));
        }

        println!(
            "Rate after failures: {:.1} conn/sec\n",
            limiter.stats().current_limit
        );
    }

    // Scenario 6: Configuration Presets
    println!("--- Scenario 6: Configuration Presets ---");
    {
        println!("Conservative preset:");
        let conservative = RateLimiterConfig::conservative();
        println!("  Max rate: {} conn/sec", conservative.max_rate);
        println!("  Burst size: {}", conservative.burst_size);

        println!("\nPermissive preset:");
        let permissive = RateLimiterConfig::permissive();
        println!("  Max rate: {} conn/sec", permissive.max_rate);
        println!("  Burst size: {}", permissive.burst_size);

        println!("\nAdaptive preset:");
        let adaptive = RateLimiterConfig::adaptive();
        println!("  Adaptive enabled: {}", adaptive.enable_adaptive);
        println!(
            "  Adaptive factor: {:.1}%",
            adaptive.adaptive_factor * 100.0
        );
        println!("  Min rate: {} conn/sec", adaptive.min_rate);
        println!("  Max rate: {} conn/sec\n", adaptive.max_adaptive_rate);
    }

    // Scenario 7: Token Refill Demonstration
    println!("--- Scenario 7: Token Refill Demonstration ---");
    {
        let config = RateLimiterConfig {
            max_rate: 10.0,
            burst_size: 5,
            ..Default::default()
        };
        let limiter = ConnectionRateLimiter::new(config);

        println!("Configuration: 10 conn/sec, burst 5");

        // Use up all tokens
        println!("Using up all tokens...");
        for i in 1..=5 {
            limiter.allow_connection(&format!("peer{}", i)).await;
        }
        println!("Tokens available: {:.2}", limiter.stats().tokens_available);

        // Should be rate limited now
        if !limiter.allow_connection("peer_new").await {
            println!("Next connection: RATE LIMITED (as expected)");
        }

        // Wait for refill
        println!("\nWaiting 0.5 seconds for refill...");
        sleep(Duration::from_millis(500)).await;

        let stats = limiter.stats();
        println!(
            "Tokens available after refill: {:.2}",
            stats.tokens_available
        );

        // Should be allowed now
        if limiter.allow_connection("peer_new").await {
            println!("Connection after refill: ALLOWED ✓\n");
        }
    }

    // Scenario 8: Real-World Usage Pattern
    println!("--- Scenario 8: Real-World Usage Pattern ---");
    {
        let config = RateLimiterConfig {
            max_rate: 20.0,
            burst_size: 10,
            enable_per_peer_limits: true,
            max_per_peer_rate: 3.0,
            enable_queuing: true,
            max_queue_size: 50,
            ..Default::default()
        };
        let limiter = ConnectionRateLimiter::new(config);

        println!("Production configuration:");
        println!("  Global: 20 conn/sec with burst of 10");
        println!("  Per-peer: 3 conn/sec");
        println!("  Queuing: enabled (max 50)");

        // Simulate realistic connection pattern
        println!("\nSimulating realistic connection pattern...");

        for round in 1..=3 {
            println!("\nRound {}:", round);

            // Mix of priorities and peers
            let peers = vec![
                ("bootstrap1", ConnectionPriority::Critical),
                ("bootstrap2", ConnectionPriority::Critical),
                ("friend_peer", ConnectionPriority::High),
                ("known_peer1", ConnectionPriority::Normal),
                ("known_peer2", ConnectionPriority::Normal),
                ("random_peer1", ConnectionPriority::Low),
                ("random_peer2", ConnectionPriority::Low),
            ];

            for (peer, priority) in peers {
                if limiter.allow_connection_with_priority(peer, priority).await {
                    limiter.record_success(peer);
                }
            }

            sleep(Duration::from_millis(200)).await;
        }

        let stats = limiter.stats();
        println!("\nFinal statistics:");
        println!("  Total attempts: {}", stats.total_attempts);
        println!("  Allowed: {}", stats.allowed);
        println!("  Rate limited: {}", stats.rate_limited);
        println!("  Queued: {}", stats.queued);
        println!(
            "  Success rate: {:.1}%",
            (stats.allowed as f64 / stats.total_attempts as f64) * 100.0
        );
    }

    println!("\n=== Connection Rate Limiting Example Complete ===");
    Ok(())
}
