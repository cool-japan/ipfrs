//! Network Policy Example
//!
//! This example demonstrates the policy engine for fine-grained control
//! over network operations including connection policies, bandwidth policies,
//! and content policies.
//!
//! Run with: `cargo run --example network_policy`

use ipfrs_network::{
    BandwidthPolicy, ConnectionPolicy, ContentPolicy, PolicyAction, PolicyConfig, PolicyEngine,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    println!("=== Network Policy Engine Demo ===\n");

    // Scenario 1: Basic connection policies
    println!("Scenario 1: Connection Policies");
    println!("----------------------------------");

    let engine = PolicyEngine::new(PolicyConfig::default());

    // Add a whitelist policy
    let whitelist = ConnectionPolicy::new("trusted-peers")
        .with_action(PolicyAction::Allow)
        .with_priority(100)
        .with_whitelist_peer("peer1")
        .with_whitelist_peer("peer2")
        .with_whitelist_peer("peer3");

    engine.add_connection_policy(whitelist)?;

    println!("Added whitelist policy for trusted peers");

    // Test connections
    let peers_to_test = vec!["peer1", "peer2", "peer4", "peer5"];

    for peer in peers_to_test {
        let allowed = engine.evaluate_connection(peer).await?;
        println!(
            "  Connection from {}: {}",
            peer,
            if allowed { "✓ ALLOWED" } else { "✗ DENIED" }
        );
    }

    // Scenario 2: Blacklist policies
    println!("\n\nScenario 2: Blacklist Policies");
    println!("--------------------------------");

    let engine2 = PolicyEngine::new(PolicyConfig::default());

    let blacklist = ConnectionPolicy::new("block-malicious")
        .with_action(PolicyAction::Deny)
        .with_priority(200)
        .with_blacklist_peer("bad_peer1")
        .with_blacklist_peer("bad_peer2");

    engine2.add_connection_policy(blacklist)?;

    println!("Added blacklist policy for malicious peers");

    let test_peers = vec!["good_peer", "bad_peer1", "bad_peer2", "another_peer"];

    for peer in test_peers {
        let allowed = engine2.evaluate_connection(peer).await?;
        println!(
            "  Connection from {}: {}",
            peer,
            if allowed { "✓ ALLOWED" } else { "✗ DENIED" }
        );
    }

    // Scenario 3: Policy priority
    println!("\n\nScenario 3: Policy Priority");
    println!("-----------------------------");

    let engine3 = PolicyEngine::new(PolicyConfig::default());

    // Low priority - deny all
    let deny_all = ConnectionPolicy::new("deny-all")
        .with_action(PolicyAction::Deny)
        .with_priority(10);

    // High priority - allow specific peer
    let allow_special = ConnectionPolicy::new("allow-special")
        .with_action(PolicyAction::Allow)
        .with_priority(100)
        .with_whitelist_peer("special_peer");

    engine3.add_connection_policy(deny_all)?;
    engine3.add_connection_policy(allow_special)?;

    println!("Added deny-all (priority 10) and allow-special (priority 100) policies");

    let priority_test = vec!["special_peer", "regular_peer", "another_peer"];

    for peer in priority_test {
        let allowed = engine3.evaluate_connection(peer).await?;
        println!(
            "  Connection from {}: {}",
            peer,
            if allowed { "✓ ALLOWED" } else { "✗ DENIED" }
        );
    }

    // Scenario 4: Connection counting and limits
    println!("\n\nScenario 4: Connection Limits");
    println!("-------------------------------");

    let engine4 = PolicyEngine::new(PolicyConfig::default());

    let rate_limit = ConnectionPolicy::new("rate-limit")
        .with_action(PolicyAction::Allow)
        .with_max_connections(2); // Max 2 connections per peer

    engine4.add_connection_policy(rate_limit)?;

    println!("Added rate limit policy (max 2 connections per peer)");

    let peer = "limited_peer";

    // First connection
    engine4.record_connection(peer);
    println!("  Connection 1 from {}: ✓ ALLOWED", peer);

    // Second connection
    engine4.record_connection(peer);
    println!("  Connection 2 from {}: ✓ ALLOWED", peer);

    // Third connection should be denied
    let can_connect = engine4.can_connect(peer);
    println!(
        "  Connection 3 from {}: {}",
        peer,
        if can_connect {
            "✓ ALLOWED"
        } else {
            "✗ DENIED (limit reached)"
        }
    );

    // Disconnect one
    engine4.record_disconnection(peer);
    let can_connect = engine4.can_connect(peer);
    println!(
        "  After disconnect - Connection 3 from {}: {}",
        peer,
        if can_connect {
            "✓ ALLOWED"
        } else {
            "✗ DENIED"
        }
    );

    // Scenario 5: Bandwidth policies
    println!("\n\nScenario 5: Bandwidth Policies");
    println!("--------------------------------");

    let engine5 = PolicyEngine::new(PolicyConfig::default());

    let bandwidth = BandwidthPolicy::new("standard-limits")
        .with_max_upload(10_000_000) // 10 Mbps
        .with_max_download(50_000_000) // 50 Mbps
        .with_per_peer_limit(1_000_000); // 1 Mbps per peer

    engine5.add_bandwidth_policy(bandwidth)?;

    println!("Added bandwidth policy:");
    println!("  Max upload: 10 Mbps");
    println!("  Max download: 50 Mbps");
    println!("  Per-peer limit: 1 Mbps");

    let policies = engine5.bandwidth_policies();
    println!("\nActive bandwidth policies: {}", policies.len());

    // Scenario 6: Content policies
    println!("\n\nScenario 6: Content Policies");
    println!("------------------------------");

    let engine6 = PolicyEngine::new(PolicyConfig::default());

    let content = ContentPolicy::new("content-filter")
        .with_allowed_pattern("^Qm.*") // Only allow CIDs starting with Qm
        .with_max_size(100_000_000); // Max 100 MB

    engine6.add_content_policy(content)?;

    println!("Added content policy:");
    println!("  Allowed pattern: ^Qm.* (IPFS CIDs)");
    println!("  Max size: 100 MB");

    let policies = engine6.content_policies();
    println!("\nActive content policies: {}", policies.len());

    // Scenario 7: Strict vs Permissive modes
    println!("\n\nScenario 7: Policy Modes");
    println!("-------------------------");

    // Strict mode - deny by default
    let strict_config = PolicyConfig::strict();
    let strict_engine = PolicyEngine::new(strict_config);

    println!("Strict mode (deny by default):");
    let allowed = strict_engine.evaluate_connection("random_peer").await?;
    println!(
        "  Connection from random_peer: {}",
        if allowed { "✓ ALLOWED" } else { "✗ DENIED" }
    );

    // Permissive mode - allow by default
    let permissive_config = PolicyConfig::permissive();
    let permissive_engine = PolicyEngine::new(permissive_config);

    println!("\nPermissive mode (allow by default):");
    let allowed = permissive_engine.evaluate_connection("random_peer").await?;
    println!(
        "  Connection from random_peer: {}",
        if allowed { "✓ ALLOWED" } else { "✗ DENIED" }
    );

    // Scenario 8: Policy removal
    println!("\n\nScenario 8: Policy Management");
    println!("-------------------------------");

    let engine8 = PolicyEngine::new(PolicyConfig::default());

    let policy1 = ConnectionPolicy::new("policy1").with_action(PolicyAction::Allow);
    let policy2 = ConnectionPolicy::new("policy2").with_action(PolicyAction::Deny);
    let policy3 = ConnectionPolicy::new("policy3").with_action(PolicyAction::Log);

    engine8.add_connection_policy(policy1)?;
    engine8.add_connection_policy(policy2)?;
    engine8.add_connection_policy(policy3)?;

    println!("Added 3 connection policies");
    println!("  Active policies: {}", engine8.connection_policies().len());

    engine8.remove_connection_policy("policy2")?;
    println!("\nRemoved policy2");
    println!("  Active policies: {}", engine8.connection_policies().len());

    // List all policies
    println!("\nRemaining policies:");
    for policy in engine8.connection_policies() {
        println!("  - {} (priority: {})", policy.name, policy.priority);
    }

    // Scenario 9: Statistics
    println!("\n\nScenario 9: Policy Statistics");
    println!("-------------------------------");

    let engine9 = PolicyEngine::new(PolicyConfig::default());

    let allow_policy = ConnectionPolicy::new("allow-some")
        .with_action(PolicyAction::Allow)
        .with_whitelist_peer("peer1");

    let deny_policy = ConnectionPolicy::new("deny-others")
        .with_action(PolicyAction::Deny)
        .with_priority(1);

    engine9.add_connection_policy(allow_policy)?;
    engine9.add_connection_policy(deny_policy)?;

    // Perform various evaluations
    for i in 1..=10 {
        let peer = if i % 3 == 0 {
            "peer1"
        } else {
            &format!("peer{}", i)
        };
        engine9.evaluate_connection(peer).await?;
    }

    let stats = engine9.stats();
    println!("Policy evaluation statistics:");
    println!("  Total evaluations: {}", stats.evaluations);
    println!("  Allowed: {}", stats.allowed);
    println!("  Denied: {}", stats.denied);
    println!("  Rate limited: {}", stats.rate_limited);
    println!(
        "  Average evaluation time: {:.3} ms",
        stats.avg_eval_time_ms
    );

    println!("\nPolicy hit counts:");
    for (policy_name, count) in &stats.policy_hits {
        println!("  {}: {} hits", policy_name, count);
    }

    // Scenario 10: Complex policy setup
    println!("\n\nScenario 10: Complex Policy Setup");
    println!("-----------------------------------");

    let complex_engine = PolicyEngine::new(PolicyConfig::default());

    // Multiple layered policies
    let tier1 = ConnectionPolicy::new("tier1-partners")
        .with_action(PolicyAction::Allow)
        .with_priority(100)
        .with_whitelist_peer("partner1")
        .with_whitelist_peer("partner2")
        .with_max_connections(10);

    let tier2 = ConnectionPolicy::new("tier2-users")
        .with_action(PolicyAction::Allow)
        .with_priority(50)
        .with_max_connections(5);

    let security = ConnectionPolicy::new("security-block")
        .with_action(PolicyAction::Deny)
        .with_priority(200)
        .with_blacklist_peer("malicious1")
        .with_blacklist_peer("malicious2");

    complex_engine.add_connection_policy(security)?;
    complex_engine.add_connection_policy(tier1)?;
    complex_engine.add_connection_policy(tier2)?;

    println!("Created complex policy hierarchy:");
    println!("  1. Security block (priority 200)");
    println!("  2. Tier 1 partners (priority 100, max 10 connections)");
    println!("  3. Tier 2 users (priority 50, max 5 connections)");

    let test_complex = vec![
        ("malicious1", "Blocked peer"),
        ("partner1", "Tier 1 partner"),
        ("partner2", "Tier 1 partner"),
        ("user1", "Regular user"),
        ("user2", "Regular user"),
    ];

    println!("\nTesting complex policies:");
    for (peer, description) in test_complex {
        let allowed = complex_engine.evaluate_connection(peer).await?;
        println!(
            "  {} ({}): {}",
            peer,
            description,
            if allowed { "✓ ALLOWED" } else { "✗ DENIED" }
        );
    }

    // Final statistics
    println!("\n\nFinal Statistics:");
    println!("------------------");
    let final_stats = complex_engine.stats();
    println!("Total evaluations: {}", final_stats.evaluations);
    println!(
        "Success rate: {:.1}%",
        (final_stats.allowed as f64 / final_stats.evaluations as f64) * 100.0
    );

    println!("\n=== Demo Complete ===");
    Ok(())
}
