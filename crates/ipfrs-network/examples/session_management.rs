//! Session Management Example
//!
//! This example demonstrates how to use the session management module
//! to track connection lifecycles, monitor activity, and gather statistics.

use ipfrs_network::session::{SessionConfig, SessionManager, SessionMetadata, SessionState};
use ipfrs_network::utils::format_bytes;
use libp2p::PeerId;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Session Management Example");
    println!("==========================\n");

    // 1. Create session manager with custom configuration
    println!("1. Creating Session Manager");
    println!("----------------------------");

    let config = SessionConfig {
        idle_timeout: Duration::from_secs(300), // 5 minutes
        max_sessions: 1000,
        auto_cleanup: true,
        cleanup_interval: Duration::from_secs(60),
    };

    let manager = SessionManager::new(config);
    println!("Session manager created");
    println!("  Max sessions: {}", 1000);
    println!("  Idle timeout: 5 minutes");
    println!();

    // 2. Create multiple sessions
    println!("2. Creating Sessions");
    println!("--------------------");

    let peer1 = PeerId::random();
    let peer2 = PeerId::random();
    let peer3 = PeerId::random();

    manager.create_session(peer1);
    manager.create_session(peer2);
    manager.create_session(peer3);

    println!("Created 3 sessions");
    println!("  Active sessions: {}", manager.session_count());
    println!();

    // 3. Activate sessions
    println!("3. Activating Sessions");
    println!("----------------------");

    manager.activate_session(&peer1);
    manager.activate_session(&peer2);
    manager.activate_session(&peer3);

    for peer_id in &[peer1, peer2, peer3] {
        if let Some(session) = manager.get_session(peer_id) {
            println!(
                "  {} - State: {:?}",
                ipfrs_network::utils::truncate_peer_id(peer_id, 8),
                session.state
            );
        }
    }
    println!();

    // 4. Update session metadata
    println!("4. Updating Session Metadata");
    println!("-----------------------------");

    let metadata1 = SessionMetadata {
        endpoint: Some("/ip4/192.168.1.100/tcp/4001".to_string()),
        protocol: Some("quic".to_string()),
        tags: vec!["bootstrap".to_string(), "high-priority".to_string()],
        quality_score: Some(0.95),
    };

    manager.update_metadata(&peer1, metadata1);

    if let Some(session) = manager.get_session(&peer1) {
        println!("  Peer 1 metadata:");
        if let Some(endpoint) = &session.metadata.endpoint {
            println!("    Endpoint: {}", endpoint);
        }
        if let Some(protocol) = &session.metadata.protocol {
            println!("    Protocol: {}", protocol);
        }
        if let Some(score) = session.metadata.quality_score {
            println!("    Quality: {:.2}", score);
        }
        println!("    Tags: {:?}", session.metadata.tags);
    }
    println!();

    // 5. Simulate bandwidth usage
    println!("5. Recording Bandwidth Usage");
    println!("----------------------------");

    manager.update_bandwidth(&peer1, 1024 * 1024, 5 * 1024 * 1024); // 1 MB sent, 5 MB received
    manager.update_bandwidth(&peer2, 512 * 1024, 2 * 1024 * 1024); // 512 KB sent, 2 MB received
    manager.update_bandwidth(&peer3, 256 * 1024, 1024 * 1024); // 256 KB sent, 1 MB received

    println!("Bandwidth statistics:");
    for (i, peer_id) in [peer1, peer2, peer3].iter().enumerate() {
        if let Some(session) = manager.get_session(peer_id) {
            println!("  Peer {}:", i + 1);
            println!("    Sent: {}", format_bytes(session.bytes_sent as usize));
            println!(
                "    Received: {}",
                format_bytes(session.bytes_received as usize)
            );
        }
    }
    println!();

    // 6. Record message activity
    println!("6. Recording Message Activity");
    println!("------------------------------");

    for _ in 0..10 {
        manager.record_message(&peer1, true); // sent
        manager.record_message(&peer1, false); // received
    }

    for _ in 0..5 {
        manager.record_message(&peer2, true);
        manager.record_message(&peer2, false);
    }

    println!("Message statistics:");
    for (i, peer_id) in [peer1, peer2].iter().enumerate() {
        if let Some(session) = manager.get_session(peer_id) {
            println!("  Peer {}:", i + 1);
            println!("    Messages sent: {}", session.messages_sent);
            println!("    Messages received: {}", session.messages_received);
        }
    }
    println!();

    // 7. Check session durations
    println!("7. Session Durations");
    println!("--------------------");

    for (i, peer_id) in [peer1, peer2, peer3].iter().enumerate() {
        if let Some(session) = manager.get_session(peer_id) {
            println!("  Peer {}: {:?}", i + 1, session.duration());
        }
    }
    println!();

    // 8. Filter sessions by state
    println!("8. Filtering Sessions by State");
    println!("-------------------------------");

    let active_sessions = manager.get_sessions_by_state(SessionState::Active);
    println!("Active sessions: {}", active_sessions.len());

    for session in &active_sessions {
        println!(
            "  {} - Duration: {:?}",
            ipfrs_network::utils::truncate_peer_id(&session.peer_id, 8),
            session.duration()
        );
    }
    println!();

    // 9. Get overall statistics
    println!("9. Overall Statistics");
    println!("---------------------");

    let stats = manager.stats();
    println!("Total sessions created: {}", stats.total_created);
    println!("Active sessions: {}", stats.active_sessions);
    println!("Idle sessions: {}", stats.idle_sessions);
    println!("Total sessions closed: {}", stats.total_closed);
    println!(
        "Total bytes sent: {}",
        format_bytes(stats.total_bytes_sent as usize)
    );
    println!(
        "Total bytes received: {}",
        format_bytes(stats.total_bytes_received as usize)
    );
    println!("Total messages sent: {}", stats.total_messages_sent);
    println!("Total messages received: {}", stats.total_messages_received);
    println!();

    // 10. Close sessions
    println!("10. Closing Sessions");
    println!("--------------------");

    manager.close_session(&peer1);
    manager.close_session(&peer2);

    println!("Closed 2 sessions");

    if let Some(session) = manager.get_session(&peer1) {
        println!("  Peer 1 state: {:?}", session.state);
        println!("  Session closed: {}", session.closed_at.is_some());
    }
    println!();

    // 11. Final statistics
    println!("11. Final Statistics");
    println!("--------------------");

    let final_stats = manager.stats();
    println!("Total sessions created: {}", final_stats.total_created);
    println!("Currently active: {}", final_stats.active_sessions);
    println!("Total closed: {}", final_stats.total_closed);
    println!(
        "Average session duration: {:?}",
        final_stats.avg_session_duration
    );
    println!();

    // 12. Remove closed sessions
    println!("12. Session Removal");
    println!("-------------------");

    let removed1 = manager.remove_session(&peer1);
    let removed2 = manager.remove_session(&peer2);

    println!(
        "Removed sessions: {}",
        if removed1.is_some() && removed2.is_some() {
            2
        } else {
            0
        }
    );
    println!("Remaining sessions: {}", manager.session_count());
    println!();

    println!("Session management demonstration complete!");

    Ok(())
}
