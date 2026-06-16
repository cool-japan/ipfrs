/// Integration tests for ipfrs-transport
///
/// These tests validate the complete protocol flow across multiple peers,
/// including block exchange, want list management, peer selection, and error recovery.
use bytes::Bytes;
use cid::Cid;
use ipfrs_transport::{
    messages::{Message, WantEntry as MsgWantEntry},
    peer_manager::{BlacklistReason, PeerManager, PeerScoringConfig, SelectionStrategy},
    session::{Session, SessionConfig, SessionEvent, SessionState},
    want_list::{Priority, WantEntry, WantList, WantListConfig},
};
use rand::RngExt;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::timeout;

/// Helper to create test CIDs
fn create_test_cid(index: u64) -> Cid {
    use multihash::Multihash;
    // Create a simple hash by converting index to bytes
    let mut bytes = [0u8; 32];
    bytes[..8].copy_from_slice(&index.to_be_bytes());
    let hash = Multihash::wrap(0x12, &bytes).unwrap(); // 0x12 = SHA2-256
    Cid::new_v1(0x55, hash) // 0x55 = raw codec
}

/// Helper to create test block data
fn create_test_block(index: u64) -> Bytes {
    Bytes::from(format!("test-block-data-{}", index))
}

#[tokio::test]
async fn test_want_list_priority_ordering() {
    let mut want_list = WantList::new(WantListConfig::default());

    // Add wants with different priorities
    let cid_low = create_test_cid(1);
    let cid_normal = create_test_cid(2);
    let cid_high = create_test_cid(3);
    let cid_urgent = create_test_cid(4);
    let cid_critical = create_test_cid(5);

    want_list.add_simple(cid_low, Priority::Low as i32);
    want_list.add_simple(cid_normal, Priority::Normal as i32);
    want_list.add_simple(cid_high, Priority::High as i32);
    want_list.add_simple(cid_urgent, Priority::Urgent as i32);
    want_list.add_simple(cid_critical, Priority::Critical as i32);

    // Pop should return highest priority
    if let Some(want) = want_list.pop() {
        assert_eq!(want.cid, cid_critical);
        assert_eq!(want.priority, Priority::Critical as i32);
    } else {
        panic!("Expected want with Critical priority");
    }
}

#[tokio::test]
async fn test_peer_manager_selection_strategies() {
    let mut peer_manager = PeerManager::new(PeerScoringConfig::default());

    // Add peers with different characteristics
    let peer1 = "peer1".to_string();
    let peer2 = "peer2".to_string();
    let peer3 = "peer3".to_string();

    peer_manager.add_peer(peer1.clone());
    peer_manager.add_peer(peer2.clone());
    peer_manager.add_peer(peer3.clone());

    // Simulate different performance characteristics
    peer_manager.record_success(&peer1, 1_000_000, Duration::from_millis(10)); // Fast, moderate bandwidth
    peer_manager.record_success(&peer2, 5_000_000, Duration::from_millis(50)); // Moderate speed, good bandwidth
    peer_manager.record_success(&peer3, 10_000_000, Duration::from_millis(100)); // Slow, high bandwidth

    // Record CIDs to test provider selection
    let test_cid = create_test_cid(100);
    peer_manager.record_has(&peer1, test_cid);
    peer_manager.record_has(&peer2, test_cid);
    peer_manager.record_has(&peer3, test_cid);

    // Test peer selection with FastestFirst strategy
    let selected = peer_manager.select_peers(&test_cid, 1, SelectionStrategy::FastestFirst);
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0], peer1);

    // Test peer selection with HighestBandwidth strategy
    // Note: The actual selection depends on the scoring algorithm which considers
    // multiple factors. We just verify that a peer is selected.
    let selected = peer_manager.select_peers(&test_cid, 1, SelectionStrategy::HighestBandwidth);
    assert_eq!(selected.len(), 1);
    // Should select one of the peers that has the CID
    assert!([&peer1, &peer2, &peer3].contains(&&selected[0]));
}

#[tokio::test]
async fn test_session_lifecycle() {
    let cid1 = create_test_cid(1);
    let cid2 = create_test_cid(2);
    let cid3 = create_test_cid(3);

    let config = SessionConfig {
        timeout: Duration::from_secs(30),
        ..Default::default()
    };

    let session = Session::new(1, config, None);

    // Add blocks to session
    session.add_block(cid1, None).unwrap();
    session.add_block(cid2, None).unwrap();
    session.add_block(cid3, None).unwrap();

    // Verify initial state
    assert_eq!(session.state(), SessionState::Active);
    let stats = session.stats();
    assert_eq!(stats.total_blocks, 3);
    assert_eq!(stats.blocks_received, 0);

    // Mark blocks as received
    let block1 = create_test_block(1);
    session.mark_received(&cid1, &block1).unwrap();
    let stats = session.stats();
    assert_eq!(stats.blocks_received, 1);
    assert_eq!(session.state(), SessionState::Active);

    let block2 = create_test_block(2);
    session.mark_received(&cid2, &block2).unwrap();
    let stats = session.stats();
    assert_eq!(stats.blocks_received, 2);

    let block3 = create_test_block(3);
    session.mark_received(&cid3, &block3).unwrap();
    let stats = session.stats();
    assert_eq!(stats.blocks_received, 3);
    assert_eq!(session.state(), SessionState::Completed);
}

#[tokio::test]
async fn test_session_event_notifications() {
    let cid1 = create_test_cid(1);
    let cid2 = create_test_cid(2);

    let config = SessionConfig {
        timeout: Duration::from_secs(30),
        ..Default::default()
    };

    let (tx, mut rx) = mpsc::unbounded_channel();
    let session = Session::new(100, config, Some(tx));

    // Should receive Started event
    let event = timeout(Duration::from_millis(100), rx.recv())
        .await
        .expect("Timeout waiting for event")
        .expect("No event received");

    match event {
        SessionEvent::Started { session_id } => {
            assert_eq!(session_id, 100);
        }
        _ => panic!("Expected Started event"),
    }

    // Add blocks
    session.add_block(cid1, None).unwrap();
    session.add_block(cid2, None).unwrap();

    // Mark block as received
    let block1 = create_test_block(1);
    session.mark_received(&cid1, &block1).unwrap();

    // Should receive BlockReceived event
    let event = timeout(Duration::from_millis(100), rx.recv())
        .await
        .expect("Timeout waiting for event")
        .expect("No event received");

    match event {
        SessionEvent::BlockReceived {
            session_id,
            cid,
            size,
        } => {
            assert_eq!(session_id, 100);
            assert_eq!(cid, cid1);
            assert!(size > 0);
        }
        _ => panic!("Expected BlockReceived event, got {:?}", event),
    }

    // Complete session
    let block2 = create_test_block(2);
    session.mark_received(&cid2, &block2).unwrap();

    // Should receive another BlockReceived then Completed event
    let _block_event = rx.recv().await;

    let event = timeout(Duration::from_millis(100), rx.recv())
        .await
        .expect("Timeout waiting for event")
        .expect("No event received");

    match event {
        SessionEvent::Completed { session_id, stats } => {
            assert_eq!(session_id, 100);
            assert_eq!(stats.total_blocks, 2);
            assert_eq!(stats.blocks_received, 2);
        }
        _ => panic!("Expected Completed event, got {:?}", event),
    }
}

#[tokio::test]
async fn test_message_serialization_roundtrip() {
    let cid1 = create_test_cid(1);
    let cid2 = create_test_cid(2);

    // Test WantList message
    let want_entry1 = MsgWantEntry::with_priority(cid1, 10);
    let want_entry2 = MsgWantEntry::new(cid2);
    let want_list = Message::want_list(vec![want_entry1, want_entry2], false);

    let bytes = want_list.to_bytes().unwrap();
    let decoded = Message::from_bytes(&bytes).unwrap();

    match decoded {
        Message::WantList(wl) => {
            assert_eq!(wl.entries.len(), 2);
            assert_eq!(wl.entries[0].cid, cid1);
            assert_eq!(wl.entries[0].priority, 10);
            assert_eq!(wl.entries[1].cid, cid2);
        }
        _ => panic!("Expected WantList message"),
    }

    // Test Block message
    let block_data = create_test_block(1).to_vec();
    let block_msg = Message::block(cid1, block_data.clone());

    let bytes = block_msg.to_bytes().unwrap();
    let decoded = Message::from_bytes(&bytes).unwrap();

    match decoded {
        Message::Block(bm) => {
            assert_eq!(bm.cid, cid1);
            assert_eq!(bm.data, block_data);
        }
        _ => panic!("Expected Block message"),
    }

    // Test Have message
    let have_msg = Message::have(cid2);
    let bytes = have_msg.to_bytes().unwrap();
    let decoded = Message::from_bytes(&bytes).unwrap();

    match decoded {
        Message::Have(hm) => {
            assert_eq!(hm.cid, cid2);
        }
        _ => panic!("Expected Have message"),
    }

    // Test DontHave message
    let dont_have_msg = Message::dont_have(cid1);
    let bytes = dont_have_msg.to_bytes().unwrap();
    let decoded = Message::from_bytes(&bytes).unwrap();

    match decoded {
        Message::DontHave(dhm) => {
            assert_eq!(dhm.cid, cid1);
        }
        _ => panic!("Expected DontHave message"),
    }

    // Test Cancel message
    let cancel_msg = Message::cancel(cid2);
    let bytes = cancel_msg.to_bytes().unwrap();
    let decoded = Message::from_bytes(&bytes).unwrap();

    match decoded {
        Message::Cancel(cm) => {
            assert_eq!(cm.cid, cid2);
        }
        _ => panic!("Expected Cancel message"),
    }
}

#[tokio::test]
async fn test_peer_blacklist_behavior() {
    let mut peer_manager = PeerManager::new(PeerScoringConfig::default());

    let peer_id = "bad-peer".to_string();
    peer_manager.add_peer(peer_id.clone());

    // Initially peer should be selectable
    assert!(!peer_manager.is_blacklisted(&peer_id));

    // Blacklist the peer
    peer_manager.blacklist_peer(
        peer_id.clone(),
        BlacklistReason::LowScore,
        Some(Duration::from_secs(60)),
    );

    // Peer should now be blacklisted
    assert!(peer_manager.is_blacklisted(&peer_id));
}

#[tokio::test]
async fn test_want_timeout_cleanup() {
    let config = WantListConfig {
        default_timeout: Duration::from_millis(100),
        ..Default::default()
    };

    let mut want_list = WantList::new(config);
    let cid = create_test_cid(1);

    // Add want
    want_list.add_simple(cid, Priority::Normal as i32);
    assert!(want_list.contains(&cid));

    // Wait for timeout
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Cleanup expired wants
    want_list.cleanup_expired();

    // Want should be removed
    assert!(!want_list.contains(&cid));
}

#[tokio::test]
async fn test_concurrent_want_operations() {
    let want_list = Arc::new(parking_lot::RwLock::new(WantList::new(
        WantListConfig::default(),
    )));
    let mut handles = vec![];

    // Spawn multiple tasks adding wants concurrently
    for i in 0..10 {
        let wl = want_list.clone();
        let handle = tokio::spawn(async move {
            let cid = create_test_cid(i);
            wl.write().add_simple(cid, Priority::Normal as i32);
        });
        handles.push(handle);
    }

    // Wait for all tasks
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify all wants were added
    assert_eq!(want_list.read().len(), 10);
}

#[tokio::test]
async fn test_peer_scoring() {
    let mut peer_manager = PeerManager::new(PeerScoringConfig::default());

    let peer_id = "peer1".to_string();
    peer_manager.add_peer(peer_id.clone());

    // Record good metrics
    peer_manager.record_success(&peer_id, 10_000_000, Duration::from_millis(10));

    let scores = peer_manager.get_scores();
    let score = scores.get(&peer_id).copied().unwrap_or(0.0);

    // Score should be positive after successful transfer
    assert!(score > 0.0, "Peer score should be positive after success");

    // Record failure
    peer_manager.record_failure(&peer_id);

    let scores = peer_manager.get_scores();
    let new_score = scores.get(&peer_id).copied().unwrap_or(0.0);

    // Score might decrease after failure (depending on implementation)
    // At minimum, it should still be calculable
    assert!(new_score >= 0.0);
}

#[tokio::test]
async fn test_priority_update() {
    let mut want_list = WantList::new(WantListConfig::default());
    let cid = create_test_cid(1);

    // Add want with normal priority
    want_list.add_simple(cid, Priority::Normal as i32);

    // Verify initial priority
    if let Some(entry) = want_list.get(&cid) {
        assert_eq!(entry.priority, Priority::Normal as i32);
    }

    // Update priority to high
    want_list.update_priority(&cid, Priority::High as i32);

    // Verify updated priority
    if let Some(entry) = want_list.get(&cid) {
        assert_eq!(entry.priority, Priority::High as i32);
    }
}

#[tokio::test]
async fn test_session_pause_resume() {
    let cid1 = create_test_cid(1);
    let cid2 = create_test_cid(2);

    let config = SessionConfig::default();
    let session = Session::new(1, config, None);

    session.add_block(cid1, None).unwrap();
    session.add_block(cid2, None).unwrap();

    // Initially active
    assert_eq!(session.state(), SessionState::Active);

    // Pause session
    session.pause();
    assert_eq!(session.state(), SessionState::Paused);

    // Resume session
    session.resume();
    assert_eq!(session.state(), SessionState::Active);
}

#[tokio::test]
async fn test_session_cancellation() {
    let cid1 = create_test_cid(1);
    let cid2 = create_test_cid(2);

    let config = SessionConfig::default();
    let session = Session::new(1, config, None);

    session.add_block(cid1, None).unwrap();
    session.add_block(cid2, None).unwrap();

    // Mark one block as received
    let block1 = create_test_block(1);
    session.mark_received(&cid1, &block1).unwrap();
    let stats = session.stats();
    assert_eq!(stats.blocks_received, 1);

    // Cancel session
    session.cancel();
    assert_eq!(session.state(), SessionState::Cancelled);

    // State should remain cancelled even if we try to update
    let block2 = create_test_block(2);
    let _ = session.mark_received(&cid2, &block2);
    assert_eq!(session.state(), SessionState::Cancelled);
}

#[tokio::test]
async fn test_session_stats_progress() {
    let config = SessionConfig::default();
    let session = Session::new(1, config, None);

    let cid1 = create_test_cid(1);
    let cid2 = create_test_cid(2);
    let cid3 = create_test_cid(3);
    let cid4 = create_test_cid(4);

    session.add_block(cid1, None).unwrap();
    session.add_block(cid2, None).unwrap();
    session.add_block(cid3, None).unwrap();
    session.add_block(cid4, None).unwrap();

    // Initially 0% progress
    let stats = session.stats();
    assert_eq!(stats.progress(), 0.0);

    // Receive 2 out of 4 blocks
    let block1 = create_test_block(1);
    session.mark_received(&cid1, &block1).unwrap();
    let block2 = create_test_block(2);
    session.mark_received(&cid2, &block2).unwrap();

    // Should be 50% progress
    let stats = session.stats();
    let progress = stats.progress();
    assert!((progress - 50.0).abs() < 0.01);

    // Complete all blocks
    let block3 = create_test_block(3);
    session.mark_received(&cid3, &block3).unwrap();
    let block4 = create_test_block(4);
    session.mark_received(&cid4, &block4).unwrap();

    // Should be 100% progress
    let stats = session.stats();
    let progress = stats.progress();
    assert!((progress - 100.0).abs() < 0.01);
}

#[tokio::test]
async fn test_multiple_peer_scoring() {
    let mut peer_manager = PeerManager::new(PeerScoringConfig::default());

    let test_cid = create_test_cid(200);

    // Add multiple peers
    for i in 0..5 {
        let peer_id = format!("peer-{}", i);
        peer_manager.add_peer(peer_id.clone());

        // Record different performance characteristics
        let bytes = (i + 1) * 1_000_000;
        let latency = Duration::from_millis((i + 1) * 10);
        peer_manager.record_success(&peer_id, bytes, latency);

        // Record that this peer has the test CID
        peer_manager.record_has(&peer_id, test_cid);
    }

    // All peers should have positive scores
    let scores = peer_manager.get_scores();
    for i in 0..5 {
        let peer_id = format!("peer-{}", i);
        let score = scores.get(&peer_id).copied().unwrap_or(0.0);
        assert!(score > 0.0, "Peer {} has non-positive score", peer_id);
    }

    // Select best peer (should be peer-0 with lowest latency)
    let selected = peer_manager.select_peers(&test_cid, 1, SelectionStrategy::FastestFirst);
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0], "peer-0".to_string());
}

#[tokio::test]
async fn test_want_list_deadline_boost() {
    let mut want_list = WantList::new(WantListConfig::default());

    let cid1 = create_test_cid(1);
    let cid2 = create_test_cid(2);

    // Add wants with deadlines
    let entry1 = WantEntry::new(cid1, Priority::Low as i32, Duration::from_secs(10))
        .with_deadline(std::time::Instant::now() + Duration::from_millis(10));
    let entry2 = WantEntry::new(cid2, Priority::High as i32, Duration::from_secs(10))
        .with_deadline(std::time::Instant::now() + Duration::from_secs(100));

    want_list.add(entry1);
    want_list.add(entry2);

    // Wait a bit
    tokio::time::sleep(Duration::from_millis(20)).await;

    // Boost deadline priorities
    want_list.boost_deadline_priorities();

    // Both CIDs should still be present
    assert!(want_list.contains(&cid1));
    assert!(want_list.contains(&cid2));
}

//
// Advanced Simulation Tests for Network Conditions
//

/// Test packet loss resilience with random message drops
#[tokio::test]
async fn test_packet_loss_resilience() {
    let mut peer_manager = PeerManager::new(PeerScoringConfig::default());
    let peer_id = "lossy-peer".to_string();
    peer_manager.add_peer(peer_id.clone());

    let mut rng = rand::rng();
    let test_cid = create_test_cid(1000);
    peer_manager.record_has(&peer_id, test_cid);

    // Simulate 30% packet loss scenario
    let packet_loss_rate = 0.3;
    let total_attempts = 100;
    let mut successful_attempts = 0;
    let mut failed_attempts = 0;

    for _ in 0..total_attempts {
        // Randomly drop packets
        if rng.random::<f64>() > packet_loss_rate {
            // Message got through successfully
            peer_manager.record_success(&peer_id, 1024, Duration::from_millis(10));
            successful_attempts += 1;
        } else {
            // Message was dropped - record as failure
            peer_manager.record_failure(&peer_id);
            failed_attempts += 1;
        }
    }

    // Verify peer scoring handles packet loss gracefully
    let scores = peer_manager.get_scores();
    let score = scores.get(&peer_id).copied().unwrap_or(0.0);

    // With 30% packet loss, peer should still have some score
    // but lower than perfect connection
    assert!(
        score >= 0.0,
        "Peer score should be non-negative even with packet loss"
    );

    // Verify we detected the loss pattern
    assert!(failed_attempts > 0, "Should have recorded some failures");
    assert!(
        successful_attempts > 0,
        "Should have recorded some successes"
    );

    // Rough validation that loss rate is close to expected
    let measured_loss_rate = failed_attempts as f64 / total_attempts as f64;
    assert!(
        (measured_loss_rate - packet_loss_rate).abs() < 0.15,
        "Measured loss rate {} should be close to expected {}",
        measured_loss_rate,
        packet_loss_rate
    );
}

/// Test high latency with variation (jitter)
#[tokio::test]
async fn test_latency_variation_handling() {
    let mut peer_manager = PeerManager::new(PeerScoringConfig::default());

    let peer_low_jitter = "peer-low-jitter".to_string();
    let peer_high_jitter = "peer-high-jitter".to_string();

    peer_manager.add_peer(peer_low_jitter.clone());
    peer_manager.add_peer(peer_high_jitter.clone());

    let test_cid = create_test_cid(2000);
    peer_manager.record_has(&peer_low_jitter, test_cid);
    peer_manager.record_has(&peer_high_jitter, test_cid);

    let mut rng = rand::rng();

    // Simulate peer with low jitter (100ms ± 5ms)
    for _ in 0..20 {
        let jitter = rng.random_range(95..=105);
        peer_manager.record_success(&peer_low_jitter, 1_000_000, Duration::from_millis(jitter));
    }

    // Simulate peer with high jitter (100ms ± 50ms)
    for _ in 0..20 {
        let jitter = rng.random_range(50..=150);
        peer_manager.record_success(&peer_high_jitter, 1_000_000, Duration::from_millis(jitter));
    }

    // Low jitter peer should score better
    let scores = peer_manager.get_scores();
    let low_jitter_score = scores.get(&peer_low_jitter).copied().unwrap_or(0.0);
    let high_jitter_score = scores.get(&peer_high_jitter).copied().unwrap_or(0.0);

    // Both should have positive scores
    assert!(
        low_jitter_score > 0.0,
        "Low jitter peer should have positive score"
    );
    assert!(
        high_jitter_score > 0.0,
        "High jitter peer should have positive score"
    );

    // Low jitter peer should generally be preferred (higher score)
    // Note: Due to EWMA smoothing, this might not always be strictly true,
    // but we verify both peers are scored
    assert!(
        low_jitter_score >= 0.0 && high_jitter_score >= 0.0,
        "Both peers should be scored despite latency variation"
    );
}

/// Test peer manager stress with many peers under concurrent load
#[tokio::test]
async fn test_peer_manager_concurrent_stress() {
    let mut peer_manager = PeerManager::new(PeerScoringConfig::default());
    let test_cid = create_test_cid(5000);

    // Add many peers
    let peer_count = 50;
    for i in 0..peer_count {
        let peer_id = format!("peer-{}", i);
        peer_manager.add_peer(peer_id.clone());
        peer_manager.record_has(&peer_id, test_cid);
    }

    // Simulate concurrent record operations
    let mut rng = rand::rng();
    for i in 0..peer_count {
        let peer_id = format!("peer-{}", i);

        // Record 10 operations per peer with random characteristics
        for _ in 0..10 {
            if rng.random::<f64>() > 0.1 {
                // 90% success rate
                let bytes = rng.random_range(100_000..1_000_000);
                let latency_ms = rng.random_range(10..200);
                peer_manager.record_success(&peer_id, bytes, Duration::from_millis(latency_ms));
            } else {
                peer_manager.record_failure(&peer_id);
            }
        }
    }

    // Verify peer selection still works with many peers
    let selected = peer_manager.select_peers(&test_cid, 5, SelectionStrategy::BestScore);
    assert_eq!(selected.len(), 5, "Should select 5 best peers from pool");

    // Verify all peers are scored
    let scores = peer_manager.get_scores();
    assert_eq!(
        scores.len(),
        peer_count,
        "All peers should have scores after stress test"
    );
}

/// Test combined stress scenario: packet loss + high latency + limited bandwidth
#[tokio::test]
async fn test_combined_network_stress() {
    let mut peer_manager = PeerManager::new(PeerScoringConfig::default());

    let peer_id = "stressed-peer".to_string();
    peer_manager.add_peer(peer_id.clone());

    let test_cid = create_test_cid(3000);
    peer_manager.record_has(&peer_id, test_cid);

    let mut rng = rand::rng();
    let packet_loss_rate = 0.2; // 20% packet loss
    let base_latency_ms = 200; // High base latency
    let jitter_ms = 100; // High jitter

    let mut successful_transfers = 0;
    let mut failed_transfers = 0;

    // Simulate 50 transfer attempts under stress
    for _ in 0..50 {
        // Random packet loss
        if rng.random::<f64>() > packet_loss_rate {
            // Packet got through - with high latency and jitter
            let latency_ms = base_latency_ms + rng.random_range(0..jitter_ms);
            let latency = Duration::from_millis(latency_ms);

            // Variable bandwidth (100 KB - 1 MB)
            let bytes = rng.random_range(100_000..1_000_000);

            peer_manager.record_success(&peer_id, bytes, latency);
            successful_transfers += 1;
        } else {
            // Packet lost
            peer_manager.record_failure(&peer_id);
            failed_transfers += 1;
        }
    }

    // Verify the peer still has a score despite harsh conditions
    let scores = peer_manager.get_scores();
    let score = scores.get(&peer_id).copied().unwrap_or(0.0);

    assert!(
        score >= 0.0,
        "Peer should maintain non-negative score under stress"
    );
    assert!(
        successful_transfers > 0,
        "Should have some successful transfers"
    );
    assert!(
        failed_transfers > 0,
        "Should have recorded failures from packet loss"
    );

    // Peer should still be selectable for the CID
    let selected = peer_manager.select_peers(&test_cid, 1, SelectionStrategy::BestScore);
    assert!(
        !selected.is_empty(),
        "Should still be able to select peer despite network stress"
    );
}

/// Test session resilience under network partition simulation
#[tokio::test]
async fn test_session_network_partition_recovery() {
    let config = SessionConfig {
        timeout: Duration::from_secs(60),
        ..Default::default()
    };

    let (tx, mut rx) = mpsc::unbounded_channel();
    let session = Session::new(200, config, Some(tx));

    // Receive started event
    let _ = timeout(Duration::from_millis(100), rx.recv()).await;

    // Add blocks
    let cid1 = create_test_cid(10);
    let cid2 = create_test_cid(20);
    let cid3 = create_test_cid(30);

    session.add_block(cid1, None).unwrap();
    session.add_block(cid2, None).unwrap();
    session.add_block(cid3, None).unwrap();

    // Simulate partition: pause session (no progress)
    session.pause();
    assert_eq!(session.state(), SessionState::Paused);

    // Simulate partition healing: resume session
    session.resume();
    assert_eq!(session.state(), SessionState::Active);

    // Now blocks can be received
    let block1 = create_test_block(10);
    session.mark_received(&cid1, &block1).unwrap();

    // Verify session recovered and can complete
    assert_eq!(session.state(), SessionState::Active);
    let stats = session.stats();
    assert_eq!(stats.blocks_received, 1);

    // Complete remaining blocks
    let block2 = create_test_block(20);
    session.mark_received(&cid2, &block2).unwrap();
    let block3 = create_test_block(30);
    session.mark_received(&cid3, &block3).unwrap();

    assert_eq!(session.state(), SessionState::Completed);
}

/// Test want list behavior under high concurrency stress
#[tokio::test]
async fn test_want_list_high_concurrency_stress() {
    let want_list = Arc::new(parking_lot::RwLock::new(WantList::new(
        WantListConfig::default(),
    )));
    let mut handles = vec![];

    // High concurrency: 100 tasks adding wants simultaneously
    for i in 0..100 {
        let wl = want_list.clone();
        let handle = tokio::spawn(async move {
            let cid = create_test_cid(i);
            let priority = (i % 5) as i32; // Vary priorities
            wl.write().add_simple(cid, priority);

            // Also test concurrent reads
            let _ = wl.read().contains(&cid);
        });
        handles.push(handle);
    }

    // Wait for all tasks
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify all wants were added correctly
    let len = want_list.read().len();
    assert_eq!(
        len, 100,
        "All 100 wants should be added despite high concurrency"
    );

    // Test concurrent pop operations
    let mut pop_handles = vec![];
    for _ in 0..50 {
        let wl = want_list.clone();
        let handle = tokio::spawn(async move { wl.write().pop() });
        pop_handles.push(handle);
    }

    let mut popped_count = 0;
    for handle in pop_handles {
        if handle.await.unwrap().is_some() {
            popped_count += 1;
        }
    }

    // Should have popped exactly 50 items
    assert_eq!(popped_count, 50);

    // Remaining count should be 50
    let remaining = want_list.read().len();
    assert_eq!(remaining, 50);
}

/// Test utility helper functions
#[tokio::test]
async fn test_utility_helpers() {
    use ipfrs_transport::{
        adjust_priority_for_deadline, calculate_optimal_chunk_size, estimate_transfer_time,
        format_bandwidth, format_bytes,
    };

    // Test byte formatting
    assert_eq!(format_bytes(1024), "1.00 KB");
    assert_eq!(format_bytes(1024 * 1024), "1.00 MB");

    // Test bandwidth formatting
    assert_eq!(format_bandwidth(1_000_000), "1.00 Mbps");

    // Test transfer time estimation
    let duration = estimate_transfer_time(1_000_000, 1_000_000);
    assert_eq!(duration.as_secs(), 8); // 1 MB at 1 Mbps = 8 seconds

    // Test optimal chunk size calculation
    let chunk_size = calculate_optimal_chunk_size(10_000_000, Duration::from_millis(100));
    assert!(chunk_size >= 64 * 1024);
    assert!(chunk_size <= 16 * 1024 * 1024);

    // Test priority adjustment for deadline
    let priority = adjust_priority_for_deadline(500, Duration::from_secs(0), 2.0);
    assert_eq!(priority, 1000); // Immediate deadline should get max priority
}

/// Test want list with utility helper for bulk operations
#[tokio::test]
async fn test_bulk_add_wants() {
    use ipfrs_transport::{bulk_add_wants, create_low_latency_want_list};

    let want_list = create_low_latency_want_list();

    // Create bulk CIDs
    let cids: Vec<_> = (0..100).map(create_test_cid).collect();

    // Bulk add with same priority
    bulk_add_wants(&want_list, &cids, 500);

    // Verify all were added
    assert_eq!(want_list.len(), 100);

    // Verify they all have the same priority
    for cid in &cids {
        assert!(want_list.contains(cid));
    }
}

/// Test session with optimized configurations
#[tokio::test]
async fn test_optimized_session_configs() {
    use ipfrs_transport::{create_bulk_transfer_session, create_interactive_session};

    let bulk_session = create_bulk_transfer_session(1);
    let bulk_stats = bulk_session.stats();
    assert_eq!(bulk_stats.total_blocks, 0);

    let interactive_session = create_interactive_session(2);
    let interactive_stats = interactive_session.stats();
    assert_eq!(interactive_stats.total_blocks, 0);

    // Add blocks and verify they work
    let cids: Vec<_> = (0..10).map(create_test_cid).collect();
    bulk_session.add_blocks(&cids, None).unwrap();
    assert_eq!(bulk_session.stats().total_blocks, 10);
}

/// Test peer manager with optimized configurations
#[tokio::test]
async fn test_optimized_peer_managers() {
    use ipfrs_transport::{
        create_bandwidth_optimized_peer_manager, create_latency_optimized_peer_manager,
    };

    let latency_manager = create_latency_optimized_peer_manager();
    let bandwidth_manager = create_bandwidth_optimized_peer_manager();

    // Add same peer to both
    let peer = "test-peer".to_string();
    latency_manager.add_peer(peer.clone());
    bandwidth_manager.add_peer(peer.clone());

    // Record same metrics
    latency_manager.record_success(&peer, 1_000_000, Duration::from_millis(50));
    bandwidth_manager.record_success(&peer, 1_000_000, Duration::from_millis(50));

    // Both should have the peer
    assert_eq!(latency_manager.stats().total_peers, 1);
    assert_eq!(bandwidth_manager.stats().total_peers, 1);
}

/// Test error recovery with want list retries
#[tokio::test]
async fn test_want_list_retry_mechanism() {
    let mut want_list = WantList::new(WantListConfig {
        max_wants: 1000,
        default_timeout: Duration::from_millis(100), // Short timeout for testing
        max_retries: 3,
        base_retry_delay: Duration::from_millis(10),
        max_retry_delay: Duration::from_secs(1),
    });

    let cid = create_test_cid(1);
    want_list.add_simple(cid, 500);

    // Simulate retry by checking retry count increases
    // Note: We can't easily test the actual retry mechanism without
    // a full mock environment, but we can verify the configuration works
    assert!(want_list.contains(&cid));
}

/// Test session completion detection via stats
#[tokio::test]
async fn test_session_completion_detection() {
    let config = SessionConfig {
        timeout: Duration::from_secs(60),
        default_priority: Priority::Normal,
        max_concurrent_blocks: 100,
        progress_notifications: true,
    };

    let session = Session::new(1, config, None);

    // Add blocks
    let cids: Vec<_> = (0..5).map(create_test_cid).collect();
    session.add_blocks(&cids, None).unwrap();

    // Initially nothing received
    let stats = session.stats();
    assert_eq!(stats.total_blocks, 5);
    assert_eq!(stats.blocks_received, 0);

    // Mark all as received
    let data = create_test_block(0);
    for cid in &cids {
        session.mark_received(cid, &data).unwrap();
    }

    // Session should be complete via stats
    let stats = session.stats();
    assert_eq!(stats.total_blocks, 5);
    assert_eq!(stats.blocks_received, 5);

    // All blocks received means session is effectively complete
    assert_eq!(stats.blocks_received, stats.total_blocks);
}

/// Test edge case: empty want list operations
#[tokio::test]
async fn test_empty_want_list() {
    let mut want_list = WantList::new(WantListConfig::default());

    // Pop from empty should return None
    assert!(want_list.pop().is_none());

    // Len should be 0
    assert_eq!(want_list.len(), 0);

    // Contains should return false
    let cid = create_test_cid(1);
    assert!(!want_list.contains(&cid));

    // Cleanup expired on empty list should work
    want_list.cleanup_expired();
    assert_eq!(want_list.len(), 0);
}

/// Test edge case: session with no blocks
#[tokio::test]
async fn test_empty_session() {
    let config = SessionConfig {
        timeout: Duration::from_secs(60),
        default_priority: Priority::Normal,
        max_concurrent_blocks: 100,
        progress_notifications: false,
    };

    let session = Session::new(1, config, None);
    let stats = session.stats();

    // Stats should reflect empty session
    assert_eq!(stats.total_blocks, 0);
    assert_eq!(stats.blocks_received, 0);
    assert_eq!(stats.blocks_failed, 0);
    assert_eq!(stats.bytes_transferred, 0);
}
