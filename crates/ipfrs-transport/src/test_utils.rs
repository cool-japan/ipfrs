//! Testing utilities for ipfrs-transport
//!
//! This module provides helper functions and utilities for writing tests
//! that use ipfrs-transport components.

use crate::{
    ConcurrentWantList, PeerManager, PeerScoringConfig, Priority, Session, SessionConfig,
    WantListConfig,
};
use ipfrs_core::Cid;
use multihash::Multihash;
use std::time::Duration;

/// Generate a deterministic CID for testing
///
/// Creates a CID from a seed value for reproducible tests.
pub fn test_cid(seed: u64) -> Cid {
    let data = seed.to_le_bytes();
    let hash = Multihash::wrap(0x12, &data)
        .expect("wrapping 8-byte seed into SHA2-256 multihash is infallible");
    Cid::new_v1(0x55, hash)
}

/// Generate multiple test CIDs
///
/// Creates a vector of CIDs with sequential seeds.
pub fn test_cids(count: usize) -> Vec<Cid> {
    (0..count).map(|i| test_cid(i as u64)).collect()
}

/// Create a test want list with default configuration
pub fn test_want_list() -> ConcurrentWantList {
    ConcurrentWantList::new(WantListConfig::default())
}

/// Create a test want list pre-populated with CIDs
pub fn test_want_list_with_cids(cids: &[Cid], priority: i32) -> ConcurrentWantList {
    let want_list = test_want_list();
    for cid in cids {
        want_list.add_simple(*cid, priority);
    }
    want_list
}

/// Create a test peer manager with default configuration
pub fn test_peer_manager() -> PeerManager {
    PeerManager::new(PeerScoringConfig::default())
}

/// Create a test peer manager with custom configuration
pub fn test_peer_manager_with_config(config: PeerScoringConfig) -> PeerManager {
    PeerManager::new(config)
}

/// Create a test session with default configuration
pub fn test_session(session_id: u64) -> Session {
    let config = SessionConfig {
        timeout: Duration::from_secs(60),
        default_priority: Priority::Normal,
        max_concurrent_blocks: 100,
        progress_notifications: false,
    };
    Session::new(session_id, config, None)
}

/// Create a test session pre-populated with blocks
pub fn test_session_with_blocks(session_id: u64, cids: &[Cid]) -> Session {
    let session = test_session(session_id);
    session.add_blocks(cids, None).ok();
    session
}

/// Generate test peer IDs
pub fn test_peer_ids(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("peer_{}", i)).collect()
}

/// Add test peers to a peer manager with default metrics
pub fn add_test_peers(manager: &mut PeerManager, count: usize) {
    for i in 0..count {
        let peer_id = format!("peer_{}", i);
        manager.add_peer(peer_id.clone());

        // Record some basic metrics to make the peer active
        manager.record_success(&peer_id, 1024, Duration::from_millis(10));
    }
}

/// Add test peers with varied performance characteristics
///
/// Creates peers with different latencies and bandwidths for testing
/// peer selection and scoring algorithms.
pub fn add_varied_test_peers(manager: &mut PeerManager, count: usize) {
    for i in 0..count {
        let peer_id = format!("peer_{}", i);
        manager.add_peer(peer_id.clone());

        // Vary latency from 5ms to 100ms
        let latency_ms = 5 + (i * 95 / count.max(1));

        // Vary bandwidth from 1KB to 10KB per transfer
        let bytes = (1024 + (i * 9216 / count.max(1))) as u64;

        manager.record_success(&peer_id, bytes, Duration::from_millis(latency_ms as u64));
    }
}

/// Create a minimal valid WantListConfig for testing
pub fn minimal_want_list_config() -> WantListConfig {
    WantListConfig {
        max_wants: 1,
        default_timeout: Duration::from_secs(1),
        max_retries: 1,
        base_retry_delay: Duration::from_millis(10),
        max_retry_delay: Duration::from_millis(100),
    }
}

/// Create a minimal valid SessionConfig for testing
pub fn minimal_session_config() -> SessionConfig {
    SessionConfig {
        timeout: Duration::from_secs(1),
        default_priority: Priority::Normal,
        max_concurrent_blocks: 1,
        progress_notifications: false,
    }
}

/// Create a minimal valid PeerScoringConfig for testing
pub fn minimal_peer_scoring_config() -> PeerScoringConfig {
    PeerScoringConfig {
        latency_weight: 0.33,
        bandwidth_weight: 0.34,
        reliability_weight: 0.33,
        ewma_alpha: 0.25,
        inactivity_decay: 0.02,
        min_score: 0.0,
        max_failures: 1,
    }
}

/// Assert that two float values are approximately equal
///
/// Useful for comparing scores, weights, and other floating-point calculations.
pub fn assert_approx_eq(a: f64, b: f64, epsilon: f64) {
    assert!(
        (a - b).abs() <= epsilon,
        "Values not approximately equal: {} vs {} (epsilon: {})",
        a,
        b,
        epsilon
    );
}

/// Assert that a value is within a range (inclusive)
pub fn assert_in_range<T: PartialOrd + std::fmt::Debug>(value: T, min: T, max: T) {
    assert!(
        value >= min && value <= max,
        "Value {:?} not in range [{:?}, {:?}]",
        value,
        min,
        max
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_test_cid() {
        let cid1 = test_cid(1);
        let cid2 = test_cid(1);
        let cid3 = test_cid(2);

        // Same seed produces same CID
        assert_eq!(cid1, cid2);
        // Different seeds produce different CIDs
        assert_ne!(cid1, cid3);
    }

    #[test]
    fn test_test_cids() {
        let cids = test_cids(10);
        assert_eq!(cids.len(), 10);

        // All CIDs should be unique
        for i in 0..cids.len() {
            for j in (i + 1)..cids.len() {
                assert_ne!(cids[i], cids[j]);
            }
        }
    }

    #[test]
    fn test_test_want_list() {
        let want_list = test_want_list();
        assert_eq!(want_list.len(), 0);
    }

    #[test]
    fn test_test_want_list_with_cids() {
        let cids = test_cids(5);
        let want_list = test_want_list_with_cids(&cids, 100);
        assert_eq!(want_list.len(), 5);
    }

    #[test]
    fn test_test_peer_manager() {
        let manager = test_peer_manager();
        let stats = manager.stats();
        assert_eq!(stats.total_peers, 0);
    }

    #[test]
    fn test_test_session() {
        let session = test_session(1);
        let stats = session.stats();
        assert_eq!(stats.total_blocks, 0);
    }

    #[test]
    fn test_test_session_with_blocks() {
        let cids = test_cids(5);
        let session = test_session_with_blocks(1, &cids);
        let stats = session.stats();
        assert_eq!(stats.total_blocks, 5);
    }

    #[test]
    fn test_test_peer_ids() {
        let peer_ids = test_peer_ids(3);
        assert_eq!(peer_ids.len(), 3);
        assert_eq!(peer_ids[0], "peer_0");
        assert_eq!(peer_ids[1], "peer_1");
        assert_eq!(peer_ids[2], "peer_2");
    }

    #[test]
    fn test_add_test_peers() {
        let mut manager = test_peer_manager();
        add_test_peers(&mut manager, 5);
        let stats = manager.stats();
        assert_eq!(stats.total_peers, 5);
    }

    #[test]
    fn test_add_varied_test_peers() {
        let mut manager = test_peer_manager();
        add_varied_test_peers(&mut manager, 5);
        let stats = manager.stats();
        assert_eq!(stats.total_peers, 5);
    }

    #[test]
    fn test_minimal_configs() {
        use crate::{
            validate_peer_scoring_config, validate_session_config, validate_want_list_config,
        };

        let want_list_config = minimal_want_list_config();
        assert!(validate_want_list_config(&want_list_config).is_ok());

        let session_config = minimal_session_config();
        assert!(validate_session_config(&session_config).is_ok());

        let peer_scoring_config = minimal_peer_scoring_config();
        assert!(validate_peer_scoring_config(&peer_scoring_config).is_ok());
    }

    #[test]
    fn test_assert_approx_eq() {
        assert_approx_eq(1.0, 1.0, 0.01);
        assert_approx_eq(1.0, 1.005, 0.01);
    }

    #[test]
    #[should_panic(expected = "Values not approximately equal")]
    fn test_assert_approx_eq_fails() {
        assert_approx_eq(1.0, 1.1, 0.01);
    }

    #[test]
    fn test_assert_in_range() {
        assert_in_range(5, 0, 10);
        assert_in_range(0, 0, 10);
        assert_in_range(10, 0, 10);
    }

    #[test]
    #[should_panic(expected = "Value")]
    fn test_assert_in_range_fails() {
        assert_in_range(11, 0, 10);
    }
}
