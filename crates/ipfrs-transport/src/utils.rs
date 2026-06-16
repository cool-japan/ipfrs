//! Utility functions and helpers for common transport operations
//!
//! This module provides convenience functions that simplify common tasks
//! when working with the ipfrs-transport crate.

use crate::{
    ConcurrentPeerManager, ConcurrentWantList, PeerScoringConfig, Priority, Session, SessionConfig,
    WantListConfig,
};
use ipfrs_core::Cid;
use std::time::Duration;

/// Quick setup for a want list with sensible defaults for high-throughput scenarios
pub fn create_high_throughput_want_list() -> ConcurrentWantList {
    let config = WantListConfig {
        max_wants: 10000,
        default_timeout: Duration::from_secs(120),
        max_retries: 5,
        base_retry_delay: Duration::from_millis(50),
        max_retry_delay: Duration::from_secs(10),
    };
    ConcurrentWantList::new(config)
}

/// Quick setup for a want list with sensible defaults for low-latency scenarios
pub fn create_low_latency_want_list() -> ConcurrentWantList {
    let config = WantListConfig {
        max_wants: 1000,
        default_timeout: Duration::from_secs(30),
        max_retries: 3,
        base_retry_delay: Duration::from_millis(10),
        max_retry_delay: Duration::from_secs(5),
    };
    ConcurrentWantList::new(config)
}

/// Quick setup for a peer manager optimized for latency-sensitive workloads
pub fn create_latency_optimized_peer_manager() -> ConcurrentPeerManager {
    let config = PeerScoringConfig {
        latency_weight: 0.6,     // Prioritize low latency
        bandwidth_weight: 0.2,   // Less important
        reliability_weight: 0.2, // Less important
        ewma_alpha: 0.3,         // More responsive to changes
        inactivity_decay: 0.05,  // Faster decay
        min_score: 0.1,
        max_failures: 3, // Lower tolerance
    };
    ConcurrentPeerManager::new(config)
}

/// Quick setup for a peer manager optimized for bandwidth-intensive workloads
pub fn create_bandwidth_optimized_peer_manager() -> ConcurrentPeerManager {
    let config = PeerScoringConfig {
        latency_weight: 0.2,     // Less important
        bandwidth_weight: 0.6,   // Prioritize high bandwidth
        reliability_weight: 0.2, // Less important
        ewma_alpha: 0.2,         // More stable
        inactivity_decay: 0.01,  // Slower decay
        min_score: 0.05,
        max_failures: 5, // Higher tolerance
    };
    ConcurrentPeerManager::new(config)
}

/// Quick setup for a session with sensible defaults for bulk transfers
pub fn create_bulk_transfer_session(session_id: u64) -> Session {
    let config = SessionConfig {
        timeout: Duration::from_secs(300), // 5 minutes
        default_priority: Priority::Normal,
        max_concurrent_blocks: 500,
        progress_notifications: true,
    };
    Session::new(session_id, config, None)
}

/// Quick setup for a session with sensible defaults for interactive transfers
pub fn create_interactive_session(session_id: u64) -> Session {
    let config = SessionConfig {
        timeout: Duration::from_secs(60), // 1 minute
        default_priority: Priority::High,
        max_concurrent_blocks: 100,
        progress_notifications: true,
    };
    Session::new(session_id, config, None)
}

/// Helper to bulk-add CIDs to a want list with the same priority
///
/// This uses the batch operation for improved performance when adding many CIDs.
pub fn bulk_add_wants(want_list: &ConcurrentWantList, cids: &[Cid], priority: i32) {
    want_list.add_batch_same_priority(cids, priority);
}

/// Helper to estimate transfer time based on size and bandwidth
///
/// Returns estimated duration in seconds
pub fn estimate_transfer_time(size_bytes: u64, bandwidth_bps: u64) -> Duration {
    if bandwidth_bps == 0 {
        return Duration::from_secs(u64::MAX);
    }
    let seconds = size_bytes / (bandwidth_bps / 8); // Convert bits to bytes
    Duration::from_secs(seconds)
}

/// Helper to calculate optimal chunk size for a given bandwidth and latency
///
/// Uses the bandwidth-delay product to determine optimal chunk size
pub fn calculate_optimal_chunk_size(bandwidth_bps: u64, latency: Duration) -> usize {
    let latency_secs = latency.as_secs_f64();
    let bdp = (bandwidth_bps as f64 * latency_secs / 8.0) as usize; // Bandwidth-delay product in bytes

    // Clamp to reasonable bounds (64 KB to 16 MB)
    bdp.clamp(64 * 1024, 16 * 1024 * 1024)
}

/// Helper to determine if a priority should be boosted based on deadline
///
/// Returns a new priority value that considers the deadline urgency
pub fn adjust_priority_for_deadline(
    base_priority: i32,
    deadline: Duration,
    boost_factor: f64,
) -> i32 {
    if deadline.as_secs() == 0 {
        return 1000; // Maximum priority for immediate deadlines
    }

    let urgency = 1.0 / deadline.as_secs_f64();
    let boost = (base_priority as f64 * boost_factor * urgency) as i32;
    (base_priority + boost).min(1000) // Cap at 1000
}

/// Helper to format byte sizes in human-readable format
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    if unit_idx == 0 {
        format!("{} {}", bytes, UNITS[0])
    } else {
        format!("{:.2} {}", size, UNITS[unit_idx])
    }
}

/// Helper to format bandwidth in human-readable format
pub fn format_bandwidth(bps: u64) -> String {
    const UNITS: &[&str] = &["bps", "Kbps", "Mbps", "Gbps", "Tbps"];
    let mut rate = bps as f64;
    let mut unit_idx = 0;

    while rate >= 1000.0 && unit_idx < UNITS.len() - 1 {
        rate /= 1000.0;
        unit_idx += 1;
    }

    format!("{:.2} {}", rate, UNITS[unit_idx])
}

/// Helper to batch-remove CIDs from a want list
///
/// This uses the batch operation for improved performance when removing many CIDs.
pub fn bulk_remove_wants(want_list: &ConcurrentWantList, cids: &[Cid]) {
    want_list.remove_batch(cids);
}

/// Helper to batch-update priorities for multiple CIDs
///
/// This uses the batch operation for improved performance when updating many priorities.
pub fn bulk_update_priorities(want_list: &ConcurrentWantList, updates: &[(Cid, i32)]) {
    want_list.update_priorities_batch(updates);
}

/// Helper to check if all CIDs are present in the want list
pub fn all_wants_present(want_list: &ConcurrentWantList, cids: &[Cid]) -> bool {
    want_list.contains_all(cids)
}

/// Helper to check if any CID is present in the want list
pub fn any_want_present(want_list: &ConcurrentWantList, cids: &[Cid]) -> bool {
    want_list.contains_any(cids)
}

/// Validate that a WantListConfig has reasonable values
///
/// Returns an error message if the configuration is invalid
pub fn validate_want_list_config(config: &WantListConfig) -> Result<(), String> {
    if config.max_wants == 0 {
        return Err("max_wants must be greater than 0".to_string());
    }
    if config.max_retries == 0 {
        return Err("max_retries must be greater than 0".to_string());
    }
    if config.base_retry_delay >= config.max_retry_delay {
        return Err("base_retry_delay must be less than max_retry_delay".to_string());
    }
    if config.default_timeout.as_secs() == 0 {
        return Err("default_timeout must be greater than 0".to_string());
    }
    Ok(())
}

/// Validate that a SessionConfig has reasonable values
///
/// Returns an error message if the configuration is invalid
pub fn validate_session_config(config: &SessionConfig) -> Result<(), String> {
    if config.max_concurrent_blocks == 0 {
        return Err("max_concurrent_blocks must be greater than 0".to_string());
    }
    if config.timeout.as_secs() == 0 {
        return Err("timeout must be greater than 0".to_string());
    }
    Ok(())
}

/// Validate that a PeerScoringConfig has reasonable values
///
/// Returns an error message if the configuration is invalid
pub fn validate_peer_scoring_config(config: &PeerScoringConfig) -> Result<(), String> {
    // Check that weights sum to approximately 1.0
    let weight_sum = config.latency_weight + config.bandwidth_weight + config.reliability_weight;
    if (weight_sum - 1.0).abs() > 0.01 {
        return Err(format!("weights must sum to 1.0, got {}", weight_sum));
    }

    // Check individual weight ranges
    if config.latency_weight < 0.0 || config.latency_weight > 1.0 {
        return Err("latency_weight must be between 0.0 and 1.0".to_string());
    }
    if config.bandwidth_weight < 0.0 || config.bandwidth_weight > 1.0 {
        return Err("bandwidth_weight must be between 0.0 and 1.0".to_string());
    }
    if config.reliability_weight < 0.0 || config.reliability_weight > 1.0 {
        return Err("reliability_weight must be between 0.0 and 1.0".to_string());
    }

    // Check EWMA alpha
    if config.ewma_alpha <= 0.0 || config.ewma_alpha >= 1.0 {
        return Err("ewma_alpha must be between 0.0 and 1.0 (exclusive)".to_string());
    }

    // Check decay rate
    if config.inactivity_decay < 0.0 || config.inactivity_decay > 1.0 {
        return Err("inactivity_decay must be between 0.0 and 1.0".to_string());
    }

    // Check score bounds
    if config.min_score < 0.0 || config.min_score > 1.0 {
        return Err("min_score must be between 0.0 and 1.0".to_string());
    }

    // Check max failures
    if config.max_failures == 0 {
        return Err("max_failures must be greater than 0".to_string());
    }

    Ok(())
}

/// Calculate the number of parallel requests based on bandwidth and latency
///
/// Uses the bandwidth-delay product to estimate optimal concurrency
pub fn calculate_optimal_concurrency(
    bandwidth_bps: u64,
    latency: Duration,
    block_size_bytes: usize,
) -> usize {
    if block_size_bytes == 0 {
        return 1;
    }

    let latency_secs = latency.as_secs_f64();
    let bytes_per_sec = bandwidth_bps / 8;
    let blocks_per_sec = bytes_per_sec / block_size_bytes as u64;
    let optimal = (blocks_per_sec as f64 * latency_secs).ceil() as usize;

    // Clamp to reasonable bounds (1 to 1000)
    optimal.clamp(1, 1000).max(1)
}

/// Helper to create a balanced peer scoring configuration
///
/// All weights are equal, suitable for general-purpose use
pub fn create_balanced_peer_scoring() -> PeerScoringConfig {
    PeerScoringConfig {
        latency_weight: 0.33,
        bandwidth_weight: 0.34,
        reliability_weight: 0.33,
        ewma_alpha: 0.25,
        inactivity_decay: 0.02,
        min_score: 0.1,
        max_failures: 5,
    }
}

/// Helper to create a reliability-focused peer scoring configuration
///
/// Prioritizes peer reliability over latency and bandwidth
pub fn create_reliability_focused_scoring() -> PeerScoringConfig {
    PeerScoringConfig {
        latency_weight: 0.15,
        bandwidth_weight: 0.15,
        reliability_weight: 0.70,
        ewma_alpha: 0.15,       // More stable
        inactivity_decay: 0.01, // Slower decay
        min_score: 0.2,         // Higher minimum
        max_failures: 2,        // Lower tolerance
    }
}

/// Quick setup for a want list optimized for edge/mobile devices
///
/// Lower limits to conserve memory and reduce overhead
pub fn create_edge_device_want_list() -> ConcurrentWantList {
    let config = WantListConfig {
        max_wants: 500,
        default_timeout: Duration::from_secs(60),
        max_retries: 2,
        base_retry_delay: Duration::from_millis(100),
        max_retry_delay: Duration::from_secs(30),
    };
    ConcurrentWantList::new(config)
}

/// Quick setup for a want list optimized for data center deployments
///
/// Higher limits for maximum throughput and parallelism
pub fn create_datacenter_want_list() -> ConcurrentWantList {
    let config = WantListConfig {
        max_wants: 50000,
        default_timeout: Duration::from_secs(180),
        max_retries: 10,
        base_retry_delay: Duration::from_millis(10),
        max_retry_delay: Duration::from_secs(30),
    };
    ConcurrentWantList::new(config)
}

/// Quick setup for a peer manager optimized for edge/mobile devices
///
/// More aggressive decay and lower tolerance to conserve resources
pub fn create_edge_device_peer_manager() -> ConcurrentPeerManager {
    let config = PeerScoringConfig {
        latency_weight: 0.4,
        bandwidth_weight: 0.3,
        reliability_weight: 0.3,
        ewma_alpha: 0.4,       // Very responsive
        inactivity_decay: 0.1, // Aggressive decay
        min_score: 0.2,
        max_failures: 2, // Low tolerance
    };
    ConcurrentPeerManager::new(config)
}

/// Quick setup for a session optimized for real-time applications
///
/// Very short timeout and high priority for minimal latency
pub fn create_realtime_session(session_id: u64) -> Session {
    let config = SessionConfig {
        timeout: Duration::from_secs(30),
        default_priority: Priority::Urgent,
        max_concurrent_blocks: 50,
        progress_notifications: true,
    };
    Session::new(session_id, config, None)
}

/// Quick setup for a session optimized for scientific computing workloads
///
/// Large concurrent blocks and long timeout for big data transfers
pub fn create_scientific_session(session_id: u64) -> Session {
    let config = SessionConfig {
        timeout: Duration::from_secs(600), // 10 minutes
        default_priority: Priority::High,
        max_concurrent_blocks: 1000,
        progress_notifications: true,
    };
    Session::new(session_id, config, None)
}

/// Calculate recommended buffer size based on bandwidth and latency
///
/// Uses the bandwidth-delay product to determine optimal buffer size
pub fn calculate_recommended_buffer_size(bandwidth_bps: u64, latency: Duration) -> usize {
    let latency_secs = latency.as_secs_f64();
    let bdp = (bandwidth_bps as f64 * latency_secs / 8.0) as usize;

    // Multiply by 2 for safety margin and clamp to reasonable bounds
    (bdp * 2).clamp(8 * 1024, 64 * 1024 * 1024)
}

/// Estimate the number of peers needed to achieve target bandwidth
///
/// Assumes even distribution of bandwidth across peers
pub fn estimate_required_peers(target_bandwidth_bps: u64, per_peer_bandwidth_bps: u64) -> usize {
    if per_peer_bandwidth_bps == 0 {
        return 0;
    }
    target_bandwidth_bps.div_ceil(per_peer_bandwidth_bps).max(1) as usize
}

/// Calculate expected throughput given current configuration
///
/// Returns estimated bytes per second based on concurrency and latency
pub fn calculate_expected_throughput(
    concurrent_blocks: usize,
    block_size_bytes: usize,
    latency: Duration,
) -> u64 {
    if latency.as_secs_f64() == 0.0 {
        return 0;
    }

    let blocks_per_second = concurrent_blocks as f64 / latency.as_secs_f64();
    (blocks_per_second * block_size_bytes as f64) as u64
}

/// Format a duration in human-readable format
pub fn format_duration(duration: Duration) -> String {
    let total_secs = duration.as_secs();

    if total_secs < 60 {
        format!("{}s", total_secs)
    } else if total_secs < 3600 {
        let mins = total_secs / 60;
        let secs = total_secs % 60;
        if secs == 0 {
            format!("{}m", mins)
        } else {
            format!("{}m {}s", mins, secs)
        }
    } else {
        let hours = total_secs / 3600;
        let mins = (total_secs % 3600) / 60;
        if mins == 0 {
            format!("{}h", hours)
        } else {
            format!("{}h {}m", hours, mins)
        }
    }
}

/// Generate a summary of WantListConfig for debugging
pub fn debug_want_list_config(config: &WantListConfig) -> String {
    format!(
        "WantListConfig {{ max_wants: {}, timeout: {}, retries: {}, base_delay: {}, max_delay: {} }}",
        config.max_wants,
        format_duration(config.default_timeout),
        config.max_retries,
        format_duration(config.base_retry_delay),
        format_duration(config.max_retry_delay)
    )
}

/// Generate a summary of PeerScoringConfig for debugging
pub fn debug_peer_scoring_config(config: &PeerScoringConfig) -> String {
    format!(
        "PeerScoringConfig {{ latency: {:.2}, bandwidth: {:.2}, reliability: {:.2}, ewma_alpha: {:.2}, decay: {:.2}, min_score: {:.2}, max_failures: {} }}",
        config.latency_weight,
        config.bandwidth_weight,
        config.reliability_weight,
        config.ewma_alpha,
        config.inactivity_decay,
        config.min_score,
        config.max_failures
    )
}

/// Generate a summary of SessionConfig for debugging
pub fn debug_session_config(config: &SessionConfig) -> String {
    format!(
        "SessionConfig {{ timeout: {}, priority: {:?}, max_concurrent: {}, notifications: {} }}",
        format_duration(config.timeout),
        config.default_priority,
        config.max_concurrent_blocks,
        config.progress_notifications
    )
}

/// Check if a WantListConfig is suitable for high-throughput scenarios
///
/// Returns true if the configuration appears optimized for throughput
pub fn is_high_throughput_config(config: &WantListConfig) -> bool {
    config.max_wants >= 5000 && config.max_retries >= 5
}

/// Check if a WantListConfig is suitable for low-latency scenarios
///
/// Returns true if the configuration appears optimized for latency
pub fn is_low_latency_config(config: &WantListConfig) -> bool {
    config.default_timeout.as_secs() <= 30 && config.base_retry_delay.as_millis() <= 50
}

/// Calculate the memory overhead estimate for a WantListConfig
///
/// Returns estimated bytes of memory overhead
pub fn estimate_want_list_memory(config: &WantListConfig) -> usize {
    // Rough estimate: each entry is about 100 bytes (CID + metadata)
    const BYTES_PER_ENTRY: usize = 100;
    config.max_wants * BYTES_PER_ENTRY
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_want_lists() {
        let high_throughput = create_high_throughput_want_list();
        assert_eq!(high_throughput.len(), 0);

        let low_latency = create_low_latency_want_list();
        assert_eq!(low_latency.len(), 0);
    }

    #[test]
    fn test_create_peer_managers() {
        let latency_optimized = create_latency_optimized_peer_manager();
        let stats = latency_optimized.stats();
        assert_eq!(stats.total_peers, 0);

        let bandwidth_optimized = create_bandwidth_optimized_peer_manager();
        let stats = bandwidth_optimized.stats();
        assert_eq!(stats.total_peers, 0);
    }

    #[test]
    fn test_create_sessions() {
        let bulk = create_bulk_transfer_session(1);
        let stats = bulk.stats();
        assert_eq!(stats.total_blocks, 0);

        let interactive = create_interactive_session(2);
        let stats = interactive.stats();
        assert_eq!(stats.total_blocks, 0);
    }

    #[test]
    fn test_estimate_transfer_time() {
        // 1 MB at 1 Mbps = 8 seconds
        let duration = estimate_transfer_time(1_000_000, 1_000_000);
        assert_eq!(duration.as_secs(), 8);

        // Zero bandwidth should return max duration
        let duration = estimate_transfer_time(1_000_000, 0);
        assert_eq!(duration.as_secs(), u64::MAX);
    }

    #[test]
    fn test_calculate_optimal_chunk_size() {
        // 1 Mbps, 100ms latency
        let chunk_size = calculate_optimal_chunk_size(1_000_000, Duration::from_millis(100));
        assert!(chunk_size >= 64 * 1024); // At least 64 KB
        assert!(chunk_size <= 16 * 1024 * 1024); // At most 16 MB

        // Very high bandwidth should be clamped
        let chunk_size = calculate_optimal_chunk_size(10_000_000_000, Duration::from_secs(1));
        assert_eq!(chunk_size, 16 * 1024 * 1024); // Clamped to 16 MB

        // Very low bandwidth should be clamped
        let chunk_size = calculate_optimal_chunk_size(100_000, Duration::from_millis(10));
        assert_eq!(chunk_size, 64 * 1024); // Clamped to 64 KB
    }

    #[test]
    fn test_adjust_priority_for_deadline() {
        // Immediate deadline should return max priority
        let priority = adjust_priority_for_deadline(500, Duration::from_secs(0), 2.0);
        assert_eq!(priority, 1000);

        // Far deadline should have minimal boost
        let priority = adjust_priority_for_deadline(500, Duration::from_secs(1000), 2.0);
        assert!(priority > 500);
        assert!(priority < 600);

        // Should be capped at 1000
        let priority = adjust_priority_for_deadline(900, Duration::from_secs(1), 10.0);
        assert_eq!(priority, 1000);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1536 * 1024), "1.50 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_format_bandwidth() {
        assert_eq!(format_bandwidth(500), "500.00 bps");
        assert_eq!(format_bandwidth(1000), "1.00 Kbps");
        assert_eq!(format_bandwidth(1_000_000), "1.00 Mbps");
        assert_eq!(format_bandwidth(1_500_000), "1.50 Mbps");
        assert_eq!(format_bandwidth(1_000_000_000), "1.00 Gbps");
    }

    #[test]
    fn test_bulk_operations() {
        use ipfrs_core::Cid;
        use multihash::Multihash;

        let want_list = create_low_latency_want_list();

        // Create test CIDs
        let cids: Vec<Cid> = (0u64..10)
            .map(|i| {
                let data = i.to_le_bytes();
                let hash = Multihash::wrap(0x12, &data).expect("test: wrap multihash for test CID");
                Cid::new_v1(0x55, hash)
            })
            .collect();

        // Test bulk_add_wants
        bulk_add_wants(&want_list, &cids, 100);
        assert_eq!(want_list.len(), 10);

        // Test all_wants_present and any_want_present
        assert!(all_wants_present(&want_list, &cids));
        assert!(any_want_present(&want_list, &cids[0..1]));

        // Test bulk_update_priorities
        let updates: Vec<(Cid, i32)> = cids.iter().map(|c| (*c, 200)).collect();
        bulk_update_priorities(&want_list, &updates);

        // Test bulk_remove_wants
        bulk_remove_wants(&want_list, &cids);
        assert_eq!(want_list.len(), 0);
        assert!(!all_wants_present(&want_list, &cids));
    }

    #[test]
    fn test_validate_want_list_config() {
        let valid_config = WantListConfig::default();
        assert!(validate_want_list_config(&valid_config).is_ok());

        let invalid_config = WantListConfig {
            max_wants: 0,
            ..Default::default()
        };
        assert!(validate_want_list_config(&invalid_config).is_err());

        let invalid_config = WantListConfig {
            max_retries: 0,
            ..Default::default()
        };
        assert!(validate_want_list_config(&invalid_config).is_err());

        let invalid_config = WantListConfig {
            base_retry_delay: Duration::from_secs(100),
            max_retry_delay: Duration::from_secs(10),
            ..Default::default()
        };
        assert!(validate_want_list_config(&invalid_config).is_err());
    }

    #[test]
    fn test_validate_session_config() {
        let valid_config = SessionConfig {
            timeout: Duration::from_secs(60),
            default_priority: Priority::Normal,
            max_concurrent_blocks: 100,
            progress_notifications: true,
        };
        assert!(validate_session_config(&valid_config).is_ok());

        let invalid_config = SessionConfig {
            max_concurrent_blocks: 0,
            ..valid_config
        };
        assert!(validate_session_config(&invalid_config).is_err());

        let invalid_config = SessionConfig {
            timeout: Duration::from_secs(0),
            ..valid_config
        };
        assert!(validate_session_config(&invalid_config).is_err());
    }

    #[test]
    fn test_validate_peer_scoring_config() {
        let valid_config = PeerScoringConfig::default();
        assert!(validate_peer_scoring_config(&valid_config).is_ok());

        // Test invalid weight sum
        let invalid_config = PeerScoringConfig {
            latency_weight: 0.5,
            bandwidth_weight: 0.5,
            reliability_weight: 0.5,
            ..Default::default()
        };
        assert!(validate_peer_scoring_config(&invalid_config).is_err());

        // Test invalid latency_weight
        let invalid_config = PeerScoringConfig {
            latency_weight: 1.5,
            ..Default::default()
        };
        assert!(validate_peer_scoring_config(&invalid_config).is_err());

        // Test invalid ewma_alpha
        let invalid_config = PeerScoringConfig {
            ewma_alpha: 1.0,
            ..Default::default()
        };
        assert!(validate_peer_scoring_config(&invalid_config).is_err());

        // Test invalid max_failures
        let invalid_config = PeerScoringConfig {
            max_failures: 0,
            ..Default::default()
        };
        assert!(validate_peer_scoring_config(&invalid_config).is_err());
    }

    #[test]
    fn test_calculate_optimal_concurrency() {
        // Test with reasonable values
        let concurrency = calculate_optimal_concurrency(
            10_000_000,                 // 10 Mbps
            Duration::from_millis(100), // 100ms latency
            256 * 1024,                 // 256 KB blocks
        );
        assert!((1..=1000).contains(&concurrency));

        // Test with zero block size
        let concurrency = calculate_optimal_concurrency(10_000_000, Duration::from_millis(100), 0);
        assert_eq!(concurrency, 1);

        // Test with very high bandwidth (should be clamped)
        let concurrency = calculate_optimal_concurrency(
            1_000_000_000_000, // 1 Tbps
            Duration::from_secs(1),
            1024,
        );
        assert_eq!(concurrency, 1000); // Should be clamped to max
    }

    #[test]
    fn test_create_balanced_peer_scoring() {
        let config = create_balanced_peer_scoring();
        assert!(validate_peer_scoring_config(&config).is_ok());

        // Weights should be roughly equal
        assert!((config.latency_weight - 0.33).abs() < 0.02);
        assert!((config.bandwidth_weight - 0.34).abs() < 0.02);
        assert!((config.reliability_weight - 0.33).abs() < 0.02);
    }

    #[test]
    fn test_create_reliability_focused_scoring() {
        let config = create_reliability_focused_scoring();
        assert!(validate_peer_scoring_config(&config).is_ok());

        // Reliability weight should be dominant
        assert!(config.reliability_weight > config.latency_weight);
        assert!(config.reliability_weight > config.bandwidth_weight);
        assert!(config.reliability_weight >= 0.7);
    }

    #[test]
    fn test_create_edge_device_want_list() {
        let want_list = create_edge_device_want_list();
        assert_eq!(want_list.len(), 0);

        // Edge device should have fewer max wants than high-throughput
        // (500 < 10000) — documented as a constant relationship, not asserted.
    }

    #[test]
    fn test_create_datacenter_want_list() {
        let want_list = create_datacenter_want_list();
        assert_eq!(want_list.len(), 0);

        // Should have higher limits than high-throughput
        // (50000 > 10000) — documented as a constant relationship, not asserted.
    }

    #[test]
    fn test_create_edge_device_peer_manager() {
        let manager = create_edge_device_peer_manager();
        let stats = manager.stats();
        assert_eq!(stats.total_peers, 0);
    }

    #[test]
    fn test_create_realtime_session() {
        let session = create_realtime_session(1);
        let stats = session.stats();
        assert_eq!(stats.total_blocks, 0);
    }

    #[test]
    fn test_create_scientific_session() {
        let session = create_scientific_session(1);
        let stats = session.stats();
        assert_eq!(stats.total_blocks, 0);
    }

    #[test]
    fn test_calculate_recommended_buffer_size() {
        // Test with reasonable values
        let buffer_size = calculate_recommended_buffer_size(
            10_000_000, // 10 Mbps
            Duration::from_millis(100),
        );
        assert!(buffer_size >= 8 * 1024); // At least 8 KB
        assert!(buffer_size <= 64 * 1024 * 1024); // At most 64 MB

        // Test with very high bandwidth (should be clamped)
        let buffer_size =
            calculate_recommended_buffer_size(10_000_000_000, Duration::from_secs(10));
        assert_eq!(buffer_size, 64 * 1024 * 1024); // Clamped to 64 MB

        // Test with very low bandwidth (should be clamped)
        let buffer_size = calculate_recommended_buffer_size(100_000, Duration::from_millis(10));
        assert_eq!(buffer_size, 8 * 1024); // Clamped to 8 KB
    }

    #[test]
    fn test_estimate_required_peers() {
        // 100 Mbps target, 10 Mbps per peer = 10 peers
        let peers = estimate_required_peers(100_000_000, 10_000_000);
        assert_eq!(peers, 10);

        // Zero per-peer bandwidth should return 0
        let peers = estimate_required_peers(100_000_000, 0);
        assert_eq!(peers, 0);

        // Uneven division should round up
        let peers = estimate_required_peers(100_000_000, 15_000_000);
        assert_eq!(peers, 7); // ceiling of 100/15
    }

    #[test]
    fn test_calculate_expected_throughput() {
        // 10 concurrent blocks, 1 MB each, 100ms latency
        // = 10 blocks / 0.1s = 100 blocks/s = 100 MB/s
        let throughput = calculate_expected_throughput(10, 1024 * 1024, Duration::from_millis(100));
        assert_eq!(throughput, 104_857_600); // 100 MB/s

        // Zero latency should return 0
        let throughput = calculate_expected_throughput(10, 1024 * 1024, Duration::from_secs(0));
        assert_eq!(throughput, 0);
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_secs(30)), "30s");
        assert_eq!(format_duration(Duration::from_secs(60)), "1m");
        assert_eq!(format_duration(Duration::from_secs(90)), "1m 30s");
        assert_eq!(format_duration(Duration::from_secs(3600)), "1h");
        assert_eq!(format_duration(Duration::from_secs(3660)), "1h 1m");
        assert_eq!(format_duration(Duration::from_secs(7200)), "2h");
    }

    #[test]
    fn test_debug_want_list_config() {
        let config = WantListConfig::default();
        let debug_str = debug_want_list_config(&config);
        assert!(debug_str.contains("WantListConfig"));
        assert!(debug_str.contains("max_wants"));
    }

    #[test]
    fn test_debug_peer_scoring_config() {
        let config = PeerScoringConfig::default();
        let debug_str = debug_peer_scoring_config(&config);
        assert!(debug_str.contains("PeerScoringConfig"));
        assert!(debug_str.contains("latency"));
    }

    #[test]
    fn test_debug_session_config() {
        let config = SessionConfig {
            timeout: Duration::from_secs(60),
            default_priority: Priority::Normal,
            max_concurrent_blocks: 100,
            progress_notifications: true,
        };
        let debug_str = debug_session_config(&config);
        assert!(debug_str.contains("SessionConfig"));
        assert!(debug_str.contains("timeout"));
    }

    #[test]
    fn test_is_high_throughput_config() {
        let config = WantListConfig {
            max_wants: 10000,
            default_timeout: Duration::from_secs(120),
            max_retries: 5,
            base_retry_delay: Duration::from_millis(50),
            max_retry_delay: Duration::from_secs(10),
        };
        assert!(is_high_throughput_config(&config));

        let config = WantListConfig {
            max_wants: 1000,
            default_timeout: Duration::from_secs(30),
            max_retries: 3,
            base_retry_delay: Duration::from_millis(10),
            max_retry_delay: Duration::from_secs(5),
        };
        assert!(!is_high_throughput_config(&config));
    }

    #[test]
    fn test_is_low_latency_config() {
        let config = WantListConfig {
            max_wants: 1000,
            default_timeout: Duration::from_secs(30),
            max_retries: 3,
            base_retry_delay: Duration::from_millis(10),
            max_retry_delay: Duration::from_secs(5),
        };
        assert!(is_low_latency_config(&config));

        let config = WantListConfig {
            max_wants: 10000,
            default_timeout: Duration::from_secs(120),
            max_retries: 5,
            base_retry_delay: Duration::from_millis(50),
            max_retry_delay: Duration::from_secs(10),
        };
        assert!(!is_low_latency_config(&config));
    }

    #[test]
    fn test_estimate_want_list_memory() {
        let config = WantListConfig {
            max_wants: 1000,
            ..Default::default()
        };
        let memory = estimate_want_list_memory(&config);
        assert_eq!(memory, 100_000); // 1000 * 100 bytes

        let config = WantListConfig {
            max_wants: 50000,
            ..Default::default()
        };
        let memory = estimate_want_list_memory(&config);
        assert_eq!(memory, 5_000_000); // 50000 * 100 bytes
    }
}
