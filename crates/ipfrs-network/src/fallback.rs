//! Fallback strategies for network error handling
//!
//! Provides comprehensive fallback mechanisms including:
//! - Alternative peer selection
//! - Relay fallback for NAT traversal
//! - Degraded mode operation
//! - Automatic retry with exponential backoff

use ipfrs_core::error::Error;
use libp2p::PeerId;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Fallback strategy configuration
#[derive(Debug, Clone)]
pub struct FallbackConfig {
    /// Maximum number of alternative peers to try
    pub max_alternatives: usize,
    /// Enable relay fallback
    pub enable_relay_fallback: bool,
    /// Enable degraded mode
    pub enable_degraded_mode: bool,
    /// Initial retry delay
    pub initial_retry_delay: Duration,
    /// Maximum retry delay
    pub max_retry_delay: Duration,
    /// Retry backoff multiplier
    pub backoff_multiplier: f64,
}

impl Default for FallbackConfig {
    fn default() -> Self {
        Self {
            max_alternatives: 5,
            enable_relay_fallback: true,
            enable_degraded_mode: true,
            initial_retry_delay: Duration::from_millis(100),
            max_retry_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
        }
    }
}

/// Fallback strategy for peer connections
#[derive(Debug, Clone)]
pub enum FallbackStrategy {
    /// Try alternative peers
    AlternativePeers {
        /// List of alternative peers to try
        alternatives: Vec<PeerId>,
    },
    /// Use relay connection
    RelayFallback {
        /// Relay peer ID
        relay_peer: PeerId,
        /// Target peer ID
        target_peer: PeerId,
    },
    /// Enter degraded mode
    DegradedMode {
        /// Reason for degraded mode
        reason: String,
    },
    /// Retry with exponential backoff
    RetryWithBackoff {
        /// Number of retries attempted
        attempt: usize,
        /// Next retry delay
        delay: Duration,
    },
}

impl FallbackStrategy {
    /// Get a description of the fallback strategy
    pub fn description(&self) -> String {
        match self {
            Self::AlternativePeers { alternatives } => {
                format!("Try {} alternative peer(s)", alternatives.len())
            }
            Self::RelayFallback {
                relay_peer,
                target_peer,
            } => {
                format!("Connect to {} via relay {}", target_peer, relay_peer)
            }
            Self::DegradedMode { reason } => {
                format!("Enter degraded mode: {}", reason)
            }
            Self::RetryWithBackoff { attempt, delay } => {
                format!("Retry attempt {} after {:?}", attempt, delay)
            }
        }
    }
}

/// Fallback manager for coordinating fallback strategies
pub struct FallbackManager {
    config: FallbackConfig,
    /// Alternative peers per content ID or operation
    alternatives: parking_lot::RwLock<HashMap<String, VecDeque<PeerId>>>,
    /// Available relay peers
    relay_peers: parking_lot::RwLock<Vec<PeerId>>,
    /// Retry state per peer
    retry_state: parking_lot::RwLock<HashMap<PeerId, RetryState>>,
    /// Degraded mode state
    degraded_mode: parking_lot::RwLock<bool>,
}

/// Retry state for a peer
#[derive(Debug, Clone)]
struct RetryState {
    attempts: usize,
    last_attempt: Instant,
    next_delay: Duration,
}

impl FallbackManager {
    /// Create a new fallback manager
    pub fn new(config: FallbackConfig) -> Self {
        Self {
            config,
            alternatives: parking_lot::RwLock::new(HashMap::new()),
            relay_peers: parking_lot::RwLock::new(Vec::new()),
            retry_state: parking_lot::RwLock::new(HashMap::new()),
            degraded_mode: parking_lot::RwLock::new(false),
        }
    }

    /// Add alternative peers for a key (content ID or operation)
    pub fn add_alternatives(&self, key: &str, peers: Vec<PeerId>) {
        let mut alternatives = self.alternatives.write();
        alternatives
            .entry(key.to_string())
            .or_default()
            .extend(peers);
    }

    /// Get next alternative peer for a key
    pub fn get_next_alternative(&self, key: &str) -> Option<PeerId> {
        let mut alternatives = self.alternatives.write();
        if let Some(peers) = alternatives.get_mut(key) {
            peers.pop_front()
        } else {
            None
        }
    }

    /// Add a relay peer
    pub fn add_relay_peer(&self, peer: PeerId) {
        let mut relay_peers = self.relay_peers.write();
        if !relay_peers.contains(&peer) {
            relay_peers.push(peer);
        }
    }

    /// Get available relay peers
    pub fn get_relay_peers(&self) -> Vec<PeerId> {
        self.relay_peers.read().clone()
    }

    /// Get fallback strategy for a failed connection
    pub fn get_fallback_strategy(
        &self,
        peer_id: PeerId,
        key: Option<&str>,
    ) -> Option<FallbackStrategy> {
        // 1. Try alternative peers first
        if let Some(key) = key {
            if let Some(alternative) = self.get_next_alternative(key) {
                let alternatives = vec![alternative];
                return Some(FallbackStrategy::AlternativePeers { alternatives });
            }
        }

        // 2. Try relay fallback
        if self.config.enable_relay_fallback {
            let relay_peers = self.get_relay_peers();
            if let Some(relay_peer) = relay_peers.first() {
                return Some(FallbackStrategy::RelayFallback {
                    relay_peer: *relay_peer,
                    target_peer: peer_id,
                });
            }
        }

        // 3. Try retry with backoff
        let mut retry_state = self.retry_state.write();
        let state = retry_state.entry(peer_id).or_insert_with(|| RetryState {
            attempts: 0,
            last_attempt: Instant::now(),
            next_delay: self.config.initial_retry_delay,
        });

        // Check if we should retry
        if state.last_attempt.elapsed() >= state.next_delay {
            state.attempts += 1;
            state.last_attempt = Instant::now();

            // Calculate next delay with exponential backoff
            let next_delay = Duration::from_secs_f64(
                state.next_delay.as_secs_f64() * self.config.backoff_multiplier,
            )
            .min(self.config.max_retry_delay);
            state.next_delay = next_delay;

            return Some(FallbackStrategy::RetryWithBackoff {
                attempt: state.attempts,
                delay: state.next_delay,
            });
        }

        // 4. Enter degraded mode if enabled
        if self.config.enable_degraded_mode {
            self.enter_degraded_mode("All fallback strategies exhausted");
            return Some(FallbackStrategy::DegradedMode {
                reason: "All fallback strategies exhausted".to_string(),
            });
        }

        None
    }

    /// Reset retry state for a peer (after successful connection)
    pub fn reset_retry_state(&self, peer_id: &PeerId) {
        let mut retry_state = self.retry_state.write();
        retry_state.remove(peer_id);
    }

    /// Enter degraded mode
    pub fn enter_degraded_mode(&self, reason: &str) {
        let mut degraded = self.degraded_mode.write();
        *degraded = true;
        tracing::warn!("Entering degraded mode: {}", reason);
    }

    /// Exit degraded mode
    pub fn exit_degraded_mode(&self) {
        let mut degraded = self.degraded_mode.write();
        *degraded = false;
        tracing::info!("Exiting degraded mode");
    }

    /// Check if in degraded mode
    pub fn is_degraded(&self) -> bool {
        *self.degraded_mode.read()
    }

    /// Get retry statistics
    pub fn retry_stats(&self) -> RetryStats {
        let retry_state = self.retry_state.read();
        let total_peers = retry_state.len();
        let total_attempts: usize = retry_state.values().map(|s| s.attempts).sum();

        RetryStats {
            total_peers_with_retries: total_peers,
            total_retry_attempts: total_attempts,
            peers_in_backoff: retry_state
                .values()
                .filter(|s| s.last_attempt.elapsed() < s.next_delay)
                .count(),
        }
    }

    /// Clear all fallback state
    pub fn clear(&self) {
        self.alternatives.write().clear();
        self.retry_state.write().clear();
        self.exit_degraded_mode();
    }
}

impl Default for FallbackManager {
    fn default() -> Self {
        Self::new(FallbackConfig::default())
    }
}

/// Retry statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct RetryStats {
    /// Number of peers with retry state
    pub total_peers_with_retries: usize,
    /// Total retry attempts across all peers
    pub total_retry_attempts: usize,
    /// Number of peers currently in backoff
    pub peers_in_backoff: usize,
}

/// Fallback result wrapper
pub enum FallbackResult<T> {
    /// Operation succeeded
    Success(T),
    /// Operation failed, fallback strategy available
    FallbackAvailable(FallbackStrategy),
    /// Operation failed, no fallback available
    Failed(Error),
}

impl<T> FallbackResult<T> {
    /// Check if the result is successful
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success(_))
    }

    /// Check if fallback is available
    pub fn has_fallback(&self) -> bool {
        matches!(self, Self::FallbackAvailable(_))
    }

    /// Unwrap the success value or panic
    pub fn unwrap(self) -> T {
        match self {
            Self::Success(value) => value,
            Self::FallbackAvailable(strategy) => {
                panic!("Called unwrap on FallbackAvailable: {:?}", strategy)
            }
            Self::Failed(error) => panic!("Called unwrap on Failed: {}", error),
        }
    }

    /// Get the success value or a default
    pub fn unwrap_or(self, default: T) -> T {
        match self {
            Self::Success(value) => value,
            _ => default,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_peer_id() -> PeerId {
        PeerId::random()
    }

    #[test]
    fn test_fallback_config_default() {
        let config = FallbackConfig::default();
        assert_eq!(config.max_alternatives, 5);
        assert!(config.enable_relay_fallback);
        assert!(config.enable_degraded_mode);
    }

    #[test]
    fn test_fallback_manager_creation() {
        let manager = FallbackManager::default();
        assert!(!manager.is_degraded());
        assert_eq!(manager.get_relay_peers().len(), 0);
    }

    #[test]
    fn test_add_and_get_alternatives() {
        let manager = FallbackManager::default();
        let peer1 = test_peer_id();
        let peer2 = test_peer_id();

        manager.add_alternatives("test_key", vec![peer1, peer2]);

        let alt1 = manager.get_next_alternative("test_key");
        assert_eq!(alt1, Some(peer1));

        let alt2 = manager.get_next_alternative("test_key");
        assert_eq!(alt2, Some(peer2));

        let alt3 = manager.get_next_alternative("test_key");
        assert_eq!(alt3, None);
    }

    #[test]
    fn test_relay_peer_management() {
        let manager = FallbackManager::default();
        let relay = test_peer_id();

        manager.add_relay_peer(relay);
        let relays = manager.get_relay_peers();

        assert_eq!(relays.len(), 1);
        assert_eq!(relays[0], relay);

        // Adding same relay again should not duplicate
        manager.add_relay_peer(relay);
        assert_eq!(manager.get_relay_peers().len(), 1);
    }

    #[test]
    fn test_fallback_strategy_alternative_peers() {
        let manager = FallbackManager::default();
        let peer = test_peer_id();
        let alt_peer = test_peer_id();

        manager.add_alternatives("test_key", vec![alt_peer]);

        let strategy = manager.get_fallback_strategy(peer, Some("test_key"));
        assert!(strategy.is_some());

        match strategy.expect("test: fallback strategy should be Some when alternative peer exists")
        {
            FallbackStrategy::AlternativePeers { alternatives } => {
                assert_eq!(alternatives.len(), 1);
                assert_eq!(alternatives[0], alt_peer);
            }
            _ => panic!("Expected AlternativePeers strategy"),
        }
    }

    #[test]
    fn test_fallback_strategy_relay() {
        let manager = FallbackManager::default();
        let peer = test_peer_id();
        let relay = test_peer_id();

        manager.add_relay_peer(relay);

        let strategy = manager.get_fallback_strategy(peer, None);
        assert!(strategy.is_some());

        match strategy.expect("test: fallback strategy should be Some when relay peer is available")
        {
            FallbackStrategy::RelayFallback {
                relay_peer,
                target_peer,
            } => {
                assert_eq!(relay_peer, relay);
                assert_eq!(target_peer, peer);
            }
            _ => panic!("Expected RelayFallback strategy"),
        }
    }

    #[test]
    fn test_retry_state_reset() {
        let manager = FallbackManager::default();
        let peer = test_peer_id();

        // Get a retry strategy to create state
        let _strategy = manager.get_fallback_strategy(peer, None);

        // Reset the state
        manager.reset_retry_state(&peer);

        // Stats should show no retries
        let stats = manager.retry_stats();
        assert_eq!(stats.total_peers_with_retries, 0);
    }

    #[test]
    fn test_degraded_mode() {
        let manager = FallbackManager::default();

        assert!(!manager.is_degraded());

        manager.enter_degraded_mode("Test reason");
        assert!(manager.is_degraded());

        manager.exit_degraded_mode();
        assert!(!manager.is_degraded());
    }

    #[test]
    fn test_retry_stats() {
        let manager = FallbackManager::default();
        let peer = test_peer_id();

        // Trigger a retry
        let _strategy = manager.get_fallback_strategy(peer, None);

        let stats = manager.retry_stats();
        assert!(stats.total_peers_with_retries > 0);
    }

    #[test]
    fn test_fallback_strategy_description() {
        let strategy = FallbackStrategy::AlternativePeers {
            alternatives: vec![test_peer_id()],
        };
        assert!(strategy.description().contains("alternative"));

        let strategy = FallbackStrategy::DegradedMode {
            reason: "test".to_string(),
        };
        assert!(strategy.description().contains("degraded"));
    }

    #[test]
    fn test_fallback_result_success() {
        let result: FallbackResult<i32> = FallbackResult::Success(42);
        assert!(result.is_success());
        assert!(!result.has_fallback());
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_fallback_result_with_fallback() {
        let strategy = FallbackStrategy::DegradedMode {
            reason: "test".to_string(),
        };
        let result: FallbackResult<i32> = FallbackResult::FallbackAvailable(strategy);
        assert!(!result.is_success());
        assert!(result.has_fallback());
        assert_eq!(result.unwrap_or(0), 0);
    }

    #[test]
    fn test_clear() {
        let manager = FallbackManager::default();
        let peer = test_peer_id();

        manager.add_alternatives("test", vec![peer]);
        manager.add_relay_peer(peer);
        manager.enter_degraded_mode("test");

        manager.clear();

        assert!(!manager.is_degraded());
        assert_eq!(manager.retry_stats().total_peers_with_retries, 0);
        assert_eq!(manager.get_next_alternative("test"), None);
    }
}
