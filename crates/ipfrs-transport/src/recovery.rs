//! Error recovery strategies for resilient operation
//!
//! Provides:
//! - Fallback peer selection
//! - Alternative provider discovery
//! - Degraded mode operation
//! - Automatic retry with backoff

use crate::peer_manager::PeerId;
use dashmap::DashMap;
use ipfrs_core::Cid;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tracing::{debug, info, warn};

/// Recovery error types
#[derive(Error, Debug)]
pub enum RecoveryError {
    #[error("No fallback peers available")]
    NoFallbackPeers,

    #[error("All providers exhausted")]
    AllProvidersExhausted,

    #[error("Degraded mode: limited functionality")]
    DegradedMode,

    #[error("Recovery failed: {0}")]
    RecoveryFailed(String),
}

/// Recovery strategy type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryStrategy {
    /// Try fallback peers in order
    FallbackPeers,
    /// Search for alternative providers
    AlternativeProviders,
    /// Enter degraded mode
    DegradedMode,
    /// Fail immediately
    FailFast,
}

/// Recovery mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RecoveryMode {
    /// Normal operation
    #[default]
    Normal,
    /// Degraded (limited functionality)
    Degraded,
    /// Emergency (minimal functionality)
    Emergency,
}

/// Recovery configuration
#[derive(Debug, Clone)]
pub struct RecoveryConfig {
    /// Primary recovery strategy
    pub strategy: RecoveryStrategy,
    /// Maximum fallback attempts
    pub max_fallback_attempts: usize,
    /// Timeout for each fallback attempt
    pub fallback_timeout: Duration,
    /// Enable automatic degraded mode
    pub auto_degrade: bool,
    /// Minimum peers for normal mode
    pub min_peers_normal: usize,
    /// Minimum peers for degraded mode
    pub min_peers_degraded: usize,
    /// Provider search timeout
    pub provider_search_timeout: Duration,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            strategy: RecoveryStrategy::FallbackPeers,
            max_fallback_attempts: 3,
            fallback_timeout: Duration::from_secs(5),
            auto_degrade: true,
            min_peers_normal: 3,
            min_peers_degraded: 1,
            provider_search_timeout: Duration::from_secs(10),
        }
    }
}

/// Recovery statistics
#[derive(Debug, Clone, Default)]
pub struct RecoveryStats {
    /// Total recovery attempts
    pub recovery_attempts: u64,
    /// Successful recoveries
    pub successful_recoveries: u64,
    /// Failed recoveries
    pub failed_recoveries: u64,
    /// Fallback peer uses
    pub fallback_uses: u64,
    /// Alternative provider finds
    pub alternative_providers_found: u64,
    /// Time in degraded mode
    pub degraded_mode_duration: Duration,
    /// Current mode
    pub current_mode: RecoveryMode,
}

/// Peer fallback information
#[derive(Debug, Clone)]
struct FallbackPeer {
    #[allow(dead_code)]
    peer_id: PeerId,
    addr: SocketAddr,
    priority: usize,
    last_used: Option<Instant>,
    success_count: usize,
    failure_count: usize,
}

/// Provider information for a CID
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ProviderInfo {
    providers: Vec<SocketAddr>,
    last_updated: Instant,
    exhausted: HashSet<SocketAddr>,
}

/// Error recovery manager
pub struct RecoveryManager {
    config: RecoveryConfig,
    mode: Arc<RwLock<RecoveryMode>>,
    fallback_peers: Arc<RwLock<HashMap<PeerId, Vec<FallbackPeer>>>>,
    providers: Arc<DashMap<Cid, ProviderInfo>>,
    stats: Arc<RwLock<RecoveryStats>>,
    degraded_since: Arc<RwLock<Option<Instant>>>,
}

impl RecoveryManager {
    /// Create a new recovery manager
    pub fn new(config: RecoveryConfig) -> Self {
        Self {
            config,
            mode: Arc::new(RwLock::new(RecoveryMode::Normal)),
            fallback_peers: Arc::new(RwLock::new(HashMap::new())),
            providers: Arc::new(DashMap::new()),
            stats: Arc::new(RwLock::new(RecoveryStats::default())),
            degraded_since: Arc::new(RwLock::new(None)),
        }
    }

    /// Register a fallback peer for a primary peer
    pub fn register_fallback(
        &self,
        primary: PeerId,
        fallback: PeerId,
        addr: SocketAddr,
        priority: usize,
    ) {
        let fallback_id = fallback.clone();
        let primary_id = primary.clone();

        let mut fallbacks = self.fallback_peers.write();
        let peer_fallbacks = fallbacks.entry(primary).or_default();

        peer_fallbacks.push(FallbackPeer {
            peer_id: fallback,
            addr,
            priority,
            last_used: None,
            success_count: 0,
            failure_count: 0,
        });

        // Sort by priority
        peer_fallbacks.sort_by_key(|f| f.priority);

        debug!(
            "Registered fallback peer {:?} for {:?} with priority {}",
            fallback_id, primary_id, priority
        );
    }

    /// Get fallback peers for a primary peer
    pub fn get_fallbacks(&self, primary: &PeerId) -> Vec<SocketAddr> {
        let fallbacks = self.fallback_peers.read();

        if let Some(peer_fallbacks) = fallbacks.get(primary) {
            peer_fallbacks
                .iter()
                .take(self.config.max_fallback_attempts)
                .map(|f| f.addr)
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Record successful fallback use
    pub fn record_fallback_success(&self, primary: &PeerId, fallback_addr: SocketAddr) {
        let mut fallbacks = self.fallback_peers.write();

        if let Some(peer_fallbacks) = fallbacks.get_mut(primary) {
            if let Some(fallback) = peer_fallbacks.iter_mut().find(|f| f.addr == fallback_addr) {
                fallback.success_count += 1;
                fallback.last_used = Some(Instant::now());

                self.stats.write().fallback_uses += 1;
                self.stats.write().successful_recoveries += 1;

                info!(
                    "Fallback peer {} succeeded for {:?}",
                    fallback_addr, primary
                );
            }
        }
    }

    /// Record failed fallback attempt
    pub fn record_fallback_failure(&self, primary: &PeerId, fallback_addr: SocketAddr) {
        let mut fallbacks = self.fallback_peers.write();

        if let Some(peer_fallbacks) = fallbacks.get_mut(primary) {
            if let Some(fallback) = peer_fallbacks.iter_mut().find(|f| f.addr == fallback_addr) {
                fallback.failure_count += 1;

                warn!("Fallback peer {} failed for {:?}", fallback_addr, primary);
            }
        }

        self.stats.write().failed_recoveries += 1;
    }

    /// Register providers for a CID
    pub fn register_providers(&self, cid: Cid, providers: Vec<SocketAddr>) {
        let provider_count = providers.len();

        let info = ProviderInfo {
            providers,
            last_updated: Instant::now(),
            exhausted: HashSet::new(),
        };

        self.providers.insert(cid, info);

        debug!("Registered {} providers for CID {}", provider_count, cid);
    }

    /// Get next available provider for a CID
    pub fn get_next_provider(&self, cid: &Cid) -> Option<SocketAddr> {
        if let Some(info) = self.providers.get_mut(cid) {
            for provider in &info.providers {
                if !info.exhausted.contains(provider) {
                    return Some(*provider);
                }
            }
        }

        None
    }

    /// Mark a provider as exhausted
    pub fn mark_provider_exhausted(&self, cid: &Cid, provider: SocketAddr) {
        if let Some(mut info) = self.providers.get_mut(cid) {
            info.exhausted.insert(provider);

            // Check if all providers exhausted
            if info.exhausted.len() >= info.providers.len() {
                warn!("All providers exhausted for CID {}", cid);
            }
        }
    }

    /// Mark a provider as successful
    pub fn mark_provider_success(&self, cid: &Cid, provider: SocketAddr) {
        if let Some(mut info) = self.providers.get_mut(cid) {
            info.exhausted.remove(&provider);
            self.stats.write().alternative_providers_found += 1;

            info!("Provider {} succeeded for CID {}", provider, cid);
        }
    }

    /// Enter degraded mode
    pub fn enter_degraded_mode(&self) {
        let mut mode = self.mode.write();
        if *mode == RecoveryMode::Normal {
            *mode = RecoveryMode::Degraded;
            *self.degraded_since.write() = Some(Instant::now());

            self.stats.write().current_mode = RecoveryMode::Degraded;

            warn!("Entering degraded mode");
        }
    }

    /// Enter emergency mode
    pub fn enter_emergency_mode(&self) {
        let mut mode = self.mode.write();
        if *mode != RecoveryMode::Emergency {
            *mode = RecoveryMode::Emergency;

            if self.degraded_since.read().is_none() {
                *self.degraded_since.write() = Some(Instant::now());
            }

            self.stats.write().current_mode = RecoveryMode::Emergency;

            warn!("Entering emergency mode");
        }
    }

    /// Exit degraded/emergency mode
    pub fn exit_degraded_mode(&self) {
        let mut mode = self.mode.write();
        if *mode != RecoveryMode::Normal {
            *mode = RecoveryMode::Normal;

            // Update degraded duration stats
            if let Some(since) = *self.degraded_since.read() {
                let mut stats = self.stats.write();
                stats.degraded_mode_duration += since.elapsed();
            }

            *self.degraded_since.write() = None;
            self.stats.write().current_mode = RecoveryMode::Normal;

            info!("Exited degraded mode - returning to normal operation");
        }
    }

    /// Get current recovery mode
    pub fn mode(&self) -> RecoveryMode {
        *self.mode.read()
    }

    /// Check if should auto-degrade based on peer count
    pub fn check_auto_degrade(&self, active_peers: usize) {
        if !self.config.auto_degrade {
            return;
        }

        let current_mode = *self.mode.read();

        if active_peers < self.config.min_peers_degraded {
            if current_mode != RecoveryMode::Emergency {
                self.enter_emergency_mode();
            }
        } else if active_peers < self.config.min_peers_normal {
            if current_mode == RecoveryMode::Normal {
                self.enter_degraded_mode();
            }
        } else if current_mode != RecoveryMode::Normal {
            self.exit_degraded_mode();
        }
    }

    /// Get statistics
    pub fn stats(&self) -> RecoveryStats {
        let mut stats = self.stats.read().clone();

        // Update degraded duration if currently degraded
        if let Some(since) = *self.degraded_since.read() {
            stats.degraded_mode_duration += since.elapsed();
        }

        stats
    }

    /// Clear all provider information
    pub fn clear_providers(&self) {
        self.providers.clear();
        info!("Cleared provider information");
    }

    /// Get recovery attempt count
    pub fn attempt_recovery(&self) -> u64 {
        let mut stats = self.stats.write();
        stats.recovery_attempts += 1;
        stats.recovery_attempts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_peer() -> PeerId {
        "test_peer_id_123".to_string()
    }

    fn dummy_cid() -> Cid {
        let data = vec![1u8; 32];
        Cid::new_v1(
            0x55,
            multihash::Multihash::wrap(0x12, &data).expect("test: wrap dummy multihash"),
        )
    }

    #[test]
    fn test_fallback_registration() {
        let manager = RecoveryManager::new(RecoveryConfig::default());

        let primary = dummy_peer();
        let fallback = "fallback_peer_id".to_string();
        let addr: SocketAddr = "127.0.0.1:8080".parse().expect("test: parse socket addr");

        manager.register_fallback(primary.clone(), fallback, addr, 1);

        let fallbacks = manager.get_fallbacks(&primary);
        assert_eq!(fallbacks.len(), 1);
        assert_eq!(fallbacks[0], addr);
    }

    #[test]
    fn test_provider_registration() {
        let manager = RecoveryManager::new(RecoveryConfig::default());
        let cid = dummy_cid();

        let providers = vec![
            "127.0.0.1:8080".parse().expect("test: parse socket addr"),
            "127.0.0.1:8081".parse().expect("test: parse socket addr"),
        ];

        manager.register_providers(cid, providers.clone());

        let provider = manager.get_next_provider(&cid);
        assert!(provider.is_some());
        assert!(providers.contains(&provider.expect("test: get next provider")));
    }

    #[test]
    fn test_provider_exhaustion() {
        let manager = RecoveryManager::new(RecoveryConfig::default());
        let cid = dummy_cid();

        let addr: SocketAddr = "127.0.0.1:8080".parse().expect("test: parse socket addr");
        manager.register_providers(cid, vec![addr]);

        let provider = manager.get_next_provider(&cid);
        assert_eq!(provider, Some(addr));

        manager.mark_provider_exhausted(&cid, addr);

        let provider = manager.get_next_provider(&cid);
        assert_eq!(provider, None);
    }

    #[test]
    fn test_degraded_mode() {
        let manager = RecoveryManager::new(RecoveryConfig::default());

        assert_eq!(manager.mode(), RecoveryMode::Normal);

        manager.enter_degraded_mode();
        assert_eq!(manager.mode(), RecoveryMode::Degraded);

        manager.exit_degraded_mode();
        assert_eq!(manager.mode(), RecoveryMode::Normal);
    }

    #[test]
    fn test_auto_degrade() {
        let config = RecoveryConfig {
            auto_degrade: true,
            min_peers_normal: 3,
            min_peers_degraded: 1,
            ..Default::default()
        };

        let manager = RecoveryManager::new(config);

        // Start normal
        assert_eq!(manager.mode(), RecoveryMode::Normal);

        // Drop below normal threshold
        manager.check_auto_degrade(2);
        assert_eq!(manager.mode(), RecoveryMode::Degraded);

        // Drop below degraded threshold
        manager.check_auto_degrade(0);
        assert_eq!(manager.mode(), RecoveryMode::Emergency);

        // Recover
        manager.check_auto_degrade(3);
        assert_eq!(manager.mode(), RecoveryMode::Normal);
    }

    #[test]
    fn test_fallback_success() {
        let manager = RecoveryManager::new(RecoveryConfig::default());

        let primary = dummy_peer();
        let fallback = "fallback_peer_id".to_string();
        let addr: SocketAddr = "127.0.0.1:8080".parse().expect("test: parse socket addr");

        manager.register_fallback(primary.clone(), fallback, addr, 1);
        manager.record_fallback_success(&primary, addr);

        let stats = manager.stats();
        assert_eq!(stats.fallback_uses, 1);
        assert_eq!(stats.successful_recoveries, 1);
    }
}
