//! QUIC Connection Migration for Mobile Support
//!
//! This module implements QUIC connection migration to handle network changes
//! seamlessly, particularly important for mobile devices switching between
//! WiFi, cellular, and other network interfaces.
//!
//! ## Features
//!
//! - Automatic detection of network interface changes
//! - Seamless connection migration without data loss
//! - State preservation during migration
//! - Retry logic for failed migrations
//! - Statistics tracking for migration events
//!
//! ## Use Cases
//!
//! - Mobile devices switching from WiFi to cellular
//! - Laptops moving between networks
//! - Devices with multiple network interfaces
//! - Network interface failures requiring fallback

use dashmap::DashMap;
use libp2p::{Multiaddr, PeerId};
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tracing::{debug, info, warn};

/// Errors that can occur during connection migration
#[derive(Debug, Error)]
pub enum MigrationError {
    #[error("No active connection to migrate")]
    NoActiveConnection,

    #[error("Migration already in progress")]
    MigrationInProgress,

    #[error("Migration failed: {0}")]
    MigrationFailed(String),

    #[error("Timeout during migration")]
    MigrationTimeout,

    #[error("Invalid migration state")]
    InvalidState,

    #[error("No suitable migration path available")]
    NoMigrationPath,
}

/// Connection migration configuration
#[derive(Debug, Clone)]
pub struct MigrationConfig {
    /// Enable automatic migration on network changes
    pub auto_migrate: bool,

    /// Maximum time to wait for migration to complete
    pub migration_timeout: Duration,

    /// Maximum number of migration retry attempts
    pub max_retry_attempts: usize,

    /// Backoff duration between retry attempts
    pub retry_backoff: Duration,

    /// Minimum time between migrations for the same connection
    pub migration_cooldown: Duration,

    /// Keep old path alive during migration (default: true for safety)
    pub keep_old_path: bool,

    /// Validate new path before closing old path
    pub validate_new_path: bool,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            auto_migrate: true,
            migration_timeout: Duration::from_secs(30),
            max_retry_attempts: 3,
            retry_backoff: Duration::from_secs(2),
            migration_cooldown: Duration::from_secs(10),
            keep_old_path: true,
            validate_new_path: true,
        }
    }
}

impl MigrationConfig {
    /// Create a mobile-optimized configuration
    ///
    /// Aggressive migration settings for mobile devices that frequently
    /// switch between WiFi and cellular networks.
    pub fn mobile() -> Self {
        Self {
            auto_migrate: true,
            migration_timeout: Duration::from_secs(15),
            max_retry_attempts: 5,
            retry_backoff: Duration::from_millis(500),
            migration_cooldown: Duration::from_secs(5),
            keep_old_path: true,
            validate_new_path: true,
        }
    }

    /// Create a conservative configuration
    ///
    /// More cautious migration settings for stable networks where
    /// migrations are rare.
    pub fn conservative() -> Self {
        Self {
            auto_migrate: false,
            migration_timeout: Duration::from_secs(60),
            max_retry_attempts: 2,
            retry_backoff: Duration::from_secs(5),
            migration_cooldown: Duration::from_secs(30),
            keep_old_path: true,
            validate_new_path: true,
        }
    }
}

/// State of a connection migration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationState {
    /// No migration in progress
    Idle,
    /// Migration initiated
    Initiated,
    /// Validating new network path
    Validating,
    /// Transferring connection state
    Migrating,
    /// Migration completed successfully
    Completed,
    /// Migration failed
    Failed,
}

/// Information about a connection migration attempt
#[derive(Debug, Clone)]
pub struct MigrationAttempt {
    /// Peer being migrated
    pub peer_id: PeerId,
    /// Old network address
    pub old_address: Multiaddr,
    /// New network address
    pub new_address: Multiaddr,
    /// Current migration state
    pub state: MigrationState,
    /// When the migration started
    pub started_at: Instant,
    /// Number of retry attempts
    pub retry_count: usize,
    /// Error message if failed
    pub error: Option<String>,
}

/// Connection migration statistics
#[derive(Debug, Clone, Default)]
pub struct MigrationStats {
    /// Total number of migration attempts
    pub total_attempts: usize,
    /// Number of successful migrations
    pub successful_migrations: usize,
    /// Number of failed migrations
    pub failed_migrations: usize,
    /// Number of migrations in progress
    pub in_progress: usize,
    /// Average migration duration (milliseconds)
    pub avg_duration_ms: u64,
    /// Total retry attempts
    pub total_retries: usize,
}

/// Manages QUIC connection migration
pub struct ConnectionMigrationManager {
    /// Configuration
    config: MigrationConfig,
    /// Active migration attempts
    active_migrations: Arc<DashMap<PeerId, MigrationAttempt>>,
    /// Last migration time per peer (for cooldown)
    last_migration: Arc<DashMap<PeerId, Instant>>,
    /// Migration statistics
    stats: Arc<RwLock<MigrationStats>>,
    /// Migration history (for tracking durations)
    migration_durations: Arc<RwLock<Vec<u64>>>,
}

impl ConnectionMigrationManager {
    /// Create a new connection migration manager
    pub fn new(config: MigrationConfig) -> Self {
        Self {
            config,
            active_migrations: Arc::new(DashMap::new()),
            last_migration: Arc::new(DashMap::new()),
            stats: Arc::new(RwLock::new(MigrationStats::default())),
            migration_durations: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Create with mobile-optimized configuration
    pub fn mobile() -> Self {
        Self::new(MigrationConfig::mobile())
    }

    /// Create with conservative configuration
    pub fn conservative() -> Self {
        Self::new(MigrationConfig::conservative())
    }

    /// Initiate migration for a peer connection
    ///
    /// # Arguments
    ///
    /// * `peer_id` - The peer to migrate
    /// * `old_address` - Current network address
    /// * `new_address` - New network address to migrate to
    ///
    /// # Returns
    ///
    /// Ok if migration was initiated, Err if migration cannot start
    pub fn initiate_migration(
        &self,
        peer_id: PeerId,
        old_address: Multiaddr,
        new_address: Multiaddr,
    ) -> Result<(), MigrationError> {
        // Check if migration is already in progress
        if self.active_migrations.contains_key(&peer_id) {
            return Err(MigrationError::MigrationInProgress);
        }

        // Check cooldown period
        if let Some(last) = self.last_migration.get(&peer_id) {
            if last.elapsed() < self.config.migration_cooldown {
                debug!(
                    "Migration cooldown active for peer {} ({:?} remaining)",
                    peer_id,
                    self.config.migration_cooldown - last.elapsed()
                );
                return Err(MigrationError::InvalidState);
            }
        }

        info!(
            "Initiating connection migration for peer {} from {} to {}",
            peer_id, old_address, new_address
        );

        // Create migration attempt
        let attempt = MigrationAttempt {
            peer_id,
            old_address,
            new_address,
            state: MigrationState::Initiated,
            started_at: Instant::now(),
            retry_count: 0,
            error: None,
        };

        // Record the attempt
        self.active_migrations.insert(peer_id, attempt);

        // Update statistics
        let mut stats = self.stats.write();
        stats.total_attempts += 1;
        stats.in_progress += 1;

        Ok(())
    }

    /// Update the state of an ongoing migration
    pub fn update_migration_state(
        &self,
        peer_id: &PeerId,
        new_state: MigrationState,
    ) -> Result<(), MigrationError> {
        if let Some(mut attempt) = self.active_migrations.get_mut(peer_id) {
            debug!(
                "Migration state change for peer {}: {:?} -> {:?}",
                peer_id, attempt.state, new_state
            );
            attempt.state = new_state;
            Ok(())
        } else {
            Err(MigrationError::NoActiveConnection)
        }
    }

    /// Mark a migration as completed successfully
    pub fn complete_migration(&self, peer_id: &PeerId) -> Result<(), MigrationError> {
        if let Some((_, attempt)) = self.active_migrations.remove(peer_id) {
            let duration_ms = attempt.started_at.elapsed().as_millis() as u64;

            info!(
                "Migration completed for peer {} in {} ms",
                peer_id, duration_ms
            );

            // Update last migration time
            self.last_migration.insert(*peer_id, Instant::now());

            // Update statistics
            {
                let mut stats = self.stats.write();
                stats.successful_migrations += 1;
                stats.in_progress = stats.in_progress.saturating_sub(1);
                stats.total_retries += attempt.retry_count;
            }

            // Record duration for average calculation
            {
                let mut durations = self.migration_durations.write();
                durations.push(duration_ms);

                // Keep only last 100 durations to prevent unbounded growth
                if durations.len() > 100 {
                    durations.remove(0);
                }

                // Update average
                let avg = durations.iter().sum::<u64>() / durations.len() as u64;
                self.stats.write().avg_duration_ms = avg;
            }

            Ok(())
        } else {
            Err(MigrationError::NoActiveConnection)
        }
    }

    /// Mark a migration as failed
    pub fn fail_migration(&self, peer_id: &PeerId, error: String) -> Result<(), MigrationError> {
        if let Some((_, mut attempt)) = self.active_migrations.remove(peer_id) {
            warn!("Migration failed for peer {}: {}", peer_id, error);

            attempt.error = Some(error);
            attempt.state = MigrationState::Failed;

            // Update statistics
            let mut stats = self.stats.write();
            stats.failed_migrations += 1;
            stats.in_progress = stats.in_progress.saturating_sub(1);
            stats.total_retries += attempt.retry_count;

            Ok(())
        } else {
            Err(MigrationError::NoActiveConnection)
        }
    }

    /// Retry a failed migration
    pub fn retry_migration(&self, peer_id: &PeerId) -> Result<(), MigrationError> {
        if let Some(mut attempt) = self.active_migrations.get_mut(peer_id) {
            if attempt.retry_count >= self.config.max_retry_attempts {
                return Err(MigrationError::MigrationFailed(
                    "Max retry attempts reached".to_string(),
                ));
            }

            attempt.retry_count += 1;
            attempt.state = MigrationState::Initiated;
            attempt.error = None;

            info!(
                "Retrying migration for peer {} (attempt {})",
                peer_id,
                attempt.retry_count + 1
            );

            Ok(())
        } else {
            Err(MigrationError::NoActiveConnection)
        }
    }

    /// Check if a migration is in progress for a peer
    pub fn is_migrating(&self, peer_id: &PeerId) -> bool {
        self.active_migrations.contains_key(peer_id)
    }

    /// Get the current migration state for a peer
    pub fn get_migration_state(&self, peer_id: &PeerId) -> Option<MigrationState> {
        self.active_migrations
            .get(peer_id)
            .map(|attempt| attempt.state)
    }

    /// Get all active migration attempts
    pub fn get_active_migrations(&self) -> Vec<MigrationAttempt> {
        self.active_migrations
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get migration statistics
    pub fn stats(&self) -> MigrationStats {
        self.stats.read().clone()
    }

    /// Check if a peer can be migrated (respects cooldown)
    pub fn can_migrate(&self, peer_id: &PeerId) -> bool {
        if self.active_migrations.contains_key(peer_id) {
            return false;
        }

        if let Some(last) = self.last_migration.get(peer_id) {
            last.elapsed() >= self.config.migration_cooldown
        } else {
            true
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &MigrationConfig {
        &self.config
    }

    /// Clean up timed-out migrations
    pub fn cleanup_timeouts(&self) {
        let timeout = self.config.migration_timeout;
        let mut timed_out = Vec::new();

        for entry in self.active_migrations.iter() {
            if entry.value().started_at.elapsed() > timeout {
                timed_out.push(*entry.key());
            }
        }

        for peer_id in timed_out {
            self.fail_migration(&peer_id, "Migration timeout".to_string())
                .ok();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn test_peer_id() -> PeerId {
        PeerId::random()
    }

    fn test_addr() -> Multiaddr {
        Multiaddr::from_str("/ip4/127.0.0.1/tcp/4001").expect("test: valid multiaddr literal")
    }

    fn test_addr2() -> Multiaddr {
        Multiaddr::from_str("/ip4/192.168.1.1/tcp/4001").expect("test: valid multiaddr literal")
    }

    #[test]
    fn test_migration_config_default() {
        let config = MigrationConfig::default();
        assert!(config.auto_migrate);
        assert!(config.keep_old_path);
        assert!(config.validate_new_path);
    }

    #[test]
    fn test_migration_config_mobile() {
        let config = MigrationConfig::mobile();
        assert!(config.auto_migrate);
        assert_eq!(config.max_retry_attempts, 5);
        assert_eq!(config.migration_cooldown, Duration::from_secs(5));
    }

    #[test]
    fn test_migration_config_conservative() {
        let config = MigrationConfig::conservative();
        assert!(!config.auto_migrate);
        assert_eq!(config.max_retry_attempts, 2);
        assert_eq!(config.migration_cooldown, Duration::from_secs(30));
    }

    #[test]
    fn test_initiate_migration() {
        let manager = ConnectionMigrationManager::new(MigrationConfig::default());
        let peer = test_peer_id();
        let old_addr = test_addr();
        let new_addr = test_addr2();

        let result = manager.initiate_migration(peer, old_addr, new_addr);
        assert!(result.is_ok());

        let stats = manager.stats();
        assert_eq!(stats.total_attempts, 1);
        assert_eq!(stats.in_progress, 1);
    }

    #[test]
    fn test_migration_in_progress_error() {
        let manager = ConnectionMigrationManager::new(MigrationConfig::default());
        let peer = test_peer_id();
        let old_addr = test_addr();
        let new_addr = test_addr2();

        manager
            .initiate_migration(peer, old_addr.clone(), new_addr.clone())
            .expect("test: first migration initiation should succeed");

        let result = manager.initiate_migration(peer, old_addr, new_addr);
        assert!(matches!(result, Err(MigrationError::MigrationInProgress)));
    }

    #[test]
    fn test_complete_migration() {
        let manager = ConnectionMigrationManager::new(MigrationConfig::default());
        let peer = test_peer_id();

        manager
            .initiate_migration(peer, test_addr(), test_addr2())
            .expect("test: migration initiation should succeed");
        let result = manager.complete_migration(&peer);

        assert!(result.is_ok());

        let stats = manager.stats();
        assert_eq!(stats.successful_migrations, 1);
        assert_eq!(stats.in_progress, 0);
    }

    #[test]
    fn test_fail_migration() {
        let manager = ConnectionMigrationManager::new(MigrationConfig::default());
        let peer = test_peer_id();

        manager
            .initiate_migration(peer, test_addr(), test_addr2())
            .expect("test: migration initiation should succeed");
        let result = manager.fail_migration(&peer, "Test error".to_string());

        assert!(result.is_ok());

        let stats = manager.stats();
        assert_eq!(stats.failed_migrations, 1);
        assert_eq!(stats.in_progress, 0);
    }

    #[test]
    fn test_retry_migration() {
        let manager = ConnectionMigrationManager::new(MigrationConfig::default());
        let peer = test_peer_id();

        manager
            .initiate_migration(peer, test_addr(), test_addr2())
            .expect("test: migration initiation should succeed");

        let result = manager.retry_migration(&peer);
        assert!(result.is_ok());

        let attempt = manager
            .active_migrations
            .get(&peer)
            .expect("test: active migration entry should exist");
        assert_eq!(attempt.retry_count, 1);
    }

    #[test]
    fn test_retry_limit() {
        let config = MigrationConfig {
            max_retry_attempts: 2,
            ..Default::default()
        };
        let manager = ConnectionMigrationManager::new(config);
        let peer = test_peer_id();

        manager
            .initiate_migration(peer, test_addr(), test_addr2())
            .expect("test: migration initiation should succeed");

        // First retry should succeed
        assert!(manager.retry_migration(&peer).is_ok());
        // Second retry should succeed
        assert!(manager.retry_migration(&peer).is_ok());
        // Third retry should fail (max reached)
        assert!(matches!(
            manager.retry_migration(&peer),
            Err(MigrationError::MigrationFailed(_))
        ));
    }

    #[test]
    fn test_is_migrating() {
        let manager = ConnectionMigrationManager::new(MigrationConfig::default());
        let peer = test_peer_id();

        assert!(!manager.is_migrating(&peer));

        manager
            .initiate_migration(peer, test_addr(), test_addr2())
            .expect("test: migration initiation should succeed");
        assert!(manager.is_migrating(&peer));

        manager
            .complete_migration(&peer)
            .expect("test: complete_migration should succeed");
        assert!(!manager.is_migrating(&peer));
    }

    #[test]
    fn test_migration_state_updates() {
        let manager = ConnectionMigrationManager::new(MigrationConfig::default());
        let peer = test_peer_id();

        manager
            .initiate_migration(peer, test_addr(), test_addr2())
            .expect("test: migration initiation should succeed");

        assert_eq!(
            manager.get_migration_state(&peer),
            Some(MigrationState::Initiated)
        );

        manager
            .update_migration_state(&peer, MigrationState::Validating)
            .expect("test: update_migration_state to Validating should succeed");
        assert_eq!(
            manager.get_migration_state(&peer),
            Some(MigrationState::Validating)
        );

        manager
            .update_migration_state(&peer, MigrationState::Migrating)
            .expect("test: update_migration_state to Migrating should succeed");
        assert_eq!(
            manager.get_migration_state(&peer),
            Some(MigrationState::Migrating)
        );
    }

    #[test]
    fn test_can_migrate() {
        let config = MigrationConfig {
            migration_cooldown: Duration::from_millis(100),
            ..Default::default()
        };
        let manager = ConnectionMigrationManager::new(config);
        let peer = test_peer_id();

        assert!(manager.can_migrate(&peer));

        manager
            .initiate_migration(peer, test_addr(), test_addr2())
            .expect("test: migration initiation should succeed");
        assert!(!manager.can_migrate(&peer));

        manager
            .complete_migration(&peer)
            .expect("test: complete_migration should succeed");
        assert!(!manager.can_migrate(&peer)); // Cooldown active

        std::thread::sleep(Duration::from_millis(150));
        assert!(manager.can_migrate(&peer)); // Cooldown expired
    }

    #[test]
    fn test_get_active_migrations() {
        let manager = ConnectionMigrationManager::new(MigrationConfig::default());
        let peer1 = test_peer_id();
        let peer2 = test_peer_id();

        manager
            .initiate_migration(peer1, test_addr(), test_addr2())
            .expect("test: peer1 migration initiation should succeed");
        manager
            .initiate_migration(peer2, test_addr(), test_addr2())
            .expect("test: peer2 migration initiation should succeed");

        let active = manager.get_active_migrations();
        assert_eq!(active.len(), 2);
    }

    #[test]
    fn test_average_duration_calculation() {
        let manager = ConnectionMigrationManager::new(MigrationConfig::default());
        let peer1 = test_peer_id();
        let peer2 = test_peer_id();

        manager
            .initiate_migration(peer1, test_addr(), test_addr2())
            .expect("test: peer1 migration initiation should succeed");
        std::thread::sleep(Duration::from_millis(10));
        manager
            .complete_migration(&peer1)
            .expect("test: peer1 complete_migration should succeed");

        manager
            .initiate_migration(peer2, test_addr(), test_addr2())
            .expect("test: peer2 migration initiation should succeed");
        std::thread::sleep(Duration::from_millis(10));
        manager
            .complete_migration(&peer2)
            .expect("test: peer2 complete_migration should succeed");

        let stats = manager.stats();
        assert!(stats.avg_duration_ms > 0);
    }
}
