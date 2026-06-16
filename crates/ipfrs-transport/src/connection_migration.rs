//! Connection migration for handling network changes
//!
//! This module provides support for migrating connections when network
//! conditions change, such as switching between WiFi and cellular,
//! or when IP addresses change due to DHCP.
//!
//! # Example
//!
//! ```
//! use ipfrs_transport::{ConnectionMigration, MigrationConfig};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = MigrationConfig::default();
//! let migration = ConnectionMigration::new(config);
//! # Ok(())
//! # }
//! ```

use dashmap::DashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Configuration for connection migration
#[derive(Debug, Clone)]
pub struct MigrationConfig {
    /// Enable automatic migration on network changes
    pub enable_auto_migration: bool,
    /// Probe interval for detecting network changes
    pub probe_interval: Duration,
    /// Timeout for completing migration
    pub migration_timeout: Duration,
    /// Number of retries for failed migrations
    pub max_retries: usize,
    /// Grace period before closing old connection
    pub grace_period: Duration,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            enable_auto_migration: true,
            probe_interval: Duration::from_secs(5),
            migration_timeout: Duration::from_secs(30),
            max_retries: 3,
            grace_period: Duration::from_secs(10),
        }
    }
}

/// Migration state for a connection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationState {
    /// No migration in progress
    Stable,
    /// Network change detected, preparing to migrate
    Preparing,
    /// Migration in progress
    Migrating,
    /// Migration completed successfully
    Completed,
    /// Migration failed
    Failed,
}

/// Migration event types
#[derive(Debug, Clone)]
pub enum MigrationEvent {
    /// Network change detected
    NetworkChangeDetected {
        old_addr: SocketAddr,
        new_addr: SocketAddr,
    },
    /// Migration started
    MigrationStarted {
        connection_id: String,
        from_addr: SocketAddr,
        to_addr: SocketAddr,
    },
    /// Migration completed successfully
    MigrationCompleted {
        connection_id: String,
        new_addr: SocketAddr,
        duration: Duration,
    },
    /// Migration failed
    MigrationFailed {
        connection_id: String,
        reason: String,
        retry_count: usize,
    },
}

/// Statistics for connection migration
#[derive(Debug, Clone, Default)]
pub struct MigrationStats {
    /// Total number of migrations attempted
    pub total_migrations: u64,
    /// Number of successful migrations
    pub successful_migrations: u64,
    /// Number of failed migrations
    pub failed_migrations: u64,
    /// Average migration duration
    pub avg_migration_duration: Duration,
    /// Maximum migration duration
    pub max_migration_duration: Duration,
}

impl MigrationStats {
    /// Calculate success rate (0.0 to 1.0)
    pub fn success_rate(&self) -> f64 {
        if self.total_migrations == 0 {
            return 0.0;
        }
        self.successful_migrations as f64 / self.total_migrations as f64
    }
}

/// Migration record for a connection
#[derive(Debug)]
struct MigrationRecord {
    /// Current state
    state: MigrationState,
    /// Old address
    #[allow(dead_code)]
    old_addr: SocketAddr,
    /// New address
    new_addr: SocketAddr,
    /// When migration started
    started_at: Instant,
    /// Retry count
    retry_count: usize,
}

/// Type alias for event callback function
type EventCallback = Box<dyn Fn(MigrationEvent) + Send + Sync>;

/// Connection migration manager
pub struct ConnectionMigration {
    /// Configuration
    config: MigrationConfig,
    /// Active migrations indexed by connection ID
    migrations: Arc<DashMap<String, MigrationRecord>>,
    /// Statistics
    stats: Arc<RwLock<MigrationStats>>,
    /// Event callbacks
    event_callbacks: Arc<RwLock<Vec<EventCallback>>>,
}

impl ConnectionMigration {
    /// Create a new connection migration manager
    pub fn new(config: MigrationConfig) -> Self {
        Self {
            config,
            migrations: Arc::new(DashMap::new()),
            stats: Arc::new(RwLock::new(MigrationStats::default())),
            event_callbacks: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a migration event callback
    pub async fn on_event<F>(&self, callback: F)
    where
        F: Fn(MigrationEvent) + Send + Sync + 'static,
    {
        let mut callbacks = self.event_callbacks.write().await;
        callbacks.push(Box::new(callback));
    }

    /// Emit a migration event
    async fn emit_event(&self, event: MigrationEvent) {
        let callbacks = self.event_callbacks.read().await;
        for callback in callbacks.iter() {
            callback(event.clone());
        }
    }

    /// Start migration for a connection
    pub async fn start_migration(
        &self,
        connection_id: String,
        old_addr: SocketAddr,
        new_addr: SocketAddr,
    ) -> Result<(), String> {
        // Check if migration is already in progress
        if self.migrations.contains_key(&connection_id) {
            return Err("Migration already in progress".to_string());
        }

        // Create migration record
        self.migrations.insert(
            connection_id.clone(),
            MigrationRecord {
                state: MigrationState::Preparing,
                old_addr,
                new_addr,
                started_at: Instant::now(),
                retry_count: 0,
            },
        );

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_migrations += 1;
        drop(stats);

        // Emit event
        self.emit_event(MigrationEvent::MigrationStarted {
            connection_id: connection_id.clone(),
            from_addr: old_addr,
            to_addr: new_addr,
        })
        .await;

        // Update state to migrating
        if let Some(mut record) = self.migrations.get_mut(&connection_id) {
            record.state = MigrationState::Migrating;
        }

        Ok(())
    }

    /// Complete a migration successfully
    pub async fn complete_migration(&self, connection_id: &str) -> Result<(), String> {
        let (new_addr, duration) = {
            if let Some(mut record) = self.migrations.get_mut(connection_id) {
                record.state = MigrationState::Completed;
                let duration = record.started_at.elapsed();
                (record.new_addr, duration)
            } else {
                return Err("Migration not found".to_string());
            }
        };

        // Update stats
        let mut stats = self.stats.write().await;
        stats.successful_migrations += 1;

        // Update average duration
        let total_duration = stats.avg_migration_duration.as_millis() as u64
            * (stats.successful_migrations - 1)
            + duration.as_millis() as u64;
        stats.avg_migration_duration =
            Duration::from_millis(total_duration / stats.successful_migrations);

        // Update max duration
        if duration > stats.max_migration_duration {
            stats.max_migration_duration = duration;
        }
        drop(stats);

        // Emit event
        self.emit_event(MigrationEvent::MigrationCompleted {
            connection_id: connection_id.to_string(),
            new_addr,
            duration,
        })
        .await;

        // Remove migration record after grace period
        let connection_id = connection_id.to_string();
        let migrations = self.migrations.clone();
        let grace_period = self.config.grace_period;
        tokio::spawn(async move {
            tokio::time::sleep(grace_period).await;
            migrations.remove(&connection_id);
        });

        Ok(())
    }

    /// Fail a migration (with possible retry)
    pub async fn fail_migration(&self, connection_id: &str, reason: String) -> Result<(), String> {
        let should_retry = {
            if let Some(mut record) = self.migrations.get_mut(connection_id) {
                record.retry_count += 1;
                record.retry_count < self.config.max_retries
            } else {
                return Err("Migration not found".to_string());
            }
        };

        if should_retry {
            // Reset state for retry
            if let Some(mut record) = self.migrations.get_mut(connection_id) {
                record.state = MigrationState::Preparing;
            }
            Ok(())
        } else {
            // Max retries exceeded, mark as failed
            let retry_count = {
                if let Some(mut record) = self.migrations.get_mut(connection_id) {
                    record.state = MigrationState::Failed;
                    record.retry_count
                } else {
                    0
                }
            };

            // Update stats
            let mut stats = self.stats.write().await;
            stats.failed_migrations += 1;
            drop(stats);

            // Emit event
            self.emit_event(MigrationEvent::MigrationFailed {
                connection_id: connection_id.to_string(),
                reason,
                retry_count,
            })
            .await;

            // Remove failed migration
            self.migrations.remove(connection_id);

            Err("Migration failed after retries".to_string())
        }
    }

    /// Get migration state for a connection
    pub fn get_state(&self, connection_id: &str) -> Option<MigrationState> {
        self.migrations
            .get(connection_id)
            .map(|record| record.state)
    }

    /// Check if a connection is currently migrating
    pub fn is_migrating(&self, connection_id: &str) -> bool {
        matches!(
            self.get_state(connection_id),
            Some(MigrationState::Preparing) | Some(MigrationState::Migrating)
        )
    }

    /// Get current statistics
    pub async fn stats(&self) -> MigrationStats {
        self.stats.read().await.clone()
    }

    /// Reset statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = MigrationStats::default();
    }

    /// Clean up timed-out migrations
    pub async fn cleanup_timeouts(&self) {
        let timeout = self.config.migration_timeout;
        let failed_ids: Vec<String> = self
            .migrations
            .iter()
            .filter(|entry| entry.started_at.elapsed() > timeout)
            .map(|entry| entry.key().clone())
            .collect();

        for id in failed_ids {
            let _ = self
                .fail_migration(&id, "Migration timeout".to_string())
                .await;
        }
    }

    /// Get number of active migrations
    pub fn active_migrations(&self) -> usize {
        self.migrations.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_connection_migration_creation() {
        let config = MigrationConfig::default();
        let migration = ConnectionMigration::new(config);
        assert_eq!(migration.active_migrations(), 0);
    }

    #[tokio::test]
    async fn test_start_migration() {
        let migration = ConnectionMigration::new(MigrationConfig::default());
        let old_addr: SocketAddr = "127.0.0.1:8000".parse().expect("test: valid socket addr");
        let new_addr: SocketAddr = "127.0.0.1:8001".parse().expect("test: valid socket addr");

        let result = migration
            .start_migration("conn1".to_string(), old_addr, new_addr)
            .await;
        assert!(result.is_ok());
        assert_eq!(migration.active_migrations(), 1);

        let state = migration.get_state("conn1");
        assert_eq!(state, Some(MigrationState::Migrating));
    }

    #[tokio::test]
    async fn test_complete_migration() {
        let migration = ConnectionMigration::new(MigrationConfig::default());
        let old_addr: SocketAddr = "127.0.0.1:8000".parse().expect("test: valid socket addr");
        let new_addr: SocketAddr = "127.0.0.1:8001".parse().expect("test: valid socket addr");

        migration
            .start_migration("conn1".to_string(), old_addr, new_addr)
            .await
            .expect("test: start migration should succeed");

        let result = migration.complete_migration("conn1").await;
        assert!(result.is_ok());

        let stats = migration.stats().await;
        assert_eq!(stats.successful_migrations, 1);
        assert_eq!(stats.success_rate(), 1.0);
    }

    #[tokio::test]
    async fn test_fail_migration_with_retry() {
        let config = MigrationConfig {
            max_retries: 3,
            ..Default::default()
        };
        let migration = ConnectionMigration::new(config);
        let old_addr: SocketAddr = "127.0.0.1:8000".parse().expect("test: valid socket addr");
        let new_addr: SocketAddr = "127.0.0.1:8001".parse().expect("test: valid socket addr");

        migration
            .start_migration("conn1".to_string(), old_addr, new_addr)
            .await
            .expect("test: start migration should succeed");

        // First failure should allow retry
        let result = migration
            .fail_migration("conn1", "Network error".to_string())
            .await;
        assert!(result.is_ok());

        // State should be reset for retry
        let state = migration.get_state("conn1");
        assert_eq!(state, Some(MigrationState::Preparing));
    }

    #[tokio::test]
    async fn test_fail_migration_max_retries() {
        let config = MigrationConfig {
            max_retries: 2,
            ..Default::default()
        };
        let migration = ConnectionMigration::new(config);
        let old_addr: SocketAddr = "127.0.0.1:8000".parse().expect("test: valid socket addr");
        let new_addr: SocketAddr = "127.0.0.1:8001".parse().expect("test: valid socket addr");

        migration
            .start_migration("conn1".to_string(), old_addr, new_addr)
            .await
            .expect("test: start migration should succeed");

        // Fail twice (max retries = 2)
        migration
            .fail_migration("conn1", "Error 1".to_string())
            .await
            .expect("test: fail migration should succeed");
        let result = migration
            .fail_migration("conn1", "Error 2".to_string())
            .await;

        assert!(result.is_err());

        let stats = migration.stats().await;
        assert_eq!(stats.failed_migrations, 1);
    }

    #[tokio::test]
    async fn test_is_migrating() {
        let migration = ConnectionMigration::new(MigrationConfig::default());
        let old_addr: SocketAddr = "127.0.0.1:8000".parse().expect("test: valid socket addr");
        let new_addr: SocketAddr = "127.0.0.1:8001".parse().expect("test: valid socket addr");

        assert!(!migration.is_migrating("conn1"));

        migration
            .start_migration("conn1".to_string(), old_addr, new_addr)
            .await
            .expect("test: start migration should succeed");

        assert!(migration.is_migrating("conn1"));

        migration
            .complete_migration("conn1")
            .await
            .expect("test: complete migration should succeed");

        // After grace period, it should be removed, but state is Completed
        let state = migration.get_state("conn1");
        if state.is_some() {
            assert_eq!(state, Some(MigrationState::Completed));
        }
    }

    #[tokio::test]
    async fn test_duplicate_migration_rejected() {
        let migration = ConnectionMigration::new(MigrationConfig::default());
        let old_addr: SocketAddr = "127.0.0.1:8000".parse().expect("test: valid socket addr");
        let new_addr: SocketAddr = "127.0.0.1:8001".parse().expect("test: valid socket addr");

        migration
            .start_migration("conn1".to_string(), old_addr, new_addr)
            .await
            .expect("test: start migration should succeed");

        // Try to start another migration for the same connection
        let result = migration
            .start_migration("conn1".to_string(), old_addr, new_addr)
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_stats_calculation() {
        let config = MigrationConfig {
            max_retries: 2,
            ..Default::default()
        };
        let migration = ConnectionMigration::new(config);
        let old_addr: SocketAddr = "127.0.0.1:8000".parse().expect("test: valid socket addr");
        let new_addr: SocketAddr = "127.0.0.1:8001".parse().expect("test: valid socket addr");

        // Successful migration
        migration
            .start_migration("conn1".to_string(), old_addr, new_addr)
            .await
            .expect("test: start migration should succeed");
        migration
            .complete_migration("conn1")
            .await
            .expect("test: complete migration should succeed");

        // Failed migration (max_retries = 2, so need to fail 2 times)
        migration
            .start_migration("conn2".to_string(), old_addr, new_addr)
            .await
            .expect("test: start migration should succeed");
        migration
            .fail_migration("conn2", "Error".to_string())
            .await
            .ok();
        migration
            .fail_migration("conn2", "Error".to_string())
            .await
            .ok();

        let stats = migration.stats().await;
        assert_eq!(stats.total_migrations, 2);
        assert_eq!(stats.successful_migrations, 1);
        assert_eq!(stats.failed_migrations, 1);
        assert_eq!(stats.success_rate(), 0.5);
    }

    #[tokio::test]
    async fn test_reset_stats() {
        let migration = ConnectionMigration::new(MigrationConfig::default());
        let old_addr: SocketAddr = "127.0.0.1:8000".parse().expect("test: valid socket addr");
        let new_addr: SocketAddr = "127.0.0.1:8001".parse().expect("test: valid socket addr");

        migration
            .start_migration("conn1".to_string(), old_addr, new_addr)
            .await
            .expect("test: start migration should succeed");

        let stats = migration.stats().await;
        assert!(stats.total_migrations > 0);

        migration.reset_stats().await;
        let stats = migration.stats().await;
        assert_eq!(stats.total_migrations, 0);
    }

    #[tokio::test]
    async fn test_event_callbacks() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let migration = ConnectionMigration::new(MigrationConfig::default());
        let event_count = Arc::new(AtomicUsize::new(0));
        let count_clone = event_count.clone();

        migration
            .on_event(move |_event| {
                count_clone.fetch_add(1, Ordering::SeqCst);
            })
            .await;

        let old_addr: SocketAddr = "127.0.0.1:8000".parse().expect("test: valid socket addr");
        let new_addr: SocketAddr = "127.0.0.1:8001".parse().expect("test: valid socket addr");

        migration
            .start_migration("conn1".to_string(), old_addr, new_addr)
            .await
            .expect("test: start migration should succeed");
        migration
            .complete_migration("conn1")
            .await
            .expect("test: complete migration should succeed");

        // Should have received 2 events: MigrationStarted and MigrationCompleted
        assert_eq!(event_count.load(Ordering::SeqCst), 2);
    }
}
