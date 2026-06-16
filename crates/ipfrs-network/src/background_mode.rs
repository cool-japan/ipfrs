//! Background mode support with pause/resume functionality
//!
//! This module provides functionality to pause and resume network operations
//! when an application goes to the background (e.g., on mobile devices).
//! This helps save battery and network resources while the app is not active.

use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tracing::{debug, info};

/// Background mode state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackgroundState {
    /// Network is running normally (foreground)
    Active,
    /// Network is paused (background)
    Paused,
    /// Network is in the process of pausing
    Pausing,
    /// Network is in the process of resuming
    Resuming,
}

/// Configuration for background mode behavior
#[derive(Debug, Clone)]
pub struct BackgroundModeConfig {
    /// Whether to pause DHT queries in background
    pub pause_dht_queries: bool,
    /// Whether to pause provider announcements in background
    pub pause_provider_announcements: bool,
    /// Whether to close idle connections when entering background
    pub close_idle_connections: bool,
    /// Minimum connection age to be considered for closing (when close_idle_connections is true)
    pub idle_connection_threshold: Duration,
    /// Whether to maintain a minimal set of connections in background
    pub keep_minimal_connections: bool,
    /// Number of connections to keep when in background mode
    pub minimal_connection_count: usize,
    /// Whether to reduce DHT query frequency in background
    pub reduce_dht_frequency: bool,
    /// DHT query interval when in background (if reduce_dht_frequency is true)
    pub background_dht_interval: Duration,
}

impl Default for BackgroundModeConfig {
    fn default() -> Self {
        Self {
            pause_dht_queries: false,
            pause_provider_announcements: true,
            close_idle_connections: true,
            idle_connection_threshold: Duration::from_secs(300), // 5 minutes
            keep_minimal_connections: true,
            minimal_connection_count: 3,
            reduce_dht_frequency: true,
            background_dht_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

impl BackgroundModeConfig {
    /// Mobile configuration - aggressive power saving
    pub fn mobile() -> Self {
        Self {
            pause_dht_queries: true, // Pause all DHT queries
            pause_provider_announcements: true,
            close_idle_connections: true,
            idle_connection_threshold: Duration::from_secs(60), // 1 minute
            keep_minimal_connections: true,
            minimal_connection_count: 2, // Keep only 2 connections
            reduce_dht_frequency: true,
            background_dht_interval: Duration::from_secs(600), // 10 minutes
        }
    }

    /// Server configuration - minimal impact
    pub fn server() -> Self {
        Self {
            pause_dht_queries: false,
            pause_provider_announcements: false,
            close_idle_connections: false,
            idle_connection_threshold: Duration::from_secs(3600), // 1 hour
            keep_minimal_connections: false,
            minimal_connection_count: 0,
            reduce_dht_frequency: false,
            background_dht_interval: Duration::from_secs(60),
        }
    }

    /// Balanced configuration
    pub fn balanced() -> Self {
        Self::default()
    }
}

/// Background mode manager
pub struct BackgroundModeManager {
    config: BackgroundModeConfig,
    state: Arc<RwLock<BackgroundState>>,
    stats: Arc<RwLock<BackgroundModeStats>>,
    /// Time when last state transition occurred
    last_transition: Arc<RwLock<Option<Instant>>>,
    /// Time spent in each state
    time_in_active: Arc<RwLock<Duration>>,
    time_in_paused: Arc<RwLock<Duration>>,
}

/// Background mode statistics
#[derive(Debug, Clone, Default)]
pub struct BackgroundModeStats {
    /// Number of times the network was paused
    pub pause_count: usize,
    /// Number of times the network was resumed
    pub resume_count: usize,
    /// Total time spent in background mode
    pub total_background_time: Duration,
    /// Total time spent in foreground mode
    pub total_foreground_time: Duration,
    /// Number of connections closed when entering background
    pub connections_closed_on_pause: usize,
    /// Number of DHT queries skipped in background
    pub dht_queries_skipped: usize,
}

/// Errors that can occur during background mode operations
#[derive(Debug, Error)]
pub enum BackgroundModeError {
    #[error("Invalid state transition from {from:?} to {to:?}")]
    InvalidStateTransition {
        from: BackgroundState,
        to: BackgroundState,
    },

    #[error("Operation not allowed in current state: {0:?}")]
    OperationNotAllowed(BackgroundState),
}

impl BackgroundModeManager {
    /// Create a new background mode manager
    pub fn new(config: BackgroundModeConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(BackgroundState::Active)),
            stats: Arc::new(RwLock::new(BackgroundModeStats::default())),
            last_transition: Arc::new(RwLock::new(Some(Instant::now()))),
            time_in_active: Arc::new(RwLock::new(Duration::ZERO)),
            time_in_paused: Arc::new(RwLock::new(Duration::ZERO)),
        }
    }

    /// Get current background state
    pub fn state(&self) -> BackgroundState {
        *self.state.read()
    }

    /// Check if the network is paused
    pub fn is_paused(&self) -> bool {
        matches!(self.state(), BackgroundState::Paused)
    }

    /// Check if the network is active
    pub fn is_active(&self) -> bool {
        matches!(self.state(), BackgroundState::Active)
    }

    /// Pause network operations (enter background mode)
    pub fn pause(&self) -> Result<(), BackgroundModeError> {
        let current_state = *self.state.read();

        match current_state {
            BackgroundState::Active => {
                info!("Pausing network for background mode");
                *self.state.write() = BackgroundState::Pausing;

                // Update time tracking
                self.update_time_tracking(current_state);

                // Perform pause operations
                self.perform_pause_operations();

                *self.state.write() = BackgroundState::Paused;
                *self.last_transition.write() = Some(Instant::now());

                let mut stats = self.stats.write();
                stats.pause_count += 1;

                debug!("Network paused successfully");
                Ok(())
            }
            BackgroundState::Paused => {
                // Already paused, no-op
                Ok(())
            }
            state => Err(BackgroundModeError::InvalidStateTransition {
                from: state,
                to: BackgroundState::Paused,
            }),
        }
    }

    /// Resume network operations (enter foreground mode)
    pub fn resume(&self) -> Result<(), BackgroundModeError> {
        let current_state = *self.state.read();

        match current_state {
            BackgroundState::Paused => {
                info!("Resuming network from background mode");
                *self.state.write() = BackgroundState::Resuming;

                // Update time tracking
                self.update_time_tracking(current_state);

                // Perform resume operations
                self.perform_resume_operations();

                *self.state.write() = BackgroundState::Active;
                *self.last_transition.write() = Some(Instant::now());

                let mut stats = self.stats.write();
                stats.resume_count += 1;

                debug!("Network resumed successfully");
                Ok(())
            }
            BackgroundState::Active => {
                // Already active, no-op
                Ok(())
            }
            state => Err(BackgroundModeError::InvalidStateTransition {
                from: state,
                to: BackgroundState::Active,
            }),
        }
    }

    /// Perform operations when entering background mode
    fn perform_pause_operations(&self) {
        // These are hooks that the NetworkNode can use
        debug!(
            "Background mode config: pause_dht={}, pause_announcements={}, close_idle={}",
            self.config.pause_dht_queries,
            self.config.pause_provider_announcements,
            self.config.close_idle_connections
        );

        // Actual implementation would be in NetworkNode
        // which would call these methods and perform the operations
    }

    /// Perform operations when resuming from background mode
    fn perform_resume_operations(&self) {
        debug!("Resuming network operations");

        // Actual implementation would be in NetworkNode
        // which would call these methods and perform the operations
    }

    /// Update time tracking when state changes
    fn update_time_tracking(&self, old_state: BackgroundState) {
        if let Some(last_transition) = *self.last_transition.read() {
            let elapsed = last_transition.elapsed();

            match old_state {
                BackgroundState::Active => {
                    *self.time_in_active.write() += elapsed;
                    let mut stats = self.stats.write();
                    stats.total_foreground_time += elapsed;
                }
                BackgroundState::Paused => {
                    *self.time_in_paused.write() += elapsed;
                    let mut stats = self.stats.write();
                    stats.total_background_time += elapsed;
                }
                _ => {}
            }
        }
    }

    /// Check if a DHT query should be allowed in current state
    pub fn should_allow_dht_query(&self) -> bool {
        match self.state() {
            BackgroundState::Active | BackgroundState::Resuming => true,
            BackgroundState::Paused | BackgroundState::Pausing => !self.config.pause_dht_queries,
        }
    }

    /// Check if provider announcements should be allowed in current state
    pub fn should_allow_provider_announcements(&self) -> bool {
        match self.state() {
            BackgroundState::Active | BackgroundState::Resuming => true,
            BackgroundState::Paused | BackgroundState::Pausing => {
                !self.config.pause_provider_announcements
            }
        }
    }

    /// Get configuration
    pub fn config(&self) -> &BackgroundModeConfig {
        &self.config
    }

    /// Get statistics
    pub fn stats(&self) -> BackgroundModeStats {
        // Update current state time before returning stats
        let current_state = *self.state.read();
        self.update_time_tracking(current_state);

        self.stats.read().clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        *self.stats.write() = BackgroundModeStats::default();
        *self.time_in_active.write() = Duration::ZERO;
        *self.time_in_paused.write() = Duration::ZERO;
        *self.last_transition.write() = Some(Instant::now());
    }

    /// Increment DHT queries skipped counter
    pub fn record_dht_query_skipped(&self) {
        self.stats.write().dht_queries_skipped += 1;
    }

    /// Record connections closed on pause
    pub fn record_connections_closed(&self, count: usize) {
        self.stats.write().connections_closed_on_pause += count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_background_mode_creation() {
        let manager = BackgroundModeManager::new(BackgroundModeConfig::default());
        assert_eq!(manager.state(), BackgroundState::Active);
        assert!(manager.is_active());
        assert!(!manager.is_paused());
    }

    #[test]
    fn test_pause_resume() {
        let manager = BackgroundModeManager::new(BackgroundModeConfig::default());

        // Pause
        assert!(manager.pause().is_ok());
        assert_eq!(manager.state(), BackgroundState::Paused);
        assert!(manager.is_paused());
        assert!(!manager.is_active());

        // Resume
        assert!(manager.resume().is_ok());
        assert_eq!(manager.state(), BackgroundState::Active);
        assert!(manager.is_active());
        assert!(!manager.is_paused());
    }

    #[test]
    fn test_pause_when_already_paused() {
        let manager = BackgroundModeManager::new(BackgroundModeConfig::default());

        assert!(manager.pause().is_ok());
        assert!(manager.pause().is_ok()); // Should be ok to pause when already paused
        assert_eq!(manager.state(), BackgroundState::Paused);
    }

    #[test]
    fn test_resume_when_already_active() {
        let manager = BackgroundModeManager::new(BackgroundModeConfig::default());

        assert!(manager.resume().is_ok()); // Should be ok to resume when already active
        assert_eq!(manager.state(), BackgroundState::Active);
    }

    #[test]
    fn test_dht_query_allowed_in_active_state() {
        let manager = BackgroundModeManager::new(BackgroundModeConfig::default());
        assert!(manager.should_allow_dht_query());
    }

    #[test]
    fn test_dht_query_behavior_in_paused_state() {
        let config = BackgroundModeConfig {
            pause_dht_queries: true,
            ..Default::default()
        };
        let manager = BackgroundModeManager::new(config);

        manager
            .pause()
            .expect("test: pause should succeed from Active state");
        assert!(!manager.should_allow_dht_query());
    }

    #[test]
    fn test_dht_query_allowed_when_not_paused_in_background() {
        let config = BackgroundModeConfig {
            pause_dht_queries: false,
            ..Default::default()
        };
        let manager = BackgroundModeManager::new(config);

        manager
            .pause()
            .expect("test: pause should succeed from Active state");
        assert!(manager.should_allow_dht_query());
    }

    #[test]
    fn test_provider_announcements_in_active_state() {
        let manager = BackgroundModeManager::new(BackgroundModeConfig::default());
        assert!(manager.should_allow_provider_announcements());
    }

    #[test]
    fn test_provider_announcements_in_paused_state() {
        let manager = BackgroundModeManager::new(BackgroundModeConfig::default());
        manager
            .pause()
            .expect("test: pause should succeed from Active state");
        // Default config pauses announcements
        assert!(!manager.should_allow_provider_announcements());
    }

    #[test]
    fn test_statistics_tracking() {
        let manager = BackgroundModeManager::new(BackgroundModeConfig::default());

        manager
            .pause()
            .expect("test: first pause should succeed from Active state");
        manager
            .resume()
            .expect("test: resume should succeed from Paused state");
        manager
            .pause()
            .expect("test: second pause should succeed from Active state");

        let stats = manager.stats();
        assert_eq!(stats.pause_count, 2);
        assert_eq!(stats.resume_count, 1);
    }

    #[test]
    fn test_mobile_config() {
        let config = BackgroundModeConfig::mobile();
        assert!(config.pause_dht_queries);
        assert!(config.pause_provider_announcements);
        assert!(config.close_idle_connections);
        assert_eq!(config.minimal_connection_count, 2);
    }

    #[test]
    fn test_server_config() {
        let config = BackgroundModeConfig::server();
        assert!(!config.pause_dht_queries);
        assert!(!config.pause_provider_announcements);
        assert!(!config.close_idle_connections);
    }

    #[test]
    fn test_record_dht_query_skipped() {
        let manager = BackgroundModeManager::new(BackgroundModeConfig::default());
        manager.record_dht_query_skipped();
        manager.record_dht_query_skipped();

        let stats = manager.stats();
        assert_eq!(stats.dht_queries_skipped, 2);
    }

    #[test]
    fn test_record_connections_closed() {
        let manager = BackgroundModeManager::new(BackgroundModeConfig::default());
        manager.record_connections_closed(5);

        let stats = manager.stats();
        assert_eq!(stats.connections_closed_on_pause, 5);
    }

    #[test]
    fn test_reset_stats() {
        let manager = BackgroundModeManager::new(BackgroundModeConfig::default());

        manager
            .pause()
            .expect("test: pause should succeed from Active state");
        manager
            .resume()
            .expect("test: resume should succeed from Paused state");
        manager.record_dht_query_skipped();

        manager.reset_stats();

        let stats = manager.stats();
        assert_eq!(stats.pause_count, 0);
        assert_eq!(stats.resume_count, 0);
        assert_eq!(stats.dht_queries_skipped, 0);
    }
}
