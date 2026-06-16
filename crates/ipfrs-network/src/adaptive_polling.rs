//! Adaptive polling intervals for power-efficient network operations
//!
//! This module provides dynamic adjustment of polling intervals based on network activity.
//! It's designed to reduce power consumption on edge devices and mobile platforms by:
//! - Increasing poll intervals during idle periods
//! - Decreasing intervals during active periods
//! - Supporting sleep mode detection
//! - Providing activity-based adjustments

use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Errors that can occur during adaptive polling operations
#[derive(Error, Debug, Clone)]
pub enum AdaptivePollingError {
    #[error("Invalid polling configuration: {0}")]
    InvalidConfig(String),

    #[error("Polling interval adjustment failed: {0}")]
    AdjustmentFailed(String),
}

/// Configuration for adaptive polling
#[derive(Debug, Clone)]
pub struct AdaptivePollingConfig {
    /// Minimum poll interval (fastest polling rate)
    pub min_interval: Duration,

    /// Maximum poll interval (slowest polling rate)
    pub max_interval: Duration,

    /// Default poll interval when activity is moderate
    pub default_interval: Duration,

    /// Interval increase factor when idle (multiplier)
    pub idle_increase_factor: f64,

    /// Interval decrease factor when active (multiplier)
    pub active_decrease_factor: f64,

    /// Time threshold to consider network idle
    pub idle_threshold: Duration,

    /// Enable sleep mode when completely inactive
    pub enable_sleep_mode: bool,

    /// Sleep mode interval (very long poll interval)
    pub sleep_interval: Duration,

    /// Time threshold to enter sleep mode
    pub sleep_threshold: Duration,
}

impl Default for AdaptivePollingConfig {
    fn default() -> Self {
        Self {
            min_interval: Duration::from_millis(50), // 50ms minimum (20 Hz max)
            max_interval: Duration::from_secs(5),    // 5s maximum
            default_interval: Duration::from_millis(500), // 500ms default (2 Hz)
            idle_increase_factor: 1.5,               // 1.5x slower when idle
            active_decrease_factor: 0.7,             // 0.7x faster when active
            idle_threshold: Duration::from_secs(10), // 10s without activity = idle
            enable_sleep_mode: true,
            sleep_interval: Duration::from_secs(30), // 30s in sleep mode
            sleep_threshold: Duration::from_secs(60), // 60s without activity = sleep
        }
    }
}

impl AdaptivePollingConfig {
    /// Configuration for mobile devices (battery-conscious)
    pub fn mobile() -> Self {
        Self {
            min_interval: Duration::from_millis(100),
            max_interval: Duration::from_secs(10),
            default_interval: Duration::from_secs(1),
            idle_increase_factor: 2.0, // More aggressive idle slowdown
            active_decrease_factor: 0.6,
            idle_threshold: Duration::from_secs(5),
            enable_sleep_mode: true,
            sleep_interval: Duration::from_secs(60),
            sleep_threshold: Duration::from_secs(30),
        }
    }

    /// Configuration for IoT/edge devices (power-saving)
    pub fn iot() -> Self {
        Self {
            min_interval: Duration::from_millis(200),
            max_interval: Duration::from_secs(30),
            default_interval: Duration::from_secs(2),
            idle_increase_factor: 2.5, // Very aggressive idle slowdown
            active_decrease_factor: 0.5,
            idle_threshold: Duration::from_secs(5),
            enable_sleep_mode: true,
            sleep_interval: Duration::from_secs(120), // 2 minutes
            sleep_threshold: Duration::from_secs(20),
        }
    }

    /// Configuration for low-power mode (maximum battery saving)
    pub fn low_power() -> Self {
        Self {
            min_interval: Duration::from_millis(500),
            max_interval: Duration::from_secs(60),
            default_interval: Duration::from_secs(5),
            idle_increase_factor: 3.0, // Very aggressive
            active_decrease_factor: 0.5,
            idle_threshold: Duration::from_secs(3),
            enable_sleep_mode: true,
            sleep_interval: Duration::from_secs(300), // 5 minutes
            sleep_threshold: Duration::from_secs(15),
        }
    }

    /// Configuration for high-performance mode (minimal latency)
    pub fn high_performance() -> Self {
        Self {
            min_interval: Duration::from_millis(10),
            max_interval: Duration::from_millis(500),
            default_interval: Duration::from_millis(50),
            idle_increase_factor: 1.2, // Gentle slowdown
            active_decrease_factor: 0.9,
            idle_threshold: Duration::from_secs(30),
            enable_sleep_mode: false, // No sleep mode
            sleep_interval: Duration::from_secs(1),
            sleep_threshold: Duration::from_secs(600),
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), AdaptivePollingError> {
        if self.min_interval.is_zero() {
            return Err(AdaptivePollingError::InvalidConfig(
                "Minimum interval must be > 0".to_string(),
            ));
        }

        if self.max_interval < self.min_interval {
            return Err(AdaptivePollingError::InvalidConfig(
                "Maximum interval must be >= minimum interval".to_string(),
            ));
        }

        if self.default_interval < self.min_interval || self.default_interval > self.max_interval {
            return Err(AdaptivePollingError::InvalidConfig(
                "Default interval must be between min and max".to_string(),
            ));
        }

        if self.idle_increase_factor < 1.0 {
            return Err(AdaptivePollingError::InvalidConfig(
                "Idle increase factor must be >= 1.0".to_string(),
            ));
        }

        if self.active_decrease_factor <= 0.0 || self.active_decrease_factor > 1.0 {
            return Err(AdaptivePollingError::InvalidConfig(
                "Active decrease factor must be in (0.0, 1.0]".to_string(),
            ));
        }

        Ok(())
    }
}

/// Activity level for adaptive polling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivityLevel {
    /// High activity (many events)
    High,
    /// Moderate activity
    Moderate,
    /// Low activity (few events)
    Low,
    /// Idle (no recent events)
    Idle,
    /// Sleep mode (completely inactive)
    Sleep,
}

/// State of the adaptive polling mechanism
#[derive(Debug)]
struct PollingState {
    /// Current poll interval
    current_interval: Duration,

    /// Current activity level
    activity_level: ActivityLevel,

    /// Last activity timestamp
    last_activity: Instant,

    /// Number of events in current window
    events_in_window: u64,

    /// Window start time
    window_start: Instant,

    /// Total adjustments made
    adjustments_made: u64,
}

impl PollingState {
    fn new(default_interval: Duration) -> Self {
        let now = Instant::now();
        Self {
            current_interval: default_interval,
            activity_level: ActivityLevel::Moderate,
            last_activity: now,
            events_in_window: 0,
            window_start: now,
            adjustments_made: 0,
        }
    }
}

/// Adaptive polling manager
pub struct AdaptivePolling {
    config: AdaptivePollingConfig,
    state: Arc<RwLock<PollingState>>,
}

impl AdaptivePolling {
    /// Create a new adaptive polling manager
    pub fn new(config: AdaptivePollingConfig) -> Result<Self, AdaptivePollingError> {
        config.validate()?;

        let state = PollingState::new(config.default_interval);

        Ok(Self {
            config: config.clone(),
            state: Arc::new(RwLock::new(state)),
        })
    }

    /// Record network activity (call this when events occur)
    pub fn record_activity(&self) {
        let mut state = self.state.write();
        let now = Instant::now();

        state.last_activity = now;
        state.events_in_window += 1;

        // Reset window every second
        if now.duration_since(state.window_start) >= Duration::from_secs(1) {
            // Determine activity level based on event count
            let events_per_sec = state.events_in_window;
            state.activity_level = if events_per_sec >= 10 {
                ActivityLevel::High
            } else if events_per_sec >= 3 {
                ActivityLevel::Moderate
            } else if events_per_sec >= 1 {
                ActivityLevel::Low
            } else {
                ActivityLevel::Idle
            };

            state.events_in_window = 0;
            state.window_start = now;
        }
    }

    /// Adjust poll interval based on current activity
    pub fn adjust_interval(&self) {
        let mut state = self.state.write();
        let now = Instant::now();
        let time_since_activity = now.duration_since(state.last_activity);

        // Check for sleep mode
        if self.config.enable_sleep_mode && time_since_activity >= self.config.sleep_threshold {
            state.activity_level = ActivityLevel::Sleep;
            state.current_interval = self.config.sleep_interval;
            state.adjustments_made += 1;
            return;
        }

        // Check for idle mode
        if time_since_activity >= self.config.idle_threshold {
            state.activity_level = ActivityLevel::Idle;
            let new_interval = Duration::from_secs_f64(
                state.current_interval.as_secs_f64() * self.config.idle_increase_factor,
            );
            state.current_interval = new_interval.min(self.config.max_interval);
            state.adjustments_made += 1;
            return;
        }

        // Adjust based on activity level
        match state.activity_level {
            ActivityLevel::High => {
                let new_interval = Duration::from_secs_f64(
                    state.current_interval.as_secs_f64() * self.config.active_decrease_factor,
                );
                state.current_interval = new_interval.max(self.config.min_interval);
                state.adjustments_made += 1;
            }
            ActivityLevel::Moderate => {
                // Gradually move towards default interval
                let diff = state.current_interval.as_secs_f64()
                    - self.config.default_interval.as_secs_f64();
                if diff.abs() > 0.1 {
                    let new_interval =
                        Duration::from_secs_f64(state.current_interval.as_secs_f64() - diff * 0.1);
                    state.current_interval = new_interval
                        .max(self.config.min_interval)
                        .min(self.config.max_interval);
                    state.adjustments_made += 1;
                }
            }
            ActivityLevel::Low => {
                let new_interval = Duration::from_secs_f64(
                    state.current_interval.as_secs_f64() * self.config.idle_increase_factor * 0.5,
                );
                state.current_interval = new_interval.min(self.config.max_interval);
                state.adjustments_made += 1;
            }
            ActivityLevel::Idle | ActivityLevel::Sleep => {
                // Already handled above
            }
        }
    }

    /// Get the current poll interval
    pub fn current_interval(&self) -> Duration {
        self.state.read().current_interval
    }

    /// Get the current activity level
    pub fn activity_level(&self) -> ActivityLevel {
        self.state.read().activity_level
    }

    /// Get the time since last activity
    pub fn time_since_activity(&self) -> Duration {
        let state = self.state.read();
        Instant::now().duration_since(state.last_activity)
    }

    /// Reset to default interval
    pub fn reset(&self) {
        let mut state = self.state.write();
        state.current_interval = self.config.default_interval;
        state.activity_level = ActivityLevel::Moderate;
        state.last_activity = Instant::now();
        state.events_in_window = 0;
        state.window_start = Instant::now();
    }

    /// Get polling statistics
    pub fn stats(&self) -> AdaptivePollingStats {
        let state = self.state.read();
        AdaptivePollingStats {
            current_interval: state.current_interval,
            activity_level: state.activity_level,
            time_since_activity: Instant::now().duration_since(state.last_activity),
            events_in_window: state.events_in_window,
            adjustments_made: state.adjustments_made,
        }
    }
}

/// Statistics for adaptive polling
#[derive(Debug, Clone)]
pub struct AdaptivePollingStats {
    pub current_interval: Duration,
    pub activity_level: ActivityLevel,
    pub time_since_activity: Duration,
    pub events_in_window: u64,
    pub adjustments_made: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_config_default() {
        let config = AdaptivePollingConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.min_interval, Duration::from_millis(50));
    }

    #[test]
    fn test_config_mobile() {
        let config = AdaptivePollingConfig::mobile();
        assert!(config.validate().is_ok());
        assert!(config.enable_sleep_mode);
    }

    #[test]
    fn test_config_iot() {
        let config = AdaptivePollingConfig::iot();
        assert!(config.validate().is_ok());
        assert_eq!(config.max_interval, Duration::from_secs(30));
    }

    #[test]
    fn test_config_low_power() {
        let config = AdaptivePollingConfig::low_power();
        assert!(config.validate().is_ok());
        assert_eq!(config.sleep_interval, Duration::from_secs(300));
    }

    #[test]
    fn test_config_high_performance() {
        let config = AdaptivePollingConfig::high_performance();
        assert!(config.validate().is_ok());
        assert!(!config.enable_sleep_mode);
    }

    #[test]
    fn test_config_validation_min_interval() {
        let config = AdaptivePollingConfig {
            min_interval: Duration::from_secs(0),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_max_less_than_min() {
        let config = AdaptivePollingConfig {
            max_interval: Duration::from_millis(10),
            min_interval: Duration::from_millis(100),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_polling_new() {
        let config = AdaptivePollingConfig::default();
        let polling =
            AdaptivePolling::new(config.clone()).expect("test: failed to create AdaptivePolling");

        assert_eq!(polling.current_interval(), config.default_interval);
        assert_eq!(polling.activity_level(), ActivityLevel::Moderate);
    }

    #[test]
    fn test_record_activity() {
        let config = AdaptivePollingConfig::default();
        let polling = AdaptivePolling::new(config).expect("test: failed to create AdaptivePolling");

        polling.record_activity();

        assert!(polling.time_since_activity() < Duration::from_millis(100));
    }

    #[test]
    fn test_adjust_interval_idle() {
        let config = AdaptivePollingConfig {
            idle_threshold: Duration::from_millis(100),
            ..Default::default()
        };
        let polling =
            AdaptivePolling::new(config.clone()).expect("test: failed to create AdaptivePolling");

        // Wait to become idle
        thread::sleep(Duration::from_millis(150));
        polling.adjust_interval();

        let new_interval = polling.current_interval();
        assert!(new_interval > config.default_interval);
    }

    #[test]
    fn test_adjust_interval_active() {
        let config = AdaptivePollingConfig::default();
        let polling = AdaptivePolling::new(config.clone())
            .expect("test: failed to create AdaptivePolling for adjust_interval_active");

        // Simulate high activity
        for _ in 0..15 {
            polling.record_activity();
        }

        // Wait for window reset
        thread::sleep(Duration::from_secs(1));
        polling.record_activity();

        polling.adjust_interval();

        let new_interval = polling.current_interval();
        assert!(new_interval < config.default_interval);
    }

    #[test]
    fn test_sleep_mode() {
        let config = AdaptivePollingConfig {
            sleep_threshold: Duration::from_millis(100),
            enable_sleep_mode: true,
            ..Default::default()
        };
        let polling = AdaptivePolling::new(config.clone())
            .expect("test: failed to create AdaptivePolling for sleep_mode");

        // Wait to enter sleep mode
        thread::sleep(Duration::from_millis(150));
        polling.adjust_interval();

        assert_eq!(polling.activity_level(), ActivityLevel::Sleep);
        assert_eq!(polling.current_interval(), config.sleep_interval);
    }

    #[test]
    fn test_reset() {
        let config = AdaptivePollingConfig::default();
        let polling = AdaptivePolling::new(config.clone())
            .expect("test: failed to create AdaptivePolling for reset");

        // Change interval
        thread::sleep(Duration::from_millis(100));
        polling.adjust_interval();

        // Reset
        polling.reset();

        assert_eq!(polling.current_interval(), config.default_interval);
        assert_eq!(polling.activity_level(), ActivityLevel::Moderate);
    }

    #[test]
    fn test_stats() {
        let config = AdaptivePollingConfig::default();
        let polling =
            AdaptivePolling::new(config).expect("test: failed to create AdaptivePolling for stats");

        polling.record_activity();

        let stats = polling.stats();
        assert!(stats.time_since_activity < Duration::from_millis(100));
        assert_eq!(stats.activity_level, ActivityLevel::Moderate);
    }

    #[test]
    fn test_interval_bounds() {
        let config = AdaptivePollingConfig::default();
        let polling = AdaptivePolling::new(config.clone())
            .expect("test: failed to create AdaptivePolling for interval_bounds");

        // Try to go below min
        for _ in 0..100 {
            polling.adjust_interval();
        }

        let interval = polling.current_interval();
        assert!(interval >= config.min_interval);
        assert!(interval <= config.max_interval);
    }
}
