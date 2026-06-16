//! Storage Health Monitor
//!
//! Provides configurable health monitoring for the storage subsystem.
//! Tracks named health checks, evaluates utilization thresholds,
//! and computes overall health status based on worst-case analysis.
//!
//! ## Example
//! ```
//! use ipfrs_storage::health_monitor::{
//!     StorageHealthMonitor, HealthMonitorConfig, MonitorHealthStatus,
//! };
//!
//! let config = HealthMonitorConfig::default();
//! let mut monitor = StorageHealthMonitor::new(config);
//! monitor.register_check("disk");
//! monitor.update_check("disk", MonitorHealthStatus::Healthy, "OK").ok();
//! assert_eq!(monitor.overall_health(), MonitorHealthStatus::Healthy);
//! ```

use std::collections::HashMap;

/// Health status for the monitor subsystem.
///
/// Distinct from `crate::health::HealthStatus` — this enum adds an `Unknown`
/// variant for checks that have not yet been evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonitorHealthStatus {
    /// All metrics within normal range.
    Healthy,
    /// One or more metrics above the degraded threshold but below unhealthy.
    Degraded,
    /// One or more metrics above the unhealthy threshold.
    Unhealthy,
    /// Check has not been evaluated yet.
    Unknown,
}

impl MonitorHealthStatus {
    /// Numeric severity for ordering (higher = worse).
    fn severity(self) -> u8 {
        match self {
            Self::Healthy => 0,
            Self::Unknown => 1,
            Self::Degraded => 2,
            Self::Unhealthy => 3,
        }
    }

    /// Return the worse of two statuses.
    fn worst(self, other: Self) -> Self {
        if self.severity() >= other.severity() {
            self
        } else {
            other
        }
    }
}

/// A single named health check.
#[derive(Debug, Clone)]
pub struct MonitorHealthCheck {
    /// Human-readable name of the check.
    pub name: String,
    /// Current status.
    pub status: MonitorHealthStatus,
    /// Descriptive message from the last evaluation.
    pub message: String,
    /// Tick at which this check was last updated.
    pub last_check_tick: u64,
}

/// Configuration for [`StorageHealthMonitor`].
#[derive(Debug, Clone)]
pub struct HealthMonitorConfig {
    /// Number of ticks between automatic check runs.
    pub check_interval_ticks: u64,
    /// Utilization ratio above which status becomes `Degraded`.
    pub degraded_threshold: f64,
    /// Utilization ratio above which status becomes `Unhealthy`.
    pub unhealthy_threshold: f64,
}

impl Default for HealthMonitorConfig {
    fn default() -> Self {
        Self {
            check_interval_ticks: 50,
            degraded_threshold: 0.8,
            unhealthy_threshold: 0.95,
        }
    }
}

/// Aggregate statistics snapshot from [`StorageHealthMonitor`].
#[derive(Debug, Clone)]
pub struct HealthMonitorStats {
    /// Total number of times `run_checks` has been called.
    pub check_count: u64,
    /// Current overall status.
    pub overall_status: MonitorHealthStatus,
    /// Number of checks with `Healthy` status.
    pub healthy: usize,
    /// Number of checks with `Degraded` status.
    pub degraded: usize,
    /// Number of checks with `Unhealthy` status.
    pub unhealthy: usize,
}

/// Monitors the health of storage subsystems.
///
/// Maintains a set of named health checks, supports tick-based scheduling,
/// and computes an overall status from the worst individual check.
pub struct StorageHealthMonitor {
    config: HealthMonitorConfig,
    checks: HashMap<String, MonitorHealthCheck>,
    current_tick: u64,
    last_check_tick: u64,
    overall_status: MonitorHealthStatus,
    check_count: u64,
}

impl StorageHealthMonitor {
    /// Create a new monitor with the given configuration.
    pub fn new(config: HealthMonitorConfig) -> Self {
        Self {
            config,
            checks: HashMap::new(),
            current_tick: 0,
            last_check_tick: 0,
            overall_status: MonitorHealthStatus::Healthy,
            check_count: 0,
        }
    }

    /// Register a named health check with `Unknown` status.
    pub fn register_check(&mut self, name: &str) {
        self.checks.insert(
            name.to_string(),
            MonitorHealthCheck {
                name: name.to_string(),
                status: MonitorHealthStatus::Unknown,
                message: String::new(),
                last_check_tick: 0,
            },
        );
    }

    /// Update the status and message of an existing check.
    ///
    /// Returns `Err` if no check with the given name is registered.
    pub fn update_check(
        &mut self,
        name: &str,
        status: MonitorHealthStatus,
        message: &str,
    ) -> Result<(), String> {
        let check = self
            .checks
            .get_mut(name)
            .ok_or_else(|| format!("check '{}' not found", name))?;
        check.status = status;
        check.message = message.to_string();
        check.last_check_tick = self.current_tick;
        Ok(())
    }

    /// Evaluate a utilization ratio against the configured thresholds.
    ///
    /// `capacity` of zero is treated as `Unhealthy`.
    pub fn evaluate_utilization(&self, current: u64, capacity: u64) -> MonitorHealthStatus {
        if capacity == 0 {
            return MonitorHealthStatus::Unhealthy;
        }
        let ratio = current as f64 / capacity as f64;
        if ratio >= self.config.unhealthy_threshold {
            MonitorHealthStatus::Unhealthy
        } else if ratio >= self.config.degraded_threshold {
            MonitorHealthStatus::Degraded
        } else {
            MonitorHealthStatus::Healthy
        }
    }

    /// Compute the overall health as the worst status among all checks.
    ///
    /// Returns `Healthy` when there are no registered checks.
    pub fn overall_health(&self) -> MonitorHealthStatus {
        self.checks
            .values()
            .fold(MonitorHealthStatus::Healthy, |acc, c| acc.worst(c.status))
    }

    /// Retrieve a reference to a named check, if it exists.
    pub fn get_check(&self, name: &str) -> Option<&MonitorHealthCheck> {
        self.checks.get(name)
    }

    /// Returns `true` when enough ticks have elapsed since the last check run.
    pub fn should_check(&self) -> bool {
        self.current_tick.saturating_sub(self.last_check_tick) >= self.config.check_interval_ticks
    }

    /// Execute a check cycle: update bookkeeping and recompute overall status.
    pub fn run_checks(&mut self) {
        self.last_check_tick = self.current_tick;
        self.check_count += 1;
        self.overall_status = self.overall_health();
    }

    /// Count of checks currently `Healthy`.
    pub fn healthy_count(&self) -> usize {
        self.checks
            .values()
            .filter(|c| c.status == MonitorHealthStatus::Healthy)
            .count()
    }

    /// Count of checks currently `Degraded`.
    pub fn degraded_count(&self) -> usize {
        self.checks
            .values()
            .filter(|c| c.status == MonitorHealthStatus::Degraded)
            .count()
    }

    /// Count of checks currently `Unhealthy`.
    pub fn unhealthy_count(&self) -> usize {
        self.checks
            .values()
            .filter(|c| c.status == MonitorHealthStatus::Unhealthy)
            .count()
    }

    /// Advance the internal tick counter by one.
    pub fn tick(&mut self) {
        self.current_tick += 1;
    }

    /// Return references to all registered checks.
    pub fn all_checks(&self) -> Vec<&MonitorHealthCheck> {
        self.checks.values().collect()
    }

    /// Produce a statistics snapshot.
    pub fn stats(&self) -> HealthMonitorStats {
        HealthMonitorStats {
            check_count: self.check_count,
            overall_status: self.overall_status,
            healthy: self.healthy_count(),
            degraded: self.degraded_count(),
            unhealthy: self.unhealthy_count(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_monitor() -> StorageHealthMonitor {
        StorageHealthMonitor::new(HealthMonitorConfig::default())
    }

    // -- construction & defaults -------------------------------------------

    #[test]
    fn test_default_config_values() {
        let cfg = HealthMonitorConfig::default();
        assert_eq!(cfg.check_interval_ticks, 50);
        assert!((cfg.degraded_threshold - 0.8).abs() < f64::EPSILON);
        assert!((cfg.unhealthy_threshold - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_new_monitor_is_healthy() {
        let m = default_monitor();
        assert_eq!(m.overall_health(), MonitorHealthStatus::Healthy);
        assert_eq!(m.overall_status, MonitorHealthStatus::Healthy);
    }

    #[test]
    fn test_new_monitor_empty_checks() {
        let m = default_monitor();
        assert!(m.all_checks().is_empty());
        assert_eq!(m.healthy_count(), 0);
        assert_eq!(m.degraded_count(), 0);
        assert_eq!(m.unhealthy_count(), 0);
    }

    // -- register / get ----------------------------------------------------

    #[test]
    fn test_register_check_unknown() {
        let mut m = default_monitor();
        m.register_check("disk");
        let c = m.get_check("disk").expect("should exist");
        assert_eq!(c.status, MonitorHealthStatus::Unknown);
        assert_eq!(c.name, "disk");
        assert!(c.message.is_empty());
    }

    #[test]
    fn test_register_multiple_checks() {
        let mut m = default_monitor();
        m.register_check("disk");
        m.register_check("memory");
        m.register_check("network");
        assert_eq!(m.all_checks().len(), 3);
    }

    #[test]
    fn test_get_check_nonexistent() {
        let m = default_monitor();
        assert!(m.get_check("nope").is_none());
    }

    // -- update ------------------------------------------------------------

    #[test]
    fn test_update_check_success() {
        let mut m = default_monitor();
        m.register_check("disk");
        let res = m.update_check("disk", MonitorHealthStatus::Healthy, "all good");
        assert!(res.is_ok());
        let c = m.get_check("disk").expect("should exist");
        assert_eq!(c.status, MonitorHealthStatus::Healthy);
        assert_eq!(c.message, "all good");
    }

    #[test]
    fn test_update_check_not_found() {
        let mut m = default_monitor();
        let res = m.update_check("missing", MonitorHealthStatus::Healthy, "");
        assert!(res.is_err());
    }

    #[test]
    fn test_update_check_records_tick() {
        let mut m = default_monitor();
        m.register_check("disk");
        for _ in 0..10 {
            m.tick();
        }
        m.update_check("disk", MonitorHealthStatus::Healthy, "ok")
            .expect("update should succeed");
        let c = m.get_check("disk").expect("should exist");
        assert_eq!(c.last_check_tick, 10);
    }

    // -- evaluate_utilization ----------------------------------------------

    #[test]
    fn test_evaluate_healthy() {
        let m = default_monitor();
        assert_eq!(
            m.evaluate_utilization(50, 100),
            MonitorHealthStatus::Healthy
        );
    }

    #[test]
    fn test_evaluate_degraded() {
        let m = default_monitor();
        assert_eq!(
            m.evaluate_utilization(85, 100),
            MonitorHealthStatus::Degraded
        );
    }

    #[test]
    fn test_evaluate_unhealthy() {
        let m = default_monitor();
        assert_eq!(
            m.evaluate_utilization(96, 100),
            MonitorHealthStatus::Unhealthy
        );
    }

    #[test]
    fn test_evaluate_zero_capacity() {
        let m = default_monitor();
        assert_eq!(m.evaluate_utilization(0, 0), MonitorHealthStatus::Unhealthy);
    }

    #[test]
    fn test_evaluate_exact_degraded_boundary() {
        let m = default_monitor();
        // 80/100 = 0.80 — exactly at the threshold => Degraded
        assert_eq!(
            m.evaluate_utilization(80, 100),
            MonitorHealthStatus::Degraded
        );
    }

    #[test]
    fn test_evaluate_just_below_degraded() {
        let m = default_monitor();
        assert_eq!(
            m.evaluate_utilization(79, 100),
            MonitorHealthStatus::Healthy
        );
    }

    #[test]
    fn test_evaluate_exact_unhealthy_boundary() {
        let m = default_monitor();
        // 95/100 = 0.95
        assert_eq!(
            m.evaluate_utilization(95, 100),
            MonitorHealthStatus::Unhealthy
        );
    }

    #[test]
    fn test_evaluate_just_below_unhealthy() {
        let m = default_monitor();
        // 94/100 = 0.94
        assert_eq!(
            m.evaluate_utilization(94, 100),
            MonitorHealthStatus::Degraded
        );
    }

    #[test]
    fn test_evaluate_full_capacity() {
        let m = default_monitor();
        assert_eq!(
            m.evaluate_utilization(100, 100),
            MonitorHealthStatus::Unhealthy
        );
    }

    #[test]
    fn test_evaluate_custom_thresholds() {
        let cfg = HealthMonitorConfig {
            check_interval_ticks: 10,
            degraded_threshold: 0.5,
            unhealthy_threshold: 0.7,
        };
        let m = StorageHealthMonitor::new(cfg);
        assert_eq!(
            m.evaluate_utilization(40, 100),
            MonitorHealthStatus::Healthy
        );
        assert_eq!(
            m.evaluate_utilization(60, 100),
            MonitorHealthStatus::Degraded
        );
        assert_eq!(
            m.evaluate_utilization(75, 100),
            MonitorHealthStatus::Unhealthy
        );
    }

    // -- overall_health ----------------------------------------------------

    #[test]
    fn test_overall_health_all_healthy() {
        let mut m = default_monitor();
        m.register_check("a");
        m.register_check("b");
        m.update_check("a", MonitorHealthStatus::Healthy, "").ok();
        m.update_check("b", MonitorHealthStatus::Healthy, "").ok();
        assert_eq!(m.overall_health(), MonitorHealthStatus::Healthy);
    }

    #[test]
    fn test_overall_health_one_degraded() {
        let mut m = default_monitor();
        m.register_check("a");
        m.register_check("b");
        m.update_check("a", MonitorHealthStatus::Healthy, "").ok();
        m.update_check("b", MonitorHealthStatus::Degraded, "").ok();
        assert_eq!(m.overall_health(), MonitorHealthStatus::Degraded);
    }

    #[test]
    fn test_overall_health_one_unhealthy() {
        let mut m = default_monitor();
        m.register_check("a");
        m.register_check("b");
        m.update_check("a", MonitorHealthStatus::Healthy, "").ok();
        m.update_check("b", MonitorHealthStatus::Unhealthy, "").ok();
        assert_eq!(m.overall_health(), MonitorHealthStatus::Unhealthy);
    }

    #[test]
    fn test_overall_health_unknown_worse_than_healthy() {
        let mut m = default_monitor();
        m.register_check("a");
        m.register_check("b");
        m.update_check("a", MonitorHealthStatus::Healthy, "").ok();
        // b stays Unknown
        assert_eq!(m.overall_health(), MonitorHealthStatus::Unknown);
    }

    #[test]
    fn test_overall_health_unhealthy_worst() {
        let mut m = default_monitor();
        m.register_check("a");
        m.register_check("b");
        m.register_check("c");
        m.update_check("a", MonitorHealthStatus::Healthy, "").ok();
        m.update_check("b", MonitorHealthStatus::Degraded, "").ok();
        m.update_check("c", MonitorHealthStatus::Unhealthy, "").ok();
        assert_eq!(m.overall_health(), MonitorHealthStatus::Unhealthy);
    }

    // -- should_check / tick -----------------------------------------------

    #[test]
    fn test_should_check_initially_true() {
        // current_tick == 0, last_check_tick == 0, interval == 50
        // 0 - 0 = 0 < 50 => false at start with default config
        let m = default_monitor();
        assert!(!m.should_check());
    }

    #[test]
    fn test_should_check_after_enough_ticks() {
        let mut m = default_monitor();
        for _ in 0..50 {
            m.tick();
        }
        assert!(m.should_check());
    }

    #[test]
    fn test_should_check_resets_after_run() {
        let mut m = default_monitor();
        for _ in 0..50 {
            m.tick();
        }
        assert!(m.should_check());
        m.run_checks();
        assert!(!m.should_check());
    }

    #[test]
    fn test_tick_increments() {
        let mut m = default_monitor();
        m.tick();
        m.tick();
        m.tick();
        assert_eq!(m.current_tick, 3);
    }

    // -- run_checks --------------------------------------------------------

    #[test]
    fn test_run_checks_increments_count() {
        let mut m = default_monitor();
        assert_eq!(m.check_count, 0);
        m.run_checks();
        assert_eq!(m.check_count, 1);
        m.run_checks();
        assert_eq!(m.check_count, 2);
    }

    #[test]
    fn test_run_checks_updates_overall() {
        let mut m = default_monitor();
        m.register_check("a");
        m.update_check("a", MonitorHealthStatus::Degraded, "warn")
            .ok();
        m.run_checks();
        assert_eq!(m.overall_status, MonitorHealthStatus::Degraded);
    }

    // -- counts ------------------------------------------------------------

    #[test]
    fn test_healthy_count() {
        let mut m = default_monitor();
        m.register_check("a");
        m.register_check("b");
        m.update_check("a", MonitorHealthStatus::Healthy, "").ok();
        m.update_check("b", MonitorHealthStatus::Healthy, "").ok();
        assert_eq!(m.healthy_count(), 2);
    }

    #[test]
    fn test_degraded_count() {
        let mut m = default_monitor();
        m.register_check("a");
        m.register_check("b");
        m.update_check("a", MonitorHealthStatus::Degraded, "").ok();
        m.update_check("b", MonitorHealthStatus::Degraded, "").ok();
        assert_eq!(m.degraded_count(), 2);
    }

    #[test]
    fn test_unhealthy_count() {
        let mut m = default_monitor();
        m.register_check("a");
        m.update_check("a", MonitorHealthStatus::Unhealthy, "").ok();
        assert_eq!(m.unhealthy_count(), 1);
    }

    #[test]
    fn test_mixed_counts() {
        let mut m = default_monitor();
        m.register_check("a");
        m.register_check("b");
        m.register_check("c");
        m.register_check("d");
        m.update_check("a", MonitorHealthStatus::Healthy, "").ok();
        m.update_check("b", MonitorHealthStatus::Degraded, "").ok();
        m.update_check("c", MonitorHealthStatus::Unhealthy, "").ok();
        // d stays Unknown
        assert_eq!(m.healthy_count(), 1);
        assert_eq!(m.degraded_count(), 1);
        assert_eq!(m.unhealthy_count(), 1);
    }

    // -- stats -------------------------------------------------------------

    #[test]
    fn test_stats_snapshot() {
        let mut m = default_monitor();
        m.register_check("a");
        m.register_check("b");
        m.update_check("a", MonitorHealthStatus::Healthy, "").ok();
        m.update_check("b", MonitorHealthStatus::Degraded, "").ok();
        m.run_checks();
        let s = m.stats();
        assert_eq!(s.check_count, 1);
        assert_eq!(s.overall_status, MonitorHealthStatus::Degraded);
        assert_eq!(s.healthy, 1);
        assert_eq!(s.degraded, 1);
        assert_eq!(s.unhealthy, 0);
    }

    #[test]
    fn test_stats_empty_monitor() {
        let m = default_monitor();
        let s = m.stats();
        assert_eq!(s.check_count, 0);
        assert_eq!(s.overall_status, MonitorHealthStatus::Healthy);
        assert_eq!(s.healthy, 0);
        assert_eq!(s.degraded, 0);
        assert_eq!(s.unhealthy, 0);
    }

    // -- severity ordering -------------------------------------------------

    #[test]
    fn test_severity_ordering() {
        assert!(MonitorHealthStatus::Healthy.severity() < MonitorHealthStatus::Unknown.severity());
        assert!(MonitorHealthStatus::Unknown.severity() < MonitorHealthStatus::Degraded.severity());
        assert!(
            MonitorHealthStatus::Degraded.severity() < MonitorHealthStatus::Unhealthy.severity()
        );
    }

    #[test]
    fn test_worst_picks_higher_severity() {
        assert_eq!(
            MonitorHealthStatus::Healthy.worst(MonitorHealthStatus::Degraded),
            MonitorHealthStatus::Degraded
        );
        assert_eq!(
            MonitorHealthStatus::Unhealthy.worst(MonitorHealthStatus::Healthy),
            MonitorHealthStatus::Unhealthy
        );
    }

    // -- re-registration overwrites ----------------------------------------

    #[test]
    fn test_register_overwrites() {
        let mut m = default_monitor();
        m.register_check("a");
        m.update_check("a", MonitorHealthStatus::Unhealthy, "bad")
            .ok();
        m.register_check("a"); // re-register resets
        let c = m.get_check("a").expect("should exist");
        assert_eq!(c.status, MonitorHealthStatus::Unknown);
        assert!(c.message.is_empty());
    }

    // -- should_check with small interval ----------------------------------

    #[test]
    fn test_should_check_interval_one() {
        let cfg = HealthMonitorConfig {
            check_interval_ticks: 1,
            ..HealthMonitorConfig::default()
        };
        let mut m = StorageHealthMonitor::new(cfg);
        assert!(!m.should_check()); // 0 - 0 = 0 < 1
        m.tick();
        assert!(m.should_check()); // 1 - 0 = 1 >= 1
        m.run_checks();
        assert!(!m.should_check()); // 1 - 1 = 0 < 1
        m.tick();
        assert!(m.should_check()); // 2 - 1 = 1 >= 1
    }

    // -- all_checks --------------------------------------------------------

    #[test]
    fn test_all_checks_returns_all() {
        let mut m = default_monitor();
        m.register_check("x");
        m.register_check("y");
        let all = m.all_checks();
        assert_eq!(all.len(), 2);
        let names: Vec<&str> = all.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"x"));
        assert!(names.contains(&"y"));
    }
}
