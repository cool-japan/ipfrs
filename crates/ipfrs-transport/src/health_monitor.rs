//! Comprehensive health monitoring for transport components
//!
//! This module provides real-time health monitoring with automatic
//! alerting and degradation detection.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Health status of a component
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentHealth {
    /// Component is healthy and operating normally
    Healthy,
    /// Component is degraded but functional
    Degraded,
    /// Component is unhealthy and may fail soon
    Unhealthy,
    /// Component has failed
    Failed,
    /// Component status is unknown
    Unknown,
}

/// Type of component being monitored
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComponentType {
    /// Peer manager
    PeerManager,
    /// Want list
    WantList,
    /// Session manager
    SessionManager,
    /// QUIC transport
    QuicTransport,
    /// TCP transport
    TcpTransport,
    /// WebSocket transport
    WebSocketTransport,
    /// Content router
    ContentRouter,
    /// CDN edge node
    EdgeNode,
    /// NAT traversal manager
    NatTraversal,
}

/// Health check result for a component
#[derive(Debug, Clone)]
pub struct HealthCheck {
    /// Component being checked
    pub component: ComponentType,
    /// Current health status
    pub status: ComponentHealth,
    /// Timestamp of the check
    pub timestamp: Instant,
    /// Optional message describing the status
    pub message: Option<String>,
    /// Metrics snapshot
    pub metrics: HashMap<String, f64>,
}

/// Health monitoring configuration
#[derive(Debug, Clone)]
pub struct HealthMonitorConfig {
    /// Interval between health checks
    pub check_interval: Duration,
    /// Number of consecutive failures before marking unhealthy
    pub failure_threshold: usize,
    /// Number of consecutive successes to recover
    pub recovery_threshold: usize,
    /// Enable automatic degradation detection
    pub auto_degradation: bool,
    /// Latency threshold for degradation (ms)
    pub latency_threshold_ms: u64,
    /// Error rate threshold for degradation (0.0 to 1.0)
    pub error_rate_threshold: f64,
}

impl Default for HealthMonitorConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(30),
            failure_threshold: 3,
            recovery_threshold: 2,
            auto_degradation: true,
            latency_threshold_ms: 1000,
            error_rate_threshold: 0.1, // 10%
        }
    }
}

/// Component state tracking
#[derive(Debug, Clone)]
struct ComponentState {
    /// Current health status
    health: ComponentHealth,
    /// Last check timestamp
    last_check: Instant,
    /// Consecutive failures
    consecutive_failures: usize,
    /// Consecutive successes
    consecutive_successes: usize,
    /// Historical health checks
    history: Vec<HealthCheck>,
    /// Maximum history size
    max_history: usize,
}

impl ComponentState {
    fn new(max_history: usize) -> Self {
        Self {
            health: ComponentHealth::Unknown,
            last_check: Instant::now(),
            consecutive_failures: 0,
            consecutive_successes: 0,
            history: Vec::with_capacity(max_history),
            max_history,
        }
    }

    fn record_check(&mut self, check: HealthCheck) {
        self.last_check = check.timestamp;
        self.health = check.status;

        // Update consecutive counters
        match check.status {
            ComponentHealth::Healthy => {
                self.consecutive_successes += 1;
                self.consecutive_failures = 0;
            }
            ComponentHealth::Failed | ComponentHealth::Unhealthy => {
                self.consecutive_failures += 1;
                self.consecutive_successes = 0;
            }
            _ => {}
        }

        // Add to history
        self.history.push(check);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    fn get_latest_check(&self) -> Option<&HealthCheck> {
        self.history.last()
    }
}

/// Health alert triggered when component status changes
#[derive(Debug, Clone)]
pub struct HealthAlert {
    /// Component that triggered the alert
    pub component: ComponentType,
    /// New health status
    pub new_status: ComponentHealth,
    /// Previous health status
    pub old_status: ComponentHealth,
    /// Timestamp when alert was triggered
    pub timestamp: SystemTime,
    /// Alert message
    pub message: String,
}

/// Callback for health alerts
pub type AlertCallback = Arc<dyn Fn(HealthAlert) + Send + Sync>;

/// Main health monitoring system
pub struct HealthMonitor {
    /// Configuration
    #[allow(dead_code)]
    config: HealthMonitorConfig,
    /// Component states
    components: Arc<RwLock<HashMap<ComponentType, ComponentState>>>,
    /// Alert callbacks
    callbacks: Arc<RwLock<Vec<AlertCallback>>>,
}

impl HealthMonitor {
    /// Create a new health monitor
    pub fn new(config: HealthMonitorConfig) -> Self {
        Self {
            config,
            components: Arc::new(RwLock::new(HashMap::new())),
            callbacks: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a component for monitoring
    pub fn register_component(&self, component: ComponentType, max_history: usize) {
        let mut components = self.components.write().unwrap_or_else(|e| e.into_inner());
        components.insert(component, ComponentState::new(max_history));
    }

    /// Record a health check for a component
    pub fn record_health_check(&self, check: HealthCheck) {
        let component = check.component;
        let new_status = check.status;

        let mut components = self.components.write().unwrap_or_else(|e| e.into_inner());
        let state = components
            .entry(component)
            .or_insert_with(|| ComponentState::new(100));

        let old_status = state.health;
        state.record_check(check);

        // Check if status changed
        if old_status != new_status {
            drop(components); // Release lock before calling callbacks
            self.trigger_alert(component, old_status, new_status);
        }
    }

    /// Get current health status of a component
    pub fn get_health(&self, component: ComponentType) -> ComponentHealth {
        let components = self.components.read().unwrap_or_else(|e| e.into_inner());
        components
            .get(&component)
            .map(|s| s.health)
            .unwrap_or(ComponentHealth::Unknown)
    }

    /// Get latest health check for a component
    pub fn get_latest_check(&self, component: ComponentType) -> Option<HealthCheck> {
        let components = self.components.read().unwrap_or_else(|e| e.into_inner());
        components
            .get(&component)
            .and_then(|s| s.get_latest_check())
            .cloned()
    }

    /// Get health history for a component
    pub fn get_history(&self, component: ComponentType, limit: usize) -> Vec<HealthCheck> {
        let components = self.components.read().unwrap_or_else(|e| e.into_inner());
        if let Some(state) = components.get(&component) {
            let len = state.history.len();
            let start = len.saturating_sub(limit);
            state.history[start..].to_vec()
        } else {
            Vec::new()
        }
    }

    /// Get overall system health
    pub fn overall_health(&self) -> ComponentHealth {
        let components = self.components.read().unwrap_or_else(|e| e.into_inner());

        if components.is_empty() {
            return ComponentHealth::Unknown;
        }

        let mut has_failed = false;
        let mut has_unhealthy = false;
        let mut has_degraded = false;

        for state in components.values() {
            match state.health {
                ComponentHealth::Failed => has_failed = true,
                ComponentHealth::Unhealthy => has_unhealthy = true,
                ComponentHealth::Degraded => has_degraded = true,
                _ => {}
            }
        }

        if has_failed {
            ComponentHealth::Failed
        } else if has_unhealthy {
            ComponentHealth::Unhealthy
        } else if has_degraded {
            ComponentHealth::Degraded
        } else {
            ComponentHealth::Healthy
        }
    }

    /// Register an alert callback
    pub fn on_alert<F>(&self, callback: F)
    where
        F: Fn(HealthAlert) + Send + Sync + 'static,
    {
        let mut callbacks = self.callbacks.write().unwrap_or_else(|e| e.into_inner());
        callbacks.push(Arc::new(callback));
    }

    /// Trigger an alert
    fn trigger_alert(
        &self,
        component: ComponentType,
        old_status: ComponentHealth,
        new_status: ComponentHealth,
    ) {
        let alert = HealthAlert {
            component,
            new_status,
            old_status,
            timestamp: SystemTime::now(),
            message: format!(
                "{:?} health changed from {:?} to {:?}",
                component, old_status, new_status
            ),
        };

        let callbacks = self.callbacks.read().unwrap_or_else(|e| e.into_inner());
        for callback in callbacks.iter() {
            callback(alert.clone());
        }
    }

    /// Get statistics for a component
    pub fn get_stats(&self, component: ComponentType) -> Option<ComponentStats> {
        let components = self.components.read().unwrap_or_else(|e| e.into_inner());
        components.get(&component).map(|state| {
            let total_checks = state.history.len();
            let healthy_count = state
                .history
                .iter()
                .filter(|c| c.status == ComponentHealth::Healthy)
                .count();

            let uptime_ratio = if total_checks > 0 {
                healthy_count as f64 / total_checks as f64
            } else {
                0.0
            };

            ComponentStats {
                component,
                current_health: state.health,
                total_checks,
                consecutive_failures: state.consecutive_failures,
                consecutive_successes: state.consecutive_successes,
                uptime_ratio,
                last_check: state.last_check,
            }
        })
    }

    /// Get all component statistics
    pub fn get_all_stats(&self) -> Vec<ComponentStats> {
        let components = self.components.read().unwrap_or_else(|e| e.into_inner());
        components
            .keys()
            .filter_map(|&comp| self.get_stats(comp))
            .collect()
    }
}

/// Statistics for a component
#[derive(Debug, Clone)]
pub struct ComponentStats {
    /// Component type
    pub component: ComponentType,
    /// Current health status
    pub current_health: ComponentHealth,
    /// Total number of health checks performed
    pub total_checks: usize,
    /// Consecutive failures
    pub consecutive_failures: usize,
    /// Consecutive successes
    pub consecutive_successes: usize,
    /// Ratio of healthy checks (0.0 to 1.0)
    pub uptime_ratio: f64,
    /// Last check timestamp
    pub last_check: Instant,
}

/// Helper to build health checks
pub struct HealthCheckBuilder {
    component: ComponentType,
    status: ComponentHealth,
    message: Option<String>,
    metrics: HashMap<String, f64>,
}

impl HealthCheckBuilder {
    /// Create a new health check builder
    pub fn new(component: ComponentType) -> Self {
        Self {
            component,
            status: ComponentHealth::Unknown,
            message: None,
            metrics: HashMap::new(),
        }
    }

    /// Set the health status
    pub fn status(mut self, status: ComponentHealth) -> Self {
        self.status = status;
        self
    }

    /// Set a message
    pub fn message<S: Into<String>>(mut self, message: S) -> Self {
        self.message = Some(message.into());
        self
    }

    /// Add a metric
    pub fn metric<S: Into<String>>(mut self, name: S, value: f64) -> Self {
        self.metrics.insert(name.into(), value);
        self
    }

    /// Build the health check
    pub fn build(self) -> HealthCheck {
        HealthCheck {
            component: self.component,
            status: self.status,
            timestamp: Instant::now(),
            message: self.message,
            metrics: self.metrics,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_monitor_creation() {
        let monitor = HealthMonitor::new(HealthMonitorConfig::default());
        assert_eq!(monitor.overall_health(), ComponentHealth::Unknown);
    }

    #[test]
    fn test_register_component() {
        let monitor = HealthMonitor::new(HealthMonitorConfig::default());
        monitor.register_component(ComponentType::PeerManager, 100);

        assert_eq!(
            monitor.get_health(ComponentType::PeerManager),
            ComponentHealth::Unknown
        );
    }

    #[test]
    fn test_record_health_check() {
        let monitor = HealthMonitor::new(HealthMonitorConfig::default());
        monitor.register_component(ComponentType::PeerManager, 100);

        let check = HealthCheckBuilder::new(ComponentType::PeerManager)
            .status(ComponentHealth::Healthy)
            .build();

        monitor.record_health_check(check);

        assert_eq!(
            monitor.get_health(ComponentType::PeerManager),
            ComponentHealth::Healthy
        );
    }

    #[test]
    fn test_health_check_builder() {
        let check = HealthCheckBuilder::new(ComponentType::WantList)
            .status(ComponentHealth::Healthy)
            .message("All systems operational")
            .metric("queue_size", 42.0)
            .build();

        assert_eq!(check.component, ComponentType::WantList);
        assert_eq!(check.status, ComponentHealth::Healthy);
        assert_eq!(
            check.message.expect("test: message field should be set"),
            "All systems operational"
        );
        assert_eq!(check.metrics.get("queue_size"), Some(&42.0));
    }

    #[test]
    fn test_overall_health() {
        let monitor = HealthMonitor::new(HealthMonitorConfig::default());

        monitor.register_component(ComponentType::PeerManager, 100);
        monitor.register_component(ComponentType::WantList, 100);

        let check1 = HealthCheckBuilder::new(ComponentType::PeerManager)
            .status(ComponentHealth::Healthy)
            .build();
        monitor.record_health_check(check1);

        let check2 = HealthCheckBuilder::new(ComponentType::WantList)
            .status(ComponentHealth::Degraded)
            .build();
        monitor.record_health_check(check2);

        assert_eq!(monitor.overall_health(), ComponentHealth::Degraded);
    }

    #[test]
    fn test_health_history() {
        let monitor = HealthMonitor::new(HealthMonitorConfig::default());
        monitor.register_component(ComponentType::PeerManager, 100);

        for _ in 0..5 {
            let check = HealthCheckBuilder::new(ComponentType::PeerManager)
                .status(ComponentHealth::Healthy)
                .build();
            monitor.record_health_check(check);
        }

        let history = monitor.get_history(ComponentType::PeerManager, 3);
        assert_eq!(history.len(), 3);
    }

    #[test]
    fn test_alert_callback() {
        let monitor = HealthMonitor::new(HealthMonitorConfig::default());
        monitor.register_component(ComponentType::PeerManager, 100);

        let alert_triggered = Arc::new(RwLock::new(false));
        let alert_triggered_clone = alert_triggered.clone();

        monitor.on_alert(move |_alert| {
            *alert_triggered_clone
                .write()
                .unwrap_or_else(|e| e.into_inner()) = true;
        });

        // First check - sets to healthy
        let check1 = HealthCheckBuilder::new(ComponentType::PeerManager)
            .status(ComponentHealth::Healthy)
            .build();
        monitor.record_health_check(check1);

        // Second check - changes to degraded, should trigger alert
        let check2 = HealthCheckBuilder::new(ComponentType::PeerManager)
            .status(ComponentHealth::Degraded)
            .build();
        monitor.record_health_check(check2);

        assert!(*alert_triggered.read().unwrap_or_else(|e| e.into_inner()));
    }

    #[test]
    fn test_component_stats() {
        let monitor = HealthMonitor::new(HealthMonitorConfig::default());
        monitor.register_component(ComponentType::PeerManager, 100);

        for i in 0..10 {
            let status = if i < 8 {
                ComponentHealth::Healthy
            } else {
                ComponentHealth::Degraded
            };

            let check = HealthCheckBuilder::new(ComponentType::PeerManager)
                .status(status)
                .build();
            monitor.record_health_check(check);
        }

        let stats = monitor
            .get_stats(ComponentType::PeerManager)
            .expect("test: stats should exist for registered component");
        assert_eq!(stats.total_checks, 10);
        assert_eq!(stats.uptime_ratio, 0.8); // 8 out of 10 healthy
    }

    #[test]
    fn test_get_all_stats() {
        let monitor = HealthMonitor::new(HealthMonitorConfig::default());
        monitor.register_component(ComponentType::PeerManager, 100);
        monitor.register_component(ComponentType::WantList, 100);

        let stats = monitor.get_all_stats();
        assert_eq!(stats.len(), 2);
    }

    #[test]
    fn test_latest_check() {
        let monitor = HealthMonitor::new(HealthMonitorConfig::default());
        monitor.register_component(ComponentType::PeerManager, 100);

        let check = HealthCheckBuilder::new(ComponentType::PeerManager)
            .status(ComponentHealth::Healthy)
            .message("Test message")
            .build();

        monitor.record_health_check(check.clone());

        let latest = monitor
            .get_latest_check(ComponentType::PeerManager)
            .expect("test: latest check should exist for registered component");
        assert_eq!(latest.status, ComponentHealth::Healthy);
        assert_eq!(latest.message, Some("Test message".to_string()));
    }
}
