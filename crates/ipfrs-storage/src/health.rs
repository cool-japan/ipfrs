//! Health check system for storage backends
//!
//! Provides standardized health checks for all storage backends including:
//! - Liveness checks (is the service running?)
//! - Readiness checks (can the service handle requests?)
//! - Detailed status reporting
//! - Aggregate health across multiple backends
//!
//! ## Example
//! ```no_run
//! use ipfrs_storage::{HealthChecker, HealthStatus};
//!
//! #[tokio::main]
//! async fn main() {
//!     let checker = HealthChecker::new();
//!
//!     let status = checker.check_liveness().await;
//!     println!("Health: {:?}", status);
//! }
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Health status of a component
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Component is healthy and operational
    Healthy,
    /// Component is degraded but operational
    Degraded,
    /// Component is unhealthy and not operational
    Unhealthy,
}

impl HealthStatus {
    /// Check if status is healthy
    pub fn is_healthy(&self) -> bool {
        matches!(self, HealthStatus::Healthy)
    }

    /// Check if status is degraded
    pub fn is_degraded(&self) -> bool {
        matches!(self, HealthStatus::Degraded)
    }

    /// Check if status is unhealthy
    pub fn is_unhealthy(&self) -> bool {
        matches!(self, HealthStatus::Unhealthy)
    }

    /// Check if component can serve requests (healthy or degraded)
    pub fn is_ready(&self) -> bool {
        !self.is_unhealthy()
    }
}

/// Detailed health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    /// Overall status
    pub status: HealthStatus,
    /// Component name
    pub component: String,
    /// Human-readable message
    pub message: String,
    /// When the check was performed
    pub checked_at: String,
    /// Check duration
    pub duration_ms: u64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl HealthCheckResult {
    /// Create a healthy result
    pub fn healthy(component: String, message: String, duration: Duration) -> Self {
        Self {
            status: HealthStatus::Healthy,
            component,
            message,
            checked_at: chrono::Utc::now().to_rfc3339(),
            duration_ms: duration.as_millis() as u64,
            metadata: HashMap::new(),
        }
    }

    /// Create a degraded result
    pub fn degraded(component: String, message: String, duration: Duration) -> Self {
        Self {
            status: HealthStatus::Degraded,
            component,
            message,
            checked_at: chrono::Utc::now().to_rfc3339(),
            duration_ms: duration.as_millis() as u64,
            metadata: HashMap::new(),
        }
    }

    /// Create an unhealthy result
    pub fn unhealthy(component: String, message: String, duration: Duration) -> Self {
        Self {
            status: HealthStatus::Unhealthy,
            component,
            message,
            checked_at: chrono::Utc::now().to_rfc3339(),
            duration_ms: duration.as_millis() as u64,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the result
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Trait for health-checkable components
#[async_trait]
pub trait HealthCheck: Send + Sync {
    /// Perform a liveness check
    ///
    /// Liveness checks verify that the component is running.
    /// A failed liveness check indicates the component should be restarted.
    async fn check_liveness(&self) -> HealthCheckResult;

    /// Perform a readiness check
    ///
    /// Readiness checks verify that the component can handle requests.
    /// A failed readiness check means the component should not receive traffic.
    async fn check_readiness(&self) -> HealthCheckResult;

    /// Get component name
    fn component_name(&self) -> String;
}

/// Aggregate health checker for multiple components
pub struct HealthChecker {
    /// Registered health checks
    checks: Arc<parking_lot::RwLock<Vec<Arc<dyn HealthCheck>>>>,
}

impl HealthChecker {
    /// Create a new health checker
    pub fn new() -> Self {
        Self {
            checks: Arc::new(parking_lot::RwLock::new(Vec::new())),
        }
    }

    /// Register a health check
    pub fn register<H: HealthCheck + 'static>(&self, check: H) {
        self.checks.write().push(Arc::new(check));
    }

    /// Check liveness of all registered components
    pub async fn check_liveness(&self) -> AggregateHealthResult {
        let checks = self.checks.read().clone();
        let mut results = Vec::new();

        for check in checks {
            results.push(check.check_liveness().await);
        }

        AggregateHealthResult::from_results(results)
    }

    /// Check readiness of all registered components
    pub async fn check_readiness(&self) -> AggregateHealthResult {
        let checks = self.checks.read().clone();
        let mut results = Vec::new();

        for check in checks {
            results.push(check.check_readiness().await);
        }

        AggregateHealthResult::from_results(results)
    }

    /// Get detailed status of all components
    pub async fn detailed_status(&self) -> DetailedHealthStatus {
        let checks = self.checks.read().clone();
        let mut liveness_results = Vec::new();
        let mut readiness_results = Vec::new();

        for check in checks {
            liveness_results.push(check.check_liveness().await);
            readiness_results.push(check.check_readiness().await);
        }

        DetailedHealthStatus {
            liveness: AggregateHealthResult::from_results(liveness_results),
            readiness: AggregateHealthResult::from_results(readiness_results),
        }
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregate health result across multiple components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateHealthResult {
    /// Overall status
    pub status: HealthStatus,
    /// Individual component results
    pub components: Vec<HealthCheckResult>,
    /// Total number of components
    pub total_components: usize,
    /// Number of healthy components
    pub healthy_count: usize,
    /// Number of degraded components
    pub degraded_count: usize,
    /// Number of unhealthy components
    pub unhealthy_count: usize,
}

impl AggregateHealthResult {
    /// Create aggregate result from individual results
    pub fn from_results(components: Vec<HealthCheckResult>) -> Self {
        let total_components = components.len();
        let mut healthy_count = 0;
        let mut degraded_count = 0;
        let mut unhealthy_count = 0;

        for result in &components {
            match result.status {
                HealthStatus::Healthy => healthy_count += 1,
                HealthStatus::Degraded => degraded_count += 1,
                HealthStatus::Unhealthy => unhealthy_count += 1,
            }
        }

        // Determine overall status
        let status = if unhealthy_count > 0 {
            HealthStatus::Unhealthy
        } else if degraded_count > 0 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };

        Self {
            status,
            components,
            total_components,
            healthy_count,
            degraded_count,
            unhealthy_count,
        }
    }

    /// Check if all components are healthy
    pub fn all_healthy(&self) -> bool {
        self.status == HealthStatus::Healthy
    }

    /// Check if any component is unhealthy
    pub fn any_unhealthy(&self) -> bool {
        self.unhealthy_count > 0
    }

    /// Get unhealthy components
    pub fn unhealthy_components(&self) -> Vec<&HealthCheckResult> {
        self.components
            .iter()
            .filter(|r| r.status == HealthStatus::Unhealthy)
            .collect()
    }
}

/// Detailed health status with liveness and readiness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedHealthStatus {
    /// Liveness check results
    pub liveness: AggregateHealthResult,
    /// Readiness check results
    pub readiness: AggregateHealthResult,
}

impl DetailedHealthStatus {
    /// Check if system is alive
    pub fn is_alive(&self) -> bool {
        self.liveness.status != HealthStatus::Unhealthy
    }

    /// Check if system is ready
    pub fn is_ready(&self) -> bool {
        self.readiness.status != HealthStatus::Unhealthy
    }
}

/// Simple health check implementation for testing
#[derive(Clone)]
pub struct SimpleHealthCheck {
    name: String,
    is_healthy: Arc<parking_lot::RwLock<bool>>,
}

impl SimpleHealthCheck {
    /// Create a new simple health check
    pub fn new(name: String) -> Self {
        Self {
            name,
            is_healthy: Arc::new(parking_lot::RwLock::new(true)),
        }
    }

    /// Set health status
    pub fn set_healthy(&self, healthy: bool) {
        *self.is_healthy.write() = healthy;
    }
}

#[async_trait]
impl HealthCheck for SimpleHealthCheck {
    async fn check_liveness(&self) -> HealthCheckResult {
        let start = Instant::now();
        let is_healthy = *self.is_healthy.read();
        let duration = start.elapsed();

        if is_healthy {
            HealthCheckResult::healthy(
                self.name.clone(),
                "Component is alive".to_string(),
                duration,
            )
        } else {
            HealthCheckResult::unhealthy(
                self.name.clone(),
                "Component is not alive".to_string(),
                duration,
            )
        }
    }

    async fn check_readiness(&self) -> HealthCheckResult {
        let start = Instant::now();
        let is_healthy = *self.is_healthy.read();
        let duration = start.elapsed();

        if is_healthy {
            HealthCheckResult::healthy(
                self.name.clone(),
                "Component is ready".to_string(),
                duration,
            )
        } else {
            HealthCheckResult::unhealthy(
                self.name.clone(),
                "Component is not ready".to_string(),
                duration,
            )
        }
    }

    fn component_name(&self) -> String {
        self.name.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_status() {
        assert!(HealthStatus::Healthy.is_healthy());
        assert!(!HealthStatus::Degraded.is_healthy());
        assert!(!HealthStatus::Unhealthy.is_healthy());

        assert!(HealthStatus::Healthy.is_ready());
        assert!(HealthStatus::Degraded.is_ready());
        assert!(!HealthStatus::Unhealthy.is_ready());
    }

    #[tokio::test]
    async fn test_simple_health_check() {
        let check = SimpleHealthCheck::new("test".to_string());

        let result = check.check_liveness().await;
        assert!(result.status.is_healthy());

        check.set_healthy(false);
        let result = check.check_liveness().await;
        assert!(result.status.is_unhealthy());
    }

    #[tokio::test]
    async fn test_health_checker_aggregate() {
        let checker = HealthChecker::new();

        let check1 = SimpleHealthCheck::new("component1".to_string());
        let check2 = SimpleHealthCheck::new("component2".to_string());

        checker.register(check1.clone());
        checker.register(check2.clone());

        let result = checker.check_liveness().await;
        assert!(result.all_healthy());
        assert_eq!(result.healthy_count, 2);

        // Make one component unhealthy
        check1.set_healthy(false);

        let result = checker.check_liveness().await;
        assert!(!result.all_healthy());
        assert!(result.any_unhealthy());
        assert_eq!(result.healthy_count, 1);
        assert_eq!(result.unhealthy_count, 1);
    }

    #[tokio::test]
    async fn test_detailed_status() {
        let checker = HealthChecker::new();
        let check = SimpleHealthCheck::new("test".to_string());
        checker.register(check);

        let status = checker.detailed_status().await;
        assert!(status.is_alive());
        assert!(status.is_ready());
    }

    #[tokio::test]
    async fn test_aggregate_health_result() {
        let results = vec![
            HealthCheckResult::healthy(
                "comp1".to_string(),
                "OK".to_string(),
                Duration::from_millis(10),
            ),
            HealthCheckResult::degraded(
                "comp2".to_string(),
                "Slow".to_string(),
                Duration::from_millis(100),
            ),
        ];

        let aggregate = AggregateHealthResult::from_results(results);
        assert_eq!(aggregate.status, HealthStatus::Degraded);
        assert_eq!(aggregate.healthy_count, 1);
        assert_eq!(aggregate.degraded_count, 1);
        assert_eq!(aggregate.unhealthy_count, 0);
    }
}
