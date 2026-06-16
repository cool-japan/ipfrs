//! Health check and liveness/readiness probes
//!
//! This module provides health check endpoints for monitoring IPFRS node status,
//! suitable for Kubernetes liveness and readiness probes.

use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Overall health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    /// System is healthy and ready
    Healthy,
    /// System is degraded but functional
    Degraded,
    /// System is unhealthy
    Unhealthy,
}

/// Component health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Component name
    pub name: String,
    /// Component status
    pub status: HealthStatus,
    /// Optional message
    pub message: Option<String>,
    /// Check duration in milliseconds
    pub duration_ms: f64,
}

/// Overall system health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    /// Overall status
    pub status: HealthStatus,
    /// Uptime in seconds
    pub uptime_seconds: u64,
    /// Individual component health
    pub components: Vec<ComponentHealth>,
    /// Timestamp
    pub timestamp: String,
}

/// Health checker for IPFRS node
pub struct HealthChecker {
    start_time: Instant,
}

impl HealthChecker {
    /// Create a new health checker
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
        }
    }

    /// Get uptime in seconds
    pub fn uptime_seconds(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    /// Check liveness (is the process running?)
    ///
    /// This is a simple check that always returns healthy if called.
    /// Use for Kubernetes liveness probes.
    pub fn check_liveness(&self) -> SystemHealth {
        SystemHealth {
            status: HealthStatus::Healthy,
            uptime_seconds: self.uptime_seconds(),
            components: vec![ComponentHealth {
                name: "process".to_string(),
                status: HealthStatus::Healthy,
                message: Some("Process is running".to_string()),
                duration_ms: 0.0,
            }],
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Check readiness (is the system ready to serve requests?)
    ///
    /// This performs comprehensive checks of all components.
    /// Use for Kubernetes readiness probes.
    pub fn check_readiness(
        &self,
        storage_healthy: bool,
        network_healthy: bool,
        semantic_healthy: bool,
        logic_healthy: bool,
    ) -> SystemHealth {
        let mut components = Vec::new();

        // Storage check
        let storage_start = Instant::now();
        components.push(ComponentHealth {
            name: "storage".to_string(),
            status: if storage_healthy {
                HealthStatus::Healthy
            } else {
                HealthStatus::Unhealthy
            },
            message: if storage_healthy {
                Some("Storage is accessible".to_string())
            } else {
                Some("Storage is not accessible".to_string())
            },
            duration_ms: storage_start.elapsed().as_secs_f64() * 1000.0,
        });

        // Network check
        let network_start = Instant::now();
        components.push(ComponentHealth {
            name: "network".to_string(),
            status: if network_healthy {
                HealthStatus::Healthy
            } else {
                HealthStatus::Degraded
            },
            message: if network_healthy {
                Some("Network is active".to_string())
            } else {
                Some("Network is not active".to_string())
            },
            duration_ms: network_start.elapsed().as_secs_f64() * 1000.0,
        });

        // Semantic search check
        let semantic_start = Instant::now();
        components.push(ComponentHealth {
            name: "semantic".to_string(),
            status: if semantic_healthy {
                HealthStatus::Healthy
            } else {
                HealthStatus::Degraded
            },
            message: if semantic_healthy {
                Some("Semantic search is enabled".to_string())
            } else {
                Some("Semantic search is disabled".to_string())
            },
            duration_ms: semantic_start.elapsed().as_secs_f64() * 1000.0,
        });

        // Logic programming check
        let logic_start = Instant::now();
        components.push(ComponentHealth {
            name: "logic".to_string(),
            status: if logic_healthy {
                HealthStatus::Healthy
            } else {
                HealthStatus::Degraded
            },
            message: if logic_healthy {
                Some("TensorLogic is enabled".to_string())
            } else {
                Some("TensorLogic is disabled".to_string())
            },
            duration_ms: logic_start.elapsed().as_secs_f64() * 1000.0,
        });

        // Determine overall status
        let overall_status = if components
            .iter()
            .any(|c| c.status == HealthStatus::Unhealthy)
        {
            HealthStatus::Unhealthy
        } else if components
            .iter()
            .any(|c| c.status == HealthStatus::Degraded)
        {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };

        SystemHealth {
            status: overall_status,
            uptime_seconds: self.uptime_seconds(),
            components,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_checker_creation() {
        let checker = HealthChecker::new();
        assert!(checker.uptime_seconds() == 0);
    }

    #[test]
    fn test_liveness_check() {
        let checker = HealthChecker::new();
        let health = checker.check_liveness();
        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.components.len(), 1);
        assert_eq!(health.components[0].name, "process");
    }

    #[test]
    fn test_readiness_check_all_healthy() {
        let checker = HealthChecker::new();
        let health = checker.check_readiness(true, true, true, true);
        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.components.len(), 4);
    }

    #[test]
    fn test_readiness_check_storage_unhealthy() {
        let checker = HealthChecker::new();
        let health = checker.check_readiness(false, true, true, true);
        assert_eq!(health.status, HealthStatus::Unhealthy);
        let storage = health
            .components
            .iter()
            .find(|c| c.name == "storage")
            .expect("test: storage component should be present");
        assert_eq!(storage.status, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_readiness_check_degraded() {
        let checker = HealthChecker::new();
        let health = checker.check_readiness(true, false, true, true);
        assert_eq!(health.status, HealthStatus::Degraded);
        let network = health
            .components
            .iter()
            .find(|c| c.name == "network")
            .expect("test: network component should be present");
        assert_eq!(network.status, HealthStatus::Degraded);
    }
}
