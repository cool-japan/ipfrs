//! Network health check endpoints
//!
//! Provides comprehensive health status reporting including:
//! - Overall network health
//! - Component health (DHT, connections, etc.)
//! - Historical health tracking
//! - Health check HTTP endpoints (optional)

use crate::{DhtHealth, DhtHealthStatus, NetworkMetrics};
use serde::Serialize;
use std::time::Instant;

/// Overall network health status
#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum NetworkHealthStatus {
    /// All systems operational
    Healthy,
    /// Some degradation but operational
    Degraded,
    /// Critical issues affecting operation
    Unhealthy,
    /// Not enough data to determine health
    Unknown,
}

/// Component health status
#[derive(Debug, Clone, Serialize)]
pub struct ComponentHealth {
    /// Component name
    pub name: String,
    /// Component status
    pub status: NetworkHealthStatus,
    /// Optional message
    pub message: Option<String>,
    /// Health score (0.0 - 1.0)
    pub score: f64,
}

/// Complete network health report
#[derive(Debug, Clone, Serialize)]
pub struct NetworkHealth {
    /// Overall status
    pub status: NetworkHealthStatus,
    /// Overall health score (0.0 - 1.0)
    pub score: f64,
    /// Component health details
    pub components: Vec<ComponentHealth>,
    /// Time of health check
    pub timestamp: u64,
    /// Uptime in seconds
    pub uptime_secs: u64,
}

impl NetworkHealth {
    /// Check if the network is healthy
    pub fn is_healthy(&self) -> bool {
        self.status == NetworkHealthStatus::Healthy
    }

    /// Check if the network is degraded
    pub fn is_degraded(&self) -> bool {
        self.status == NetworkHealthStatus::Degraded
    }

    /// Check if the network is unhealthy
    pub fn is_unhealthy(&self) -> bool {
        self.status == NetworkHealthStatus::Unhealthy
    }
}

/// Health checker for network components
pub struct HealthChecker {
    /// Last health check result
    last_check: parking_lot::RwLock<Option<NetworkHealth>>,
    /// Health check history (last 100 checks)
    history: parking_lot::RwLock<Vec<(Instant, NetworkHealthStatus)>>,
    /// Maximum history size
    max_history: usize,
}

impl HealthChecker {
    /// Create a new health checker
    pub fn new() -> Self {
        Self {
            last_check: parking_lot::RwLock::new(None),
            history: parking_lot::RwLock::new(Vec::new()),
            max_history: 100,
        }
    }

    /// Perform a health check
    pub fn check_health(
        &self,
        metrics: &NetworkMetrics,
        dht_health: Option<&DhtHealth>,
    ) -> NetworkHealth {
        let mut components = Vec::new();
        let mut total_score = 0.0;
        let mut component_count = 0;

        // Check connection health
        let connection_health = self.check_connection_health(metrics);
        total_score += connection_health.score;
        component_count += 1;
        components.push(connection_health);

        // Check DHT health if available
        if let Some(dht) = dht_health {
            let dht_component = self.check_dht_health(dht);
            total_score += dht_component.score;
            component_count += 1;
            components.push(dht_component);
        }

        // Check bandwidth health
        let bandwidth_health = self.check_bandwidth_health(metrics);
        total_score += bandwidth_health.score;
        component_count += 1;
        components.push(bandwidth_health);

        // Calculate overall score and status
        let overall_score = if component_count > 0 {
            total_score / component_count as f64
        } else {
            0.0
        };

        let overall_status = if overall_score >= 0.8 {
            NetworkHealthStatus::Healthy
        } else if overall_score >= 0.5 {
            NetworkHealthStatus::Degraded
        } else if overall_score > 0.0 {
            NetworkHealthStatus::Unhealthy
        } else {
            NetworkHealthStatus::Unknown
        };

        let health = NetworkHealth {
            status: overall_status.clone(),
            score: overall_score,
            components,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time is after UNIX epoch")
                .as_secs(),
            uptime_secs: metrics.uptime().as_secs(),
        };

        // Store in history
        let mut history = self.history.write();
        history.push((Instant::now(), overall_status));
        if history.len() > self.max_history {
            history.remove(0);
        }

        // Store last check
        *self.last_check.write() = Some(health.clone());

        health
    }

    /// Get the last health check result
    pub fn last_health(&self) -> Option<NetworkHealth> {
        self.last_check.read().clone()
    }

    /// Get health history summary
    pub fn health_history(&self) -> HealthHistory {
        let history = self.history.read();
        let total = history.len();

        if total == 0 {
            return HealthHistory {
                total_checks: 0,
                healthy_count: 0,
                degraded_count: 0,
                unhealthy_count: 0,
                unknown_count: 0,
                healthy_percentage: 0.0,
            };
        }

        let mut healthy_count = 0;
        let mut degraded_count = 0;
        let mut unhealthy_count = 0;
        let mut unknown_count = 0;

        for (_, status) in history.iter() {
            match status {
                NetworkHealthStatus::Healthy => healthy_count += 1,
                NetworkHealthStatus::Degraded => degraded_count += 1,
                NetworkHealthStatus::Unhealthy => unhealthy_count += 1,
                NetworkHealthStatus::Unknown => unknown_count += 1,
            }
        }

        HealthHistory {
            total_checks: total,
            healthy_count,
            degraded_count,
            unhealthy_count,
            unknown_count,
            healthy_percentage: (healthy_count as f64 / total as f64) * 100.0,
        }
    }

    /// Check connection health
    fn check_connection_health(&self, metrics: &NetworkMetrics) -> ComponentHealth {
        let snapshot = metrics.connections().snapshot();
        let total = snapshot.total_established;
        let failed = snapshot.total_failed;
        let active = snapshot.active;

        let success_rate = if total > 0 {
            (total - failed) as f64 / total as f64
        } else {
            1.0 // No connections yet, assume healthy
        };

        let has_connections = active > 0;

        let score = if !has_connections && total == 0 {
            0.5 // Starting up, no connections yet
        } else if !has_connections {
            0.3 // Had connections but lost them all
        } else {
            success_rate
        };

        let status = if score >= 0.8 {
            NetworkHealthStatus::Healthy
        } else if score >= 0.5 {
            NetworkHealthStatus::Degraded
        } else {
            NetworkHealthStatus::Unhealthy
        };

        let message = if !has_connections && total > 0 {
            Some("No active connections".to_string())
        } else if success_rate < 0.5 {
            Some(format!(
                "High connection failure rate: {:.1}%",
                (1.0 - success_rate) * 100.0
            ))
        } else {
            None
        };

        ComponentHealth {
            name: "connections".to_string(),
            status,
            message,
            score,
        }
    }

    /// Check DHT health
    fn check_dht_health(&self, dht_health: &DhtHealth) -> ComponentHealth {
        let status = match dht_health.status {
            DhtHealthStatus::Healthy => NetworkHealthStatus::Healthy,
            DhtHealthStatus::Degraded => NetworkHealthStatus::Degraded,
            DhtHealthStatus::Unhealthy => NetworkHealthStatus::Unhealthy,
            DhtHealthStatus::Unknown => NetworkHealthStatus::Unknown,
        };

        let message = if dht_health.peer_count == 0 {
            Some("No peers in routing table".to_string())
        } else if dht_health.query_success_rate < 0.5 {
            Some(format!(
                "Low query success rate: {:.1}%",
                dht_health.query_success_rate * 100.0
            ))
        } else {
            None
        };

        ComponentHealth {
            name: "dht".to_string(),
            status,
            message,
            score: dht_health.health_score,
        }
    }

    /// Check bandwidth health
    fn check_bandwidth_health(&self, metrics: &NetworkMetrics) -> ComponentHealth {
        let snapshot = metrics.bandwidth().snapshot();
        let total_traffic = snapshot.total_sent + snapshot.total_received;

        // Simple heuristic: if we have active connections but no traffic, something might be wrong
        let connections = metrics.connections().active();

        let score = if connections == 0 {
            0.8 // No connections, can't judge bandwidth
        } else if total_traffic == 0 {
            0.5 // Have connections but no traffic (might be normal for new connections)
        } else {
            1.0 // Have traffic, all good
        };

        let status = if score >= 0.8 {
            NetworkHealthStatus::Healthy
        } else if score >= 0.5 {
            NetworkHealthStatus::Degraded
        } else {
            NetworkHealthStatus::Unhealthy
        };

        ComponentHealth {
            name: "bandwidth".to_string(),
            status,
            message: None,
            score,
        }
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Health history summary
#[derive(Debug, Clone, Serialize)]
pub struct HealthHistory {
    /// Total health checks performed
    pub total_checks: usize,
    /// Number of healthy checks
    pub healthy_count: usize,
    /// Number of degraded checks
    pub degraded_count: usize,
    /// Number of unhealthy checks
    pub unhealthy_count: usize,
    /// Number of unknown checks
    pub unknown_count: usize,
    /// Percentage of healthy checks
    pub healthy_percentage: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::NetworkMetrics;

    #[test]
    fn test_health_checker_creation() {
        let checker = HealthChecker::new();
        assert!(checker.last_health().is_none());
    }

    #[test]
    fn test_health_check_no_connections() {
        let checker = HealthChecker::new();
        let metrics = NetworkMetrics::new();

        let health = checker.check_health(&metrics, None);

        // Should be degraded or unknown with no connections
        assert!(
            health.status == NetworkHealthStatus::Degraded
                || health.status == NetworkHealthStatus::Unknown
        );
    }

    #[test]
    fn test_health_check_with_connections() {
        let checker = HealthChecker::new();
        let metrics = NetworkMetrics::new();

        // Simulate successful connections
        metrics.connections().connection_established(true);
        metrics.connections().connection_established(false);

        let health = checker.check_health(&metrics, None);

        // Should be healthy with successful connections
        assert_eq!(health.components.len(), 2); // connections + bandwidth
        assert_eq!(health.components[0].name, "connections");
    }

    #[test]
    fn test_health_history() {
        let checker = HealthChecker::new();
        let metrics = NetworkMetrics::new();

        // Perform multiple checks
        for _ in 0..5 {
            checker.check_health(&metrics, None);
        }

        let history = checker.health_history();
        assert_eq!(history.total_checks, 5);
    }

    #[test]
    fn test_health_status_determination() {
        let checker = HealthChecker::new();
        let metrics = NetworkMetrics::new();

        // Add connections and traffic
        metrics.connections().connection_established(true);
        metrics.bandwidth().record_sent(1000);
        metrics.bandwidth().record_received(2000);

        let health = checker.check_health(&metrics, None);

        // Should be healthy with active connections and traffic
        assert!(health.score > 0.5);
    }

    #[test]
    fn test_last_health_stored() {
        let checker = HealthChecker::new();
        let metrics = NetworkMetrics::new();

        let health1 = checker.check_health(&metrics, None);
        let last = checker
            .last_health()
            .expect("test: last health should be stored after check");

        assert_eq!(health1.timestamp, last.timestamp);
        assert_eq!(health1.score, last.score);
    }
}
