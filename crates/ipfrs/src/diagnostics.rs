//! Diagnostic utilities for IPFRS nodes
//!
//! This module provides comprehensive diagnostic information about node health,
//! resource usage, and performance metrics.

use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

/// Comprehensive node diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDiagnostics {
    /// Timestamp when diagnostics were collected
    pub timestamp: SystemTime,
    /// Node uptime
    pub uptime: Duration,
    /// Storage diagnostics
    pub storage: StorageDiagnostics,
    /// Semantic router diagnostics (if enabled)
    pub semantic: Option<SemanticDiagnostics>,
    /// TensorLogic diagnostics (if enabled)
    pub tensorlogic: Option<TensorLogicDiagnostics>,
    /// Network diagnostics (if enabled)
    pub network: Option<NetworkDiagnostics>,
    /// System resource usage
    pub resources: ResourceUsage,
}

/// Storage-related diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageDiagnostics {
    /// Total number of blocks stored
    pub total_blocks: u64,
    /// Total size in bytes
    pub total_bytes: u64,
    /// Average block size
    pub avg_block_size: u64,
    /// Storage path
    pub storage_path: String,
    /// Database health status
    pub health: HealthStatus,
}

/// Semantic router diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticDiagnostics {
    /// Number of indexed vectors
    pub num_vectors: usize,
    /// Index dimensions
    pub dimensions: usize,
    /// Index health status
    pub health: HealthStatus,
    /// Cache hit rate (if available)
    pub cache_hit_rate: Option<f64>,
}

/// TensorLogic diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorLogicDiagnostics {
    /// Number of facts in knowledge base
    pub num_facts: usize,
    /// Number of rules
    pub num_rules: usize,
    /// Knowledge base health status
    pub health: HealthStatus,
    /// Average inference time (if available)
    pub avg_inference_ms: Option<f64>,
}

/// Network diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkDiagnostics {
    /// Number of connected peers
    pub connected_peers: usize,
    /// Network health status
    pub health: HealthStatus,
    /// Bytes sent
    pub bytes_sent: u64,
    /// Bytes received
    pub bytes_received: u64,
}

/// System resource usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Memory usage in bytes (approximate)
    pub memory_bytes: u64,
    /// CPU usage percentage (if available)
    pub cpu_percent: Option<f64>,
}

/// Health status indicator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Component is healthy
    Healthy,
    /// Component is degraded but functional
    Degraded,
    /// Component is unhealthy
    Unhealthy,
    /// Component status is unknown
    Unknown,
}

impl HealthStatus {
    /// Check if the status is healthy
    pub fn is_healthy(&self) -> bool {
        matches!(self, HealthStatus::Healthy)
    }

    /// Check if the status indicates problems
    pub fn has_issues(&self) -> bool {
        matches!(self, HealthStatus::Degraded | HealthStatus::Unhealthy)
    }
}

/// Diagnostic recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticRecommendation {
    /// Recommendation severity
    pub severity: RecommendationSeverity,
    /// Component this recommendation applies to
    pub component: String,
    /// Recommendation message
    pub message: String,
    /// Optional action to take
    pub action: Option<String>,
}

/// Recommendation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationSeverity {
    /// Informational
    Info,
    /// Warning
    Warning,
    /// Critical issue
    Critical,
}

/// Diagnostic analyzer that generates recommendations
pub struct DiagnosticAnalyzer;

impl DiagnosticAnalyzer {
    /// Analyze diagnostics and generate recommendations
    pub fn analyze(diagnostics: &NodeDiagnostics) -> Vec<DiagnosticRecommendation> {
        let mut recommendations = Vec::new();

        // Check storage health
        if diagnostics.storage.health.has_issues() {
            recommendations.push(DiagnosticRecommendation {
                severity: RecommendationSeverity::Critical,
                component: "Storage".to_string(),
                message: "Storage health is degraded".to_string(),
                action: Some("Check disk space and database integrity".to_string()),
            });
        }

        // Check storage utilization
        if diagnostics.storage.total_blocks > 1_000_000 {
            recommendations.push(DiagnosticRecommendation {
                severity: RecommendationSeverity::Info,
                component: "Storage".to_string(),
                message: format!(
                    "Large number of blocks: {}",
                    diagnostics.storage.total_blocks
                ),
                action: Some("Consider running garbage collection".to_string()),
            });
        }

        // Check semantic router
        if let Some(ref semantic) = diagnostics.semantic {
            if semantic.health.has_issues() {
                recommendations.push(DiagnosticRecommendation {
                    severity: RecommendationSeverity::Warning,
                    component: "Semantic".to_string(),
                    message: "Semantic router health is degraded".to_string(),
                    action: Some("Check index integrity and rebuild if necessary".to_string()),
                });
            }

            if semantic.num_vectors > 100_000 {
                recommendations.push(DiagnosticRecommendation {
                    severity: RecommendationSeverity::Info,
                    component: "Semantic".to_string(),
                    message: format!("Large vector index: {} vectors", semantic.num_vectors),
                    action: Some("Consider index optimization or sharding".to_string()),
                });
            }
        }

        // Check TensorLogic
        if let Some(ref logic) = diagnostics.tensorlogic {
            if logic.health.has_issues() {
                recommendations.push(DiagnosticRecommendation {
                    severity: RecommendationSeverity::Warning,
                    component: "TensorLogic".to_string(),
                    message: "Knowledge base health is degraded".to_string(),
                    action: Some("Verify KB integrity and reload if necessary".to_string()),
                });
            }

            if logic.num_facts > 10_000 {
                recommendations.push(DiagnosticRecommendation {
                    severity: RecommendationSeverity::Info,
                    component: "TensorLogic".to_string(),
                    message: format!("Large knowledge base: {} facts", logic.num_facts),
                    action: Some("Consider query optimization or KB pruning".to_string()),
                });
            }
        }

        // Check network
        if let Some(ref network) = diagnostics.network {
            if network.health.has_issues() {
                recommendations.push(DiagnosticRecommendation {
                    severity: RecommendationSeverity::Warning,
                    component: "Network".to_string(),
                    message: "Network health is degraded".to_string(),
                    action: Some("Check network connectivity and peer connections".to_string()),
                });
            }

            if network.connected_peers == 0 {
                recommendations.push(DiagnosticRecommendation {
                    severity: RecommendationSeverity::Warning,
                    component: "Network".to_string(),
                    message: "No connected peers".to_string(),
                    action: Some("Check bootstrap nodes and network configuration".to_string()),
                });
            }
        }

        // Check memory usage
        if diagnostics.resources.memory_bytes > 1_000_000_000 {
            // > 1GB
            recommendations.push(DiagnosticRecommendation {
                severity: RecommendationSeverity::Info,
                component: "Resources".to_string(),
                message: format!(
                    "High memory usage: {} MB",
                    diagnostics.resources.memory_bytes / 1_000_000
                ),
                action: Some("Consider reducing cache sizes or restarting node".to_string()),
            });
        }

        recommendations
    }

    /// Generate a health report string
    pub fn health_report(diagnostics: &NodeDiagnostics) -> String {
        let mut report = String::new();
        report.push_str("=== IPFRS Node Health Report ===\n\n");

        report.push_str(&format!("Uptime: {:?}\n", diagnostics.uptime));
        report.push_str(&format!(
            "Storage: {:?} ({} blocks, {} bytes)\n",
            diagnostics.storage.health,
            diagnostics.storage.total_blocks,
            diagnostics.storage.total_bytes
        ));

        if let Some(ref semantic) = diagnostics.semantic {
            report.push_str(&format!(
                "Semantic: {:?} ({} vectors, {} dims)\n",
                semantic.health, semantic.num_vectors, semantic.dimensions
            ));
        }

        if let Some(ref logic) = diagnostics.tensorlogic {
            report.push_str(&format!(
                "TensorLogic: {:?} ({} facts, {} rules)\n",
                logic.health, logic.num_facts, logic.num_rules
            ));
        }

        if let Some(ref network) = diagnostics.network {
            report.push_str(&format!(
                "Network: {:?} ({} peers)\n",
                network.health, network.connected_peers
            ));
        }

        report.push_str(&format!(
            "Memory: {} MB\n",
            diagnostics.resources.memory_bytes / 1_000_000
        ));

        report.push_str("\n--- Recommendations ---\n");
        let recommendations = Self::analyze(diagnostics);
        if recommendations.is_empty() {
            report.push_str("No issues detected\n");
        } else {
            for rec in recommendations {
                report.push_str(&format!(
                    "[{:?}] {}: {}\n",
                    rec.severity, rec.component, rec.message
                ));
                if let Some(ref action) = rec.action {
                    report.push_str(&format!("  Action: {}\n", action));
                }
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status() {
        assert!(HealthStatus::Healthy.is_healthy());
        assert!(!HealthStatus::Degraded.is_healthy());
        assert!(HealthStatus::Degraded.has_issues());
        assert!(HealthStatus::Unhealthy.has_issues());
    }

    #[test]
    fn test_diagnostic_analyzer_no_issues() {
        let diagnostics = NodeDiagnostics {
            timestamp: SystemTime::now(),
            uptime: Duration::from_secs(60),
            storage: StorageDiagnostics {
                total_blocks: 100,
                total_bytes: 10000,
                avg_block_size: 100,
                storage_path: "/tmp/ipfrs".to_string(),
                health: HealthStatus::Healthy,
            },
            semantic: Some(SemanticDiagnostics {
                num_vectors: 50,
                dimensions: 768,
                health: HealthStatus::Healthy,
                cache_hit_rate: Some(0.8),
            }),
            tensorlogic: Some(TensorLogicDiagnostics {
                num_facts: 20,
                num_rules: 5,
                health: HealthStatus::Healthy,
                avg_inference_ms: Some(1.5),
            }),
            network: Some(NetworkDiagnostics {
                connected_peers: 5,
                health: HealthStatus::Healthy,
                bytes_sent: 1000,
                bytes_received: 2000,
            }),
            resources: ResourceUsage {
                memory_bytes: 100_000_000, // 100MB
                cpu_percent: Some(10.0),
            },
        };

        let recommendations = DiagnosticAnalyzer::analyze(&diagnostics);
        assert_eq!(recommendations.len(), 0);
    }

    #[test]
    fn test_diagnostic_analyzer_storage_issues() {
        let diagnostics = NodeDiagnostics {
            timestamp: SystemTime::now(),
            uptime: Duration::from_secs(60),
            storage: StorageDiagnostics {
                total_blocks: 2_000_000,
                total_bytes: 10_000_000_000,
                avg_block_size: 5000,
                storage_path: "/tmp/ipfrs".to_string(),
                health: HealthStatus::Degraded,
            },
            semantic: None,
            tensorlogic: None,
            network: None,
            resources: ResourceUsage {
                memory_bytes: 100_000_000,
                cpu_percent: None,
            },
        };

        let recommendations = DiagnosticAnalyzer::analyze(&diagnostics);
        assert!(recommendations.len() >= 2);
        assert!(recommendations
            .iter()
            .any(|r| r.severity == RecommendationSeverity::Critical));
    }

    #[test]
    fn test_health_report_generation() {
        let diagnostics = NodeDiagnostics {
            timestamp: SystemTime::now(),
            uptime: Duration::from_secs(3600),
            storage: StorageDiagnostics {
                total_blocks: 100,
                total_bytes: 10000,
                avg_block_size: 100,
                storage_path: "/tmp/ipfrs".to_string(),
                health: HealthStatus::Healthy,
            },
            semantic: None,
            tensorlogic: None,
            network: None,
            resources: ResourceUsage {
                memory_bytes: 100_000_000,
                cpu_percent: Some(5.0),
            },
        };

        let report = DiagnosticAnalyzer::health_report(&diagnostics);
        assert!(report.contains("Health Report"));
        assert!(report.contains("Storage"));
        assert!(report.contains("Healthy"));
    }
}
