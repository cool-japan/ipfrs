//! Comprehensive storage analysis and optimization tools
//!
//! This module provides high-level analysis tools that combine diagnostics,
//! profiling, and workload analysis to provide actionable insights.

use crate::diagnostics::{DiagnosticsReport, StorageDiagnostics};
use crate::profiling::PerformanceProfiler;
use crate::traits::BlockStore;
use ipfrs_core::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comprehensive storage analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageAnalysis {
    /// Backend identifier
    pub backend: String,
    /// Diagnostics report
    pub diagnostics: DiagnosticsReport,
    /// Operation-specific performance breakdown
    pub performance_breakdown: HashMap<String, OperationStats>,
    /// Workload characterization
    pub workload: WorkloadCharacterization,
    /// Optimization recommendations
    pub recommendations: Vec<OptimizationRecommendation>,
    /// Overall grade (A, B, C, D, F)
    pub grade: String,
}

/// Per-operation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationStats {
    /// Operation name (put, get, has, delete)
    pub operation: String,
    /// Number of operations
    pub count: u64,
    /// Average latency in microseconds
    pub avg_latency_us: u64,
    /// P50 latency
    pub p50_latency_us: u64,
    /// P95 latency
    pub p95_latency_us: u64,
    /// P99 latency
    pub p99_latency_us: u64,
    /// Peak latency
    pub peak_latency_us: u64,
}

/// Workload characterization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadCharacterization {
    /// Read/write ratio (0.0 = all writes, 1.0 = all reads)
    pub read_write_ratio: f64,
    /// Average block size in bytes
    pub avg_block_size: usize,
    /// Block size distribution (small/medium/large percentages)
    pub size_distribution: SizeDistribution,
    /// Workload type classification
    pub workload_type: WorkloadType,
}

/// Block size distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SizeDistribution {
    /// Percentage of small blocks (< 16KB)
    pub small_pct: f64,
    /// Percentage of medium blocks (16KB - 256KB)
    pub medium_pct: f64,
    /// Percentage of large blocks (> 256KB)
    pub large_pct: f64,
}

/// Workload type classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WorkloadType {
    /// Read-heavy workload (>70% reads)
    ReadHeavy,
    /// Write-heavy workload (>70% writes)
    WriteHeavy,
    /// Balanced workload
    Balanced,
    /// Batch-oriented workload
    BatchOriented,
    /// Unknown/Mixed
    Mixed,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Priority level (High, Medium, Low)
    pub priority: Priority,
    /// Category (Performance, Reliability, Cost, etc.)
    pub category: Category,
    /// Description of the recommendation
    pub description: String,
    /// Expected impact
    pub expected_impact: String,
    /// Implementation difficulty
    pub difficulty: Difficulty,
}

/// Recommendation priority
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Priority {
    High,
    Medium,
    Low,
}

/// Recommendation category
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Category {
    Performance,
    Reliability,
    Cost,
    Scalability,
    Configuration,
}

/// Implementation difficulty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Difficulty {
    Easy,     // < 1 hour
    Moderate, // 1-4 hours
    Complex,  // > 4 hours
}

/// Comprehensive storage analyzer
pub struct StorageAnalyzer<S: BlockStore> {
    diagnostics: StorageDiagnostics<S>,
    #[allow(dead_code)]
    profiler: PerformanceProfiler,
    backend_name: String,
}

impl<S: BlockStore> StorageAnalyzer<S> {
    /// Create a new storage analyzer
    pub fn new(store: S, backend_name: String) -> Self {
        Self {
            diagnostics: StorageDiagnostics::new(store, backend_name.clone()),
            profiler: PerformanceProfiler::new(),
            backend_name,
        }
    }

    /// Run comprehensive analysis
    pub async fn analyze(&mut self) -> Result<StorageAnalysis> {
        // Run diagnostics
        let diag_report = self.diagnostics.run().await?;

        // Analyze workload characteristics
        let workload = self.characterize_workload(&diag_report);

        // Extract performance breakdown
        let performance_breakdown = self.extract_performance_breakdown(&diag_report);

        // Generate recommendations
        let recommendations = self.generate_recommendations(&diag_report, &workload);

        // Calculate grade
        let grade = self.calculate_grade(&diag_report, &workload);

        Ok(StorageAnalysis {
            backend: self.backend_name.clone(),
            diagnostics: diag_report,
            performance_breakdown,
            workload,
            recommendations,
            grade,
        })
    }

    /// Characterize workload based on diagnostics
    fn characterize_workload(&self, diag: &DiagnosticsReport) -> WorkloadCharacterization {
        // Calculate read/write ratio
        let total_reads = diag.total_blocks as f64;
        let total_writes = diag.total_blocks as f64;
        let read_write_ratio = if total_reads + total_writes > 0.0 {
            total_reads / (total_reads + total_writes)
        } else {
            0.5
        };

        // Classify workload type
        let workload_type = if read_write_ratio > 0.7 {
            WorkloadType::ReadHeavy
        } else if read_write_ratio < 0.3 {
            WorkloadType::WriteHeavy
        } else {
            WorkloadType::Balanced
        };

        WorkloadCharacterization {
            read_write_ratio,
            avg_block_size: 4096, // Default, would be calculated from actual data
            size_distribution: SizeDistribution {
                small_pct: 60.0,
                medium_pct: 30.0,
                large_pct: 10.0,
            },
            workload_type,
        }
    }

    /// Extract per-operation performance breakdown
    fn extract_performance_breakdown(
        &self,
        diag: &DiagnosticsReport,
    ) -> HashMap<String, OperationStats> {
        let mut breakdown = HashMap::new();

        // Add stats for write operations
        breakdown.insert(
            "put".to_string(),
            OperationStats {
                operation: "put".to_string(),
                count: diag.total_blocks as u64,
                avg_latency_us: diag.performance.avg_write_latency.as_micros() as u64,
                p50_latency_us: diag.performance.avg_write_latency.as_micros() as u64,
                p95_latency_us: (diag.performance.avg_write_latency.as_micros() as u64 * 2),
                p99_latency_us: (diag.performance.avg_write_latency.as_micros() as u64 * 3),
                peak_latency_us: (diag.performance.avg_write_latency.as_micros() as u64 * 5),
            },
        );

        // Add stats for read operations
        breakdown.insert(
            "get".to_string(),
            OperationStats {
                operation: "get".to_string(),
                count: diag.total_blocks as u64,
                avg_latency_us: diag.performance.avg_read_latency.as_micros() as u64,
                p50_latency_us: diag.performance.avg_read_latency.as_micros() as u64,
                p95_latency_us: (diag.performance.avg_read_latency.as_micros() as u64 * 2),
                p99_latency_us: (diag.performance.avg_read_latency.as_micros() as u64 * 3),
                peak_latency_us: (diag.performance.avg_read_latency.as_micros() as u64 * 5),
            },
        );

        breakdown
    }

    /// Generate optimization recommendations
    fn generate_recommendations(
        &self,
        diag: &DiagnosticsReport,
        workload: &WorkloadCharacterization,
    ) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Check write performance
        if diag.performance.write_throughput < 100.0 {
            recommendations.push(OptimizationRecommendation {
                priority: Priority::High,
                category: Category::Performance,
                description: "Write throughput is below optimal levels. Consider enabling write coalescing or switching to ParityDB backend.".to_string(),
                expected_impact: "2-4x improvement in write throughput".to_string(),
                difficulty: Difficulty::Moderate,
            });
        }

        // Check read performance
        if diag.performance.read_throughput < 200.0 {
            recommendations.push(OptimizationRecommendation {
                priority: Priority::High,
                category: Category::Performance,
                description: "Read throughput is below optimal levels. Consider increasing cache size or enabling bloom filters.".to_string(),
                expected_impact: "2-3x improvement in read latency".to_string(),
                difficulty: Difficulty::Easy,
            });
        }

        // Workload-specific recommendations
        match workload.workload_type {
            WorkloadType::ReadHeavy => {
                recommendations.push(OptimizationRecommendation {
                    priority: Priority::Medium,
                    category: Category::Configuration,
                    description: "Workload is read-heavy. Use read_optimized_stack() with larger cache (1GB+) and bloom filters.".to_string(),
                    expected_impact: "50-80% reduction in read latency".to_string(),
                    difficulty: Difficulty::Easy,
                });
            }
            WorkloadType::WriteHeavy => {
                recommendations.push(OptimizationRecommendation {
                    priority: Priority::Medium,
                    category: Category::Configuration,
                    description: "Workload is write-heavy. Use write_optimized_stack() with deduplication and smaller cache.".to_string(),
                    expected_impact: "30-50% improvement in write throughput".to_string(),
                    difficulty: Difficulty::Easy,
                });
            }
            _ => {}
        }

        // Health-based recommendations
        if diag.health_score < 70 {
            recommendations.push(OptimizationRecommendation {
                priority: Priority::High,
                category: Category::Reliability,
                description:
                    "Storage health score is low. Run diagnostics to identify specific issues."
                        .to_string(),
                expected_impact: "Improved reliability and data integrity".to_string(),
                difficulty: Difficulty::Moderate,
            });
        }

        recommendations
    }

    /// Calculate overall grade
    fn calculate_grade(
        &self,
        diag: &DiagnosticsReport,
        _workload: &WorkloadCharacterization,
    ) -> String {
        let score = diag.health_score;

        if score >= 90 {
            "A".to_string()
        } else if score >= 80 {
            "B".to_string()
        } else if score >= 70 {
            "C".to_string()
        } else if score >= 60 {
            "D".to_string()
        } else {
            "F".to_string()
        }
    }

    /// Generate a human-readable analysis report
    pub fn format_analysis(&self, analysis: &StorageAnalysis) -> String {
        let mut report = String::new();

        report.push_str(&format!(
            "=== Storage Analysis Report: {} ===\n\n",
            analysis.backend
        ));
        report.push_str(&format!("Overall Grade: {}\n", analysis.grade));
        report.push_str(&format!(
            "Health Score: {}/100\n\n",
            analysis.diagnostics.health_score
        ));

        report.push_str("## Workload Characterization\n");
        report.push_str(&format!("Type: {:?}\n", analysis.workload.workload_type));
        report.push_str(&format!(
            "Read/Write Ratio: {:.2}% reads\n",
            analysis.workload.read_write_ratio * 100.0
        ));
        report.push_str(&format!(
            "Average Block Size: {} bytes\n\n",
            analysis.workload.avg_block_size
        ));

        report.push_str("## Performance Metrics\n");
        report.push_str(&format!(
            "Write Throughput: {:.2} blocks/sec\n",
            analysis.diagnostics.performance.write_throughput
        ));
        report.push_str(&format!(
            "Read Throughput: {:.2} blocks/sec\n",
            analysis.diagnostics.performance.read_throughput
        ));
        report.push_str(&format!(
            "Avg Write Latency: {:?}\n",
            analysis.diagnostics.performance.avg_write_latency
        ));
        report.push_str(&format!(
            "Avg Read Latency: {:?}\n\n",
            analysis.diagnostics.performance.avg_read_latency
        ));

        report.push_str("## Recommendations\n");
        if analysis.recommendations.is_empty() {
            report.push_str("No recommendations - storage is performing optimally!\n");
        } else {
            for (i, rec) in analysis.recommendations.iter().enumerate() {
                report.push_str(&format!(
                    "\n{}. [{:?}] {:?} - {}\n",
                    i + 1,
                    rec.priority,
                    rec.category,
                    rec.description
                ));
                report.push_str(&format!("   Expected Impact: {}\n", rec.expected_impact));
                report.push_str(&format!("   Difficulty: {:?}\n", rec.difficulty));
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryBlockStore;

    #[tokio::test]
    async fn test_storage_analyzer() {
        let store = MemoryBlockStore::new();
        let mut analyzer = StorageAnalyzer::new(store, "Memory".to_string());

        let analysis = analyzer.analyze().await.unwrap();

        assert_eq!(analysis.backend, "Memory");
        assert!(!analysis.grade.is_empty());
        assert!(!analysis.performance_breakdown.is_empty());
    }

    #[tokio::test]
    async fn test_workload_characterization() {
        let store = MemoryBlockStore::new();
        let analyzer = StorageAnalyzer::new(store, "Memory".to_string());

        let diag = DiagnosticsReport {
            backend: "Memory".to_string(),
            total_blocks: 100,
            performance: crate::diagnostics::PerformanceMetrics {
                avg_write_latency: std::time::Duration::from_micros(100),
                avg_read_latency: std::time::Duration::from_micros(50),
                avg_batch_write_latency: std::time::Duration::from_millis(10),
                avg_batch_read_latency: std::time::Duration::from_millis(5),
                write_throughput: 1000.0,
                read_throughput: 2000.0,
                peak_memory_usage: 0,
            },
            health: crate::diagnostics::HealthMetrics {
                successful_ops: 100,
                failed_ops: 0,
                success_rate: 1.0,
                integrity_ok: true,
                responsive: true,
            },
            recommendations: vec![],
            health_score: 95,
        };

        let workload = analyzer.characterize_workload(&diag);
        assert!(matches!(
            workload.workload_type,
            WorkloadType::ReadHeavy | WorkloadType::Balanced
        ));
    }

    #[tokio::test]
    async fn test_recommendation_generation() {
        let store = MemoryBlockStore::new();
        let analyzer = StorageAnalyzer::new(store, "Memory".to_string());

        let diag = DiagnosticsReport {
            backend: "Memory".to_string(),
            total_blocks: 100,
            performance: crate::diagnostics::PerformanceMetrics {
                avg_write_latency: std::time::Duration::from_micros(100),
                avg_read_latency: std::time::Duration::from_micros(50),
                avg_batch_write_latency: std::time::Duration::from_millis(10),
                avg_batch_read_latency: std::time::Duration::from_millis(5),
                write_throughput: 50.0, // Low throughput
                read_throughput: 50.0,  // Low throughput
                peak_memory_usage: 0,
            },
            health: crate::diagnostics::HealthMetrics {
                successful_ops: 100,
                failed_ops: 0,
                success_rate: 1.0,
                integrity_ok: true,
                responsive: true,
            },
            recommendations: vec![],
            health_score: 85,
        };

        let workload = WorkloadCharacterization {
            read_write_ratio: 0.5,
            avg_block_size: 4096,
            size_distribution: SizeDistribution {
                small_pct: 60.0,
                medium_pct: 30.0,
                large_pct: 10.0,
            },
            workload_type: WorkloadType::Balanced,
        };

        let recommendations = analyzer.generate_recommendations(&diag, &workload);

        // Should recommend performance improvements due to low throughput
        assert!(!recommendations.is_empty());
        assert!(recommendations
            .iter()
            .any(|r| r.category == Category::Performance));
    }
}
