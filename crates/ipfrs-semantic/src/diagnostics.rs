//! Index diagnostics and health monitoring
//!
//! This module provides comprehensive diagnostic tools for monitoring
//! index health, detecting performance issues, and providing actionable insights.

use crate::hnsw::VectorIndex;
use std::time::{Duration, Instant};

/// Overall health status of an index
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Index is healthy and performing optimally
    Healthy,
    /// Index has minor issues but is functional
    Warning,
    /// Index has significant issues affecting performance
    Degraded,
    /// Index is critically impaired
    Critical,
}

/// Detailed diagnostic report for an index
#[derive(Debug, Clone)]
pub struct DiagnosticReport {
    /// Overall health status
    pub status: HealthStatus,
    /// Index size (number of vectors)
    pub size: usize,
    /// Memory usage estimate in bytes
    pub memory_usage: usize,
    /// Issues detected
    pub issues: Vec<DiagnosticIssue>,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
    /// Performance metrics
    pub performance: PerformanceMetrics,
}

/// A specific issue detected during diagnostics
#[derive(Debug, Clone)]
pub struct DiagnosticIssue {
    /// Severity level
    pub severity: IssueSeverity,
    /// Issue category
    pub category: IssueCategory,
    /// Human-readable description
    pub description: String,
    /// Suggested fix or mitigation
    pub suggested_fix: Option<String>,
}

/// Severity of a diagnostic issue
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IssueSeverity {
    /// Informational message
    Info,
    /// Warning - should be addressed
    Warning,
    /// Error - significantly impacts functionality
    Error,
    /// Critical - immediate attention required
    Critical,
}

/// Category of diagnostic issue
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueCategory {
    /// Memory-related issues
    Memory,
    /// Performance-related issues
    Performance,
    /// Configuration issues
    Configuration,
    /// Data quality issues
    DataQuality,
    /// Index structure issues
    IndexStructure,
}

/// Performance metrics for the index
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Average query latency (if available)
    pub avg_query_latency: Option<Duration>,
    /// Cache hit rate (0.0 - 1.0)
    pub cache_hit_rate: Option<f32>,
    /// Estimated queries per second capacity
    pub estimated_qps: Option<f32>,
}

/// Run comprehensive diagnostics on an index
pub fn diagnose_index(index: &VectorIndex) -> DiagnosticReport {
    let mut issues = Vec::new();
    let mut recommendations = Vec::new();

    let size = index.len();
    let dimension = index.dimension();

    // Estimate memory usage
    // Each vector: dimension * 4 bytes (f32) + overhead
    // HNSW graph: ~(M * 2 * size * 8 bytes) for connections
    let vector_memory = size * dimension * 4;
    let graph_memory = size * 16 * 8; // Assuming M=16
    let overhead = size * 100; // Mappings and other overhead
    let memory_usage = vector_memory + graph_memory + overhead;

    // Check for size-related issues
    if size == 0 {
        issues.push(DiagnosticIssue {
            severity: IssueSeverity::Warning,
            category: IssueCategory::IndexStructure,
            description: "Index is empty".to_string(),
            suggested_fix: Some("Add vectors to the index before querying".to_string()),
        });
    } else if size > 10_000_000 {
        issues.push(DiagnosticIssue {
            severity: IssueSeverity::Warning,
            category: IssueCategory::Performance,
            description: format!("Very large index ({} vectors)", size),
            suggested_fix: Some("Consider using DiskANN for datasets > 10M vectors".to_string()),
        });
        recommendations
            .push("Consider partitioning the index or using distributed search".to_string());
    }

    // Check memory usage
    if memory_usage > 10 * 1024 * 1024 * 1024 {
        // > 10GB
        issues.push(DiagnosticIssue {
            severity: IssueSeverity::Warning,
            category: IssueCategory::Memory,
            description: format!("High memory usage: ~{:.2} GB", memory_usage as f64 / 1e9),
            suggested_fix: Some("Consider using quantization or DiskANN".to_string()),
        });
    }

    // Check dimension
    if dimension > 2048 {
        issues.push(DiagnosticIssue {
            severity: IssueSeverity::Info,
            category: IssueCategory::Performance,
            description: format!("High dimensionality: {}", dimension),
            suggested_fix: Some("Consider dimensionality reduction or PCA".to_string()),
        });
        recommendations
            .push("High-dimensional vectors may benefit from dimensionality reduction".to_string());
    }

    // Determine overall health status
    let status = if issues.iter().any(|i| i.severity == IssueSeverity::Critical) {
        HealthStatus::Critical
    } else if issues.iter().any(|i| i.severity == IssueSeverity::Error) {
        HealthStatus::Degraded
    } else if issues.iter().any(|i| i.severity == IssueSeverity::Warning) {
        HealthStatus::Warning
    } else {
        HealthStatus::Healthy
    };

    DiagnosticReport {
        status,
        size,
        memory_usage,
        issues,
        recommendations,
        performance: PerformanceMetrics {
            avg_query_latency: None,
            cache_hit_rate: None,
            estimated_qps: None,
        },
    }
}

/// Performance profiler for search operations
pub struct SearchProfiler {
    start_time: Instant,
    query_count: usize,
    total_duration: Duration,
    min_latency: Option<Duration>,
    max_latency: Option<Duration>,
}

impl SearchProfiler {
    /// Create a new search profiler
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            query_count: 0,
            total_duration: Duration::from_secs(0),
            min_latency: None,
            max_latency: None,
        }
    }

    /// Record a query execution
    pub fn record_query(&mut self, duration: Duration) {
        self.query_count += 1;
        self.total_duration += duration;

        self.min_latency = Some(match self.min_latency {
            Some(min) => min.min(duration),
            None => duration,
        });

        self.max_latency = Some(match self.max_latency {
            Some(max) => max.max(duration),
            None => duration,
        });
    }

    /// Get profiling statistics
    pub fn stats(&self) -> ProfilerStats {
        let avg_latency = if self.query_count > 0 {
            self.total_duration / self.query_count as u32
        } else {
            Duration::from_secs(0)
        };

        let elapsed = self.start_time.elapsed();
        let qps = if elapsed.as_secs() > 0 {
            self.query_count as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        ProfilerStats {
            total_queries: self.query_count,
            avg_latency,
            min_latency: self.min_latency,
            max_latency: self.max_latency,
            qps,
            elapsed,
        }
    }

    /// Reset the profiler
    pub fn reset(&mut self) {
        self.start_time = Instant::now();
        self.query_count = 0;
        self.total_duration = Duration::from_secs(0);
        self.min_latency = None;
        self.max_latency = None;
    }
}

impl Default for SearchProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics from search profiling
#[derive(Debug, Clone)]
pub struct ProfilerStats {
    /// Total number of queries executed
    pub total_queries: usize,
    /// Average query latency
    pub avg_latency: Duration,
    /// Minimum query latency
    pub min_latency: Option<Duration>,
    /// Maximum query latency
    pub max_latency: Option<Duration>,
    /// Queries per second
    pub qps: f64,
    /// Total elapsed time
    pub elapsed: Duration,
}

/// Index health monitor with periodic checks
pub struct HealthMonitor {
    /// Last diagnostic report
    last_report: Option<DiagnosticReport>,
    /// Last check time
    last_check: Option<Instant>,
    /// Check interval
    check_interval: Duration,
}

impl HealthMonitor {
    /// Create a new health monitor
    pub fn new(check_interval: Duration) -> Self {
        Self {
            last_report: None,
            last_check: None,
            check_interval,
        }
    }

    /// Check if a health check is due
    pub fn should_check(&self) -> bool {
        match self.last_check {
            Some(last) => last.elapsed() >= self.check_interval,
            None => true,
        }
    }

    /// Perform a health check
    pub fn check(&mut self, index: &VectorIndex) -> &DiagnosticReport {
        self.last_report = Some(diagnose_index(index));
        self.last_check = Some(Instant::now());
        self.last_report.as_ref().expect("just assigned above")
    }

    /// Get the last diagnostic report
    pub fn last_report(&self) -> Option<&DiagnosticReport> {
        self.last_report.as_ref()
    }

    /// Get time since last check
    pub fn time_since_last_check(&self) -> Option<Duration> {
        self.last_check.map(|t| t.elapsed())
    }
}

impl Default for HealthMonitor {
    fn default() -> Self {
        Self::new(Duration::from_secs(300)) // 5 minutes default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnose_empty_index() {
        let index =
            VectorIndex::with_defaults(128).expect("test: VectorIndex::with_defaults failed");
        let report = diagnose_index(&index);

        assert_eq!(report.size, 0);
        assert!(!report.issues.is_empty());
        assert!(report
            .issues
            .iter()
            .any(|i| i.category == IssueCategory::IndexStructure));
    }

    #[test]
    fn test_diagnose_normal_index() {
        let mut index =
            VectorIndex::with_defaults(128).expect("test: VectorIndex::with_defaults failed");
        let cid: ipfrs_core::Cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse()
            .expect("test: CID parse failed");
        index
            .insert(&cid, &vec![0.1; 128])
            .expect("test: index insert failed");

        let report = diagnose_index(&index);

        assert_eq!(report.size, 1);
        assert!(report.status == HealthStatus::Healthy || report.status == HealthStatus::Warning);
    }

    #[test]
    fn test_search_profiler() {
        let mut profiler = SearchProfiler::new();

        profiler.record_query(Duration::from_millis(10));
        profiler.record_query(Duration::from_millis(20));
        profiler.record_query(Duration::from_millis(15));

        let stats = profiler.stats();

        assert_eq!(stats.total_queries, 3);
        assert!(stats.avg_latency.as_millis() >= 10);
        assert!(stats.avg_latency.as_millis() <= 20);
        assert_eq!(stats.min_latency, Some(Duration::from_millis(10)));
        assert_eq!(stats.max_latency, Some(Duration::from_millis(20)));
    }

    #[test]
    fn test_health_monitor() {
        let mut monitor = HealthMonitor::new(Duration::from_millis(100));
        let index = VectorIndex::with_defaults(128)
            .expect("test: VectorIndex creation with dim 128 should succeed");

        assert!(monitor.should_check());

        monitor.check(&index);

        assert!(!monitor.should_check());
        assert!(monitor.last_report().is_some());

        std::thread::sleep(Duration::from_millis(150));
        assert!(monitor.should_check());
    }

    #[test]
    fn test_profiler_reset() {
        let mut profiler = SearchProfiler::new();

        profiler.record_query(Duration::from_millis(10));
        profiler.record_query(Duration::from_millis(20));

        assert_eq!(profiler.stats().total_queries, 2);

        profiler.reset();

        assert_eq!(profiler.stats().total_queries, 0);
        assert_eq!(profiler.stats().min_latency, None);
        assert_eq!(profiler.stats().max_latency, None);
    }
}
