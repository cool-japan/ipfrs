//! Comprehensive storage profiling and optimization
//!
//! This module provides a unified interface for profiling storage performance,
//! analyzing workload characteristics, and generating optimization recommendations.

use crate::analyzer::{StorageAnalysis, StorageAnalyzer, WorkloadType};
use crate::auto_tuner::{AutoTuner, AutoTunerConfig, TuningReport};
use crate::diagnostics::{DiagnosticsReport, StorageDiagnostics};
use crate::traits::BlockStore;
use crate::workload::{WorkloadConfig, WorkloadResult, WorkloadSimulator};
use ipfrs_core::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Comprehensive profiling report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileReport {
    /// Storage backend name
    pub backend: String,
    /// Diagnostics results
    pub diagnostics: DiagnosticsReport,
    /// Workload simulation results
    pub workload_results: Vec<WorkloadResult>,
    /// Storage analysis
    pub analysis: StorageAnalysis,
    /// Auto-tuning recommendations
    pub tuning_report: TuningReport,
    /// Overall performance score (0-100)
    pub performance_score: u8,
    /// Profiling duration
    pub duration: Duration,
}

/// Profiling configuration
#[derive(Debug, Clone)]
pub struct ProfileConfig {
    /// Run diagnostics tests
    pub run_diagnostics: bool,
    /// Workload configurations to test
    pub workload_configs: Vec<WorkloadConfig>,
    /// Auto-tuner configuration
    pub tuner_config: AutoTunerConfig,
    /// Include detailed analysis
    pub detailed_analysis: bool,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            run_diagnostics: true,
            workload_configs: vec![
                crate::workload::WorkloadPresets::light_test(),
                crate::workload::WorkloadPresets::medium_stress(),
            ],
            tuner_config: AutoTunerConfig::default(),
            detailed_analysis: true,
        }
    }
}

impl ProfileConfig {
    /// Create a quick profiling configuration (fast, minimal tests)
    pub fn quick() -> Self {
        Self {
            run_diagnostics: true,
            workload_configs: vec![crate::workload::WorkloadPresets::light_test()],
            tuner_config: AutoTunerConfig::default(),
            detailed_analysis: false,
        }
    }

    /// Create a comprehensive profiling configuration (thorough, all tests)
    pub fn comprehensive() -> Self {
        Self {
            run_diagnostics: true,
            workload_configs: vec![
                crate::workload::WorkloadPresets::light_test(),
                crate::workload::WorkloadPresets::medium_stress(),
                crate::workload::WorkloadPresets::cdn_cache(),
                crate::workload::WorkloadPresets::ingestion_pipeline(),
                crate::workload::WorkloadPresets::time_series(),
            ],
            tuner_config: AutoTunerConfig::default(),
            detailed_analysis: true,
        }
    }

    /// Create a performance-focused profiling configuration
    pub fn performance() -> Self {
        Self {
            run_diagnostics: true,
            workload_configs: vec![
                crate::workload::WorkloadPresets::medium_stress(),
                crate::workload::WorkloadPresets::heavy_stress(),
            ],
            tuner_config: crate::auto_tuner::TuningPresets::performance(),
            detailed_analysis: true,
        }
    }
}

/// Storage profiler for comprehensive performance analysis
pub struct StorageProfiler<S: BlockStore> {
    store: Arc<S>,
    backend_name: String,
    config: ProfileConfig,
}

impl<S: BlockStore + Send + Sync + 'static> StorageProfiler<S> {
    /// Create a new storage profiler
    pub fn new(store: Arc<S>, backend_name: String, config: ProfileConfig) -> Self {
        Self {
            store,
            backend_name,
            config,
        }
    }

    /// Create a profiler with default configuration
    pub fn with_defaults(store: Arc<S>, backend_name: String) -> Self {
        Self::new(store, backend_name, ProfileConfig::default())
    }

    /// Run comprehensive profiling
    pub async fn profile(&self) -> Result<ProfileReport> {
        let start = Instant::now();

        // Step 1: Run diagnostics
        let diagnostics = if self.config.run_diagnostics {
            let mut diagnostics_runner =
                StorageDiagnostics::new(Arc::clone(&self.store), self.backend_name.clone());
            diagnostics_runner.run().await?
        } else {
            // Create minimal diagnostics report
            DiagnosticsReport {
                backend: self.backend_name.clone(),
                total_blocks: 0,
                performance: crate::diagnostics::PerformanceMetrics {
                    avg_write_latency: Duration::from_micros(0),
                    avg_read_latency: Duration::from_micros(0),
                    avg_batch_write_latency: Duration::from_micros(0),
                    avg_batch_read_latency: Duration::from_micros(0),
                    write_throughput: 0.0,
                    read_throughput: 0.0,
                    peak_memory_usage: 0,
                },
                health: crate::diagnostics::HealthMetrics {
                    successful_ops: 0,
                    failed_ops: 0,
                    success_rate: 1.0,
                    integrity_ok: true,
                    responsive: true,
                },
                recommendations: Vec::new(),
                health_score: 100,
            }
        };

        // Step 2: Run workload simulations
        let mut workload_results = Vec::new();
        for workload_config in &self.config.workload_configs {
            let mut simulator = WorkloadSimulator::new(workload_config.clone());
            simulator.generate_dataset();
            let result = simulator.run(self.store.clone()).await?;
            workload_results.push(result);
        }

        // Step 3: Analyze storage characteristics
        let mut analyzer = StorageAnalyzer::new(Arc::clone(&self.store), self.backend_name.clone());
        let analysis = if self.config.detailed_analysis {
            analyzer.analyze().await?
        } else {
            // Create basic analysis from workload results
            self.create_basic_analysis(&diagnostics, &workload_results)
        };

        // Step 4: Generate tuning recommendations
        let tuner = AutoTuner::new(self.config.tuner_config.clone());
        let tuning_report = tuner.analyze_and_tune(&*self.store, &analysis).await?;

        // Step 5: Calculate performance score
        let performance_score = self.calculate_performance_score(&analysis, &tuning_report);

        let duration = start.elapsed();

        Ok(ProfileReport {
            backend: self.backend_name.clone(),
            diagnostics,
            workload_results,
            analysis,
            tuning_report,
            performance_score,
            duration,
        })
    }

    /// Create basic analysis from workload results
    fn create_basic_analysis(
        &self,
        diagnostics: &DiagnosticsReport,
        workload_results: &[WorkloadResult],
    ) -> StorageAnalysis {
        // Determine workload type based on operation counts
        let mut total_gets = 0usize;
        let mut total_puts = 0usize;
        for result in workload_results {
            total_gets += result.operation_counts.get("get").copied().unwrap_or(0);
            total_puts += result.operation_counts.get("put").copied().unwrap_or(0);
        }

        let read_write_ratio = if total_puts > 0 {
            total_gets as f64 / (total_gets + total_puts) as f64
        } else {
            1.0
        };

        let workload_type = if read_write_ratio > 0.7 {
            WorkloadType::ReadHeavy
        } else if read_write_ratio < 0.3 {
            WorkloadType::WriteHeavy
        } else {
            WorkloadType::Balanced
        };

        StorageAnalysis {
            backend: self.backend_name.clone(),
            diagnostics: diagnostics.clone(),
            performance_breakdown: HashMap::new(),
            workload: crate::analyzer::WorkloadCharacterization {
                read_write_ratio,
                avg_block_size: 16384, // Default assumption
                size_distribution: crate::analyzer::SizeDistribution {
                    small_pct: 0.3,
                    medium_pct: 0.5,
                    large_pct: 0.2,
                },
                workload_type,
            },
            recommendations: Vec::new(),
            grade: self.calculate_grade(diagnostics.health_score),
        }
    }

    /// Calculate performance score
    fn calculate_performance_score(&self, analysis: &StorageAnalysis, tuning: &TuningReport) -> u8 {
        let mut score = 100u8;

        // Penalize based on diagnostics health score
        let health_penalty = (100 - analysis.diagnostics.health_score) / 2;
        score = score.saturating_sub(health_penalty);

        // Penalize based on number of high-priority recommendations
        let high_priority_recs = tuning
            .recommendations
            .iter()
            .filter(|r| r.expected_impact > 20.0)
            .count();
        score = score.saturating_sub((high_priority_recs * 5) as u8);

        // Bonus for good tuning score
        score = score.saturating_add(tuning.score / 10);

        score.min(100)
    }

    /// Calculate grade from score
    fn calculate_grade(&self, score: u8) -> String {
        match score {
            90..=100 => "A",
            80..=89 => "B",
            70..=79 => "C",
            60..=69 => "D",
            _ => "F",
        }
        .to_string()
    }
}

/// Comparative profiling for multiple storage configurations
pub struct ComparativeProfiler;

impl ComparativeProfiler {
    /// Compare multiple storage backends
    pub async fn compare<S1, S2>(
        store1: Arc<S1>,
        name1: &str,
        store2: Arc<S2>,
        name2: &str,
        config: ProfileConfig,
    ) -> Result<ComparisonReport>
    where
        S1: BlockStore + Send + Sync + 'static,
        S2: BlockStore + Send + Sync + 'static,
    {
        let profiler1 = StorageProfiler::new(store1, name1.to_string(), config.clone());
        let profiler2 = StorageProfiler::new(store2, name2.to_string(), config);

        let report1 = profiler1.profile().await?;
        let report2 = profiler2.profile().await?;

        Ok(ComparisonReport {
            profiles: vec![report1, report2],
            winner: Self::determine_winner(name1, name2, &[]),
        })
    }

    /// Determine the better configuration
    fn determine_winner(name1: &str, name2: &str, profiles: &[ProfileReport]) -> String {
        if profiles.len() < 2 {
            return "Insufficient data".to_string();
        }

        if profiles[0].performance_score > profiles[1].performance_score {
            name1.to_string()
        } else if profiles[1].performance_score > profiles[0].performance_score {
            name2.to_string()
        } else {
            "Tie".to_string()
        }
    }
}

/// Comparison report for multiple configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    /// Individual profile reports
    pub profiles: Vec<ProfileReport>,
    /// Winner determination
    pub winner: String,
}

/// Performance regression detector
pub struct RegressionDetector {
    baseline: ProfileReport,
    threshold: f64,
}

impl RegressionDetector {
    /// Create a new regression detector with baseline
    pub fn new(baseline: ProfileReport, threshold: f64) -> Self {
        Self {
            baseline,
            threshold,
        }
    }

    /// Check for performance regression
    pub fn check_regression(&self, current: &ProfileReport) -> RegressionResult {
        let mut regressions = Vec::new();

        // Check performance score regression
        if current.performance_score < self.baseline.performance_score {
            let diff = self.baseline.performance_score - current.performance_score;
            if (diff as f64) > self.threshold {
                regressions.push(format!("Performance score decreased by {diff} points"));
            }
        }

        // Check workload throughput regression
        for (i, baseline_result) in self.baseline.workload_results.iter().enumerate() {
            if let Some(current_result) = current.workload_results.get(i) {
                let throughput_ratio =
                    current_result.ops_per_second / baseline_result.ops_per_second;
                if throughput_ratio < (1.0 - self.threshold) {
                    regressions.push(format!(
                        "Workload {} throughput decreased by {:.1}%",
                        i,
                        (1.0 - throughput_ratio) * 100.0
                    ));
                }
            }
        }

        RegressionResult {
            has_regression: !regressions.is_empty(),
            regressions,
            improvement_pct: self.calculate_improvement(current),
        }
    }

    /// Calculate overall improvement percentage
    fn calculate_improvement(&self, current: &ProfileReport) -> f64 {
        let score_improvement = (current.performance_score as f64
            - self.baseline.performance_score as f64)
            / self.baseline.performance_score as f64;
        score_improvement * 100.0
    }
}

/// Regression detection result
#[derive(Debug, Clone)]
pub struct RegressionResult {
    /// Whether regression was detected
    pub has_regression: bool,
    /// List of detected regressions
    pub regressions: Vec<String>,
    /// Overall improvement percentage (negative if regression)
    pub improvement_pct: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryBlockStore;

    #[tokio::test]
    async fn test_quick_profile() {
        let config = ProfileConfig::quick();
        let profiler = StorageProfiler::new(
            Arc::new(MemoryBlockStore::new()),
            "MemoryBlockStore".to_string(),
            config,
        );

        let report = profiler.profile().await.unwrap();

        assert_eq!(report.backend, "MemoryBlockStore");
        assert!(!report.workload_results.is_empty());
        assert!(report.performance_score <= 100);
    }

    #[tokio::test]
    async fn test_performance_score_calculation() {
        let store = Arc::new(MemoryBlockStore::new());
        let config = ProfileConfig::quick();
        let profiler = StorageProfiler::new(store, "Test".to_string(), config);

        let report = profiler.profile().await.unwrap();
        assert!(report.performance_score > 0);
        assert!(report.performance_score <= 100);
    }

    #[test]
    fn test_profile_config_presets() {
        let _quick = ProfileConfig::quick();
        let _comprehensive = ProfileConfig::comprehensive();
        let _performance = ProfileConfig::performance();
    }
}
