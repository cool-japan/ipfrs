//! Automatic configuration tuning based on workload patterns
//!
//! This module provides automatic tuning of storage configuration parameters
//! based on observed workload characteristics and performance metrics.

use crate::analyzer::{StorageAnalysis, WorkloadType};
use crate::traits::BlockStore;
use ipfrs_core::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Tuning recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningRecommendation {
    /// Parameter name
    pub parameter: String,
    /// Current value
    pub current_value: String,
    /// Recommended value
    pub recommended_value: String,
    /// Rationale for the recommendation
    pub rationale: String,
    /// Expected impact (percentage improvement)
    pub expected_impact: f64,
    /// Confidence level (0.0 - 1.0)
    pub confidence: f64,
}

/// Auto-tuning configuration
#[derive(Debug, Clone)]
pub struct AutoTunerConfig {
    /// Minimum observation period before making recommendations
    pub observation_period: Duration,
    /// Minimum confidence threshold for recommendations (0.0 - 1.0)
    pub confidence_threshold: f64,
    /// Target cache hit rate (0.0 - 1.0)
    pub target_cache_hit_rate: f64,
    /// Target bloom filter false positive rate (0.0 - 1.0)
    pub target_bloom_fp_rate: f64,
    /// Enable aggressive tuning (may suggest larger changes)
    pub aggressive: bool,
}

impl Default for AutoTunerConfig {
    fn default() -> Self {
        Self {
            observation_period: Duration::from_secs(300), // 5 minutes
            confidence_threshold: 0.7,
            target_cache_hit_rate: 0.85,
            target_bloom_fp_rate: 0.01,
            aggressive: false,
        }
    }
}

/// Tuning report with recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningReport {
    /// Workload analysis
    pub analysis: Option<String>,
    /// List of recommendations
    pub recommendations: Vec<TuningRecommendation>,
    /// Overall tuning score (0-100)
    pub score: u8,
    /// Summary of findings
    pub summary: String,
}

/// Automatic configuration tuner
pub struct AutoTuner {
    config: AutoTunerConfig,
}

impl AutoTuner {
    /// Create a new auto-tuner with the given configuration
    pub fn new(config: AutoTunerConfig) -> Self {
        Self { config }
    }

    /// Create an auto-tuner with default configuration
    pub fn default_config() -> Self {
        Self::new(AutoTunerConfig::default())
    }

    /// Analyze storage and generate tuning recommendations
    #[allow(clippy::unused_async)]
    pub async fn analyze_and_tune<S: BlockStore>(
        &self,
        _store: &S,
        analysis: &StorageAnalysis,
    ) -> Result<TuningReport> {
        let mut recommendations = Vec::new();
        let mut score = 100u8;

        // Analyze cache performance
        if let Some(cache_rec) = self.tune_cache(analysis) {
            if cache_rec.confidence >= self.config.confidence_threshold {
                score = score.saturating_sub(5);
                recommendations.push(cache_rec);
            }
        }

        // Analyze bloom filter
        if let Some(bloom_rec) = self.tune_bloom_filter(analysis) {
            if bloom_rec.confidence >= self.config.confidence_threshold {
                score = score.saturating_sub(5);
                recommendations.push(bloom_rec);
            }
        }

        // Analyze concurrency settings
        if let Some(concurrency_rec) = self.tune_concurrency(analysis) {
            if concurrency_rec.confidence >= self.config.confidence_threshold {
                score = score.saturating_sub(5);
                recommendations.push(concurrency_rec);
            }
        }

        // Analyze compression settings
        if let Some(compression_rec) = self.tune_compression(analysis) {
            if compression_rec.confidence >= self.config.confidence_threshold {
                score = score.saturating_sub(5);
                recommendations.push(compression_rec);
            }
        }

        // Analyze deduplication settings
        if let Some(dedup_rec) = self.tune_deduplication(analysis) {
            if dedup_rec.confidence >= self.config.confidence_threshold {
                score = score.saturating_sub(5);
                recommendations.push(dedup_rec);
            }
        }

        // Analyze backend selection
        if let Some(backend_rec) = self.tune_backend_selection(analysis) {
            if backend_rec.confidence >= self.config.confidence_threshold {
                score = score.saturating_sub(10);
                recommendations.push(backend_rec);
            }
        }

        let summary = self.generate_summary(&recommendations, &analysis.workload.workload_type);

        Ok(TuningReport {
            analysis: Some(format!("Workload: {:?}", analysis.workload.workload_type)),
            recommendations,
            score,
            summary,
        })
    }

    /// Tune cache size based on workload
    fn tune_cache(&self, analysis: &StorageAnalysis) -> Option<TuningRecommendation> {
        // Use success rate as a proxy for cache effectiveness
        // In a real implementation, this would come from cache statistics
        let cache_hit_rate = analysis.diagnostics.health.success_rate * 0.7; // Approximation

        if cache_hit_rate < self.config.target_cache_hit_rate {
            let current_size = "current"; // Would be extracted from actual config
            let increase_factor = if self.config.aggressive { 2.0 } else { 1.5 };
            let recommended_size = format!("{increase_factor}x current");

            let confidence = if analysis.workload.read_write_ratio > 0.7 {
                0.9 // High confidence for read-heavy workloads
            } else {
                0.7
            };

            Some(TuningRecommendation {
                parameter: "cache_size".to_string(),
                current_value: current_size.to_string(),
                recommended_value: recommended_size,
                rationale: format!(
                    "Cache hit rate ({:.1}%) is below target ({:.1}%). \
                     Increasing cache size will improve read performance.",
                    cache_hit_rate * 100.0,
                    self.config.target_cache_hit_rate * 100.0
                ),
                expected_impact: (self.config.target_cache_hit_rate - cache_hit_rate) * 50.0,
                confidence,
            })
        } else {
            None
        }
    }

    /// Tune bloom filter parameters
    fn tune_bloom_filter(&self, analysis: &StorageAnalysis) -> Option<TuningRecommendation> {
        // For read-heavy workloads, a larger bloom filter helps
        if analysis.workload.read_write_ratio > 0.7 {
            Some(TuningRecommendation {
                parameter: "bloom_filter_size".to_string(),
                current_value: "current".to_string(),
                recommended_value: "2x expected items".to_string(),
                rationale: "Read-heavy workload benefits from larger bloom filter \
                           to reduce false positives and unnecessary disk lookups."
                    .to_string(),
                expected_impact: 5.0,
                confidence: 0.8,
            })
        } else {
            None
        }
    }

    /// Tune concurrency settings
    fn tune_concurrency(&self, analysis: &StorageAnalysis) -> Option<TuningRecommendation> {
        // Check if there are latency issues
        let avg_latency = analysis
            .performance_breakdown
            .values()
            .map(|stats| stats.avg_latency_us)
            .max()
            .unwrap_or(0);

        if avg_latency > 10_000 {
            // > 10ms
            Some(TuningRecommendation {
                parameter: "concurrency_limit".to_string(),
                current_value: "unlimited".to_string(),
                recommended_value: "8-16 concurrent operations".to_string(),
                rationale: "High latency detected. Limiting concurrency \
                           can reduce contention and improve throughput."
                    .to_string(),
                expected_impact: 15.0,
                confidence: 0.75,
            })
        } else {
            None
        }
    }

    /// Tune compression settings
    fn tune_compression(&self, analysis: &StorageAnalysis) -> Option<TuningRecommendation> {
        let avg_block_size = analysis.workload.avg_block_size;

        // Recommend compression for larger blocks
        if avg_block_size > 16384 {
            // > 16KB
            Some(TuningRecommendation {
                parameter: "compression".to_string(),
                current_value: "disabled".to_string(),
                recommended_value: "Zstd level 3".to_string(),
                rationale: format!(
                    "Average block size ({avg_block_size} bytes) is large enough to benefit \
                     from compression. Zstd level 3 provides good balance."
                ),
                expected_impact: 30.0, // 30% storage reduction
                confidence: 0.85,
            })
        } else {
            None
        }
    }

    /// Tune deduplication settings
    fn tune_deduplication(&self, analysis: &StorageAnalysis) -> Option<TuningRecommendation> {
        // For write-heavy workloads with redundancy, recommend dedup
        if matches!(analysis.workload.workload_type, WorkloadType::WriteHeavy) {
            Some(TuningRecommendation {
                parameter: "deduplication".to_string(),
                current_value: "disabled".to_string(),
                recommended_value: "enabled with 16KB chunks".to_string(),
                rationale: "Write-heavy workload likely has redundant data. \
                           Deduplication can significantly reduce storage."
                    .to_string(),
                expected_impact: 25.0, // 25% storage reduction
                confidence: 0.7,
            })
        } else {
            None
        }
    }

    /// Tune backend selection
    fn tune_backend_selection(&self, analysis: &StorageAnalysis) -> Option<TuningRecommendation> {
        match analysis.workload.workload_type {
            WorkloadType::WriteHeavy if analysis.backend == "Sled" => {
                Some(TuningRecommendation {
                    parameter: "backend".to_string(),
                    current_value: "Sled".to_string(),
                    recommended_value: "ParityDB".to_string(),
                    rationale: "Write-heavy workload. ParityDB offers 2-5x better \
                               write performance with lower write amplification."
                        .to_string(),
                    expected_impact: 100.0, // 2x throughput
                    confidence: 0.9,
                })
            }
            WorkloadType::ReadHeavy if analysis.backend == "ParityDB" => {
                Some(TuningRecommendation {
                    parameter: "backend".to_string(),
                    current_value: "ParityDB".to_string(),
                    recommended_value: "Sled".to_string(),
                    rationale: "Read-heavy workload. Sled offers better read \
                               performance with B-tree indexing."
                        .to_string(),
                    expected_impact: 50.0, // 1.5x read throughput
                    confidence: 0.85,
                })
            }
            _ => None,
        }
    }

    /// Generate summary text
    fn generate_summary(
        &self,
        recommendations: &[TuningRecommendation],
        workload: &WorkloadType,
    ) -> String {
        if recommendations.is_empty() {
            return "Configuration is well-tuned for current workload. No changes recommended."
                .to_string();
        }

        let high_impact: Vec<_> = recommendations
            .iter()
            .filter(|r| r.expected_impact > 20.0)
            .collect();

        if high_impact.is_empty() {
            format!(
                "Found {} minor optimization opportunities for {:?} workload.",
                recommendations.len(),
                workload
            )
        } else {
            format!(
                "Found {} optimization opportunities for {:?} workload, \
                 including {} high-impact changes. Implementing all recommendations \
                 could improve performance by up to {:.0}%.",
                recommendations.len(),
                workload,
                high_impact.len(),
                recommendations
                    .iter()
                    .map(|r| r.expected_impact)
                    .sum::<f64>()
            )
        }
    }

    /// Apply automatic tuning (returns recommended configuration as key-value pairs)
    pub fn apply_recommendations(&self, report: &TuningReport) -> HashMap<String, String> {
        let mut config = HashMap::new();

        for rec in &report.recommendations {
            if rec.confidence >= self.config.confidence_threshold {
                config.insert(rec.parameter.clone(), rec.recommended_value.clone());
            }
        }

        config
    }

    /// Quick tune based on workload type (doesn't require analysis)
    pub fn quick_tune(&self, workload_type: WorkloadType) -> HashMap<String, String> {
        let mut config = HashMap::new();

        match workload_type {
            WorkloadType::ReadHeavy => {
                config.insert("cache_size".to_string(), "1GB".to_string());
                config.insert("bloom_filter".to_string(), "large".to_string());
                config.insert("backend".to_string(), "Sled".to_string());
                config.insert("compression".to_string(), "disabled".to_string());
            }
            WorkloadType::WriteHeavy => {
                config.insert("cache_size".to_string(), "256MB".to_string());
                config.insert("backend".to_string(), "ParityDB".to_string());
                config.insert("deduplication".to_string(), "enabled".to_string());
                config.insert("batch_size".to_string(), "100".to_string());
            }
            WorkloadType::Balanced => {
                config.insert("cache_size".to_string(), "512MB".to_string());
                config.insert("bloom_filter".to_string(), "medium".to_string());
                config.insert("backend".to_string(), "ParityDB".to_string());
            }
            WorkloadType::BatchOriented => {
                config.insert("batch_size".to_string(), "1000".to_string());
                config.insert("concurrency".to_string(), "16".to_string());
                config.insert("backend".to_string(), "ParityDB".to_string());
            }
            WorkloadType::Mixed => {
                config.insert("cache_size".to_string(), "512MB".to_string());
                config.insert("backend".to_string(), "ParityDB".to_string());
            }
        }

        config
    }
}

/// Tuning presets for common scenarios
pub struct TuningPresets;

impl TuningPresets {
    /// Conservative tuning (minimal changes, high confidence)
    pub fn conservative() -> AutoTunerConfig {
        AutoTunerConfig {
            observation_period: Duration::from_secs(600), // 10 minutes
            confidence_threshold: 0.85,
            target_cache_hit_rate: 0.80,
            target_bloom_fp_rate: 0.01,
            aggressive: false,
        }
    }

    /// Balanced tuning (moderate changes, good confidence)
    pub fn balanced() -> AutoTunerConfig {
        AutoTunerConfig::default()
    }

    /// Aggressive tuning (larger changes, lower confidence threshold)
    pub fn aggressive() -> AutoTunerConfig {
        AutoTunerConfig {
            observation_period: Duration::from_secs(300), // 5 minutes
            confidence_threshold: 0.6,
            target_cache_hit_rate: 0.90,
            target_bloom_fp_rate: 0.005,
            aggressive: true,
        }
    }

    /// Performance-focused tuning (maximize throughput)
    pub fn performance() -> AutoTunerConfig {
        AutoTunerConfig {
            observation_period: Duration::from_secs(300),
            confidence_threshold: 0.7,
            target_cache_hit_rate: 0.95,
            target_bloom_fp_rate: 0.001,
            aggressive: true,
        }
    }

    /// Cost-focused tuning (minimize resource usage)
    pub fn cost_optimized() -> AutoTunerConfig {
        AutoTunerConfig {
            observation_period: Duration::from_secs(900), // 15 minutes
            confidence_threshold: 0.9,
            target_cache_hit_rate: 0.75,
            target_bloom_fp_rate: 0.02,
            aggressive: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::{SizeDistribution, StorageAnalysis, WorkloadCharacterization};
    use crate::diagnostics::{DiagnosticsReport, HealthMetrics, PerformanceMetrics};

    fn create_test_analysis(workload_type: WorkloadType) -> StorageAnalysis {
        let diagnostics = DiagnosticsReport {
            backend: "Sled".to_string(),
            total_blocks: 10_000,
            performance: PerformanceMetrics {
                avg_write_latency: Duration::from_micros(1000),
                avg_read_latency: Duration::from_micros(500),
                avg_batch_write_latency: Duration::from_millis(50),
                avg_batch_read_latency: Duration::from_millis(20),
                write_throughput: 500.0,
                read_throughput: 1000.0,
                peak_memory_usage: 100_000_000,
            },
            health: HealthMetrics {
                successful_ops: 9000,
                failed_ops: 0,
                success_rate: 1.0,
                integrity_ok: true,
                responsive: true,
            },
            recommendations: Vec::new(),
            health_score: 85,
        };

        StorageAnalysis {
            backend: "Sled".to_string(),
            diagnostics,
            performance_breakdown: HashMap::new(),
            workload: WorkloadCharacterization {
                read_write_ratio: 0.7,
                avg_block_size: 32768,
                size_distribution: SizeDistribution {
                    small_pct: 0.3,
                    medium_pct: 0.5,
                    large_pct: 0.2,
                },
                workload_type,
            },
            recommendations: Vec::new(),
            grade: "B".to_string(),
        }
    }

    #[tokio::test]
    async fn test_auto_tuner_cache_recommendation() {
        let tuner = AutoTuner::default_config();
        let analysis = create_test_analysis(WorkloadType::ReadHeavy);

        // Should recommend cache increase for low hit rate
        let cache_rec = tuner.tune_cache(&analysis);
        assert!(cache_rec.is_some());

        let rec = cache_rec.unwrap();
        assert_eq!(rec.parameter, "cache_size");
        assert!(rec.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_auto_tuner_backend_recommendation() {
        let tuner = AutoTuner::default_config();
        let mut analysis = create_test_analysis(WorkloadType::WriteHeavy);
        analysis.backend = "Sled".to_string();

        let backend_rec = tuner.tune_backend_selection(&analysis);
        assert!(backend_rec.is_some());

        let rec = backend_rec.unwrap();
        assert_eq!(rec.parameter, "backend");
        assert_eq!(rec.recommended_value, "ParityDB");
    }

    #[tokio::test]
    async fn test_quick_tune() {
        let tuner = AutoTuner::default_config();

        let read_config = tuner.quick_tune(WorkloadType::ReadHeavy);
        assert_eq!(read_config.get("backend"), Some(&"Sled".to_string()));

        let write_config = tuner.quick_tune(WorkloadType::WriteHeavy);
        assert_eq!(write_config.get("backend"), Some(&"ParityDB".to_string()));
    }

    #[test]
    fn test_tuning_presets() {
        let _conservative = TuningPresets::conservative();
        let _balanced = TuningPresets::balanced();
        let _aggressive = TuningPresets::aggressive();
        let _performance = TuningPresets::performance();
        let _cost = TuningPresets::cost_optimized();
    }
}
