//! Metric exporters for integration with monitoring systems
//!
//! Provides exporters for various formats: JSON, CSV, Prometheus, and custom formats
//! for easy integration with monitoring and observability platforms.

use crate::analyzer::StorageAnalysis;
use crate::diagnostics::DiagnosticsReport;
use crate::metrics::StorageMetrics;
use serde_json;
use std::fmt::Write as FmtWrite;

/// Export format
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExportFormat {
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// Prometheus metrics format
    Prometheus,
    /// Human-readable text format
    Text,
}

/// Metric exporter
pub struct MetricExporter;

impl MetricExporter {
    /// Export storage metrics to specified format
    pub fn export_metrics(metrics: &StorageMetrics, format: ExportFormat) -> String {
        match format {
            ExportFormat::Json => Self::export_metrics_json(metrics),
            ExportFormat::Csv => Self::export_metrics_csv(metrics),
            ExportFormat::Prometheus => Self::export_metrics_prometheus(metrics),
            ExportFormat::Text => Self::export_metrics_text(metrics),
        }
    }

    /// Export diagnostics report
    pub fn export_diagnostics(report: &DiagnosticsReport, format: ExportFormat) -> String {
        match format {
            ExportFormat::Json => serde_json::to_string_pretty(report).unwrap_or_default(),
            ExportFormat::Csv => Self::export_diagnostics_csv(report),
            ExportFormat::Prometheus => Self::export_diagnostics_prometheus(report),
            ExportFormat::Text => Self::export_diagnostics_text(report),
        }
    }

    /// Export storage analysis
    pub fn export_analysis(analysis: &StorageAnalysis, format: ExportFormat) -> String {
        match format {
            ExportFormat::Json => serde_json::to_string_pretty(analysis).unwrap_or_default(),
            ExportFormat::Csv => Self::export_analysis_csv(analysis),
            ExportFormat::Prometheus => Self::export_analysis_prometheus(analysis),
            ExportFormat::Text => Self::export_analysis_text(analysis),
        }
    }

    // JSON exports
    fn export_metrics_json(metrics: &StorageMetrics) -> String {
        serde_json::to_string_pretty(metrics).unwrap_or_default()
    }

    // CSV exports
    fn export_metrics_csv(metrics: &StorageMetrics) -> String {
        let mut csv = String::new();
        csv.push_str("metric,value\n");
        csv.push_str(&format!("put_count,{}\n", metrics.put_count));
        csv.push_str(&format!("get_count,{}\n", metrics.get_count));
        csv.push_str(&format!("has_count,{}\n", metrics.has_count));
        csv.push_str(&format!("delete_count,{}\n", metrics.delete_count));
        csv.push_str(&format!("get_hits,{}\n", metrics.get_hits));
        csv.push_str(&format!("get_misses,{}\n", metrics.get_misses));
        csv.push_str(&format!("bytes_written,{}\n", metrics.bytes_written));
        csv.push_str(&format!("bytes_read,{}\n", metrics.bytes_read));
        csv.push_str(&format!(
            "avg_put_latency_us,{}\n",
            metrics.avg_put_latency_us
        ));
        csv.push_str(&format!(
            "avg_get_latency_us,{}\n",
            metrics.avg_get_latency_us
        ));
        csv.push_str(&format!("cache_hit_rate,{:.4}\n", metrics.cache_hit_rate()));
        csv
    }

    fn export_diagnostics_csv(report: &DiagnosticsReport) -> String {
        let mut csv = String::new();
        csv.push_str("metric,value\n");
        csv.push_str(&format!("backend,{}\n", report.backend));
        csv.push_str(&format!("total_blocks,{}\n", report.total_blocks));
        csv.push_str(&format!("health_score,{}\n", report.health_score));
        csv.push_str(&format!(
            "write_throughput,{:.2}\n",
            report.performance.write_throughput
        ));
        csv.push_str(&format!(
            "read_throughput,{:.2}\n",
            report.performance.read_throughput
        ));
        csv.push_str(&format!(
            "peak_memory_usage,{}\n",
            report.performance.peak_memory_usage
        ));
        csv.push_str(&format!(
            "successful_ops,{}\n",
            report.health.successful_ops
        ));
        csv.push_str(&format!("failed_ops,{}\n", report.health.failed_ops));
        csv.push_str(&format!("success_rate,{:.4}\n", report.health.success_rate));
        csv
    }

    fn export_analysis_csv(analysis: &StorageAnalysis) -> String {
        let mut csv = String::new();
        csv.push_str("metric,value\n");
        csv.push_str(&format!("backend,{}\n", analysis.backend));
        csv.push_str(&format!("grade,{}\n", analysis.grade));
        csv.push_str(&format!(
            "health_score,{}\n",
            analysis.diagnostics.health_score
        ));
        csv.push_str(&format!(
            "read_write_ratio,{:.4}\n",
            analysis.workload.read_write_ratio
        ));
        csv.push_str(&format!(
            "avg_block_size,{}\n",
            analysis.workload.avg_block_size
        ));
        csv.push_str(&format!(
            "workload_type,{:?}\n",
            analysis.workload.workload_type
        ));
        csv.push_str(&format!(
            "recommendation_count,{}\n",
            analysis.recommendations.len()
        ));
        csv
    }

    // Prometheus format exports
    fn export_metrics_prometheus(metrics: &StorageMetrics) -> String {
        let mut prom = String::new();

        // Counters
        let _ = writeln!(
            prom,
            "# HELP ipfrs_storage_put_total Total number of put operations"
        );
        let _ = writeln!(prom, "# TYPE ipfrs_storage_put_total counter");
        let _ = writeln!(prom, "ipfrs_storage_put_total {}", metrics.put_count);

        let _ = writeln!(
            prom,
            "# HELP ipfrs_storage_get_total Total number of get operations"
        );
        let _ = writeln!(prom, "# TYPE ipfrs_storage_get_total counter");
        let _ = writeln!(prom, "ipfrs_storage_get_total {}", metrics.get_count);

        let _ = writeln!(
            prom,
            "# HELP ipfrs_storage_bytes_written_total Total bytes written"
        );
        let _ = writeln!(prom, "# TYPE ipfrs_storage_bytes_written_total counter");
        let _ = writeln!(
            prom,
            "ipfrs_storage_bytes_written_total {}",
            metrics.bytes_written
        );

        let _ = writeln!(
            prom,
            "# HELP ipfrs_storage_bytes_read_total Total bytes read"
        );
        let _ = writeln!(prom, "# TYPE ipfrs_storage_bytes_read_total counter");
        let _ = writeln!(
            prom,
            "ipfrs_storage_bytes_read_total {}",
            metrics.bytes_read
        );

        // Gauges
        let _ = writeln!(
            prom,
            "# HELP ipfrs_storage_avg_put_latency_us Average put latency in microseconds"
        );
        let _ = writeln!(prom, "# TYPE ipfrs_storage_avg_put_latency_us gauge");
        let _ = writeln!(
            prom,
            "ipfrs_storage_avg_put_latency_us {}",
            metrics.avg_put_latency_us
        );

        let _ = writeln!(
            prom,
            "# HELP ipfrs_storage_cache_hit_rate Cache hit rate (0-1)"
        );
        let _ = writeln!(prom, "# TYPE ipfrs_storage_cache_hit_rate gauge");
        let _ = writeln!(
            prom,
            "ipfrs_storage_cache_hit_rate {:.4}",
            metrics.cache_hit_rate()
        );

        prom
    }

    fn export_diagnostics_prometheus(report: &DiagnosticsReport) -> String {
        let mut prom = String::new();

        let _ = writeln!(
            prom,
            "# HELP ipfrs_storage_health_score Storage health score (0-100)"
        );
        let _ = writeln!(prom, "# TYPE ipfrs_storage_health_score gauge");
        let _ = writeln!(
            prom,
            "ipfrs_storage_health_score{{backend=\"{}\"}} {}",
            report.backend, report.health_score
        );

        let _ = writeln!(
            prom,
            "# HELP ipfrs_storage_write_throughput Write throughput in blocks/sec"
        );
        let _ = writeln!(prom, "# TYPE ipfrs_storage_write_throughput gauge");
        let _ = writeln!(
            prom,
            "ipfrs_storage_write_throughput{{backend=\"{}\"}} {:.2}",
            report.backend, report.performance.write_throughput
        );

        let _ = writeln!(
            prom,
            "# HELP ipfrs_storage_read_throughput Read throughput in blocks/sec"
        );
        let _ = writeln!(prom, "# TYPE ipfrs_storage_read_throughput gauge");
        let _ = writeln!(
            prom,
            "ipfrs_storage_read_throughput{{backend=\"{}\"}} {:.2}",
            report.backend, report.performance.read_throughput
        );

        let _ = writeln!(
            prom,
            "# HELP ipfrs_storage_peak_memory_usage Peak memory usage in bytes"
        );
        let _ = writeln!(prom, "# TYPE ipfrs_storage_peak_memory_usage gauge");
        let _ = writeln!(
            prom,
            "ipfrs_storage_peak_memory_usage{{backend=\"{}\"}} {}",
            report.backend, report.performance.peak_memory_usage
        );

        prom
    }

    fn export_analysis_prometheus(analysis: &StorageAnalysis) -> String {
        let mut prom = String::new();

        let grade_score = match analysis.grade.as_str() {
            "A" => 5,
            "B" => 4,
            "C" => 3,
            "D" => 2,
            _ => 1,
        };

        let _ = writeln!(prom, "# HELP ipfrs_storage_grade Storage grade (1-5)");
        let _ = writeln!(prom, "# TYPE ipfrs_storage_grade gauge");
        let _ = writeln!(
            prom,
            "ipfrs_storage_grade{{backend=\"{}\"}} {}",
            analysis.backend, grade_score
        );

        let _ = writeln!(
            prom,
            "# HELP ipfrs_storage_recommendation_count Number of recommendations"
        );
        let _ = writeln!(prom, "# TYPE ipfrs_storage_recommendation_count gauge");
        let _ = writeln!(
            prom,
            "ipfrs_storage_recommendation_count{{backend=\"{}\"}} {}",
            analysis.backend,
            analysis.recommendations.len()
        );

        prom
    }

    // Text format exports
    fn export_metrics_text(metrics: &StorageMetrics) -> String {
        format!(
            "Storage Metrics:\n\
             Put Operations: {}\n\
             Get Operations: {}\n\
             Has Operations: {}\n\
             Delete Operations: {}\n\
             Cache Hits: {}\n\
             Cache Misses: {}\n\
             Cache Hit Rate: {:.2}%\n\
             Bytes Written: {}\n\
             Bytes Read: {}\n\
             Avg Put Latency: {}μs\n\
             Avg Get Latency: {}μs\n\
             Avg Has Latency: {}μs\n\
             Peak Put Latency: {}μs\n\
             Peak Get Latency: {}μs\n\
             Errors: {}\n",
            metrics.put_count,
            metrics.get_count,
            metrics.has_count,
            metrics.delete_count,
            metrics.get_hits,
            metrics.get_misses,
            metrics.cache_hit_rate() * 100.0,
            metrics.bytes_written,
            metrics.bytes_read,
            metrics.avg_put_latency_us,
            metrics.avg_get_latency_us,
            metrics.avg_has_latency_us,
            metrics.peak_put_latency_us,
            metrics.peak_get_latency_us,
            metrics.error_count,
        )
    }

    fn export_diagnostics_text(report: &DiagnosticsReport) -> String {
        format!(
            "=== Diagnostics Report: {} ===\n\
             Total Blocks: {}\n\
             Health Score: {}/100\n\n\
             Performance:\n\
             - Write Throughput: {:.2} blocks/sec\n\
             - Read Throughput: {:.2} blocks/sec\n\
             - Avg Write Latency: {:?}\n\
             - Avg Read Latency: {:?}\n\
             - Peak Memory Usage: {:.2} MB\n\n\
             Health:\n\
             - Successful Ops: {}\n\
             - Failed Ops: {}\n\
             - Success Rate: {:.2}%\n\
             - Integrity OK: {}\n\
             - Responsive: {}\n",
            report.backend,
            report.total_blocks,
            report.health_score,
            report.performance.write_throughput,
            report.performance.read_throughput,
            report.performance.avg_write_latency,
            report.performance.avg_read_latency,
            report.performance.peak_memory_usage as f64 / (1024.0 * 1024.0),
            report.health.successful_ops,
            report.health.failed_ops,
            report.health.success_rate * 100.0,
            report.health.integrity_ok,
            report.health.responsive,
        )
    }

    fn export_analysis_text(analysis: &StorageAnalysis) -> String {
        format!(
            "=== Storage Analysis: {} ===\n\
             Grade: {}\n\
             Health Score: {}/100\n\
             Workload Type: {:?}\n\
             Read/Write Ratio: {:.2}% reads\n\
             Avg Block Size: {} bytes\n\
             Recommendations: {}\n",
            analysis.backend,
            analysis.grade,
            analysis.diagnostics.health_score,
            analysis.workload.workload_type,
            analysis.workload.read_write_ratio * 100.0,
            analysis.workload.avg_block_size,
            analysis.recommendations.len(),
        )
    }
}

/// Batch exporter for exporting multiple metrics at once
pub struct BatchExporter {
    exports: Vec<(String, String)>,
}

impl BatchExporter {
    /// Create a new batch exporter
    pub fn new() -> Self {
        Self {
            exports: Vec::new(),
        }
    }

    /// Add metrics to export
    pub fn add_metrics(&mut self, name: &str, metrics: &StorageMetrics, format: ExportFormat) {
        let exported = MetricExporter::export_metrics(metrics, format);
        self.exports.push((name.to_string(), exported));
    }

    /// Add diagnostics to export
    pub fn add_diagnostics(
        &mut self,
        name: &str,
        report: &DiagnosticsReport,
        format: ExportFormat,
    ) {
        let exported = MetricExporter::export_diagnostics(report, format);
        self.exports.push((name.to_string(), exported));
    }

    /// Get all exports
    pub fn get_exports(&self) -> &[(String, String)] {
        &self.exports
    }

    /// Export all as a single document
    pub fn export_all(&self) -> String {
        let mut result = String::new();
        for (name, content) in &self.exports {
            result.push_str(&format!("=== {name} ===\n"));
            result.push_str(content);
            result.push_str("\n\n");
        }
        result
    }
}

impl Default for BatchExporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metrics() -> StorageMetrics {
        StorageMetrics {
            put_count: 1000,
            get_count: 2000,
            has_count: 500,
            delete_count: 100,
            get_hits: 1800,
            get_misses: 200,
            bytes_written: 1024000,
            bytes_read: 2048000,
            avg_put_latency_us: 100,
            avg_get_latency_us: 50,
            avg_has_latency_us: 25,
            peak_put_latency_us: 500,
            peak_get_latency_us: 200,
            error_count: 10,
            batch_op_count: 50,
            batch_items_count: 500,
            avg_batch_size: 10,
        }
    }

    #[test]
    fn test_export_metrics_json() {
        let metrics = sample_metrics();
        let exported = MetricExporter::export_metrics(&metrics, ExportFormat::Json);
        assert!(exported.contains("put_count"));
        assert!(exported.contains("1000"));
    }

    #[test]
    fn test_export_metrics_csv() {
        let metrics = sample_metrics();
        let exported = MetricExporter::export_metrics(&metrics, ExportFormat::Csv);
        assert!(exported.contains("metric,value"));
        assert!(exported.contains("put_count,1000"));
    }

    #[test]
    fn test_export_metrics_prometheus() {
        let metrics = sample_metrics();
        let exported = MetricExporter::export_metrics(&metrics, ExportFormat::Prometheus);
        assert!(exported.contains("# HELP"));
        assert!(exported.contains("# TYPE"));
        assert!(exported.contains("ipfrs_storage_put_total"));
    }

    #[test]
    fn test_export_metrics_text() {
        let metrics = sample_metrics();
        let exported = MetricExporter::export_metrics(&metrics, ExportFormat::Text);
        assert!(exported.contains("Storage Metrics"));
        assert!(exported.contains("Put Operations: 1000"));
    }

    #[test]
    fn test_batch_exporter() {
        let mut exporter = BatchExporter::new();
        let metrics = sample_metrics();

        exporter.add_metrics("test1", &metrics, ExportFormat::Json);
        exporter.add_metrics("test2", &metrics, ExportFormat::Csv);

        assert_eq!(exporter.get_exports().len(), 2);

        let all = exporter.export_all();
        assert!(all.contains("=== test1 ==="));
        assert!(all.contains("=== test2 ==="));
    }
}
