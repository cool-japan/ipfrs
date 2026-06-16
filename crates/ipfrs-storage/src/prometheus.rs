//! Prometheus metrics exporter
//!
//! This module provides integration with Prometheus for production monitoring.
//! It exports storage metrics in Prometheus text format for scraping.
//!
//! # Example
//!
//! ```rust,no_run
//! use ipfrs_storage::{PrometheusExporter, StorageMetrics};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let metrics = StorageMetrics::default();
//! let exporter = PrometheusExporter::new("ipfrs_storage".to_string());
//! let prometheus_text = exporter.export(&metrics);
//! println!("{}", prometheus_text);
//! # Ok(())
//! # }
//! ```

use crate::StorageMetrics;
use std::fmt::Write;

/// Prometheus metrics exporter
#[derive(Debug, Clone)]
pub struct PrometheusExporter {
    /// Namespace for metrics (e.g., "ipfrs_storage")
    namespace: String,
    /// Additional labels to add to all metrics
    labels: Vec<(String, String)>,
}

impl PrometheusExporter {
    /// Create a new Prometheus exporter
    pub fn new(namespace: String) -> Self {
        Self {
            namespace,
            labels: Vec::new(),
        }
    }

    /// Add a label to all exported metrics
    pub fn with_label(mut self, key: String, value: String) -> Self {
        self.labels.push((key, value));
        self
    }

    /// Export metrics in Prometheus text format
    pub fn export(&self, metrics: &StorageMetrics) -> String {
        let mut output = String::new();
        let labels = self.format_labels();

        // Helper macro to write metrics
        macro_rules! write_metric {
            ($name:expr, $type:expr, $help:expr, $value:expr) => {
                writeln!(output, "# HELP {}_{} {}", self.namespace, $name, $help)
                    .expect("write to String is infallible");
                writeln!(output, "# TYPE {}_{} {}", self.namespace, $name, $type)
                    .expect("write to String is infallible");
                writeln!(output, "{}_{}{} {}", self.namespace, $name, labels, $value)
                    .expect("write to String is infallible");
            };
        }

        // Operation counters
        write_metric!(
            "put_total",
            "counter",
            "Total number of put operations",
            metrics.put_count
        );
        write_metric!(
            "get_total",
            "counter",
            "Total number of get operations",
            metrics.get_count
        );
        write_metric!(
            "has_total",
            "counter",
            "Total number of has operations",
            metrics.has_count
        );
        write_metric!(
            "delete_total",
            "counter",
            "Total number of delete operations",
            metrics.delete_count
        );

        // Cache metrics
        write_metric!(
            "get_hits_total",
            "counter",
            "Total number of successful gets",
            metrics.get_hits
        );
        write_metric!(
            "get_misses_total",
            "counter",
            "Total number of failed gets",
            metrics.get_misses
        );
        write_metric!(
            "cache_hit_rate",
            "gauge",
            "Cache hit rate (0.0 to 1.0)",
            metrics.cache_hit_rate()
        );

        // Bytes transferred
        write_metric!(
            "bytes_written_total",
            "counter",
            "Total bytes written",
            metrics.bytes_written
        );
        write_metric!(
            "bytes_read_total",
            "counter",
            "Total bytes read",
            metrics.bytes_read
        );

        // Latency metrics
        write_metric!(
            "put_latency_microseconds",
            "gauge",
            "Average put operation latency in microseconds",
            metrics.avg_put_latency_us
        );
        write_metric!(
            "get_latency_microseconds",
            "gauge",
            "Average get operation latency in microseconds",
            metrics.avg_get_latency_us
        );
        write_metric!(
            "has_latency_microseconds",
            "gauge",
            "Average has operation latency in microseconds",
            metrics.avg_has_latency_us
        );
        write_metric!(
            "peak_put_latency_microseconds",
            "gauge",
            "Peak put operation latency in microseconds",
            metrics.peak_put_latency_us
        );
        write_metric!(
            "peak_get_latency_microseconds",
            "gauge",
            "Peak get operation latency in microseconds",
            metrics.peak_get_latency_us
        );
        write_metric!(
            "operation_latency_microseconds",
            "gauge",
            "Average operation latency in microseconds",
            metrics.avg_operation_latency_us()
        );

        // Error metrics
        write_metric!(
            "errors_total",
            "counter",
            "Total number of errors encountered",
            metrics.error_count
        );

        output
    }

    /// Format labels for Prometheus
    fn format_labels(&self) -> String {
        if self.labels.is_empty() {
            String::new()
        } else {
            let label_str = self
                .labels
                .iter()
                .map(|(k, v)| format!("{k}=\"{v}\""))
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{label_str}}}")
        }
    }

    /// Export metrics as HTTP response body (suitable for /metrics endpoint)
    pub fn export_http(&self, metrics: &StorageMetrics) -> (String, String) {
        let body = self.export(metrics);
        let content_type = "text/plain; version=0.0.4; charset=utf-8".to_string();
        (content_type, body)
    }
}

/// Builder for creating a Prometheus exporter with multiple configurations
#[derive(Debug, Default)]
pub struct PrometheusExporterBuilder {
    namespace: Option<String>,
    labels: Vec<(String, String)>,
}

impl PrometheusExporterBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the namespace for metrics
    pub fn namespace(mut self, namespace: String) -> Self {
        self.namespace = Some(namespace);
        self
    }

    /// Add a label to all metrics
    pub fn label(mut self, key: String, value: String) -> Self {
        self.labels.push((key, value));
        self
    }

    /// Add the instance label (common for Prometheus)
    pub fn instance(self, instance: String) -> Self {
        self.label("instance".to_string(), instance)
    }

    /// Add the job label (common for Prometheus)
    pub fn job(self, job: String) -> Self {
        self.label("job".to_string(), job)
    }

    /// Build the exporter
    pub fn build(self) -> PrometheusExporter {
        let namespace = self
            .namespace
            .unwrap_or_else(|| "ipfrs_storage".to_string());
        PrometheusExporter {
            namespace,
            labels: self.labels,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prometheus_export_basic() {
        let metrics = StorageMetrics {
            put_count: 100,
            get_count: 200,
            get_hits: 180,
            get_misses: 20,
            bytes_written: 1024000,
            bytes_read: 2048000,
            ..StorageMetrics::default()
        };

        let exporter = PrometheusExporter::new("test".to_string());
        let output = exporter.export(&metrics);

        // Check that output contains expected metrics
        assert!(output.contains("# HELP test_put_total"));
        assert!(output.contains("# TYPE test_put_total counter"));
        assert!(output.contains("test_put_total 100"));
        assert!(output.contains("test_get_total 200"));
        assert!(output.contains("test_get_hits_total 180"));
        assert!(output.contains("test_get_misses_total 20"));
        assert!(output.contains("test_bytes_written_total 1024000"));
        assert!(output.contains("test_bytes_read_total 2048000"));
    }

    #[test]
    fn test_prometheus_export_with_labels() {
        let metrics = StorageMetrics::default();
        let exporter = PrometheusExporter::new("test".to_string())
            .with_label("instance".to_string(), "node1".to_string())
            .with_label("datacenter".to_string(), "us-west".to_string());

        let output = exporter.export(&metrics);

        // Check that labels are included
        assert!(output.contains("{instance=\"node1\",datacenter=\"us-west\"}"));
    }

    #[test]
    fn test_prometheus_export_cache_hit_rate() {
        let metrics = StorageMetrics {
            get_hits: 90,
            get_misses: 10,
            ..StorageMetrics::default()
        };

        let exporter = PrometheusExporter::new("test".to_string());
        let output = exporter.export(&metrics);

        // Cache hit rate should be 0.9
        assert!(output.contains("test_cache_hit_rate 0.9"));
    }

    #[test]
    fn test_prometheus_export_builder() {
        let exporter = PrometheusExporterBuilder::new()
            .namespace("custom".to_string())
            .instance("node1".to_string())
            .job("storage".to_string())
            .label("region".to_string(), "us-east".to_string())
            .build();

        let metrics = StorageMetrics::default();
        let output = exporter.export(&metrics);

        assert!(output.contains("custom_put_total"));
        assert!(output.contains("instance=\"node1\""));
        assert!(output.contains("job=\"storage\""));
        assert!(output.contains("region=\"us-east\""));
    }

    #[test]
    fn test_http_export() {
        let metrics = StorageMetrics::default();
        let exporter = PrometheusExporter::new("test".to_string());
        let (content_type, body) = exporter.export_http(&metrics);

        assert_eq!(content_type, "text/plain; version=0.0.4; charset=utf-8");
        assert!(body.contains("# HELP"));
        assert!(body.contains("# TYPE"));
    }
}
