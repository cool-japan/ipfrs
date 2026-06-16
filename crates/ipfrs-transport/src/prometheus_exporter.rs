//! Prometheus metrics exporter for transport layer
//!
//! This module provides utilities to export transport metrics in Prometheus format
//! for monitoring and alerting.
//!
//! # Example
//!
//! ```
//! use ipfrs_transport::prometheus_exporter::PrometheusExporter;
//!
//! let mut exporter = PrometheusExporter::new();
//!
//! // Record metrics
//! exporter.record_block_request("peer1", 1024);
//! exporter.record_block_latency("peer1", 50);
//! exporter.record_peer_connection("peer1", "QUIC");
//!
//! // Export in Prometheus format
//! let metrics = exporter.export();
//! assert!(metrics.contains("ipfrs_transport_blocks_requested_total"));
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Prometheus metric type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    /// Counter - monotonically increasing value
    Counter,
    /// Gauge - can go up or down
    Gauge,
    /// Histogram - distribution of values
    Histogram,
}

/// A single metric value
#[derive(Debug, Clone)]
struct MetricValue {
    /// Metric type
    metric_type: MetricType,
    /// Current value
    value: f64,
    /// Help text
    help: String,
}

/// Prometheus metrics exporter for transport layer
pub struct PrometheusExporter {
    metrics: Arc<Mutex<HashMap<String, MetricValue>>>,
    labels: Arc<Mutex<HashMap<String, HashMap<String, String>>>>,
}

impl PrometheusExporter {
    /// Create a new Prometheus exporter
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(HashMap::new())),
            labels: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Record a counter metric
    pub fn record_counter(&mut self, name: &str, value: f64, help: &str) {
        let mut metrics = self.metrics.lock().unwrap_or_else(|e| e.into_inner());
        let entry = metrics.entry(name.to_string()).or_insert(MetricValue {
            metric_type: MetricType::Counter,
            value: 0.0,
            help: help.to_string(),
        });
        entry.value += value;
    }

    /// Record a gauge metric
    pub fn record_gauge(&mut self, name: &str, value: f64, help: &str) {
        let mut metrics = self.metrics.lock().unwrap_or_else(|e| e.into_inner());
        metrics.insert(
            name.to_string(),
            MetricValue {
                metric_type: MetricType::Gauge,
                value,
                help: help.to_string(),
            },
        );
    }

    /// Add labels to a metric
    pub fn add_labels(&mut self, name: &str, labels: HashMap<String, String>) {
        let mut all_labels = self.labels.lock().unwrap_or_else(|e| e.into_inner());
        all_labels.insert(name.to_string(), labels);
    }

    /// Record a block request
    pub fn record_block_request(&mut self, peer_id: &str, bytes: usize) {
        self.record_counter(
            "ipfrs_transport_blocks_requested_total",
            1.0,
            "Total number of blocks requested",
        );
        self.record_counter(
            "ipfrs_transport_bytes_requested_total",
            bytes as f64,
            "Total bytes requested",
        );

        let mut labels = HashMap::new();
        labels.insert("peer_id".to_string(), peer_id.to_string());
        self.add_labels("ipfrs_transport_blocks_requested_total", labels);
    }

    /// Record a block received
    pub fn record_block_received(&mut self, peer_id: &str, bytes: usize) {
        self.record_counter(
            "ipfrs_transport_blocks_received_total",
            1.0,
            "Total number of blocks received",
        );
        self.record_counter(
            "ipfrs_transport_bytes_received_total",
            bytes as f64,
            "Total bytes received",
        );

        let mut labels = HashMap::new();
        labels.insert("peer_id".to_string(), peer_id.to_string());
        self.add_labels("ipfrs_transport_blocks_received_total", labels);
    }

    /// Record block request latency
    pub fn record_block_latency(&mut self, peer_id: &str, latency_ms: u64) {
        self.record_gauge(
            "ipfrs_transport_block_latency_ms",
            latency_ms as f64,
            "Block request latency in milliseconds",
        );

        let mut labels = HashMap::new();
        labels.insert("peer_id".to_string(), peer_id.to_string());
        self.add_labels("ipfrs_transport_block_latency_ms", labels);
    }

    /// Record a peer connection
    pub fn record_peer_connection(&mut self, peer_id: &str, transport_type: &str) {
        self.record_gauge(
            "ipfrs_transport_peers_connected",
            1.0,
            "Number of connected peers",
        );

        let mut labels = HashMap::new();
        labels.insert("peer_id".to_string(), peer_id.to_string());
        labels.insert("transport".to_string(), transport_type.to_string());
        self.add_labels("ipfrs_transport_peers_connected", labels);
    }

    /// Record a peer disconnection
    pub fn record_peer_disconnection(&mut self, peer_id: &str) {
        self.record_gauge(
            "ipfrs_transport_peers_connected",
            0.0,
            "Number of connected peers",
        );

        let mut labels = HashMap::new();
        labels.insert("peer_id".to_string(), peer_id.to_string());
        self.add_labels("ipfrs_transport_peers_connected", labels);
    }

    /// Record a request failure
    pub fn record_request_failure(&mut self, peer_id: &str, error_type: &str) {
        self.record_counter(
            "ipfrs_transport_requests_failed_total",
            1.0,
            "Total number of failed requests",
        );

        let mut labels = HashMap::new();
        labels.insert("peer_id".to_string(), peer_id.to_string());
        labels.insert("error_type".to_string(), error_type.to_string());
        self.add_labels("ipfrs_transport_requests_failed_total", labels);
    }

    /// Record session metrics
    pub fn record_session_metrics(
        &mut self,
        session_id: &str,
        blocks: usize,
        bytes: u64,
        duration_ms: u64,
    ) {
        self.record_counter(
            "ipfrs_transport_sessions_total",
            1.0,
            "Total number of sessions",
        );

        self.record_gauge(
            "ipfrs_transport_session_blocks",
            blocks as f64,
            "Number of blocks in session",
        );

        self.record_gauge(
            "ipfrs_transport_session_bytes",
            bytes as f64,
            "Bytes transferred in session",
        );

        self.record_gauge(
            "ipfrs_transport_session_duration_ms",
            duration_ms as f64,
            "Session duration in milliseconds",
        );

        let mut labels = HashMap::new();
        labels.insert("session_id".to_string(), session_id.to_string());
        self.add_labels("ipfrs_transport_sessions_total", labels.clone());
        self.add_labels("ipfrs_transport_session_blocks", labels.clone());
        self.add_labels("ipfrs_transport_session_bytes", labels.clone());
        self.add_labels("ipfrs_transport_session_duration_ms", labels);
    }

    /// Record want list metrics
    pub fn record_want_list_size(&mut self, size: usize) {
        self.record_gauge(
            "ipfrs_transport_want_list_size",
            size as f64,
            "Current size of want list",
        );
    }

    /// Record peer manager metrics
    pub fn record_peer_count(&mut self, count: usize) {
        self.record_gauge(
            "ipfrs_transport_peer_count",
            count as f64,
            "Number of known peers",
        );
    }

    /// Export metrics in Prometheus text format
    pub fn export(&self) -> String {
        let metrics = self.metrics.lock().unwrap_or_else(|e| e.into_inner());
        let labels = self.labels.lock().unwrap_or_else(|e| e.into_inner());

        let mut output = String::new();

        for (name, metric) in metrics.iter() {
            // Write HELP line
            output.push_str(&format!("# HELP {} {}\n", name, metric.help));

            // Write TYPE line
            let type_str = match metric.metric_type {
                MetricType::Counter => "counter",
                MetricType::Gauge => "gauge",
                MetricType::Histogram => "histogram",
            };
            output.push_str(&format!("# TYPE {} {}\n", name, type_str));

            // Write metric value with labels
            if let Some(metric_labels) = labels.get(name) {
                if !metric_labels.is_empty() {
                    let labels_str: Vec<String> = metric_labels
                        .iter()
                        .map(|(k, v)| format!("{}=\"{}\"", k, v))
                        .collect();
                    output.push_str(&format!(
                        "{}{{{}}} {}\n",
                        name,
                        labels_str.join(","),
                        metric.value
                    ));
                } else {
                    output.push_str(&format!("{} {}\n", name, metric.value));
                }
            } else {
                output.push_str(&format!("{} {}\n", name, metric.value));
            }
        }

        output
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        let mut metrics = self.metrics.lock().unwrap_or_else(|e| e.into_inner());
        let mut labels = self.labels.lock().unwrap_or_else(|e| e.into_inner());
        metrics.clear();
        labels.clear();
    }

    /// Get metric count
    pub fn metric_count(&self) -> usize {
        let metrics = self.metrics.lock().unwrap_or_else(|e| e.into_inner());
        metrics.len()
    }
}

impl Default for PrometheusExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for PrometheusExporter {
    fn clone(&self) -> Self {
        Self {
            metrics: Arc::clone(&self.metrics),
            labels: Arc::clone(&self.labels),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prometheus_exporter_creation() {
        let exporter = PrometheusExporter::new();
        assert_eq!(exporter.metric_count(), 0);
    }

    #[test]
    fn test_record_counter() {
        let mut exporter = PrometheusExporter::new();
        exporter.record_counter("test_counter", 1.0, "Test counter");
        exporter.record_counter("test_counter", 2.0, "Test counter");

        let output = exporter.export();
        assert!(output.contains("# HELP test_counter Test counter"));
        assert!(output.contains("# TYPE test_counter counter"));
        assert!(output.contains("test_counter 3"));
    }

    #[test]
    fn test_record_gauge() {
        let mut exporter = PrometheusExporter::new();
        exporter.record_gauge("test_gauge", 42.0, "Test gauge");

        let output = exporter.export();
        assert!(output.contains("# HELP test_gauge Test gauge"));
        assert!(output.contains("# TYPE test_gauge gauge"));
        assert!(output.contains("test_gauge 42"));
    }

    #[test]
    fn test_record_block_request() {
        let mut exporter = PrometheusExporter::new();
        exporter.record_block_request("peer1", 1024);

        let output = exporter.export();
        assert!(output.contains("ipfrs_transport_blocks_requested_total"));
        assert!(output.contains("ipfrs_transport_bytes_requested_total"));
    }

    #[test]
    fn test_record_block_received() {
        let mut exporter = PrometheusExporter::new();
        exporter.record_block_received("peer1", 2048);

        let output = exporter.export();
        assert!(output.contains("ipfrs_transport_blocks_received_total"));
        assert!(output.contains("ipfrs_transport_bytes_received_total"));
    }

    #[test]
    fn test_record_block_latency() {
        let mut exporter = PrometheusExporter::new();
        exporter.record_block_latency("peer1", 50);

        let output = exporter.export();
        assert!(output.contains("ipfrs_transport_block_latency_ms"));
        assert!(output.contains("peer_id=\"peer1\""));
    }

    #[test]
    fn test_record_peer_connection() {
        let mut exporter = PrometheusExporter::new();
        exporter.record_peer_connection("peer1", "QUIC");

        let output = exporter.export();
        assert!(output.contains("ipfrs_transport_peers_connected"));
        assert!(output.contains("transport=\"QUIC\""));
    }

    #[test]
    fn test_record_request_failure() {
        let mut exporter = PrometheusExporter::new();
        exporter.record_request_failure("peer1", "timeout");

        let output = exporter.export();
        assert!(output.contains("ipfrs_transport_requests_failed_total"));
        assert!(output.contains("error_type=\"timeout\""));
    }

    #[test]
    fn test_record_session_metrics() {
        let mut exporter = PrometheusExporter::new();
        exporter.record_session_metrics("session1", 100, 1024000, 5000);

        let output = exporter.export();
        assert!(output.contains("ipfrs_transport_sessions_total"));
        assert!(output.contains("ipfrs_transport_session_blocks"));
        assert!(output.contains("ipfrs_transport_session_bytes"));
        assert!(output.contains("ipfrs_transport_session_duration_ms"));
    }

    #[test]
    fn test_reset() {
        let mut exporter = PrometheusExporter::new();
        exporter.record_counter("test", 1.0, "Test");

        assert_eq!(exporter.metric_count(), 1);
        exporter.reset();
        assert_eq!(exporter.metric_count(), 0);
    }

    #[test]
    fn test_clone_exporter() {
        let mut exporter1 = PrometheusExporter::new();
        exporter1.record_counter("test", 1.0, "Test");

        let exporter2 = exporter1.clone();
        assert_eq!(exporter2.metric_count(), 1);
    }

    #[test]
    fn test_multiple_metrics() {
        let mut exporter = PrometheusExporter::new();
        exporter.record_counter("counter1", 1.0, "Counter 1");
        exporter.record_gauge("gauge1", 42.0, "Gauge 1");
        exporter.record_counter("counter2", 2.0, "Counter 2");

        assert_eq!(exporter.metric_count(), 3);

        let output = exporter.export();
        assert!(output.contains("counter1"));
        assert!(output.contains("gauge1"));
        assert!(output.contains("counter2"));
    }

    #[test]
    fn test_labels() {
        let mut exporter = PrometheusExporter::new();
        exporter.record_counter("test_metric", 1.0, "Test");

        let mut labels = HashMap::new();
        labels.insert("key1".to_string(), "value1".to_string());
        labels.insert("key2".to_string(), "value2".to_string());
        exporter.add_labels("test_metric", labels);

        let output = exporter.export();
        assert!(output.contains("key1=\"value1\""));
        assert!(output.contains("key2=\"value2\""));
    }
}
