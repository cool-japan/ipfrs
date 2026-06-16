//! Metrics collection and Prometheus integration
//!
//! This module provides comprehensive metrics collection for IPFRS operations,
//! including block storage, semantic search, logic programming, and network operations.

use metrics::{counter, gauge, histogram};
use metrics_exporter_prometheus::PrometheusBuilder;
use std::net::SocketAddr;
use std::time::Instant;

/// Metrics registry for IPFRS
pub struct MetricsRegistry {
    start_time: Instant,
}

impl MetricsRegistry {
    /// Create a new metrics registry
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
        }
    }

    /// Initialize Prometheus exporter on the given address
    ///
    /// This sets up a Prometheus metrics endpoint that can be scraped
    /// by Prometheus server.
    ///
    /// # Arguments
    /// * `addr` - Socket address to bind the metrics endpoint
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::metrics::MetricsRegistry;
    /// use std::net::SocketAddr;
    ///
    /// let registry = MetricsRegistry::new();
    /// let addr: SocketAddr = "127.0.0.1:9000".parse().expect("valid socket address");
    /// registry.init_prometheus(addr).expect("prometheus init should succeed");
    /// ```
    pub fn init_prometheus(&self, addr: SocketAddr) -> Result<(), Box<dyn std::error::Error>> {
        PrometheusBuilder::new()
            .with_http_listener(addr)
            .install()?;
        tracing::info!(
            "Prometheus metrics endpoint initialized at http://{}/metrics",
            addr
        );
        Ok(())
    }

    /// Get uptime in seconds
    pub fn uptime_seconds(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// Block Storage Metrics

/// Record a block put operation
pub fn record_block_put(size_bytes: usize, duration_ms: f64) {
    counter!("ipfrs_block_put_total").increment(1);
    histogram!("ipfrs_block_put_duration_ms").record(duration_ms);
    histogram!("ipfrs_block_size_bytes").record(size_bytes as f64);
    counter!("ipfrs_block_bytes_written").increment(size_bytes as u64);
}

/// Record a block get operation
pub fn record_block_get(found: bool, duration_ms: f64) {
    counter!("ipfrs_block_get_total").increment(1);
    if found {
        counter!("ipfrs_block_get_found").increment(1);
    } else {
        counter!("ipfrs_block_get_not_found").increment(1);
    }
    histogram!("ipfrs_block_get_duration_ms").record(duration_ms);
}

/// Record a block delete operation
pub fn record_block_delete(duration_ms: f64) {
    counter!("ipfrs_block_delete_total").increment(1);
    histogram!("ipfrs_block_delete_duration_ms").record(duration_ms);
}

/// Update block count gauge
pub fn set_block_count(count: usize) {
    gauge!("ipfrs_block_count").set(count as f64);
}

/// Update total storage size gauge
pub fn set_storage_size_bytes(size: u64) {
    gauge!("ipfrs_storage_size_bytes").set(size as f64);
}

// Semantic Search Metrics

/// Record vector indexing operation
pub fn record_vector_index(dimension: usize, duration_ms: f64) {
    counter!("ipfrs_vector_index_total").increment(1);
    histogram!("ipfrs_vector_index_duration_ms").record(duration_ms);
    gauge!("ipfrs_vector_dimension").set(dimension as f64);
}

/// Record similarity search operation
pub fn record_similarity_search(k: usize, results: usize, duration_ms: f64) {
    counter!("ipfrs_similarity_search_total").increment(1);
    histogram!("ipfrs_similarity_search_duration_ms").record(duration_ms);
    histogram!("ipfrs_similarity_search_k").record(k as f64);
    histogram!("ipfrs_similarity_search_results").record(results as f64);
}

/// Update vector count gauge
pub fn set_vector_count(count: usize) {
    gauge!("ipfrs_vector_count").set(count as f64);
}

/// Record cache hit/miss
pub fn record_cache_access(hit: bool) {
    counter!("ipfrs_cache_access_total").increment(1);
    if hit {
        counter!("ipfrs_cache_hit").increment(1);
    } else {
        counter!("ipfrs_cache_miss").increment(1);
    }
}

// Logic Programming Metrics

/// Record fact addition
pub fn record_fact_add(duration_ms: f64) {
    counter!("ipfrs_logic_fact_add_total").increment(1);
    histogram!("ipfrs_logic_fact_add_duration_ms").record(duration_ms);
}

/// Record rule addition
pub fn record_rule_add(duration_ms: f64) {
    counter!("ipfrs_logic_rule_add_total").increment(1);
    histogram!("ipfrs_logic_rule_add_duration_ms").record(duration_ms);
}

/// Record inference operation
pub fn record_inference(results: usize, duration_ms: f64) {
    counter!("ipfrs_logic_inference_total").increment(1);
    histogram!("ipfrs_logic_inference_duration_ms").record(duration_ms);
    histogram!("ipfrs_logic_inference_results").record(results as f64);
}

/// Record proof generation
pub fn record_proof_generation(success: bool, duration_ms: f64) {
    counter!("ipfrs_logic_proof_total").increment(1);
    if success {
        counter!("ipfrs_logic_proof_success").increment(1);
    } else {
        counter!("ipfrs_logic_proof_failure").increment(1);
    }
    histogram!("ipfrs_logic_proof_duration_ms").record(duration_ms);
}

/// Update knowledge base stats
pub fn set_kb_stats(facts: usize, rules: usize) {
    gauge!("ipfrs_logic_facts_count").set(facts as f64);
    gauge!("ipfrs_logic_rules_count").set(rules as f64);
}

// Network Metrics

/// Record peer connection
pub fn record_peer_connect() {
    counter!("ipfrs_network_peer_connect_total").increment(1);
}

/// Record peer disconnection
pub fn record_peer_disconnect() {
    counter!("ipfrs_network_peer_disconnect_total").increment(1);
}

/// Update peer count gauge
pub fn set_peer_count(count: usize) {
    gauge!("ipfrs_network_peer_count").set(count as f64);
}

/// Record bytes sent
pub fn record_bytes_sent(bytes: usize) {
    counter!("ipfrs_network_bytes_sent").increment(bytes as u64);
}

/// Record bytes received
pub fn record_bytes_received(bytes: usize) {
    counter!("ipfrs_network_bytes_received").increment(bytes as u64);
}

/// Record DHT query
pub fn record_dht_query(success: bool, duration_ms: f64) {
    counter!("ipfrs_network_dht_query_total").increment(1);
    if success {
        counter!("ipfrs_network_dht_query_success").increment(1);
    } else {
        counter!("ipfrs_network_dht_query_failure").increment(1);
    }
    histogram!("ipfrs_network_dht_query_duration_ms").record(duration_ms);
}

// HTTP API Metrics

/// Record HTTP request
pub fn record_http_request(method: &str, path: &str, status: u16, duration_ms: f64) {
    counter!("ipfrs_http_requests_total", "method" => method.to_string(), "path" => path.to_string(), "status" => status.to_string()).increment(1);
    histogram!("ipfrs_http_request_duration_ms", "method" => method.to_string(), "path" => path.to_string()).record(duration_ms);
}

/// Record HTTP error
pub fn record_http_error(method: &str, path: &str, status: u16) {
    counter!("ipfrs_http_errors_total", "method" => method.to_string(), "path" => path.to_string(), "status" => status.to_string()).increment(1);
}

// System Metrics

/// Update system uptime
pub fn set_uptime_seconds(seconds: u64) {
    gauge!("ipfrs_uptime_seconds").set(seconds as f64);
}

/// Record error
pub fn record_error(component: &str, error_type: &str) {
    counter!("ipfrs_errors_total", "component" => component.to_string(), "type" => error_type.to_string()).increment(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_registry_creation() {
        let registry = MetricsRegistry::new();
        assert!(registry.uptime_seconds() == 0);
    }

    #[test]
    fn test_block_metrics() {
        record_block_put(1024, 10.5);
        record_block_get(true, 5.2);
        record_block_delete(3.1);
        set_block_count(100);
        set_storage_size_bytes(1024000);
    }

    #[test]
    fn test_semantic_metrics() {
        record_vector_index(768, 15.0);
        record_similarity_search(10, 8, 20.5);
        set_vector_count(1000);
        record_cache_access(true);
        record_cache_access(false);
    }

    #[test]
    fn test_logic_metrics() {
        record_fact_add(2.0);
        record_rule_add(3.5);
        record_inference(5, 50.0);
        record_proof_generation(true, 100.0);
        set_kb_stats(100, 10);
    }

    #[test]
    fn test_network_metrics() {
        record_peer_connect();
        record_peer_disconnect();
        set_peer_count(5);
        record_bytes_sent(1024);
        record_bytes_received(2048);
        record_dht_query(true, 25.0);
    }

    #[test]
    fn test_http_metrics() {
        record_http_request("GET", "/api/v0/block/get", 200, 15.5);
        record_http_error("POST", "/api/v0/block/put", 500);
    }

    #[test]
    fn test_system_metrics() {
        set_uptime_seconds(3600);
        record_error("storage", "disk_full");
    }
}
