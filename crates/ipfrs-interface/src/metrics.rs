//! Prometheus metrics for observability
//!
//! This module provides comprehensive metrics collection for monitoring
//! IPFRS interface performance, usage patterns, and system health.

use lazy_static::lazy_static;
use prometheus::{
    register_counter_vec, register_gauge_vec, register_histogram_vec, register_int_counter_vec,
    register_int_gauge_vec, CounterVec, Encoder, GaugeVec, HistogramVec, IntCounterVec,
    IntGaugeVec, TextEncoder,
};
use std::time::Instant;

lazy_static! {
    // HTTP Request Metrics

    /// Total number of HTTP requests by endpoint and method
    pub static ref HTTP_REQUESTS_TOTAL: IntCounterVec = register_int_counter_vec!(
        "ipfrs_http_requests_total",
        "Total number of HTTP requests",
        &["endpoint", "method", "status"]
    )
    .unwrap();

    /// HTTP request duration in seconds
    pub static ref HTTP_REQUEST_DURATION_SECONDS: HistogramVec = register_histogram_vec!(
        "ipfrs_http_request_duration_seconds",
        "HTTP request duration in seconds",
        &["endpoint", "method"],
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    .unwrap();

    /// HTTP request body size in bytes
    pub static ref HTTP_REQUEST_SIZE_BYTES: HistogramVec = register_histogram_vec!(
        "ipfrs_http_request_size_bytes",
        "HTTP request body size in bytes",
        &["endpoint", "method"],
        vec![
            100.0,
            1_000.0,
            10_000.0,
            100_000.0,
            1_000_000.0,
            10_000_000.0,
            100_000_000.0
        ]
    )
    .unwrap();

    /// HTTP response size in bytes
    pub static ref HTTP_RESPONSE_SIZE_BYTES: HistogramVec = register_histogram_vec!(
        "ipfrs_http_response_size_bytes",
        "HTTP response body size in bytes",
        &["endpoint", "method"],
        vec![
            100.0,
            1_000.0,
            10_000.0,
            100_000.0,
            1_000_000.0,
            10_000_000.0,
            100_000_000.0
        ]
    )
    .unwrap();

    /// Currently active HTTP connections
    pub static ref HTTP_CONNECTIONS_ACTIVE: IntGaugeVec = register_int_gauge_vec!(
        "ipfrs_http_connections_active",
        "Currently active HTTP connections",
        &["endpoint"]
    )
    .unwrap();

    // Block Operations Metrics

    /// Total blocks retrieved
    pub static ref BLOCKS_RETRIEVED_TOTAL: IntCounterVec = register_int_counter_vec!(
        "ipfrs_blocks_retrieved_total",
        "Total number of blocks retrieved",
        &["source"]
    )
    .unwrap();

    /// Total blocks stored
    pub static ref BLOCKS_STORED_TOTAL: IntCounterVec = register_int_counter_vec!(
        "ipfrs_blocks_stored_total",
        "Total number of blocks stored",
        &["destination"]
    )
    .unwrap();

    /// Block operation errors
    pub static ref BLOCK_ERRORS_TOTAL: IntCounterVec = register_int_counter_vec!(
        "ipfrs_block_errors_total",
        "Total number of block operation errors",
        &["operation", "error_type"]
    )
    .unwrap();

    /// Block retrieval duration
    pub static ref BLOCK_RETRIEVAL_DURATION_SECONDS: HistogramVec = register_histogram_vec!(
        "ipfrs_block_retrieval_duration_seconds",
        "Block retrieval duration in seconds",
        &["source"],
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    )
    .unwrap();

    // Batch Operations Metrics

    /// Batch operation size (number of items)
    pub static ref BATCH_OPERATION_SIZE: HistogramVec = register_histogram_vec!(
        "ipfrs_batch_operation_size",
        "Number of items in batch operations",
        &["operation"],
        vec![1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    )
    .unwrap();

    /// Batch operation duration
    pub static ref BATCH_OPERATION_DURATION_SECONDS: HistogramVec = register_histogram_vec!(
        "ipfrs_batch_operation_duration_seconds",
        "Batch operation duration in seconds",
        &["operation"],
        vec![0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0]
    )
    .unwrap();

    // Streaming Metrics

    /// Total bytes uploaded
    pub static ref UPLOAD_BYTES_TOTAL: CounterVec = register_counter_vec!(
        "ipfrs_upload_bytes_total",
        "Total bytes uploaded",
        &["endpoint"]
    )
    .unwrap();

    /// Total bytes downloaded
    pub static ref DOWNLOAD_BYTES_TOTAL: CounterVec = register_counter_vec!(
        "ipfrs_download_bytes_total",
        "Total bytes downloaded",
        &["endpoint"]
    )
    .unwrap();

    /// Active streaming operations
    pub static ref STREAMING_OPERATIONS_ACTIVE: IntGaugeVec = register_int_gauge_vec!(
        "ipfrs_streaming_operations_active",
        "Currently active streaming operations",
        &["type"]
    )
    .unwrap();

    /// Streaming chunk size
    pub static ref STREAMING_CHUNK_SIZE_BYTES: HistogramVec = register_histogram_vec!(
        "ipfrs_streaming_chunk_size_bytes",
        "Streaming chunk size in bytes",
        &["operation"],
        vec![
            1024.0,
            4096.0,
            16384.0,
            65536.0,
            262144.0,
            1048576.0
        ]
    )
    .unwrap();

    // Cache Metrics

    /// Cache hits
    pub static ref CACHE_HITS_TOTAL: IntCounterVec = register_int_counter_vec!(
        "ipfrs_cache_hits_total",
        "Total cache hits",
        &["cache_type"]
    )
    .unwrap();

    /// Cache misses
    pub static ref CACHE_MISSES_TOTAL: IntCounterVec = register_int_counter_vec!(
        "ipfrs_cache_misses_total",
        "Total cache misses",
        &["cache_type"]
    )
    .unwrap();

    /// Current cache size
    pub static ref CACHE_SIZE_BYTES: GaugeVec = register_gauge_vec!(
        "ipfrs_cache_size_bytes",
        "Current cache size in bytes",
        &["cache_type"]
    )
    .unwrap();

    // Authentication Metrics

    /// Authentication attempts
    pub static ref AUTH_ATTEMPTS_TOTAL: IntCounterVec = register_int_counter_vec!(
        "ipfrs_auth_attempts_total",
        "Total authentication attempts",
        &["method", "result"]
    )
    .unwrap();

    /// Active authenticated sessions
    pub static ref AUTH_SESSIONS_ACTIVE: IntGaugeVec = register_int_gauge_vec!(
        "ipfrs_auth_sessions_active",
        "Currently active authenticated sessions",
        &["user"]
    )
    .unwrap();

    // Rate Limiting Metrics

    /// Rate limit hits (requests blocked)
    pub static ref RATE_LIMIT_HITS_TOTAL: IntCounterVec = register_int_counter_vec!(
        "ipfrs_rate_limit_hits_total",
        "Total rate limit hits (requests blocked)",
        &["endpoint", "client_ip"]
    )
    .unwrap();

    /// Available rate limit tokens
    pub static ref RATE_LIMIT_TOKENS_AVAILABLE: GaugeVec = register_gauge_vec!(
        "ipfrs_rate_limit_tokens_available",
        "Available rate limit tokens",
        &["client_ip"]
    )
    .unwrap();

    // WebSocket Metrics

    /// Active WebSocket connections
    pub static ref WEBSOCKET_CONNECTIONS_ACTIVE: IntGaugeVec = register_int_gauge_vec!(
        "ipfrs_websocket_connections_active",
        "Currently active WebSocket connections",
        &["topic"]
    )
    .unwrap();

    /// WebSocket messages sent
    pub static ref WEBSOCKET_MESSAGES_SENT_TOTAL: IntCounterVec = register_int_counter_vec!(
        "ipfrs_websocket_messages_sent_total",
        "Total WebSocket messages sent",
        &["topic", "event_type"]
    )
    .unwrap();

    /// WebSocket messages received
    pub static ref WEBSOCKET_MESSAGES_RECEIVED_TOTAL: IntCounterVec = register_int_counter_vec!(
        "ipfrs_websocket_messages_received_total",
        "Total WebSocket messages received",
        &["message_type"]
    )
    .unwrap();

    // gRPC Metrics

    /// gRPC requests total
    pub static ref GRPC_REQUESTS_TOTAL: IntCounterVec = register_int_counter_vec!(
        "ipfrs_grpc_requests_total",
        "Total gRPC requests",
        &["service", "method", "status"]
    )
    .unwrap();

    /// gRPC request duration
    pub static ref GRPC_REQUEST_DURATION_SECONDS: HistogramVec = register_histogram_vec!(
        "ipfrs_grpc_request_duration_seconds",
        "gRPC request duration in seconds",
        &["service", "method"],
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
    )
    .unwrap();

    // Tensor Operations Metrics

    /// Tensor operations total
    pub static ref TENSOR_OPERATIONS_TOTAL: IntCounterVec = register_int_counter_vec!(
        "ipfrs_tensor_operations_total",
        "Total tensor operations",
        &["operation", "dtype"]
    )
    .unwrap();

    /// Tensor slice operations
    pub static ref TENSOR_SLICE_OPERATIONS_TOTAL: IntCounterVec = register_int_counter_vec!(
        "ipfrs_tensor_slice_operations_total",
        "Total tensor slice operations",
        &["dimensions"]
    )
    .unwrap();

    /// Tensor size in bytes
    pub static ref TENSOR_SIZE_BYTES: HistogramVec = register_histogram_vec!(
        "ipfrs_tensor_size_bytes",
        "Tensor size in bytes",
        &["dtype"],
        vec![
            1000.0,
            10_000.0,
            100_000.0,
            1_000_000.0,
            10_000_000.0,
            100_000_000.0,
            1_000_000_000.0
        ]
    )
    .unwrap();

    // System Metrics

    /// Total memory allocated (in bytes)
    pub static ref MEMORY_ALLOCATED_BYTES: IntGaugeVec = register_int_gauge_vec!(
        "ipfrs_memory_allocated_bytes",
        "Total memory allocated in bytes",
        &["component"]
    )
    .unwrap();

    /// Number of goroutines (async tasks)
    pub static ref ASYNC_TASKS_ACTIVE: IntGaugeVec = register_int_gauge_vec!(
        "ipfrs_async_tasks_active",
        "Currently active async tasks",
        &["type"]
    )
    .unwrap();
}

/// Helper struct for timing operations
pub struct Timer {
    start: Instant,
    labels: Vec<String>,
}

impl Timer {
    /// Create a new timer with labels
    pub fn new(labels: Vec<String>) -> Self {
        Self {
            start: Instant::now(),
            labels,
        }
    }

    /// Observe the duration and record it to the given histogram
    pub fn observe_duration(self, histogram: &HistogramVec) {
        let duration = self.start.elapsed().as_secs_f64();
        histogram
            .with_label_values(&self.labels.iter().map(|s| s.as_str()).collect::<Vec<_>>())
            .observe(duration);
    }
}

/// Record an HTTP request
#[allow(dead_code)]
pub fn record_http_request(endpoint: &str, method: &str, status: u16) {
    HTTP_REQUESTS_TOTAL
        .with_label_values(&[endpoint, method, &status.to_string()])
        .inc();
}

/// Start timing an HTTP request
#[allow(dead_code)]
pub fn start_http_request_timer(endpoint: &str, method: &str) -> Timer {
    HTTP_CONNECTIONS_ACTIVE.with_label_values(&[endpoint]).inc();
    Timer::new(vec![endpoint.to_string(), method.to_string()])
}

/// Finish timing an HTTP request
#[allow(dead_code)]
pub fn finish_http_request_timer(timer: Timer, endpoint: &str) {
    timer.observe_duration(&HTTP_REQUEST_DURATION_SECONDS);
    HTTP_CONNECTIONS_ACTIVE.with_label_values(&[endpoint]).dec();
}

/// Record HTTP request size
#[allow(dead_code)]
pub fn record_http_request_size(endpoint: &str, method: &str, size: usize) {
    HTTP_REQUEST_SIZE_BYTES
        .with_label_values(&[endpoint, method])
        .observe(size as f64);
}

/// Record HTTP response size
#[allow(dead_code)]
pub fn record_http_response_size(endpoint: &str, method: &str, size: usize) {
    HTTP_RESPONSE_SIZE_BYTES
        .with_label_values(&[endpoint, method])
        .observe(size as f64);
}

/// Record block retrieval
#[allow(dead_code)]
pub fn record_block_retrieved(source: &str) {
    BLOCKS_RETRIEVED_TOTAL.with_label_values(&[source]).inc();
}

/// Record block storage
#[allow(dead_code)]
pub fn record_block_stored(destination: &str) {
    BLOCKS_STORED_TOTAL.with_label_values(&[destination]).inc();
}

/// Record block error
#[allow(dead_code)]
pub fn record_block_error(operation: &str, error_type: &str) {
    BLOCK_ERRORS_TOTAL
        .with_label_values(&[operation, error_type])
        .inc();
}

/// Record upload bytes
#[allow(dead_code)]
pub fn record_upload_bytes(endpoint: &str, bytes: u64) {
    UPLOAD_BYTES_TOTAL
        .with_label_values(&[endpoint])
        .inc_by(bytes as f64);
}

/// Record download bytes
#[allow(dead_code)]
pub fn record_download_bytes(endpoint: &str, bytes: u64) {
    DOWNLOAD_BYTES_TOTAL
        .with_label_values(&[endpoint])
        .inc_by(bytes as f64);
}

/// Record cache hit
#[allow(dead_code)]
pub fn record_cache_hit(cache_type: &str) {
    CACHE_HITS_TOTAL.with_label_values(&[cache_type]).inc();
}

/// Record cache miss
#[allow(dead_code)]
pub fn record_cache_miss(cache_type: &str) {
    CACHE_MISSES_TOTAL.with_label_values(&[cache_type]).inc();
}

/// Record authentication attempt
#[allow(dead_code)]
pub fn record_auth_attempt(method: &str, result: &str) {
    AUTH_ATTEMPTS_TOTAL
        .with_label_values(&[method, result])
        .inc();
}

/// Record rate limit hit
#[allow(dead_code)]
pub fn record_rate_limit_hit(endpoint: &str, client_ip: &str) {
    RATE_LIMIT_HITS_TOTAL
        .with_label_values(&[endpoint, client_ip])
        .inc();
}

/// Encode all metrics in Prometheus text format
pub fn encode_metrics() -> Result<String, prometheus::Error> {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer)?;
    String::from_utf8(buffer)
        .map_err(|e| prometheus::Error::Msg(format!("Failed to encode metrics as UTF-8: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_http_request() {
        record_http_request("/api/v0/add", "POST", 200);
        let metrics = encode_metrics().unwrap();
        assert!(metrics.contains("ipfrs_http_requests_total"));
    }

    #[test]
    fn test_timer() {
        let timer = Timer::new(vec!["test".to_string(), "GET".to_string()]);
        std::thread::sleep(std::time::Duration::from_millis(10));
        timer.observe_duration(&HTTP_REQUEST_DURATION_SECONDS);

        let metrics = encode_metrics().unwrap();
        assert!(metrics.contains("ipfrs_http_request_duration_seconds"));
    }

    #[test]
    fn test_record_block_operations() {
        record_block_retrieved("local");
        record_block_stored("blockstore");
        record_block_error("get", "not_found");

        let metrics = encode_metrics().unwrap();
        assert!(metrics.contains("ipfrs_blocks_retrieved_total"));
        assert!(metrics.contains("ipfrs_blocks_stored_total"));
        assert!(metrics.contains("ipfrs_block_errors_total"));
    }

    #[test]
    fn test_record_cache_operations() {
        record_cache_hit("block_cache");
        record_cache_miss("block_cache");

        let metrics = encode_metrics().unwrap();
        assert!(metrics.contains("ipfrs_cache_hits_total"));
        assert!(metrics.contains("ipfrs_cache_misses_total"));
    }

    #[test]
    fn test_encode_metrics() {
        // Record some metrics to ensure encoder has data
        record_http_request("/test", "GET", 200);
        record_block_retrieved("test_store");

        let result = encode_metrics();
        assert!(result.is_ok());

        let metrics = result.unwrap();
        // Metrics should include at least the recorded ones
        assert!(
            metrics.contains("ipfrs_http_requests_total")
                || metrics.contains("ipfrs_blocks_retrieved_total")
        );
    }
}
