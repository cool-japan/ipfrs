//! Prometheus metrics for observability
//!
//! This module provides comprehensive metrics collection for monitoring
//! IPFRS interface performance, usage patterns, and system health.
//!
//! # Global registry (lazy_static)
//!
//! The module exposes a set of well-known `lazy_static` metrics that register
//! themselves in the **default** prometheus registry.  These are used by the
//! HTTP middleware and route helpers.
//!
//! # Per-node registry (`IpfrsMetrics`)
//!
//! For fine-grained per-node observability (block operations, DHT, inference
//! sessions, GossipSub, storage and GC) each `Node` or gateway instance
//! can hold an [`IpfrsMetrics`] value.  Metrics are registered in a
//! **private** [`prometheus::Registry`] so multiple test instances do not
//! collide.

use lazy_static::lazy_static;
use prometheus::{
    register_counter_vec, register_gauge_vec, register_histogram_vec, register_int_counter_vec,
    register_int_gauge_vec, Counter, CounterVec, Encoder, Gauge, GaugeVec, Histogram,
    HistogramOpts, HistogramVec, IntCounterVec, IntGaugeVec, Opts, Registry, TextEncoder,
};
use std::sync::Arc;
use std::time::Instant;

lazy_static! {
    // HTTP Request Metrics

    /// Total number of HTTP requests by endpoint and method
    pub static ref HTTP_REQUESTS_TOTAL: IntCounterVec = register_int_counter_vec!(
        "ipfrs_http_requests_total",
        "Total number of HTTP requests",
        &["endpoint", "method", "status"]
    )
    .expect("prometheus metric registration failed");

    /// HTTP request duration in seconds
    pub static ref HTTP_REQUEST_DURATION_SECONDS: HistogramVec = register_histogram_vec!(
        "ipfrs_http_request_duration_seconds",
        "HTTP request duration in seconds",
        &["endpoint", "method"],
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    .expect("prometheus metric registration failed");

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
    .expect("prometheus metric registration failed");

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
    .expect("prometheus metric registration failed");

    /// Currently active HTTP connections
    pub static ref HTTP_CONNECTIONS_ACTIVE: IntGaugeVec = register_int_gauge_vec!(
        "ipfrs_http_connections_active",
        "Currently active HTTP connections",
        &["endpoint"]
    )
    .expect("prometheus metric registration failed");

    // Block Operations Metrics

    /// Total blocks retrieved
    pub static ref BLOCKS_RETRIEVED_TOTAL: IntCounterVec = register_int_counter_vec!(
        "ipfrs_blocks_retrieved_total",
        "Total number of blocks retrieved",
        &["source"]
    )
    .expect("prometheus metric registration failed");

    /// Total blocks stored
    pub static ref BLOCKS_STORED_TOTAL: IntCounterVec = register_int_counter_vec!(
        "ipfrs_blocks_stored_total",
        "Total number of blocks stored",
        &["destination"]
    )
    .expect("prometheus metric registration failed");

    /// Block operation errors
    pub static ref BLOCK_ERRORS_TOTAL: IntCounterVec = register_int_counter_vec!(
        "ipfrs_block_errors_total",
        "Total number of block operation errors",
        &["operation", "error_type"]
    )
    .expect("prometheus metric registration failed");

    /// Block retrieval duration
    pub static ref BLOCK_RETRIEVAL_DURATION_SECONDS: HistogramVec = register_histogram_vec!(
        "ipfrs_block_retrieval_duration_seconds",
        "Block retrieval duration in seconds",
        &["source"],
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    )
    .expect("prometheus metric registration failed");

    // Batch Operations Metrics

    /// Batch operation size (number of items)
    pub static ref BATCH_OPERATION_SIZE: HistogramVec = register_histogram_vec!(
        "ipfrs_batch_operation_size",
        "Number of items in batch operations",
        &["operation"],
        vec![1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    )
    .expect("prometheus metric registration failed");

    /// Batch operation duration
    pub static ref BATCH_OPERATION_DURATION_SECONDS: HistogramVec = register_histogram_vec!(
        "ipfrs_batch_operation_duration_seconds",
        "Batch operation duration in seconds",
        &["operation"],
        vec![0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0]
    )
    .expect("prometheus metric registration failed");

    // Streaming Metrics

    /// Total bytes uploaded
    pub static ref UPLOAD_BYTES_TOTAL: CounterVec = register_counter_vec!(
        "ipfrs_upload_bytes_total",
        "Total bytes uploaded",
        &["endpoint"]
    )
    .expect("prometheus metric registration failed");

    /// Total bytes downloaded
    pub static ref DOWNLOAD_BYTES_TOTAL: CounterVec = register_counter_vec!(
        "ipfrs_download_bytes_total",
        "Total bytes downloaded",
        &["endpoint"]
    )
    .expect("prometheus metric registration failed");

    /// Active streaming operations
    pub static ref STREAMING_OPERATIONS_ACTIVE: IntGaugeVec = register_int_gauge_vec!(
        "ipfrs_streaming_operations_active",
        "Currently active streaming operations",
        &["type"]
    )
    .expect("prometheus metric registration failed");

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
    .expect("prometheus metric registration failed");

    // Cache Metrics

    /// Cache hits
    pub static ref CACHE_HITS_TOTAL: IntCounterVec = register_int_counter_vec!(
        "ipfrs_cache_hits_total",
        "Total cache hits",
        &["cache_type"]
    )
    .expect("prometheus metric registration failed");

    /// Cache misses
    pub static ref CACHE_MISSES_TOTAL: IntCounterVec = register_int_counter_vec!(
        "ipfrs_cache_misses_total",
        "Total cache misses",
        &["cache_type"]
    )
    .expect("prometheus metric registration failed");

    /// Current cache size
    pub static ref CACHE_SIZE_BYTES: GaugeVec = register_gauge_vec!(
        "ipfrs_cache_size_bytes",
        "Current cache size in bytes",
        &["cache_type"]
    )
    .expect("prometheus metric registration failed");

    // Authentication Metrics

    /// Authentication attempts
    pub static ref AUTH_ATTEMPTS_TOTAL: IntCounterVec = register_int_counter_vec!(
        "ipfrs_auth_attempts_total",
        "Total authentication attempts",
        &["method", "result"]
    )
    .expect("prometheus metric registration failed");

    /// Active authenticated sessions
    pub static ref AUTH_SESSIONS_ACTIVE: IntGaugeVec = register_int_gauge_vec!(
        "ipfrs_auth_sessions_active",
        "Currently active authenticated sessions",
        &["user"]
    )
    .expect("prometheus metric registration failed");

    // Rate Limiting Metrics

    /// Rate limit hits (requests blocked)
    pub static ref RATE_LIMIT_HITS_TOTAL: IntCounterVec = register_int_counter_vec!(
        "ipfrs_rate_limit_hits_total",
        "Total rate limit hits (requests blocked)",
        &["endpoint", "client_ip"]
    )
    .expect("prometheus metric registration failed");

    /// Available rate limit tokens
    pub static ref RATE_LIMIT_TOKENS_AVAILABLE: GaugeVec = register_gauge_vec!(
        "ipfrs_rate_limit_tokens_available",
        "Available rate limit tokens",
        &["client_ip"]
    )
    .expect("prometheus metric registration failed");

    // WebSocket Metrics

    /// Active WebSocket connections
    pub static ref WEBSOCKET_CONNECTIONS_ACTIVE: IntGaugeVec = register_int_gauge_vec!(
        "ipfrs_websocket_connections_active",
        "Currently active WebSocket connections",
        &["topic"]
    )
    .expect("prometheus metric registration failed");

    /// WebSocket messages sent
    pub static ref WEBSOCKET_MESSAGES_SENT_TOTAL: IntCounterVec = register_int_counter_vec!(
        "ipfrs_websocket_messages_sent_total",
        "Total WebSocket messages sent",
        &["topic", "event_type"]
    )
    .expect("prometheus metric registration failed");

    /// WebSocket messages received
    pub static ref WEBSOCKET_MESSAGES_RECEIVED_TOTAL: IntCounterVec = register_int_counter_vec!(
        "ipfrs_websocket_messages_received_total",
        "Total WebSocket messages received",
        &["message_type"]
    )
    .expect("prometheus metric registration failed");

    // gRPC Metrics

    /// gRPC requests total
    pub static ref GRPC_REQUESTS_TOTAL: IntCounterVec = register_int_counter_vec!(
        "ipfrs_grpc_requests_total",
        "Total gRPC requests",
        &["service", "method", "status"]
    )
    .expect("prometheus metric registration failed");

    /// gRPC request duration
    pub static ref GRPC_REQUEST_DURATION_SECONDS: HistogramVec = register_histogram_vec!(
        "ipfrs_grpc_request_duration_seconds",
        "gRPC request duration in seconds",
        &["service", "method"],
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
    )
    .expect("prometheus metric registration failed");

    // Tensor Operations Metrics

    /// Tensor operations total
    pub static ref TENSOR_OPERATIONS_TOTAL: IntCounterVec = register_int_counter_vec!(
        "ipfrs_tensor_operations_total",
        "Total tensor operations",
        &["operation", "dtype"]
    )
    .expect("prometheus metric registration failed");

    /// Tensor slice operations
    pub static ref TENSOR_SLICE_OPERATIONS_TOTAL: IntCounterVec = register_int_counter_vec!(
        "ipfrs_tensor_slice_operations_total",
        "Total tensor slice operations",
        &["dimensions"]
    )
    .expect("prometheus metric registration failed");

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
    .expect("prometheus metric registration failed");

    // System Metrics

    /// Total memory allocated (in bytes)
    pub static ref MEMORY_ALLOCATED_BYTES: IntGaugeVec = register_int_gauge_vec!(
        "ipfrs_memory_allocated_bytes",
        "Total memory allocated in bytes",
        &["component"]
    )
    .expect("prometheus metric registration failed");

    /// Number of goroutines (async tasks)
    pub static ref ASYNC_TASKS_ACTIVE: IntGaugeVec = register_int_gauge_vec!(
        "ipfrs_async_tasks_active",
        "Currently active async tasks",
        &["type"]
    )
    .expect("prometheus metric registration failed");
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

// ============================================================================
// IpfrsMetrics — per-node registry
// ============================================================================

/// Global metrics registry for IPFRS.
///
/// Each field is a Prometheus primitive registered in a **private** registry
/// so that multiple instances (e.g., in tests) do not conflict with each other.
///
/// ```rust
/// use ipfrs_interface::metrics::IpfrsMetrics;
///
/// let m = IpfrsMetrics::new().expect("failed to create metrics registry");
/// m.blocks_added.inc();
/// let text = m.render();
/// assert!(text.contains("ipfrs_blocks_added_total"));
/// ```
pub struct IpfrsMetrics {
    /// Private registry — metrics exported via `render()`.
    pub registry: Registry,

    // ----- Block operations --------------------------------------------------
    /// Total blocks added to local storage.
    pub blocks_added: Counter,
    /// Total blocks fetched from local storage or network.
    pub blocks_fetched: Counter,
    /// Total blocks deleted.
    pub blocks_deleted: Counter,
    /// Total bytes written via `add_bytes` / `add_file`.
    pub block_add_bytes: Counter,
    /// Latency (seconds) for block fetch operations.
    pub block_fetch_latency: Histogram,
    /// Rolling cache hit-rate (0.0 – 1.0).
    pub cache_hit_rate: Gauge,

    // ----- DHT ---------------------------------------------------------------
    /// Number of DHT `provide` calls.
    pub dht_provide_calls: Counter,
    /// Number of DHT `find_providers` calls.
    pub dht_find_providers_calls: Counter,
    /// Current number of provider records held by this node.
    pub dht_provider_records: Gauge,

    // ----- Inference sessions ------------------------------------------------
    /// Inference sessions started.
    pub inference_sessions_started: Counter,
    /// Inference sessions completed successfully.
    pub inference_sessions_completed: Counter,
    /// End-to-end latency (seconds) for completed inference sessions.
    pub inference_session_latency: Histogram,
    /// Depth of proof trees emitted during inference.
    pub proof_tree_depth: Histogram,

    // ----- GossipSub ---------------------------------------------------------
    /// Messages published to GossipSub topics.
    pub gossipsub_messages_sent: Counter,
    /// Messages received from GossipSub topics.
    pub gossipsub_messages_received: Counter,
    /// Current number of mesh peers across all topics.
    pub gossipsub_mesh_peers: Gauge,

    // ----- Storage / GC ------------------------------------------------------
    /// Total bytes currently stored on disk (updated periodically).
    pub storage_bytes_total: Gauge,
    /// Total number of blocks currently stored on disk.
    pub storage_blocks_total: Gauge,
    /// Number of GC runs completed.
    pub gc_runs: Counter,
    /// Total blocks collected (deleted) across all GC runs.
    pub gc_blocks_collected: Counter,
}

impl IpfrsMetrics {
    /// Create a new metrics registry with all counters, gauges, and histograms
    /// initialised to zero.
    ///
    /// # Errors
    ///
    /// Returns a [`prometheus::Error`] if any metric registration fails (which
    /// should only happen if two metrics share the same name).
    pub fn new() -> Result<Self, prometheus::Error> {
        let registry = Registry::new();

        // ---- Block operations ----
        let blocks_added = Counter::with_opts(Opts::new(
            "ipfrs_blocks_added_total",
            "Total number of blocks added to local storage",
        ))?;
        registry.register(Box::new(blocks_added.clone()))?;

        let blocks_fetched = Counter::with_opts(Opts::new(
            "ipfrs_blocks_fetched_total",
            "Total number of blocks fetched from storage or network",
        ))?;
        registry.register(Box::new(blocks_fetched.clone()))?;

        let blocks_deleted = Counter::with_opts(Opts::new(
            "ipfrs_blocks_deleted_total",
            "Total number of blocks deleted from local storage",
        ))?;
        registry.register(Box::new(blocks_deleted.clone()))?;

        let block_add_bytes = Counter::with_opts(Opts::new(
            "ipfrs_block_add_bytes_total",
            "Total bytes written to local storage via add operations",
        ))?;
        registry.register(Box::new(block_add_bytes.clone()))?;

        let block_fetch_latency = Histogram::with_opts(
            HistogramOpts::new(
                "ipfrs_block_fetch_latency_seconds",
                "Latency of block fetch operations in seconds",
            )
            .buckets(vec![
                0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
            ]),
        )?;
        registry.register(Box::new(block_fetch_latency.clone()))?;

        let cache_hit_rate = Gauge::with_opts(Opts::new(
            "ipfrs_cache_hit_rate",
            "Rolling cache hit rate in the range [0.0, 1.0]",
        ))?;
        registry.register(Box::new(cache_hit_rate.clone()))?;

        // ---- DHT ----
        let dht_provide_calls = Counter::with_opts(Opts::new(
            "ipfrs_dht_provide_calls_total",
            "Total number of DHT provide calls",
        ))?;
        registry.register(Box::new(dht_provide_calls.clone()))?;

        let dht_find_providers_calls = Counter::with_opts(Opts::new(
            "ipfrs_dht_find_providers_calls_total",
            "Total number of DHT find_providers calls",
        ))?;
        registry.register(Box::new(dht_find_providers_calls.clone()))?;

        let dht_provider_records = Gauge::with_opts(Opts::new(
            "ipfrs_dht_provider_records",
            "Current number of DHT provider records held by this node",
        ))?;
        registry.register(Box::new(dht_provider_records.clone()))?;

        // ---- Inference sessions ----
        let inference_sessions_started = Counter::with_opts(Opts::new(
            "ipfrs_inference_sessions_started_total",
            "Total number of inference sessions started",
        ))?;
        registry.register(Box::new(inference_sessions_started.clone()))?;

        let inference_sessions_completed = Counter::with_opts(Opts::new(
            "ipfrs_inference_sessions_completed_total",
            "Total number of inference sessions completed successfully",
        ))?;
        registry.register(Box::new(inference_sessions_completed.clone()))?;

        let inference_session_latency = Histogram::with_opts(
            HistogramOpts::new(
                "ipfrs_inference_session_latency_seconds",
                "End-to-end latency of completed inference sessions in seconds",
            )
            .buckets(vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]),
        )?;
        registry.register(Box::new(inference_session_latency.clone()))?;

        let proof_tree_depth = Histogram::with_opts(
            HistogramOpts::new(
                "ipfrs_proof_tree_depth",
                "Depth of proof trees emitted during inference",
            )
            .buckets(vec![1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0]),
        )?;
        registry.register(Box::new(proof_tree_depth.clone()))?;

        // ---- GossipSub ----
        let gossipsub_messages_sent = Counter::with_opts(Opts::new(
            "ipfrs_gossipsub_messages_sent_total",
            "Total number of messages published to GossipSub topics",
        ))?;
        registry.register(Box::new(gossipsub_messages_sent.clone()))?;

        let gossipsub_messages_received = Counter::with_opts(Opts::new(
            "ipfrs_gossipsub_messages_received_total",
            "Total number of messages received from GossipSub topics",
        ))?;
        registry.register(Box::new(gossipsub_messages_received.clone()))?;

        let gossipsub_mesh_peers = Gauge::with_opts(Opts::new(
            "ipfrs_gossipsub_mesh_peers",
            "Current number of GossipSub mesh peers across all topics",
        ))?;
        registry.register(Box::new(gossipsub_mesh_peers.clone()))?;

        // ---- Storage / GC ----
        let storage_bytes_total = Gauge::with_opts(Opts::new(
            "ipfrs_storage_bytes_total",
            "Total bytes currently stored on disk",
        ))?;
        registry.register(Box::new(storage_bytes_total.clone()))?;

        let storage_blocks_total = Gauge::with_opts(Opts::new(
            "ipfrs_storage_blocks_total",
            "Total number of blocks currently stored on disk",
        ))?;
        registry.register(Box::new(storage_blocks_total.clone()))?;

        let gc_runs = Counter::with_opts(Opts::new(
            "ipfrs_gc_runs_total",
            "Total number of garbage collection runs completed",
        ))?;
        registry.register(Box::new(gc_runs.clone()))?;

        let gc_blocks_collected = Counter::with_opts(Opts::new(
            "ipfrs_gc_blocks_collected_total",
            "Total number of blocks collected (deleted) across all GC runs",
        ))?;
        registry.register(Box::new(gc_blocks_collected.clone()))?;

        Ok(Self {
            registry,
            blocks_added,
            blocks_fetched,
            blocks_deleted,
            block_add_bytes,
            block_fetch_latency,
            cache_hit_rate,
            dht_provide_calls,
            dht_find_providers_calls,
            dht_provider_records,
            inference_sessions_started,
            inference_sessions_completed,
            inference_session_latency,
            proof_tree_depth,
            gossipsub_messages_sent,
            gossipsub_messages_received,
            gossipsub_mesh_peers,
            storage_bytes_total,
            storage_blocks_total,
            gc_runs,
            gc_blocks_collected,
        })
    }

    /// Render all registered metrics in Prometheus text exposition format.
    ///
    /// Returns an empty string if encoding fails.
    pub fn render(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        if encoder.encode(&metric_families, &mut buffer).is_err() {
            return String::new();
        }
        String::from_utf8(buffer).unwrap_or_default()
    }
}

impl Default for IpfrsMetrics {
    fn default() -> Self {
        Self::new().expect("IpfrsMetrics::default() failed to create registry")
    }
}

/// A thread-safe shared handle to an [`IpfrsMetrics`] instance.
pub type SharedMetrics = Arc<IpfrsMetrics>;

/// Create a new [`SharedMetrics`] instance.
///
/// # Errors
///
/// Returns a [`prometheus::Error`] if metric registration fails.
pub fn new_shared_metrics() -> Result<SharedMetrics, prometheus::Error> {
    IpfrsMetrics::new().map(Arc::new)
}

// ============================================================================
// Global encode helper
// ============================================================================

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
        let metrics = encode_metrics().expect("test: metrics encoding should succeed");
        assert!(metrics.contains("ipfrs_http_requests_total"));
    }

    #[test]
    fn test_timer() {
        let timer = Timer::new(vec!["test".to_string(), "GET".to_string()]);
        std::thread::sleep(std::time::Duration::from_millis(10));
        timer.observe_duration(&HTTP_REQUEST_DURATION_SECONDS);

        let metrics = encode_metrics().expect("test: metrics encoding should succeed");
        assert!(metrics.contains("ipfrs_http_request_duration_seconds"));
    }

    #[test]
    fn test_record_block_operations() {
        record_block_retrieved("local");
        record_block_stored("blockstore");
        record_block_error("get", "not_found");

        let metrics = encode_metrics().expect("test: metrics encoding should succeed");
        assert!(metrics.contains("ipfrs_blocks_retrieved_total"));
        assert!(metrics.contains("ipfrs_blocks_stored_total"));
        assert!(metrics.contains("ipfrs_block_errors_total"));
    }

    #[test]
    fn test_record_cache_operations() {
        record_cache_hit("block_cache");
        record_cache_miss("block_cache");

        let metrics = encode_metrics().expect("test: metrics encoding should succeed");
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

        let metrics = result.expect("test: encode_metrics should return valid UTF-8 metrics");
        // Metrics should include at least the recorded ones
        assert!(
            metrics.contains("ipfrs_http_requests_total")
                || metrics.contains("ipfrs_blocks_retrieved_total")
        );
    }

    // ---- IpfrsMetrics (private registry) tests ----

    /// All counters, gauges and histograms must start at zero after construction.
    #[test]
    fn test_metrics_default_zero() {
        let m = IpfrsMetrics::new().expect("should create metrics");

        // Counters start at 0
        assert_eq!(m.blocks_added.get() as u64, 0);
        assert_eq!(m.blocks_fetched.get() as u64, 0);
        assert_eq!(m.blocks_deleted.get() as u64, 0);
        assert_eq!(m.block_add_bytes.get() as u64, 0);
        assert_eq!(m.dht_provide_calls.get() as u64, 0);
        assert_eq!(m.dht_find_providers_calls.get() as u64, 0);
        assert_eq!(m.inference_sessions_started.get() as u64, 0);
        assert_eq!(m.inference_sessions_completed.get() as u64, 0);
        assert_eq!(m.gossipsub_messages_sent.get() as u64, 0);
        assert_eq!(m.gossipsub_messages_received.get() as u64, 0);
        assert_eq!(m.gc_runs.get() as u64, 0);
        assert_eq!(m.gc_blocks_collected.get() as u64, 0);

        // Gauges start at 0
        assert_eq!(m.cache_hit_rate.get(), 0.0_f64);
        assert_eq!(m.dht_provider_records.get(), 0.0_f64);
        assert_eq!(m.gossipsub_mesh_peers.get(), 0.0_f64);
        assert_eq!(m.storage_bytes_total.get(), 0.0_f64);
        assert_eq!(m.storage_blocks_total.get(), 0.0_f64);
    }

    /// Incrementing `blocks_added` must be reflected in `render()`.
    #[test]
    fn test_metrics_increment() {
        let m = IpfrsMetrics::new().expect("should create metrics");
        m.blocks_added.inc();
        m.blocks_added.inc();

        let text = m.render();
        // The rendered text must contain the metric name and the value "2"
        assert!(
            text.contains("ipfrs_blocks_added_total"),
            "render() must include blocks_added counter"
        );
        assert!(
            text.contains("ipfrs_blocks_added_total 2"),
            "render() value should be 2 but got:\n{text}"
        );
    }

    /// `render()` must produce valid Prometheus text format output.
    #[test]
    fn test_metrics_render_format() {
        let m = IpfrsMetrics::new().expect("should create metrics");
        // Touch at least one metric so the output is non-empty.
        m.blocks_added.inc();

        let text = m.render();
        assert!(
            text.starts_with("# HELP"),
            "render() should start with '# HELP' but got:\n{text}"
        );
        assert!(
            text.contains("ipfrs_"),
            "render() should contain 'ipfrs_' prefix metrics"
        );
    }

    /// Observing a latency value must make histogram sum positive.
    #[test]
    fn test_histogram_observe() {
        let m = IpfrsMetrics::new().expect("should create metrics");
        m.block_fetch_latency.observe(0.042);

        let text = m.render();
        assert!(
            text.contains("ipfrs_block_fetch_latency_seconds_sum"),
            "render() must include histogram sum"
        );
        // The sum should be > 0, encoded as a non-zero float
        // Find the sum line and verify it is not "0"
        let sum_line = text
            .lines()
            .find(|l| l.contains("ipfrs_block_fetch_latency_seconds_sum"))
            .unwrap_or("");
        assert!(
            !sum_line.ends_with(" 0"),
            "histogram sum should be non-zero after observe(0.042)"
        );
    }

    /// Setting `storage_bytes_total` gauge must be reflected in `render()`.
    #[test]
    fn test_gauge_set() {
        let m = IpfrsMetrics::new().expect("should create metrics");
        m.storage_bytes_total.set(12345.0);

        let text = m.render();
        assert!(
            text.contains("ipfrs_storage_bytes_total"),
            "render() must include storage_bytes_total gauge"
        );
        assert!(
            text.contains("ipfrs_storage_bytes_total 12345"),
            "render() gauge value should be 12345 but got:\n{text}"
        );
    }
}
