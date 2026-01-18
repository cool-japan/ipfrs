//! Distributed tracing setup with OpenTelemetry
//!
//! This module provides distributed tracing configuration using OpenTelemetry,
//! enabling trace collection across distributed IPFRS nodes.

use opentelemetry::trace::TracerProvider as _;
use opentelemetry::KeyValue;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::trace::SdkTracerProvider;
use opentelemetry_sdk::Resource;
use std::error::Error;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

/// Tracing configuration
#[derive(Debug, Clone)]
pub struct TracingConfig {
    /// Service name for traces
    pub service_name: String,
    /// Service version
    pub service_version: String,
    /// OTLP endpoint (e.g., "http://localhost:4317")
    pub otlp_endpoint: Option<String>,
    /// Enable JSON logging
    pub json_logs: bool,
    /// Log level filter
    pub log_level: String,
}

impl TracingConfig {
    /// Create a new tracing configuration
    ///
    /// # Example
    /// ```rust
    /// use ipfrs::tracing_setup::TracingConfig;
    ///
    /// let config = TracingConfig::new("ipfrs-node".to_string());
    /// ```
    pub fn new(service_name: String) -> Self {
        Self {
            service_name,
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            otlp_endpoint: None,
            json_logs: false,
            log_level: "info".to_string(),
        }
    }

    /// Set OTLP endpoint for trace export
    pub fn with_otlp_endpoint(mut self, endpoint: String) -> Self {
        self.otlp_endpoint = Some(endpoint);
        self
    }

    /// Enable JSON logging
    pub fn with_json_logs(mut self, enabled: bool) -> Self {
        self.json_logs = enabled;
        self
    }

    /// Set log level
    pub fn with_log_level(mut self, level: String) -> Self {
        self.log_level = level;
        self
    }
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self::new("ipfrs".to_string())
    }
}

/// Initialize tracing with OpenTelemetry
///
/// Sets up distributed tracing with OTLP export if configured,
/// along with structured logging.
///
/// # Arguments
/// * `config` - Tracing configuration
///
/// # Returns
/// A guard that must be kept alive for tracing to work
///
/// # Example
/// ```rust,no_run
/// use ipfrs::tracing_setup::{TracingConfig, init_tracing};
///
/// #[tokio::main]
/// async fn main() {
///     let config = TracingConfig::new("my-service".to_string())
///         .with_otlp_endpoint("http://localhost:4317".to_string());
///
///     let _guard = init_tracing(config).expect("Failed to initialize tracing");
///
///     // Your application code
/// }
/// ```
pub fn init_tracing(config: TracingConfig) -> Result<TracingGuard, Box<dyn Error>> {
    // Create resource with service information
    let resource = Resource::builder()
        .with_attributes(vec![
            KeyValue::new("service.name", config.service_name.clone()),
            KeyValue::new("service.version", config.service_version.clone()),
        ])
        .build();

    // Set up tracing layer
    let tracer_provider = if let Some(endpoint) = &config.otlp_endpoint {
        // Configure OTLP exporter with batch processor
        // Note: opentelemetry 0.31 changed the API significantly
        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_http()
            .with_endpoint(endpoint.clone())
            .build()?;

        let batch_processor =
            opentelemetry_sdk::trace::BatchSpanProcessor::builder(exporter).build();

        SdkTracerProvider::builder()
            .with_resource(resource.clone())
            .with_span_processor(batch_processor)
            .build()
    } else {
        // No OTLP export, just local tracing
        SdkTracerProvider::builder().with_resource(resource).build()
    };

    // Create OpenTelemetry layer
    let telemetry_layer =
        tracing_opentelemetry::layer().with_tracer(tracer_provider.tracer("ipfrs"));

    // Create env filter
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.log_level));

    // Set up subscriber with layers
    if config.json_logs {
        // JSON formatting for structured logs
        let fmt_layer = tracing_subscriber::fmt::layer()
            .json()
            .with_current_span(true)
            .with_span_list(true);

        tracing_subscriber::registry()
            .with(env_filter)
            .with(telemetry_layer)
            .with(fmt_layer)
            .init();
    } else {
        // Human-readable formatting
        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_target(true)
            .with_thread_ids(true)
            .with_line_number(true);

        tracing_subscriber::registry()
            .with(env_filter)
            .with(telemetry_layer)
            .with(fmt_layer)
            .init();
    }

    tracing::info!(
        service_name = %config.service_name,
        service_version = %config.service_version,
        otlp_endpoint = ?config.otlp_endpoint,
        "Distributed tracing initialized"
    );

    Ok(TracingGuard {
        tracer_provider: Some(tracer_provider),
    })
}

/// Guard for tracing that ensures proper shutdown
///
/// The tracer provider is shut down when this guard is dropped.
pub struct TracingGuard {
    tracer_provider: Option<SdkTracerProvider>,
}

impl Drop for TracingGuard {
    fn drop(&mut self) {
        if let Some(provider) = self.tracer_provider.take() {
            // SdkTracerProvider::shutdown() now uses a different error type
            let _ = provider.shutdown();
        }
    }
}

/// Trace span attributes for IPFRS operations
pub mod attributes {
    use opentelemetry::KeyValue;

    /// Create block operation attributes
    pub fn block_op(operation: &str, cid: &str) -> Vec<KeyValue> {
        vec![
            KeyValue::new("ipfrs.operation", operation.to_string()),
            KeyValue::new("ipfrs.block.cid", cid.to_string()),
        ]
    }

    /// Create semantic search attributes
    pub fn semantic_search(k: usize, results: usize) -> Vec<KeyValue> {
        vec![
            KeyValue::new("ipfrs.semantic.k", k as i64),
            KeyValue::new("ipfrs.semantic.results", results as i64),
        ]
    }

    /// Create logic operation attributes
    pub fn logic_op(operation: &str, predicate: &str) -> Vec<KeyValue> {
        vec![
            KeyValue::new("ipfrs.logic.operation", operation.to_string()),
            KeyValue::new("ipfrs.logic.predicate", predicate.to_string()),
        ]
    }

    /// Create network operation attributes
    pub fn network_op(operation: &str, peer_count: usize) -> Vec<KeyValue> {
        vec![
            KeyValue::new("ipfrs.network.operation", operation.to_string()),
            KeyValue::new("ipfrs.network.peers", peer_count as i64),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracing_config_creation() {
        let config = TracingConfig::new("test-service".to_string());
        assert_eq!(config.service_name, "test-service");
        assert_eq!(config.service_version, env!("CARGO_PKG_VERSION"));
        assert_eq!(config.otlp_endpoint, None);
        assert!(!config.json_logs);
        assert_eq!(config.log_level, "info");
    }

    #[test]
    fn test_tracing_config_builder() {
        let config = TracingConfig::new("test-service".to_string())
            .with_otlp_endpoint("http://localhost:4317".to_string())
            .with_json_logs(true)
            .with_log_level("debug".to_string());

        assert_eq!(config.service_name, "test-service");
        assert_eq!(
            config.otlp_endpoint,
            Some("http://localhost:4317".to_string())
        );
        assert!(config.json_logs);
        assert_eq!(config.log_level, "debug");
    }

    #[test]
    fn test_tracing_config_default() {
        let config = TracingConfig::default();
        assert_eq!(config.service_name, "ipfrs");
        assert_eq!(config.service_version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn test_block_op_attributes() {
        let attrs = attributes::block_op("put", "QmTest123");
        assert_eq!(attrs.len(), 2);
    }

    #[test]
    fn test_semantic_search_attributes() {
        let attrs = attributes::semantic_search(10, 8);
        assert_eq!(attrs.len(), 2);
    }

    #[test]
    fn test_logic_op_attributes() {
        let attrs = attributes::logic_op("infer", "parent(X, Y)");
        assert_eq!(attrs.len(), 2);
    }

    #[test]
    fn test_network_op_attributes() {
        let attrs = attributes::network_op("connect", 5);
        assert_eq!(attrs.len(), 2);
    }
}
