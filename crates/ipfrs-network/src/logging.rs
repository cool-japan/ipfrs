//! Structured logging and tracing support
//!
//! Provides enhanced logging capabilities with:
//! - Structured event logging
//! - Tracing spans for operations
//! - Context propagation
//! - Log filtering and formatting

use tracing::{span, Level, Span};

/// Network event types for structured logging
#[derive(Debug, Clone, Copy)]
pub enum NetworkEventType {
    /// Connection establishment
    ConnectionEstablished,
    /// Connection closed
    ConnectionClosed,
    /// Connection failed
    ConnectionFailed,
    /// DHT query started
    DhtQuery,
    /// DHT query completed
    DhtQueryComplete,
    /// Peer discovered
    PeerDiscovered,
    /// Content announced
    ContentAnnounced,
    /// Content found
    ContentFound,
    /// Protocol negotiation
    ProtocolNegotiation,
}

impl NetworkEventType {
    /// Get the event type name as a string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ConnectionEstablished => "connection_established",
            Self::ConnectionClosed => "connection_closed",
            Self::ConnectionFailed => "connection_failed",
            Self::DhtQuery => "dht_query",
            Self::DhtQueryComplete => "dht_query_complete",
            Self::PeerDiscovered => "peer_discovered",
            Self::ContentAnnounced => "content_announced",
            Self::ContentFound => "content_found",
            Self::ProtocolNegotiation => "protocol_negotiation",
        }
    }
}

/// Create a span for a network operation
pub fn network_span(operation: &str) -> Span {
    span!(Level::INFO, "network_operation", operation = operation)
}

/// Create a span for a DHT operation
pub fn dht_span(operation: &str, key: Option<&str>) -> Span {
    if let Some(k) = key {
        span!(Level::INFO, "dht_operation", operation = operation, key = k)
    } else {
        span!(Level::INFO, "dht_operation", operation = operation)
    }
}

/// Create a span for a connection operation
pub fn connection_span(peer_id: &str, direction: &str) -> Span {
    span!(
        Level::INFO,
        "connection",
        peer_id = peer_id,
        direction = direction
    )
}

/// Log a structured network event
#[macro_export]
macro_rules! log_network_event {
    ($event_type:expr, $($key:ident = $value:expr),* $(,)?) => {
        tracing::info!(
            event_type = $event_type.as_str(),
            $($key = tracing::field::debug(&$value)),*
        );
    };
}

/// Log a network error with context
#[macro_export]
macro_rules! log_network_error {
    ($message:expr, $($key:ident = $value:expr),* $(,)?) => {
        tracing::error!(
            message = $message,
            $($key = tracing::field::debug(&$value)),*
        );
    };
}

/// Log a network warning with context
#[macro_export]
macro_rules! log_network_warn {
    ($message:expr, $($key:ident = $value:expr),* $(,)?) => {
        tracing::warn!(
            message = $message,
            $($key = tracing::field::debug(&$value)),*
        );
    };
}

/// Logging configuration
#[derive(Debug, Clone)]
pub struct LoggingConfig {
    /// Log level
    pub level: LogLevel,
    /// Enable JSON formatting
    pub json_format: bool,
    /// Enable timestamp
    pub with_timestamp: bool,
    /// Enable target (module path)
    pub with_target: bool,
    /// Enable thread ID
    pub with_thread_id: bool,
    /// Enable span information
    pub with_spans: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            json_format: false,
            with_timestamp: true,
            with_target: true,
            with_thread_id: false,
            with_spans: true,
        }
    }
}

/// Log level configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// Trace level (most verbose)
    Trace,
    /// Debug level
    Debug,
    /// Info level
    Info,
    /// Warn level
    Warn,
    /// Error level
    Error,
}

impl LogLevel {
    /// Convert to tracing Level
    pub fn to_tracing_level(&self) -> Level {
        match self {
            Self::Trace => Level::TRACE,
            Self::Debug => Level::DEBUG,
            Self::Info => Level::INFO,
            Self::Warn => Level::WARN,
            Self::Error => Level::ERROR,
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "trace" => Some(Self::Trace),
            "debug" => Some(Self::Debug),
            "info" => Some(Self::Info),
            "warn" | "warning" => Some(Self::Warn),
            "error" => Some(Self::Error),
            _ => None,
        }
    }
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Trace => write!(f, "TRACE"),
            Self::Debug => write!(f, "DEBUG"),
            Self::Info => write!(f, "INFO"),
            Self::Warn => write!(f, "WARN"),
            Self::Error => write!(f, "ERROR"),
        }
    }
}

/// Network operation context for tracing
pub struct OperationContext {
    span: Span,
}

impl OperationContext {
    /// Create a new operation context
    pub fn new(name: &str) -> Self {
        Self {
            span: span!(Level::INFO, "operation", name = name),
        }
    }

    /// Enter the context
    pub fn enter(&self) -> tracing::span::Entered<'_> {
        self.span.enter()
    }

    /// Add a field to the context
    pub fn record<T: std::fmt::Debug>(&self, field: &str, value: T) {
        self.span.record(field, tracing::field::debug(&value));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_conversion() {
        assert_eq!(LogLevel::Info.to_tracing_level(), Level::INFO);
        assert_eq!(LogLevel::Debug.to_tracing_level(), Level::DEBUG);
        assert_eq!(LogLevel::Error.to_tracing_level(), Level::ERROR);
    }

    #[test]
    fn test_log_level_from_string() {
        assert_eq!(LogLevel::parse("info"), Some(LogLevel::Info));
        assert_eq!(LogLevel::parse("INFO"), Some(LogLevel::Info));
        assert_eq!(LogLevel::parse("debug"), Some(LogLevel::Debug));
        assert_eq!(LogLevel::parse("warn"), Some(LogLevel::Warn));
        assert_eq!(LogLevel::parse("warning"), Some(LogLevel::Warn));
        assert_eq!(LogLevel::parse("error"), Some(LogLevel::Error));
        assert_eq!(LogLevel::parse("trace"), Some(LogLevel::Trace));
        assert_eq!(LogLevel::parse("invalid"), None);
    }

    #[test]
    fn test_log_level_display() {
        assert_eq!(format!("{}", LogLevel::Info), "INFO");
        assert_eq!(format!("{}", LogLevel::Debug), "DEBUG");
        assert_eq!(format!("{}", LogLevel::Error), "ERROR");
    }

    #[test]
    fn test_event_type_str() {
        assert_eq!(
            NetworkEventType::ConnectionEstablished.as_str(),
            "connection_established"
        );
        assert_eq!(NetworkEventType::DhtQuery.as_str(), "dht_query");
    }

    #[test]
    fn test_network_span_creation() {
        let _span = network_span("test_operation");
        // Span creation succeeds (may be disabled without subscriber)
    }

    #[test]
    fn test_dht_span_creation() {
        let _span = dht_span("query", Some("test_key"));
        let _span_no_key = dht_span("bootstrap", None);
        // Span creation succeeds (may be disabled without subscriber)
    }

    #[test]
    fn test_connection_span_creation() {
        let _span = connection_span("12D3KooTest", "outbound");
        // Span creation succeeds (may be disabled without subscriber)
    }

    #[test]
    fn test_operation_context() {
        let ctx = OperationContext::new("test_op");
        let _guard = ctx.enter();
        ctx.record("status", "success");
    }

    #[test]
    fn test_logging_config_default() {
        let config = LoggingConfig::default();
        assert_eq!(config.level, LogLevel::Info);
        assert!(config.with_timestamp);
        assert!(config.with_target);
        assert!(!config.json_format);
    }
}
