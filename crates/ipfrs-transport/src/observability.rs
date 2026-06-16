//! Observability module for structured logging and event tracking
//!
//! This module provides utilities for tracking transport layer events,
//! structured logging, and integration with observability platforms.
//!
//! # Example
//!
//! ```
//! use ipfrs_transport::observability::{EventLogger, LogLevel, TransportEvent};
//!
//! let mut logger = EventLogger::new();
//! logger.log_event(TransportEvent::BlockRequested {
//!     cid: "QmTest".to_string(),
//!     peer_id: "peer1".to_string(),
//!     priority: "High".to_string(),
//! });
//!
//! let events = logger.get_recent_events(10);
//! assert_eq!(events.len(), 1);
//! ```

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Log level for events
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    /// Debug level - detailed information for debugging
    Debug,
    /// Info level - general informational messages
    Info,
    /// Warn level - warning messages for potentially problematic situations
    Warn,
    /// Error level - error messages for failures
    Error,
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warn => write!(f, "WARN"),
            LogLevel::Error => write!(f, "ERROR"),
        }
    }
}

/// Transport layer events that can be logged
#[derive(Debug, Clone)]
pub enum TransportEvent {
    /// Block requested from a peer
    BlockRequested {
        cid: String,
        peer_id: String,
        priority: String,
    },
    /// Block received from a peer
    BlockReceived {
        cid: String,
        peer_id: String,
        bytes: usize,
        latency_ms: u64,
    },
    /// Block request failed
    BlockRequestFailed {
        cid: String,
        peer_id: String,
        error: String,
    },
    /// Peer connected
    PeerConnected {
        peer_id: String,
        transport_type: String,
        address: String,
    },
    /// Peer disconnected
    PeerDisconnected { peer_id: String, reason: String },
    /// Session started
    SessionStarted {
        session_id: String,
        block_count: usize,
    },
    /// Session completed
    SessionCompleted {
        session_id: String,
        duration_ms: u64,
        bytes_transferred: u64,
    },
    /// Circuit breaker opened
    CircuitBreakerOpened {
        peer_id: String,
        failure_count: usize,
    },
    /// Network partition detected
    PartitionDetected {
        peer_count: usize,
        suspected_peers: Vec<String>,
    },
    /// Network partition recovered
    PartitionRecovered { duration_ms: u64 },
    /// Custom event with arbitrary key-value pairs
    Custom {
        event_type: String,
        data: Vec<(String, String)>,
    },
}

impl std::fmt::Display for TransportEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransportEvent::BlockRequested {
                cid,
                peer_id,
                priority,
            } => {
                write!(
                    f,
                    "BlockRequested(cid={}, peer={}, priority={})",
                    cid, peer_id, priority
                )
            }
            TransportEvent::BlockReceived {
                cid,
                peer_id,
                bytes,
                latency_ms,
            } => {
                write!(
                    f,
                    "BlockReceived(cid={}, peer={}, bytes={}, latency={}ms)",
                    cid, peer_id, bytes, latency_ms
                )
            }
            TransportEvent::BlockRequestFailed {
                cid,
                peer_id,
                error,
            } => {
                write!(
                    f,
                    "BlockRequestFailed(cid={}, peer={}, error={})",
                    cid, peer_id, error
                )
            }
            TransportEvent::PeerConnected {
                peer_id,
                transport_type,
                address,
            } => {
                write!(
                    f,
                    "PeerConnected(peer={}, transport={}, addr={})",
                    peer_id, transport_type, address
                )
            }
            TransportEvent::PeerDisconnected { peer_id, reason } => {
                write!(f, "PeerDisconnected(peer={}, reason={})", peer_id, reason)
            }
            TransportEvent::SessionStarted {
                session_id,
                block_count,
            } => {
                write!(
                    f,
                    "SessionStarted(id={}, blocks={})",
                    session_id, block_count
                )
            }
            TransportEvent::SessionCompleted {
                session_id,
                duration_ms,
                bytes_transferred,
            } => {
                write!(
                    f,
                    "SessionCompleted(id={}, duration={}ms, bytes={})",
                    session_id, duration_ms, bytes_transferred
                )
            }
            TransportEvent::CircuitBreakerOpened {
                peer_id,
                failure_count,
            } => {
                write!(
                    f,
                    "CircuitBreakerOpened(peer={}, failures={})",
                    peer_id, failure_count
                )
            }
            TransportEvent::PartitionDetected {
                peer_count,
                suspected_peers,
            } => {
                write!(
                    f,
                    "PartitionDetected(peers={}, suspected={:?})",
                    peer_count, suspected_peers
                )
            }
            TransportEvent::PartitionRecovered { duration_ms } => {
                write!(f, "PartitionRecovered(duration={}ms)", duration_ms)
            }
            TransportEvent::Custom { event_type, data } => {
                write!(f, "{}(", event_type)?;
                for (i, (k, v)) in data.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}={}", k, v)?;
                }
                write!(f, ")")
            }
        }
    }
}

/// Log entry with timestamp and level
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// Timestamp in milliseconds since UNIX epoch
    pub timestamp_ms: u64,
    /// Log level
    pub level: LogLevel,
    /// Event that occurred
    pub event: TransportEvent,
}

impl LogEntry {
    /// Create a new log entry with current timestamp
    pub fn new(level: LogLevel, event: TransportEvent) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            timestamp_ms,
            level,
            event,
        }
    }
}

impl std::fmt::Display for LogEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {} {}", self.timestamp_ms, self.level, self.event)
    }
}

/// Configuration for event logger
#[derive(Debug, Clone)]
pub struct LoggerConfig {
    /// Maximum number of events to keep in memory
    pub max_events: usize,
    /// Minimum log level to record
    pub min_level: LogLevel,
    /// Whether to print events to stdout
    pub print_to_stdout: bool,
}

impl Default for LoggerConfig {
    fn default() -> Self {
        Self {
            max_events: 10000,
            min_level: LogLevel::Info,
            print_to_stdout: false,
        }
    }
}

/// Event logger for transport layer events
pub struct EventLogger {
    config: LoggerConfig,
    events: Arc<Mutex<VecDeque<LogEntry>>>,
}

impl EventLogger {
    /// Create a new event logger with default configuration
    pub fn new() -> Self {
        Self::with_config(LoggerConfig::default())
    }

    /// Create a new event logger with custom configuration
    pub fn with_config(config: LoggerConfig) -> Self {
        Self {
            config,
            events: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Log an event with specified level
    pub fn log(&mut self, level: LogLevel, event: TransportEvent) {
        if level < self.config.min_level {
            return;
        }

        let entry = LogEntry::new(level, event);

        if self.config.print_to_stdout {
            println!("{}", entry);
        }

        let mut events = self.events.lock().unwrap_or_else(|e| e.into_inner());
        events.push_back(entry);

        // Trim to max size
        while events.len() > self.config.max_events {
            events.pop_front();
        }
    }

    /// Log an event with Info level
    pub fn log_event(&mut self, event: TransportEvent) {
        self.log(LogLevel::Info, event);
    }

    /// Log a debug event
    pub fn debug(&mut self, event: TransportEvent) {
        self.log(LogLevel::Debug, event);
    }

    /// Log an info event
    pub fn info(&mut self, event: TransportEvent) {
        self.log(LogLevel::Info, event);
    }

    /// Log a warning event
    pub fn warn(&mut self, event: TransportEvent) {
        self.log(LogLevel::Warn, event);
    }

    /// Log an error event
    pub fn error(&mut self, event: TransportEvent) {
        self.log(LogLevel::Error, event);
    }

    /// Get recent events (most recent first)
    pub fn get_recent_events(&self, count: usize) -> Vec<LogEntry> {
        let events = self.events.lock().unwrap_or_else(|e| e.into_inner());
        events.iter().rev().take(count).cloned().collect()
    }

    /// Get all events matching a log level
    pub fn get_events_by_level(&self, level: LogLevel) -> Vec<LogEntry> {
        let events = self.events.lock().unwrap_or_else(|e| e.into_inner());
        events
            .iter()
            .filter(|e| e.level == level)
            .cloned()
            .collect()
    }

    /// Get events within a time range (milliseconds since UNIX epoch)
    pub fn get_events_by_time(&self, start_ms: u64, end_ms: u64) -> Vec<LogEntry> {
        let events = self.events.lock().unwrap_or_else(|e| e.into_inner());
        events
            .iter()
            .filter(|e| e.timestamp_ms >= start_ms && e.timestamp_ms <= end_ms)
            .cloned()
            .collect()
    }

    /// Clear all logged events
    pub fn clear(&mut self) {
        let mut events = self.events.lock().unwrap_or_else(|e| e.into_inner());
        events.clear();
    }

    /// Get total number of logged events
    pub fn event_count(&self) -> usize {
        let events = self.events.lock().unwrap_or_else(|e| e.into_inner());
        events.len()
    }

    /// Get configuration
    pub fn config(&self) -> &LoggerConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: LoggerConfig) {
        self.config = config;
    }
}

impl Default for EventLogger {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for EventLogger {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            events: Arc::clone(&self.events),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_logger_creation() {
        let logger = EventLogger::new();
        assert_eq!(logger.event_count(), 0);
    }

    #[test]
    fn test_log_event() {
        let mut logger = EventLogger::new();
        logger.log_event(TransportEvent::BlockRequested {
            cid: "QmTest".to_string(),
            peer_id: "peer1".to_string(),
            priority: "High".to_string(),
        });

        assert_eq!(logger.event_count(), 1);
        let events = logger.get_recent_events(1);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].level, LogLevel::Info);
    }

    #[test]
    fn test_log_levels() {
        let mut logger = EventLogger::new();

        logger.debug(TransportEvent::Custom {
            event_type: "test".to_string(),
            data: vec![],
        });
        logger.info(TransportEvent::Custom {
            event_type: "test".to_string(),
            data: vec![],
        });
        logger.warn(TransportEvent::Custom {
            event_type: "test".to_string(),
            data: vec![],
        });
        logger.error(TransportEvent::Custom {
            event_type: "test".to_string(),
            data: vec![],
        });

        // Debug events are filtered by default (min_level is Info)
        assert_eq!(logger.event_count(), 3);
    }

    #[test]
    fn test_min_level_filtering() {
        let config = LoggerConfig {
            max_events: 100,
            min_level: LogLevel::Warn,
            print_to_stdout: false,
        };
        let mut logger = EventLogger::with_config(config);

        logger.debug(TransportEvent::Custom {
            event_type: "debug".to_string(),
            data: vec![],
        });
        logger.info(TransportEvent::Custom {
            event_type: "info".to_string(),
            data: vec![],
        });
        logger.warn(TransportEvent::Custom {
            event_type: "warn".to_string(),
            data: vec![],
        });
        logger.error(TransportEvent::Custom {
            event_type: "error".to_string(),
            data: vec![],
        });

        // Only warn and error should be logged
        assert_eq!(logger.event_count(), 2);
    }

    #[test]
    fn test_max_events_limit() {
        let config = LoggerConfig {
            max_events: 5,
            min_level: LogLevel::Debug,
            print_to_stdout: false,
        };
        let mut logger = EventLogger::with_config(config);

        for i in 0..10 {
            logger.info(TransportEvent::Custom {
                event_type: format!("event{}", i),
                data: vec![],
            });
        }

        assert_eq!(logger.event_count(), 5);
        let events = logger.get_recent_events(10);
        assert_eq!(events.len(), 5);
    }

    #[test]
    fn test_get_events_by_level() {
        let mut logger = EventLogger::with_config(LoggerConfig {
            min_level: LogLevel::Debug,
            ..Default::default()
        });

        logger.info(TransportEvent::Custom {
            event_type: "info1".to_string(),
            data: vec![],
        });
        logger.error(TransportEvent::Custom {
            event_type: "error1".to_string(),
            data: vec![],
        });
        logger.info(TransportEvent::Custom {
            event_type: "info2".to_string(),
            data: vec![],
        });

        let info_events = logger.get_events_by_level(LogLevel::Info);
        assert_eq!(info_events.len(), 2);

        let error_events = logger.get_events_by_level(LogLevel::Error);
        assert_eq!(error_events.len(), 1);
    }

    #[test]
    fn test_clear_events() {
        let mut logger = EventLogger::new();

        logger.info(TransportEvent::Custom {
            event_type: "test".to_string(),
            data: vec![],
        });
        assert_eq!(logger.event_count(), 1);

        logger.clear();
        assert_eq!(logger.event_count(), 0);
    }

    #[test]
    fn test_event_display() {
        let event = TransportEvent::BlockRequested {
            cid: "QmTest".to_string(),
            peer_id: "peer1".to_string(),
            priority: "High".to_string(),
        };

        let display = format!("{}", event);
        assert!(display.contains("BlockRequested"));
        assert!(display.contains("QmTest"));
        assert!(display.contains("peer1"));
    }

    #[test]
    fn test_log_entry_display() {
        let entry = LogEntry::new(
            LogLevel::Info,
            TransportEvent::Custom {
                event_type: "test".to_string(),
                data: vec![("key".to_string(), "value".to_string())],
            },
        );

        let display = format!("{}", entry);
        assert!(display.contains("INFO"));
        assert!(display.contains("test"));
    }

    #[test]
    fn test_clone_logger() {
        let mut logger1 = EventLogger::new();
        logger1.info(TransportEvent::Custom {
            event_type: "test".to_string(),
            data: vec![],
        });

        let logger2 = logger1.clone();
        assert_eq!(logger2.event_count(), 1);
    }

    #[test]
    fn test_update_config() {
        let mut logger = EventLogger::new();

        let new_config = LoggerConfig {
            max_events: 50,
            min_level: LogLevel::Debug,
            print_to_stdout: true,
        };

        logger.update_config(new_config.clone());
        assert_eq!(logger.config().max_events, 50);
        assert_eq!(logger.config().min_level, LogLevel::Debug);
    }

    #[test]
    fn test_all_event_types() {
        let mut logger = EventLogger::new();

        logger.info(TransportEvent::BlockRequested {
            cid: "QmTest1".to_string(),
            peer_id: "peer1".to_string(),
            priority: "High".to_string(),
        });

        logger.info(TransportEvent::BlockReceived {
            cid: "QmTest2".to_string(),
            peer_id: "peer2".to_string(),
            bytes: 1024,
            latency_ms: 50,
        });

        logger.info(TransportEvent::PeerConnected {
            peer_id: "peer3".to_string(),
            transport_type: "QUIC".to_string(),
            address: "127.0.0.1:4001".to_string(),
        });

        logger.info(TransportEvent::SessionStarted {
            session_id: "session1".to_string(),
            block_count: 100,
        });

        assert_eq!(logger.event_count(), 4);
    }
}
