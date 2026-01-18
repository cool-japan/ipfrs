//! Transport abstraction for multi-transport support
//!
//! This module provides a common interface for different transport protocols:
//! - QUIC (primary, high-performance)
//! - TCP (fallback, universal compatibility)
//! - WebSocket (gateway compatibility)
//! - WebTransport (future browser support)

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use std::fmt;
use std::net::SocketAddr;
use std::time::Duration;
use thiserror::Error;

/// Transport error types
#[derive(Error, Debug)]
pub enum TransportError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Connection closed: {0}")]
    ConnectionClosed(String),

    #[error("Send failed: {0}")]
    SendFailed(String),

    #[error("Receive failed: {0}")]
    ReceiveFailed(String),

    #[error("Timeout after {0:?}")]
    Timeout(Duration),

    #[error("Transport not available: {0}")]
    NotAvailable(String),

    #[error("Protocol error: {0}")]
    ProtocolError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Transport type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransportType {
    /// QUIC transport (primary)
    Quic,
    /// TCP transport (fallback)
    Tcp,
    /// WebSocket transport
    WebSocket,
    /// WebTransport (HTTP/3 based)
    WebTransport,
}

impl fmt::Display for TransportType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransportType::Quic => write!(f, "QUIC"),
            TransportType::Tcp => write!(f, "TCP"),
            TransportType::WebSocket => write!(f, "WebSocket"),
            TransportType::WebTransport => write!(f, "WebTransport"),
        }
    }
}

/// Transport capabilities for feature detection
#[derive(Debug, Clone, Copy)]
pub struct TransportCapabilities {
    /// Supports multiplexed streams
    pub multiplexing: bool,
    /// Supports 0-RTT connection establishment
    pub zero_rtt: bool,
    /// Supports connection migration
    pub migration: bool,
    /// Native encryption support
    pub encryption: bool,
    /// Supports unreliable datagrams
    pub datagrams: bool,
    /// Maximum message size (None = unlimited)
    pub max_message_size: Option<usize>,
}

impl TransportCapabilities {
    /// QUIC capabilities
    pub fn quic() -> Self {
        Self {
            multiplexing: true,
            zero_rtt: true,
            migration: true,
            encryption: true,
            datagrams: true,
            max_message_size: None,
        }
    }

    /// TCP capabilities
    pub fn tcp() -> Self {
        Self {
            multiplexing: false,
            zero_rtt: false,
            migration: false,
            encryption: false,
            datagrams: false,
            max_message_size: None,
        }
    }

    /// WebSocket capabilities
    pub fn websocket() -> Self {
        Self {
            multiplexing: false,
            zero_rtt: false,
            migration: false,
            encryption: true,
            datagrams: false,
            max_message_size: Some(16 * 1024 * 1024), // 16MB typical limit
        }
    }

    /// WebTransport capabilities
    pub fn webtransport() -> Self {
        Self {
            multiplexing: true,
            zero_rtt: false,
            migration: false,
            encryption: true,
            datagrams: true,
            max_message_size: None,
        }
    }
}

/// Connection metrics for monitoring
#[derive(Debug, Clone, Default)]
pub struct ConnectionMetrics {
    /// Bytes sent
    pub bytes_sent: u64,
    /// Bytes received
    pub bytes_received: u64,
    /// Round-trip time estimate
    pub rtt: Option<Duration>,
    /// Number of active streams
    pub active_streams: usize,
    /// Connection uptime
    pub uptime: Duration,
}

/// A transport connection handle
#[async_trait]
pub trait Connection: Send + Sync {
    /// Send data over the connection
    async fn send(&mut self, data: Bytes) -> Result<(), TransportError>;

    /// Receive data from the connection
    async fn receive(&mut self) -> Result<Bytes, TransportError>;

    /// Close the connection gracefully
    async fn close(&mut self) -> Result<(), TransportError>;

    /// Check if the connection is still alive
    fn is_alive(&self) -> bool;

    /// Get connection metrics
    fn metrics(&self) -> ConnectionMetrics;

    /// Get the remote address
    fn remote_addr(&self) -> SocketAddr;

    /// Get the transport type
    fn transport_type(&self) -> TransportType;
}

/// Transport trait for different protocols
#[async_trait]
pub trait Transport: Send + Sync {
    /// Get the transport type
    fn transport_type(&self) -> TransportType;

    /// Get transport capabilities
    fn capabilities(&self) -> TransportCapabilities;

    /// Check if this transport is available in the current environment
    fn is_available(&self) -> bool;

    /// Connect to a remote peer
    async fn connect(&self, addr: SocketAddr) -> Result<Box<dyn Connection>, TransportError>;

    /// Start listening for incoming connections
    async fn listen(&self, addr: SocketAddr) -> Result<(), TransportError>;

    /// Accept an incoming connection
    async fn accept(&self) -> Result<Box<dyn Connection>, TransportError>;

    /// Get transport-specific statistics
    fn stats(&self) -> TransportStats;
}

/// Statistics for a transport
#[derive(Debug, Clone, Default)]
pub struct TransportStats {
    /// Total connections established
    pub connections_established: u64,
    /// Total connections failed
    pub connections_failed: u64,
    /// Currently active connections
    pub active_connections: usize,
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Total bytes received
    pub bytes_received: u64,
    /// Average RTT
    pub avg_rtt: Option<Duration>,
}

/// Transport selection strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportSelectionStrategy {
    /// Prefer lowest latency
    LowestLatency,
    /// Prefer highest bandwidth
    HighestBandwidth,
    /// Prefer most features
    MostCapable,
    /// Use first available
    FirstAvailable,
    /// Prefer specific transport type
    PreferType(TransportType),
}

/// Transport selector for automatic transport selection
pub struct TransportSelector {
    transports: Vec<Box<dyn Transport>>,
    strategy: TransportSelectionStrategy,
}

impl TransportSelector {
    /// Create a new transport selector
    pub fn new(strategy: TransportSelectionStrategy) -> Self {
        Self {
            transports: Vec::new(),
            strategy,
        }
    }

    /// Register a transport
    pub fn register(&mut self, transport: Box<dyn Transport>) {
        self.transports.push(transport);
    }

    /// Select the best transport based on strategy
    pub fn select(&self) -> Option<&dyn Transport> {
        match self.strategy {
            TransportSelectionStrategy::FirstAvailable => self
                .transports
                .iter()
                .find(|t| t.is_available())
                .map(|b| b.as_ref()),
            TransportSelectionStrategy::PreferType(transport_type) => {
                // Try preferred type first
                self.transports
                    .iter()
                    .find(|t| t.transport_type() == transport_type && t.is_available())
                    .map(|b| b.as_ref())
                    .or_else(|| {
                        // Fallback to any available
                        self.transports
                            .iter()
                            .find(|t| t.is_available())
                            .map(|b| b.as_ref())
                    })
            }
            TransportSelectionStrategy::MostCapable => {
                // Score transports by capability count
                self.transports
                    .iter()
                    .filter(|t| t.is_available())
                    .max_by_key(|t| {
                        let cap = t.capabilities();
                        let mut score = 0;
                        if cap.multiplexing {
                            score += 10;
                        }
                        if cap.zero_rtt {
                            score += 5;
                        }
                        if cap.migration {
                            score += 3;
                        }
                        if cap.encryption {
                            score += 8;
                        }
                        if cap.datagrams {
                            score += 2;
                        }
                        score
                    })
                    .map(|b| b.as_ref())
            }
            TransportSelectionStrategy::LowestLatency => {
                // Use transport with lowest average RTT
                self.transports
                    .iter()
                    .filter(|t| t.is_available())
                    .min_by_key(|t| t.stats().avg_rtt.unwrap_or(Duration::MAX))
                    .map(|b| b.as_ref())
            }
            TransportSelectionStrategy::HighestBandwidth => {
                // Use transport type ranking (QUIC > TCP > WebSocket)
                let preference = [
                    TransportType::Quic,
                    TransportType::WebTransport,
                    TransportType::Tcp,
                    TransportType::WebSocket,
                ];
                preference.iter().find_map(|&preferred| {
                    self.transports
                        .iter()
                        .find(|t| t.transport_type() == preferred && t.is_available())
                        .map(|b| b.as_ref())
                })
            }
        }
    }

    /// Get all available transports
    pub fn available_transports(&self) -> Vec<TransportType> {
        self.transports
            .iter()
            .filter(|t| t.is_available())
            .map(|t| t.transport_type())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transport_type_display() {
        assert_eq!(TransportType::Quic.to_string(), "QUIC");
        assert_eq!(TransportType::Tcp.to_string(), "TCP");
        assert_eq!(TransportType::WebSocket.to_string(), "WebSocket");
        assert_eq!(TransportType::WebTransport.to_string(), "WebTransport");
    }

    #[test]
    fn test_capabilities() {
        let quic_cap = TransportCapabilities::quic();
        assert!(quic_cap.multiplexing);
        assert!(quic_cap.zero_rtt);
        assert!(quic_cap.encryption);

        let tcp_cap = TransportCapabilities::tcp();
        assert!(!tcp_cap.multiplexing);
        assert!(!tcp_cap.zero_rtt);
        assert!(!tcp_cap.encryption);

        let ws_cap = TransportCapabilities::websocket();
        assert!(!ws_cap.multiplexing);
        assert!(ws_cap.encryption);
        assert!(ws_cap.max_message_size.is_some());
    }

    #[test]
    fn test_transport_selector_empty() {
        let selector = TransportSelector::new(TransportSelectionStrategy::FirstAvailable);
        assert!(selector.available_transports().is_empty());
    }
}
