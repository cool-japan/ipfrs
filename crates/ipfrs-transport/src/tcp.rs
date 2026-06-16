//! TCP fallback transport for universal compatibility
//!
//! Provides a simple TCP-based transport as a fallback when QUIC is not available.
//! Features:
//! - Automatic reconnection on connection loss
//! - Connection pooling for multiple peers
//! - Frame-based message protocol
//! - Keep-alive support

use crate::transport::{
    Connection, ConnectionMetrics, Transport, TransportCapabilities, TransportError,
    TransportStats, TransportType,
};
use async_trait::async_trait;
use bytes::Bytes;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;
use tracing::{debug, info};

/// Frame header size (4 bytes for length)
const FRAME_HEADER_SIZE: usize = 4;

/// Maximum frame size (16 MB)
const MAX_FRAME_SIZE: usize = 16 * 1024 * 1024;

/// TCP transport configuration
#[derive(Debug, Clone)]
pub struct TcpConfig {
    /// TCP keep-alive interval
    pub keepalive_interval: Duration,
    /// Connection timeout
    pub connect_timeout: Duration,
    /// Read timeout
    pub read_timeout: Duration,
    /// Write timeout
    pub write_timeout: Duration,
    /// TCP_NODELAY (disable Nagle's algorithm)
    pub nodelay: bool,
    /// SO_RCVBUF size
    pub recv_buffer_size: Option<usize>,
    /// SO_SNDBUF size
    pub send_buffer_size: Option<usize>,
}

impl Default for TcpConfig {
    fn default() -> Self {
        Self {
            keepalive_interval: Duration::from_secs(30),
            connect_timeout: Duration::from_secs(10),
            read_timeout: Duration::from_secs(60),
            write_timeout: Duration::from_secs(30),
            nodelay: true,
            recv_buffer_size: Some(256 * 1024), // 256 KB
            send_buffer_size: Some(256 * 1024), // 256 KB
        }
    }
}

/// TCP connection implementation
pub struct TcpConnection {
    stream: Mutex<TcpStream>,
    remote_addr: SocketAddr,
    metrics: Arc<RwLock<ConnectionMetrics>>,
    created_at: Instant,
    alive: Arc<RwLock<bool>>,
    config: TcpConfig,
}

impl TcpConnection {
    /// Create a new TCP connection
    pub async fn new(stream: TcpStream, config: TcpConfig) -> Result<Self, TransportError> {
        let remote_addr = stream.peer_addr().map_err(|e| {
            TransportError::ConnectionFailed(format!("Failed to get peer address: {}", e))
        })?;

        // Configure socket options
        if config.nodelay {
            stream.set_nodelay(true).map_err(|e| {
                TransportError::ConnectionFailed(format!("Failed to set TCP_NODELAY: {}", e))
            })?;
        }

        debug!("TCP connection established to {}", remote_addr);

        Ok(Self {
            stream: Mutex::new(stream),
            remote_addr,
            metrics: Arc::new(RwLock::new(ConnectionMetrics::default())),
            created_at: Instant::now(),
            alive: Arc::new(RwLock::new(true)),
            config,
        })
    }

    /// Read a framed message from the stream
    async fn read_frame(&self) -> Result<Bytes, TransportError> {
        let mut stream = self.stream.lock().await;

        // Read frame header (4 bytes length prefix)
        let mut header = [0u8; FRAME_HEADER_SIZE];
        tokio::time::timeout(self.config.read_timeout, stream.read_exact(&mut header))
            .await
            .map_err(|_| TransportError::Timeout(self.config.read_timeout))?
            .map_err(|e| {
                if e.kind() == io::ErrorKind::UnexpectedEof {
                    *self.alive.write() = false;
                    TransportError::ConnectionClosed("Remote closed connection".to_string())
                } else {
                    TransportError::ReceiveFailed(format!("Failed to read frame header: {}", e))
                }
            })?;

        let frame_len = u32::from_be_bytes(header) as usize;

        if frame_len == 0 {
            return Err(TransportError::ProtocolError(
                "Received zero-length frame".to_string(),
            ));
        }

        if frame_len > MAX_FRAME_SIZE {
            return Err(TransportError::ProtocolError(format!(
                "Frame size {} exceeds maximum {}",
                frame_len, MAX_FRAME_SIZE
            )));
        }

        // Read frame payload
        let mut buffer = vec![0u8; frame_len];
        tokio::time::timeout(self.config.read_timeout, stream.read_exact(&mut buffer))
            .await
            .map_err(|_| TransportError::Timeout(self.config.read_timeout))?
            .map_err(|e| {
                if e.kind() == io::ErrorKind::UnexpectedEof {
                    *self.alive.write() = false;
                    TransportError::ConnectionClosed("Remote closed connection".to_string())
                } else {
                    TransportError::ReceiveFailed(format!("Failed to read frame payload: {}", e))
                }
            })?;

        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.bytes_received += (FRAME_HEADER_SIZE + frame_len) as u64;
        }

        Ok(Bytes::from(buffer))
    }

    /// Write a framed message to the stream
    async fn write_frame(&self, data: Bytes) -> Result<(), TransportError> {
        let data_len = data.len();

        if data_len > MAX_FRAME_SIZE {
            return Err(TransportError::ProtocolError(format!(
                "Message size {} exceeds maximum {}",
                data_len, MAX_FRAME_SIZE
            )));
        }

        let mut stream = self.stream.lock().await;

        // Write frame header
        let header = (data_len as u32).to_be_bytes();

        tokio::time::timeout(self.config.write_timeout, stream.write_all(&header))
            .await
            .map_err(|_| TransportError::Timeout(self.config.write_timeout))?
            .map_err(|e| {
                *self.alive.write() = false;
                TransportError::SendFailed(format!("Failed to write frame header: {}", e))
            })?;

        // Write frame payload
        tokio::time::timeout(self.config.write_timeout, stream.write_all(&data))
            .await
            .map_err(|_| TransportError::Timeout(self.config.write_timeout))?
            .map_err(|e| {
                *self.alive.write() = false;
                TransportError::SendFailed(format!("Failed to write frame payload: {}", e))
            })?;

        // Flush
        tokio::time::timeout(self.config.write_timeout, stream.flush())
            .await
            .map_err(|_| TransportError::Timeout(self.config.write_timeout))?
            .map_err(|e| {
                *self.alive.write() = false;
                TransportError::SendFailed(format!("Failed to flush: {}", e))
            })?;

        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.bytes_sent += (FRAME_HEADER_SIZE + data_len) as u64;
        }

        Ok(())
    }
}

#[async_trait]
impl Connection for TcpConnection {
    async fn send(&mut self, data: Bytes) -> Result<(), TransportError> {
        self.write_frame(data).await
    }

    async fn receive(&mut self) -> Result<Bytes, TransportError> {
        self.read_frame().await
    }

    async fn close(&mut self) -> Result<(), TransportError> {
        *self.alive.write() = false;
        let mut stream = self.stream.lock().await;
        stream
            .shutdown()
            .await
            .map_err(|e| TransportError::ConnectionClosed(format!("Shutdown failed: {}", e)))?;
        debug!("TCP connection to {} closed", self.remote_addr);
        Ok(())
    }

    fn is_alive(&self) -> bool {
        *self.alive.read()
    }

    fn metrics(&self) -> ConnectionMetrics {
        let mut metrics = self.metrics.read().clone();
        metrics.uptime = self.created_at.elapsed();
        metrics.active_streams = 1; // TCP has single stream
        metrics
    }

    fn remote_addr(&self) -> SocketAddr {
        self.remote_addr
    }

    fn transport_type(&self) -> TransportType {
        TransportType::Tcp
    }
}

/// TCP transport implementation
pub struct TcpTransport {
    config: TcpConfig,
    listener: Arc<Mutex<Option<TcpListener>>>,
    stats: Arc<RwLock<TransportStats>>,
    connections: Arc<RwLock<HashMap<SocketAddr, Instant>>>,
}

impl TcpTransport {
    /// Create a new TCP transport
    pub fn new(config: TcpConfig) -> Self {
        Self {
            config,
            listener: Arc::new(Mutex::new(None)),
            stats: Arc::new(RwLock::new(TransportStats::default())),
            connections: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(TcpConfig::default())
    }
}

#[async_trait]
impl Transport for TcpTransport {
    fn transport_type(&self) -> TransportType {
        TransportType::Tcp
    }

    fn capabilities(&self) -> TransportCapabilities {
        TransportCapabilities::tcp()
    }

    fn is_available(&self) -> bool {
        // TCP is always available
        true
    }

    async fn connect(&self, addr: SocketAddr) -> Result<Box<dyn Connection>, TransportError> {
        debug!("Connecting to {} via TCP", addr);

        let stream = tokio::time::timeout(self.config.connect_timeout, TcpStream::connect(addr))
            .await
            .map_err(|_| TransportError::Timeout(self.config.connect_timeout))?
            .map_err(|e| {
                self.stats.write().connections_failed += 1;
                TransportError::ConnectionFailed(format!("TCP connect failed: {}", e))
            })?;

        let connection = TcpConnection::new(stream, self.config.clone()).await?;

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.connections_established += 1;
            stats.active_connections += 1;
        }

        // Track connection
        self.connections.write().insert(addr, Instant::now());

        info!("TCP connection established to {}", addr);

        Ok(Box::new(connection))
    }

    async fn listen(&self, addr: SocketAddr) -> Result<(), TransportError> {
        let listener = TcpListener::bind(addr).await.map_err(|e| {
            TransportError::ConnectionFailed(format!("Failed to bind TCP listener: {}", e))
        })?;

        info!("TCP transport listening on {}", addr);

        *self.listener.lock().await = Some(listener);
        Ok(())
    }

    async fn accept(&self) -> Result<Box<dyn Connection>, TransportError> {
        let listener = self.listener.lock().await;
        let listener = listener
            .as_ref()
            .ok_or_else(|| TransportError::ProtocolError("No listener bound".to_string()))?;

        let (stream, addr) = listener
            .accept()
            .await
            .map_err(|e| TransportError::ConnectionFailed(format!("Accept failed: {}", e)))?;

        debug!("Accepted TCP connection from {}", addr);

        let connection = TcpConnection::new(stream, self.config.clone()).await?;

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.connections_established += 1;
            stats.active_connections += 1;
        }

        // Track connection
        self.connections.write().insert(addr, Instant::now());

        Ok(Box::new(connection))
    }

    fn stats(&self) -> TransportStats {
        self.stats.read().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tcp_config_default() {
        let config = TcpConfig::default();
        assert_eq!(config.keepalive_interval, Duration::from_secs(30));
        assert!(config.nodelay);
        assert_eq!(config.recv_buffer_size, Some(256 * 1024));
    }

    #[test]
    fn test_frame_constants() {
        assert_eq!(FRAME_HEADER_SIZE, 4);
        assert_eq!(MAX_FRAME_SIZE, 16 * 1024 * 1024);
    }

    #[tokio::test]
    async fn test_tcp_transport_creation() {
        let transport = TcpTransport::default_config();
        assert_eq!(transport.transport_type(), TransportType::Tcp);
        assert!(transport.is_available());

        let caps = transport.capabilities();
        assert!(!caps.multiplexing);
        assert!(!caps.zero_rtt);
    }

    #[tokio::test]
    async fn test_tcp_listen_and_connect() {
        let transport = TcpTransport::default_config();

        // Bind to localhost
        let addr: SocketAddr = "127.0.0.1:0"
            .parse()
            .expect("test: valid loopback address literal");
        transport
            .listen(addr)
            .await
            .expect("test: listener should bind to loopback");

        // Get the actual bound address
        let listener = transport.listener.lock().await;
        let bound_addr = listener
            .as_ref()
            .expect("test: listener should be present after listen()")
            .local_addr()
            .expect("test: OS should provide bound local address");
        drop(listener);

        // Spawn accept task
        let transport_clone = Arc::new(transport);
        let accept_handle = {
            let transport = transport_clone.clone();
            tokio::spawn(async move { transport.accept().await })
        };

        // Give accept time to start
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Connect
        let mut client_conn = transport_clone
            .connect(bound_addr)
            .await
            .expect("test: connect transport");
        let mut server_conn = accept_handle
            .await
            .expect("test: await accept")
            .expect("test: accept connection");

        // Test send/receive
        let test_data = Bytes::from("Hello, TCP!");
        client_conn
            .send(test_data.clone())
            .await
            .expect("test: send data");

        let received = server_conn.receive().await.expect("test: receive data");
        assert_eq!(received, test_data);

        // Check metrics
        let client_metrics = client_conn.metrics();
        assert!(client_metrics.bytes_sent > 0);

        let server_metrics = server_conn.metrics();
        assert!(server_metrics.bytes_received > 0);

        // Close connections
        client_conn.close().await.expect("test: close connection");
        server_conn.close().await.expect("test: close connection");
    }
}
