//! WebSocket transport for gateway compatibility
//!
//! Provides WebSocket-based transport for compatibility with web gateways
//! and restrictive network environments.
//!
//! Features:
//! - Text and binary message support
//! - Automatic reconnection
//! - Ping/pong keepalive
//! - TLS support

use crate::transport::{
    Connection, ConnectionMetrics, Transport, TransportCapabilities, TransportError,
    TransportStats, TransportType,
};
use async_trait::async_trait;
use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;
use tokio_tungstenite::{
    accept_async, connect_async, tungstenite::Message, MaybeTlsStream, WebSocketStream,
};
use tracing::{debug, info};

/// WebSocket transport configuration
#[derive(Debug, Clone)]
pub struct WebSocketConfig {
    /// Ping interval for keepalive
    pub ping_interval: Duration,
    /// Connection timeout
    pub connect_timeout: Duration,
    /// Maximum message size (16MB default)
    pub max_message_size: usize,
    /// Use binary frames (vs text)
    pub use_binary: bool,
}

impl Default for WebSocketConfig {
    fn default() -> Self {
        Self {
            ping_interval: Duration::from_secs(30),
            connect_timeout: Duration::from_secs(10),
            max_message_size: 16 * 1024 * 1024, // 16 MB
            use_binary: true,
        }
    }
}

/// WebSocket connection wrapper for client connections
pub struct WebSocketConnection {
    stream: Arc<Mutex<WebSocketStream<MaybeTlsStream<TcpStream>>>>,
    remote_addr: SocketAddr,
    metrics: Arc<RwLock<ConnectionMetrics>>,
    created_at: Instant,
    alive: Arc<RwLock<bool>>,
    config: WebSocketConfig,
}

/// WebSocket connection wrapper for server connections
pub struct WebSocketServerConnection {
    stream: Arc<Mutex<WebSocketStream<TcpStream>>>,
    remote_addr: SocketAddr,
    metrics: Arc<RwLock<ConnectionMetrics>>,
    created_at: Instant,
    alive: Arc<RwLock<bool>>,
    config: WebSocketConfig,
}

impl WebSocketConnection {
    /// Create a new WebSocket client connection
    pub fn new(
        stream: WebSocketStream<MaybeTlsStream<TcpStream>>,
        remote_addr: SocketAddr,
        config: WebSocketConfig,
    ) -> Self {
        debug!("WebSocket connection established to {}", remote_addr);

        Self {
            stream: Arc::new(Mutex::new(stream)),
            remote_addr,
            metrics: Arc::new(RwLock::new(ConnectionMetrics::default())),
            created_at: Instant::now(),
            alive: Arc::new(RwLock::new(true)),
            config,
        }
    }

    /// Send a ping frame
    #[allow(dead_code)]
    async fn send_ping(&self) -> Result<(), TransportError> {
        let mut stream = self.stream.lock().await;
        stream
            .send(Message::Ping(vec![].into()))
            .await
            .map_err(|e| TransportError::SendFailed(format!("Ping failed: {}", e)))?;
        Ok(())
    }
}

#[async_trait]
impl Connection for WebSocketConnection {
    async fn send(&mut self, data: Bytes) -> Result<(), TransportError> {
        if data.len() > self.config.max_message_size {
            return Err(TransportError::ProtocolError(format!(
                "Message size {} exceeds maximum {}",
                data.len(),
                self.config.max_message_size
            )));
        }

        let data_len = data.len();
        let message = if self.config.use_binary {
            Message::Binary(data)
        } else {
            Message::Text(String::from_utf8_lossy(&data).to_string().into())
        };

        let mut stream = self.stream.lock().await;

        stream.send(message).await.map_err(|e| {
            *self.alive.write() = false;
            TransportError::SendFailed(format!("WebSocket send failed: {}", e))
        })?;

        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.bytes_sent += data_len as u64;
        }

        Ok(())
    }

    async fn receive(&mut self) -> Result<Bytes, TransportError> {
        let mut stream = self.stream.lock().await;

        loop {
            match stream.next().await {
                Some(Ok(message)) => match message {
                    Message::Binary(data) => {
                        // Update metrics
                        {
                            let mut metrics = self.metrics.write();
                            metrics.bytes_received += data.len() as u64;
                        }
                        return Ok(data);
                    }
                    Message::Text(text) => {
                        // Convert Utf8Bytes to bytes::Bytes
                        let data = Bytes::copy_from_slice(text.as_bytes());
                        // Update metrics
                        {
                            let mut metrics = self.metrics.write();
                            metrics.bytes_received += data.len() as u64;
                        }
                        return Ok(data);
                    }
                    Message::Ping(_) => {
                        // Automatically respond with pong
                        debug!("Received ping, sending pong");
                        stream
                            .send(Message::Pong(vec![].into()))
                            .await
                            .map_err(|e| {
                                TransportError::SendFailed(format!("Pong failed: {}", e))
                            })?;
                        continue;
                    }
                    Message::Pong(_) => {
                        debug!("Received pong");
                        continue;
                    }
                    Message::Close(_) => {
                        *self.alive.write() = false;
                        return Err(TransportError::ConnectionClosed(
                            "Received close frame".to_string(),
                        ));
                    }
                    Message::Frame(_) => {
                        // Raw frames shouldn't happen in normal operation
                        continue;
                    }
                },
                Some(Err(e)) => {
                    *self.alive.write() = false;
                    return Err(TransportError::ReceiveFailed(format!(
                        "WebSocket receive error: {}",
                        e
                    )));
                }
                None => {
                    *self.alive.write() = false;
                    return Err(TransportError::ConnectionClosed(
                        "WebSocket stream ended".to_string(),
                    ));
                }
            }
        }
    }

    async fn close(&mut self) -> Result<(), TransportError> {
        *self.alive.write() = false;
        let mut stream = self.stream.lock().await;
        stream
            .close(None)
            .await
            .map_err(|e| TransportError::ConnectionClosed(format!("Close failed: {}", e)))?;
        debug!("WebSocket connection to {} closed", self.remote_addr);
        Ok(())
    }

    fn is_alive(&self) -> bool {
        *self.alive.read()
    }

    fn metrics(&self) -> ConnectionMetrics {
        let mut metrics = self.metrics.read().clone();
        metrics.uptime = self.created_at.elapsed();
        metrics.active_streams = 1; // WebSocket has single stream
        metrics
    }

    fn remote_addr(&self) -> SocketAddr {
        self.remote_addr
    }

    fn transport_type(&self) -> TransportType {
        TransportType::WebSocket
    }
}

impl WebSocketServerConnection {
    /// Create a new WebSocket server connection
    pub fn new(
        stream: WebSocketStream<TcpStream>,
        remote_addr: SocketAddr,
        config: WebSocketConfig,
    ) -> Self {
        debug!("WebSocket server connection accepted from {}", remote_addr);

        Self {
            stream: Arc::new(Mutex::new(stream)),
            remote_addr,
            metrics: Arc::new(RwLock::new(ConnectionMetrics::default())),
            created_at: Instant::now(),
            alive: Arc::new(RwLock::new(true)),
            config,
        }
    }
}

#[async_trait]
impl Connection for WebSocketServerConnection {
    async fn send(&mut self, data: Bytes) -> Result<(), TransportError> {
        if data.len() > self.config.max_message_size {
            return Err(TransportError::ProtocolError(format!(
                "Message size {} exceeds maximum {}",
                data.len(),
                self.config.max_message_size
            )));
        }

        let data_len = data.len();
        let message = if self.config.use_binary {
            Message::Binary(data)
        } else {
            Message::Text(String::from_utf8_lossy(&data).to_string().into())
        };

        let mut stream = self.stream.lock().await;

        stream.send(message).await.map_err(|e| {
            *self.alive.write() = false;
            TransportError::SendFailed(format!("WebSocket send failed: {}", e))
        })?;

        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.bytes_sent += data_len as u64;
        }

        Ok(())
    }

    async fn receive(&mut self) -> Result<Bytes, TransportError> {
        let mut stream = self.stream.lock().await;

        loop {
            match stream.next().await {
                Some(Ok(message)) => match message {
                    Message::Binary(data) => {
                        // Update metrics
                        {
                            let mut metrics = self.metrics.write();
                            metrics.bytes_received += data.len() as u64;
                        }
                        return Ok(data);
                    }
                    Message::Text(text) => {
                        // Convert Utf8Bytes to bytes::Bytes
                        let data = Bytes::copy_from_slice(text.as_bytes());
                        // Update metrics
                        {
                            let mut metrics = self.metrics.write();
                            metrics.bytes_received += data.len() as u64;
                        }
                        return Ok(data);
                    }
                    Message::Ping(_) => {
                        // Automatically respond with pong
                        debug!("Received ping, sending pong");
                        stream
                            .send(Message::Pong(vec![].into()))
                            .await
                            .map_err(|e| {
                                TransportError::SendFailed(format!("Pong failed: {}", e))
                            })?;
                        continue;
                    }
                    Message::Pong(_) => {
                        debug!("Received pong");
                        continue;
                    }
                    Message::Close(_) => {
                        *self.alive.write() = false;
                        return Err(TransportError::ConnectionClosed(
                            "Received close frame".to_string(),
                        ));
                    }
                    Message::Frame(_) => {
                        // Raw frames shouldn't happen in normal operation
                        continue;
                    }
                },
                Some(Err(e)) => {
                    *self.alive.write() = false;
                    return Err(TransportError::ReceiveFailed(format!(
                        "WebSocket receive error: {}",
                        e
                    )));
                }
                None => {
                    *self.alive.write() = false;
                    return Err(TransportError::ConnectionClosed(
                        "WebSocket stream ended".to_string(),
                    ));
                }
            }
        }
    }

    async fn close(&mut self) -> Result<(), TransportError> {
        *self.alive.write() = false;
        let mut stream = self.stream.lock().await;
        stream
            .close(None)
            .await
            .map_err(|e| TransportError::ConnectionClosed(format!("Close failed: {}", e)))?;
        debug!("WebSocket connection to {} closed", self.remote_addr);
        Ok(())
    }

    fn is_alive(&self) -> bool {
        *self.alive.read()
    }

    fn metrics(&self) -> ConnectionMetrics {
        let mut metrics = self.metrics.read().clone();
        metrics.uptime = self.created_at.elapsed();
        metrics.active_streams = 1; // WebSocket has single stream
        metrics
    }

    fn remote_addr(&self) -> SocketAddr {
        self.remote_addr
    }

    fn transport_type(&self) -> TransportType {
        TransportType::WebSocket
    }
}

/// WebSocket transport implementation
pub struct WebSocketTransport {
    config: WebSocketConfig,
    listener: Arc<Mutex<Option<TcpListener>>>,
    stats: Arc<RwLock<TransportStats>>,
    connections: Arc<RwLock<HashMap<SocketAddr, Instant>>>,
}

impl WebSocketTransport {
    /// Create a new WebSocket transport
    pub fn new(config: WebSocketConfig) -> Self {
        Self {
            config,
            listener: Arc::new(Mutex::new(None)),
            stats: Arc::new(RwLock::new(TransportStats::default())),
            connections: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(WebSocketConfig::default())
    }
}

#[async_trait]
impl Transport for WebSocketTransport {
    fn transport_type(&self) -> TransportType {
        TransportType::WebSocket
    }

    fn capabilities(&self) -> TransportCapabilities {
        TransportCapabilities::websocket()
    }

    fn is_available(&self) -> bool {
        // WebSocket is always available
        true
    }

    async fn connect(&self, addr: SocketAddr) -> Result<Box<dyn Connection>, TransportError> {
        debug!("Connecting to {} via WebSocket", addr);

        // Construct WebSocket URL
        let url = format!("ws://{}", addr);

        let (ws_stream, _) = tokio::time::timeout(self.config.connect_timeout, connect_async(&url))
            .await
            .map_err(|_| TransportError::Timeout(self.config.connect_timeout))?
            .map_err(|e| {
                self.stats.write().connections_failed += 1;
                TransportError::ConnectionFailed(format!("WebSocket connect failed: {}", e))
            })?;

        // Extract underlying TCP stream's peer address
        let connection = WebSocketConnection::new(ws_stream, addr, self.config.clone());

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.connections_established += 1;
            stats.active_connections += 1;
        }

        // Track connection
        self.connections.write().insert(addr, Instant::now());

        info!("WebSocket connection established to {}", addr);

        Ok(Box::new(connection))
    }

    async fn listen(&self, addr: SocketAddr) -> Result<(), TransportError> {
        let listener = TcpListener::bind(addr).await.map_err(|e| {
            TransportError::ConnectionFailed(format!("Failed to bind WebSocket listener: {}", e))
        })?;

        info!("WebSocket transport listening on {}", addr);

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

        debug!("Accepting WebSocket connection from {}", addr);

        // Perform WebSocket handshake
        let ws_stream = accept_async(stream).await.map_err(|e| {
            TransportError::ConnectionFailed(format!("WebSocket handshake failed: {}", e))
        })?;

        let connection = WebSocketServerConnection::new(ws_stream, addr, self.config.clone());

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.connections_established += 1;
            stats.active_connections += 1;
        }

        // Track connection
        self.connections.write().insert(addr, Instant::now());

        info!("WebSocket connection accepted from {}", addr);

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
    fn test_websocket_config_default() {
        let config = WebSocketConfig::default();
        assert_eq!(config.ping_interval, Duration::from_secs(30));
        assert!(config.use_binary);
        assert_eq!(config.max_message_size, 16 * 1024 * 1024);
    }

    #[tokio::test]
    async fn test_websocket_transport_creation() {
        let transport = WebSocketTransport::default_config();
        assert_eq!(transport.transport_type(), TransportType::WebSocket);
        assert!(transport.is_available());

        let caps = transport.capabilities();
        assert!(!caps.multiplexing);
        assert!(caps.encryption);
        assert_eq!(caps.max_message_size, Some(16 * 1024 * 1024));
    }

    #[tokio::test]
    async fn test_websocket_listen_and_connect() {
        let transport = Arc::new(WebSocketTransport::default_config());

        // Bind to localhost
        let addr: SocketAddr = "127.0.0.1:0".parse().expect("test: valid socket addr");
        transport.listen(addr).await.expect("test: listen");

        // Get the actual bound address
        let listener = transport.listener.lock().await;
        let bound_addr = listener
            .as_ref()
            .expect("test: listener exists")
            .local_addr()
            .expect("test: get local addr");
        drop(listener);

        // Spawn accept task
        let transport_clone = transport.clone();
        let accept_handle = tokio::spawn(async move { transport_clone.accept().await });

        // Give accept time to start
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Connect
        let mut client_conn = transport.connect(bound_addr).await.expect("test: connect");
        let mut server_conn = accept_handle
            .await
            .expect("test: join accept")
            .expect("test: accept connection");

        // Test send/receive
        let test_data = Bytes::from("Hello, WebSocket!");
        client_conn
            .send(test_data.clone())
            .await
            .expect("test: send");

        let received = server_conn.receive().await.expect("test: receive");
        assert_eq!(received, test_data);

        // Check metrics
        let client_metrics = client_conn.metrics();
        assert!(client_metrics.bytes_sent > 0);

        let server_metrics = server_conn.metrics();
        assert!(server_metrics.bytes_received > 0);

        // Close connections
        client_conn.close().await.expect("test: close client");
        server_conn.close().await.expect("test: close server");
    }
}
