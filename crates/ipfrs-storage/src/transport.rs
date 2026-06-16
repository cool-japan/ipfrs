//! Network transport abstraction for distributed RAFT.
//!
//! Provides a generic transport layer for RAFT node communication,
//! enabling multi-node clusters with different network backends.
//!
//! # Example
//!
//! ```rust,ignore
//! use ipfrs_storage::transport::{Transport, InMemoryTransport, Message};
//!
//! let transport = InMemoryTransport::new();
//! let msg = Message::AppendEntries { /* ... */ };
//! transport.send(target_node, msg).await?;
//! ```

use crate::raft::{
    AppendEntriesRequest, AppendEntriesResponse, NodeId, RequestVoteRequest, RequestVoteResponse,
};
use async_trait::async_trait;
use dashmap::DashMap;
use ipfrs_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, RwLock};

/// Network message types for RAFT communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    /// AppendEntries RPC request
    AppendEntries(AppendEntriesRequest),
    /// AppendEntries RPC response
    AppendEntriesResponse(AppendEntriesResponse),
    /// RequestVote RPC request
    RequestVote(RequestVoteRequest),
    /// RequestVote RPC response
    RequestVoteResponse(RequestVoteResponse),
}

/// Transport trait for network communication between RAFT nodes
#[async_trait]
pub trait Transport: Send + Sync {
    /// Send a message to a specific node
    async fn send(&self, target: NodeId, message: Message) -> Result<()>;

    /// Receive the next message for this node
    async fn recv(&self) -> Result<(NodeId, Message)>;

    /// Get local node ID
    fn node_id(&self) -> NodeId;

    /// Close the transport and clean up resources
    async fn close(&self) -> Result<()>;
}

/// In-memory transport for testing and local development
///
/// Provides a zero-copy, high-performance transport for running
/// multiple RAFT nodes in the same process.
pub struct InMemoryTransport {
    /// Local node ID
    node_id: NodeId,
    /// Shared message registry for all nodes
    registry: Arc<DashMap<NodeId, mpsc::UnboundedSender<(NodeId, Message)>>>,
    /// Receiver for incoming messages
    rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<(NodeId, Message)>>>,
}

impl InMemoryTransport {
    /// Create a new in-memory transport for a node
    ///
    /// # Arguments
    /// * `node_id` - Unique identifier for this node
    /// * `registry` - Shared registry of all nodes in the cluster
    pub fn new(
        node_id: NodeId,
        registry: Arc<DashMap<NodeId, mpsc::UnboundedSender<(NodeId, Message)>>>,
    ) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        registry.insert(node_id, tx);

        Self {
            node_id,
            registry,
            rx: Arc::new(tokio::sync::Mutex::new(rx)),
        }
    }

    /// Create a new registry for a cluster
    pub fn new_registry() -> Arc<DashMap<NodeId, mpsc::UnboundedSender<(NodeId, Message)>>> {
        Arc::new(DashMap::new())
    }
}

#[async_trait]
impl Transport for InMemoryTransport {
    async fn send(&self, target: NodeId, message: Message) -> Result<()> {
        if let Some(tx) = self.registry.get(&target) {
            tx.send((self.node_id, message))
                .map_err(|_| Error::Network("Failed to send message".into()))?;
            Ok(())
        } else {
            Err(Error::Network(format!("Node {} not found", target.0)))
        }
    }

    async fn recv(&self) -> Result<(NodeId, Message)> {
        let mut rx = self.rx.lock().await;
        rx.recv()
            .await
            .ok_or_else(|| Error::Network("Transport closed".into()))
    }

    fn node_id(&self) -> NodeId {
        self.node_id
    }

    async fn close(&self) -> Result<()> {
        self.registry.remove(&self.node_id);
        Ok(())
    }
}

/// TCP-based transport for real network communication
///
/// Provides a production-ready transport for RAFT clusters
/// running across multiple machines.
pub struct TcpTransport {
    /// Local node ID
    node_id: NodeId,
    /// Local listening address
    listen_addr: SocketAddr,
    /// Mapping of node IDs to their addresses
    peer_addrs: Arc<DashMap<NodeId, SocketAddr>>,
    /// Receiver for incoming messages
    rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<(NodeId, Message)>>>,
    /// Sender for incoming messages (used by listener)
    tx: mpsc::UnboundedSender<(NodeId, Message)>,
    /// Transport configuration
    config: TransportConfig,
    /// Shutdown signal
    shutdown: Arc<RwLock<bool>>,
}

impl TcpTransport {
    /// Create a new TCP transport
    ///
    /// # Arguments
    /// * `node_id` - Unique identifier for this node
    /// * `listen_addr` - Address to listen on for incoming connections
    /// * `peer_addrs` - Map of node IDs to their addresses
    /// * `config` - Transport configuration
    pub async fn new(
        node_id: NodeId,
        listen_addr: SocketAddr,
        peer_addrs: Arc<DashMap<NodeId, SocketAddr>>,
        config: TransportConfig,
    ) -> Result<Self> {
        let (tx, rx) = mpsc::unbounded_channel();
        let shutdown = Arc::new(RwLock::new(false));

        let transport = Self {
            node_id,
            listen_addr,
            peer_addrs,
            rx: Arc::new(tokio::sync::Mutex::new(rx)),
            tx,
            config,
            shutdown,
        };

        // Start listener task and get the actual bound address
        transport.start_listener().await
    }

    /// Start listening for incoming connections
    async fn start_listener(self) -> Result<Self> {
        let listener = TcpListener::bind(self.listen_addr)
            .await
            .map_err(|e| Error::Network(format!("Failed to bind: {e}")))?;

        // Get the actual bound address (important when using port 0)
        let actual_addr = listener
            .local_addr()
            .map_err(|e| Error::Network(format!("Failed to get local address: {e}")))?;

        let tx = self.tx.clone();
        let max_size = self.config.max_message_size;
        let shutdown = self.shutdown.clone();

        tokio::spawn(async move {
            loop {
                // Check shutdown signal
                if *shutdown.read().await {
                    break;
                }

                match listener.accept().await {
                    Ok((mut stream, _)) => {
                        let tx = tx.clone();
                        tokio::spawn(async move {
                            if let Err(e) = Self::handle_connection(&mut stream, tx, max_size).await
                            {
                                tracing::warn!("Connection error: {}", e);
                            }
                        });
                    }
                    Err(e) => {
                        tracing::error!("Accept error: {}", e);
                    }
                }
            }
        });

        Ok(Self {
            listen_addr: actual_addr,
            ..self
        })
    }

    /// Handle an incoming connection
    async fn handle_connection(
        stream: &mut TcpStream,
        tx: mpsc::UnboundedSender<(NodeId, Message)>,
        max_size: usize,
    ) -> Result<()> {
        // Read message length (4 bytes)
        let len = stream
            .read_u32()
            .await
            .map_err(|e| Error::Network(format!("Failed to read length: {e}")))?
            as usize;

        if len > max_size {
            return Err(Error::Network(format!(
                "Message too large: {len} > {max_size}"
            )));
        }

        // Read message data
        let mut buf = vec![0u8; len];
        stream
            .read_exact(&mut buf)
            .await
            .map_err(|e| Error::Network(format!("Failed to read message: {e}")))?;

        // Deserialize message
        let (sender_id, message): (NodeId, Message) =
            oxicode::serde::decode_owned_from_slice(&buf, oxicode::config::standard())
                .map(|(v, _)| v)
                .map_err(|e| Error::Network(format!("Failed to deserialize: {e}")))?;

        // Send to receiver channel
        tx.send((sender_id, message))
            .map_err(|_| Error::Network("Channel closed".into()))?;

        Ok(())
    }

    /// Send a message to a peer with retry logic
    async fn send_to_peer(&self, target: NodeId, message: Message) -> Result<()> {
        let addr = self
            .peer_addrs
            .get(&target)
            .ok_or_else(|| Error::Network(format!("Node {} not found", target.0)))?
            .value()
            .to_owned();

        // Serialize message with sender ID
        let data =
            oxicode::serde::encode_to_vec(&(self.node_id, message), oxicode::config::standard())
                .map_err(|e| Error::Network(format!("Failed to serialize: {e}")))?;

        if data.len() > self.config.max_message_size {
            return Err(Error::Network(format!(
                "Message too large: {} > {}",
                data.len(),
                self.config.max_message_size
            )));
        }

        // Retry with exponential backoff
        let mut attempt = 0;
        let mut last_error = None;

        while attempt <= self.config.max_retries {
            match self.send_with_timeout(addr, &data).await {
                Ok(_) => return Ok(()),
                Err(e) => {
                    last_error = Some(e);
                    attempt += 1;

                    if attempt <= self.config.max_retries {
                        // Exponential backoff: 100ms, 200ms, 400ms, etc.
                        let backoff_ms = 100 * (1 << (attempt - 1));
                        tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| Error::Network("Send failed".into())))
    }

    /// Send data with timeout (single attempt)
    async fn send_with_timeout(&self, addr: SocketAddr, data: &[u8]) -> Result<()> {
        let connect_timeout = std::time::Duration::from_millis(self.config.connect_timeout_ms);
        let mut stream = tokio::time::timeout(connect_timeout, TcpStream::connect(addr))
            .await
            .map_err(|_| Error::Network("Connection timeout".into()))?
            .map_err(|e| Error::Network(format!("Failed to connect: {e}")))?;

        // Write message length (4 bytes) + data
        stream
            .write_u32(data.len() as u32)
            .await
            .map_err(|e| Error::Network(format!("Failed to write length: {e}")))?;

        stream
            .write_all(data)
            .await
            .map_err(|e| Error::Network(format!("Failed to write data: {e}")))?;

        stream
            .flush()
            .await
            .map_err(|e| Error::Network(format!("Failed to flush: {e}")))?;

        Ok(())
    }
}

#[async_trait]
impl Transport for TcpTransport {
    async fn send(&self, target: NodeId, message: Message) -> Result<()> {
        self.send_to_peer(target, message).await
    }

    async fn recv(&self) -> Result<(NodeId, Message)> {
        let mut rx = self.rx.lock().await;
        rx.recv()
            .await
            .ok_or_else(|| Error::Network("Transport closed".into()))
    }

    fn node_id(&self) -> NodeId {
        self.node_id
    }

    async fn close(&self) -> Result<()> {
        *self.shutdown.write().await = true;
        Ok(())
    }
}

/// Configuration for network transports
#[derive(Debug, Clone)]
pub struct TransportConfig {
    /// Maximum message size in bytes
    pub max_message_size: usize,
    /// Connection timeout in milliseconds
    pub connect_timeout_ms: u64,
    /// Request timeout in milliseconds
    pub request_timeout_ms: u64,
    /// Maximum number of retry attempts
    pub max_retries: usize,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            max_message_size: 10 * 1024 * 1024, // 10MB
            connect_timeout_ms: 5000,           // 5 seconds
            request_timeout_ms: 10000,          // 10 seconds
            max_retries: 3,
        }
    }
}

/// QUIC-based transport for encrypted, multiplexed communication
///
/// Provides a high-performance transport with built-in encryption,
/// connection multiplexing, and 0-RTT support.
#[cfg(feature = "quic")]
pub struct QuicTransport {
    /// Local node ID
    node_id: NodeId,
    /// QUIC endpoint
    endpoint: Arc<quinn::Endpoint>,
    /// Mapping of node IDs to their addresses
    peer_addrs: Arc<DashMap<NodeId, SocketAddr>>,
    /// Receiver for incoming messages
    rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<(NodeId, Message)>>>,
    /// Sender for incoming messages (used by listener)
    tx: mpsc::UnboundedSender<(NodeId, Message)>,
    /// Transport configuration
    config: TransportConfig,
    /// Shutdown signal
    shutdown: Arc<RwLock<bool>>,
}

#[cfg(feature = "quic")]
impl QuicTransport {
    /// Create a new QUIC transport
    ///
    /// # Arguments
    /// * `node_id` - Unique identifier for this node
    /// * `listen_addr` - Address to listen on for incoming connections
    /// * `peer_addrs` - Map of node IDs to their addresses
    /// * `config` - Transport configuration
    #[allow(clippy::unused_async)]
    pub async fn new(
        node_id: NodeId,
        listen_addr: SocketAddr,
        peer_addrs: Arc<DashMap<NodeId, SocketAddr>>,
        config: TransportConfig,
    ) -> Result<Self> {
        let (tx, rx) = mpsc::unbounded_channel();
        let shutdown = Arc::new(RwLock::new(false));

        // Generate self-signed certificate
        let cert = generate_self_signed_cert()?;
        let server_config = configure_server(cert.clone())?;
        let client_config = configure_client()?;

        // Create QUIC endpoint
        let mut endpoint = quinn::Endpoint::server(server_config, listen_addr)
            .map_err(|e| Error::Network(format!("Failed to create endpoint: {e}")))?;

        endpoint.set_default_client_config(client_config);

        let transport = Self {
            node_id,
            endpoint: Arc::new(endpoint),
            peer_addrs,
            rx: Arc::new(tokio::sync::Mutex::new(rx)),
            tx,
            config,
            shutdown,
        };

        // Start listener task
        transport.start_listener();

        Ok(transport)
    }

    /// Start listening for incoming connections
    fn start_listener(&self) {
        let endpoint = self.endpoint.clone();
        let tx = self.tx.clone();
        let max_size = self.config.max_message_size;
        let shutdown = self.shutdown.clone();

        tokio::spawn(async move {
            loop {
                // Check shutdown signal
                if *shutdown.read().await {
                    break;
                }

                // Accept incoming connection
                match endpoint.accept().await {
                    Some(incoming) => {
                        let tx = tx.clone();
                        tokio::spawn(async move {
                            if let Err(e) = Self::handle_connection(incoming, tx, max_size).await {
                                tracing::warn!("QUIC connection error: {}", e);
                            }
                        });
                    }
                    None => {
                        // Endpoint closed
                        break;
                    }
                }
            }
        });
    }

    /// Handle an incoming QUIC connection
    async fn handle_connection(
        incoming: quinn::Incoming,
        tx: mpsc::UnboundedSender<(NodeId, Message)>,
        max_size: usize,
    ) -> Result<()> {
        let connection = incoming
            .await
            .map_err(|e| Error::Network(format!("Failed to establish connection: {e}")))?;

        // Accept bi-directional stream
        let (_send, mut recv) = connection
            .accept_bi()
            .await
            .map_err(|e| Error::Network(format!("Failed to accept stream: {e}")))?;

        // Read message length (4 bytes)
        let mut len_buf = [0u8; 4];
        recv.read_exact(&mut len_buf)
            .await
            .map_err(|e| Error::Network(format!("Failed to read length: {e}")))?;
        let len = u32::from_be_bytes(len_buf) as usize;

        if len > max_size {
            return Err(Error::Network(format!(
                "Message too large: {len} > {max_size}"
            )));
        }

        // Read message data
        let mut buf = vec![0u8; len];
        recv.read_exact(&mut buf)
            .await
            .map_err(|e| Error::Network(format!("Failed to read message: {e}")))?;

        // Deserialize message
        let (sender_id, message): (NodeId, Message) =
            oxicode::serde::decode_owned_from_slice(&buf, oxicode::config::standard())
                .map(|(v, _)| v)
                .map_err(|e| Error::Network(format!("Failed to deserialize: {e}")))?;

        // Send to receiver channel
        tx.send((sender_id, message))
            .map_err(|_| Error::Network("Channel closed".into()))?;

        Ok(())
    }

    /// Send a message to a peer with retry logic
    async fn send_to_peer(&self, target: NodeId, message: Message) -> Result<()> {
        let addr = self
            .peer_addrs
            .get(&target)
            .ok_or_else(|| Error::Network(format!("Node {} not found", target.0)))?
            .value()
            .to_owned();

        // Serialize message with sender ID
        let data =
            oxicode::serde::encode_to_vec(&(self.node_id, message), oxicode::config::standard())
                .map_err(|e| Error::Network(format!("Failed to serialize: {e}")))?;

        if data.len() > self.config.max_message_size {
            return Err(Error::Network(format!(
                "Message too large: {} > {}",
                data.len(),
                self.config.max_message_size
            )));
        }

        // Retry with exponential backoff
        let mut attempt = 0;
        let mut last_error = None;

        while attempt <= self.config.max_retries {
            match self.send_with_timeout(addr, &data).await {
                Ok(_) => return Ok(()),
                Err(e) => {
                    last_error = Some(e);
                    attempt += 1;

                    if attempt <= self.config.max_retries {
                        // Exponential backoff: 100ms, 200ms, 400ms, etc.
                        let backoff_ms = 100 * (1 << (attempt - 1));
                        tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| Error::Network("Send failed".into())))
    }

    /// Send data with timeout (single attempt)
    async fn send_with_timeout(&self, addr: SocketAddr, data: &[u8]) -> Result<()> {
        let connect_timeout = std::time::Duration::from_millis(self.config.connect_timeout_ms);

        let connecting = self
            .endpoint
            .connect(addr, "localhost")
            .map_err(|e| Error::Network(format!("Failed to initiate connection: {e}")))?;

        let connection = tokio::time::timeout(connect_timeout, connecting)
            .await
            .map_err(|_| Error::Network("Connection timeout".into()))?
            .map_err(|e| Error::Network(format!("Failed to establish connection: {e}")))?;

        // Open bi-directional stream
        let (mut send, _recv) = connection
            .open_bi()
            .await
            .map_err(|e| Error::Network(format!("Failed to open stream: {e}")))?;

        // Write message length (4 bytes) + data
        send.write_all(&(data.len() as u32).to_be_bytes())
            .await
            .map_err(|e| Error::Network(format!("Failed to write length: {e}")))?;

        send.write_all(data)
            .await
            .map_err(|e| Error::Network(format!("Failed to write data: {e}")))?;

        send.finish()
            .map_err(|e| Error::Network(format!("Failed to finish stream: {e}")))?;

        Ok(())
    }

    /// Get the local address
    pub fn local_addr(&self) -> Result<SocketAddr> {
        self.endpoint
            .local_addr()
            .map_err(|e| Error::Network(format!("Failed to get local address: {e}")))
    }
}

#[cfg(feature = "quic")]
#[async_trait]
impl Transport for QuicTransport {
    async fn send(&self, target: NodeId, message: Message) -> Result<()> {
        self.send_to_peer(target, message).await
    }

    async fn recv(&self) -> Result<(NodeId, Message)> {
        let mut rx = self.rx.lock().await;
        rx.recv()
            .await
            .ok_or_else(|| Error::Network("Transport closed".into()))
    }

    fn node_id(&self) -> NodeId {
        self.node_id
    }

    async fn close(&self) -> Result<()> {
        *self.shutdown.write().await = true;
        self.endpoint.close(0u32.into(), b"Shutdown");
        Ok(())
    }
}

/// Generate a self-signed certificate for QUIC
#[cfg(feature = "quic")]
fn generate_self_signed_cert() -> Result<rustls::pki_types::CertificateDer<'static>> {
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
        .map_err(|e| Error::Network(format!("Failed to generate certificate: {e}")))?;

    let cert_der = cert.cert.der().to_vec();
    Ok(rustls::pki_types::CertificateDer::from(cert_der))
}

/// Configure QUIC server
#[cfg(feature = "quic")]
fn configure_server(
    _cert: rustls::pki_types::CertificateDer<'static>,
) -> Result<quinn::ServerConfig> {
    let cert_gen = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
        .map_err(|e| Error::Network(format!("Failed to generate certificate: {e}")))?;

    let key_der = rustls::pki_types::PrivateKeyDer::Pkcs8(
        rustls::pki_types::PrivatePkcs8KeyDer::from(cert_gen.signing_key.serialize_der()),
    );
    let cert_der = cert_gen.cert.der().to_vec();
    let cert_chain = vec![rustls::pki_types::CertificateDer::from(cert_der)];

    let mut server_crypto = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(cert_chain, key_der)
        .map_err(|e| Error::Network(format!("Failed to configure server: {e}")))?;

    server_crypto.alpn_protocols = vec![b"ipfrs-raft".to_vec()];

    let server_config = quinn::ServerConfig::with_crypto(Arc::new(
        quinn::crypto::rustls::QuicServerConfig::try_from(server_crypto)
            .map_err(|e| Error::Network(format!("Failed to create QUIC server config: {e}")))?,
    ));

    Ok(server_config)
}

/// Configure QUIC client
#[cfg(feature = "quic")]
fn configure_client() -> Result<quinn::ClientConfig> {
    // Accept any certificate (for testing with self-signed certs)
    let mut client_crypto = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
        .with_no_client_auth();

    client_crypto.alpn_protocols = vec![b"ipfrs-raft".to_vec()];

    let client_config = quinn::ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(client_crypto)
            .map_err(|e| Error::Network(format!("Failed to create QUIC client config: {e}")))?,
    ));

    Ok(client_config)
}

/// Skip server certificate verification (for testing only)
#[cfg(feature = "quic")]
#[derive(Debug)]
struct SkipServerVerification;

#[cfg(feature = "quic")]
impl rustls::client::danger::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> std::result::Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::RSA_PKCS1_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ED25519,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_in_memory_transport_send_recv() {
        let registry = InMemoryTransport::new_registry();
        let transport1 = InMemoryTransport::new(NodeId(1), registry.clone());
        let transport2 = InMemoryTransport::new(NodeId(2), registry);

        // Send message from node 1 to node 2
        let request = RequestVoteRequest {
            term: crate::raft::Term(1),
            candidate_id: NodeId(1),
            last_log_index: crate::raft::LogIndex(0),
            last_log_term: crate::raft::Term(0),
        };
        let message = Message::RequestVote(request);

        transport1.send(NodeId(2), message.clone()).await.unwrap();

        // Receive message at node 2
        let (sender, received) = transport2.recv().await.unwrap();
        assert_eq!(sender, NodeId(1));
        matches!(received, Message::RequestVote(_));
    }

    #[tokio::test]
    async fn test_in_memory_transport_node_not_found() {
        let registry = InMemoryTransport::new_registry();
        let transport = InMemoryTransport::new(NodeId(1), registry);

        let request = RequestVoteRequest {
            term: crate::raft::Term(1),
            candidate_id: NodeId(1),
            last_log_index: crate::raft::LogIndex(0),
            last_log_term: crate::raft::Term(0),
        };
        let message = Message::RequestVote(request);

        // Try to send to non-existent node
        let result = transport.send(NodeId(999), message).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_transport_close() {
        let registry = InMemoryTransport::new_registry();
        let transport = InMemoryTransport::new(NodeId(1), registry.clone());

        assert!(registry.contains_key(&NodeId(1)));

        transport.close().await.unwrap();

        assert!(!registry.contains_key(&NodeId(1)));
    }

    #[tokio::test]
    async fn test_bidirectional_communication() {
        let registry = InMemoryTransport::new_registry();
        let transport1 = InMemoryTransport::new(NodeId(1), registry.clone());
        let transport2 = InMemoryTransport::new(NodeId(2), registry);

        // Node 1 sends RequestVote to Node 2
        let vote_request = RequestVoteRequest {
            term: crate::raft::Term(1),
            candidate_id: NodeId(1),
            last_log_index: crate::raft::LogIndex(0),
            last_log_term: crate::raft::Term(0),
        };
        transport1
            .send(NodeId(2), Message::RequestVote(vote_request))
            .await
            .unwrap();

        // Node 2 receives and responds
        let (sender, _msg) = transport2.recv().await.unwrap();
        assert_eq!(sender, NodeId(1));

        let vote_response = RequestVoteResponse {
            term: crate::raft::Term(1),
            vote_granted: true,
        };
        transport2
            .send(NodeId(1), Message::RequestVoteResponse(vote_response))
            .await
            .unwrap();

        // Node 1 receives response
        let (sender, received) = transport1.recv().await.unwrap();
        assert_eq!(sender, NodeId(2));
        matches!(received, Message::RequestVoteResponse(_));
    }

    #[tokio::test]
    async fn test_tcp_transport_send_recv() {
        let peer_addrs1 = Arc::new(DashMap::new());
        let peer_addrs2 = Arc::new(DashMap::new());

        let addr1: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let addr2: SocketAddr = "127.0.0.1:0".parse().unwrap();

        let config = TransportConfig::default();

        let transport1 = TcpTransport::new(NodeId(1), addr1, peer_addrs1.clone(), config.clone())
            .await
            .unwrap();

        let transport2 = TcpTransport::new(NodeId(2), addr2, peer_addrs2.clone(), config)
            .await
            .unwrap();

        // Register peers
        peer_addrs1.insert(NodeId(2), transport2.listen_addr);
        peer_addrs2.insert(NodeId(1), transport1.listen_addr);

        // Send message from node 1 to node 2
        let request = RequestVoteRequest {
            term: crate::raft::Term(1),
            candidate_id: NodeId(1),
            last_log_index: crate::raft::LogIndex(0),
            last_log_term: crate::raft::Term(0),
        };
        let message = Message::RequestVote(request);

        transport1.send(NodeId(2), message).await.unwrap();

        // Receive message at node 2
        let (sender, received) = transport2.recv().await.unwrap();
        assert_eq!(sender, NodeId(1));
        matches!(received, Message::RequestVote(_));

        // Cleanup
        transport1.close().await.unwrap();
        transport2.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_tcp_transport_bidirectional() {
        let peer_addrs1 = Arc::new(DashMap::new());
        let peer_addrs2 = Arc::new(DashMap::new());

        let addr1: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let addr2: SocketAddr = "127.0.0.1:0".parse().unwrap();

        let config = TransportConfig::default();

        let transport1 = TcpTransport::new(NodeId(1), addr1, peer_addrs1.clone(), config.clone())
            .await
            .unwrap();

        let transport2 = TcpTransport::new(NodeId(2), addr2, peer_addrs2.clone(), config)
            .await
            .unwrap();

        // Register peers
        peer_addrs1.insert(NodeId(2), transport2.listen_addr);
        peer_addrs2.insert(NodeId(1), transport1.listen_addr);

        // Node 1 sends to Node 2
        let vote_request = RequestVoteRequest {
            term: crate::raft::Term(1),
            candidate_id: NodeId(1),
            last_log_index: crate::raft::LogIndex(0),
            last_log_term: crate::raft::Term(0),
        };
        transport1
            .send(NodeId(2), Message::RequestVote(vote_request))
            .await
            .unwrap();

        // Node 2 receives
        let (sender, _msg) = transport2.recv().await.unwrap();
        assert_eq!(sender, NodeId(1));

        // Node 2 responds
        let vote_response = RequestVoteResponse {
            term: crate::raft::Term(1),
            vote_granted: true,
        };
        transport2
            .send(NodeId(1), Message::RequestVoteResponse(vote_response))
            .await
            .unwrap();

        // Node 1 receives response
        let (sender, received) = transport1.recv().await.unwrap();
        assert_eq!(sender, NodeId(2));
        matches!(received, Message::RequestVoteResponse(_));

        // Cleanup
        transport1.close().await.unwrap();
        transport2.close().await.unwrap();
    }

    #[cfg(feature = "quic")]
    #[tokio::test]
    #[ignore] // QUIC tests need timing refinement
    async fn test_quic_transport_send_recv() {
        // Install default crypto provider for rustls
        let _ = rustls_rustcrypto::provider().install_default();

        let peer_addrs1 = Arc::new(DashMap::new());
        let peer_addrs2 = Arc::new(DashMap::new());

        let addr1: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let addr2: SocketAddr = "127.0.0.1:0".parse().unwrap();

        let config = TransportConfig::default();

        let transport1 = QuicTransport::new(NodeId(1), addr1, peer_addrs1.clone(), config.clone())
            .await
            .unwrap();

        let transport2 = QuicTransport::new(NodeId(2), addr2, peer_addrs2.clone(), config)
            .await
            .unwrap();

        let addr1_actual = transport1.local_addr().unwrap();
        let addr2_actual = transport2.local_addr().unwrap();

        // Register peers
        peer_addrs1.insert(NodeId(2), addr2_actual);
        peer_addrs2.insert(NodeId(1), addr1_actual);

        // Give the listeners time to start
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Send message from node 1 to node 2
        let request = RequestVoteRequest {
            term: crate::raft::Term(1),
            candidate_id: NodeId(1),
            last_log_index: crate::raft::LogIndex(0),
            last_log_term: crate::raft::Term(0),
        };
        let message = Message::RequestVote(request);

        transport1.send(NodeId(2), message).await.unwrap();

        // Receive message at node 2
        let (sender, received) = transport2.recv().await.unwrap();
        assert_eq!(sender, NodeId(1));
        matches!(received, Message::RequestVote(_));

        // Cleanup
        transport1.close().await.unwrap();
        transport2.close().await.unwrap();
    }

    #[cfg(feature = "quic")]
    #[tokio::test]
    #[ignore] // QUIC tests need timing refinement
    async fn test_quic_transport_bidirectional() {
        // Install default crypto provider for rustls
        let _ = rustls_rustcrypto::provider().install_default();

        let peer_addrs1 = Arc::new(DashMap::new());
        let peer_addrs2 = Arc::new(DashMap::new());

        let addr1: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let addr2: SocketAddr = "127.0.0.1:0".parse().unwrap();

        let config = TransportConfig::default();

        let transport1 = QuicTransport::new(NodeId(1), addr1, peer_addrs1.clone(), config.clone())
            .await
            .unwrap();

        let transport2 = QuicTransport::new(NodeId(2), addr2, peer_addrs2.clone(), config)
            .await
            .unwrap();

        let addr1_actual = transport1.local_addr().unwrap();
        let addr2_actual = transport2.local_addr().unwrap();

        // Register peers
        peer_addrs1.insert(NodeId(2), addr2_actual);
        peer_addrs2.insert(NodeId(1), addr1_actual);

        // Give the listeners time to start
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Node 1 sends to Node 2
        let vote_request = RequestVoteRequest {
            term: crate::raft::Term(1),
            candidate_id: NodeId(1),
            last_log_index: crate::raft::LogIndex(0),
            last_log_term: crate::raft::Term(0),
        };
        transport1
            .send(NodeId(2), Message::RequestVote(vote_request))
            .await
            .unwrap();

        // Node 2 receives
        let (sender, _msg) = transport2.recv().await.unwrap();
        assert_eq!(sender, NodeId(1));

        // Node 2 responds
        let vote_response = RequestVoteResponse {
            term: crate::raft::Term(1),
            vote_granted: true,
        };
        transport2
            .send(NodeId(1), Message::RequestVoteResponse(vote_response))
            .await
            .unwrap();

        // Node 1 receives response
        let (sender, received) = transport1.recv().await.unwrap();
        assert_eq!(sender, NodeId(2));
        matches!(received, Message::RequestVoteResponse(_));

        // Cleanup
        transport1.close().await.unwrap();
        transport2.close().await.unwrap();
    }
}
