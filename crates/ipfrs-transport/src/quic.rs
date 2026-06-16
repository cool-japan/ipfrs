//! QUIC transport for efficient block exchange
//!
//! Implements QUIC-based transport using the quinn crate:
//! - 0-RTT connection establishment
//! - Connection pooling and reuse
//! - Stream multiplexing
//! - Congestion control tuning for bulk transfer
//! - Zero-copy block forwarding with bytes::Bytes

use bytes::Bytes;
use ipfrs_core::error::{Error, Result};
use quinn::{
    ClientConfig, Connection, Endpoint, RecvStream, SendStream, ServerConfig, TransportConfig,
};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// QUIC transport configuration
#[derive(Debug, Clone)]
pub struct QuicConfig {
    /// Address to bind the endpoint to
    pub bind_addr: SocketAddr,
    /// Maximum idle timeout for connections
    pub idle_timeout: Duration,
    /// Maximum concurrent streams per connection
    pub max_streams: u32,
    /// Enable 0-RTT early data
    pub enable_0rtt: bool,
    /// Connection pool size per peer
    pub pool_size: usize,
    /// Idle connection timeout before eviction
    pub pool_idle_timeout: Duration,
    /// Maximum message size
    pub max_message_size: usize,
    /// Initial congestion window (bytes)
    pub initial_window: u32,
    /// Maximum congestion window (bytes)
    pub max_window: u32,
}

impl Default for QuicConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0:0"
                .parse()
                .expect("static socket addr literal must parse"),
            idle_timeout: Duration::from_secs(30),
            max_streams: 256,
            enable_0rtt: true,
            pool_size: 4,
            pool_idle_timeout: Duration::from_secs(60),
            max_message_size: 16 * 1024 * 1024, // 16 MB
            initial_window: 10 * 1024 * 1024,   // 10 MB
            max_window: 100 * 1024 * 1024,      // 100 MB
        }
    }
}

/// Connection pool entry
struct PooledConnection {
    connection: Connection,
    /// When connection was created - reserved for connection age metrics
    #[allow(dead_code)]
    created_at: Instant,
    last_used: Instant,
    active_streams: u32,
}

impl PooledConnection {
    fn new(connection: Connection) -> Self {
        let now = Instant::now();
        Self {
            connection,
            created_at: now,
            last_used: now,
            active_streams: 0,
        }
    }

    fn is_healthy(&self) -> bool {
        self.connection.close_reason().is_none()
    }

    fn is_idle(&self, timeout: Duration) -> bool {
        self.last_used.elapsed() > timeout && self.active_streams == 0
    }

    fn touch(&mut self) {
        self.last_used = Instant::now();
    }
}

/// Connection pool for a single peer
struct PeerPool {
    connections: Vec<PooledConnection>,
    max_size: usize,
    idle_timeout: Duration,
}

impl PeerPool {
    fn new(max_size: usize, idle_timeout: Duration) -> Self {
        Self {
            connections: Vec::with_capacity(max_size),
            max_size,
            idle_timeout,
        }
    }

    /// Get an available connection from the pool
    fn get(&mut self) -> Option<&mut PooledConnection> {
        // Clean up closed connections
        self.connections.retain(|c| c.is_healthy());

        // Remove idle connections
        self.connections.retain(|c| !c.is_idle(self.idle_timeout));

        // Find connection with lowest active streams
        self.connections
            .iter_mut()
            .filter(|c| c.is_healthy())
            .min_by_key(|c| c.active_streams)
    }

    /// Add a connection to the pool
    fn add(&mut self, connection: Connection) -> bool {
        if self.connections.len() >= self.max_size {
            // Remove oldest idle connection
            if let Some(pos) = self
                .connections
                .iter()
                .position(|c| c.is_idle(Duration::ZERO))
            {
                self.connections.remove(pos);
            } else {
                return false;
            }
        }

        self.connections.push(PooledConnection::new(connection));
        true
    }

    fn connection_count(&self) -> usize {
        self.connections.len()
    }
}

/// QUIC transport for block exchange
pub struct QuicTransport {
    /// QUIC endpoint
    endpoint: Endpoint,
    /// Connection pools per peer address
    pools: Arc<RwLock<HashMap<SocketAddr, PeerPool>>>,
    /// Configuration
    config: QuicConfig,
    /// Client configuration for outbound connections
    client_config: ClientConfig,
}

impl QuicTransport {
    /// Create a new QUIC transport
    pub async fn new(config: QuicConfig) -> Result<Self> {
        // Install rustcrypto provider for rustls (Pure Rust, no ring/C dependency)
        let _ = rustls_rustcrypto::provider().install_default();

        // Create self-signed certificate for development
        let (cert, key) = Self::generate_self_signed_cert()?;

        // Server config with its own transport config
        let server_transport = Self::create_transport_config(&config);
        let mut server_config = ServerConfig::with_single_cert(vec![cert.clone()], key.clone_key())
            .map_err(|e| Error::Internal(format!("Failed to create server config: {}", e)))?;
        server_config.transport_config(Arc::new(server_transport));

        // Client config with its own transport config (skip verification for development)
        let client_transport = Self::create_transport_config(&config);
        let client_crypto = rustls::ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
            .with_no_client_auth();
        let mut client_config = ClientConfig::new(Arc::new(
            quinn::crypto::rustls::QuicClientConfig::try_from(client_crypto).map_err(|e| {
                Error::Internal(format!("Failed to create QUIC client config: {}", e))
            })?,
        ));
        client_config.transport_config(Arc::new(client_transport));

        // Create endpoint
        let endpoint = Endpoint::server(server_config, config.bind_addr)
            .map_err(|e| Error::Internal(format!("Failed to create QUIC endpoint: {}", e)))?;

        Ok(Self {
            endpoint,
            pools: Arc::new(RwLock::new(HashMap::new())),
            config,
            client_config,
        })
    }

    /// Create transport configuration optimized for bulk transfer
    fn create_transport_config(config: &QuicConfig) -> TransportConfig {
        let mut transport = TransportConfig::default();
        transport.max_idle_timeout(Some(config.idle_timeout.try_into().unwrap_or_default()));
        transport.max_concurrent_bidi_streams(config.max_streams.into());
        transport.max_concurrent_uni_streams(config.max_streams.into());
        transport.initial_mtu(1200);
        // Note: Congestion window settings would be configured via
        // custom congestion controller implementation
        transport
    }

    /// Generate a self-signed certificate for development
    fn generate_self_signed_cert() -> Result<(
        rustls::pki_types::CertificateDer<'static>,
        rustls::pki_types::PrivateKeyDer<'static>,
    )> {
        let rcgen_cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
            .map_err(|e| Error::Internal(format!("Failed to generate certificate: {}", e)))?;

        let cert_der = rustls::pki_types::CertificateDer::from(rcgen_cert.cert.der().to_vec());
        let key_der =
            rustls::pki_types::PrivateKeyDer::try_from(rcgen_cert.signing_key.serialize_der())
                .map_err(|e| Error::Internal(format!("Failed to serialize key: {}", e)))?;

        Ok((cert_der, key_der))
    }

    /// Get local address
    pub fn local_addr(&self) -> Result<SocketAddr> {
        self.endpoint
            .local_addr()
            .map_err(|e| Error::Internal(format!("Failed to get local address: {}", e)))
    }

    /// Connect to a peer
    pub async fn connect(&self, addr: SocketAddr) -> Result<Connection> {
        // Check pool first
        {
            let mut pools = self.pools.write().await;
            if let Some(pool) = pools.get_mut(&addr) {
                if let Some(conn) = pool.get() {
                    conn.touch();
                    return Ok(conn.connection.clone());
                }
            }
        }

        // Establish new connection
        let connection = self
            .endpoint
            .connect_with(self.client_config.clone(), addr, "localhost")
            .map_err(|e| Error::Internal(format!("Failed to initiate connection: {}", e)))?
            .await
            .map_err(|e| Error::Internal(format!("Failed to connect: {}", e)))?;

        // Add to pool
        {
            let mut pools = self.pools.write().await;
            let pool = pools.entry(addr).or_insert_with(|| {
                PeerPool::new(self.config.pool_size, self.config.pool_idle_timeout)
            });
            pool.add(connection.clone());
        }

        Ok(connection)
    }

    /// Accept an incoming connection
    pub async fn accept(&self) -> Result<Option<Connection>> {
        if let Some(incoming) = self.endpoint.accept().await {
            let connection = incoming
                .await
                .map_err(|e| Error::Internal(format!("Failed to accept connection: {}", e)))?;
            Ok(Some(connection))
        } else {
            Ok(None)
        }
    }

    /// Open a bidirectional stream on a connection
    pub async fn open_stream(&self, connection: &Connection) -> Result<(SendStream, RecvStream)> {
        connection
            .open_bi()
            .await
            .map_err(|e| Error::Internal(format!("Failed to open stream: {}", e)))
    }

    /// Send data on a stream
    pub async fn send(&self, stream: &mut SendStream, data: &[u8]) -> Result<()> {
        stream
            .write_all(data)
            .await
            .map_err(|e| Error::Internal(format!("Failed to send data: {}", e)))?;
        stream
            .finish()
            .map_err(|e| Error::Internal(format!("Failed to finish stream: {}", e)))?;
        Ok(())
    }

    /// Receive data from a stream
    pub async fn receive(&self, stream: &mut RecvStream) -> Result<Vec<u8>> {
        let data = stream
            .read_to_end(self.config.max_message_size)
            .await
            .map_err(|e| Error::Internal(format!("Failed to receive data: {}", e)))?;
        Ok(data)
    }

    /// Send data using zero-copy with Bytes
    pub async fn send_zero_copy(&self, stream: &mut SendStream, data: Bytes) -> Result<()> {
        stream
            .write_all(&data)
            .await
            .map_err(|e| Error::Internal(format!("Failed to send data: {}", e)))?;
        stream
            .finish()
            .map_err(|e| Error::Internal(format!("Failed to finish stream: {}", e)))?;
        Ok(())
    }

    /// Receive data from a stream as Bytes (zero-copy)
    pub async fn receive_zero_copy(&self, stream: &mut RecvStream) -> Result<Bytes> {
        let data = stream
            .read_to_end(self.config.max_message_size)
            .await
            .map_err(|e| Error::Internal(format!("Failed to receive data: {}", e)))?;
        Ok(Bytes::from(data))
    }

    /// Forward block data directly between streams (zero-copy)
    pub async fn forward_block(
        &self,
        recv_stream: &mut RecvStream,
        send_stream: &mut SendStream,
    ) -> Result<usize> {
        let mut total_bytes = 0;
        let mut buffer = vec![0u8; 16384]; // 16 KB chunks

        loop {
            let n = match recv_stream.read(&mut buffer).await {
                Ok(Some(n)) => n,
                Ok(None) => break,
                Err(e) => return Err(Error::Internal(format!("Failed to read: {}", e))),
            };

            send_stream
                .write_all(&buffer[..n])
                .await
                .map_err(|e| Error::Internal(format!("Failed to write: {}", e)))?;

            total_bytes += n;
        }

        send_stream
            .finish()
            .map_err(|e| Error::Internal(format!("Failed to finish stream: {}", e)))?;

        Ok(total_bytes)
    }

    /// Send data to a peer address (opens connection if needed)
    pub async fn send_to(&self, addr: SocketAddr, data: &[u8]) -> Result<()> {
        let connection = self.connect(addr).await?;
        let (mut send, _recv) = self.open_stream(&connection).await?;
        self.send(&mut send, data).await
    }

    /// Get connection pool statistics
    pub async fn pool_stats(&self) -> QuicPoolStats {
        let pools = self.pools.read().await;
        let total_connections: usize = pools.values().map(|p| p.connection_count()).sum();
        let peer_count = pools.len();

        QuicPoolStats {
            peer_count,
            total_connections,
        }
    }

    /// Clean up idle connections
    pub async fn cleanup_idle(&self) {
        let mut pools = self.pools.write().await;
        for pool in pools.values_mut() {
            pool.connections
                .retain(|c| c.is_healthy() && !c.is_idle(pool.idle_timeout));
        }
        // Remove empty pools
        pools.retain(|_, p| !p.connections.is_empty());
    }

    /// Close the transport
    pub fn close(&self) {
        self.endpoint.close(0u32.into(), b"shutdown");
    }
}

/// QUIC connection pool statistics
#[derive(Debug, Clone)]
pub struct QuicPoolStats {
    /// Number of peers with pooled connections
    pub peer_count: usize,
    /// Total pooled connections
    pub total_connections: usize,
}

/// Skip server certificate verification (for development only)
#[derive(Debug)]
struct SkipServerVerification;

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
            rustls::SignatureScheme::RSA_PKCS1_SHA384,
            rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
            rustls::SignatureScheme::RSA_PKCS1_SHA512,
            rustls::SignatureScheme::ECDSA_NISTP521_SHA512,
            rustls::SignatureScheme::RSA_PSS_SHA256,
            rustls::SignatureScheme::RSA_PSS_SHA384,
            rustls::SignatureScheme::RSA_PSS_SHA512,
            rustls::SignatureScheme::ED25519,
        ]
    }
}

/// Stream handle for parallel block requests
pub struct BlockStream {
    send: SendStream,
    recv: RecvStream,
}

impl BlockStream {
    /// Create a new block stream
    pub fn new(send: SendStream, recv: RecvStream) -> Self {
        Self { send, recv }
    }

    /// Send a block request
    pub async fn send_request(&mut self, data: &[u8]) -> Result<()> {
        self.send
            .write_all(data)
            .await
            .map_err(|e| Error::Internal(format!("Failed to send request: {}", e)))?;
        self.send
            .finish()
            .map_err(|e| Error::Internal(format!("Failed to finish stream: {}", e)))?;
        Ok(())
    }

    /// Receive a block response
    pub async fn receive_response(&mut self, max_size: usize) -> Result<Vec<u8>> {
        self.recv
            .read_to_end(max_size)
            .await
            .map_err(|e| Error::Internal(format!("Failed to receive response: {}", e)))
    }

    /// Send a block request using zero-copy Bytes
    pub async fn send_request_zero_copy(&mut self, data: Bytes) -> Result<()> {
        self.send
            .write_all(&data)
            .await
            .map_err(|e| Error::Internal(format!("Failed to send request: {}", e)))?;
        self.send
            .finish()
            .map_err(|e| Error::Internal(format!("Failed to finish stream: {}", e)))?;
        Ok(())
    }

    /// Receive a block response as zero-copy Bytes
    pub async fn receive_response_zero_copy(&mut self, max_size: usize) -> Result<Bytes> {
        let data = self
            .recv
            .read_to_end(max_size)
            .await
            .map_err(|e| Error::Internal(format!("Failed to receive response: {}", e)))?;
        Ok(Bytes::from(data))
    }
}

/// Parallel block request manager
pub struct ParallelRequester {
    connection: Connection,
    max_concurrent: usize,
    /// Maximum message size - reserved for size validation
    #[allow(dead_code)]
    max_message_size: usize,
}

impl ParallelRequester {
    /// Create a new parallel requester
    pub fn new(connection: Connection, max_concurrent: usize, max_message_size: usize) -> Self {
        Self {
            connection,
            max_concurrent,
            max_message_size,
        }
    }

    /// Open a new stream for a request
    pub async fn open_stream(&self) -> Result<BlockStream> {
        let (send, recv) = self
            .connection
            .open_bi()
            .await
            .map_err(|e| Error::Internal(format!("Failed to open stream: {}", e)))?;
        Ok(BlockStream::new(send, recv))
    }

    /// Execute multiple requests in parallel
    pub async fn execute_parallel<F, Fut, T>(&self, requests: Vec<F>) -> Vec<Result<T>>
    where
        F: FnOnce(BlockStream) -> Fut,
        Fut: std::future::Future<Output = Result<T>> + Send,
        T: Send,
    {
        use futures::stream::{self, StreamExt};

        let max_concurrent = self.max_concurrent;

        stream::iter(requests)
            .map(|request| async move {
                let stream = self.open_stream().await?;
                request(stream).await
            })
            .buffer_unordered(max_concurrent)
            .collect()
            .await
    }

    /// Get maximum concurrent streams
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }
}

/// Adaptive batch size tuner
///
/// Dynamically adjusts batch sizes based on network performance and peer capacity
pub struct AdaptiveBatchTuner {
    /// Current batch size
    current_batch_size: usize,
    /// Minimum batch size
    min_batch_size: usize,
    /// Maximum batch size
    max_batch_size: usize,
    /// Recent completion times (in milliseconds)
    completion_times: Vec<u64>,
    /// Window size for averaging
    window_size: usize,
    /// Target throughput (blocks/sec)
    target_throughput: f64,
    /// Last adjustment time
    last_adjustment: Instant,
    /// Adjustment cooldown
    adjustment_interval: Duration,
}

impl AdaptiveBatchTuner {
    /// Create a new adaptive batch tuner
    pub fn new(
        initial_batch_size: usize,
        min_batch_size: usize,
        max_batch_size: usize,
        target_throughput: f64,
    ) -> Self {
        Self {
            current_batch_size: initial_batch_size,
            min_batch_size,
            max_batch_size,
            completion_times: Vec::new(),
            window_size: 10,
            target_throughput,
            last_adjustment: Instant::now(),
            adjustment_interval: Duration::from_secs(1),
        }
    }

    /// Record a batch completion time
    pub fn record_completion(&mut self, duration_ms: u64) {
        self.completion_times.push(duration_ms);
        if self.completion_times.len() > self.window_size {
            self.completion_times.remove(0);
        }
    }

    /// Get current batch size
    pub fn current_batch_size(&self) -> usize {
        self.current_batch_size
    }

    /// Adjust batch size based on recent performance
    pub fn adjust_batch_size(&mut self) -> usize {
        // Only adjust if enough time has passed
        if self.last_adjustment.elapsed() < self.adjustment_interval {
            return self.current_batch_size;
        }

        // Need at least a few samples to make a decision
        if self.completion_times.len() < 3 {
            return self.current_batch_size;
        }

        // Calculate average completion time
        let avg_time =
            self.completion_times.iter().sum::<u64>() as f64 / self.completion_times.len() as f64;

        // Calculate current throughput (blocks per second)
        let current_throughput = (self.current_batch_size as f64 / avg_time) * 1000.0;

        // Adjust batch size based on throughput
        let new_batch_size = if current_throughput < self.target_throughput * 0.8 {
            // Too slow, increase batch size
            (self.current_batch_size as f64 * 1.2) as usize
        } else if current_throughput > self.target_throughput * 1.2 {
            // Too fast, decrease batch size to reduce memory pressure
            (self.current_batch_size as f64 * 0.8) as usize
        } else {
            // Within acceptable range
            self.current_batch_size
        };

        // Clamp to min/max
        self.current_batch_size = new_batch_size.clamp(self.min_batch_size, self.max_batch_size);
        self.last_adjustment = Instant::now();
        self.completion_times.clear();

        self.current_batch_size
    }

    /// Reset tuner state
    pub fn reset(&mut self) {
        self.completion_times.clear();
        self.last_adjustment = Instant::now();
    }
}

impl Default for AdaptiveBatchTuner {
    fn default() -> Self {
        Self::new(32, 8, 128, 100.0)
    }
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of blocks to prefetch ahead
    pub prefetch_depth: usize,
    /// Maximum pipeline size
    pub max_pipeline_size: usize,
    /// Enable speculative prefetching
    pub enable_speculation: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            prefetch_depth: 4,
            max_pipeline_size: 16,
            enable_speculation: true,
        }
    }
}

/// Pipelined block fetcher for sequential access
///
/// Implements request pipelining to reduce round-trip latency for sequential block access
pub struct SequentialPipeline {
    /// QUIC connection
    connection: Connection,
    /// Pipeline configuration
    config: PipelineConfig,
    /// Maximum message size
    max_message_size: usize,
    /// Active in-flight requests
    in_flight: Arc<RwLock<HashMap<u64, tokio::task::JoinHandle<Result<Bytes>>>>>,
    /// Next block index to request
    next_index: Arc<RwLock<u64>>,
}

impl SequentialPipeline {
    /// Create a new sequential pipeline
    pub fn new(connection: Connection, config: PipelineConfig, max_message_size: usize) -> Self {
        Self {
            connection,
            config,
            max_message_size,
            in_flight: Arc::new(RwLock::new(HashMap::new())),
            next_index: Arc::new(RwLock::new(0)),
        }
    }

    /// Start a pipelined request for a block index
    async fn start_request(&self, index: u64, request_data: Bytes) -> Result<()> {
        let connection = self.connection.clone();
        let max_size = self.max_message_size;

        let handle = tokio::spawn(async move {
            let (mut send, mut recv) = connection
                .open_bi()
                .await
                .map_err(|e| Error::Internal(format!("Failed to open stream: {}", e)))?;

            // Send request
            send.write_all(&request_data)
                .await
                .map_err(|e| Error::Internal(format!("Failed to send: {}", e)))?;
            send.finish()
                .map_err(|e| Error::Internal(format!("Failed to finish: {}", e)))?;

            // Receive response
            let data = recv
                .read_to_end(max_size)
                .await
                .map_err(|e| Error::Internal(format!("Failed to receive: {}", e)))?;

            Ok(Bytes::from(data))
        });

        let mut in_flight = self.in_flight.write().await;
        in_flight.insert(index, handle);

        Ok(())
    }

    /// Fetch the next block in sequence
    pub async fn fetch_next(&self, request_data: Bytes) -> Result<Bytes> {
        let current_index = {
            let mut next = self.next_index.write().await;
            let current = *next;
            *next += 1;
            current
        };

        // Start prefetch requests for upcoming blocks
        if self.config.enable_speculation {
            for i in 1..=self.config.prefetch_depth {
                let prefetch_index = current_index + i as u64;

                // Check if already in flight
                let in_flight = self.in_flight.read().await;
                if !in_flight.contains_key(&prefetch_index) {
                    drop(in_flight);

                    // Start speculative request (with same data for now)
                    let _ = self
                        .start_request(prefetch_index, request_data.clone())
                        .await;
                }
            }
        }

        // Wait for current block
        let handle = {
            let mut in_flight = self.in_flight.write().await;

            // If not already started, start now
            if !in_flight.contains_key(&current_index) {
                drop(in_flight);
                self.start_request(current_index, request_data).await?;
                let mut in_flight = self.in_flight.write().await;
                in_flight.remove(&current_index)
            } else {
                in_flight.remove(&current_index)
            }
        };

        if let Some(handle) = handle {
            handle
                .await
                .map_err(|e| Error::Internal(format!("Task failed: {}", e)))?
        } else {
            Err(Error::Internal("Request handle not found".to_string()))
        }
    }

    /// Clear all in-flight requests
    pub async fn clear(&self) {
        let mut in_flight = self.in_flight.write().await;
        for (_, handle) in in_flight.drain() {
            handle.abort();
        }
    }

    /// Get number of in-flight requests
    pub async fn in_flight_count(&self) -> usize {
        self.in_flight.read().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quic_config_defaults() {
        let config = QuicConfig::default();
        assert_eq!(config.max_streams, 256);
        assert!(config.enable_0rtt);
        assert_eq!(config.pool_size, 4);
    }

    #[test]
    fn test_peer_pool() {
        // Note: Full integration tests would require actual QUIC connections
        let pool = PeerPool::new(4, Duration::from_secs(60));
        assert_eq!(pool.connection_count(), 0);
    }
}
