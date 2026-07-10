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
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::WebPkiSupportedAlgorithms;
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, SignatureScheme};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// A certificate pin that the QUIC client will accept for a server.
///
/// IPFRS peers authenticate one another with *ephemeral, self-signed* certificates
/// that chain to no public certificate authority. Instead of classic webpki path
/// validation (which is meaningless for keys that no public root vouches for), the
/// client authenticates a peer by matching a SHA-256 pin, in the same spirit as SSH
/// `known_hosts` or HPKP-style key pinning. See [`PinnedServerVerifier`] for the full
/// trust model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CertPin {
    /// SHA-256 of the end-entity certificate's `SubjectPublicKeyInfo` (DER).
    ///
    /// Survives certificate re-issuance as long as the key is unchanged. Preferred.
    Spki([u8; 32]),
    /// SHA-256 of the full end-entity certificate DER (exact-certificate match).
    Cert([u8; 32]),
}

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
    /// Certificate pins the outbound client will accept for the peers it dials.
    ///
    /// Each successful handshake must match at least one of these pins (see
    /// [`PinnedServerVerifier`]). Defaults to an empty vector, which — unless
    /// [`Self::dangerous_accept_any_cert`] is set — makes [`QuicTransport::new`]
    /// **fail closed** rather than silently trust every peer.
    pub server_pins: Vec<CertPin>,
    /// Development-only escape hatch: when `true`, the client accepts **any**
    /// server certificate and performs **no** authentication whatsoever.
    ///
    /// This re-enables full man-in-the-middle exposure and must never be set in
    /// production. Defaults to `false`.
    pub dangerous_accept_any_cert: bool,
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
            // Secure by default: no pins configured and the dangerous escape hatch
            // disabled, so `QuicTransport::new` fails closed until the caller supplies
            // the expected peer's SPKI/cert pin.
            server_pins: Vec::new(),
            dangerous_accept_any_cert: false,
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
    /// SHA-256 of this endpoint's own SPKI DER. Hand this to a peer so it can pin
    /// us with [`CertPin::Spki`]. Computed with the same helper the client verifier
    /// uses, so both sides agree byte-for-byte.
    spki_pin: [u8; 32],
    /// SHA-256 of this endpoint's own full certificate DER (see [`CertPin::Cert`]).
    cert_fingerprint: [u8; 32],
}

impl QuicTransport {
    /// Create a new QUIC transport.
    ///
    /// # Security model
    ///
    /// The outbound client authenticates every peer it dials by **SPKI/certificate
    /// pinning** (see [`PinnedServerVerifier`]) using [`QuicConfig::server_pins`].
    /// Because IPFRS certificates are ephemeral and self-signed, an empty pin set has
    /// no safe interpretation, so this constructor **fails closed**: with no pins and
    /// [`QuicConfig::dangerous_accept_any_cert`] left `false`, it returns an error
    /// rather than trusting arbitrary peers.
    ///
    /// Set [`QuicConfig::dangerous_accept_any_cert`] to `true` only in development to
    /// restore the old (insecure) "accept any certificate" behaviour.
    pub async fn new(config: QuicConfig) -> Result<Self> {
        // Explicit QUIC-capable crypto provider. quinn requires TLS 1.3 with a QUIC-capable
        // initial cipher suite (`TLS13_AES_128_GCM_SHA256`). The pure-Rust `rustls_rustcrypto`
        // provider sets `quic: None` on every one of its TLS 1.3 suites, so quinn rejects it
        // ("no initial cipher suite found") and the transport can never be built. We therefore
        // use `ring`, which is already unconditionally in this crate's dependency graph
        // (quinn-proto, libp2p-tls, rustls-webpki, snow) — selecting it here adds no new
        // dependency and is exactly what quinn itself uses internally. Building the provider
        // explicitly (instead of relying on a process-installed default) keeps this transport's
        // crypto self-contained and free of global mutable state.
        let provider = Arc::new(rustls::crypto::ring::default_provider());

        // Ephemeral self-signed certificate for this endpoint's server side.
        let (cert, key) = Self::generate_self_signed_cert()?;

        // Publish our own pins so a remote peer can pin us. Computed with the SAME
        // helpers the client verifier uses, guaranteeing client/server agree byte-for-byte.
        let spki_pin = spki_sha256(&cert)
            .map_err(|e| Error::Internal(format!("Failed to compute SPKI pin: {}", e)))?;
        let cert_fingerprint = cert_sha256(&cert);

        // Server rustls config: same explicit provider, TLS 1.3 only (mandatory for QUIC).
        let server_transport = Self::create_transport_config(&config);
        let mut server_crypto = rustls::ServerConfig::builder_with_provider(provider.clone())
            .with_protocol_versions(&[&rustls::version::TLS13])
            .map_err(|e| Error::Internal(format!("Failed to set server TLS versions: {}", e)))?
            .with_no_client_auth()
            .with_single_cert(vec![cert.clone()], key.clone_key())
            .map_err(|e| Error::Internal(format!("Failed to create server config: {}", e)))?;
        // QUIC requires the max early-data size to be exactly 0 or u32::MAX; u32::MAX enables 0-RTT.
        server_crypto.max_early_data_size = u32::MAX;
        let quic_server_crypto = quinn::crypto::rustls::QuicServerConfig::try_from(server_crypto)
            .map_err(|e| {
            Error::Internal(format!("Failed to create QUIC server config: {}", e))
        })?;
        let mut server_config = ServerConfig::with_crypto(Arc::new(quic_server_crypto));
        server_config.transport_config(Arc::new(server_transport));

        // Select the server-certificate verifier for our outbound client side.
        // Default: genuine SPKI/cert pinning. Escape hatch: accept-any (dev only), loudly logged.
        let verifier: Arc<dyn ServerCertVerifier> = if config.dangerous_accept_any_cert {
            tracing::warn!(
                "QUIC: dangerous_accept_any_cert=true — server certificates are NOT verified; \
                 this disables peer authentication and re-enables man-in-the-middle exposure"
            );
            Arc::new(SkipServerVerification)
        } else {
            Arc::new(
                PinnedServerVerifier::new(config.server_pins.clone(), &provider).map_err(|e| {
                    Error::Internal(format!(
                        "Failed to build pinned certificate verifier: {}",
                        e
                    ))
                })?,
            )
        };

        // Client rustls config: same explicit provider, TLS 1.3 only, with our pinning verifier.
        let client_transport = Self::create_transport_config(&config);
        let client_crypto = rustls::ClientConfig::builder_with_provider(provider.clone())
            .with_protocol_versions(&[&rustls::version::TLS13])
            .map_err(|e| Error::Internal(format!("Failed to set client TLS versions: {}", e)))?
            .dangerous()
            .with_custom_certificate_verifier(verifier)
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
            spki_pin,
            cert_fingerprint,
        })
    }

    /// SHA-256 of this endpoint's own `SubjectPublicKeyInfo` (DER).
    ///
    /// Give this to a remote peer so it can authenticate us with
    /// `CertPin::Spki(transport.spki_pin())`. The pin is stable across certificate
    /// re-issuance as long as this endpoint keeps the same key pair.
    pub fn spki_pin(&self) -> [u8; 32] {
        self.spki_pin
    }

    /// SHA-256 of this endpoint's own full certificate DER (exact-certificate pin).
    ///
    /// Give this to a remote peer so it can authenticate us with
    /// `CertPin::Cert(transport.cert_fingerprint())`.
    pub fn cert_fingerprint(&self) -> [u8; 32] {
        self.cert_fingerprint
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

    /// Generate an ephemeral self-signed certificate for this endpoint.
    ///
    /// The certificate carries an explicit, absolute validity window (`not_before` /
    /// `not_after`) rather than relying on wall-clock-relative defaults. This makes the
    /// client's expiry check ([`PinnedServerVerifier`] step (b)) deterministic so that
    /// loopback tests cannot flake on a clock boundary. The window is intentionally very
    /// wide because these certificates are authenticated by *key pin*, not by lifetime.
    fn generate_self_signed_cert() -> Result<(
        rustls::pki_types::CertificateDer<'static>,
        rustls::pki_types::PrivateKeyDer<'static>,
    )> {
        let mut params = rcgen::CertificateParams::new(vec!["localhost".to_string()])
            .map_err(|e| Error::Internal(format!("Failed to build certificate params: {}", e)))?;
        // Fixed, absolute window well around any realistic handshake clock.
        params.not_before = rcgen::date_time_ymd(2000, 1, 1);
        params.not_after = rcgen::date_time_ymd(4096, 1, 1);

        let signing_key = rcgen::KeyPair::generate()
            .map_err(|e| Error::Internal(format!("Failed to generate key pair: {}", e)))?;
        let rcgen_cert = params
            .self_signed(&signing_key)
            .map_err(|e| Error::Internal(format!("Failed to self-sign certificate: {}", e)))?;

        let cert_der = rustls::pki_types::CertificateDer::from(rcgen_cert.der().to_vec());
        let key_der = rustls::pki_types::PrivateKeyDer::try_from(signing_key.serialize_der())
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

/// Compute the SHA-256 of a byte slice into a fixed 32-byte array.
///
/// `Sha256::digest` yields a length-32 output, so the copy is total and infallible.
fn sha256_array(bytes: &[u8]) -> [u8; 32] {
    let digest = Sha256::digest(bytes);
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

/// SHA-256 of a certificate's `SubjectPublicKeyInfo` (SPKI) in DER form.
///
/// This is the canonical input for SPKI pinning: it hashes the full SPKI structure
/// (algorithm identifier + subject public key), so the pin survives certificate
/// re-issuance as long as the underlying key pair is unchanged. It matches the value
/// produced by `openssl x509 -pubkey | openssl pkey -pubin -outform der | sha256sum`.
///
/// Only the pure-Rust `x509-parser` DER parser is used here; its `verify` feature is
/// intentionally not required. The handshake signature is verified separately by rustls.
///
/// Returns an error if `cert` is not parseable as X.509 DER.
fn spki_sha256(cert: &CertificateDer<'_>) -> std::result::Result<[u8; 32], rustls::Error> {
    let (_, parsed) = x509_parser::parse_x509_certificate(cert.as_ref())
        .map_err(|e| rustls::Error::General(format!("failed to parse certificate DER: {}", e)))?;
    Ok(sha256_array(parsed.public_key().raw))
}

/// SHA-256 of the full end-entity certificate DER (exact-certificate fingerprint).
fn cert_sha256(cert: &CertificateDer<'_>) -> [u8; 32] {
    sha256_array(cert.as_ref())
}

/// A [`ServerCertVerifier`] that authenticates a QUIC peer by **pinning** its
/// end-entity certificate — either its public key (SPKI) or the exact certificate —
/// instead of chaining to a public certificate authority / webpki root store.
///
/// # Security model
///
/// IPFRS peers present *ephemeral, self-signed* certificates that chain to no public
/// root. Classic webpki path validation is therefore meaningless here: it would reject
/// every peer (there is no trusted anchor) or, worse, "succeed" only for certificates
/// issued by unrelated public CAs that have nothing to do with the peer we intend to
/// reach. The correct trust model for this setting is to authenticate the peer's *key*
/// directly, exactly as SSH `known_hosts` and HPKP-style pinning do.
///
/// [`Self::verify_server_cert`] enforces, in order:
/// 1. the certificate parses as X.509 DER;
/// 2. the certificate is temporally valid at the handshake time `now` (not expired,
///    not yet valid) — see [`UnixTime`];
/// 3. the SHA-256 of the SPKI DER (for [`CertPin::Spki`]) or of the whole certificate
///    (for [`CertPin::Cert`]) matches one of the configured pins.
///
/// If nothing matches, the handshake is aborted with an error (**fail closed**).
///
/// The TLS handshake signature itself is verified *for real* in
/// [`Self::verify_tls13_signature`] / [`Self::verify_tls12_signature`], which delegate
/// to rustls' webpki-backed verifiers using the installed [`CryptoProvider`]'s
/// algorithms. Both checks are required: the pin binds the *identity* of the key, and
/// the signature check proves the peer actually *holds* the matching private key. A pin
/// match without a signature check would let an attacker replay a certificate it copied
/// but does not own.
///
/// ## Why is there no hostname / SAN check?
///
/// Hostname verification answers "does this CA-issued certificate authorize the DNS
/// name I dialed?". That question is irrelevant for direct key pinning: the pin already
/// identifies *the* acceptable key with cryptographic precision, and IPFRS dials peers
/// by [`SocketAddr`], not by a certificate-bound DNS identity. The peer's certificate
/// carries a throwaway `localhost` SAN that authenticates nothing. Enforcing SAN
/// matching here would add no security while coupling us to an irrelevant,
/// attacker-influenced field. This omission is **deliberate**, not an oversight.
///
/// [`CryptoProvider`]: rustls::crypto::CryptoProvider
pub struct PinnedServerVerifier {
    /// Accepted pins. Guaranteed non-empty by [`Self::new`].
    pins: Vec<CertPin>,
    /// Signature-verification algorithms sourced from the [`CryptoProvider`] passed to
    /// [`Self::new`] (the same provider that drives the QUIC handshake).
    ///
    /// [`WebPkiSupportedAlgorithms`] is `Copy` and holds only `'static` references, so
    /// snapshotting it at construction is cheap and keeps verification self-contained.
    ///
    /// [`CryptoProvider`]: rustls::crypto::CryptoProvider
    algs: WebPkiSupportedAlgorithms,
}

impl std::fmt::Debug for PinnedServerVerifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `WebPkiSupportedAlgorithms` does not implement `Debug`; surface the schemes it
        // advertises instead of the opaque algorithm table.
        f.debug_struct("PinnedServerVerifier")
            .field("pins", &self.pins)
            .field("supported_schemes", &self.algs.supported_schemes())
            .finish()
    }
}

impl PinnedServerVerifier {
    /// Build a pinning verifier from a **non-empty** set of [`CertPin`]s.
    ///
    /// # Fail closed on an empty pin set
    ///
    /// An empty pin set is rejected with an error instead of being interpreted as
    /// "trust everything" (a MITM footgun) or "trust nothing" (a silent, confusing
    /// connection failure). Because IPFRS certificates are ephemeral and self-signed,
    /// there is no meaningful fallback — we cannot "fall back to the platform root
    /// store", since these certificates chain to no public root. Returning `Err` forces
    /// the caller to make an explicit choice: populate [`QuicConfig::server_pins`] with
    /// the expected peer's SPKI/certificate hash, or (development only) set
    /// [`QuicConfig::dangerous_accept_any_cert`].
    ///
    /// `provider` supplies the signature-verification algorithms used by
    /// [`Self::verify_tls13_signature`] / [`Self::verify_tls12_signature`] and advertised by
    /// [`Self::supported_verify_schemes`]. Pass the SAME [`CryptoProvider`] that drives the QUIC
    /// handshake so the enforced and advertised schemes match. Threading it explicitly (rather
    /// than reading a globally-installed default) avoids any dependence on process-wide state.
    ///
    /// [`CryptoProvider`]: rustls::crypto::CryptoProvider
    pub fn new(
        pins: Vec<CertPin>,
        provider: &rustls::crypto::CryptoProvider,
    ) -> std::result::Result<Self, rustls::Error> {
        if pins.is_empty() {
            return Err(rustls::Error::General(
                "PinnedServerVerifier requires at least one certificate pin: set \
                 QuicConfig::server_pins to the expected peer's SPKI/certificate hash, or \
                 (development only) set QuicConfig::dangerous_accept_any_cert = true"
                    .to_string(),
            ));
        }
        Ok(Self {
            pins,
            // `WebPkiSupportedAlgorithms` is `Copy`, so this snapshots the table from the
            // caller-supplied provider (the same provider that drives the QUIC handshake).
            algs: provider.signature_verification_algorithms,
        })
    }
}

impl ServerCertVerifier for PinnedServerVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        now: UnixTime,
    ) -> std::result::Result<ServerCertVerified, rustls::Error> {
        // (a) Parse the end-entity certificate as X.509 DER.
        let (_, parsed) =
            x509_parser::parse_x509_certificate(end_entity.as_ref()).map_err(|e| {
                rustls::Error::General(format!("failed to parse server certificate: {}", e))
            })?;

        // (b) Enforce temporal validity against the handshake clock `now`: a pin match on
        //     an expired or not-yet-valid certificate must still be rejected.
        let asn1_now = x509_parser::time::ASN1Time::from_timestamp(now.as_secs() as i64)
            .map_err(|e| rustls::Error::General(format!("invalid handshake timestamp: {}", e)))?;
        if !parsed.validity().is_valid_at(asn1_now) {
            return Err(rustls::Error::General(
                "server certificate is expired or not yet valid".to_string(),
            ));
        }

        // (c) Compute the SPKI and full-certificate fingerprints via the shared helpers —
        //     the exact code path the server side uses to publish its pins, so both ends
        //     compare identical bytes. (The cheap re-parse inside `spki_sha256` keeps a
        //     single source of truth for SPKI extraction.)
        let spki_hash = spki_sha256(end_entity)?;
        let cert_hash = cert_sha256(end_entity);

        // (d) Accept only if some configured pin matches; otherwise fail closed.
        let matched = self.pins.iter().any(|pin| match pin {
            CertPin::Spki(expected) => *expected == spki_hash,
            CertPin::Cert(expected) => *expected == cert_hash,
        });
        if !matched {
            return Err(rustls::Error::General(
                "server certificate pin mismatch".to_string(),
            ));
        }

        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        // Real signature verification against the peer's public key — never a blanket
        // assertion. Proves the peer holds the private key for the pinned certificate.
        rustls::crypto::verify_tls12_signature(message, cert, dss, &self.algs)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        // Real signature verification against the peer's public key — never a blanket
        // assertion. QUIC always uses TLS 1.3, so this is the hot path.
        rustls::crypto::verify_tls13_signature(message, cert, dss, &self.algs)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.algs.supported_schemes()
    }
}

/// **Development-only** certificate verifier that disables all authentication.
///
/// Every method returns success without inspecting the certificate or verifying any
/// handshake signature, so a client using it accepts **any** certificate from **any**
/// peer — full man-in-the-middle exposure. It is reachable only when
/// [`QuicConfig::dangerous_accept_any_cert`] is explicitly set to `true`, and its use
/// is logged at `warn`. Never enable it in production; prefer [`PinnedServerVerifier`].
#[derive(Debug)]
struct SkipServerVerification;

impl ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> std::result::Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        vec![
            SignatureScheme::RSA_PKCS1_SHA256,
            SignatureScheme::ECDSA_NISTP256_SHA256,
            SignatureScheme::RSA_PKCS1_SHA384,
            SignatureScheme::ECDSA_NISTP384_SHA384,
            SignatureScheme::RSA_PKCS1_SHA512,
            SignatureScheme::ECDSA_NISTP521_SHA512,
            SignatureScheme::RSA_PSS_SHA256,
            SignatureScheme::RSA_PSS_SHA384,
            SignatureScheme::RSA_PSS_SHA512,
            SignatureScheme::ED25519,
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

    /// Build a loopback-bound config with the given pinning policy. `bind_addr` is
    /// `127.0.0.1:0` so `local_addr()` yields a routable loopback address the peer can dial.
    fn loopback_config(server_pins: Vec<CertPin>, dangerous_accept_any_cert: bool) -> QuicConfig {
        QuicConfig {
            bind_addr: "127.0.0.1:0"
                .parse()
                .expect("loopback socket addr literal must parse"),
            server_pins,
            dangerous_accept_any_cert,
            ..QuicConfig::default()
        }
    }

    /// A non-empty dummy pin for a transport whose *client* side is never exercised
    /// (e.g. a pure echo server). It satisfies the fail-closed constructor without
    /// reaching for the dangerous escape hatch; it is never actually matched.
    fn unused_server_side_pin() -> Vec<CertPin> {
        vec![CertPin::Cert([0u8; 32])]
    }

    /// Accept exactly one connection and echo the first bidirectional stream's payload,
    /// then hold the connection open until the client closes so the echo is delivered.
    async fn echo_once(transport: Arc<QuicTransport>) -> Result<()> {
        let conn = transport
            .accept()
            .await?
            .ok_or_else(|| Error::Internal("no incoming connection".to_string()))?;
        let (mut send, mut recv) = conn
            .accept_bi()
            .await
            .map_err(|e| Error::Internal(format!("accept_bi failed: {}", e)))?;
        let req = recv
            .read_to_end(64 * 1024)
            .await
            .map_err(|e| Error::Internal(format!("read failed: {}", e)))?;
        send.write_all(&req)
            .await
            .map_err(|e| Error::Internal(format!("write failed: {}", e)))?;
        send.finish()
            .map_err(|e| Error::Internal(format!("finish failed: {}", e)))?;
        // Keep the connection alive until the client consumes the echo and closes, so the
        // finished stream is actually delivered before this endpoint is dropped.
        conn.closed().await;
        Ok(())
    }

    /// Drive a single request/echo round-trip from the client side.
    async fn client_round_trip(client: &QuicTransport, addr: SocketAddr, payload: &[u8]) {
        let conn = client.connect(addr).await.expect("client connect");
        let (mut send, mut recv) = client.open_stream(&conn).await.expect("open stream");
        client.send(&mut send, payload).await.expect("send payload");
        let response = client.receive(&mut recv).await.expect("receive echo");
        assert_eq!(response, payload, "echoed payload must match");
        conn.close(0u32.into(), b"done");
    }

    #[test]
    fn test_quic_config_defaults() {
        let config = QuicConfig::default();
        assert_eq!(config.max_streams, 256);
        assert!(config.enable_0rtt);
        assert_eq!(config.pool_size, 4);
        // Secure-by-default: no pins, escape hatch disabled -> `new` fails closed.
        assert!(config.server_pins.is_empty());
        assert!(!config.dangerous_accept_any_cert);
    }

    #[test]
    fn test_peer_pool() {
        // Note: Full integration tests would require actual QUIC connections
        let pool = PeerPool::new(4, Duration::from_secs(60));
        assert_eq!(pool.connection_count(), 0);
    }

    /// The client pins the server's real SPKI: the handshake and a message round-trip
    /// both succeed. Exercises the full happy path including real signature verification.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spki_pin_roundtrip_ok() {
        let server = Arc::new(
            QuicTransport::new(loopback_config(unused_server_side_pin(), false))
                .await
                .expect("server transport"),
        );
        let server_addr = server.local_addr().expect("server local addr");
        let server_pin = server.spki_pin();

        let client = QuicTransport::new(loopback_config(vec![CertPin::Spki(server_pin)], false))
            .await
            .expect("client transport");

        let server_task = tokio::spawn(echo_once(server.clone()));

        tokio::time::timeout(
            Duration::from_secs(10),
            client_round_trip(&client, server_addr, b"ping-roundtrip"),
        )
        .await
        .expect("round-trip timed out");

        let _ = server_task.await;
    }

    /// The client pins a hash that does not match the server's SPKI: the pin check fails,
    /// the certificate is rejected, and `connect` returns an error (no MITM slips through).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pin_mismatch_rejected() {
        let server = Arc::new(
            QuicTransport::new(loopback_config(unused_server_side_pin(), false))
                .await
                .expect("server transport"),
        );
        let server_addr = server.local_addr().expect("server local addr");

        // A pin that cannot match any real certificate's SPKI hash.
        let client = QuicTransport::new(loopback_config(vec![CertPin::Spki([0u8; 32])], false))
            .await
            .expect("client transport");

        // The server must be actively accepting so the handshake advances to certificate
        // verification (rather than the client merely timing out with no server response).
        let server_task = tokio::spawn(async move {
            let _ = echo_once(server).await;
        });

        let outcome =
            tokio::time::timeout(Duration::from_secs(10), client.connect(server_addr)).await;
        match outcome {
            Ok(result) => assert!(
                result.is_err(),
                "connect must fail when the server certificate pin does not match"
            ),
            Err(_elapsed) => panic!("connect neither succeeded nor failed within the timeout"),
        }

        server_task.abort();
    }

    /// With no pins and `dangerous_accept_any_cert = false`, `QuicTransport::new` refuses
    /// to build a client that would trust arbitrary peers: it fails closed.
    #[tokio::test]
    async fn empty_pins_fail_closed() {
        let result = QuicTransport::new(loopback_config(vec![], false)).await;
        assert!(
            result.is_err(),
            "empty pins with dangerous_accept_any_cert=false must fail closed"
        );
    }

    /// The dangerous escape hatch (`dangerous_accept_any_cert = true`) accepts any
    /// certificate with no pins configured. This documents the opt-in gate; never use it
    /// in production.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dangerous_flag_accepts_any() {
        let server = Arc::new(
            QuicTransport::new(loopback_config(unused_server_side_pin(), false))
                .await
                .expect("server transport"),
        );
        let server_addr = server.local_addr().expect("server local addr");

        // No pins, but the dangerous flag is set: the client accepts the server cert.
        let client = QuicTransport::new(loopback_config(vec![], true))
            .await
            .expect("client transport");

        let server_task = tokio::spawn(echo_once(server.clone()));

        tokio::time::timeout(
            Duration::from_secs(10),
            client_round_trip(&client, server_addr, b"dangerous-but-connected"),
        )
        .await
        .expect("dangerous round-trip timed out");

        let _ = server_task.await;
    }

    /// Cross-check the SPKI extraction: `spki_sha256(cert)` must equal the SHA-256 of the
    /// canonical `SubjectPublicKeyInfo` DER that rcgen emits for the same key. This proves
    /// the bytes x509-parser extracts from the certificate are the canonical SPKI bytes.
    #[test]
    fn spki_extraction_matches_rcgen() {
        use rcgen::PublicKeyData;

        let certified = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
            .expect("generate self-signed cert");
        let cert_der = CertificateDer::from(certified.cert.der().to_vec());

        // Canonical SPKI DER straight from rcgen's key material.
        let rcgen_spki_der = certified.signing_key.subject_public_key_info();
        let expected = sha256_array(&rcgen_spki_der);

        let extracted = spki_sha256(&cert_der).expect("extract SPKI hash");
        assert_eq!(
            extracted, expected,
            "spki_sha256(cert) must equal SHA-256 of rcgen's canonical SPKI DER"
        );
        // The SPKI pin and the full-certificate fingerprint hash different inputs.
        assert_ne!(
            extracted,
            cert_sha256(&cert_der),
            "SPKI hash must differ from the full-certificate fingerprint"
        );
    }

    /// A QUIC-capable crypto provider for the direct verifier unit tests — the same provider
    /// `QuicTransport::new` uses. These tests need only its signature-algorithm table.
    fn test_provider() -> rustls::crypto::CryptoProvider {
        rustls::crypto::ring::default_provider()
    }

    /// Generate a self-signed certificate valid at "now", returning its DER plus its SPKI and
    /// full-certificate SHA-256 pins (computed with the production helpers).
    fn make_test_cert() -> (CertificateDer<'static>, [u8; 32], [u8; 32]) {
        let mut params = rcgen::CertificateParams::new(vec!["localhost".to_string()])
            .expect("certificate params");
        params.not_before = rcgen::date_time_ymd(2000, 1, 1);
        params.not_after = rcgen::date_time_ymd(4096, 1, 1);
        let signing_key = rcgen::KeyPair::generate().expect("key pair");
        let cert = params.self_signed(&signing_key).expect("self-sign");
        let cert_der = CertificateDer::from(cert.der().to_vec());
        let spki = spki_sha256(&cert_der).expect("spki hash");
        let fingerprint = cert_sha256(&cert_der);
        (cert_der, spki, fingerprint)
    }

    /// Direct verifier unit test (no live handshake): a matching SPKI pin verifies.
    #[test]
    fn verifier_accepts_matching_spki_pin() {
        let provider = test_provider();
        let (cert_der, spki, _fingerprint) = make_test_cert();
        let verifier =
            PinnedServerVerifier::new(vec![CertPin::Spki(spki)], &provider).expect("verifier");
        let server_name = ServerName::try_from("localhost").expect("server name");
        if let Err(e) =
            verifier.verify_server_cert(&cert_der, &[], &server_name, &[], UnixTime::now())
        {
            panic!("a matching SPKI pin must verify, got error: {}", e);
        }
    }

    /// Direct verifier unit test: a mismatched pin is rejected (fail closed).
    #[test]
    fn verifier_rejects_mismatched_pin() {
        let provider = test_provider();
        let (cert_der, _spki, _fingerprint) = make_test_cert();
        let verifier =
            PinnedServerVerifier::new(vec![CertPin::Spki([0u8; 32])], &provider).expect("verifier");
        let server_name = ServerName::try_from("localhost").expect("server name");
        let result =
            verifier.verify_server_cert(&cert_der, &[], &server_name, &[], UnixTime::now());
        assert!(result.is_err(), "a mismatched pin must be rejected");
    }

    /// Direct verifier unit test: a matching full-certificate fingerprint (`CertPin::Cert`) verifies.
    #[test]
    fn verifier_accepts_matching_cert_fingerprint() {
        let provider = test_provider();
        let (cert_der, _spki, fingerprint) = make_test_cert();
        let verifier = PinnedServerVerifier::new(vec![CertPin::Cert(fingerprint)], &provider)
            .expect("verifier");
        let server_name = ServerName::try_from("localhost").expect("server name");
        if let Err(e) =
            verifier.verify_server_cert(&cert_der, &[], &server_name, &[], UnixTime::now())
        {
            panic!(
                "a matching full-certificate fingerprint must verify, got error: {}",
                e
            );
        }
    }

    /// An expired certificate is rejected even when its pin matches: the temporal-validity
    /// check ((b)) runs independently of the pin check ((d)).
    #[test]
    fn expired_certificate_rejected() {
        let provider = test_provider();

        // A certificate whose validity window lies entirely in the past.
        let mut params = rcgen::CertificateParams::new(vec!["localhost".to_string()])
            .expect("certificate params");
        params.not_before = rcgen::date_time_ymd(2000, 1, 1);
        params.not_after = rcgen::date_time_ymd(2001, 1, 1);
        let signing_key = rcgen::KeyPair::generate().expect("key pair");
        let cert = params.self_signed(&signing_key).expect("self-sign");
        let cert_der = CertificateDer::from(cert.der().to_vec());

        // Pin the real SPKI so ONLY the expiry check can reject the certificate.
        let spki = spki_sha256(&cert_der).expect("spki hash");
        let verifier = PinnedServerVerifier::new(vec![CertPin::Spki(spki)], &provider)
            .expect("build pinned verifier");

        let now = UnixTime::now();
        let server_name = ServerName::try_from("localhost").expect("server name");
        let result = verifier.verify_server_cert(&cert_der, &[], &server_name, &[], now);
        assert!(
            result.is_err(),
            "an expired certificate must be rejected even when its pin matches"
        );
    }

    /// A `PinnedServerVerifier` built with an empty pin set is rejected at construction —
    /// the same fail-closed guarantee `QuicTransport::new` relies on.
    #[test]
    fn pinned_verifier_rejects_empty_pins() {
        let provider = test_provider();
        assert!(
            PinnedServerVerifier::new(vec![], &provider).is_err(),
            "PinnedServerVerifier::new must reject an empty pin set"
        );
    }
}
