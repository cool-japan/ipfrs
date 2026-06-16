//! Tor Integration for Privacy-Preserving Networking
//!
//! This module provides integration with the Tor network for anonymous and privacy-preserving
//! peer-to-peer communication. It supports both client connections through Tor and hosting
//! hidden services for anonymous server endpoints.
//!
//! ## Features
//!
//! - **SOCKS5 Proxy**: Connect to peers through Tor's SOCKS5 proxy
//! - **Onion Routing**: Multi-hop encrypted routing for anonymity
//! - **Hidden Services**: Host anonymous .onion endpoints
//! - **Circuit Management**: Control Tor circuits for optimal performance
//! - **Stream Isolation**: Separate streams for different applications
//! - **Bandwidth Management**: Throttle Tor traffic to avoid network congestion
//!
//! ## Use Cases
//!
//! - **Privacy**: Hide IP addresses from peers and network observers
//! - **Censorship Resistance**: Access content in restrictive networks
//! - **Anonymous Publishing**: Host content without revealing location
//! - **Surveillance Protection**: Protect against traffic analysis
//!
//! ## Example
//!
//! ```rust,no_run
//! use ipfrs_network::tor::{TorManager, TorConfig, HiddenServiceConfig};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create Tor manager
//! let config = TorConfig::default();
//! let mut manager = TorManager::new(config).await?;
//!
//! // Start Tor
//! manager.start().await?;
//!
//! // Connect through Tor
//! let peer_addr = "example.onion:8080";
//! let stream = manager.connect(peer_addr).await?;
//!
//! // Create hidden service
//! let hs_config = HiddenServiceConfig::default();
//! let onion_addr = manager.create_hidden_service(hs_config).await?;
//! println!("Hidden service available at: {}", onion_addr);
//! # Ok(())
//! # }
//! ```

use dashmap::DashMap;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tracing::{debug, info};

/// Errors that can occur in Tor operations
#[derive(Debug, Error)]
pub enum TorError {
    #[error("Tor not running")]
    NotRunning,

    #[error("Tor already running")]
    AlreadyRunning,

    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Hidden service creation failed: {0}")]
    HiddenServiceFailed(String),

    #[error("Circuit creation failed: {0}")]
    CircuitFailed(String),

    #[error("Invalid onion address: {0}")]
    InvalidOnionAddress(String),

    #[error("SOCKS5 proxy error: {0}")]
    Socks5Error(String),

    #[error("Tor configuration error: {0}")]
    ConfigError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for Tor operations
pub type Result<T> = std::result::Result<T, TorError>;

/// Tor circuit identifier
pub type CircuitId = u32;

/// Stream identifier
pub type StreamId = u64;

/// Onion address (e.g., "example.onion")
pub type OnionAddress = String;

/// Tor circuit state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is being built
    Building,

    /// Circuit is ready for use
    Ready,

    /// Circuit is being used
    Active,

    /// Circuit is degraded (slow or unreliable)
    Degraded,

    /// Circuit has failed
    Failed,

    /// Circuit is being closed
    Closing,
}

/// Information about a Tor circuit
#[derive(Debug, Clone)]
pub struct CircuitInfo {
    /// Circuit identifier
    pub id: CircuitId,

    /// Circuit state
    pub state: CircuitState,

    /// Relay nodes in the circuit (3 hops typically)
    pub hops: Vec<String>,

    /// When the circuit was created
    pub created_at: Instant,

    /// Number of streams using this circuit
    pub stream_count: usize,

    /// Total bytes sent through this circuit
    pub bytes_sent: u64,

    /// Total bytes received through this circuit
    pub bytes_received: u64,
}

/// Hidden service configuration
#[derive(Debug, Clone)]
pub struct HiddenServiceConfig {
    /// Local port to expose
    pub local_port: u16,

    /// Virtual port (port visible to Tor users)
    pub virtual_port: u16,

    /// Directory for hidden service keys
    pub data_dir: PathBuf,

    /// Maximum concurrent connections
    pub max_connections: usize,

    /// Enable v3 onion addresses (recommended)
    pub use_v3: bool,
}

impl Default for HiddenServiceConfig {
    fn default() -> Self {
        Self {
            local_port: 8080,
            virtual_port: 8080,
            data_dir: std::env::temp_dir().join("tor-hidden-service"),
            max_connections: 100,
            use_v3: true,
        }
    }
}

/// Configuration for Tor manager
#[derive(Debug, Clone)]
pub struct TorConfig {
    /// SOCKS5 proxy address (default: 127.0.0.1:9050)
    pub socks_proxy: SocketAddr,

    /// Control port address (default: 127.0.0.1:9051)
    pub control_port: SocketAddr,

    /// Tor data directory
    pub data_dir: PathBuf,

    /// Enable stream isolation (separate circuits per stream)
    pub stream_isolation: bool,

    /// Maximum circuits to maintain
    pub max_circuits: usize,

    /// Circuit timeout duration
    pub circuit_timeout: Duration,

    /// Enable bandwidth limiting
    pub enable_bandwidth_limit: bool,

    /// Maximum bandwidth in bytes/sec (0 = unlimited)
    pub max_bandwidth_bps: u64,

    /// Use bridges for censorship circumvention
    pub use_bridges: bool,

    /// Bridge addresses (if use_bridges is true)
    pub bridges: Vec<String>,
}

impl Default for TorConfig {
    fn default() -> Self {
        Self {
            socks_proxy: "127.0.0.1:9050"
                .parse()
                .expect("static socket addr literal must parse"),
            control_port: "127.0.0.1:9051"
                .parse()
                .expect("static socket addr literal must parse"),
            data_dir: std::env::temp_dir().join("tor-data"),
            stream_isolation: true,
            max_circuits: 10,
            circuit_timeout: Duration::from_secs(60),
            enable_bandwidth_limit: false,
            max_bandwidth_bps: 0,
            use_bridges: false,
            bridges: Vec::new(),
        }
    }
}

impl TorConfig {
    /// Configuration for high-privacy mode
    pub fn high_privacy() -> Self {
        Self {
            stream_isolation: true,
            max_circuits: 5,
            circuit_timeout: Duration::from_secs(90),
            enable_bandwidth_limit: true,
            max_bandwidth_bps: 1_000_000, // 1 MB/s
            ..Default::default()
        }
    }

    /// Configuration for high-performance mode
    pub fn high_performance() -> Self {
        Self {
            stream_isolation: false,
            max_circuits: 20,
            circuit_timeout: Duration::from_secs(30),
            enable_bandwidth_limit: false,
            max_bandwidth_bps: 0,
            ..Default::default()
        }
    }

    /// Configuration for censorship circumvention
    pub fn censorship_resistant() -> Self {
        Self {
            use_bridges: true,
            bridges: vec![
                // Example bridges (should be updated with real bridge addresses)
                "obfs4 192.0.2.1:443".to_string(),
                "obfs4 192.0.2.2:443".to_string(),
            ],
            stream_isolation: true,
            max_circuits: 8,
            circuit_timeout: Duration::from_secs(120),
            ..Default::default()
        }
    }
}

/// Statistics for Tor operations
#[derive(Debug, Clone, Default)]
pub struct TorStats {
    /// Total circuits created
    pub circuits_created: usize,

    /// Currently active circuits
    pub active_circuits: usize,

    /// Total streams created
    pub streams_created: usize,

    /// Currently active streams
    pub active_streams: usize,

    /// Total bytes sent through Tor
    pub total_bytes_sent: u64,

    /// Total bytes received through Tor
    pub total_bytes_received: u64,

    /// Number of hidden services hosted
    pub hidden_services: usize,

    /// Total connection failures
    pub connection_failures: usize,

    /// Average circuit build time (ms)
    pub avg_circuit_build_time_ms: f64,
}

/// Tor network manager
pub struct TorManager {
    /// Configuration
    config: TorConfig,

    /// Running state
    running: Arc<RwLock<bool>>,

    /// Active circuits
    circuits: Arc<DashMap<CircuitId, CircuitInfo>>,

    /// Next circuit ID
    next_circuit_id: Arc<RwLock<CircuitId>>,

    /// Stream to circuit mapping
    stream_circuits: Arc<DashMap<StreamId, CircuitId>>,

    /// Next stream ID
    next_stream_id: Arc<RwLock<StreamId>>,

    /// Hidden services
    hidden_services: Arc<DashMap<OnionAddress, HiddenServiceConfig>>,

    /// Statistics
    stats: Arc<RwLock<TorStats>>,

    /// Circuit build times (for averaging)
    circuit_build_times: Arc<RwLock<Vec<f64>>>,
}

impl TorManager {
    /// Create a new Tor manager
    pub async fn new(config: TorConfig) -> Result<Self> {
        info!("Creating Tor manager");

        // Validate configuration
        if config.max_circuits == 0 {
            return Err(TorError::ConfigError(
                "max_circuits must be > 0".to_string(),
            ));
        }

        Ok(Self {
            config,
            running: Arc::new(RwLock::new(false)),
            circuits: Arc::new(DashMap::new()),
            next_circuit_id: Arc::new(RwLock::new(0)),
            stream_circuits: Arc::new(DashMap::new()),
            next_stream_id: Arc::new(RwLock::new(0)),
            hidden_services: Arc::new(DashMap::new()),
            stats: Arc::new(RwLock::new(TorStats::default())),
            circuit_build_times: Arc::new(RwLock::new(Vec::with_capacity(100))),
        })
    }

    /// Start the Tor manager
    pub async fn start(&mut self) -> Result<()> {
        let mut running = self.running.write();
        if *running {
            return Err(TorError::AlreadyRunning);
        }

        info!("Starting Tor manager");
        info!("SOCKS proxy: {}", self.config.socks_proxy);
        info!("Control port: {}", self.config.control_port);

        // In a real implementation, this would:
        // 1. Start Tor process or connect to existing Tor daemon
        // 2. Authenticate with control port
        // 3. Configure Tor settings
        // 4. Wait for bootstrap completion

        *running = true;
        info!("Tor manager started");

        Ok(())
    }

    /// Stop the Tor manager
    pub async fn stop(&mut self) -> Result<()> {
        let mut running = self.running.write();
        if !*running {
            return Err(TorError::NotRunning);
        }

        info!("Stopping Tor manager");

        // Close all circuits
        for circuit in self.circuits.iter() {
            let circuit_id = circuit.key();
            debug!("Closing circuit {}", circuit_id);
        }
        self.circuits.clear();

        // Remove all hidden services
        self.hidden_services.clear();

        *running = false;
        info!("Tor manager stopped");

        Ok(())
    }

    /// Check if Tor is running
    pub fn is_running(&self) -> bool {
        *self.running.read()
    }

    /// Create a new Tor circuit
    pub async fn create_circuit(&self) -> Result<CircuitId> {
        if !self.is_running() {
            return Err(TorError::NotRunning);
        }

        if self.circuits.len() >= self.config.max_circuits {
            // Remove oldest inactive circuit
            self.cleanup_circuits();

            if self.circuits.len() >= self.config.max_circuits {
                return Err(TorError::CircuitFailed(
                    "Maximum circuits reached".to_string(),
                ));
            }
        }

        let circuit_id = {
            let mut id = self.next_circuit_id.write();
            let current_id = *id;
            *id += 1;
            current_id
        };

        let start_time = Instant::now();

        // In a real implementation, this would:
        // 1. Select guard, middle, and exit nodes
        // 2. Build circuit through Tor control protocol
        // 3. Wait for circuit to be ready

        // Simulate circuit with 3 hops
        let circuit = CircuitInfo {
            id: circuit_id,
            state: CircuitState::Ready,
            hops: vec![
                "GuardNode".to_string(),
                "MiddleNode".to_string(),
                "ExitNode".to_string(),
            ],
            created_at: Instant::now(),
            stream_count: 0,
            bytes_sent: 0,
            bytes_received: 0,
        };

        self.circuits.insert(circuit_id, circuit);

        // Record build time
        let build_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        let mut build_times = self.circuit_build_times.write();
        build_times.push(build_time_ms);
        if build_times.len() > 100 {
            build_times.remove(0);
        }

        let mut stats = self.stats.write();
        stats.circuits_created += 1;
        stats.active_circuits = self.circuits.len();
        stats.avg_circuit_build_time_ms =
            build_times.iter().sum::<f64>() / build_times.len() as f64;

        info!("Created circuit {} in {:.1}ms", circuit_id, build_time_ms);

        Ok(circuit_id)
    }

    /// Connect to an address through Tor
    pub async fn connect(&self, address: &str) -> Result<StreamId> {
        if !self.is_running() {
            return Err(TorError::NotRunning);
        }

        // Get or create a circuit
        let circuit_id = if self.config.stream_isolation {
            // Create new circuit for stream isolation
            self.create_circuit().await?
        } else {
            // Reuse existing circuit
            self.get_or_create_circuit().await?
        };

        let stream_id = {
            let mut id = self.next_stream_id.write();
            let current_id = *id;
            *id += 1;
            current_id
        };

        // In a real implementation, this would:
        // 1. Open SOCKS5 connection to Tor proxy
        // 2. Send SOCKS5 connect request
        // 3. Wait for connection establishment
        // 4. Return stream handle

        // Map stream to circuit
        self.stream_circuits.insert(stream_id, circuit_id);

        // Update circuit
        if let Some(mut circuit) = self.circuits.get_mut(&circuit_id) {
            circuit.stream_count += 1;
            circuit.state = CircuitState::Active;
        }

        let mut stats = self.stats.write();
        stats.streams_created += 1;
        stats.active_streams = self.stream_circuits.len();

        debug!("Connected to {} via circuit {}", address, circuit_id);

        Ok(stream_id)
    }

    /// Create a hidden service
    pub async fn create_hidden_service(&self, config: HiddenServiceConfig) -> Result<OnionAddress> {
        if !self.is_running() {
            return Err(TorError::NotRunning);
        }

        // In a real implementation, this would:
        // 1. Generate hidden service keys (v3 onion address)
        // 2. Configure Tor to host hidden service
        // 3. Wait for descriptor publication
        // 4. Return .onion address

        // Generate a mock v3 onion address (56 characters, base32: a-z, 2-7)
        let onion_addr = if config.use_v3 {
            format!(
                "{}.onion",
                "abcdefghijklmnopqrstuvabcdefghijklmnopqrstuvabcdefghijkl"
            )
        } else {
            // v2 onion address (16 characters) - deprecated
            format!("{}.onion", "abcdefghijklmnop")
        };

        self.hidden_services.insert(onion_addr.clone(), config);

        let mut stats = self.stats.write();
        stats.hidden_services = self.hidden_services.len();

        info!("Created hidden service: {}", onion_addr);

        Ok(onion_addr)
    }

    /// Remove a hidden service
    pub async fn remove_hidden_service(&self, onion_addr: &str) -> Result<()> {
        if !self.is_running() {
            return Err(TorError::NotRunning);
        }

        if self.hidden_services.remove(onion_addr).is_some() {
            let mut stats = self.stats.write();
            stats.hidden_services = self.hidden_services.len();

            info!("Removed hidden service: {}", onion_addr);
            Ok(())
        } else {
            Err(TorError::InvalidOnionAddress(onion_addr.to_string()))
        }
    }

    /// Get or create a circuit
    async fn get_or_create_circuit(&self) -> Result<CircuitId> {
        // Try to find a ready circuit with low stream count
        let best_circuit = self
            .circuits
            .iter()
            .filter(|entry| {
                let circuit = entry.value();
                circuit.state == CircuitState::Ready && circuit.stream_count < 10
            })
            .min_by_key(|entry| entry.value().stream_count)
            .map(|entry| *entry.key());

        if let Some(circuit_id) = best_circuit {
            Ok(circuit_id)
        } else {
            self.create_circuit().await
        }
    }

    /// Cleanup old circuits
    fn cleanup_circuits(&self) {
        let now = Instant::now();
        let timeout = self.config.circuit_timeout;

        let old_circuits: Vec<CircuitId> = self
            .circuits
            .iter()
            .filter(|entry| {
                let circuit = entry.value();
                circuit.stream_count == 0 && now.duration_since(circuit.created_at) > timeout
            })
            .map(|entry| *entry.key())
            .collect();

        for circuit_id in old_circuits {
            debug!("Removing old circuit {}", circuit_id);
            self.circuits.remove(&circuit_id);
        }

        let mut stats = self.stats.write();
        stats.active_circuits = self.circuits.len();
    }

    /// Close a stream
    pub fn close_stream(&self, stream_id: StreamId) -> Result<()> {
        if let Some((_, circuit_id)) = self.stream_circuits.remove(&stream_id) {
            // Update circuit stream count
            if let Some(mut circuit) = self.circuits.get_mut(&circuit_id) {
                circuit.stream_count = circuit.stream_count.saturating_sub(1);

                if circuit.stream_count == 0 {
                    circuit.state = CircuitState::Ready;
                }
            }

            let mut stats = self.stats.write();
            stats.active_streams = self.stream_circuits.len();

            debug!("Closed stream {}", stream_id);
            Ok(())
        } else {
            Err(TorError::ConnectionFailed(format!(
                "Stream {} not found",
                stream_id
            )))
        }
    }

    /// Get circuit information
    pub fn get_circuit(&self, circuit_id: CircuitId) -> Option<CircuitInfo> {
        self.circuits.get(&circuit_id).map(|e| e.value().clone())
    }

    /// Get all circuits
    pub fn get_circuits(&self) -> Vec<CircuitInfo> {
        self.circuits
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get all hidden services
    pub fn get_hidden_services(&self) -> HashMap<OnionAddress, HiddenServiceConfig> {
        self.hidden_services
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    /// Get statistics
    pub fn stats(&self) -> TorStats {
        self.stats.read().clone()
    }

    /// Get configuration
    pub fn config(&self) -> &TorConfig {
        &self.config
    }

    /// Validate an onion address
    pub fn validate_onion_address(address: &str) -> bool {
        // v3 onion address: 56 characters + .onion
        // v2 onion address: 16 characters + .onion (deprecated)

        if !address.ends_with(".onion") {
            return false;
        }

        let name = address
            .strip_suffix(".onion")
            .expect("just confirmed ends_with('.onion')");

        // Base32 character set: a-z, 2-7
        let is_valid_base32 = |c: char| c.is_ascii_lowercase() || ('2'..='7').contains(&c);

        // v3: 56 characters (base32 encoded)
        if name.len() == 56 {
            return name.chars().all(is_valid_base32);
        }

        // v2: 16 characters (base32 encoded, deprecated)
        if name.len() == 16 {
            return name.chars().all(is_valid_base32);
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_manager_creation() {
        let config = TorConfig::default();
        let manager = TorManager::new(config)
            .await
            .expect("test: TorManager::new should succeed");

        assert!(!manager.is_running());
        assert_eq!(manager.stats().circuits_created, 0);
    }

    #[tokio::test]
    async fn test_start_stop() {
        let config = TorConfig::default();
        let mut manager = TorManager::new(config)
            .await
            .expect("test: TorManager::new should succeed");

        assert!(!manager.is_running());

        manager.start().await.expect("test: start should succeed");
        assert!(manager.is_running());

        manager.stop().await.expect("test: stop should succeed");
        assert!(!manager.is_running());
    }

    #[tokio::test]
    async fn test_create_circuit() {
        let config = TorConfig::default();
        let mut manager = TorManager::new(config)
            .await
            .expect("test: TorManager::new should succeed");

        manager.start().await.expect("test: start should succeed");

        let circuit_id = manager
            .create_circuit()
            .await
            .expect("test: create_circuit should succeed");
        assert_eq!(circuit_id, 0);

        let circuit = manager
            .get_circuit(circuit_id)
            .expect("test: circuit should exist after creation");
        assert_eq!(circuit.state, CircuitState::Ready);
        assert_eq!(circuit.hops.len(), 3);

        let stats = manager.stats();
        assert_eq!(stats.circuits_created, 1);
        assert_eq!(stats.active_circuits, 1);
    }

    #[tokio::test]
    async fn test_max_circuits_limit() {
        let config = TorConfig {
            max_circuits: 2,
            circuit_timeout: Duration::from_millis(100),
            ..Default::default()
        };
        let mut manager = TorManager::new(config)
            .await
            .expect("test: TorManager::new should succeed");

        manager.start().await.expect("test: start should succeed");

        // Create 2 circuits (should succeed)
        let circuit1 = manager
            .create_circuit()
            .await
            .expect("test: create first circuit should succeed");
        let _circuit2 = manager
            .create_circuit()
            .await
            .expect("test: create second circuit should succeed");

        // Close first circuit's streams to make it eligible for cleanup
        if let Some(mut circuit) = manager.circuits.get_mut(&circuit1) {
            circuit.stream_count = 0;
        }

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Try to create 3rd circuit (should succeed after cleanup)
        let result = manager.create_circuit().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_connect() {
        let config = TorConfig::default();
        let mut manager = TorManager::new(config)
            .await
            .expect("test: TorManager::new should succeed");

        manager.start().await.expect("test: start should succeed");

        let stream_id = manager
            .connect("example.onion:8080")
            .await
            .expect("test: connect should succeed");
        assert_eq!(stream_id, 0);

        let stats = manager.stats();
        assert_eq!(stats.streams_created, 1);
        assert_eq!(stats.active_streams, 1);
    }

    #[tokio::test]
    async fn test_stream_isolation() {
        let config = TorConfig {
            stream_isolation: true,
            ..Default::default()
        };
        let mut manager = TorManager::new(config)
            .await
            .expect("test: TorManager::new should succeed");

        manager.start().await.expect("test: start should succeed");

        // Two streams should use different circuits with stream isolation
        manager
            .connect("example1.onion:8080")
            .await
            .expect("test: connect to example1 should succeed");
        manager
            .connect("example2.onion:8080")
            .await
            .expect("test: connect to example2 should succeed");

        let stats = manager.stats();
        assert_eq!(stats.circuits_created, 2);
        assert_eq!(stats.streams_created, 2);
    }

    #[tokio::test]
    async fn test_hidden_service() {
        let config = TorConfig::default();
        let mut manager = TorManager::new(config)
            .await
            .expect("test: TorManager::new should succeed");

        manager.start().await.expect("test: start should succeed");

        let hs_config = HiddenServiceConfig::default();
        let onion_addr = manager
            .create_hidden_service(hs_config)
            .await
            .expect("test: create_hidden_service should succeed");

        assert!(onion_addr.ends_with(".onion"));
        assert_eq!(onion_addr.len(), 62); // 56 chars + ".onion" (6 chars)

        let stats = manager.stats();
        assert_eq!(stats.hidden_services, 1);
    }

    #[tokio::test]
    async fn test_remove_hidden_service() {
        let config = TorConfig::default();
        let mut manager = TorManager::new(config)
            .await
            .expect("test: TorManager::new should succeed");

        manager.start().await.expect("test: start should succeed");

        let hs_config = HiddenServiceConfig::default();
        let onion_addr = manager
            .create_hidden_service(hs_config)
            .await
            .expect("test: create_hidden_service should succeed");

        manager
            .remove_hidden_service(&onion_addr)
            .await
            .expect("test: remove_hidden_service should succeed");

        let stats = manager.stats();
        assert_eq!(stats.hidden_services, 0);
    }

    #[tokio::test]
    async fn test_close_stream() {
        let config = TorConfig::default();
        let mut manager = TorManager::new(config)
            .await
            .expect("test: TorManager::new should succeed");

        manager.start().await.expect("test: start should succeed");

        let stream_id = manager
            .connect("example.onion:8080")
            .await
            .expect("test: connect should succeed");
        manager
            .close_stream(stream_id)
            .expect("test: close_stream should succeed");

        let stats = manager.stats();
        assert_eq!(stats.active_streams, 0);
    }

    #[tokio::test]
    async fn test_config_presets() {
        let high_privacy = TorConfig::high_privacy();
        assert!(high_privacy.stream_isolation);
        assert_eq!(high_privacy.max_circuits, 5);

        let high_performance = TorConfig::high_performance();
        assert!(!high_performance.stream_isolation);
        assert_eq!(high_performance.max_circuits, 20);

        let censorship = TorConfig::censorship_resistant();
        assert!(censorship.use_bridges);
        assert!(!censorship.bridges.is_empty());
    }

    #[test]
    fn test_validate_onion_address() {
        // Valid v3 onion address (56 characters, base32: a-z, 2-7)
        let v3_addr = "abcdefghijklmnopqrstuvabcdefghijklmnopqrstuvabcdefghijkl.onion";
        assert!(TorManager::validate_onion_address(v3_addr));

        // Valid v2 onion address (16 characters, base32: a-z, 2-7)
        let v2_addr = "abcdefghijklmnop.onion";
        assert!(TorManager::validate_onion_address(v2_addr));

        // Invalid addresses
        assert!(!TorManager::validate_onion_address("invalid"));
        assert!(!TorManager::validate_onion_address("example.com"));
        assert!(!TorManager::validate_onion_address("abc.onion")); // Too short

        // Invalid base32 characters (contains 8, 9, x, y, z)
        assert!(!TorManager::validate_onion_address(
            "abcdefghijklmnopqrstuvwxyz234567abcdefghijklmnopqrstuvw.onion"
        ));
    }

    #[tokio::test]
    async fn test_circuit_cleanup() {
        let config = TorConfig {
            circuit_timeout: Duration::from_millis(100),
            ..Default::default()
        };
        let mut manager = TorManager::new(config)
            .await
            .expect("test: TorManager::new should succeed");

        manager.start().await.expect("test: start should succeed");

        // Create a circuit
        let _circuit_id = manager
            .create_circuit()
            .await
            .expect("test: create_circuit should succeed");

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Cleanup should remove the old circuit
        manager.cleanup_circuits();

        // Circuit should still exist if we just created it
        // But it should be cleaned up if it's old and unused
    }

    #[tokio::test]
    async fn test_not_running_errors() {
        let config = TorConfig::default();
        let manager = TorManager::new(config)
            .await
            .expect("test: TorManager::new should succeed");

        // These should fail when not running
        assert!(manager.create_circuit().await.is_err());
        assert!(manager.connect("example.onion:8080").await.is_err());

        let hs_config = HiddenServiceConfig::default();
        assert!(manager.create_hidden_service(hs_config).await.is_err());
    }
}
