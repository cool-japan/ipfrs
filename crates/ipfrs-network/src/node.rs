//! Network node implementation with full libp2p integration

use dashmap::DashSet;
use futures::StreamExt;
use libp2p::{
    autonat,
    core::Transport as _,
    dcutr, identify, identity, kad, mdns, noise, ping, relay,
    swarm::{NetworkBehaviour, SwarmEvent},
    Multiaddr, PeerId, Swarm,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot, Mutex};
use tracing::{debug, info, warn};

// Type alias for IPFRS results to avoid conflicts with libp2p types
type IpfrsResult<T> = ipfrs_core::error::Result<T>;

/// Type alias for provider waiters map to reduce type complexity
type ProviderWaiters = Arc<Mutex<HashMap<String, Vec<oneshot::Sender<Vec<PeerId>>>>>>;

/// Type alias for inference response waiters, keyed by session/request ID.
///
/// When `distributed_infer()` fires a request over GossipSub it registers a
/// oneshot sender here; the event loop wakes it when a matching
/// `InferenceResponse` arrives on the `INFERENCE_RESULT` topic.
pub type InferenceWaiters =
    Arc<Mutex<HashMap<String, Vec<oneshot::Sender<ipfrs_tensorlogic::InferenceResponse>>>>>;

/// Kademlia DHT configuration
#[derive(Debug, Clone)]
pub struct KademliaConfig {
    /// Query timeout in seconds
    pub query_timeout_secs: u64,
    /// Replication factor (k) - number of replicas to store
    pub replication_factor: usize,
    /// Alpha (α) - concurrency parameter for iterative queries
    pub alpha: usize,
    /// K-bucket size - maximum peers per bucket
    pub kbucket_size: usize,
}

impl Default for KademliaConfig {
    fn default() -> Self {
        Self {
            // Standard Kademlia timeout
            query_timeout_secs: 60,
            // IPFS uses 20 for replication
            replication_factor: 20,
            // Standard Kademlia alpha (3 is common, IPFS uses 3)
            alpha: 3,
            // Standard Kademlia k-bucket size (20 is standard)
            kbucket_size: 20,
        }
    }
}

/// Network configuration
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Listen addresses
    pub listen_addrs: Vec<String>,
    /// Bootstrap peers
    pub bootstrap_peers: Vec<String>,
    /// Enable QUIC transport
    pub enable_quic: bool,
    /// Data directory
    pub data_dir: PathBuf,
    /// Enable mDNS peer discovery
    pub enable_mdns: bool,
    /// Enable NAT traversal (AutoNAT + DCUtR)
    pub enable_nat_traversal: bool,
    /// Relay server addresses for NAT traversal
    pub relay_servers: Vec<String>,
    /// Kademlia DHT configuration
    pub kademlia: KademliaConfig,
    /// Maximum number of concurrent connections (None = unlimited)
    pub max_connections: Option<usize>,
    /// Maximum number of inbound connections (None = unlimited)
    pub max_inbound_connections: Option<usize>,
    /// Maximum number of outbound connections (None = unlimited)
    pub max_outbound_connections: Option<usize>,
    /// Connection buffer size in bytes
    pub connection_buffer_size: usize,
    /// Enable aggressive memory optimizations
    pub low_memory_mode: bool,
    /// Enable DCUtR hole-punching (default: true)
    ///
    /// When true the `dcutr` behaviour actively participates in NAT hole-punch
    /// coordination with peers.  Disabling this is useful for low-memory
    /// constrained environments where the small overhead of maintaining the
    /// DCUtR state machine is undesirable.
    pub dcutr_enabled: bool,
    /// Enable Circuit Relay v2 client behaviour (default: true)
    ///
    /// The relay client transport is always compiled in (because removing it
    /// from the combined transport would break the swarm type), but this flag
    /// controls whether the node actively seeks relay reservations.
    pub relay_v2_enabled: bool,
    /// Timeout for a single hole-punch attempt (default: 30 s)
    pub hole_punch_timeout: Duration,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            listen_addrs: vec![
                "/ip4/0.0.0.0/udp/0/quic-v1".to_string(),
                "/ip6/::/udp/0/quic-v1".to_string(),
            ],
            bootstrap_peers: vec![],
            enable_quic: true,
            enable_mdns: false,
            enable_nat_traversal: true,
            relay_servers: vec![],
            data_dir: PathBuf::from(".ipfrs"),
            kademlia: KademliaConfig::default(),
            max_connections: None,
            max_inbound_connections: None,
            max_outbound_connections: None,
            connection_buffer_size: 64 * 1024, // 64 KB default
            low_memory_mode: false,
            // NAT traversal defaults – enabled by default for production use
            dcutr_enabled: true,
            relay_v2_enabled: true,
            hole_punch_timeout: Duration::from_secs(30),
        }
    }
}

impl NetworkConfig {
    /// Create a low-memory configuration for constrained devices
    ///
    /// This configuration minimizes memory usage at the cost of some features:
    /// - Limited to 16 total connections
    /// - Smaller connection buffers (8 KB)
    /// - Reduced DHT parameters
    /// - mDNS disabled
    /// - NAT traversal disabled
    ///
    /// Suitable for embedded devices with < 128 MB RAM
    pub fn low_memory() -> Self {
        Self {
            listen_addrs: vec!["/ip4/0.0.0.0/udp/0/quic-v1".to_string()],
            bootstrap_peers: vec![],
            enable_quic: true,
            enable_mdns: false,          // Disabled to save memory
            enable_nat_traversal: false, // Disabled to save memory
            relay_servers: vec![],
            data_dir: PathBuf::from(".ipfrs"),
            kademlia: KademliaConfig {
                query_timeout_secs: 30, // Shorter timeout
                replication_factor: 10, // Reduced from 20
                alpha: 2,               // Reduced from 3
                kbucket_size: 10,       // Reduced from 20
            },
            max_connections: Some(16), // Very limited connections
            max_inbound_connections: Some(8),
            max_outbound_connections: Some(8),
            connection_buffer_size: 8 * 1024, // 8 KB buffers
            low_memory_mode: true,
            // NAT traversal disabled for low-memory environments
            dcutr_enabled: false,
            relay_v2_enabled: false,
            hole_punch_timeout: Duration::from_secs(30),
        }
    }

    /// Create an IoT device configuration
    ///
    /// Balanced configuration for IoT devices:
    /// - Limited to 32 total connections
    /// - Moderate connection buffers (16 KB)
    /// - Reduced DHT parameters
    /// - mDNS enabled for local discovery
    /// - NAT traversal enabled
    ///
    /// Suitable for IoT devices with 128-512 MB RAM
    pub fn iot() -> Self {
        Self {
            listen_addrs: vec!["/ip4/0.0.0.0/udp/0/quic-v1".to_string()],
            bootstrap_peers: vec![],
            enable_quic: true,
            enable_mdns: true, // Local discovery useful for IoT
            enable_nat_traversal: true,
            relay_servers: vec![],
            data_dir: PathBuf::from(".ipfrs"),
            kademlia: KademliaConfig {
                query_timeout_secs: 45,
                replication_factor: 15,
                alpha: 2,
                kbucket_size: 15,
            },
            max_connections: Some(32),
            max_inbound_connections: Some(16),
            max_outbound_connections: Some(16),
            connection_buffer_size: 16 * 1024, // 16 KB buffers
            low_memory_mode: false,
            dcutr_enabled: true,
            relay_v2_enabled: true,
            hole_punch_timeout: Duration::from_secs(30),
        }
    }

    /// Create a mobile device configuration
    ///
    /// Power and bandwidth-aware configuration for mobile devices:
    /// - Limited to 64 total connections
    /// - Standard connection buffers (32 KB)
    /// - Standard DHT parameters
    /// - mDNS disabled (battery saving)
    /// - NAT traversal enabled
    ///
    /// Suitable for mobile devices with network switching
    pub fn mobile() -> Self {
        Self {
            listen_addrs: vec!["/ip4/0.0.0.0/udp/0/quic-v1".to_string()],
            bootstrap_peers: vec![],
            enable_quic: true,
            enable_mdns: false, // Battery saving
            enable_nat_traversal: true,
            relay_servers: vec![],
            data_dir: PathBuf::from(".ipfrs"),
            kademlia: KademliaConfig {
                query_timeout_secs: 60,
                replication_factor: 20,
                alpha: 3,
                kbucket_size: 20,
            },
            dcutr_enabled: true,
            relay_v2_enabled: true,
            hole_punch_timeout: Duration::from_secs(30),
            max_connections: Some(64),
            max_inbound_connections: Some(32),
            max_outbound_connections: Some(32),
            connection_buffer_size: 32 * 1024, // 32 KB buffers
            low_memory_mode: false,
        }
    }

    /// Create a high-performance configuration for servers
    ///
    /// Optimized for high throughput and many connections:
    /// - Unlimited connections
    /// - Large connection buffers (128 KB)
    /// - Aggressive DHT parameters
    /// - All features enabled
    ///
    /// Suitable for servers with > 2 GB RAM
    pub fn high_performance() -> Self {
        Self {
            listen_addrs: vec![
                "/ip4/0.0.0.0/udp/0/quic-v1".to_string(),
                "/ip6/::/udp/0/quic-v1".to_string(),
            ],
            bootstrap_peers: vec![],
            enable_quic: true,
            enable_mdns: true,
            enable_nat_traversal: true,
            relay_servers: vec![],
            data_dir: PathBuf::from(".ipfrs"),
            kademlia: KademliaConfig {
                query_timeout_secs: 60,
                replication_factor: 20,
                alpha: 3,
                kbucket_size: 20,
            },
            max_connections: None, // Unlimited
            max_inbound_connections: None,
            max_outbound_connections: None,
            connection_buffer_size: 128 * 1024, // 128 KB buffers
            low_memory_mode: false,
            dcutr_enabled: true,
            relay_v2_enabled: true,
            hole_punch_timeout: Duration::from_secs(30),
        }
    }
}

/// Network behavior combining multiple protocols
#[derive(NetworkBehaviour)]
#[behaviour(to_swarm = "IpfrsBehaviourEvent")]
pub struct IpfrsBehaviour {
    /// Kademlia DHT for content and peer discovery
    pub kademlia: kad::Behaviour<kad::store::MemoryStore>,
    /// Identify protocol for peer information
    pub identify: identify::Behaviour,
    /// Ping protocol for connectivity checks
    pub ping: ping::Behaviour,
    /// AutoNAT for NAT detection and address confirmation
    pub autonat: autonat::Behaviour,
    /// DCUtR for hole punching through NAT
    pub dcutr: dcutr::Behaviour,
    /// mDNS for local network peer discovery
    pub mdns: mdns::tokio::Behaviour,
    /// Relay client for NAT traversal
    pub relay_client: relay::client::Behaviour,
}

/// Events generated by IpfrsBehaviour
#[derive(Debug)]
pub enum IpfrsBehaviourEvent {
    Kademlia(kad::Event),
    Identify(Box<identify::Event>),
    Ping(ping::Event),
    Autonat(autonat::Event),
    Dcutr(dcutr::Event),
    Mdns(mdns::Event),
    RelayClient(relay::client::Event),
}

impl From<kad::Event> for IpfrsBehaviourEvent {
    fn from(event: kad::Event) -> Self {
        IpfrsBehaviourEvent::Kademlia(event)
    }
}

impl From<identify::Event> for IpfrsBehaviourEvent {
    fn from(event: identify::Event) -> Self {
        IpfrsBehaviourEvent::Identify(Box::new(event))
    }
}

impl From<ping::Event> for IpfrsBehaviourEvent {
    fn from(event: ping::Event) -> Self {
        IpfrsBehaviourEvent::Ping(event)
    }
}

impl From<autonat::Event> for IpfrsBehaviourEvent {
    fn from(event: autonat::Event) -> Self {
        IpfrsBehaviourEvent::Autonat(event)
    }
}

impl From<dcutr::Event> for IpfrsBehaviourEvent {
    fn from(event: dcutr::Event) -> Self {
        IpfrsBehaviourEvent::Dcutr(event)
    }
}

impl From<mdns::Event> for IpfrsBehaviourEvent {
    fn from(event: mdns::Event) -> Self {
        IpfrsBehaviourEvent::Mdns(event)
    }
}

impl From<relay::client::Event> for IpfrsBehaviourEvent {
    fn from(event: relay::client::Event) -> Self {
        IpfrsBehaviourEvent::RelayClient(event)
    }
}

/// Commands forwarded from `NetworkNode` to the background swarm event loop.
///
/// After `start()` the swarm lives in a spawned task.  All operations that
/// need to call into the swarm (dial, provide, get_providers, …) are sent
/// over this channel and executed inside the event-loop task.
enum SwarmCommand {
    /// Dial a remote address
    Dial(Multiaddr),
    /// Disconnect a specific peer
    Disconnect(PeerId),
    /// Announce local content to the Kademlia DHT
    Provide(cid::Cid),
    /// Query the DHT for providers of a CID (fire-and-forget; waiters handle the result)
    GetProviders(cid::Cid),
    /// Ask the Kademlia routing table for the k-closest peers to our own ID
    Bootstrap,
    /// Add a peer address to the Kademlia routing table
    AddPeerAddress(PeerId, Multiaddr),
}

/// Circuit Relay v2 configuration.
///
/// Controls whether the node actively seeks relay reservations for NAT
/// traversal via the `/libp2p/circuit/relay/0.2.0/hop` protocol.
#[derive(Debug, Clone)]
pub struct RelayConfig {
    /// Enable Circuit Relay v2 client (reservation) support.
    pub relay_v2_enabled: bool,
    /// Maximum number of simultaneous relay reservations to maintain.
    pub max_reservations: usize,
    /// Duration in seconds for which a relay reservation is considered valid.
    pub reservation_duration_secs: u64,
}

impl Default for RelayConfig {
    fn default() -> Self {
        Self {
            relay_v2_enabled: true,
            max_reservations: 4,
            reservation_duration_secs: 3600,
        }
    }
}

/// IPFRS network node
pub struct NetworkNode {
    config: NetworkConfig,
    peer_id: PeerId,
    swarm: Option<Swarm<IpfrsBehaviour>>,
    shutdown_tx: Option<mpsc::Sender<()>>,
    /// Command channel to the background swarm event loop (set after `start()`)
    swarm_cmd_tx: Option<mpsc::Sender<SwarmCommand>>,
    event_tx: mpsc::Sender<NetworkEvent>,
    event_rx: Option<mpsc::Receiver<NetworkEvent>>,
    /// External addresses discovered via AutoNAT
    external_addrs: Arc<parking_lot::RwLock<Vec<Multiaddr>>>,
    /// Set of currently connected peers
    connected_peers: Arc<DashSet<PeerId>>,
    /// Bandwidth tracking (bytes sent/received)
    bandwidth_stats: Arc<parking_lot::RwLock<BandwidthStats>>,
    /// Waiters for provider query results, keyed by CID string
    provider_waiters: ProviderWaiters,
    /// NAT traversal (DCUtR hole-punch) metrics
    nat_metrics: Arc<parking_lot::RwLock<NatTraversalMetrics>>,
    /// In-process GossipSub manager for topic-based pub/sub messaging.
    ///
    /// Shared with callers so that external code (e.g. `distributed_infer`)
    /// can publish messages directly without going through the swarm command
    /// channel.  The manager is `Arc`-wrapped so it can be cloned cheaply.
    pub gossipsub: Arc<crate::gossipsub::GossipSubManager>,
    /// Waiters for inference responses, keyed by request/session ID.
    pub inference_waiters: InferenceWaiters,
    /// Active Circuit Relay v2 reservations, keyed by relay peer ID.
    ///
    /// Each entry records the [`std::time::Instant`] at which the reservation
    /// was obtained so that expired reservations can be detected and renewed.
    pub active_relay_reservations: Arc<parking_lot::RwLock<HashMap<PeerId, std::time::Instant>>>,
    /// Circuit Relay v2 configuration.
    pub relay_config: RelayConfig,
}

/// Bandwidth statistics
#[derive(Debug, Clone, Default)]
struct BandwidthStats {
    bytes_sent: u64,
    bytes_received: u64,
}

/// Network events
#[derive(Debug, Clone)]
pub enum NetworkEvent {
    /// Peer connected
    PeerConnected {
        peer_id: PeerId,
        endpoint: ConnectionEndpoint,
        established_in: std::time::Duration,
    },
    /// Peer disconnected
    PeerDisconnected {
        peer_id: PeerId,
        cause: Option<String>,
    },
    /// Content found in DHT
    ContentFound { cid: String, providers: Vec<PeerId> },
    /// New peer discovered
    PeerDiscovered {
        peer_id: PeerId,
        addrs: Vec<Multiaddr>,
    },
    /// Listening on a new address
    ListeningOn { address: Multiaddr },
    /// Connection error occurred
    ConnectionError {
        peer_id: Option<PeerId>,
        error: String,
    },
    /// DHT bootstrap completed
    DhtBootstrapCompleted,
    /// NAT status changed
    NatStatusChanged {
        old_status: String,
        new_status: String,
    },
}

/// Connection endpoint information
#[derive(Debug, Clone)]
pub enum ConnectionEndpoint {
    /// Dialer (outbound connection)
    Dialer { address: Multiaddr },
    /// Listener (inbound connection)
    Listener {
        local_addr: Multiaddr,
        send_back_addr: Multiaddr,
    },
}

/// Default keypair filename
const KEYPAIR_FILENAME: &str = "identity.key";

impl NetworkNode {
    /// Create a new network node
    pub fn new(config: NetworkConfig) -> IpfrsResult<Self> {
        info!("Creating network node with libp2p");

        // Load or generate keypair for stable identity
        let keypair = Self::load_or_generate_keypair(&config.data_dir)?;
        let peer_id = keypair.public().to_peer_id();

        info!("Local peer ID: {}", peer_id);

        // Create event channel
        let (event_tx, event_rx) = mpsc::channel(1024);

        // Build the swarm
        let swarm = Self::build_swarm(keypair, &config)?;

        // Build the GossipSub manager and subscribe to all inference topics.
        let gossipsub = {
            use crate::gossipsub::{GossipSubConfig, GossipSubManager};
            let mgr = GossipSubManager::new(GossipSubConfig::default());
            // Ignore errors – AlreadySubscribed is harmless on a fresh instance.
            let _ = mgr.subscribe_inference_topics();
            Arc::new(mgr)
        };

        Ok(Self {
            config,
            peer_id,
            swarm: Some(swarm),
            shutdown_tx: None,
            swarm_cmd_tx: None,
            event_tx,
            event_rx: Some(event_rx),
            external_addrs: Arc::new(RwLock::new(Vec::new())),
            connected_peers: Arc::new(DashSet::new()),
            bandwidth_stats: Arc::new(RwLock::new(BandwidthStats::default())),
            provider_waiters: Arc::new(Mutex::new(HashMap::new())),
            nat_metrics: Arc::new(RwLock::new(NatTraversalMetrics::default())),
            gossipsub,
            inference_waiters: Arc::new(Mutex::new(HashMap::new())),
            active_relay_reservations: Arc::new(RwLock::new(HashMap::new())),
            relay_config: RelayConfig::default(),
        })
    }

    /// Load existing keypair or generate a new one
    fn load_or_generate_keypair(data_dir: &Path) -> IpfrsResult<identity::Keypair> {
        let key_path = data_dir.join(KEYPAIR_FILENAME);

        if key_path.exists() {
            info!("Loading existing identity from {:?}", key_path);
            Self::load_keypair(&key_path)
        } else {
            info!("Generating new identity");
            let keypair = identity::Keypair::generate_ed25519();

            // Create data directory if it doesn't exist
            if !data_dir.exists() {
                fs::create_dir_all(data_dir).map_err(ipfrs_core::error::Error::Io)?;
            }

            // Save the new keypair
            Self::save_keypair(&keypair, &key_path)?;
            info!("Saved new identity to {:?}", key_path);

            Ok(keypair)
        }
    }

    /// Load keypair from file
    fn load_keypair(path: &Path) -> IpfrsResult<identity::Keypair> {
        let bytes = fs::read(path).map_err(ipfrs_core::error::Error::Io)?;

        identity::Keypair::from_protobuf_encoding(&bytes).map_err(|e| {
            ipfrs_core::error::Error::Network(format!("Failed to decode keypair: {}", e))
        })
    }

    /// Save keypair to file
    fn save_keypair(keypair: &identity::Keypair, path: &Path) -> IpfrsResult<()> {
        let bytes = keypair.to_protobuf_encoding().map_err(|e| {
            ipfrs_core::error::Error::Network(format!("Failed to encode keypair: {}", e))
        })?;

        fs::write(path, bytes).map_err(ipfrs_core::error::Error::Io)?;

        // Set restrictive permissions on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let permissions = fs::Permissions::from_mode(0o600);
            fs::set_permissions(path, permissions).map_err(ipfrs_core::error::Error::Io)?;
        }

        Ok(())
    }

    /// Build libp2p swarm with all protocols
    #[allow(clippy::too_many_lines)]
    fn build_swarm(
        keypair: identity::Keypair,
        config: &NetworkConfig,
    ) -> IpfrsResult<Swarm<IpfrsBehaviour>> {
        let peer_id = keypair.public().to_peer_id();

        // Create relay client for NAT traversal.
        // The relay *transport* must remain alive alongside the relay *behaviour*
        // because they communicate via an internal channel.  Including it in the
        // combined transport (even when nat-traversal is disabled) keeps the
        // channel open and prevents the "polled after channel closed" panic.
        let (relay_transport, relay_client) = relay::client::new(peer_id);

        // Upgrade the relay transport so it can be combined with the others
        let relay_transport = relay_transport
            .upgrade(libp2p::core::upgrade::Version::V1)
            .authenticate(noise::Config::new(&keypair).map_err(std::io::Error::other)?)
            .multiplex(libp2p::yamux::Config::default())
            .map(|(peer_id, muxer), _| (peer_id, libp2p::core::muxing::StreamMuxerBox::new(muxer)));

        // Build TCP transport with noise and yamux
        let tcp_transport = libp2p::tcp::tokio::Transport::default()
            .upgrade(libp2p::core::upgrade::Version::V1)
            .authenticate(noise::Config::new(&keypair).map_err(std::io::Error::other)?)
            .multiplex(libp2p::yamux::Config::default())
            .map(|(peer_id, muxer), _| (peer_id, libp2p::core::muxing::StreamMuxerBox::new(muxer)));

        // Build QUIC transport
        let quic_transport = libp2p::quic::tokio::Transport::new(libp2p::quic::Config::new(
            &keypair,
        ))
        .map(|(peer_id, muxer), _| (peer_id, libp2p::core::muxing::StreamMuxerBox::new(muxer)));

        // Combine transports: relay (for circuit relay v2) → QUIC → TCP
        // The relay transport handles /p2p-circuit addresses; QUIC and TCP handle
        // direct connections.  Order matters: relay is tried first for circuit
        // addresses, then QUIC, then TCP.
        let transport = if config.enable_quic {
            relay_transport
                .or_transport(quic_transport)
                .map(|either, _| either.into_inner())
                .or_transport(tcp_transport)
                .map(|either, _| either.into_inner())
                .boxed()
        } else {
            relay_transport
                .or_transport(tcp_transport)
                .map(|either, _| either.into_inner())
                .boxed()
        };

        // Create Kademlia DHT with tunable config
        let store = kad::store::MemoryStore::new(peer_id);
        let mut kad_config = kad::Config::default();

        // Apply tunable parameters
        kad_config.set_query_timeout(Duration::from_secs(config.kademlia.query_timeout_secs));
        kad_config.set_replication_factor(
            std::num::NonZeroUsize::new(config.kademlia.replication_factor)
                .expect("Replication factor must be > 0"),
        );
        kad_config.set_parallelism(
            std::num::NonZeroUsize::new(config.kademlia.alpha).expect("Alpha must be > 0"),
        );
        kad_config.set_kbucket_inserts(kad::BucketInserts::OnConnected);

        // Set max k-bucket size if possible (note: libp2p has a fixed K=20 in current versions)
        // The kbucket_size parameter is kept for future compatibility

        let kademlia = kad::Behaviour::with_config(peer_id, store, kad_config);

        // Create Identify protocol
        let identify = identify::Behaviour::new(
            identify::Config::new("/ipfrs/1.0.0".to_string(), keypair.public())
                .with_agent_version(format!("ipfrs/{}", env!("CARGO_PKG_VERSION"))),
        );

        // Create Ping protocol for connectivity checks
        let ping = ping::Behaviour::new(ping::Config::new().with_interval(Duration::from_secs(15)));

        // Create AutoNAT for NAT detection
        let autonat = autonat::Behaviour::new(
            peer_id,
            autonat::Config {
                only_global_ips: false,
                ..Default::default()
            },
        );

        // Create DCUtR for hole punching
        let dcutr = dcutr::Behaviour::new(peer_id);

        // Create mDNS for local network discovery (if enabled)
        let mdns = if config.enable_mdns {
            mdns::tokio::Behaviour::new(mdns::Config::default(), peer_id)
                .map_err(|e| ipfrs_core::error::Error::Network(e.to_string()))?
        } else {
            // Disabled mDNS - still need to create one but it won't discover
            mdns::tokio::Behaviour::new(
                mdns::Config {
                    ttl: Duration::from_secs(1),
                    query_interval: Duration::from_secs(3600), // Very long interval
                    enable_ipv6: false,
                },
                peer_id,
            )
            .map_err(|e| ipfrs_core::error::Error::Network(e.to_string()))?
        };

        // Combine into network behavior
        let behaviour = IpfrsBehaviour {
            kademlia,
            identify,
            ping,
            autonat,
            dcutr,
            mdns,
            relay_client,
        };

        // Create swarm with tokio executor
        let mut swarm_config = libp2p::swarm::Config::with_executor(|fut| {
            tokio::spawn(fut);
        });
        swarm_config = swarm_config.with_idle_connection_timeout(Duration::from_secs(60));

        let swarm = Swarm::new(transport, behaviour, peer_id, swarm_config);

        Ok(swarm)
    }

    /// Start the network node
    pub async fn start(&mut self) -> IpfrsResult<()> {
        info!("🚀 IPFRS Network Node Starting");
        info!("   Peer ID: {}", self.peer_id);
        info!("   QUIC enabled: {}", self.config.enable_quic);

        let mut swarm = self.swarm.take().ok_or_else(|| {
            ipfrs_core::error::Error::Network("Swarm already started".to_string())
        })?;

        // Listen on configured addresses
        for addr_str in &self.config.listen_addrs {
            let addr: Multiaddr = addr_str.parse().map_err(|e| {
                ipfrs_core::error::Error::Network(format!("Invalid multiaddr: {}", e))
            })?;

            swarm
                .listen_on(addr.clone())
                .map_err(|e| ipfrs_core::error::Error::Network(e.to_string()))?;

            info!("   Listening on: {}", addr);
        }

        // Bootstrap DHT with configured peers
        for peer_str in &self.config.bootstrap_peers {
            match peer_str.parse::<Multiaddr>() {
                Ok(addr) => {
                    if let Err(e) = swarm.dial(addr.clone()) {
                        warn!("Failed to dial bootstrap peer {}: {}", addr, e);
                    } else {
                        info!("   Dialing bootstrap peer: {}", addr);
                    }
                }
                Err(e) => {
                    warn!("Invalid bootstrap peer address {}: {}", peer_str, e);
                }
            }
        }

        // Put DHT into server mode
        swarm
            .behaviour_mut()
            .kademlia
            .set_mode(Some(kad::Mode::Server));

        // Bootstrap the DHT
        if let Err(e) = swarm.behaviour_mut().kademlia.bootstrap() {
            warn!("DHT bootstrap failed: {}", e);
        }

        // Create shutdown channel
        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);
        self.shutdown_tx = Some(shutdown_tx);

        // Create swarm command channel so callers can drive the swarm from
        // outside the event-loop task after start().
        let (swarm_cmd_tx, mut swarm_cmd_rx) = mpsc::channel::<SwarmCommand>(256);
        self.swarm_cmd_tx = Some(swarm_cmd_tx);

        let event_tx = self.event_tx.clone();
        let external_addrs = Arc::clone(&self.external_addrs);
        let connected_peers = Arc::clone(&self.connected_peers);
        let provider_waiters = Arc::clone(&self.provider_waiters);
        let nat_metrics = Arc::clone(&self.nat_metrics);

        info!("✅ Network node ready");
        info!(
            "   Transport: {}",
            if self.config.enable_quic {
                "QUIC"
            } else {
                "TCP"
            }
        );
        info!("   DHT mode: Server");

        // Event loop
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    event = swarm.select_next_some() => {
                        Self::handle_swarm_event(event, &event_tx, swarm.behaviour_mut(), &external_addrs, &connected_peers, &provider_waiters, &nat_metrics).await;
                    }
                    Some(cmd) = swarm_cmd_rx.recv() => {
                        Self::handle_swarm_command(cmd, &mut swarm, &provider_waiters).await;
                    }
                    _ = shutdown_rx.recv() => {
                        info!("Shutting down network node");
                        break;
                    }
                }
            }
        });

        Ok(())
    }

    /// Handle swarm events
    async fn handle_swarm_event(
        event: SwarmEvent<IpfrsBehaviourEvent>,
        event_tx: &mpsc::Sender<NetworkEvent>,
        _behaviour: &mut IpfrsBehaviour,
        external_addrs: &Arc<RwLock<Vec<Multiaddr>>>,
        connected_peers: &Arc<DashSet<PeerId>>,
        provider_waiters: &ProviderWaiters,
        nat_metrics: &Arc<RwLock<NatTraversalMetrics>>,
    ) {
        match event {
            SwarmEvent::NewListenAddr { address, .. } => {
                info!("Listening on {}", address);
                let _ = event_tx
                    .send(NetworkEvent::ListeningOn {
                        address: address.clone(),
                    })
                    .await;
            }
            SwarmEvent::Behaviour(IpfrsBehaviourEvent::Identify(ev)) => {
                if let identify::Event::Received { peer_id, info, .. } = *ev {
                    debug!("Identified peer {}: {:?}", peer_id, info);
                    let _ = event_tx
                        .send(NetworkEvent::PeerDiscovered {
                            peer_id,
                            addrs: info.listen_addrs,
                        })
                        .await;
                }
            }
            SwarmEvent::Behaviour(IpfrsBehaviourEvent::Kademlia(
                kad::Event::OutboundQueryProgressed { result, .. },
            )) => match result {
                kad::QueryResult::GetProviders(Ok(kad::GetProvidersOk::FoundProviders {
                    key,
                    providers,
                })) => {
                    let cid = String::from_utf8_lossy(key.as_ref()).to_string();
                    let provider_list: Vec<PeerId> = providers.into_iter().collect();
                    debug!("Found {} providers for {}", provider_list.len(), cid);

                    // Notify any registered waiters for this CID
                    {
                        let mut waiters = provider_waiters.lock().await;
                        if let Some(senders) = waiters.remove(&cid) {
                            for tx in senders {
                                // Best-effort: ignore send errors (receiver may have timed out)
                                let _ = tx.send(provider_list.clone());
                            }
                        }
                    }

                    let _ = event_tx
                        .send(NetworkEvent::ContentFound {
                            cid,
                            providers: provider_list,
                        })
                        .await;
                }
                kad::QueryResult::GetProviders(Err(e)) => {
                    debug!("GetProviders query failed: {:?}", e);
                }
                kad::QueryResult::Bootstrap(Ok(_)) => {
                    info!("DHT bootstrap completed");
                    let _ = event_tx.send(NetworkEvent::DhtBootstrapCompleted).await;
                }
                kad::QueryResult::Bootstrap(Err(e)) => {
                    warn!("DHT bootstrap failed: {:?}", e);
                }
                _ => {}
            },
            SwarmEvent::ConnectionEstablished {
                peer_id,
                endpoint,
                established_in,
                ..
            } => {
                info!("Connected to peer: {} in {:?}", peer_id, established_in);

                // Track connected peer
                connected_peers.insert(peer_id);

                let conn_endpoint = if endpoint.is_dialer() {
                    ConnectionEndpoint::Dialer {
                        address: endpoint.get_remote_address().clone(),
                    }
                } else {
                    ConnectionEndpoint::Listener {
                        local_addr: endpoint.get_remote_address().clone(),
                        send_back_addr: endpoint.get_remote_address().clone(),
                    }
                };

                let _ = event_tx
                    .send(NetworkEvent::PeerConnected {
                        peer_id,
                        endpoint: conn_endpoint,
                        established_in,
                    })
                    .await;
            }
            SwarmEvent::ConnectionClosed {
                peer_id,
                cause,
                num_established,
                ..
            } => {
                info!("Disconnected from peer {}: {:?}", peer_id, cause);

                // Remove peer from tracking if no more connections remain
                if num_established == 0 {
                    connected_peers.remove(&peer_id);
                }

                let _ = event_tx
                    .send(NetworkEvent::PeerDisconnected {
                        peer_id,
                        cause: cause.map(|c| format!("{:?}", c)),
                    })
                    .await;
            }
            SwarmEvent::IncomingConnection { .. } => {
                debug!("Incoming connection");
            }
            SwarmEvent::IncomingConnectionError { error, .. } => {
                debug!("Incoming connection error: {}", error);
                let _ = event_tx
                    .send(NetworkEvent::ConnectionError {
                        peer_id: None,
                        error: error.to_string(),
                    })
                    .await;
            }
            SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
                warn!("Outgoing connection error to {:?}: {}", peer_id, error);
                let _ = event_tx
                    .send(NetworkEvent::ConnectionError {
                        peer_id,
                        error: error.to_string(),
                    })
                    .await;
            }
            SwarmEvent::Behaviour(IpfrsBehaviourEvent::Autonat(autonat_event)) => {
                match autonat_event {
                    autonat::Event::InboundProbe(_) => {
                        debug!("AutoNAT inbound probe");
                    }
                    autonat::Event::OutboundProbe(_) => {
                        debug!("AutoNAT outbound probe");
                    }
                    autonat::Event::StatusChanged { old, new } => {
                        info!("AutoNAT status changed from {:?} to {:?}", old, new);

                        let old_status = format!("{:?}", old);
                        let new_status = format!("{:?}", new);

                        let _ = event_tx
                            .send(NetworkEvent::NatStatusChanged {
                                old_status,
                                new_status,
                            })
                            .await;

                        match new {
                            autonat::NatStatus::Public(addr) => {
                                info!("Public address confirmed: {}", addr);
                                // Track external address
                                let mut addrs = external_addrs.write();
                                if !addrs.contains(&addr) {
                                    addrs.push(addr);
                                }
                            }
                            autonat::NatStatus::Private => {
                                info!("Node is behind NAT");
                                // Clear external addresses when behind NAT
                                external_addrs.write().clear();
                            }
                            autonat::NatStatus::Unknown => {
                                debug!("NAT status unknown");
                            }
                        }
                    }
                }
            }
            SwarmEvent::Behaviour(IpfrsBehaviourEvent::Dcutr(dcutr_event)) => {
                debug!("DCUtR event: {:?}", dcutr_event);
                match dcutr_event {
                    dcutr::Event { result: Ok(_), .. } => {
                        let mut m = nat_metrics.write();
                        m.hole_punch_attempts = m.hole_punch_attempts.saturating_add(1);
                        m.hole_punch_successes = m.hole_punch_successes.saturating_add(1);
                        info!("DCUtR hole-punch succeeded");
                    }
                    dcutr::Event {
                        result: Err(ref e), ..
                    } => {
                        let mut m = nat_metrics.write();
                        m.hole_punch_attempts = m.hole_punch_attempts.saturating_add(1);
                        m.hole_punch_failures = m.hole_punch_failures.saturating_add(1);
                        warn!("DCUtR hole-punch failed: {}", e);
                    }
                }
            }
            SwarmEvent::Behaviour(IpfrsBehaviourEvent::Mdns(mdns_event)) => match mdns_event {
                mdns::Event::Discovered(peers) => {
                    for (peer_id, addr) in peers {
                        info!("mDNS discovered peer {} at {}", peer_id, addr);
                        let _ = event_tx
                            .send(NetworkEvent::PeerDiscovered {
                                peer_id,
                                addrs: vec![addr],
                            })
                            .await;
                    }
                }
                mdns::Event::Expired(peers) => {
                    for (peer_id, addr) in peers {
                        debug!("mDNS peer expired: {} at {}", peer_id, addr);
                    }
                }
            },
            SwarmEvent::Behaviour(IpfrsBehaviourEvent::RelayClient(relay_event)) => {
                debug!("Relay client event: {:?}", relay_event);
                match &relay_event {
                    relay::client::Event::ReservationReqAccepted { .. } => {
                        let mut m = nat_metrics.write();
                        m.relay_connections = m.relay_connections.saturating_add(1);
                        info!("Relay reservation accepted");
                    }
                    relay::client::Event::OutboundCircuitEstablished { .. } => {
                        let mut m = nat_metrics.write();
                        m.relay_connections = m.relay_connections.saturating_add(1);
                        debug!("Outbound relay circuit established");
                    }
                    _ => {}
                }
            }
            SwarmEvent::Behaviour(IpfrsBehaviourEvent::Ping(ping_event)) => {
                if let Ok(rtt) = ping_event.result {
                    debug!("Ping to {:?}: RTT = {:?}", ping_event.peer, rtt);
                }
            }
            _ => {}
        }
    }

    /// Handle a command sent to the background swarm event loop.
    ///
    /// This runs inside the spawned task that owns the swarm, so it can call
    /// swarm methods directly.
    async fn handle_swarm_command(
        cmd: SwarmCommand,
        swarm: &mut Swarm<IpfrsBehaviour>,
        provider_waiters: &ProviderWaiters,
    ) {
        match cmd {
            SwarmCommand::Dial(addr) => match swarm.dial(addr.clone()) {
                Ok(()) => info!("Dialing peer: {}", addr),
                Err(e) => warn!("Dial error for {}: {}", addr, e),
            },
            SwarmCommand::Disconnect(peer_id) => {
                let _ = swarm.disconnect_peer_id(peer_id);
                info!("Disconnecting from peer: {}", peer_id);
            }
            SwarmCommand::Provide(cid) => {
                let key = kad::RecordKey::new(&cid.to_bytes());
                match swarm.behaviour_mut().kademlia.start_providing(key) {
                    Ok(_) => debug!("Announcing content: {}", cid),
                    Err(e) => warn!("Failed to announce {}: {}", cid, e),
                }
            }
            SwarmCommand::GetProviders(cid) => {
                let cid_str = String::from_utf8_lossy(&cid.to_bytes()).to_string();
                let key = kad::RecordKey::new(&cid.to_bytes());
                swarm.behaviour_mut().kademlia.get_providers(key);
                debug!("Querying DHT providers for: {}", cid_str);
            }
            SwarmCommand::Bootstrap => match swarm.behaviour_mut().kademlia.bootstrap() {
                Ok(_) => info!("DHT bootstrap initiated"),
                Err(e) => warn!("DHT bootstrap failed: {}", e),
            },
            SwarmCommand::AddPeerAddress(peer_id, addr) => {
                swarm
                    .behaviour_mut()
                    .kademlia
                    .add_address(&peer_id, addr.clone());
                debug!("Added address {} for peer {}", addr, peer_id);
                // Also try to proactively add the peer to our routing table
                // by dialing if not already connected
                if !swarm.is_connected(&peer_id) {
                    if let Err(e) = swarm.dial(addr.clone()) {
                        debug!("Auto-dial for routing table peer {}: {}", peer_id, e);
                    }
                }
            }
        }
        // Suppress unused warning on provider_waiters (it's used by GetProviders
        // via the event handler, not directly here)
        let _ = provider_waiters;
    }

    /// Stop the network node
    pub async fn stop(&mut self) -> IpfrsResult<()> {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(()).await;
        }
        self.swarm_cmd_tx = None;
        Ok(())
    }

    /// Get local peer ID
    pub fn peer_id(&self) -> PeerId {
        self.peer_id
    }

    /// Get listening addresses
    pub fn listeners(&self) -> Vec<String> {
        self.config.listen_addrs.clone()
    }

    /// Get connected peers
    pub fn connected_peers(&self) -> Vec<PeerId> {
        self.connected_peers
            .iter()
            .map(|entry| *entry.key())
            .collect()
    }

    /// Helper: send a command to the background swarm event loop.
    ///
    /// Before `start()` we still have the swarm locally, so we handle commands
    /// inline.  After `start()` we forward via the command channel.
    fn send_swarm_cmd(&self, cmd: SwarmCommand) -> IpfrsResult<()> {
        match &self.swarm_cmd_tx {
            Some(tx) => tx.try_send(cmd).map_err(|e| {
                ipfrs_core::error::Error::Network(format!("Swarm command channel error: {}", e))
            }),
            None => {
                // Node not yet started – silently ignore (pre-start dial attempts etc.)
                Ok(())
            }
        }
    }

    /// Connect to a peer
    pub async fn connect(&mut self, addr: Multiaddr) -> IpfrsResult<()> {
        if let Some(swarm) = &mut self.swarm {
            // Node not yet started: drive swarm directly
            swarm
                .dial(addr.clone())
                .map_err(|e| ipfrs_core::error::Error::Network(e.to_string()))?;
            info!("Dialing peer: {}", addr);
        } else {
            // Node is running: forward to event-loop task
            self.send_swarm_cmd(SwarmCommand::Dial(addr))?;
        }
        Ok(())
    }

    /// Disconnect from a peer
    pub async fn disconnect(&mut self, peer_id: PeerId) -> IpfrsResult<()> {
        if let Some(swarm) = &mut self.swarm {
            let _ = swarm.disconnect_peer_id(peer_id);
            info!("Disconnecting from peer: {}", peer_id);
        } else {
            self.send_swarm_cmd(SwarmCommand::Disconnect(peer_id))?;
        }
        Ok(())
    }

    /// Announce content to DHT (provide)
    pub async fn provide(&mut self, cid: &cid::Cid) -> IpfrsResult<()> {
        if let Some(swarm) = &mut self.swarm {
            let key = kad::RecordKey::new(&cid.to_bytes());
            swarm
                .behaviour_mut()
                .kademlia
                .start_providing(key)
                .map_err(|e| ipfrs_core::error::Error::Network(e.to_string()))?;
            debug!("Announcing content: {}", cid);
        } else {
            self.send_swarm_cmd(SwarmCommand::Provide(*cid))?;
        }
        Ok(())
    }

    /// Find providers for content in DHT (fire and forget)
    pub async fn find_providers(&mut self, cid: &cid::Cid) -> IpfrsResult<()> {
        if let Some(swarm) = &mut self.swarm {
            let key = kad::RecordKey::new(&cid.to_bytes());
            swarm.behaviour_mut().kademlia.get_providers(key);
            debug!("Searching for providers of: {}", cid);
        } else {
            self.send_swarm_cmd(SwarmCommand::GetProviders(*cid))?;
        }
        Ok(())
    }

    /// Find providers for content in DHT and wait for results
    ///
    /// Queries the Kademlia DHT for providers of the given CID and waits up to
    /// `timeout` for the first set of results. Returns the provider peer IDs.
    pub async fn find_providers_await(
        &mut self,
        cid: &cid::Cid,
        timeout: Duration,
    ) -> IpfrsResult<Vec<PeerId>> {
        let cid_str = String::from_utf8_lossy(&cid.to_bytes()).to_string();

        // Register a waiter before firing the query so we don't miss early responses
        let (tx, rx) = oneshot::channel::<Vec<PeerId>>();
        {
            let mut waiters = self.provider_waiters.lock().await;
            waiters.entry(cid_str.clone()).or_default().push(tx);
        }

        // Fire the DHT query via command channel or directly
        if let Some(swarm) = &mut self.swarm {
            let key = kad::RecordKey::new(&cid.to_bytes());
            swarm.behaviour_mut().kademlia.get_providers(key);
            debug!(
                "Querying DHT providers for: {} (with timeout {:?})",
                cid, timeout
            );
        } else {
            match self.send_swarm_cmd(SwarmCommand::GetProviders(*cid)) {
                Ok(()) => {
                    debug!(
                        "Querying DHT providers for: {} (with timeout {:?})",
                        cid, timeout
                    );
                }
                Err(_) => {
                    // Command channel broken – clean up waiter and return empty
                    let mut waiters = self.provider_waiters.lock().await;
                    if let Some(senders) = waiters.get_mut(&cid_str) {
                        senders.retain(|_| false);
                    }
                    return Ok(Vec::new());
                }
            }
        }

        // Wait for the result with a timeout
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(providers)) => {
                debug!("Received {} providers for {}", providers.len(), cid);
                Ok(providers)
            }
            Ok(Err(_)) => {
                // Sender was dropped without sending (e.g. query failed)
                debug!("Provider query for {} completed with no results", cid);
                Ok(Vec::new())
            }
            Err(_) => {
                // Timeout – clean up the stale waiter
                debug!("Provider query for {} timed out after {:?}", cid, timeout);
                let mut waiters = self.provider_waiters.lock().await;
                if let Some(senders) = waiters.get_mut(&cid_str) {
                    senders.retain(|_| false);
                }
                Ok(Vec::new())
            }
        }
    }

    /// Fetch a block from a specific peer via Bitswap
    ///
    /// This is a best-effort implementation. If the peer is connected and has the
    /// block, it will be returned. Otherwise an error is returned and the caller
    /// should try the next provider.
    pub async fn fetch_block_from_peer(
        &mut self,
        peer: &PeerId,
        cid: &cid::Cid,
    ) -> IpfrsResult<ipfrs_core::Block> {
        // For now, verify the peer is connected before attempting fetch
        if !self.connected_peers.contains(peer) {
            return Err(ipfrs_core::error::Error::Network(format!(
                "Peer {} is not connected; cannot fetch block {}",
                peer, cid
            )));
        }

        // Full Bitswap block exchange over a live connection is a future milestone.
        // The plumbing (connected peer check, DHT provider discovery) is in place;
        // the actual Bitswap wire protocol exchange will be wired here in Task E.
        Err(ipfrs_core::error::Error::NotFound(format!(
            "Block {} not yet retrievable from peer {} (Bitswap exchange pending Task E)",
            cid, peer
        )))
    }

    /// Find node (closest peers to a given peer ID) using Kademlia
    pub async fn find_node(&mut self, peer_id: PeerId) -> IpfrsResult<()> {
        if let Some(swarm) = &mut self.swarm {
            swarm.behaviour_mut().kademlia.get_closest_peers(peer_id);
            debug!("Finding closest peers to: {}", peer_id);
        }
        Ok(())
    }

    /// Get the k-closest peers to our local peer ID
    pub async fn get_closest_local_peers(&mut self) -> IpfrsResult<Vec<PeerId>> {
        if let Some(swarm) = &mut self.swarm {
            let mut closest_peers = Vec::new();

            // Get peers from the routing table
            for bucket in swarm.behaviour_mut().kademlia.kbuckets() {
                for entry in bucket.iter() {
                    closest_peers.push(*entry.node.key.preimage());
                }
            }

            debug!("Found {} peers in routing table", closest_peers.len());
            Ok(closest_peers)
        } else {
            Ok(Vec::new())
        }
    }

    /// Bootstrap the DHT (search for our own peer ID to populate routing table)
    pub async fn bootstrap_dht(&mut self) -> IpfrsResult<()> {
        if let Some(swarm) = &mut self.swarm {
            swarm
                .behaviour_mut()
                .kademlia
                .bootstrap()
                .map_err(|e| ipfrs_core::error::Error::Network(e.to_string()))?;
            info!("DHT bootstrap initiated");
        } else {
            self.send_swarm_cmd(SwarmCommand::Bootstrap)?;
        }
        Ok(())
    }

    /// Add an address for a peer to the routing table
    pub fn add_peer_address(&mut self, peer_id: PeerId, addr: Multiaddr) -> IpfrsResult<()> {
        if let Some(swarm) = &mut self.swarm {
            swarm
                .behaviour_mut()
                .kademlia
                .add_address(&peer_id, addr.clone());
            debug!("Added address {} for peer {}", addr, peer_id);
        } else {
            self.send_swarm_cmd(SwarmCommand::AddPeerAddress(peer_id, addr))?;
        }
        Ok(())
    }

    /// Get routing table information
    pub fn get_routing_table_info(&mut self) -> IpfrsResult<RoutingTableInfo> {
        if let Some(swarm) = &mut self.swarm {
            let mut total_peers = 0;
            let mut buckets_info = Vec::new();

            for (index, bucket) in swarm.behaviour_mut().kademlia.kbuckets().enumerate() {
                let num_entries = bucket.iter().count();
                total_peers += num_entries;
                buckets_info.push(BucketInfo { index, num_entries });
            }

            Ok(RoutingTableInfo {
                total_peers,
                num_buckets: buckets_info.len(),
                buckets: buckets_info,
            })
        } else {
            Ok(RoutingTableInfo {
                total_peers: 0,
                num_buckets: 0,
                buckets: Vec::new(),
            })
        }
    }

    /// Get network statistics
    pub fn stats(&self) -> NetworkStats {
        let bandwidth = self.bandwidth_stats.read();
        NetworkStats {
            peer_id: self.peer_id.to_string(),
            listen_addrs: self.config.listen_addrs.clone(),
            connected_peers: self.connected_peers.len(),
            quic_enabled: self.config.enable_quic,
            bytes_received: bandwidth.bytes_received,
            bytes_sent: bandwidth.bytes_sent,
            bootstrap_peers: self.config.bootstrap_peers.clone(),
        }
    }

    /// Take the event receiver
    pub fn take_event_receiver(&mut self) -> Option<mpsc::Receiver<NetworkEvent>> {
        self.event_rx.take()
    }

    /// Get confirmed external addresses
    pub fn get_external_addresses(&self) -> Vec<Multiaddr> {
        self.external_addrs.read().clone()
    }

    /// Check if node has public reachability
    pub fn is_publicly_reachable(&self) -> bool {
        !self.external_addrs.read().is_empty()
    }

    /// Check if connected to a specific peer
    pub fn is_connected_to(&self, peer_id: &PeerId) -> bool {
        self.connected_peers.contains(peer_id)
    }

    /// Get number of connected peers
    pub fn get_peer_count(&self) -> usize {
        self.connected_peers.len()
    }

    /// Connect to multiple peers in batch
    pub async fn connect_to_peers(&mut self, addrs: Vec<Multiaddr>) -> Vec<IpfrsResult<()>> {
        let mut results = Vec::with_capacity(addrs.len());

        for addr in addrs {
            let result = self.connect(addr).await;
            results.push(result);
        }

        results
    }

    /// Disconnect from all connected peers
    pub async fn disconnect_all(&mut self) -> IpfrsResult<()> {
        let peers: Vec<PeerId> = self.connected_peers().clone();

        for peer in peers {
            let _ = self.disconnect(peer).await;
        }

        Ok(())
    }

    /// Update bandwidth statistics manually (for custom tracking)
    pub fn update_bandwidth(&self, bytes_sent: u64, bytes_received: u64) {
        let mut stats = self.bandwidth_stats.write();
        stats.bytes_sent += bytes_sent;
        stats.bytes_received += bytes_received;
    }

    /// Get total bandwidth sent
    pub fn get_bytes_sent(&self) -> u64 {
        self.bandwidth_stats.read().bytes_sent
    }

    /// Get total bandwidth received
    pub fn get_bytes_received(&self) -> u64 {
        self.bandwidth_stats.read().bytes_received
    }

    /// Reset bandwidth statistics
    pub fn reset_bandwidth_stats(&self) {
        let mut stats = self.bandwidth_stats.write();
        stats.bytes_sent = 0;
        stats.bytes_received = 0;
    }

    /// Get network health summary
    pub fn get_network_health(&self) -> NetworkHealthSummary {
        let peer_count = self.get_peer_count();
        let is_public = self.is_publicly_reachable();
        let has_external_addrs = !self.external_addrs.read().is_empty();

        // Determine health status
        let status = if peer_count >= 10 && is_public {
            NetworkHealthLevel::Healthy
        } else if peer_count >= 3 || has_external_addrs {
            NetworkHealthLevel::Degraded
        } else if peer_count > 0 {
            NetworkHealthLevel::Limited
        } else {
            NetworkHealthLevel::Disconnected
        };

        NetworkHealthSummary {
            status,
            connected_peers: peer_count,
            is_publicly_reachable: is_public,
            external_addresses: self.get_external_addresses().len(),
        }
    }

    /// Check if node is healthy
    pub fn is_healthy(&self) -> bool {
        matches!(
            self.get_network_health().status,
            NetworkHealthLevel::Healthy
        )
    }

    /// Get a snapshot of NAT traversal (hole-punch) metrics.
    pub fn nat_traversal_metrics(&self) -> NatTraversalMetrics {
        self.nat_metrics.read().clone()
    }

    // ─── Distributed inference transport ─────────────────────────────────────

    /// Publish an `InferenceRequest` to the GossipSub `INFERENCE_REQUEST`
    /// topic.
    ///
    /// Serialises `request` as JSON and hands it to the local
    /// `GossipSubManager`.  The manager fan-out to all subscribed peers is
    /// simulated in-process; wire integration is provided by the event loop
    /// once a real GossipSub swarm behaviour is wired in.
    ///
    /// # Errors
    /// Returns an error when JSON serialisation fails.
    pub fn publish_inference_request(
        &self,
        request: &ipfrs_tensorlogic::InferenceRequest,
    ) -> IpfrsResult<()> {
        let json = serde_json::to_vec(request).map_err(|e| {
            ipfrs_core::error::Error::Network(format!("Failed to serialize InferenceRequest: {e}"))
        })?;
        let peer_id_str = self.peer_id.to_string();
        self.gossipsub
            .publish_inference_request(&json, &peer_id_str)
            .map_err(|e| {
                ipfrs_core::error::Error::Network(format!(
                    "GossipSub publish_inference_request failed: {e}"
                ))
            })
    }

    /// Register a one-shot waiter that will be resolved when an
    /// `InferenceResponse` with the given `request_id` is delivered to this
    /// node via `deliver_inference_response`.
    ///
    /// Returns the receiving half of the oneshot channel.  The caller should
    /// wrap the `await` with [`tokio::time::timeout`] to bound the wait.
    pub async fn register_inference_waiter(
        &self,
        request_id: String,
    ) -> oneshot::Receiver<ipfrs_tensorlogic::InferenceResponse> {
        let (tx, rx) = oneshot::channel();
        let mut waiters = self.inference_waiters.lock().await;
        waiters.entry(request_id).or_default().push(tx);
        rx
    }

    /// Deliver an `InferenceResponse` to any registered waiters for
    /// `response.request_id`.
    ///
    /// This is the counterpart of `register_inference_waiter`.  Typically
    /// called from the event loop when a GossipSub message arrives on the
    /// `INFERENCE_RESULT` topic.
    pub async fn deliver_inference_response(&self, response: ipfrs_tensorlogic::InferenceResponse) {
        let mut waiters = self.inference_waiters.lock().await;
        if let Some(senders) = waiters.remove(&response.request_id) {
            for tx in senders {
                // Best-effort delivery – ignore closed receivers.
                let _ = tx.send(response.clone());
            }
        }
    }

    /// Publish a local `InferenceResponse` to the GossipSub
    /// `INFERENCE_RESULT` topic so remote requesters can collect it.
    ///
    /// # Errors
    /// Returns an error when JSON serialisation fails.
    pub fn publish_inference_response(
        &self,
        response: &ipfrs_tensorlogic::InferenceResponse,
    ) -> IpfrsResult<()> {
        let json = serde_json::to_vec(response).map_err(|e| {
            ipfrs_core::error::Error::Network(format!("Failed to serialize InferenceResponse: {e}"))
        })?;
        let peer_id_str = self.peer_id.to_string();
        self.gossipsub
            .publish_inference_result(&json, &peer_id_str)
            .map_err(|e| {
                ipfrs_core::error::Error::Network(format!(
                    "GossipSub publish_inference_result failed: {e}"
                ))
            })
    }
}

/// NAT traversal (hole-punching) metrics
///
/// Tracks the outcome of DCUtR hole-punch attempts so operators can assess
/// whether relay fallback is being relied on too heavily.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct NatTraversalMetrics {
    /// Total number of hole-punch attempts initiated (both sides)
    pub hole_punch_attempts: u64,
    /// Hole-punch attempts that resulted in a direct connection
    pub hole_punch_successes: u64,
    /// Hole-punch attempts that failed (connection remained via relay)
    pub hole_punch_failures: u64,
    /// Number of connections currently established via a relay circuit
    pub relay_connections: u64,
}

impl NatTraversalMetrics {
    /// Fraction of hole-punch attempts that succeeded (0.0 if no attempts).
    pub fn success_rate(&self) -> f32 {
        if self.hole_punch_attempts == 0 {
            return 0.0;
        }
        self.hole_punch_successes as f32 / self.hole_punch_attempts as f32
    }
}

/// Network statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct NetworkStats {
    pub peer_id: String,
    pub listen_addrs: Vec<String>,
    pub connected_peers: usize,
    pub quic_enabled: bool,
    /// Total bytes received
    pub bytes_received: u64,
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Bootstrap peers
    pub bootstrap_peers: Vec<String>,
}

/// Information about a k-bucket in the routing table
#[derive(Debug, Clone, serde::Serialize)]
pub struct BucketInfo {
    /// Bucket index
    pub index: usize,
    /// Number of entries in this bucket
    pub num_entries: usize,
}

/// Routing table information
#[derive(Debug, Clone, serde::Serialize)]
pub struct RoutingTableInfo {
    /// Total number of peers in routing table
    pub total_peers: usize,
    /// Number of buckets
    pub num_buckets: usize,
    /// Information about each bucket
    pub buckets: Vec<BucketInfo>,
}

/// Network health summary
#[derive(Debug, Clone, serde::Serialize)]
pub struct NetworkHealthSummary {
    /// Overall health status
    pub status: NetworkHealthLevel,
    /// Number of connected peers
    pub connected_peers: usize,
    /// Whether node is publicly reachable
    pub is_publicly_reachable: bool,
    /// Number of external addresses
    pub external_addresses: usize,
}

/// Network health level
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum NetworkHealthLevel {
    /// Fully connected with good peer count and public reachability
    Healthy,
    /// Connected but with limited peers or no public reachability
    Degraded,
    /// Minimal connectivity
    Limited,
    /// No connections
    Disconnected,
}

// ============================================================================
// Circuit Relay v2 — reservation management
// ============================================================================

impl NetworkNode {
    /// Attempt to obtain a Circuit Relay v2 reservation from `relay_peer`.
    ///
    /// The method dials the relay peer (if not already connected) and records
    /// the reservation in `active_relay_reservations`.  In a full
    /// implementation the swarm's `relay::client::Behaviour` would send the
    /// actual reservation request; here we perform the dial and record the
    /// reservation optimistically, returning an error if relay v2 is disabled
    /// in the node's configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Relay v2 is disabled in the node or relay config.
    /// - The maximum number of simultaneous reservations is already reached.
    /// - The swarm command channel is not available (node not started).
    pub async fn reserve_relay(&mut self, relay_peer: PeerId) -> IpfrsResult<()> {
        if !self.config.relay_v2_enabled || !self.relay_config.relay_v2_enabled {
            return Err(ipfrs_core::error::Error::Network(
                "Circuit Relay v2 is disabled".to_string(),
            ));
        }

        // Check max reservations limit.
        {
            let reservations = self.active_relay_reservations.read();
            if reservations.len() >= self.relay_config.max_reservations {
                return Err(ipfrs_core::error::Error::Network(format!(
                    "Maximum relay reservations ({}) already reached",
                    self.relay_config.max_reservations
                )));
            }
        }

        // Build the relay circuit address:
        // /p2p/<relay_peer_id>/p2p-circuit
        let relay_addr: Multiaddr =
            format!("/p2p/{}/p2p-circuit", relay_peer)
                .parse()
                .map_err(|e| {
                    ipfrs_core::error::Error::Network(format!(
                        "Invalid relay address for peer {}: {}",
                        relay_peer, e
                    ))
                })?;

        debug!(
            relay_peer = %relay_peer,
            addr = %relay_addr,
            "Requesting Circuit Relay v2 reservation"
        );

        // Send the dial command to the background swarm event-loop.
        // The actual `/libp2p/circuit/relay/0.2.0/hop` RESERVE message is
        // handled by the relay::client::Behaviour inside the swarm.
        if let Some(ref cmd_tx) = self.swarm_cmd_tx {
            cmd_tx
                .send(SwarmCommand::Dial(relay_addr))
                .await
                .map_err(|_| {
                    ipfrs_core::error::Error::Network("Swarm command channel closed".to_string())
                })?;
        } else {
            // Node not started yet: record the reservation intent anyway so
            // that the caller can check it later.
            warn!(
                relay_peer = %relay_peer,
                "reserve_relay called before node.start(); \
                 reservation recorded but dial not sent"
            );
        }

        // Record the reservation with the current timestamp.
        {
            let mut reservations = self.active_relay_reservations.write();
            reservations.insert(relay_peer, std::time::Instant::now());
        }

        info!(
            relay_peer = %relay_peer,
            "Circuit Relay v2 reservation recorded"
        );

        Ok(())
    }

    /// Return a snapshot of the currently active relay reservations.
    ///
    /// Each entry maps a relay [`PeerId`] to the [`std::time::Instant`] at
    /// which the reservation was obtained.
    pub fn relay_reservations(&self) -> HashMap<PeerId, std::time::Instant> {
        self.active_relay_reservations.read().clone()
    }

    /// Remove a relay reservation (e.g., when the relay peer disconnects).
    pub fn remove_relay_reservation(&mut self, relay_peer: &PeerId) {
        self.active_relay_reservations.write().remove(relay_peer);
    }

    /// Remove reservations older than `max_age`.
    pub fn prune_expired_relay_reservations(&mut self, max_age: std::time::Duration) {
        let now = std::time::Instant::now();
        self.active_relay_reservations
            .write()
            .retain(|_, instant| now.duration_since(*instant) < max_age);
    }
}
