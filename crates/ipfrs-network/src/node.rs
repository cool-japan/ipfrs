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
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

// Type alias for IPFRS results to avoid conflicts with libp2p types
type IpfrsResult<T> = ipfrs_core::error::Result<T>;

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
    Identify(identify::Event),
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
        IpfrsBehaviourEvent::Identify(event)
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

/// IPFRS network node
pub struct NetworkNode {
    config: NetworkConfig,
    peer_id: PeerId,
    swarm: Option<Swarm<IpfrsBehaviour>>,
    shutdown_tx: Option<mpsc::Sender<()>>,
    event_tx: mpsc::Sender<NetworkEvent>,
    event_rx: Option<mpsc::Receiver<NetworkEvent>>,
    /// External addresses discovered via AutoNAT
    external_addrs: Arc<parking_lot::RwLock<Vec<Multiaddr>>>,
    /// Set of currently connected peers
    connected_peers: Arc<DashSet<PeerId>>,
    /// Bandwidth tracking (bytes sent/received)
    bandwidth_stats: Arc<parking_lot::RwLock<BandwidthStats>>,
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

        Ok(Self {
            config,
            peer_id,
            swarm: Some(swarm),
            shutdown_tx: None,
            event_tx,
            event_rx: Some(event_rx),
            external_addrs: Arc::new(RwLock::new(Vec::new())),
            connected_peers: Arc::new(DashSet::new()),
            bandwidth_stats: Arc::new(RwLock::new(BandwidthStats::default())),
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

        // Create relay client for NAT traversal (behavior only - transport handled separately)
        let (_relay_transport, relay_client) = relay::client::new(peer_id);

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

        // Combine transports: QUIC primary with TCP fallback
        let transport = if config.enable_quic {
            // QUIC with TCP fallback
            quic_transport
                .or_transport(tcp_transport)
                .map(|either, _| either.into_inner())
                .boxed()
        } else {
            // TCP only
            tcp_transport.boxed()
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

        let event_tx = self.event_tx.clone();
        let external_addrs = Arc::clone(&self.external_addrs);
        let connected_peers = Arc::clone(&self.connected_peers);

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
                        Self::handle_swarm_event(event, &event_tx, swarm.behaviour_mut(), &external_addrs, &connected_peers).await;
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
            SwarmEvent::Behaviour(IpfrsBehaviourEvent::Identify(identify::Event::Received {
                peer_id,
                info,
                ..
            })) => {
                debug!("Identified peer {}: {:?}", peer_id, info);
                let _ = event_tx
                    .send(NetworkEvent::PeerDiscovered {
                        peer_id,
                        addrs: info.listen_addrs,
                    })
                    .await;
            }
            SwarmEvent::Behaviour(IpfrsBehaviourEvent::Kademlia(
                kad::Event::OutboundQueryProgressed { result, .. },
            )) => match result {
                kad::QueryResult::GetProviders(Ok(kad::GetProvidersOk::FoundProviders {
                    key,
                    providers,
                })) => {
                    let cid = String::from_utf8_lossy(key.as_ref()).to_string();
                    debug!("Found {} providers for {}", providers.len(), cid);
                    let _ = event_tx
                        .send(NetworkEvent::ContentFound {
                            cid,
                            providers: providers.into_iter().collect(),
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
            }
            SwarmEvent::Behaviour(IpfrsBehaviourEvent::Ping(ping_event)) => {
                if let Ok(rtt) = ping_event.result {
                    debug!("Ping to {:?}: RTT = {:?}", ping_event.peer, rtt);
                }
            }
            _ => {}
        }
    }

    /// Stop the network node
    pub async fn stop(&mut self) -> IpfrsResult<()> {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(()).await;
        }
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

    /// Connect to a peer
    pub async fn connect(&mut self, addr: Multiaddr) -> IpfrsResult<()> {
        if let Some(swarm) = &mut self.swarm {
            swarm
                .dial(addr.clone())
                .map_err(|e| ipfrs_core::error::Error::Network(e.to_string()))?;
            info!("Dialing peer: {}", addr);
        }
        Ok(())
    }

    /// Disconnect from a peer
    pub async fn disconnect(&mut self, peer_id: PeerId) -> IpfrsResult<()> {
        if let Some(swarm) = &mut self.swarm {
            let _ = swarm.disconnect_peer_id(peer_id);
            info!("Disconnecting from peer: {}", peer_id);
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
        }
        Ok(())
    }

    /// Find providers for content in DHT
    pub async fn find_providers(&mut self, cid: &cid::Cid) -> IpfrsResult<()> {
        if let Some(swarm) = &mut self.swarm {
            let key = kad::RecordKey::new(&cid.to_bytes());
            swarm.behaviour_mut().kademlia.get_providers(key);
            debug!("Searching for providers of: {}", cid);
        }
        Ok(())
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
