//! IPFRS Network - libp2p-based networking layer
//!
//! This crate provides the networking infrastructure for IPFRS including:
//! - libp2p node management with full protocol support
//! - QUIC transport with TCP fallback for reliable connectivity
//! - Kademlia DHT for content and peer discovery
//! - Bitswap protocol for block exchange
//! - mDNS for local peer discovery
//! - Bootstrap peer management with retry logic and circuit breaker
//! - Provider record caching with TTL and LRU eviction
//! - Connection limits and intelligent pruning
//! - Network metrics and Prometheus export
//! - NAT traversal (AutoNAT, DCUtR, Circuit Relay v2)
//! - Query optimization with early termination and pipelining
//! - Comprehensive health monitoring and logging
//!
//! ## Features
//!
//! ### Core Networking
//! - **Multi-transport**: QUIC (primary) with TCP fallback for maximum compatibility
//! - **NAT Traversal**: AutoNAT for detection, DCUtR for hole punching, Circuit Relay for fallback
//! - **Peer Discovery**: Kademlia DHT, mDNS for local peers, configurable bootstrap nodes
//! - **Connection Management**: Intelligent limits, priority-based pruning, bandwidth tracking
//!
//! ### DHT Operations
//! - **Content Routing**: Provider record publishing and discovery with automatic refresh
//! - **Peer Routing**: Find closest peers, routing table management
//! - **Query Optimization**: Early termination, pipelining, quality scoring
//! - **Caching**: Query results and provider records with TTL-based expiration
//! - **Semantic Routing**: Vector-based content discovery using embeddings and LSH
//!
//! ### Pub/Sub Messaging
//! - **GossipSub**: Topic-based publish/subscribe messaging
//! - **Mesh Formation**: Automatic peer mesh optimization for topic propagation
//! - **Message Deduplication**: Efficient duplicate detection
//! - **Peer Scoring**: Quality-based peer selection for reliable delivery
//!
//! ### Reliability
//! - **Retry Logic**: Exponential backoff with configurable limits
//! - **Circuit Breaker**: Prevent cascading failures with failing peers
//! - **Fallback Strategies**: Alternative peers, relay fallback, degraded mode
//! - **Health Monitoring**: DHT health, connection health, bandwidth metrics
//!
//! ### Monitoring & Observability
//! - **Metrics**: Connection stats, DHT stats, bandwidth tracking, query performance
//! - **Prometheus Export**: Ready-to-use metrics export for monitoring systems
//! - **Structured Logging**: Tracing spans with context propagation
//! - **Health Checks**: Component-level and overall health assessment
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use ipfrs_network::{NetworkConfig, NetworkNode};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create configuration
//!     let config = NetworkConfig {
//!         listen_addrs: vec!["/ip4/0.0.0.0/udp/0/quic-v1".to_string()],
//!         enable_quic: true,
//!         enable_mdns: true,
//!         enable_nat_traversal: true,
//!         ..Default::default()
//!     };
//!
//!     // Create and start network node
//!     let mut node = NetworkNode::new(config)?;
//!     node.start().await?;
//!
//!     // Check network health
//!     let health = node.get_network_health();
//!     println!("Network status: {:?}", health.status);
//!
//!     // Announce content to DHT
//!     let cid = cid::Cid::default();
//!     node.provide(&cid).await?;
//!
//!     // Get network statistics
//!     let stats = node.stats();
//!     println!("Connected peers: {}", stats.connected_peers);
//!
//!     Ok(())
//! }
//! ```
//!
//! ## High-Level Facade
//!
//! For easy integration of all features, use the `NetworkFacade`:
//!
//! ```rust,no_run
//! use ipfrs_network::NetworkFacadeBuilder;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a mobile-optimized node with advanced features
//!     let mut facade = NetworkFacadeBuilder::new()
//!         .with_preset_mobile()
//!         .with_semantic_dht()
//!         .with_gossipsub()
//!         .build()?;
//!
//!     facade.start().await?;
//!     println!("Peer ID: {}", facade.peer_id());
//!     Ok(())
//! }
//! ```
//!
//! ## Architecture
//!
//! The crate is organized into several modules:
//!
//! - **facade**: High-level facade for easy integration of all modules
//! - **auto_tuner**: Automatic network configuration tuning based on system resources
//! - **benchmarking**: Performance benchmarking utilities for network components
//! - **node**: Core `NetworkNode` implementation with libp2p swarm management
//! - **dht**: Kademlia DHT operations and caching
//! - **peer**: Peer store for tracking known peers and their metadata
//! - **connection_manager**: Connection limits and intelligent pruning
//! - **bootstrap**: Bootstrap peer management with retry logic
//! - **providers**: Provider record caching with TTL
//! - **query_optimizer**: Query optimization and performance tracking
//! - **metrics**: Network metrics and Prometheus export
//! - **health**: Health monitoring for network components
//! - **logging**: Structured logging and tracing
//! - **protocol**: Custom protocol support and version negotiation
//! - **fallback**: Fallback strategies for resilience
//! - **semantic_dht**: Vector-based semantic DHT for content routing by similarity
//! - **gossipsub**: Topic-based pub/sub messaging with mesh optimization
//! - **geo_routing**: Geographic routing optimization for proximity-based peer selection
//! - **dht_provider**: Pluggable DHT provider interface for custom DHT implementations
//! - **peer_selector**: Intelligent peer selection combining geographic proximity and quality metrics
//! - **multipath_quic**: QUIC multipath support for using multiple network paths simultaneously
//! - **tor**: Tor integration for privacy-preserving networking with onion routing and hidden services
//! - **diagnostics**: Network diagnostics and troubleshooting utilities
//! - **session**: Connection session management with lifecycle tracking and statistics
//! - **rate_limiter**: Connection rate limiting for preventing connection storms and resource exhaustion
//! - **reputation**: Peer reputation system for tracking and scoring peer behavior over time
//! - **metrics_aggregator**: Time-series metrics aggregation with statistical analysis and trend tracking
//! - **load_tester**: Load testing and stress testing utilities for performance validation
//! - **traffic_analyzer**: Traffic pattern analysis and anomaly detection for network insights
//! - **network_simulator**: Network condition simulation for testing under adverse conditions
//! - **policy**: Network policy engine for fine-grained control over operations
//! - **utils**: Common utility functions for formatting, parsing, and network operations
//!
//! ## Examples
//!
//! See the `examples/` directory for more comprehensive examples:
//! - `network_facade_demo.rs`: Using the NetworkFacade for easy integration
//! - `basic_node.rs`: Creating and starting a basic network node
//! - `dht_operations.rs`: Content announcement and provider discovery
//! - `connection_management.rs`: Connection tracking and bandwidth monitoring

pub mod adaptive_bandwidth_allocator;
pub use adaptive_bandwidth_allocator::{
    AdaptiveBandwidthAllocator,
    AllocationPolicy,
    AllocatorConfig,
    AllocatorError,
    BandwidthClass,
    BandwidthEvent,
    // BandwidthStats collides with bandwidth_monitor::BandwidthStats → Aba prefix.
    BandwidthStats as AbaBandwidthStats,
    BandwidthWindow,
    PeerBandwidthProfile,
};

pub mod adaptive_peer_scheduler;
pub use adaptive_peer_scheduler::{
    AdaptivePeerScheduler,
    // BackpressureSignal conflicts with rate_limiter::BackpressureSignal → alias.
    BackpressureSignal as ApsBackpressureSignal,
    // PeerMetrics conflicts with peer_scoring::PeerMetrics → alias.
    PeerMetrics as ApsPeerMetrics,
    ScheduleSlot,
    SchedulerConfig as ApsSchedulerConfig,
    SchedulerStats as ApsSchedulerStats,
};

pub mod anti_entropy;
pub use anti_entropy::{
    AntiEntropyConfig, DigestEntry, GossipAntiEntropy, MerkleDigest, ReconcileResult,
};

pub mod block_transfer;
pub use block_transfer::{
    BlockTransfer, StreamingBlockTransfer, TransferChunk, TransferDirection, TransferManagerStats,
    TransferState,
};

pub mod bandwidth_allocator;
pub use bandwidth_allocator::{
    AllocationStats, AllocationStrategy, PeerAllocation, PeerBandwidthAllocator,
};

pub mod bandwidth_budget;
pub use bandwidth_budget::{
    BandwidthBudgetManager, BandwidthQuota, BudgetConfig, BudgetStats, PeerBucket,
};

pub mod bandwidth_monitor;
pub use bandwidth_monitor::{
    BandwidthAnomaly,
    BandwidthMonitor,
    BandwidthMonitorStats,
    BandwidthSample,
    BandwidthStats,
    BandwidthStatsSnapshot,
    Direction,
    MonitorConfig,
    PeerBandwidth,
    // Tick-based peer bandwidth monitor
    PeerBandwidthMonitor,
    PeerBandwidthWindow,
    TickBandwidthSample,
};

pub mod adaptive_timeout;
pub use adaptive_timeout::{
    AdaptiveTimeoutConfig, PeerAdaptiveTimeout, RttSample, TimeoutEstimate, TimeoutStats,
};

pub mod adaptive_lookup;
pub mod adaptive_polling;
pub mod arm_profiler;
pub mod auto_tuner;
pub mod background_mode;
pub mod batch_resolver;
pub mod benchmarking;
pub mod bitswap;
pub mod bootstrap;
pub mod bootstrap_coordinator;
pub mod capability_registry;
pub mod cert_pin;
pub mod churn_resilience;
pub mod connection_drainer;
pub mod content_routing_optimizer;
pub mod identity;
pub mod mesh_repair;
pub mod relay;
pub use connection_drainer::{
    ConnectionDrainer, DrainState, DrainableConnection, DrainerConfig, DrainerStats,
};
pub mod connection_health_monitor;
pub use connection_health_monitor::{
    AlertThresholds, ChmAlertSeverity, ChmHealthAlert, ChmHealthMetric, ChmHealthSample,
    ChmMonitorStats, ConnectionHealth, ConnectionHealthMonitor,
};

pub mod connection_manager;
pub mod connection_migration;
pub mod dht;
pub mod dht_optimizer;
pub mod dht_provider;
pub mod diagnostics;
pub mod event_bus;
pub mod facade;
pub mod fallback;
pub mod geo_routing;
pub mod gossipsub;
pub mod health;
pub mod ipfs_compat;
pub mod load_tester;
pub mod logging;
pub mod memory_monitor;
pub mod metrics;
pub mod metrics_aggregator;
pub mod multipath_quic;
pub mod nat_traversal;
pub mod nat_traversal_manager;
pub use nat_traversal_manager::{
    fnv1a_64 as ntm_fnv1a_64,
    xorshift64 as ntm_xorshift64,
    CandidateAddress,
    CandidateType,
    IcePair,
    // NatTraversalManager collides with nat_traversal::NatTraversalManager → Ntm prefix.
    NatTraversalManager as NtmNatTraversalManager,
    // NatType collides with nat_traversal::NatType → Ntm prefix.
    NatType as NtmNatType,
    PairState,
    StunAttribute,
    StunMessage,
    StunMessageType,
    TraversalConfig as NtmTraversalConfig,
    TraversalError,
    TraversalStats as NtmTraversalStats,
};
pub mod network_monitor;
pub mod network_simulator;
pub mod node;
pub mod offline_queue;
pub mod peer;
pub mod peer_blacklist;
pub use peer_blacklist::{
    BlacklistConfig, BlacklistEntry, BlacklistReason, BlacklistStats, PeerBlacklist,
};

pub mod peer_capabilities;
pub mod peer_discovery;
pub mod peer_exchange;
pub mod peer_migration;
pub use peer_exchange::{PeerExchangeProtocol, PeerSource, PexConfig, PexPeerRecord, PexStats};
pub use peer_migration::{
    MigrationItem as PmMigrationItem, PeerMigrationConfig as PmMigrationConfig,
    PeerMigrationManager, PeerMigrationRecord, PeerMigrationState as PmMigrationState,
    PeerMigrationStats as PmMigrationStats,
};
pub mod lookup_cache;
pub mod peer_health;
pub mod peer_score;
pub mod peer_selector;
pub mod policy;
pub mod presets;
pub mod protocol;
pub mod provider_renewal;
pub mod providers;
pub mod quality_predictor;
pub mod query_batcher;
pub mod query_optimizer;
pub mod quic;
pub mod rate_limiter;
pub mod reputation;
pub mod semantic_dht;
pub mod session;
pub mod throttle;
pub mod topic_router;
pub mod tor;
pub mod traffic_shaper;
pub use traffic_shaper::{
    tokens_available,
    xorshift64 as ts_xorshift64,
    xorshift_f64 as ts_xorshift_f64,
    DropPolicy,
    PeerShaperStats,
    PeerTokenBucket,
    // Legacy peer shaper types (renamed to avoid clashes)
    PeerTrafficClass,
    PeerTrafficShaper,
    QueueEntry,
    QueuingDiscipline,
    ShaperConfig,
    ShaperError,
    ShaperEvent,
    ShaperStats,
    // New discipline-based shaper types
    TrafficClass,
    TrafficShaper,
    TrafficToken,
};
pub mod routing_table;
pub mod traffic_analyzer;
pub mod utils;

pub use adaptive_lookup::{
    AdaptiveLookupScheduler, LookupSchedulerStats, PeerLatencyTracker, ALPHA_DEFAULT, ALPHA_MAX,
    ALPHA_MIN,
};
pub use adaptive_polling::{
    ActivityLevel, AdaptivePolling, AdaptivePollingConfig, AdaptivePollingError,
    AdaptivePollingStats,
};
pub use arm_profiler::{
    ArmDevice, ArmProfiler, PerformanceSample, PerformanceStats, ProfilerConfig, ProfilerError,
};
pub use auto_tuner::{
    AutoTuner, AutoTunerConfig, AutoTunerError, AutoTunerStats, SystemResources, WorkloadProfile,
};
pub use background_mode::{
    BackgroundModeConfig, BackgroundModeError, BackgroundModeManager, BackgroundModeStats,
    BackgroundState,
};
pub use batch_resolver::{
    BatchCidResolver, BatchResolverStats, BatchResolverStatsSnapshot, CachedResult, LookupHandle,
    PendingLookup, PrefetchScheduler,
};
pub use benchmarking::{
    BenchmarkConfig, BenchmarkError, BenchmarkResult, BenchmarkType, PerformanceBenchmark,
};
pub use bitswap::{Bitswap, BitswapEvent, BitswapMessage, BitswapStats};
pub use bootstrap::{BootstrapConfig, BootstrapManager, BootstrapStats};
pub use bootstrap_coordinator::{
    BootstrapCoordinator, BootstrapPeer, BootstrapStats as CoordinatorBootstrapStats,
    BootstrapStatsSnapshot, DiscoveryRecord,
};
pub use capability_registry::{
    // Protocol-level capability types (PeerCapabilityRegistry)
    Capability,
    CapabilityAdvertisement,
    CapabilityRegistryStats,
    NodeCapabilities,
    // Node-level capability types (NodeCapabilityRegistry)
    NodeCapability,
    NodeCapabilityRegistry,
    PeerCapabilityRegistry,
};
pub use cert_pin::{CertFingerprint, CertPinStore, PeerCertPin, PinPolicy, VerificationResult};
pub use churn_resilience::{
    AdaptiveRefreshConfig, AdaptiveRefreshScheduler, ChurnEventType, ChurnMetrics,
    ChurnResilienceManager, PeerChurnEvent, PeerChurnTracker,
};
pub use connection_manager::{
    ConnectionDirection, ConnectionLimitsConfig, ConnectionManager, ConnectionManagerStats,
};
pub use connection_migration::{
    ConnectionMigrationManager, MigrationAttempt, MigrationConfig, MigrationError, MigrationState,
    MigrationStats,
};
pub use dht::{
    DhtConfig, DhtHealth, DhtHealthStatus, DhtManager, DhtStats, ProviderReannouncer,
    ReannounceStats,
};
pub use dht_provider::{
    DhtCapabilities, DhtPeerInfo, DhtProvider, DhtProviderError, DhtProviderRegistry,
    DhtProviderStats, DhtQueryResult,
};
pub use diagnostics::{
    ConfigDiagnostics, ConfigIssue, DiagnosticResult, DiagnosticTest, NetworkDiagnostics,
    PerformanceMetrics, TroubleshootingGuide,
};
pub use facade::{NetworkFacade, NetworkFacadeBuilder};
pub use fallback::{FallbackConfig, FallbackManager, FallbackResult, FallbackStrategy, RetryStats};
pub use geo_routing::{
    GeoLocation, GeoPeer, GeoRegion, GeoRouter, GeoRouterConfig, GeoRouterStats,
};
pub use gossipsub::{
    topics as gossipsub_topics, GossipSubConfig, GossipSubError, GossipSubManager,
    GossipSubMessage, GossipSubStats, IpfrsTopic, MeshHealthMonitor, MeshHealthStatus, MessageId,
    PeerScore, TopicId, TopicMessage, TopicSubscription,
};
pub use health::{
    ComponentHealth, HealthChecker, HealthHistory, NetworkHealth, NetworkHealthStatus,
};
pub use identity::{IdentityError, PeerIdentityManager, PreviousKeypair, RotationRecord};
pub use ipfs_compat::{
    ipfs_test_config, test_ipfs_connectivity, IpfsCompatTestResults, IPFS_BOOTSTRAP_NODES,
    TEST_CIDS,
};
pub use load_tester::{
    LoadTestConfig, LoadTestError, LoadTestMetrics, LoadTestResults, LoadTestType, LoadTester,
};
pub use logging::{
    connection_span, dht_span, network_span, LogLevel, LoggingConfig, NetworkEventType,
    OperationContext,
};
pub use lookup_cache::{
    CachedProviders, LookupCache, LookupCacheConfig, LookupCacheStats, ParallelLookupConfig,
    ParallelLookupExecutor, ParallelLookupResult,
};
pub use memory_monitor::{
    ComponentMemory, MemoryMonitor, MemoryMonitorConfig, MemoryMonitorError, MemoryStats,
};
pub use mesh_repair::{MeshRepairConfig, MeshRepairCoordinator, MeshRepairState};
pub use metrics::{
    BandwidthMetricsSnapshot, ConnectionMetricsSnapshot, DhtMetricsSnapshot, MetricsSnapshot,
    NetworkMetrics, SharedMetrics,
};
pub use metrics_aggregator::{
    AggregatedStatistics, AggregatorConfig, MetricStatistics, MetricsAggregator, TimeWindow,
};
pub use multipath_quic::{
    MultipathConfig, MultipathError, MultipathQuicManager, MultipathStats, NetworkPath, PathId,
    PathQuality, PathSelectionStrategy, PathState,
};
pub use nat_traversal::{
    HolePunchAttempt, HolePunchStatus, NatTraversalConfig, NatTraversalManager, NatTraversalStats,
    NatType, StunBinding, TraversalStrategy,
};
pub use network_monitor::{
    InterfaceType, NetworkChange, NetworkInterface, NetworkMonitor, NetworkMonitorConfig,
    NetworkMonitorError, NetworkMonitorStats,
};
pub use network_simulator::{
    NetworkCondition, NetworkSimulator, SimulatorConfig, SimulatorError, SimulatorStats,
};
pub use node::{
    BucketInfo, ConnectionEndpoint, InferenceWaiters, KademliaConfig, NatTraversalMetrics,
    NetworkConfig, NetworkEvent, NetworkHealthLevel, NetworkHealthSummary, NetworkNode,
    NetworkStats, RelayConfig, RoutingTableInfo,
};
pub use offline_queue::{
    OfflineQueue, OfflineQueueConfig, OfflineQueueError, OfflineQueueStats, QueuedRequest,
    QueuedRequestType, RequestPriority,
};
pub use peer::{PeerInfo, PeerStore, PeerStoreConfig, PeerStoreStats};
pub use peer_capabilities::{
    CapabilityConfig,
    CapabilityRegistry,
    CapabilityStats,
    // Advertised as PeerCapability to avoid collision with capability_registry::Capability
    PeerCapability,
    PeerCapabilitySet,
};
pub use peer_selector::{
    PeerSelector, PeerSelectorConfig, PeerSelectorStats, SelectedPeer, SelectionCriteria,
};
pub use policy::{
    BandwidthPolicy, ConnectionPolicy, ContentPolicy, PolicyAction, PolicyConfig, PolicyEngine,
    PolicyError, PolicyResult, PolicyStats,
};
pub use presets::NetworkPreset;
pub use protocol::{
    ProtocolCapabilities, ProtocolHandler, ProtocolId, ProtocolRegistry, ProtocolVersion,
};
pub use provider_renewal::{
    ProviderRecord, ProviderRenewalScheduler, RenewalConfig, DEFAULT_PROVIDER_TTL_SECS,
    DEFAULT_RENEWAL_THRESHOLD,
};
pub use providers::{ProviderCache, ProviderCacheConfig, ProviderCacheStats};
pub use quality_predictor::{
    QualityPrediction, QualityPredictor, QualityPredictorConfig, QualityPredictorError,
    QualityPredictorStats,
};
pub use query_batcher::{
    PendingQuery, QueryBatchResult, QueryBatcher, QueryBatcherConfig, QueryBatcherError,
    QueryBatcherStats, QueryType,
};
pub use query_optimizer::{QueryMetrics, QueryOptimizer, QueryOptimizerConfig, QueryResult};
pub use quic::{
    CongestionControl, QuicConfig, QuicConnectionInfo, QuicConnectionState, QuicMonitor, QuicStats,
};
pub use rate_limiter::{
    AtomicRateLimiterStats,
    // New atomic types
    AtomicTokenBucket,
    BackpressureController,
    BackpressureSignal,
    ConnectionPriority,
    ConnectionRateLimiter,
    GlobalLimiter,
    PeerLimiter,
    PeerRateLimiter,
    PeerRateLimiterConfig,
    PeerRateLimiterStats,
    RateLimitDecision,
    // PeerRateLimiter
    RateLimitResult,
    RateLimiter,
    RateLimiterConfig,
    RateLimiterError,
    RateLimiterStats,
    RateLimiterStatsSnapshot,
};
pub use relay::{RelayError, RelayManager, RelayReservation};
pub use reputation::{
    PeerReputation, PeerReputationEvent, PeerReputationStats, PeerReputationStatsSnapshot,
    PeerReputationTracker, PeerTier, ReputationConfig, ReputationEvent, ReputationManager,
    ReputationScore, ReputationStats,
};
pub use semantic_dht::{
    DistanceMetric, LshConfig, LshHash, MergeResult, NamespaceId, PartialSyncConfig,
    PartialSyncStats, SearchResult, SemanticDht, SemanticDhtConfig, SemanticDhtError,
    SemanticDhtMetrics, SemanticDhtStats, SemanticNamespace, SemanticQuery, SemanticResult,
    ShardBalancer, ShardBalancerConfig, VectorAnnotatedRecord,
};
pub use session::{
    Session, SessionConfig, SessionManager, SessionMetadata,
    SessionState as ConnectionSessionState, SessionStats,
};
pub use throttle::{
    BandwidthThrottle, ThrottleConfig, ThrottleError, ThrottleStats, TrafficDirection,
};
pub use topic_router::{
    PrioritizedMessage, TopicConfig, TopicError, TopicRouter, TopicRouterStats,
    TopicRouterStatsSnapshot,
};
pub use tor::{
    CircuitId, CircuitInfo, CircuitState, HiddenServiceConfig, OnionAddress, StreamId, TorConfig,
    TorError, TorManager, TorStats,
};
pub use traffic_analyzer::{
    AnomalyType, PatternType, PeerProfile, TrafficAnalysis, TrafficAnalyzer, TrafficAnalyzerConfig,
    TrafficAnalyzerError, TrafficAnalyzerStats, TrafficAnomaly, TrafficEvent, TrafficPattern,
    TrendDirection,
};

pub use routing_table::{
    ContentRoutingTable, RoutingEntry, RoutingError, RoutingTableStats, DEFAULT_ENTRY_TTL,
    DEFAULT_MAX_PROVIDERS,
};

pub mod content_routing_cache;
pub use content_routing_cache::{
    CacheConfig as CrcCacheConfig,
    CacheStats as CrcCacheStats,
    ContentRoutingCache,
    // ProviderRecord conflicts with provider_renewal::ProviderRecord → Crc prefix.
    CrcProviderRecord,
    NegativeCacheEntry,
    RoutingHint,
    DEFAULT_HINT_TTL_MS as CRC_DEFAULT_HINT_TTL_MS,
    DEFAULT_MAX_HINTS as CRC_DEFAULT_MAX_HINTS,
    DEFAULT_MAX_NEGATIVE as CRC_DEFAULT_MAX_NEGATIVE,
    DEFAULT_MAX_PROVIDERS as CRC_DEFAULT_MAX_PROVIDERS,
    DEFAULT_NEGATIVE_TTL_MS as CRC_DEFAULT_NEGATIVE_TTL_MS,
    DEFAULT_PROVIDER_TTL_MS as CRC_DEFAULT_PROVIDER_TTL_MS,
};

pub mod routing_table_sharding;
pub use routing_table_sharding::{
    EvictionPolicy,
    // NodeId conflicts with routing_table_manager::NodeId → alias.
    NodeId as RtsNodeId,
    // RoutingEntry conflicts with routing_table::RoutingEntry → alias.
    RoutingEntry as RtsRoutingEntry,
    RoutingTableSharding,
    ShardConfig,
    ShardId,
    ShardStats,
};

pub mod routing_table_manager;
pub use routing_table_manager::{
    bucket_index as rtm_bucket_index, xor_distance as rtm_xor_distance, BucketEntry, KBucket,
    NodeId, RoutingTableManager, RoutingTableStats as RtmRoutingTableStats,
    DEFAULT_ALPHA as RTM_DEFAULT_ALPHA, DEFAULT_K as RTM_DEFAULT_K,
    DEFAULT_REPLACEMENT_CACHE_SIZE as RTM_DEFAULT_REPLACEMENT_CACHE_SIZE,
};

pub mod peer_capability_negotiator;
pub use peer_capability_negotiator::{
    CapabilityPolicy,
    CapabilityVersion,
    NegotiationOffer,
    NegotiationRecord,
    // NegotiationResult conflicts with protocol_negotiator::NegotiationResult → Pcn prefix.
    NegotiationResult as PcnNegotiationResult,
    // NegotiatorConfig conflicts with protocol_negotiator::NegotiatorConfig → Pcn prefix.
    NegotiatorConfig as PcnNegotiatorConfig,
    NegotiatorStats,
    // PeerCapability conflicts with peer_capabilities::PeerCapability → Pcn prefix.
    PeerCapability as PcnPeerCapability,
    PeerCapabilityNegotiator,
};

pub mod protocol_handshake;
pub mod protocol_negotiator;
pub use protocol_negotiator::{
    NegotiationResult,
    NegotiatorConfig,
    PeerNegotiationResult,
    PeerNegotiatorStats,
    PeerProtocolNegotiator,
    PeerProtocolVersion,
    // Pn-prefixed rich negotiation system
    PnNegotiatedSession,
    PnNegotiationOutcome,
    PnNegotiationRecord,
    PnNegotiationStats,
    PnNegotiatorConfig,
    PnProtocolId,
    PnProtocolNegotiator,
    PnProtocolVersion,
    PnSessionId,
    ProtocolFeature,
    ProtocolNegotiator,
    ProtocolOffer,
};
pub mod protocol_version;
pub use protocol_handshake::{
    FeatureFlag, HandshakeError, HandshakeOffer, HandshakeResult, HandshakeStats,
    HandshakeStatsSnapshot, ProtocolHandshaker, ProtocolVersion as HandshakeProtocolVersion,
    DEFAULT_MAX_FRAME_SIZE,
};
pub use protocol_version::{
    CompatibilityLevel, NegotiationResult as PvNegotiationResult, ProtocolDescriptor,
    ProtocolVersion as PvProtocolVersion, ProtocolVersionManager, VersionStats,
};

pub mod gossip_metrics;
pub use gossip_metrics::{GossipEvent, GossipMetricsSnapshot, MessageTrace, PeerGossipMetrics};

pub mod gossip_overlay;
pub use gossip_overlay::{
    GossipFanout, GossipMessage, GossipOverlayManager, GossipState, GossipStats,
    GossipStatsSnapshot,
};

pub mod connection_pool;
pub use connection_pool::{
    ConnectionPool,
    ConnectionState,
    // Tick-based peer pool
    PeerConnectionPool,
    PeerPoolConfig,
    PeerPoolStats,
    PeerPooledConnection,
    PoolConfig,
    PoolConnectionState,
    PoolError,
    PoolStats,
    PoolStatsSnapshot,
    PooledConnection,
};

pub mod connection_tracker;

pub mod message_router;
pub use message_router::{
    HandlerRegistration,
    MessagePriority,
    MessageRouter,
    // PeerMessageRouter and related types
    MessageRouterStats,
    MessageType,
    PeerMessageRouter,
    PeerRoutedMessage,
    RouteRule,
    RoutedMessage,
    RouterError,
    RouterStats,
    RouterStatsSnapshot,
};

pub mod subscription_router;
pub use subscription_router::{
    // FNV-1a helper exposed for downstream use
    fnv1a_64 as mr_fnv1a_64,
    // SubscriptionRouter and related types
    DeliveryRecord,
    MessageTopic,
    RoutingMessage,
    SubRouterStats,
    Subscription,
    SubscriptionFilter,
    SubscriptionRouter,
};

pub mod request_dedup;

pub use request_dedup::{
    AcquireResult, DedupStats, DedupStatsSnapshot, RequestDeduplicator, ResolveResult, WaiterHandle,
};

// Re-export commonly used utility functions
pub use utils::{
    exponential_backoff, format_bandwidth, format_bytes, format_duration, is_local_addr,
    is_public_addr, jittered_backoff, moving_average, parse_multiaddr, parse_multiaddrs,
    peers_match, percentage, truncate_peer_id, validate_alpha,
};

pub mod peer_discovery_manager;
pub use peer_discovery_manager::{
    ConnectOutcome, DiscoveryConfig as PdmDiscoveryConfig, DiscoveryMethod,
    DiscoveryStats as PdmDiscoveryStats, PeerCandidate,
    PeerDiscoveryManager as PdmPeerDiscoveryManager,
};

pub mod discovery_cache;
pub use discovery_cache::{DiscoveryCacheStats, DiscoverySource, PeerDiscoveryCache, PeerRecord};

pub mod discovery_lru;
pub use discovery_lru::{
    CacheEntry as LruCacheEntry, DiscoveryCacheConfig as LruDiscoveryCacheConfig,
    LruDiscoveryCacheStats, LruPeerDiscoveryCache,
};

pub mod circuit_breaker;
pub use circuit_breaker::{
    CallResult, CircuitBreakerRegistry, CircuitBreakerState, CircuitConfig, CircuitStats,
    PeerCircuit, PeerCircuitBreaker, PeerCircuitState, RegistryStats,
};

pub mod topology_mapper;
pub use topology_mapper::{
    // Primary type — aliased to avoid collision with network_topology_mapper::NetworkTopologyMapper.
    NetworkTopologyMapper as LegacyNetworkTopologyMapper,
    PathResult,
    // Legacy aliases kept for backwards compatibility
    PeerEdge,
    TopoEdge,
    TopoNode,
    TopologyNode,
    TopologySnapshot,
    TopologyStats,
};

pub mod message_dedup;
pub use message_dedup::{DedupConfig, DedupEntry, MessageDeduplicator, MsgDedupStats, MsgId};

pub mod message_batcher;
pub use message_batcher::{
    BatchConfig, BatchFlush, BatchMessage, BatcherStats, FlushReason, PeerMessageBatcher,
};

pub mod message_codec;
pub use message_codec::{CodecConfig, CodecError, CodecStats, EncodedMessage, PeerMessageCodec};

pub mod message_prioritizer;
pub use message_prioritizer::{
    AgingConfig, MessagePriority as PeerMessagePriority, PeerMessagePrioritizer,
    PrioritizedMessage as PeerPrioritizedMessage, PrioritizerStats,
};

pub mod priority_queue;
pub use priority_queue::{
    MessagePriority as PeerQueuePriority, PeerPriorityQueue, QueueConfig, QueueStats, QueuedMessage,
};

pub mod latency_tracker;
pub use latency_tracker::{
    LatencyBucket,
    LatencyTrackerStats,
    PeerLatency,
    // Re-export with alias to avoid collision with adaptive_lookup::PeerLatencyTracker.
    PeerLatencyTracker as HistogramLatencyTracker,
};

pub mod latency_predictor;
pub use latency_predictor::{
    LatencySample, PeerLatencyPredictor, PeerLatencyState, PredictorConfig, PredictorStats,
    TrendDirection as LatencyTrendDirection,
};

pub mod peer_session;
pub use peer_session::{
    PeerSession as AuthPeerSession, PeerSessionManager as AuthPeerSessionManager,
    SessionCapability, SessionManagerConfig as AuthSessionManagerConfig, SessionToken,
};

pub mod session_manager;
pub use session_manager::{
    PeerSession, PeerSessionEntry, PeerSessionManager, PeerSessionState, SessionDirection,
    SessionManagerConfig, SessionManagerStats, SessionState,
};

pub mod connection_limiter;
pub use connection_limiter::{
    LimiterConfig, LimiterStats, PeerConnectionInfo as LimiterPeerConnectionInfo,
    PeerConnectionLimiter,
};

pub mod connection_health;
pub use connection_health::{
    ConnectionEvent, ConnectionHealthChecker, ConnectionHealthState, ConnectionRecord,
    HealthCheckerConfig,
};

pub mod announcement_manager;
pub use announcement_manager::{
    AnnouncementChannel, AnnouncementConfig, AnnouncementRecord, AnnouncementStats,
    PeerAnnouncementManager,
};

pub mod routing_auditor;
pub use routing_auditor::{
    AuditFinding, AuditSeverity, AuditorConfig, BucketInfo as AuditorBucketInfo,
    RoutingTableAuditor, DEFAULT_MAX_CAPACITY,
};

pub mod peer_reputation;
pub use peer_reputation::{
    PeerReputationManager, PrReputationConfig, PrReputationEvent, PrReputationScore,
    PrReputationStats,
};

pub mod ban_list;
pub use ban_list::{BanConfig, BanEntry, BanKind, BanListStats, PeerBanList};

pub mod behavior_classifier;
pub use behavior_classifier::{
    BehaviorProfile, BehaviorSignal, ClassifierStats, PeerBehaviorClassifier,
};

pub mod sync_coordinator;
pub use sync_coordinator::{PeerSyncCoordinator, SyncDirection, SyncPhase, SyncSession, SyncStats};

pub mod peer_sync_protocol;
pub use peer_sync_protocol::{
    ConflictPolicy, PeerSyncProtocol, PspSyncStats, SyncEntry, SyncError, SyncOperation, SyncState,
    VectorClock,
};

pub mod congestion_controller;
pub use congestion_controller::{
    CccAlgorithm,
    CccCongestionController,
    CccConnId,
    CccConnection,
    CccControllerConfig,
    CccControllerStats,
    CccDecision,
    CccEvent,
    CccEventType,
    CccState,
    CongestionConfig,
    // New multi-algorithm controller.
    CongestionController,
    CongestionEvent,
    CongestionState,
    Decision,
    MultiPeerCongestionManager,
    PeerCongestionController,
    WindowStats,
};

pub mod load_balancer;
pub use load_balancer::{
    AdaptiveLbStats, AdaptiveLoadBalancer, LbAlgorithm, LbDecision, LbPeer, LbRequest,
};
pub use load_balancer::{LbStats, LbStrategy, PeerLoad, PeerLoadBalancer};

pub mod trust_manager;
pub use trust_manager::{
    PeerTrustManager, TrustAttestation, TrustLevel, TrustManagerStats, TrustRecord,
};

pub mod flow_control;
pub use flow_control::{FlowControlConfig, FlowControlStats, FlowWindow, PeerFlowControl};

pub mod gossip_filter;
pub use gossip_filter::{
    FilterConfig, FilterStats, FilterVerdict, GossipMessage as FilterGossipMessage,
    PeerGossipFilter,
};

pub mod gossip_content_filter;
pub use gossip_content_filter::{ContentGossipFilter, FilterAction, FilterRule, GossipFilterStats};

pub mod bloom_filter;
pub use bloom_filter::{BloomConfig, BloomFilter, BloomStats, PeerBloomFilter};

pub mod churn_manager;
pub use churn_manager::{
    ChurnEvent, ChurnManagerConfig, ChurnStats, ChurnWindow, PeerChurnManager, PeerLifetime,
};

pub mod request_priority_queue;
pub use request_priority_queue::{
    PeerRequest, PeerRequestQueue, PriorityQueueStats as RequestQueueStats,
    RequestPriority as PeerRequestPriority,
};

pub mod health_checker;
pub use health_checker::{
    HealthCheckerStats as PeerHealthCheckerStats,
    // HealthTier may clash with other tier enums; export with a peer-specific alias.
    HealthTier as PeerHealthTier,
    HeartbeatRecord,
    PeerHealthChecker,
    PeerHealthCheckerConfig,
};

pub mod scoreboard;
pub use scoreboard::{PeerScoreboard, SbPeerScore, ScoreComponent, ScoreboardStats};

pub mod peer_scoring;
pub use peer_scoring::{
    PeerMetrics,
    PeerScoringSystem,
    // PeerScore already exported from gossipsub; use prefixed alias.
    PsPeerScore,
    // ScoreComponent already exported from scoreboard (as a struct); use prefixed alias.
    PsScoringDimension,
    ScoreTier,
    ScoringError,
    ScoringStats,
    ScoringWeights,
};

pub mod overlay_network;
pub use overlay_network::{
    OverlayError, OverlayMessage, OverlayNetworkManager, OverlayNode, OverlayRoute, OverlayStats,
    OverlayTopology,
};

pub mod overlay_network_manager;
pub use overlay_network_manager::{
    fnv1a_64 as onm_fnv1a_64,
    xorshift64 as onm_xorshift64,
    // New types with no crate-root collision.
    OverlayConfig,
    // Collide with overlay_network exports → Onm prefix.
    OverlayError as OnmOverlayError,
    OverlayLink,
    OverlayNetworkManager as OnmOverlayNetworkManager,
    OverlayNode as OnmOverlayNode,
    OverlayStats as OnmOverlayStats,
    OverlayTopology as OnmOverlayTopology,
    RoutingPolicy,
    VirtualRoute,
};

pub mod flood_protection;
pub use flood_protection::{
    fnv1a_message_id,
    CheckResult as FpCheckResult,
    FloodConfig,
    FloodProtection,
    FloodStats,
    // Export MessageId with Fp prefix to avoid collision with gossipsub::MessageId.
    MessageId as FpMessageId,
    PeerState as FpPeerState,
    ViolationRecord,
    ViolationType,
};

pub mod security_monitor;
pub use security_monitor::{
    IncidentStatus, NetworkSecurityMonitor, SecurityEvent, SecurityIncident, SecurityMonitorStats,
    ThreatLevel, ThreatScore, ThreatType,
};

pub mod stream_multiplexer;
pub use stream_multiplexer::{
    priority_from_u8 as smx_priority_from_u8,
    xorshift64 as smx_xorshift64,
    FrameFlags,
    LogicalStream,
    MultiplexerConfig,
    MultiplexerStats,
    MuxError,
    MuxEvent,
    MuxStats,
    StreamFrame,
    // StreamId is already exported from tor; alias to avoid collision.
    StreamId as SmxStreamId,
    StreamInfo,
    StreamMultiplexer,
    StreamPriority,
    // StreamState is already exported from tor; alias to avoid collision.
    StreamState as SmxStreamState,
    FLAG_FIN as SMX_FLAG_FIN,
    FLAG_RST as SMX_FLAG_RST,
    FLAG_SYN as SMX_FLAG_SYN,
};

pub mod merkle_proof_verifier;
pub use merkle_proof_verifier::{
    sha256 as mpv_sha256,
    MerkleHashAlgo,
    MerkleNode,
    MerkleProof,
    MerkleProofVerifier,
    MerkleTree,
    ProofStep,
    // VerificationResult conflicts with cert_pin::VerificationResult → alias.
    VerificationResult as MpvVerificationResult,
};

pub mod peer_bandwidth_manager;
pub use peer_bandwidth_manager::{
    BandwidthDirection, BandwidthLimit, BandwidthManagerConfig, BandwidthManagerStats,
    BandwidthUsage, FairnessPolicy, PeerBandwidthManager, PeerBandwidthState,
};

pub mod gossip_message_filter;
pub use gossip_message_filter::{
    fnv1a_64 as gmf_fnv1a_64,
    // FilterConfig conflicts with gossip_filter::FilterConfig → Gmf prefix.
    FilterConfig as GmfFilterConfig,
    // FilterRule conflicts with gossip_content_filter::FilterRule → Gmf prefix.
    FilterRule as GmfFilterRule,
    // FilterStats conflicts with gossip_filter::FilterStats → Gmf prefix.
    FilterStats as GmfFilterStats,
    // FilterVerdict conflicts with gossip_filter::FilterVerdict → Gmf prefix.
    FilterVerdict as GmfFilterVerdict,
    // GossipMessage conflicts with gossip_overlay::GossipMessage → Gmf prefix.
    GossipMessage as GmfGossipMessage,
    GossipMessageFilter,
    // MessageId conflicts with gossipsub::MessageId → Gmf prefix.
    MessageId as GmfMessageId,
};

pub mod peer_trust_scorer;
pub use peer_trust_scorer::{
    PeerTrustProfile, PeerTrustScorer, TrustBand, TrustConfig, TrustDimension, TrustEvent,
    TrustScorerStats,
};

pub mod network_event_bus;
pub use network_event_bus::{
    BusError,
    EventBusConfig,
    EventBusStats,
    EventFilter,
    EventTopic,
    // NebNetworkEvent is the bus-specific event struct; node::NetworkEvent is
    // a different type, so we keep the Neb prefix to avoid collision.
    NebNetworkEvent,
    // NebSubscription avoids collision with subscription_router::Subscription.
    NebSubscription,
    NetworkEventBus,
    SubscriberId,
};

pub mod network_qos_manager;
pub use network_qos_manager::{
    NetworkQoSManager,
    QoSConfig,
    QoSPacket,
    QoSStats,
    QueueMetrics,
    SLASpec,
    SLAViolation,
    // TrafficClass conflicts with traffic_shaper::TrafficClass which is not re-exported
    // at crate root, so no alias needed here.
    TrafficClass as QosTrafficClass,
};

pub mod flood_sub_router;
pub use flood_sub_router::{
    FloodMessage,
    FloodMessageId,
    FloodSubRouter,
    FloodTopic,
    ForwardDecision,
    // FsrRouterStats is already prefixed to avoid collision with
    // message_router::RouterStats which is exported as RouterStats.
    FsrRouterStats,
    // RouterConfig conflicts with nothing at crate root → no alias needed.
    RouterConfig as FsrRouterConfig,
    SubscriptionRecord as FsrSubscriptionRecord,
};

pub mod peer_reputation_graph;
pub use peer_reputation_graph::{
    GraphConfig,
    GraphError,
    GraphStats,
    PeerReputationGraph,
    // ReputationEvent collides with reputation::ReputationEvent → Prg prefix.
    ReputationEvent as PrgReputationEvent,
    // ReputationScore collides with reputation::ReputationScore → Prg prefix.
    ReputationScore as PrgReputationScore,
    TrustEdge,
};

pub mod gossip_protocol_engine;
pub use gossip_protocol_engine::{
    fnv1a_64 as gpe_fnv1a_64,
    xorshift64 as gpe_xorshift64,
    EngineError,
    FanoutStrategy,
    GossipConfig,
    // GossipEvent conflicts with gossip_metrics::GossipEvent → Gpe prefix.
    GossipEvent as GpeGossipEvent,
    // GossipMessage conflicts with gossip_overlay::GossipMessage → Gpe prefix.
    GossipMessage as GpeGossipMessage,
    GossipPeer,
    GossipProtocolEngine,
    // GossipStats conflicts with gossip_overlay::GossipStats → Gpe prefix.
    GossipStats as GpeGossipStats,
};

pub mod network_circuit_breaker;
pub use network_circuit_breaker::{
    xorshift64 as ncb_xorshift64,
    BreakerError,
    CircuitCallGuard,
    CircuitEvent,
    CircuitMetrics,
    CircuitOutcome,
    // CircuitConfig collides with circuit_breaker::CircuitConfig → Ncb prefix.
    NcbCircuitConfig,
    // CircuitState collides with tor::CircuitState → Ncb prefix.
    NcbCircuitState,
    NetworkCircuitBreaker,
};

pub mod peer_discovery_protocol;
pub use peer_discovery_protocol::{
    // PRNG helper (aliased to avoid collision with other xorshift64 exports).
    xorshift64 as pdp_xorshift64,
    // Type aliases (unprefixed) — no collision at crate root.
    DiscoveredPeer,
    DiscoveryConfig,
    DiscoveryError,
    DiscoveryEvent,
    DiscoveryStats,
    // Core structs and enums with Pdp prefix (canonical names).
    PdpDiscoveredPeer,
    PdpDiscoveryConfig,
    PdpDiscoveryError,
    PdpDiscoveryEvent,
    PdpDiscoveryMethod,
    PdpDiscoveryStats,
    PeerDiscoveryProtocol,
};

pub mod message_authenticator;
pub use message_authenticator::{
    fnv1a_64 as mau_fnv1a_64, hmac_fnv64 as mau_hmac_fnv64, xorshift64 as mau_xorshift64,
    AuthAlgorithm, AuthError, AuthKey, AuthPolicy, AuthStats, MessageAuthenticator, ReplayWindow,
    SignedMessage,
};

pub mod connection_pool_manager;
pub use connection_pool_manager::{
    xorshift64 as cpm_xorshift64,
    AcquirePolicy,
    ConnState,
    ConnectionPoolManager,
    // PoolConfig collides with connection_pool::PoolConfig → Cpm prefix.
    CpmPoolConfig,
    // PoolError collides with connection_pool::PoolError → Cpm prefix.
    CpmPoolError,
    // PoolStats collides with connection_pool::PoolStats → Cpm prefix.
    CpmPoolStats,
    PoolEvent,
    // PooledConnection collides with connection_pool::PooledConnection → Cpm prefix.
    PooledConnection as CpmPooledConnection,
};

pub mod adaptive_routing_engine;
pub use adaptive_routing_engine::{
    AdaptiveRoutingEngine,
    AreRouteEntry,
    AreRouteKey,
    AreRoutingConfig,
    // RoutingPolicy is already exported from overlay_network_manager → use Are prefix.
    AreRoutingPolicy,
    AreRoutingStats,
    RoutingEngineError,
};

pub mod peer_load_balancer;
pub use peer_load_balancer::{
    // PeerLoadBalancer collides with load_balancer::PeerLoadBalancer → Plb prefix.
    PeerLoadBalancer as PlbPeerLoadBalancer,
    PlbBalancerConfig,
    PlbBalancerStats,
    PlbError,
    PlbPeerId,
    PlbPeerState,
    PlbPeerStats,
    PlbRequestRecord,
    PlbStrategy,
};

pub mod network_topology_mapper;
pub use network_topology_mapper::{
    NetworkTopologyMapper, NtmEdge, NtmMapperConfig, NtmMapperError, NtmNetworkTopologyMapper,
    NtmNode, NtmSnapshot, NtmTopologyMetrics,
};

pub mod stream_priority_scheduler;
pub use stream_priority_scheduler::{
    xorshift64 as sps_xorshift64, SpsError, SpsSchedulerConfig, SpsSchedulerStats,
    SpsSchedulingPolicy, SpsStream, SpsStreamId, StreamPriorityScheduler,
};

pub mod packet_fragmentation_assembler;
pub use packet_fragmentation_assembler::{
    fnv1a_64 as pfa_fnv1a_64, xorshift64 as pfa_xorshift64, PacketFragmentationAssembler,
    PfaAssemblerConfig, PfaAssemblerStats, PfaFragment, PfaFragmentRecord, PfaMessageId,
    PfaPacketFragmentationAssembler, PfaReassemblyBuffer, PfaReceiveResult,
};

/// Re-export libp2p types
pub use libp2p;
