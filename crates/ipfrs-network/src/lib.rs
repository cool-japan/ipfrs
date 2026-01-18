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

pub mod adaptive_polling;
pub mod arm_profiler;
pub mod auto_tuner;
pub mod background_mode;
pub mod benchmarking;
pub mod bitswap;
pub mod bootstrap;
pub mod connection_manager;
pub mod connection_migration;
pub mod dht;
pub mod dht_provider;
pub mod diagnostics;
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
pub mod network_monitor;
pub mod network_simulator;
pub mod node;
pub mod offline_queue;
pub mod peer;
pub mod peer_selector;
pub mod policy;
pub mod presets;
pub mod protocol;
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
pub mod tor;
pub mod traffic_analyzer;
pub mod utils;

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
pub use benchmarking::{
    BenchmarkConfig, BenchmarkError, BenchmarkResult, BenchmarkType, PerformanceBenchmark,
};
pub use bitswap::{Bitswap, BitswapEvent, BitswapMessage, BitswapStats};
pub use bootstrap::{BootstrapConfig, BootstrapManager, BootstrapStats};
pub use connection_manager::{
    ConnectionDirection, ConnectionLimitsConfig, ConnectionManager, ConnectionManagerStats,
};
pub use connection_migration::{
    ConnectionMigrationManager, MigrationAttempt, MigrationConfig, MigrationError, MigrationState,
    MigrationStats,
};
pub use dht::{DhtConfig, DhtHealth, DhtHealthStatus, DhtManager, DhtStats};
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
    GossipSubConfig, GossipSubError, GossipSubManager, GossipSubMessage, GossipSubStats, MessageId,
    PeerScore, TopicId, TopicSubscription,
};
pub use health::{
    ComponentHealth, HealthChecker, HealthHistory, NetworkHealth, NetworkHealthStatus,
};
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
pub use memory_monitor::{
    ComponentMemory, MemoryMonitor, MemoryMonitorConfig, MemoryMonitorError, MemoryStats,
};
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
pub use network_monitor::{
    InterfaceType, NetworkChange, NetworkInterface, NetworkMonitor, NetworkMonitorConfig,
    NetworkMonitorError, NetworkMonitorStats,
};
pub use network_simulator::{
    NetworkCondition, NetworkSimulator, SimulatorConfig, SimulatorError, SimulatorStats,
};
pub use node::{
    BucketInfo, ConnectionEndpoint, KademliaConfig, NetworkConfig, NetworkEvent,
    NetworkHealthLevel, NetworkHealthSummary, NetworkNode, NetworkStats, RoutingTableInfo,
};
pub use offline_queue::{
    OfflineQueue, OfflineQueueConfig, OfflineQueueError, OfflineQueueStats, QueuedRequest,
    QueuedRequestType, RequestPriority,
};
pub use peer::{PeerInfo, PeerStore, PeerStoreConfig, PeerStoreStats};
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
    ConnectionPriority, ConnectionRateLimiter, RateLimiterConfig, RateLimiterError,
    RateLimiterStats,
};
pub use reputation::{
    ReputationConfig, ReputationEvent, ReputationManager, ReputationScore, ReputationStats,
};
pub use semantic_dht::{
    DistanceMetric, LshConfig, LshHash, NamespaceId, SemanticDht, SemanticDhtConfig,
    SemanticDhtError, SemanticDhtStats, SemanticNamespace, SemanticQuery, SemanticResult,
};
pub use session::{
    Session, SessionConfig, SessionManager, SessionMetadata, SessionState, SessionStats,
};
pub use throttle::{
    BandwidthThrottle, ThrottleConfig, ThrottleError, ThrottleStats, TrafficDirection,
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

// Re-export commonly used utility functions
pub use utils::{
    exponential_backoff, format_bandwidth, format_bytes, format_duration, is_local_addr,
    is_public_addr, jittered_backoff, moving_average, parse_multiaddr, parse_multiaddrs,
    peers_match, percentage, truncate_peer_id, validate_alpha,
};

/// Re-export libp2p types
pub use libp2p;
