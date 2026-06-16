//! IPFRS Transport - TensorSwap and data exchange protocols
//!
//! This crate implements the transport layer for IPFRS including:
//! - TensorSwap protocol for efficient tensor streaming
//! - Bitswap-compatible block exchange
//! - GraphSync for DAG traversal
//! - Enhanced want list management with priority queues
//! - Peer scoring and selection strategies
//! - QUIC transport with connection pooling
//! - Content routing with DHT integration
//! - CDN edge node caching
//! - Comprehensive diagnostics and health monitoring
//! - Automatic configuration tuning
//! - Performance statistics aggregation
//! - Observability with structured logging
//! - Prometheus metrics export
//! - Load testing utilities

pub mod advanced_scheduling;
pub mod arrow_deframer;
pub mod auto_tuner;
pub mod bitswap;
pub mod cdn_edge;
pub mod config_advisor;
pub mod connection_migration;
pub mod content_routing;
pub mod diagnostics;
pub mod erasure;
pub mod facade;
pub mod graphsync;
pub mod health_monitor;
pub mod load_tester;
pub mod messages;
pub mod metrics;
pub mod multi_transport;
pub mod multicast;
pub mod nat_traversal;
pub mod observability;
pub mod partition;
pub mod peer_manager;
pub mod prefetch;
pub mod prometheus_exporter;
pub mod quic;
pub mod range_request;
pub mod recovery;
pub mod request_coalescing;
pub mod schema_migration;
pub mod schema_registry;
pub mod session;
pub mod session_config;
pub mod stats_aggregator;
pub mod tcp;
pub mod tensorswap;
pub mod test_utils;
pub mod throttle;
pub mod transport;
pub mod utils;
pub mod want_list;
pub mod websocket;

pub use graphsync::{
    AggregationStrategy, DagTraversal, GradientAggregator, GradientAggregatorStats,
    GradientMessage, GradientStream, GraphSync, Selector, TraversalCheckpoint, TraversalMode,
    TraversalState, TraversalStats,
};

pub use bitswap::{BitswapConfig, BitswapExchange, BitswapStats};
pub use cdn_edge::{
    EdgeConfig, EdgeNode, EdgeStats, EvictionPolicy, InvalidationReason, InvalidationRequest,
    OriginServer,
};
pub use content_routing::{
    ContentRouter, ContentRoutingConfig, ContentRoutingStats, ProviderRecord,
};
pub use erasure::{
    ErasureConfig, ErasureError, ErasureManager, ErasureMetadata, Shard, SimpleErasureEncoder,
};
pub use messages::{
    BlockMessage, CancelMessage, DontHaveMessage, HaveMessage, Message,
    WantEntry as MessageWantEntry, WantList as MessageWantList,
};
pub use metrics::{
    LatencyStats, LatencyTracker, MemoryStats, MemoryTracker, MetricsConfig, ThroughputTracker,
    Timer,
};
pub use multi_transport::{
    ConnectionAttempt, MultiTransportConfig, MultiTransportManager, MultiTransportManagerBuilder,
};
pub use multicast::{
    BlockAnnouncement, MulticastConfig, MulticastError, MulticastManager, MulticastStats,
    Subscription, SubscriptionFilter, Topic,
};
pub use nat_traversal::{
    CandidatePair, CandidateType, ConnectivityEvent, IceCandidate, NatTraversalConfig,
    NatTraversalError, NatTraversalManager, NatTraversalStats, NatType, PairState, StunConfig,
    TurnConfig,
};
pub use partition::{
    PartitionConfig, PartitionDetector, PartitionError, PartitionState, PartitionStats,
};
pub use peer_manager::{
    BlacklistReason, CircuitBreaker, CircuitBreakerConfig, CircuitBreakerStats, CircuitState,
    ConcurrentPeerManager, PeerId, PeerManager, PeerManagerStats, PeerMetrics, PeerScoringConfig,
    PeerState, RetryConfig, RetryPolicy, SelectionStrategy,
};
pub use prefetch::{
    Prediction, PredictionReason, PrefetchConfig, PrefetchPredictor, PrefetchStats,
    PrefetchStrategy,
};
pub use quic::{
    AdaptiveBatchTuner, BlockStream, ParallelRequester, PipelineConfig, QuicConfig, QuicPoolStats,
    QuicTransport, SequentialPipeline,
};
pub use range_request::{ByteRange, RangeAssembler, RangeError, RangeRequest, RangeResponse};
pub use recovery::{
    RecoveryConfig, RecoveryError, RecoveryManager, RecoveryMode, RecoveryStats, RecoveryStrategy,
};
pub use schema_migration::{
    FieldDefault, FieldMigration, MigrationError, SchemaEvolutionManager, SchemaMigration,
};
pub use schema_registry::{
    ipc_bytes_to_schema, schema_to_ipc_bytes, EvolutionStrategy, SchemaError, SchemaEvolutionFrame,
    SchemaRegistry, SchemaVersion,
};
pub use session::{
    Session, SessionConfig, SessionError, SessionEvent, SessionId, SessionManager, SessionState,
    SessionStats,
};
pub use session_config::{
    SessionConfig as BlockExchangeSessionConfig, SessionMetrics, SessionMetricsSnapshot,
    SessionMetricsStore,
};
pub use tcp::{TcpConfig, TcpConnection, TcpTransport};
pub use tensorswap::{
    BackpressureConfig, BackpressureController, ChunkInfo, EinsumExpression, EinsumGraph,
    SafetensorEntry, SafetensorsHeader, StreamProgress, StreamRequest, StreamRequestQueue,
    TensorMetadata, TensorStream, TensorSwap, TensorSwapConfig, TensorSwapStats,
};
pub use throttle::{
    BandwidthConfig, BandwidthThrottle, QosPriority, ThrottleError, ThrottleStats, TokenBucket,
};
pub use transport::{
    Connection, ConnectionMetrics, Transport, TransportCapabilities, TransportError,
    TransportSelectionStrategy, TransportSelector, TransportStats, TransportType,
};
pub use want_list::{ConcurrentWantList, Priority, WantEntry, WantList, WantListConfig};
pub use websocket::{
    WebSocketConfig, WebSocketConnection, WebSocketServerConnection, WebSocketTransport,
};

// Diagnostics and monitoring
pub use auto_tuner::{AutoTuner, AutoTunerConfig, NetworkCondition, NetworkMetrics, TuningProfile};
pub use diagnostics::{
    DiagnosticConfig, DiagnosticEngine, DiagnosticIssue, DiagnosticReport, HealthStatus,
    IssueCategory, IssueSeverity, PeerManagerDiagnostics, SessionDiagnostics, WantListDiagnostics,
};
pub use health_monitor::{
    AlertCallback, ComponentHealth, ComponentStats, ComponentType, HealthAlert, HealthCheck,
    HealthCheckBuilder, HealthMonitor, HealthMonitorConfig,
};
pub use stats_aggregator::{
    AggregatedSessionStats, AggregatedStats, AggregatedStatsBuilder, AggregatedTransportStats,
    DataPoint, PerformanceMetrics, StatsCollector,
};

// Re-export utility functions for convenience
pub use utils::{
    adjust_priority_for_deadline, all_wants_present, any_want_present, bulk_add_wants,
    bulk_remove_wants, bulk_update_priorities, calculate_expected_throughput,
    calculate_optimal_chunk_size, calculate_optimal_concurrency, calculate_recommended_buffer_size,
    create_balanced_peer_scoring, create_bandwidth_optimized_peer_manager,
    create_bulk_transfer_session, create_datacenter_want_list, create_edge_device_peer_manager,
    create_edge_device_want_list, create_high_throughput_want_list, create_interactive_session,
    create_latency_optimized_peer_manager, create_low_latency_want_list, create_realtime_session,
    create_reliability_focused_scoring, create_scientific_session, debug_peer_scoring_config,
    debug_session_config, debug_want_list_config, estimate_required_peers, estimate_transfer_time,
    estimate_want_list_memory, format_bandwidth, format_bytes, format_duration,
    is_high_throughput_config, is_low_latency_config, validate_peer_scoring_config,
    validate_session_config, validate_want_list_config,
};

// Re-export facade for easy setup
pub use facade::{TransportFacade, TransportFacadeBuilder, TransportFacadeConfig, TransportPreset};

// Re-export test utilities for testing (feature-gated in the future if needed)
pub use test_utils::{
    add_test_peers, add_varied_test_peers, assert_approx_eq, assert_in_range,
    minimal_peer_scoring_config, minimal_session_config, minimal_want_list_config, test_cid,
    test_cids, test_peer_ids, test_peer_manager, test_peer_manager_with_config, test_session,
    test_session_with_blocks, test_want_list, test_want_list_with_cids,
};

// Re-export advanced features
pub use advanced_scheduling::{
    AdvancedScheduler, SchedulePriority, ScheduledRequest, SchedulerStats, SchedulingPolicy,
};
pub use connection_migration::{
    ConnectionMigration, MigrationConfig, MigrationEvent, MigrationState, MigrationStats,
};
pub use request_coalescing::{CoalescerConfig, CoalescerStats, RequestCoalescer};

// Re-export observability
pub use observability::{EventLogger, LogEntry, LogLevel, LoggerConfig, TransportEvent};
pub use prometheus_exporter::{MetricType, PrometheusExporter};

// Re-export load testing
pub use load_tester::{
    LoadPattern, LoadTestConfig, LoadTestConfigBuilder, LoadTestStats, LoadTester,
};

// Re-export configuration advisor
pub use config_advisor::{
    ConfigAdvisor, ConfigRequirements, NetworkQuality, PerformanceProfile, RecommendedConfig,
    ResourceLevel, UseCase,
};

// Arrow IPC stream deframer
pub use arrow_deframer::{
    build_test_eos, build_test_frame, ArrowFrame, ArrowFrameType, ArrowStreamDeframer,
    DeframerError, DeframerStats, DeframerStatsSnapshot,
};
