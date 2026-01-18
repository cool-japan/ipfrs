//! IPFRS Storage - Block storage and retrieval
//!
//! This crate provides the storage layer for IPFRS including:
//! - Sled, ParityDB, and in-memory block stores
//! - LRU and tiered caching
//! - Bloom filter for fast existence checks
//! - Streaming interface for large blocks
//! - Content-defined chunking and deduplication
//! - Transparent block compression (Zstd, Lz4, Snappy)
//! - Pin management for preventing GC
//! - Hot/cold tiering with access tracking
//! - Garbage collection (mark-and-sweep)
//! - CAR format export/import for backups
//! - Encryption at rest (ChaCha20-Poly1305, AES-256-GCM)
//! - Version Control System for differentiable storage (Git for Tensors)
//! - RAFT consensus protocol for distributed storage
//! - Network transport abstraction (TCP, QUIC with TLS) for multi-node RAFT clusters
//! - Cluster coordinator with automatic failover and re-election
//! - Multi-datacenter support with latency-aware routing and cross-datacenter replication
//! - Eventual consistency with version vectors and conflict resolution
//! - GraphQL query interface for flexible metadata querying
//! - ARM profiler with NEON SIMD optimization and low-power tuning
//! - Production-grade metrics and observability for monitoring (Prometheus, OpenTelemetry)
//! - Circuit breaker pattern for fault-tolerant external service calls
//! - Unified health check system for liveness and readiness monitoring
//! - TTL support for automatic block expiration
//! - Advanced retry logic with exponential backoff and jitter
//! - Optimized S3 multipart uploads for large blocks
//! - Rate limiting for controlling request rates to backends
//! - Write coalescing for batching similar writes
//! - Workload simulation for testing and benchmarking
//! - Automatic configuration tuning based on workload patterns
//! - Comprehensive profiling system with comparative analysis and regression detection
//! - Storage pool manager for multi-backend routing with intelligent load balancing
//! - Quota management system for per-tenant storage limits and bandwidth control
//! - Lifecycle policies for automatic data tiering, archival, and expiration
//! - Predictive prefetching with access pattern learning and adaptive depth control
//! - Cost analytics and optimization for cloud storage (AWS, Azure, GCP)

pub mod analyzer;
pub mod arm_profiler;
pub mod auto_tuner;
pub mod batch;
pub mod blockstore;
pub mod bloom;
pub mod cache;
pub mod car;
pub mod circuit_breaker;
pub mod cluster;
pub mod coalesce;
#[cfg(feature = "compression")]
pub mod compression;
pub mod cost_analytics;
pub mod datacenter;
pub mod dedup;
pub mod diagnostics;
#[cfg(feature = "encryption")]
pub mod encryption;
pub mod eventual_consistency;
pub mod exporters;
#[cfg(feature = "gateway")]
pub mod gateway;
pub mod gc;
pub mod gradient;
#[cfg(feature = "graphql")]
pub mod graphql;
pub mod health;
pub mod helpers;
// pub mod incremental_backup;  // Module file not found - commented out
pub mod lifecycle;
pub mod memory;
pub mod metrics;
pub mod migration;
#[cfg(feature = "mmap")]
pub mod mmap;
pub mod otel;
pub mod paritydb;
pub mod pinning;
pub mod pool;
pub mod prefetch;
pub mod profiler;
pub mod profiling;
pub mod prometheus;
pub mod query_optimizer;
pub mod quota;
pub mod raft;
pub mod rate_limit;
pub mod replication;
pub mod retry;
#[cfg(feature = "s3")]
pub mod s3;
pub mod safetensors;
pub mod streaming;
pub mod tiering;
pub mod traits;
pub mod transport;
pub mod ttl;
pub mod utils;
pub mod vcs;
pub mod workload;

pub use analyzer::{
    Category, Difficulty, OperationStats, OptimizationRecommendation, Priority, SizeDistribution,
    StorageAnalysis, StorageAnalyzer, WorkloadCharacterization, WorkloadType,
};
pub use arm_profiler::{
    hash_block, ArmFeatures, ArmPerfCounter, ArmPerfReport, LowPowerBatcher, PowerProfile,
    PowerStats,
};
pub use auto_tuner::{
    AutoTuner, AutoTunerConfig, TuningPresets, TuningRecommendation, TuningReport,
};
pub use batch::{batch_delete, batch_get, batch_has, batch_put, BatchConfig, BatchResult};
pub use blockstore::{BlockStoreConfig, SledBlockStore};
pub use bloom::{BloomBlockStore, BloomConfig, BloomFilter, BloomStats};
pub use cache::{
    BlockCache, CacheStats, CachedBlockStore, TieredBlockCache, TieredCacheStats,
    TieredCachedBlockStore,
};
pub use car::{
    export_to_car, import_from_car, CarHeader, CarReadStats, CarReader, CarWriteStats, CarWriter,
};
pub use circuit_breaker::{CircuitBreaker, CircuitState, CircuitStats};
pub use cluster::{ClusterConfig, ClusterCoordinator, ClusterStats, NodeHealth, NodeInfo};
pub use coalesce::{CoalesceConfig, CoalesceStats, CoalescingBlockStore};
#[cfg(feature = "compression")]
pub use compression::{
    BlockCompressionStats, CompressionAlgorithm, CompressionBlockStore, CompressionConfig,
};
pub use cost_analytics::{
    CloudProvider, CostAnalyzer, CostBreakdown, CostProjection, CostTier, TierCostModel,
    TierOption, TierRecommendation,
};
pub use datacenter::{
    CrossDcStats, Datacenter, DatacenterId, LatencyAwareSelector, MultiDatacenterCoordinator,
    Region, ReplicationPolicy,
};
pub use dedup::{ChunkingConfig, DedupBlockStore, DedupStats};
pub use diagnostics::{
    BenchmarkComparison, DiagnosticsReport, HealthMetrics, PerformanceMetrics, StorageDiagnostics,
};
#[cfg(feature = "encryption")]
pub use encryption::{Cipher, EncryptedBlockStore, EncryptionConfig, EncryptionKey};
pub use eventual_consistency::{
    ConflictResolution, ConsistencyLevel, EventualStore, EventualStoreStats, VersionVector,
    VersionedValue,
};
pub use exporters::{BatchExporter, ExportFormat, MetricExporter};
#[cfg(feature = "gateway")]
pub use gateway::{GatewayBlockStore, GatewayConfig, HybridBlockStore};
pub use gc::{
    GarbageCollector, GcConfig, GcPolicy, GcResult, GcScheduler, GcStats, GcStatsSnapshot,
};
pub use gradient::{
    CompressionStats, DeltaEncoder, GradientData, GradientStore, ProvenanceMetadata,
};
#[cfg(feature = "graphql")]
pub use graphql::{
    create_schema, BlockConnection, BlockFilter, BlockMetadata, BlockQuerySchema, BlockStats,
    QueryRoot, SortField, SortOrder,
};
pub use health::{
    AggregateHealthResult, DetailedHealthStatus, HealthCheck, HealthCheckResult, HealthChecker,
    HealthStatus, SimpleHealthCheck,
};
#[cfg(feature = "encryption")]
pub use helpers::encrypted_production_stack;
pub use helpers::{
    blockchain_stack, cache_stack, cdn_edge_stack, coalescing_memory_stack,
    deduplicated_production_stack, development_stack, embedded_stack, ingestion_stack, iot_stack,
    media_streaming_stack, memory_stack, ml_model_stack, monitored_production_stack,
    production_stack, read_optimized_stack, resilient_stack, testing_stack, ttl_production_stack,
    write_optimized_stack, StorageStackBuilder,
};
#[cfg(feature = "compression")]
pub use helpers::{compressed_production_stack, ultimate_production_stack};
#[cfg(feature = "compression")]
pub use helpers::{distributed_fs_stack, scientific_archive_stack};
// pub use incremental_backup::{BackupStats, BackupType, IncrementalBackup, Snapshot};  // Module not found
pub use lifecycle::{
    BlockMetadata as LifecycleBlockMetadata, LifecycleAction, LifecycleActionResult,
    LifecycleCondition, LifecyclePolicyConfig, LifecyclePolicyManager, LifecycleRule,
    LifecycleStatsSnapshot, StorageTier,
};
pub use memory::MemoryBlockStore;
pub use metrics::{MetricsBlockStore, StorageMetrics};
pub use migration::{
    estimate_migration, migrate_storage, migrate_storage_batched, migrate_storage_verified,
    migrate_storage_with_progress, validate_migration, MigrationConfig, MigrationEstimate,
    MigrationStats, StorageMigrator,
};
#[cfg(feature = "mmap")]
pub use mmap::{MmapBlockStore, MmapConfig};
pub use otel::OtelBlockStore;
pub use paritydb::{ParityDbBlockStore, ParityDbConfig, ParityDbPreset};
pub use pinning::{PinInfo, PinManager, PinSet, PinStatsSnapshot, PinType};
pub use pool::{BackendConfig, BackendId, PoolConfig, PoolStats, RoutingStrategy, StoragePool};
pub use prefetch::{
    AccessPattern, PredictivePrefetcher, PrefetchConfig, PrefetchPrediction, PrefetchStatsSnapshot,
};
pub use profiler::{
    ComparativeProfiler, ComparisonReport, ProfileConfig, ProfileReport, RegressionDetector,
    RegressionResult, StorageProfiler,
};
pub use profiling::{BatchProfiler, LatencyHistogram, PerformanceProfiler, ThroughputTracker};
pub use prometheus::{PrometheusExporter, PrometheusExporterBuilder};
pub use query_optimizer::{
    OptimizerConfig, QueryLogEntry, QueryOptimizer, QueryPlan, QueryStrategy, Recommendation,
    RecommendationCategory, RecommendationPriority,
};
pub use quota::{
    QuotaBlockStore, QuotaConfig, QuotaManager, QuotaManagerConfig, QuotaReport, QuotaStatus,
    QuotaUsageSnapshot, TenantId, ViolationType,
};
pub use raft::{
    AppendEntriesRequest, AppendEntriesResponse, Command, LogEntry, LogIndex, NodeId, NodeState,
    RaftConfig, RaftNode, RaftStats, RequestVoteRequest, RequestVoteResponse, Term,
};
pub use rate_limit::{RateLimitAlgorithm, RateLimitConfig, RateLimitStats, RateLimiter};
pub use replication::{
    ConflictStrategy, ReplicationManager, ReplicationState, Replicator, SyncResult, SyncStrategy,
};
pub use retry::{BackoffStrategy, JitterType, RetryPolicy, RetryStats, Retryable};
#[cfg(feature = "s3")]
pub use s3::{S3BlockStore, S3Config};
pub use safetensors::{
    ChunkConfig, ChunkedTensor, DType, ModelStats, SafetensorsHeader, SafetensorsManifest,
    SafetensorsStore, TensorInfo,
};
pub use streaming::{
    BlockReader, ByteRange, PartialBlock, StreamConfig, StreamingBlockStore, StreamingWriter,
};
pub use tiering::{AccessStats, AccessTracker, Tier, TierConfig, TierStatsSnapshot, TieredStore};
pub use traits::BlockStore as BlockStoreTrait;
#[cfg(feature = "quic")]
pub use transport::QuicTransport;
pub use transport::{
    InMemoryTransport, Message as TransportMessage, TcpTransport, Transport, TransportConfig,
};
pub use ttl::{TtlBlockStore, TtlCleanupResult, TtlConfig, TtlStats};
pub use utils::{
    compute_cid, compute_total_size, create_block, create_blocks_batch, deduplicate_blocks,
    estimate_compression_ratio, extract_cids, filter_blocks_by_size, find_duplicates,
    generate_compressible_blocks, generate_compressible_data, generate_dedup_dataset,
    generate_incompressible_data, generate_mixed_size_blocks, generate_pattern_blocks,
    generate_random_block, generate_random_blocks, group_blocks_by_size, sample_blocks,
    sort_blocks_by_size_asc, sort_blocks_by_size_desc, validate_block_integrity,
    validate_blocks_batch, BlockStatistics,
};
pub use vcs::{
    Author, Commit, CommitBuilder, MergeResult, MergeStrategy, Ref, RefType, VersionControl,
};
pub use workload::{
    OperationMix, SizeDistribution as WorkloadSizeDistribution, WorkloadConfig, WorkloadPattern,
    WorkloadPresets, WorkloadResult, WorkloadSimulator,
};
