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
//! - Safe, transactional schema migration framework with rollback support

pub mod block_garbage_collector;
pub use block_garbage_collector::{
    BgcBlockRecord, BgcCollectorConfig, BgcCollectorStats, BgcGcPolicy, BgcGcResult,
    BgcSweepResult, BlockGarbageCollector,
};

pub mod content_dedup_index;
pub use content_dedup_index::{
    ContentDedupConfig, ContentDedupResult, ContentDedupStats, ContentDeduplicationIndex,
    ContentHash, DedupIndexError,
};

pub mod block_validator;
pub use block_validator::{
    StorageBlockValidator, ValidationReport, ValidationResult, ValidationRule, ValidatorStats,
};

pub mod block_verifier;
pub use block_verifier::{
    BlockRecord, StorageBlockVerifier, VerificationReport, VerificationResult, VerifierStats,
};

pub mod block_index;
pub mod block_manifest;
pub mod block_packer;
pub mod index_builder;
pub mod index_recovery;
pub use index_recovery::{
    IndexEntry as IrIndexEntry, IndexRecovery, RawBlock, RecoveryConfig, RecoveryStats,
    RecoveryStatus,
};
pub mod metrics_collector;
pub mod storage_metrics_collector;
pub use storage_metrics_collector::{
    AggregatedStats, CollectorConfig, CollectorStats, MetricKind, MetricSample, MetricSeries,
    StorageMetricsCollector as SmcStorageMetricsCollector, TimeBucket,
};
pub mod eviction_policy;
pub use eviction_policy::{
    CacheEntry as EvictionCacheEntry, EvictionCandidate, EvictionStrategy,
    PolicyStats as EvictionPolicyStats, StorageEvictionPolicy,
};
pub mod access_logger;
pub mod access_predictor;
pub mod analyzer;
pub mod audit_trail;
pub mod eviction_simulator;
pub mod fragmentation_analyzer;
pub mod heatmap_tracker;
pub use audit_trail::{
    AuditEntry, AuditEventType, AuditFilter, AuditTrail, AuditTrailConfig, AuditTrailStats,
};
pub mod block_migration_planner;
pub use block_migration_planner::{
    BlockMigrationPlanner, BmpBlockMeta, BmpMigrationPlan, BmpPlanStatus, BmpPlannerConfig,
    BmpPlannerStats, BmpPriorityPolicy,
};
pub mod arm_profiler;
pub mod auto_tuner;
pub mod batch;
pub mod block_cache;
pub mod block_stream;
pub mod blockstore;
pub mod bloom;
pub mod compaction;
pub mod compaction_advisor;
pub mod compression_advisor;
pub mod compression_pipeline;
pub use compression_pipeline::{
    CompressionAlgo as CpCompressionAlgo, CompressionError as CpCompressionError,
    CompressionHint as CpCompressionHint, CompressionResult as CpCompressionResult,
    PipelineConfig as CpPipelineConfig, PipelineStage as CpPipelineStage,
    PipelineStats as CpPipelineStats, StorageCompressionPipeline,
};
pub mod compression_registry;
pub use compression_registry::{
    CodecProfile, CodecRecommendation, CompressionCodec as RegistryCompressionCodec,
    CompressionRegistryStats, DataCharacteristics, StorageCompressionRegistry,
};
pub mod compactor;
pub use compactor::{
    CompactionResult as RegionCompactionResult, CompactionState as RegionCompactionState,
    CompactorConfig as RegionCompactorConfig, CompactorStats as RegionCompactorStats,
    FragmentationReport as RegionFragmentationReport, StorageCompactor,
};
pub mod cache;
pub mod car;
pub mod chunk_manager;
pub mod circuit_breaker;
pub mod cold_storage;
pub use cold_storage::{
    fnv1a_cold, ColdStorageManager, ColdStorageStats, StorageTier as CsStorageTier,
    TierPolicy as CsTierPolicy, TieredBlock as CsTieredBlock,
};
pub mod cluster;
pub mod coalesce;
#[cfg(feature = "compression")]
pub mod compression;
pub mod corruption_repair;
pub use corruption_repair::{
    BlockWithParity, CorruptionRepairer, CorruptionReport, CorruptionType, RepairAction,
    RepairConfig, RepairStats,
};
pub mod cost_analytics;
pub mod cost_estimator;
pub mod data_integrity_auditor;
pub use data_integrity_auditor::{
    compute_adler32 as dia_compute_adler32, compute_checksum as dia_compute_checksum,
    compute_crc32 as dia_compute_crc32, compute_fnv_xor64 as dia_compute_fnv_xor64, AuditConfig,
    AuditResult, AuditorStats, BlockChecksum, ChecksumAlgo, DataIntegrityAuditor, RepairRecord,
};
pub mod data_integrity_checker;
pub use data_integrity_checker::{
    BlockRecord as DicBlockRecord, DataIntegrityChecker, IntegrityCheckerConfig, IntegrityReport,
    IntegrityStats, IntegrityStatus,
};
pub mod datacenter;
pub mod dedup;
pub mod dedup_tracker;
pub mod deduplication_pipeline;
pub use deduplication_pipeline::{
    BlockEntry, DedupPipelineStats, DedupResult as DpDedupResult, DedupStage,
    DeduplicationPipeline, PipelineConfig,
};
pub mod diagnostics;
#[cfg(feature = "encryption")]
pub mod encryption;
pub mod eventual_consistency;
pub mod exporters;
pub mod garbage_collector;
#[cfg(feature = "gateway")]
pub mod gateway;
pub use garbage_collector::{
    GcError, GcObject, GcObjectId, GcPhase, StorageGarbageCollector, StorageGcConfig, StorageGcRun,
    StorageGcStats,
};

pub mod gc;
pub mod gc_planner;
pub mod gradient;
#[cfg(feature = "graphql")]
pub mod graphql;
pub mod health;
#[cfg(feature = "sled-backend")]
pub mod helpers;
pub mod integrity_checker;
pub mod integrity_scanner;
// pub mod incremental_backup;  // Module file not found - commented out
pub mod lifecycle;
pub mod memory;
pub mod metrics;
pub mod migration;
#[cfg(feature = "mmap")]
pub mod mmap;
pub mod otel;
#[cfg(feature = "parity-db-backend")]
pub mod paritydb;
pub mod pinning;
pub mod pool;
pub mod prefetch;
pub mod profiler;
pub mod profiling;
pub mod prometheus;
pub mod query_optimizer;
pub mod quota;
pub mod quota_enforcer;
pub mod read_ahead;
pub use quota_enforcer::{
    NamespaceQuota as EnforcerNamespaceQuota, QuotaEnforcerConfig, QuotaEnforcerStats, QuotaLevel,
    StorageQuotaEnforcer,
};
pub mod quota_manager;
pub mod raft;
pub mod rate_limit;
pub mod replication;
pub mod replication_tracker;
pub mod retention_engine;
pub mod retention_policy;
pub mod retry;
#[cfg(feature = "s3")]
pub mod s3;
pub mod safetensors;
pub mod snapshot_diff;
pub use snapshot_diff::{
    DiffEntry, DiffKind, DiffStats as SnapshotDiffStats, SnapshotDiffResult,
    SnapshotEntry as SnapshotDiffEntry, StorageSnapshotDiff,
};
pub mod snapshot_differ;
pub mod snapshot_manager;
pub use snapshot_manager::{
    fnv1a_64 as snapshot_fnv1a_64,
    LegacySnapshot,
    LegacySnapshotDiff as ManagerSnapshotDiff,
    LegacySnapshotEntry,
    // Legacy block-snapshot API
    LegacyStorageSnapshotManager,
    SnapshotConfig,
    SnapshotDelta,
    SnapshotEntry as SsmSnapshotEntry,
    SnapshotError,
    SnapshotId,
    SnapshotKind,
    SnapshotManagerStats,
    SnapshotState,
    SnapshotStats as SsmSnapshotStats,
    SsmSnapshot,
    StorageSnapshot,
    // New production-grade API
    StorageSnapshotManager,
    StorageState as SsmStorageState,
};

pub mod streaming;
pub mod tier_migration_engine;
pub use tier_migration_engine::{
    BlockMeta as TmBlockMeta, MigrationAction as TmMigrationAction,
    MigrationReason as TmMigrationReason, MigrationResult as TmMigrationResult, MigratorConfig,
    MigratorStats, StorageTier as TmStorageTier, StorageTierMigrator, TierPolicy as TmTierPolicy,
};

pub mod content_addressable_cache;
pub mod tier_balancer;
pub mod tier_manager;
pub mod tier_migrator;
pub mod tiering;
pub mod traits;
pub mod transport;
pub mod ttl;
pub mod utils;
pub mod vcs;
pub mod wal;
pub mod workload;
pub mod write_buffer;
pub mod write_journal;
pub use content_addressable_cache::{
    CacheConfig as CacCacheConfig, CacheEntry as CacCacheEntry, CacheError,
    CacheStats as CacCacheStats, ContentAddressableCache, EvictionPolicy as CacEvictionPolicy,
    LruNode,
};

pub mod content_addressed_cache_v2;
pub use content_addressed_cache_v2::{
    Cac2CacheConfig, Cac2CacheStats, Cac2Cid, Cac2Entry, Cac2EvictionRecord, Cac2Tier,
    ContentAddressedCacheV2,
};

pub mod hotspot_detector;

pub use access_predictor::{
    AccessEvent, AccessPattern as PredictorAccessPattern, PredictionResult, PredictorStats,
    StorageAccessPredictor,
};
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
pub use block_index::{
    BlockIndexEntry, BlockIndexStats, IndexEntry, IndexKey, IndexQuery, IndexStats,
    SecondaryBlockIndex, StorageBlockIndex,
};
pub use block_manifest::{
    fnv1a, ManifestEntry, ManifestFilter, ManifestStats, StorageBlockManifest,
};
pub use block_packer::{
    fnv1a as packer_fnv1a, Pack, PackEntry, PackerConfig, PackerStats, StorageBlockPacker,
};
pub use block_stream::{
    BlockChunk, BlockStreamIterator, BlockStreamState, StreamConfig as BlockStreamConfig,
    StreamStats as BlockStreamStats,
};
#[cfg(feature = "sled-backend")]
pub use blockstore::SledBlockStore;
pub use blockstore::{BlockStoreConfig, DeduplicationStats, DeduplicationStatsSnapshot};
pub use bloom::{
    BloomBlockStore, BloomConfig, BloomFilter, BloomFilterConfig, BloomSnapshot, BloomStats,
    CidBloomFilter,
};
pub use cache::{
    BlockCache, CacheConfig, CacheStats, CacheStatsSnapshot, CachedBlockStore, LegacyCacheStats,
    TieredBlockCache, TieredCacheStats, TieredCachedBlockStore,
};
pub use car::{
    export_to_car, import_from_car, CarHeader, CarReadStats, CarReader, CarWriteStats, CarWriter,
};
pub use circuit_breaker::{CircuitBreaker, CircuitState, CircuitStats};
pub use cluster::{ClusterConfig, ClusterCoordinator, ClusterStats, NodeHealth, NodeInfo};
pub use coalesce::{CoalesceConfig, CoalesceStats, CoalescingBlockStore};
pub use compaction::{CompactionConfig, CompactionScheduler};
#[cfg(feature = "compression")]
pub use compression::{
    compress_block, compress_block_with_algorithm, compress_block_with_level,
    compress_block_with_stats, decompress_block, BlockCompressStats, BlockCompressionStats,
    CompressionAlgorithm, CompressionBlockStore, CompressionConfig, MAGIC_LZ4, MAGIC_RAW,
    MAGIC_SNAPPY, MAGIC_ZSTD, MIN_COMPRESS_SIZE,
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
#[cfg(feature = "sled-backend")]
pub use gc::SledSnapshotPinRegistry;
pub use gc::{
    snapshot_pin_id, GarbageCollector, GcConfig, GcPolicy, GcResult, GcScheduler, GcStats,
    GcStatsSnapshot, OrphanGarbageCollector, OrphanGcConfig, OrphanGcResult, SnapshotPinRegistry,
};
pub use gc_planner::{GCCandidate, GCConfig, GCPlan, GCPlannerStats, StorageGCPlanner};
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
pub mod health_monitor;
pub use health_monitor::{
    HealthMonitorConfig, HealthMonitorStats, MonitorHealthCheck, MonitorHealthStatus,
    StorageHealthMonitor,
};

pub mod storage_health_monitor;
#[cfg(all(feature = "sled-backend", feature = "encryption"))]
pub use helpers::encrypted_production_stack;
#[cfg(feature = "sled-backend")]
pub use helpers::{
    blockchain_stack, cache_stack, cdn_edge_stack, coalescing_memory_stack,
    deduplicated_production_stack, development_stack, embedded_stack, ingestion_stack, iot_stack,
    media_streaming_stack, memory_stack, ml_model_stack, monitored_production_stack,
    production_stack, read_optimized_stack, resilient_stack, testing_stack, ttl_production_stack,
    write_optimized_stack, StorageStackBuilder,
};
#[cfg(all(feature = "sled-backend", feature = "compression"))]
pub use helpers::{compressed_production_stack, ultimate_production_stack};
#[cfg(all(feature = "sled-backend", feature = "compression"))]
pub use helpers::{distributed_fs_stack, scientific_archive_stack};
pub use storage_health_monitor::{
    ShmAlert, ShmCategory, ShmHealthSnapshot, ShmMonitorConfig, ShmMonitorStats, ShmProbe,
    ShmProbeId, ShmSeverity, ShmStatus, StorageHealthMonitor as ShmStorageHealthMonitor,
};
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
    migrate_storage_with_progress, validate_migration, BlockMigrationStats, MigrationConfig,
    MigrationEstimate, StorageMigrator,
};
pub use migration::{
    MigrationError, MigrationPlan, MigrationRecord, MigrationRunner, MigrationStats,
    MigrationStatsSnapshot, MigrationStatus, MigrationStep, SchemaVersion,
};
#[cfg(feature = "mmap")]
pub use mmap::{MmapBlockStore, MmapConfig};
pub use otel::OtelBlockStore;
#[cfg(feature = "parity-db-backend")]
pub use paritydb::{ParityDbBlockStore, ParityDbConfig, ParityDbPreset};
pub use pinning::{PinInfo, PinManager, PinSet, PinStatsSnapshot, PinType};
pub use pool::{BackendConfig, BackendId, PoolConfig, PoolStats, RoutingStrategy, StoragePool};
pub use prefetch::{
    AccessPattern as PrefetchAccessPattern, PredictivePrefetcher, PrefetchConfig,
    PrefetchPrediction, PrefetchStatsSnapshot,
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
pub use read_ahead::{PrefetchHint, ReadAheadPattern, ReadAheadScheduler, ReadAheadStats};
pub mod quota_registry;
pub use chunk_manager::{
    fnv1a_u64 as chunk_fnv1a_u64, Chunk, ChunkManagerStats, ChunkState, ChunkedObject,
    StorageChunkManager,
};
pub use fragmentation_analyzer::{
    CompactionCandidate, FragmentationReport, StorageExtent, StorageFragmentationAnalyzer,
};
pub use heatmap_tracker::{HeatBucket, HeatEntry, HeatmapStats, StorageHeatmapTracker};
pub use quota_registry::{
    QuotaEntry, QuotaKind, QuotaViolation, RegistryStats, StorageQuotaRegistry,
    ViolationType as RegistryViolationType,
};
pub use raft::{
    AppendEntriesRequest, AppendEntriesResponse, Command, LogEntry, LogIndex, NodeId, NodeState,
    RaftConfig, RaftNode, RaftStats, RequestVoteRequest, RequestVoteResponse, Term,
};
pub use rate_limit::{RateLimitAlgorithm, RateLimitConfig, RateLimitStats, RateLimiter};
pub use replication::{
    ConflictStrategy, ReplicationManager, ReplicationState, Replicator, SyncResult, SyncStrategy,
};
pub use replication_tracker::{
    BlockReplicationEntry, ReplicaLocation, ReplicationStats as BlockReplicationStats,
    ReplicationStatus, ReplicationTask, StorageReplicationTracker,
};
pub use retention_policy::{
    BlockRecord as RetentionBlockRecord, PolicyDecision, RetentionAction, RetentionEntry,
    RetentionPolicyStats, RetentionRule, RetentionStats, StorageRetentionPolicy,
    StorageRetentionPolicyEngine, TickRetentionAction, TickRetentionRule,
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
pub use tier_balancer::{BalancerStats, MoveTask, StorageTierBalancer, TierKind, TierStatus};
pub use tier_manager::{
    BlockTierRecord, StorageTier as TierManagerStorageTier, StorageTierManager, TierPolicy,
    TierStats, TierStatsSnapshot as TierManagerStatsSnapshot, TierTransition,
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
pub use wal::{
    fnv1a_32, StorageWalEntry, StorageWalStats, StorageWriteAheadLog, WalConfig, WalEntry,
    WalEntryKind, WalError, WalOp, WalStats, WalStatsSnapshot, WriteAheadLog,
};
pub use workload::{
    OperationMix, SizeDistribution as WorkloadSizeDistribution, WorkloadConfig, WorkloadPattern,
    WorkloadPresets, WorkloadResult, WorkloadSimulator,
};
pub use write_buffer::{
    BufferConfig, BufferedEntry, FlushResult, StorageWriteAheadBuffer, WriteOp,
};
pub use write_journal::{
    fnv1a as journal_fnv1a, JournalCursor, JournalEntry, JournalEntryKind, JournalStats,
    StorageWriteJournal,
};
pub mod object_store;
pub use hotspot_detector::{
    AccessEvent as HotspotAccessEvent, AccessType, BlockAccessRecord, DetectorConfig,
    HotspotDetectorStats, HotspotScore, StorageHotspotDetector,
};
pub use object_store::{ObjectStoreStats, ObjectVersion, StorageObjectStore, StoredObject};

pub mod blockstore_sharding;
pub use blockstore_sharding::{
    BlockRecord as BssBlockRecord, BlockStoreGlobalStats, BlockStoreSharding, ShardKey,
    ShardMetrics, ShardingConfig, ShardingError,
};

pub mod migration_planner;
pub use migration_planner::{
    MigrationDirection, MigrationRecord as PlannerMigrationRecord,
    MigrationStatus as PlannerMigrationStatus, MigrationTask, PlannerStats,
    StorageMigrationPlanner, StorageTier as MigrationStorageTier,
};

pub mod transaction_log;
pub use transaction_log::{
    StorageTransactionLog, Transaction, TransactionId, TransactionStatus, TxError, TxOperation,
    TxStats,
};

pub mod block_compactor;
pub use block_compactor::{
    BlockFragment, CompactionPlan, CompactionSegment, CompactorConfig, CompactorStats,
    StorageBlockCompactor,
};

pub mod access_log;
pub use access_log::{
    AccessLogEntry, AccessLogStats, AccessPattern as LogAccessPattern, LogConfig, LogOperation,
    StorageAccessLog,
};

pub mod access_tracker;
pub use access_tracker::{
    AccessRecord, AccessTracker as AtAccessTracker, TrackerConfig, TrackerStats,
};

pub mod metadata_index;
pub use metadata_index::{
    MetadataField, MetadataIndexEntry, MetadataIndexStats, MetadataQuery, MetadataQueryResult,
    MetadataSortField, StorageMetadataIndex,
};

pub mod replication_manager;
pub use replication_manager::{
    BlockReplicas, ReplicaInfo, ReplicaNode, ReplicationConfig as ReplicationManagerConfig,
    ReplicationError, ReplicationManagerStats, ReplicationState as ReplicaReplicationState,
    RmReplicaLocation, RmReplicationPolicy, RmReplicationStats, RmReplicationStatus,
    StorageReplicationManager,
};

pub mod io_scheduler;
pub use io_scheduler::{
    IODirection, IOPriority, IORequest, IOSchedulerStats, SchedulerConfig, StorageIOScheduler,
};

pub mod encryption_layer;
pub use encryption_layer::{
    CipherMode, EncryptedBlock, EncryptionLayerConfig, EncryptionLayerStats, StorageEncryptionLayer,
};

pub mod storage_encryption_layer;
pub use storage_encryption_layer::{
    SelCipher, SelEncryptedBlockRecord, SelEncryptionConfig, SelEncryptionStats,
    StorageEncryptionLayer as SelStorageEncryptionLayer,
};

pub mod event_log;
pub use event_log::{EventLogStats, EventSeverity, EventType, StorageEvent, StorageEventLog};

pub mod storage_benchmark;
pub use storage_benchmark::{
    BenchmarkConfig, BenchmarkOp, BenchmarkResult, BenchmarkStats, LatencySample, StorageBenchmark,
};

pub mod block_cache_manager;
pub use block_cache_manager::{
    BcmCacheConfig, BcmCacheStats, BcmCachedBlock, BlockCacheManager, CacheTier, EvictionPolicy,
};

pub mod storage_quota_manager;
pub use storage_quota_manager::{
    ObjectRecord, QuotaError, QuotaNamespace, QuotaPolicy, QuotaStats, SqmEvictionStrategy,
    SqmQuotaEntry, SqmQuotaViolation, StorageQuotaManager,
};

pub mod content_addressed_archive;
pub use content_addressed_archive::{
    compute_cid as caa_compute_cid, ArchiveBlock, ArchiveConfig, ArchiveEntry, ArchiveError,
    ArchiveIndex, ArchiveStats, ContentAddressedArchive,
};

pub mod storage_event_log;
pub use storage_event_log::{
    event_checksum,
    event_cid,
    sel_fnv1a_64,
    EventAggregation,
    EventFilter,
    // New production-grade API
    EventId,
    EventLogConfig,
    EventLogError,
    EventQuery,
    RetentionPolicy,
    SelEventLogStats,
    SelEventType,
    SelStorageEvent,
    SelStorageEventLog,
    // Legacy shim API (previous StorageEventKind-based interface)
    StorageEventKind,
};

pub mod object_lifecycle;
pub use object_lifecycle::{
    LifecycleError, ManagedObject, ObjectLifecycleManager, OlmLifecycleAction, OlmLifecycleState,
    OlmLifecycleStats, OlmRetentionRule,
};

pub mod prefetch_engine;
pub use prefetch_engine::{
    CoAccessPair, PeAccessEvent, PeAccessPattern, PeAccessType, PeConfig, PePrefetchHint,
    PePrefetchStats, StoragePrefetchEngine,
};

pub mod block_fragment_store;
pub use block_fragment_store::{
    bfs_fnv1a_32, AssembledBlock, BfsFragment, BlockFragmentStore, FragmentError, FragmentId,
    FragmentSet, FragmentSetState, FragmentStats,
};

pub mod mirror_sync;
pub use mirror_sync::{
    fnv1a_64 as mirror_sync_fnv1a_64, ConflictType, MirrorId, MirrorSyncStats,
    MsConflictResolution, MsSyncResult, StorageMirrorSync, SyncConflict, SyncItem, SyncOperation,
    SyncPlan,
};

pub mod checksum_engine;
pub use checksum_engine::{
    adler32 as ce_adler32, blake3_256_simple as ce_blake3_256_simple, crc32_iso as ce_crc32_iso,
    djb2 as ce_djb2, fnv1a_64 as ce_fnv1a_64, murmur3_32 as ce_murmur3_32,
    xxhash64_simple as ce_xxhash64_simple, CeVerificationResult, Checksum, ChecksumAlgorithm,
    ChecksumRecord as CeChecksumRecord, ChecksumStats, StorageChecksumEngine,
};

pub mod wal_replay;
pub use wal_replay::{
    ReplayPolicy as WrReplayPolicy, ReplayState as WrReplayState, ReplayStats as WrReplayStats,
    StorageWALReplay, WalEntry as WrWalEntry, WalEntryType as WrWalEntryType,
    WalStats as WrWalStats,
};

pub mod block_access_optimizer;
pub use block_access_optimizer::{
    AccessEvent as BaoAccessEvent, AccessPattern as BaoAccessPattern, BlockAccessOptimizer,
    CoAccessPair as BaoCoAccessPair, OptimizerConfig as BaoOptimizerConfig, OptimizerStats,
    PrefetchRecommendation,
};

pub mod block_index_rebuild;
pub use block_index_rebuild::{
    BlockIndexRebuild, BlockScanEntry, IndexEntry as BirIndexEntry, RebuildConfig, RebuildPhase,
    RebuildProgress, RebuildStats,
};

pub mod object_version_store;
pub use object_version_store::{
    ObjectVersionStore, OvsGcPolicy, OvsObjectVersion, VersionBranch, VersionQuery,
    VersionStoreConfig, VsError, VsStats,
};

pub mod merkle_proof_verifier;
pub use merkle_proof_verifier::{
    MerkleHash, MerkleLeaf, MerkleProof, MerkleProofVerifier, ProofStep,
    TreeStats as MerkleTreeStats, UpdateProof as MerkleUpdateProof,
    VerifierError as MerkleVerifierError,
};

pub mod storage_shard_balancer;
pub use storage_shard_balancer::{
    fnv1a_64 as ssb_fnv1a_64, xorshift64 as ssb_xorshift64, BalancerConfig, BalancerError,
    RebalanceOp, RebalancePolicy, ShardAssignment, ShardNode, SsbBalancerStats,
    StorageShardBalancer,
};

pub mod storage_quota_enforcer;
pub use storage_quota_enforcer::{
    EnforcementPolicy, EnforcerConfig, GrowthForecast, NamespaceId, QuotaLimit, QuotaUsage,
    QuotaViolation as SqeQuotaViolation, SqeStorageQuotaEnforcer, UsageSample, ViolationKind,
};

pub mod block_deduplicator;
pub use block_deduplicator::{
    fnv1a_64 as bdd_fnv1a_64, BlockDeduplicator, Chunk as BddChunk, ChunkHash, ChunkRef,
    ChunkingConfig as BddChunkingConfig, DeduplicationStats as BddDeduplicationStats,
    DeduplicatorError, ObjectManifest,
};

pub mod object_storage_tiering;
pub use object_storage_tiering::{
    ObjectStorageTiering, OstStorageTier, OstTierConfig, OstTierPolicy, OstTierTransition,
    TieredObject, TieringError, TieringStats,
};

pub mod write_ahead_log;
pub use write_ahead_log::{
    fnv1a_64 as wal2_fnv1a_64, RecoveryResult, Transaction as WalTransaction, TxState,
    WalConfig as WalWalConfig, WalEntry as WalWalEntry, WalError as WalWalError, WalOpType,
    WalStats as WalWalStats, WalWriteAheadLog, WAL_MAGIC,
};

pub mod storage_compression_pipeline;
pub use storage_compression_pipeline::{
    delta_decode, delta_encode, fnv1a_64 as scp_fnv1a_64, lz77_decode, lz77_encode, rle_decode,
    rle_encode, xor_transform, CompressedBlock, CompressionStage, ScpCompressionAlgorithm,
    ScpPipelineConfig, ScpPipelineError, ScpPipelineStats, ScpStorageCompressionPipeline,
};

pub mod storage_access_controller;
pub use storage_access_controller::{
    AccessDecision, AclConfig, AclError, AclStats, Permission, PolicyEffect, ResourcePolicy,
    SacAuditEntry, SacRole, StorageAccessController, SubjectAttributes,
};

pub mod object_integrity_checker;
pub use object_integrity_checker::{
    CheckerConfig, CheckerError, CheckerStats, IntegrityHash, IntegrityLevel,
    ObjectIntegrityChecker, OicIntegrityStatus, OicObjectRecord, OicVerificationResult,
};

pub mod storage_replication_manager;
pub use storage_replication_manager::{
    ReplicaTarget as SrmReplicaTargetDirect, ReplicationPolicy as SrmReplicationPolicyDirect,
    SrmError as SrmReplicationError, SrmReplicaTarget, SrmReplicationConfig, SrmReplicationOp,
    SrmReplicationPolicy, SrmReplicationStats,
    StorageReplicationManager as SrmStorageReplicationManager,
};

pub mod storage_query_planner;
pub use storage_query_planner::{
    SqpCostModel, SqpPlanStep, SqpPlannerConfig, SqpPlannerStats, SqpQuery, SqpQueryPlan,
    StorageQueryPlanner,
};

pub mod storage_snapshot_manager;
pub use storage_snapshot_manager::{
    fnv1a_64 as ssm_fnv1a_64,
    CoWMapping,
    Page,
    PageId,
    SnapshotConfig as CowSnapshotConfig,
    SnapshotDiff as CowSnapshotDiff,
    SnapshotError as CowSnapshotError,
    SnapshotId as CowSnapshotId,
    SnapshotMetadata,
    SnapshotStats as CowSnapshotStats,
    // Collisions with snapshot_manager exports — use Cow* prefix:
    StorageSnapshotManager as CowStorageSnapshotManager,
};
