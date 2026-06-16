//! IPFRS TensorLogic - Integration with TensorLogic IR
//!
//! This crate provides comprehensive integration between IPFRS and TensorLogic including:
//!
//! ## Core Features
//!
//! - **IR (Intermediate Representation)**: Serialization and storage of logic terms
//! - **Term and Predicate Storage**: Content-addressed storage for logical statements
//! - **Distributed Reasoning**: Query caching, goal decomposition, and proof assembly
//! - **Zero-Copy Tensor Transport**: Apache Arrow integration for efficient data sharing
//! - **Safetensors Support**: Read/write Safetensors format for ML models
//! - **PyTorch Checkpoint Support**: Load/save PyTorch .pt/.pth model checkpoints
//! - **Shared Memory**: Cross-process memory mapping for large tensors
//! - **Gradient Management**: Compression, aggregation, and differential privacy
//! - **Model Version Control**: Git-like versioning for ML models
//! - **Provenance Tracking**: Complete lineage tracking for datasets and models
//! - **Computation Graphs**: IPLD-based graph storage with optimization
//! - **Device Management**: Heterogeneous device support with adaptive batch sizing
//! - **FFI Profiling**: Overhead measurement and bottleneck identification
//! - **Allocation Optimization**: Buffer pooling and zero-copy conversions
//! - **GPU Support**: Stub implementation for future CUDA/OpenCL/Vulkan integration
//!
//! ## Performance Targets
//!
//! - FFI call overhead: < 1μs
//! - Zero-copy tensor access: < 100ns
//! - Query cache lookup: < 1μs
//! - Term serialization: < 10μs for small terms
//!
//! # Examples
//!
//! ## Basic Term and Predicate Creation
//!
//! ```
//! use ipfrs_tensorlogic::{Term, Predicate, Constant};
//!
//! // Create terms
//! let alice = Term::Const(Constant::String("Alice".to_string()));
//! let bob = Term::Const(Constant::String("Bob".to_string()));
//! let x = Term::Var("X".to_string());
//!
//! // Create a predicate: parent(Alice, Bob)
//! let pred = Predicate::new("parent".to_string(), vec![alice, bob]);
//! assert!(pred.is_ground());
//! ```
//!
//! ## Zero-Copy Tensor Operations
//!
//! ```
//! use ipfrs_tensorlogic::{ArrowTensor, ArrowTensorStore};
//!
//! // Create a tensor from f32 data
//! let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
//! let tensor = ArrowTensor::from_slice_f32("my_tensor", vec![4], &data);
//!
//! // Zero-copy access to the data
//! let slice = tensor.as_slice_f32().expect("example: should succeed in docs");
//! assert_eq!(slice[0], 1.0);
//!
//! // Create a tensor store
//! let mut store = ArrowTensorStore::new();
//! store.insert(tensor);
//! assert_eq!(store.len(), 1);
//! ```
//!
//! ## Query Caching
//!
//! ```
//! use ipfrs_tensorlogic::{QueryCache, QueryKey};
//!
//! // Create a cache with capacity 100
//! let cache = QueryCache::new(100);
//!
//! // Insert a query result
//! let key = QueryKey {
//!     predicate_name: "parent".to_string(),
//!     ground_args: vec![],
//! };
//! cache.insert(key.clone(), vec![]);
//!
//! // Retrieve from cache
//! let result = cache.get(&key);
//! assert!(result.is_some());
//! ```
//!
//! ## Gradient Compression
//!
//! ```
//! use ipfrs_tensorlogic::GradientCompressor;
//!
//! let gradient = vec![0.1, 0.5, 0.01, 0.8, 0.02];
//!
//! // Top-k compression (keep largest 2 values)
//! let sparse = GradientCompressor::top_k(&gradient, vec![5], 2).expect("example: should succeed in docs");
//! assert_eq!(sparse.nnz(), 2); // Only 2 non-zero elements
//!
//! // Quantization to int8
//! let quantized = GradientCompressor::quantize(&gradient, vec![5]);
//! assert!(quantized.compression_ratio() > 1.0);
//! ```
//!
//! ## Device-Aware Batch Sizing
//!
//! ```
//! use ipfrs_tensorlogic::{DeviceCapabilities, AdaptiveBatchSizer};
//! use std::sync::Arc;
//!
//! // Detect device capabilities
//! let caps = DeviceCapabilities::detect().expect("example: should succeed in docs");
//! println!("Device: {:?}, Memory: {} GB",
//!          caps.device_type,
//!          caps.memory.total_bytes / 1024 / 1024 / 1024);
//!
//! // Create adaptive batch sizer
//! let sizer = AdaptiveBatchSizer::new(Arc::new(caps))
//!     .with_min_batch_size(1)
//!     .with_max_batch_size(256);
//!
//! // Calculate optimal batch size
//! let model_size = 500 * 1024 * 1024;  // 500MB model
//! let item_size = 256 * 1024;          // 256KB per item
//! let batch_size = sizer.calculate(item_size, model_size);
//! println!("Optimal batch size: {}", batch_size);
//! ```
//!
//! ## FFI Profiling
//!
//! ```
//! use ipfrs_tensorlogic::{FfiProfiler, global_profiler};
//!
//! let profiler = FfiProfiler::new();
//!
//! // Profile a function call
//! {
//!     let _guard = profiler.start("my_ffi_function");
//!     // Your FFI code here
//! }
//!
//! // Get statistics
//! let stats = profiler.get_stats("my_ffi_function").expect("example: should succeed in docs");
//! println!("Calls: {}, Avg: {:?}", stats.call_count, stats.avg_duration);
//!
//! // Use global profiler
//! let global = global_profiler();
//! let _guard = global.start("global_operation");
//! ```
//!
//! ## Buffer Pooling
//!
//! ```
//! use ipfrs_tensorlogic::BufferPool;
//!
//! let pool = BufferPool::new(4096, 10); // 4KB buffers, max 10 pooled
//!
//! // Acquire buffer from pool
//! let mut buffer = pool.acquire();
//! buffer.as_mut().extend_from_slice(&[1, 2, 3, 4]);
//!
//! // Buffer automatically returned to pool when dropped
//! drop(buffer);
//! assert!(pool.size() > 0); // Buffer available for reuse
//! ```
//!
//! ## Zero-Copy Conversions
//!
//! ```
//! use ipfrs_tensorlogic::ZeroCopyConverter;
//!
//! let floats: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
//!
//! // Zero-copy conversion to bytes
//! let bytes = ZeroCopyConverter::slice_to_bytes(&floats);
//! assert_eq!(bytes.len(), 16); // 4 floats * 4 bytes
//!
//! // Zero-copy conversion back
//! let floats_back: &[f32] = ZeroCopyConverter::bytes_to_slice(bytes);
//! assert_eq!(floats, floats_back);
//! ```
//!
//! ## Safetensors Multi-Dtype Support
//!
//! ```
//! use ipfrs_tensorlogic::{SafetensorsWriter, SafetensorsReader, ArrowTensor};
//! use bytes::Bytes;
//!
//! // Create a model with multiple data types
//! let mut writer = SafetensorsWriter::new();
//!
//! // Add float32 weights
//! writer.add_f32("layer1.weights", vec![128, 64], &vec![0.1; 8192]);
//!
//! // Add float64 biases for high precision
//! writer.add_f64("layer1.bias", vec![64], &vec![0.01; 64]);
//!
//! // Add int32 indices
//! writer.add_i32("vocab_indices", vec![1000], &vec![42; 1000]);
//!
//! // Add int64 large IDs
//! writer.add_i64("entity_ids", vec![100], &vec![1000000; 100]);
//!
//! // Serialize and read back
//! let bytes = writer.serialize().expect("example: should succeed in docs");
//! let reader = SafetensorsReader::from_bytes(Bytes::from(bytes)).expect("example: should succeed in docs");
//!
//! // Load as Arrow tensors for zero-copy access
//! let weights = reader.load_as_arrow("layer1.weights").expect("example: should succeed in docs");
//! let bias = reader.load_as_arrow("layer1.bias").expect("example: should succeed in docs");
//! let indices = reader.load_as_arrow("vocab_indices").expect("example: should succeed in docs");
//! let ids = reader.load_as_arrow("entity_ids").expect("example: should succeed in docs");
//!
//! assert!(weights.as_slice_f32().is_some());
//! assert!(bias.as_slice_f64().is_some());
//! assert!(indices.as_slice_i32().is_some());
//! assert!(ids.as_slice_i64().is_some());
//! ```
//!
//! ## Memory Profiling
//!
//! ```
//! use ipfrs_tensorlogic::MemoryProfiler;
//! use std::time::Duration;
//!
//! let profiler = MemoryProfiler::new();
//!
//! {
//!     let _guard = profiler.start_tracking("tensor_allocation");
//!     let data: Vec<f32> = vec![1.0; 1000000]; // ~4 MB
//!     std::thread::sleep(Duration::from_millis(10));
//!     drop(data);
//! }
//!
//! let stats = profiler.get_stats("tensor_allocation").expect("example: should succeed in docs");
//! assert_eq!(stats.track_count, 1);
//! assert!(stats.total_duration >= Duration::from_millis(10));
//!
//! // Generate and print a report
//! let report = profiler.generate_report();
//! println!("Total operations tracked: {}", report.total_operations);
//! ```
//!
//! ## Model Quantization
//!
//! ```
//! use ipfrs_tensorlogic::{QuantizedTensor, QuantizationConfig};
//!
//! // Per-tensor INT8 symmetric quantization
//! let weights = vec![0.5, -0.3, 0.8, -0.1];
//! let config = QuantizationConfig::int8_symmetric();
//! let quantized = QuantizedTensor::quantize_per_tensor(&weights, vec![4], config).expect("example: should succeed in docs");
//!
//! // Dequantize back to f32
//! let dequantized = quantized.dequantize();
//! assert_eq!(dequantized.len(), 4);
//!
//! // Check compression ratio
//! println!("Compression: {:.2}x", quantized.compression_ratio());
//! ```
//!
//! ## More Examples
//!
//! For complete examples, see the `examples/` directory:
//! - `basic_reasoning.rs` - TensorLogic inference and backward chaining
//! - `query_optimization.rs` - Materialized views and query caching
//! - `proof_storage.rs` - Proof fragment management and compression
//! - `proof_explanation_demo.rs` - Automatic proof explanation in natural language (multiple styles)
//! - `model_versioning.rs` - Git-like version control for ML models
//! - `model_quantization.rs` - Model quantization for edge deployment (INT4/INT8, per-channel, dynamic)
//! - `tensor_storage.rs` - Safetensors and Arrow integration
//! - `device_aware_training.rs` - Device detection and adaptive batching
//! - `federated_learning.rs` - Gradient compression and differential privacy
//! - `allocation_optimization.rs` - Buffer pooling and zero-copy techniques
//! - `ffi_profiling.rs` - FFI overhead measurement
//! - `distributed_graph_execution.rs` - Graph partitioning across multiple workers
//! - `memory_profiling.rs` - Memory usage tracking and profiling
//! - `visualization_demo.rs` - Graph and proof visualization with DOT format

use ipfrs_core::Cid;
use serde::{Deserialize, Deserializer, Serializer};

pub mod kernel_registry;
pub use kernel_registry::{
    KernelDescriptor, KernelPrecision, KernelQuery, KernelRegistryStats, KernelTarget,
    TensorKernelRegistry,
};

pub mod allocation_optimizer;
pub mod arrow;
pub mod audit_log;
pub mod cache;
pub mod checkpoint_manager;
pub mod checkpoint_v2;
pub mod codec_registry;
pub mod computation_graph;
pub mod consensus;
pub mod constraint_solver;
pub mod datalog;
pub mod device;
pub mod distributed_backward_chainer;
pub mod feed_forward;
pub mod ffi_profiler;
pub mod gpu;
pub mod gradient;
pub mod gradient_accumulator;
pub mod gradient_clipper;
pub mod gradient_noise;
pub mod gradient_sparsify;
pub mod graph_partitioner;
pub mod inference_cache;
pub mod inference_trace;
pub mod ipld_codec;
pub mod ipld_path;
pub mod ir;
pub mod kb_federation;
pub mod kg_traversal;
pub mod memory_profiler;
pub mod memory_tracker;
pub mod multi_hop;
pub mod op_scheduler;
pub mod optimizer;
pub mod privacy_budget;
pub mod proof_cache;
pub mod proof_explanation;
pub mod proof_serializer;
pub mod proof_storage;
pub mod proof_tree;
pub mod proof_tree_export;
pub mod proof_tree_streaming;
pub mod proof_verifier;
pub mod provenance;
pub mod pytorch_checkpoint;
pub mod quantization;
pub mod reasoning;
pub mod recursive_reasoning;
pub mod remote_reasoning;
pub mod rule_conflict_v2;
pub mod rule_dependency;
pub mod rule_profiler;
pub mod rule_versioning;
pub mod safetensors_support;
pub mod session_manager;
pub mod session_replay;
pub mod shared_memory;
pub mod storage;
pub mod tensor_arena;
pub mod tensor_diff;
pub mod tensor_pool;
pub mod term_index;
pub mod utils;
pub mod version_control;
pub mod versioned_cache;
pub mod visualization;
// ConstraintSolver — CSP solver with AC-3, backtracking, and MRV heuristics.
// Note: `SolverConfig` and `SolverResult` are aliased as `CspSolverConfig` and
// `CspSolverResult` to avoid collision with the same names already exported
// from `markov_decision_process` (as `MdpSolverConfig` / `MdpSolverResult`).
pub use constraint_solver::{
    Assignment as CspAssignment, Constraint as CspConstraint, ConstraintSolver, CspError, CspStats,
    CspVarId, CspVariable, Domain as CspDomain, SolverConfig as CspSolverConfig,
    SolverResult as CspSolverResult,
};
pub mod budget_manager;
pub mod early_stopping;
pub mod rule_migrator;
pub mod slice_manager;
pub mod tensor_checksum;
pub mod tensor_gc;
pub use early_stopping::{
    EarlyStoppingConfig, EarlyStoppingMonitor, EarlyStoppingStats, EpochMetrics, StopCriterion,
    StopDecision,
};
pub mod dependency_graph;
pub mod event_bus_v2;
pub mod feature_extractor;
pub mod flow_controller;
pub mod inference_scheduler;
pub mod memory_layout;
pub mod memory_pool;
pub mod ml_feature_extractor;
pub mod op_dispatcher;
pub mod op_fusion;
pub mod query_optimizer;
pub mod rule_index;
pub mod rule_validator;

// TensorMemoryLayout — manages tensor memory layout descriptors
// Note: `TensorShape` is also exported as `MemoryLayoutShape` for callers
// that need to import it alongside other crates that define their own
// `TensorShape`.
pub use memory_layout::{
    LayoutDescriptor, LayoutOrder, LayoutStats, MemoryLayoutShape, TensorMemoryLayout,
    TensorShape as MemoryTensorShape,
};

// TensorFeatureExtractor — statistical and structural feature extraction
pub use feature_extractor::{
    ExtractedFeature, ExtractionResult, ExtractorConfig, ExtractorStats, FeatureKind,
    TensorFeatureExtractor,
};

// FeatureExtractor — composable ML preprocessing pipeline
// Note: `ExtractorStats` from `ml_feature_extractor` is re-exported as
// `FeExtractorStats` to avoid collision with `feature_extractor::ExtractorStats`.
pub use ml_feature_extractor::{
    fit_minmax_scaler, fit_onehot, fit_standard_scaler, ExtractedFeatures, FePipelineStats,
    FeatureError, FeatureExtractor, FeatureSpec, FeatureTransform, FeatureValue,
};

// TensorOpDispatcher — routes tensor operations to registered backends
// (CPU/GPU/Remote/Simulated) with priority-ordered fallback chains and
// per-backend statistics.
pub use op_dispatcher::{
    BackendKind, BackendRegistration, BackendStats, DispatchOp, DispatchResult, DispatcherStats,
    TensorOpDispatcher,
};

// TensorOpFusion — detects and fuses sequences of tensor operations into
// optimised compound operations, reducing memory bandwidth overhead.
// Note: `TensorOp` is re-exported as `FusionTensorOp` to avoid collision
// with `computation_graph::TensorOp` which is already exported at crate root.
pub use op_fusion::{FusedOp, FusionPlan, FusionStats, TensorOp as FusionTensorOp, TensorOpFusion};

// TensorMemoryPool — slab-based memory pool with size-class bucketing
pub use memory_pool::{MemoryPoolStats, PoolSlot, SizeClass, TensorMemoryPool};

// TensorBlockPool — pre-allocated fixed-size block pool with owner tracking,
// reservation, defragmentation, and shrink-to-fit.
pub use memory_pool::{BlockPoolStats, BlockStatus, MemoryBlock, PoolConfig, TensorBlockPool};

// TensorRuleIndex — multi-dimensional index over TensorLogic rules
pub use rule_index::{IndexedRule, RuleArity, RuleIndexStats, RuleQuery, TensorRuleIndex};

// TensorInferenceScheduler — deadline-aware priority scheduling for inference jobs
pub use inference_scheduler::{
    InferenceJob, JobStatus, SchedulerConfig, SchedulerStats, TensorInferenceScheduler,
};

// TensorQueryOptimizer — query plan rewriting
pub use query_optimizer::{
    estimated_cost, OptimizationResult, OptimizationRule, OptimizerStats, QueryNode,
    TensorQueryOptimizer,
};

// Tensor flow controller — backpressure, rate limiting, priority-based admission
pub use flow_controller::{
    FlowControllerConfig, FlowItem, FlowPriority, FlowState, FlowStats, TensorFlowController,
};

// Tensor rule validator
pub use rule_validator::{
    RuleSpec, TensorRuleValidator, ValidationError, ValidationResult, ValidatorConfig,
};

// Tensor dependency graph
pub use dependency_graph::{
    DependencyEdge, DependencyKind, DirtySet, GraphStats, TensorDependencyGraph,
};

// Checkpoint pruning and validation
pub use checkpoint_manager::{
    crc32, CheckpointPruner, CheckpointRecord, CheckpointValidator, RetentionPolicy,
    ValidationError as CheckpointValidationError,
};

// Distributed backward chaining
pub use distributed_backward_chainer::{Binding, DistributedBackwardChainer};

// Multi-hop rule resolution
pub use multi_hop::{
    HopRecord, HopTrace, MultiHopConfig, MultiHopResolver, MultiHopResult, VisitedSet,
};

// Proof tree
pub use proof_tree::{ProofNode, ProofTree};

// Proof tree streaming
pub use proof_tree_streaming::{
    ProofTreeStreamSummary, ProofTreeStreamer, ProofTreeUpdate, ProofTreeUpdateSink,
};

// Allocation optimization
pub use allocation_optimizer::{
    AdaptiveBuffer, AllocationError, BufferPool, PooledBuffer, StackBuffer, TypedBufferPool,
    TypedPooledBuffer, ZeroCopyConverter,
};

// Tensor pool (slab-based, power-of-two bucket pool for Arrow IPC zero-copy operations)
pub use tensor_pool::{
    bucket_for, PooledBuffer as TensorPoolBuffer, TensorPool, TensorPoolConfig, TensorPoolSnapshot,
    TensorPoolStats, NUM_BUCKETS,
};

// Arrow integration
pub use arrow::{ArrowTensor, ArrowTensorStore, TensorDtype, TensorMetadata, ZeroCopyAccessor};

// Caching
pub use cache::{
    CacheManager, CacheStats, CacheStatsSnapshot, CombinedCacheStats, QueryCache, QueryKey,
    RemoteFactCache,
};

// Computation graphs
pub use computation_graph::{
    BatchScheduler, ComputationGraph, DistributedExecutor, ExecutionBatch, GraphError, GraphNode,
    GraphOptimizer, GraphPartition, LazyCache, NodeAssignment, ParallelExecutor, StreamChunk,
    StreamingExecutor, TensorOp,
};

// Datalog parsing
pub use datalog::{parse_fact, parse_query, parse_rule, DatalogParser, ParseError, Statement};

// Device capabilities
pub use device::{
    AdaptiveBatchSizer, CpuInfo, DeviceArch, DeviceCapabilities, DeviceError,
    DevicePerformanceTier, DeviceProfiler, DeviceType, MemoryInfo,
};

// FFI profiling
pub use ffi_profiler::{
    global_profiler, FfiCallGuard, FfiCallStats, FfiProfiler, OverheadSummary, ProfilingReport,
};

// Feedforward network layer for transformer blocks
pub use feed_forward::{
    FFLayer, FFStats, FeedForwardActivation, FeedForwardConfig, FeedForwardNetwork,
};

// GPU execution (stub for future integration)
pub use gpu::{
    GpuBackend, GpuBuffer, GpuDevice, GpuError, GpuExecutor, GpuKernel, GpuMemoryManager,
};

// Gradient storage
pub use gradient::{
    clip_gradient_norm, federated_average, load_gradient_from_arrow, store_gradient_as_arrow,
    AggregationMethod, BackwardPassConfig, BackwardPassCoordinator, BackwardPassStats,
    BackwardPassStep, BackwardStepStatus, ClientInfo, ClientState, ComputationGraphError,
    ComputationGraphStore, ComputationNode, ConvergenceDetector, DPMechanism, DifferentialPrivacy,
    DistributedGradientAccumulator, FederatedRound, GradientAggregator, GradientCheckpoint,
    GradientCompressor, GradientDelta, GradientError, GradientVerifier, LayerGradient,
    ModelSyncProtocol, PrivacyBudget as GradientPrivacyBudget, QuantizedGradient,
    SecureAggregation, SparseGradient,
};

// IR types
pub use ir::{Constant, KnowledgeBase, KnowledgeBaseStats, Predicate, Rule, Term, TermRef};

// Memory profiling
pub use memory_profiler::{
    MemoryProfiler, MemoryProfilingReport, MemoryStats, MemoryTrackingGuard,
};

// Query optimization
pub use optimizer::{
    OptimizationRecommendation, PlanNode, PredicateStats, QueryOptimizer, QueryPlan,
};

// Reasoning
pub use reasoning::{
    apply_subst_predicate, rename_rule_vars, unify_predicates, CycleDetector, DistributedReasoner,
    GoalDecomposition, InferenceEngine, MemoizedInferenceEngine, Proof, ProofRule, Substitution,
};

// Recursive reasoning
pub use recursive_reasoning::{
    FixpointEngine, StratificationAnalyzer, StratificationResult, TableStats, TabledInferenceEngine,
};

// Remote reasoning
pub use remote_reasoning::{
    DistributedGoalResolver, DistributedInferenceSession, DistributedProofAssembler,
    DistributedReasonerConfig, DistributedReasonerV2, FactDiscoveryRequest, FactDiscoveryResponse,
    GoalResolutionRequest, GoalResolutionResponse, IncrementalLoadRequest, IncrementalLoadResponse,
    InferenceRequest, InferenceResponse, InferenceResultStream, MockRemoteKnowledgeProvider,
    PartialResult, QueryRequest, QueryResponse, ReasoningError, RemoteKnowledgeProvider,
    RemoteReasoningError, RemoteResult, SessionMetrics, SessionStats,
};

// Proof storage
pub use proof_storage::{
    ProofAssembler, ProofFragment, ProofFragmentRef, ProofFragmentStore, ProofMetadata, RuleRef,
};

// Proof explanation
pub use proof_explanation::{
    ExplanationConfig, ExplanationStyle, FragmentProofExplainer, ProofExplainer,
    ProofExplanationBuilder,
};

// Provenance tracking
pub use provenance::{
    Attribution, DatasetProvenance, Hyperparameters, License, LineageTrace, ProvenanceError,
    ProvenanceGraph, TrainingProvenance,
};

// PyTorch checkpoint support
pub use pytorch_checkpoint::{
    CheckpointMetadata, OptimizerState, ParamState, PyTorchCheckpoint, StateDict, TensorData,
};

// Quantization support
pub use quantization::{
    CalibrationMethod, DynamicQuantizer, QuantizationConfig, QuantizationError,
    QuantizationGranularity, QuantizationParams, QuantizationScheme, QuantizedTensor,
};

// Safetensors support
pub use safetensors_support::{
    ChunkedModelStorage, ModelSummary, SafetensorError, SafetensorsReader, SafetensorsWriter,
    TensorInfo,
};

// Shared memory
pub use shared_memory::{
    SharedMemoryError, SharedMemoryPool, SharedTensorBuffer, SharedTensorBufferReadOnly,
    SharedTensorInfo,
};

// IPLD codec
pub use ipld_codec::{
    block_to_fact, block_to_kb, block_to_rule, fact_cid, fact_ipld_to_predicate, fact_to_block,
    kb_to_block, predicate_to_fact_ipld, predicate_to_term_ipld, rule_cid, rule_ipld_to_rule,
    rule_to_block, rule_to_rule_ipld, term_ipld_to_predicate, FactIpld, KnowledgeBaseIpld,
    RuleIpld, TermIpld,
};

// IPLD path resolution
pub use ipld_path::{IpldPathResolver, IpldPathValue, PathError};

// Storage
pub use storage::{
    FactSnapshot, KnowledgeBaseSnapshot, RuleSnapshot, TensorLogicError,
    TensorLogicPersistenceConfig, TensorLogicStore, TensorLogicStoreStats,
};

// Inference cache
pub use inference_cache::{
    hash_goal as inference_hash_goal, CacheStats as InferenceCacheStats, CachedResult,
    InferenceCache, InferenceCacheKey,
};

// Versioned inference cache
pub use versioned_cache::{
    CacheEntry, CacheError, CacheKey, CacheStatsSnapshot as VersionedCacheStatsSnapshot,
    VersionedInferenceCache,
};

// Knowledge base federation
pub use kb_federation::{
    export_kb_as_cid, import_remote_kb, merge_knowledge_bases, KbConflict, KbMergeDiff,
};

// Utilities
pub use utils::{KnowledgeBaseUtils, PredicateBuilder, QueryUtils, RuleBuilder, TermUtils};

// Version control
pub use version_control::{
    Branch, LayerDiff, ModelCommit, ModelDiff, ModelDiffer, ModelRepository, VersionControlError,
};

// Rule set versioning and conflict resolution
pub use rule_versioning::{
    ConflictResolver, ConflictStrategy, ResolvedRuleSet, RuleSetDiff, RuleSetVersion,
    VersionedRuleSet,
};

// Distributed session manager
pub use session_manager::{
    DistributedSessionManager, PeerId as SessionPeerId, SessionError, SessionId,
    SessionMetrics as SessionManagerMetrics, SessionMetricsSnapshot, SessionStatus,
    MAX_CONCURRENT_SESSIONS,
};

// Visualization
pub use visualization::{GraphVisualizer, ProofVisualizer};

// Privacy budget accounting (differential privacy)
pub use privacy_budget::{
    BudgetError, BudgetSnapshot, PerRoundBudget, PrivacyBudget as DpPrivacyBudget, RenyiAccountant,
    RoundGuard,
};

// Knowledge graph traversal
pub use kg_traversal::{
    EdgeType, KgEdge, KgError, KgNode, KnowledgeGraph, KnowledgeGraphTraverser, NodeType,
};

// Round consensus tracking for federated learning
pub use consensus::{
    ConsensusError, ConsensusStats, ConsensusStatsSnapshot, PeerVote, QuorumPolicy, QuorumResult,
    RoundConsensusTracker, RoundId, RoundStatus, Vote,
};

// Codec registry — compression/encoding codec selection and peer negotiation
pub use codec_registry::{
    CodecDescriptor, CodecError, CodecId, CodecNegotiationRecord, CodecRegistry, SpeedClass,
};

// Inference audit log — immutable compliance trail for distributed inference queries
pub use audit_log::{
    AuditEntry, AuditError, AuditEvent, AuditStats, AuditStatsSnapshot, InferenceAuditLog,
};

// Rule dependency graph — topological evaluation scheduling
pub use rule_dependency::{
    DepError, DependencyType, EvaluationSchedule, RuleDependency, RuleDependencyGraph, RuleId,
};

// Gradient sparsification and delta encoding for bandwidth-constrained federated learning.
// Note: `SparseGradientV2` / `GradientDeltaV2` are used as aliases to avoid
// collisions with the existing types in the `gradient` module.
pub use gradient_sparsify::{
    DeltaEncoder, DeltaStats, GradientDelta as GradientDeltaV2, GradientSparsifier,
    SparseGradient as SparseGradientV2, SparsifierStats, SparsityConfig,
};

// Gradient noise injection — training regularization via configurable noise distributions.
pub use gradient_noise::{
    GradientNoiseConfig, GradientNoiseInjector, NoiseSample, NoiseStats, NoiseType,
};

// Gradient clipping — norm and value clipping strategies to prevent gradient explosion.
pub use gradient_clipper::{
    ClipperStats, ClippingResult, ClippingStrategy, GradientTensor, TensorGradientClipper,
};

// Proof serializer — serialize/deserialize distributed proof trees for IPLD.
pub use proof_serializer::{
    ProofNodeInput, ProofNodeRecord, ProofSerError, ProofSerializer, ProofSerializerStats,
    ProofSerializerStatsSnapshot, ProofTreeInput, SerializedProof,
};

// Tensor arena — bump-allocating slab arena for inference pipeline tensors.
pub use tensor_arena::{ArenaError, ArenaRegion, ArenaSlice, ArenaStats, TensorArena};

// Proof caching layer — LFU-evicting, TTL-expiring cache for proof results.
pub use proof_cache::{
    fnv1a_hash, CachedProof, ProofCacheConfig, ProofCacheKey, ProofCacheStats, ProofCachingLayer,
};

// TensorStateSnapshot — capture/restore TensorLogic runtime state.
pub mod state_snapshot;
pub use state_snapshot::{
    fnv1a_u64, FieldData, SnapshotDelta, SnapshotField, StateSnapshot, StateSnapshotStats,
    TensorStateSnapshot,
};

// TensorProvenanceTracker — full lineage tracking for tensor values and
// inference results.
pub mod provenance_tracker;
pub use provenance_tracker::{
    ProvenanceChain, ProvenanceKind, ProvenanceRecord, ProvenanceStats, TensorProvenanceTracker,
};

// TensorExecutionTracer — records detailed execution traces of TensorLogic
// inference operations for debugging, profiling, and replay purposes.
pub mod execution_tracer;

// TensorOptimizationHistory — records and analyzes optimization steps to
// detect convergence, track best results, and guide adaptive LR schedules.
pub mod optimization_history;
pub use execution_tracer::{
    TensorExecutionTracer, TraceEvent, TraceEventKind, TraceSummary, TracerConfig, TracerStats,
};
pub use optimization_history::{
    ConvergenceStatus, HistoryStats, OptimizationHistoryConfig, OptimizationStep,
    TensorOptimizationHistory,
};

// TensorCheckpointScheduler — step/tick/loss-triggered automatic checkpointing.
// Note: `CheckpointRecord`, `SchedulerConfig`, and `SchedulerStats` are aliased
// to avoid collisions with the same names already exported from
// `checkpoint_manager` / `inference_scheduler`.
pub mod checkpoint_scheduler;
pub use checkpoint_scheduler::{
    CheckpointRecord as SchedulerCheckpointRecord, CheckpointTrigger,
    SchedulerConfig as CheckpointSchedulerConfig, SchedulerStats as CheckpointSchedulerStats,
    TensorCheckpointScheduler,
};

// TensorGradAccumulator — mini-batch gradient accumulation before optimizer step.
// Note: `AccumulatorStats` is aliased to `GradAccumulatorStats` to avoid
// collision with `gradient_accumulator::AccumulatorStats` already exported
// at crate root.
pub mod grad_accumulator;
pub use grad_accumulator::{
    AccumulationMode, AccumulatorConfig as GradAccumulatorConfig,
    AccumulatorStats as GradAccumulatorStats, GradBuffer, TensorGradAccumulator,
};

// Reverse-mode automatic differentiation (autograd) for scalar-output functions.
pub mod autograd;
pub use autograd::{AutogradGraph, AutogradNode, AutogradOp, NodeId};

// TensorSliceView — zero-copy logical views via offset+stride descriptors.
pub mod slice_view;
pub use slice_view::{BroadcastShape, SliceRange, SliceViewStats, TensorSliceView, ViewDescriptor};

// Batch normalisation layer with running statistics tracking, training/inference
// modes, and configurable epsilon/momentum parameters.
pub mod batch_norm;
pub use batch_norm::{BatchNormConfig, BatchNormStats, NormMode, TensorBatchNorm};

// TensorQuantizer — symmetric/asymmetric calibration-based quantization (INT8/INT4).
pub mod quantizer;
pub use quantizer::{QuantBits, QuantMode, QuantParams, QuantizerStats, TensorQuantizer};

// MultiPrecisionQuantizer — multi-precision tensor quantization (INT8, INT4, FP16, BF16).
// Note: Several names conflict with existing exports:
//   - `QuantizedTensor`  → `TqQuantizedTensor`   (conflicts with quantization module)
//   - `QuantizerStats`   → `TqQuantizerStats`     (conflicts with quantizer module)
//   - `TensorQuantizer`  → `MultiPrecisionQuantizer` (conflicts with quantizer module)
pub mod tensor_quantizer;
pub use tensor_quantizer::{
    percentile as tq_percentile, DequantizedTensor as TqDequantizedTensor, QuantizationMode,
    QuantizedTensor as TqQuantizedTensor, QuantizerConfig, QuantizerError,
    QuantizerStats as TqQuantizerStats, TensorQuantizer as MultiPrecisionQuantizer,
};

// TensorCheckpointer — periodic checkpointing of tensor computation state with rollback.
pub mod checkpointer;
pub use checkpointer::{Checkpoint, CheckpointConfig, CheckpointerStats, TensorCheckpointer};

// TensorProfiler — operation profiling for tensor computations.
// Note: `ProfilerStats` is aliased to `TensorProfilerStats` to avoid
// collision with `rule_profiler::ProfilerStats`.
pub mod profiler;
pub use profiler::{OpProfile, ProfileEntry, ProfilerStats as TensorProfilerStats, TensorProfiler};

// TensorDataLoader — batch data loading with shuffling and epoch tracking.
pub mod data_loader;
pub use data_loader::{DataBatch, DataLoaderConfig, DataLoaderStats, TensorDataLoader};

// TensorShapeInference — static shape inference for tensor operation graphs.
// Note: `TensorShape` is re-exported as `InferenceTensorShape` to avoid
// collision with `memory_layout::TensorShape` (already exported as
// `MemoryTensorShape`).
pub mod shape_inference;
pub use shape_inference::{
    InferenceRule, ShapeInferenceStats, ShapeOp, TensorShape as InferenceTensorShape,
    TensorShapeInference,
};

// TensorLossFunction — common loss functions for tensor computations
// (MSE, MAE, CrossEntropy, Huber, Hinge) with gradient support.
pub mod loss_function;
pub use loss_function::{LossConfig, LossFunctionStats, LossType, Reduction, TensorLossFunction};

// TensorActivation — activation functions with forward/backward passes.
pub mod activation;
pub use activation::{ActivationConfig, ActivationStats, ActivationType, TensorActivation};

// ActivationFunction — richer activation layer with derivative, vectorised
// ops, dead-ReLU tracking, and extended type set (ELU, Mish, HardSwish, …).
// Names collide with `activation`; re-export under `Af*` aliases.
pub mod activation_function;
pub use activation_function::ActivationConfig as AfActivationConfig;
pub use activation_function::ActivationFunction;
pub use activation_function::ActivationStats as AfActivationStats;
pub use activation_function::ActivationType as AfActivationType;

// TensorRegularizer — L1/L2/ElasticNet regularization for tensor parameters.
pub mod regularizer;
pub use regularizer::{RegularizerConfig, RegularizerStats, RegularizerType, TensorRegularizer};

// LossScaler — dynamic loss scaling for mixed-precision training to prevent
// gradient underflow (Static, Dynamic, Gradual policies).
pub mod loss_scaler;
pub use loss_scaler::{LossScaler, LossScalerConfig, ScaleUpdatePolicy, ScalerStats};

// TensorLRScheduler — learning rate scheduling strategies (Constant,
// StepDecay, ExponentialDecay, CosineAnnealing, WarmupLinear, OneCycleLR).
pub mod lr_scheduler;
pub use lr_scheduler::{
    LRSchedulerConfig,
    LRSchedulerStats,
    // Multi-strategy LearningRateScheduler
    LearningRateScheduler,
    LrHistory,
    LrSchedulerState,
    LrStats,
    ScheduleType,
    SchedulerStrategy,
    TensorLRScheduler,
};

// WeightInitializer — weight initialization strategies (Xavier, He, LeCun,
// Orthogonal, Sparse, TruncatedNormal) for tensor operations.
pub mod weight_initializer;
pub use weight_initializer::{
    FanMode, InitDistribution, InitStats, InitStrategy, TensorShape as InitTensorShape,
    WeightInitConfig, WeightInitializer,
};

// TensorOptimizer — SGD optimizer variants (vanilla, momentum, Nesterov)
// with weight decay and dampening support.
pub mod sgd_optimizer;
pub use sgd_optimizer::{
    OptimizerType, ParameterState, SGDConfig, SGDOptimizer, SGDOptimizerStats,
};

// ModelPruner — weight pruning with magnitude, structured, percentile,
// random, and gradual scheduling strategies.
pub mod model_pruner;
pub use model_pruner::{
    LayerWeights, ModelPruner, PrunerConfig, PrunerStats, PruningResult, PruningStrategy,
};

// AttentionMechanism — production-grade multi-head scaled dot-product attention
// with positional encoding, masking, and attention pattern analysis.
//
// Name collision notes:
//   - `AttentionStats`       → `AttnStats`                (new production type)
//   - `SimpleAttentionStats` replaces the old `AttentionStats` in the simple API
//   - `AttentionOutput`      → `AttentionOutput` (new, uses AttentionMatrix fields)
//   - Old simple variants exported with `Simple*` prefix
pub mod attention_mechanism;
pub use attention_mechanism::{
    causal_mask,
    matmul as attn_matmul,
    // Free-standing utilities
    scaled_dot_product_attention,
    softmax_1d,
    transpose as attn_transpose,
    // Configuration
    AttentionConfig,
    // Sub-types
    AttentionHead,
    // Matrix primitive
    AttentionMatrix,
    // Production-grade mechanism
    AttentionMechanism,
    // Output / stats
    AttentionOutput,
    // Error type
    AttnError,
    AttnStats,
    PositionalEncoding,
    // Simple / backward-compatible API
    SimpleAttentionConfig,
    SimpleAttentionMechanism,
    SimpleAttentionOutput,
    SimpleAttentionStats,
};

// GradientCheckpointer — gradient accumulation, checkpointing, and replay for
// distributed training with fault tolerance.
// Note: Several names conflict with existing exports:
//   - `GradientTensor`      → `GcGradientTensor`      (conflicts with gradient_clipper)
//   - `GradientCheckpoint`  → `GcGradientCheckpoint`  (conflicts with gradient module)
//   - `CheckpointerStats`   → `GcCheckpointerStats`   (conflicts with checkpointer module)
//   - `AccumulationMode`    → `GcAccumulationMode`    (conflicts with grad_accumulator)
pub mod gradient_checkpointer;
pub use gradient_checkpointer::{
    fnv1a_f64_slice, CheckpointId, CheckpointerConfig, GcAccumulationMode, GcCheckpointerStats,
    GcGradientCheckpoint, GcGradientTensor, GradientCheckpointer, GradientCheckpointerError,
};

// ModelEnsemble — multi-model ensemble aggregator supporting voting, averaging,
// and stacking strategies for distributed inference.
pub mod model_ensemble;
pub use model_ensemble::{
    EnsembleConfig, EnsembleError, EnsembleResult, EnsembleStats, EnsembleStrategy, ModelEnsemble,
    ModelMember, ModelPrediction,
};

// OnlineLearner — online / incremental learning algorithms for streaming data.
// Implements Perceptron, Passive-Aggressive (PA-I), and SGD with Momentum.
// Note: `LossFunction` is re-exported as `OlLossFunction` to avoid collision
// with `loss_function::LossType` and related names already at crate root.
pub mod online_learner;
pub use online_learner::{
    LearnerError, OlLossFunction, OnlineAlgorithm, OnlineLearner, OnlineLearnerStats,
    TrainingSample,
};

// AdaptiveOptimizer — Adam, AdaGrad, RMSProp, and AdamW optimizers for
// distributed gradient descent.
// Note: Several names conflict with existing exports:
//   - `OptimizerState`  → `AoOptimizerState`  (conflicts with pytorch_checkpoint)
//   - `OptimizerStats`  → `AoOptimizerStats`  (conflicts with query_optimizer)
pub mod adaptive_optimizer;
pub use adaptive_optimizer::{
    AdaptiveOptimizer, OptimizerAlgorithm, OptimizerError, OptimizerState as AoOptimizerState,
    OptimizerStats as AoOptimizerStats, ParameterGroup,
};

// NeuralArchitectureSearch — random / evolutionary NAS for discovering optimal network structures.
// Note: All public types are prefixed with `Nas` to avoid any future collision risk.
pub mod neural_arch_search;
pub use neural_arch_search::{
    fnv1a_nas, NasArchitecture, NasConfig, NasEvaluationResult, NasLayerType, NasSearchStrategy,
    NasStats, NeuralArchitectureSearch,
};

// HyperparameterTuner — Bayesian optimization, random search, and grid search
// for hyperparameter tuning with UCB acquisition and importance scoring.
pub mod hyperparameter_tuner;
pub use hyperparameter_tuner::{
    HpConfig, HpSpec, HpTunerError, HpType, HpValue, HyperparameterTuner, TunerConfig, TunerStats,
    TuningResult, TuningStrategy,
};

// MetaLearner — MAML-inspired meta-learning system that learns to learn.
pub mod meta_learner;
pub use meta_learner::{
    MetaError, MetaLearner, MetaLearnerConfig, MetaLearnerStats, MetaParameters, MetaTask,
    TaskAdaptation, TaskExample, TaskId, TaskType,
};

// ReinforcementLearner — tabular Q-learning, SARSA, and Double Q-learning agents
// for discrete reinforcement learning.
pub mod reinforcement_learner;
pub use reinforcement_learner::{
    ActionId, Experience, Policy, ReinforcementLearner, RlAlgorithm, RlError, RlStats, StateId,
};

// CausalInferenceEngine — do-calculus, interventional distributions,
// and counterfactual reasoning over Gaussian structural causal models.
pub mod causal_inference;
pub use causal_inference::{
    CausalEdge, CausalEdgeType, CausalError, CausalGraph, CausalInferenceEngine, CausalNode,
    CausalNodeId, CausalStats, CounterfactualQuery, InferenceResult, Intervention,
};

// DistributedOptimizer — coordinates distributed gradient aggregation across
// workers with staleness handling and fault tolerance.
// Note: `WorkerState` is re-exported as `DoWorkerState` to avoid collision
// with any future `WorkerState` names at crate root.
// BayesianUpdateEngine — conjugate-prior Bayesian belief updating.
pub mod bayesian_updater;
pub use bayesian_updater::{
    BayesError, BayesianUpdateEngine, CredibleInterval, Observation as BayesObservation,
    Posterior as BayesPosterior, Prior as BayesPrior,
};

pub mod distributed_optimizer;
pub use distributed_optimizer::{
    AggregatedGradient, AggregationStrategy, DistOptimizerStats, DistributedOptimizer,
    GradientUpdate as DoGradientUpdate, OptimizerDistError, WorkerId as DoWorkerId,
    WorkerState as DoWorkerState,
};

// GraphNeuralNetwork — message-passing GNN with node feature aggregation,
// edge weighting, and multi-layer propagation.
pub mod graph_neural_network;
pub use graph_neural_network::{
    xorshift64 as gnn_xorshift64, GnnActivation, GnnAggregation, GnnConfig, GnnEdge, GnnError,
    GnnLayer, GnnNodeId, GnnStats, GraphNeuralNetwork, NodeFeatures,
};

// DifferentialPrivacyEngine — Laplace/Gaussian/Randomized noise mechanisms,
// sensitivity clipping, privacy budget tracking, and composition theorems.
pub mod differential_privacy;
pub use differential_privacy::{
    BudgetTracker as DpBudgetTracker, DifferentialPrivacyEngine, DpError, DpQuery, DpResult,
    NoiseScale, PrivacyMechanism, PrivacyParameters as DpPrivacyParameters,
};

// FuzzyLogicEngine — membership functions, fuzzy rules, Mamdani/Sugeno
// inference, and centroid / mean-of-max / largest-of-max defuzzification.
pub mod fuzzy_logic;
pub use fuzzy_logic::{
    DefuzzMethod, FuzzyError, FuzzyLogicEngine, FuzzyProposition, FuzzyRule, FuzzySet, FuzzyStats,
    FuzzyVariable, InferenceMethod, MembershipFunction,
};

// FuzzyLogicEngine (full Mamdani) — production-quality engine with all MF
// variants (Triangle, Trapezoid, Gaussian, Bell, Sigmoid, Singleton, Linear),
// tree-structured FuzzyExpr antecedents (And/Or/Not/Very/Somewhat), Mamdani
// inference, and five defuzzification methods.
//
// Collision notes:
//   • `MembershipFunction`, `FuzzySet`, `FuzzyVariable`, `FuzzyRule`,
//     `FuzzyError`, `DefuzzMethod`, `FuzzyLogicEngine` already exported from
//     `fuzzy_logic`; prefixed with `Fle`.
//   • `InferenceResult` already exported from `causal_inference`; prefixed
//     with `Fle`.
pub mod fuzzy_logic_engine;
pub use fuzzy_logic_engine::{
    DefuzzMethod as FleDefuzzMethod, EngineConfig, EngineStats, FuzzyError as FleFuzzyError,
    FuzzyExpr, FuzzyLogicEngine as FleFuzzyLogicEngine, FuzzyRule as FleFuzzyRule,
    FuzzySet as FleFuzzySet, FuzzyVariable as FleFuzzyVariable,
    InferenceResult as FleInferenceResult, MembershipFunction as FleMembershipFunction,
};

// TemporalReasoningEngine — Allen's interval algebra, temporal constraints,
// event chains and windowed queries.
pub mod temporal_reasoning;
pub use temporal_reasoning::{
    AllenRelation, ConstraintViolation, TemporalConstraint, TemporalError, TemporalEvent,
    TemporalReasoningEngine, TemporalStats, TimeInterval, TimePoint,
};

// MarkovDecisionProcess — tabular MDP solver (Value Iteration, Policy Iteration, Q-values).
// Note: `StateId`, `ActionId`, and `Policy` are already exported from
// `reinforcement_learner`; the MDP equivalents are exported under the `Mdp*`
// prefix to avoid name collisions.
pub mod markov_decision_process;
pub use markov_decision_process::{
    xorshift64 as mdp_xorshift64, xorshift_f64 as mdp_xorshift_f64, MarkovDecisionProcess,
    MdpActionId, MdpError, MdpPolicy, MdpState, MdpStateId, MdpStats,
    SolverConfig as MdpSolverConfig, SolverResult as MdpSolverResult, SolverType as MdpSolverType,
    Transition as MdpTransition, ValueFunction as MdpValueFunction,
};

// NeuralSymbolicIntegrator — hybrid neural + symbolic inference engine.
pub mod neural_symbolic;
pub use neural_symbolic::{
    InferenceMode, IntegratorConfig, LogicalRule, NeuralSymbolicIntegrator, NsError, NsQuery,
    NsResult, NsStats, RuleType, Symbol, SymbolId,
};

// EpistemicLogicReasoner — multi-agent epistemic logic over finite Kripke structures.
pub mod epistemic_logic;
pub use epistemic_logic::{
    AccessibilityRelation, AgentId, EpistemicError, EpistemicFormula, EpistemicLogicReasoner,
    EpistemicStats, KripkeModel, PossibleWorld, WorldId,
};

// SymbolicNeuralOptimizer — hybrid symbolic + gradient-based optimizer.
// Note: `OptimizationStep`   is re-exported as `SnoOptimizationStep`  to avoid
//        collision with `optimization_history::OptimizationStep`.
//       `OptimizationResult` is re-exported as `SnoOptimizationResult` to avoid
//        collision with `query_optimizer::OptimizationResult`.
pub mod symbolic_neural_optimizer;
pub use symbolic_neural_optimizer::{
    parse_constraint_bound, xorshift64 as sno_xorshift64, ConstraintBound, OptimizationObjective,
    ParameterVector, SnoOptimizationResult, SnoOptimizationStep, SnoOptimizerConfig,
    SymbolicConstraint, SymbolicNeuralOptimizer,
};

// TemporalPatternMatcher — NFA-based temporal sequence pattern matching.
//
// Collision note: `TemporalConstraint` is already exported from
// `temporal_reasoning`; the version from this module is re-exported under the
// alias `TpmTemporalConstraint` to avoid the name conflict.
// Similarly `xorshift64` is already exported from `graph_neural_network` and
// `symbolic_neural_optimizer`; this one is exported as `tpm_xorshift64`.
pub mod temporal_pattern_matcher;
pub use temporal_pattern_matcher::{
    xorshift64 as tpm_xorshift64, EventLabel, MatchResult as TpmMatchResult, MatcherConfig,
    MatcherError, MatcherStats, NfaState, PatternStep, RepeatSpec,
    TemporalConstraint as TpmTemporalConstraint, TemporalPattern, TemporalPatternMatcher,
    TimedEvent,
};

// CausalChainTracer — production-quality causal chain tracing for event sequences.
// Collision note: `CausalEdge` and `CausalNode` already exist in `causal_inference`;
// those from this module are re-exported under `CctCausalEdge` / `CctCausalNode` aliases.
// `xorshift64` is re-exported as `cct_xorshift64`.
pub mod causal_chain_tracer;
pub use causal_chain_tracer::{
    xorshift64 as cct_xorshift64, CausalChain, CausalChainTracer, CausalEdge as CctCausalEdge,
    CausalNode as CctCausalNode, CausalRelation, TraceQuery, TracerConfig as CctTracerConfig,
    TracerError, TracerStats as CctTracerStats,
};

// RuleConflictResolver — production-quality logic rule conflict detection and resolution.
//
// Collision notes:
//   • `LogicRule` is new (distinct from `LogicalRule` in `neural_symbolic`).
//   • `ConflictType`, `ResolutionStrategy`, `ConflictRecord` exist inside
//     `rule_conflict_v2` but are NOT exported at crate root — no aliases needed.
//   • `xorshift64` is re-exported as `rcr_xorshift64` (consistent with gnn/sno/tpm/cct).
pub mod rule_conflict_resolver;
pub use rule_conflict_resolver::{
    xorshift64 as rcr_xorshift64, ConflictRecord, ConflictType, LogicRule, ResolutionStrategy,
    ResolverConfig, ResolverError, ResolverStats, RuleConflictResolver,
};

// BeliefRevisionEngine — AGM-style belief revision (expansion, contraction, revision,
// consolidation) with epistemic entrenchment, recency bias, source-priority, and
// minimal-change retention functions.
//
// Collision note: `xorshift64` is already exported from several modules; this one is
// re-exported as `bre_xorshift64`.
pub mod belief_revision_engine;
pub use belief_revision_engine::{
    xorshift64 as bre_xorshift64, Belief, BeliefRevisionEngine, BeliefSet, ConsistencyCheck,
    RetentionFunction, RevisionConfig, RevisionError, RevisionOp, RevisionStats,
};

// ProbabilisticLogicNetwork — indefinite truth values, PLN inference rules
// (Deduction, Induction, Abduction, Revision, Conjunction, Disjunction,
// Negation, ModusPonens), and hypergraph atom/link store.
//
// Collision notes:
//   • `InferenceResult` is already exported from `causal_inference`;
//     this module's type is re-exported as `PlnInferenceResult`.
//   • `InferenceRule` is already exported from `shape_inference`;
//     this module's type is re-exported as `PlnInferenceRule`.
pub mod probabilistic_logic_network;
pub use probabilistic_logic_network::{
    AtomType, LinkType, PlnAtom, PlnConfig, PlnError, PlnInferenceResult, PlnInferenceRule,
    PlnLink, PlnStats, ProbabilisticLogicNetwork, TruthValue,
};

// Collision notes:
//   • `EngineConfig` is already exported from `fuzzy_logic_engine`;
//     this module's type is re-exported as `HteEngineConfig`.
pub mod hypothesis_test_engine;
pub use hypothesis_test_engine::{
    chi2_p_value, normal_cdf, sample_stats, t_cdf_approx, xorshift64, xorshift_normal,
    EngineConfig as HteEngineConfig, Hypothesis, HypothesisTestEngine, SampleData, TestError,
    TestResult, TestStatistic, TestStats, TestType,
};

// ReinforcementLearningAgent — tabular RL with multiple algorithms
// (SARSA, Q-Learning, Expected SARSA, Double Q-Learning, N-Step TD)
// and multiple policies (EpsilonGreedy, Boltzmann, UCB, Random).
//
// Collision notes:
//   • `RlError` is already exported from `reinforcement_learner`; the new
//     error type is re-exported as `RlaRlError`.
//   • `xorshift64` / `xorshift_f64` are re-exported as `rla_xorshift64` /
//     `rla_xorshift_f64` (consistent naming convention).
pub mod reinforcement_learning_agent;
pub use reinforcement_learning_agent::{
    xorshift64 as rla_xorshift64, xorshift_f64 as rla_xorshift_f64, AgentConfig, AgentPolicy,
    AgentStats, AlgorithmType, EpisodeStats, ExperienceReplay, ReinforcementLearningAgent,
    RlAction, RlAgentError, RlState, Transition as RlaTransition,
};

// BayesianNetworkInference — variable elimination, belief propagation, and
// sampling-based inference over discrete Bayesian networks.
//
// Collision notes:
//   • `xorshift64` is already exported from `hypothesis_test_engine` at crate
//     root; this module's copy is re-exported as `bni_xorshift64`.
pub mod bayesian_network_inference;
pub use bayesian_network_inference::{
    bni_xorshift64, BayesianNetwork, BayesianNetworkInference, BniConfig, BniError, BniStats,
    ConditionalProbabilityTable, EliminationOrder, Evidence, Factor, InferenceAlgorithm,
    InferenceQuery, QueryResult, RandomVariable,
};

// MetaLearningOptimizer — MAML, Reptile, FOMAML, and ProtoNet meta-learning
// over a linear regression model.
//
// Collision notes:
//   • `TaskId`      is already exported from `meta_learner`; aliased as `MloTaskId`.
//   • `TaskExample` is already exported from `meta_learner`; aliased as `MloTaskExample`.
//   • `MetaTask`    is already exported from `meta_learner`; aliased as `MloMetaTask`.
//   • `MetaError`   is already exported from `meta_learner`; aliased as `MloMetaError`.
//   • `xorshift64`  is internal to this module and NOT re-exported at crate root.
pub mod meta_learning_optimizer;
pub use meta_learning_optimizer::{
    AdaptationStep, MetaAlgorithm, MetaError as MloMetaError, MetaLearningOptimizer, MetaStats,
    MetaTask as MloMetaTask, ModelParams, OptimizerConfig, TaskExample as MloTaskExample,
    TaskId as MloTaskId,
};

// TemporalKnowledgeGraph — tracks facts and relationships over time.
//
// Collision notes:
//   • `NodeId` is already exported from `autograd`; aliased as `TkgNodeId`.
//   • `QueryResult` is already exported from `bayesian_network_inference`; TkgQueryResult has no
//     collision because it carries the `Tkg` prefix already.
pub mod temporal_knowledge_graph;
pub use temporal_knowledge_graph::{
    EdgeId as TkgEdgeId, NodeId as TkgNodeId, TemporalKnowledgeGraph, TkgEdge, TkgError, TkgEvent,
    TkgGraphStats, TkgMergePolicy, TkgNode, TkgQuery, TkgQueryResult, TkgSnapshot,
};

// ProbabilisticProgramEngine — Bayesian reasoning and posterior sampling.
//
// Collision notes:
//   • `xorshift64` is already exported from `hypothesis_test_engine`; the copy
//     in this module is NOT re-exported at crate root (it is `pub(crate)` only
//     inside the module). All external references use `ppe_xorshift64` if
//     needed but we do not re-export it here to keep the API surface clean.
pub mod probabilistic_program_engine;
pub use probabilistic_program_engine::{
    PpeEngineConfig, PpePrior, PpeSampleResult, PpeSamplingMethod, PpeSamplingStats,
    ProbabilisticProgramEngine,
};

pub mod constraint_propagation_engine;
pub use constraint_propagation_engine::{
    ConstraintPropagationEngine, CpeConstraint, CpeDomain, CpeEngineConfig, CpePropagationResult,
    CpePropagationStats, CpeVariable,
};

// SymbolicExpressionSimplifier — multi-pass rewriting engine for symbolic math expressions.
pub mod symbolic_expression_simplifier;
pub use symbolic_expression_simplifier::{
    SesExpr, SesRewriteRule, SesSimplifierConfig, SesSimplifierStats, SymbolicExpressionSimplifier,
};

// DecisionTreeLearner — ID3/C4.5-style decision tree with training, prediction,
// feature importance, pruning, and rich statistics.
pub mod decision_tree_learner;
pub use decision_tree_learner::{
    DecisionTreeLearner, DtlCriterion, DtlLearnerConfig, DtlLearnerStats, DtlNode, DtlPrediction,
    DtlSample,
};

// AbductiveReasoningEngine — infers the best explanation for observed facts.
// All exported names use the `Abr` prefix to avoid collision with the `Are*`
// names already used by `adaptive_routing_engine`.
pub mod abductive_reasoning_engine;
pub use abductive_reasoning_engine::{
    abr_xorshift64, fnv1a_64 as abr_fnv1a_64, set_fingerprint as abr_set_fingerprint,
    AbductiveReasoningEngine, AbrAbductiveReasoningEngine, AbrCostFunction, AbrEngineConfig,
    AbrExplanation, AbrExplanationRecord, AbrHypothesis, AbrReasoningStats, AbrRule, AbrTerm,
};

// EnsembleLearner — Bagging, AdaBoost, Gradient Boosting, Random Forest, and Stacking.
//
// Collision notes:
//   • `ElEnsembleLearner` is a type alias for `EnsembleLearner` — no collision.
//   • All exported names carry the `El` prefix; no crate-root conflicts expected.
pub mod ensemble_learner;
pub use ensemble_learner::{
    ElBaseModel, ElEnsembleLearner, ElError, ElLearnerConfig, ElLearnerStats, ElMethod,
    ElPrediction, ElSample, ElTrainingRecord, EnsembleLearner,
};

/// Serialize CID as string
pub(crate) fn serialize_cid<S>(cid: &Cid, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_str(&cid.to_string())
}

/// Deserialize CID from string
pub(crate) fn deserialize_cid<'de, D>(deserializer: D) -> Result<Cid, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    s.parse().map_err(serde::de::Error::custom)
}
