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
//! let slice = tensor.as_slice_f32().unwrap();
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
//! let sparse = GradientCompressor::top_k(&gradient, vec![5], 2).unwrap();
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
//! let caps = DeviceCapabilities::detect().unwrap();
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
//! let stats = profiler.get_stats("my_ffi_function").unwrap();
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
//! let bytes = writer.serialize().unwrap();
//! let reader = SafetensorsReader::from_bytes(Bytes::from(bytes)).unwrap();
//!
//! // Load as Arrow tensors for zero-copy access
//! let weights = reader.load_as_arrow("layer1.weights").unwrap();
//! let bias = reader.load_as_arrow("layer1.bias").unwrap();
//! let indices = reader.load_as_arrow("vocab_indices").unwrap();
//! let ids = reader.load_as_arrow("entity_ids").unwrap();
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
//! let stats = profiler.get_stats("tensor_allocation").unwrap();
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
//! let quantized = QuantizedTensor::quantize_per_tensor(&weights, vec![4], config).unwrap();
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

pub mod allocation_optimizer;
pub mod arrow;
pub mod cache;
pub mod computation_graph;
pub mod datalog;
pub mod device;
pub mod ffi_profiler;
pub mod gpu;
pub mod gradient;
pub mod ir;
pub mod memory_profiler;
pub mod optimizer;
pub mod proof_explanation;
pub mod proof_storage;
pub mod provenance;
pub mod pytorch_checkpoint;
pub mod quantization;
pub mod reasoning;
pub mod recursive_reasoning;
pub mod remote_reasoning;
pub mod safetensors_support;
pub mod shared_memory;
pub mod storage;
pub mod utils;
pub mod version_control;
pub mod visualization;

// Allocation optimization
pub use allocation_optimizer::{
    AdaptiveBuffer, AllocationError, BufferPool, PooledBuffer, StackBuffer, TypedBufferPool,
    TypedPooledBuffer, ZeroCopyConverter,
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

// GPU execution (stub for future integration)
pub use gpu::{
    GpuBackend, GpuBuffer, GpuDevice, GpuError, GpuExecutor, GpuKernel, GpuMemoryManager,
};

// Gradient storage
pub use gradient::{
    ClientInfo, ClientState, ConvergenceDetector, DPMechanism, DifferentialPrivacy, FederatedRound,
    GradientAggregator, GradientCompressor, GradientDelta, GradientError, GradientVerifier,
    LayerGradient, ModelSyncProtocol, PrivacyBudget, QuantizedGradient, SecureAggregation,
    SparseGradient,
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
    CycleDetector, DistributedReasoner, GoalDecomposition, InferenceEngine,
    MemoizedInferenceEngine, Proof, ProofRule, Substitution,
};

// Recursive reasoning
pub use recursive_reasoning::{
    FixpointEngine, StratificationAnalyzer, StratificationResult, TableStats, TabledInferenceEngine,
};

// Remote reasoning
pub use remote_reasoning::{
    DistributedGoalResolver, DistributedProofAssembler, FactDiscoveryRequest,
    FactDiscoveryResponse, GoalResolutionRequest, GoalResolutionResponse, IncrementalLoadRequest,
    IncrementalLoadResponse, MockRemoteKnowledgeProvider, QueryRequest, QueryResponse,
    RemoteKnowledgeProvider, RemoteReasoningError,
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

// Storage
pub use storage::TensorLogicStore;

// Utilities
pub use utils::{KnowledgeBaseUtils, PredicateBuilder, QueryUtils, RuleBuilder, TermUtils};

// Version control
pub use version_control::{
    Branch, LayerDiff, ModelCommit, ModelDiff, ModelDiffer, ModelRepository, VersionControlError,
};

// Visualization
pub use visualization::{GraphVisualizer, ProofVisualizer};

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
