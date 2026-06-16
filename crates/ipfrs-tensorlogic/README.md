# ipfrs-tensorlogic

**[Alpha]** | v0.2.1 | 3,402 public items | 1 stub (GPU backend) | 2026-06-16

TensorLogic integration layer for IPFRS.

## Overview

`ipfrs-tensorlogic` bridges IPFRS with the TensorLogic AI language:

- **Zero-Copy Binding**: Direct memory sharing via Apache Arrow
- **Distributed Reasoning**: Network-wide backward chaining
- **Gradient Storage**: Version-controlled learning history
- **Proof Provenance**: Merkle DAG for inference traces

## Key Features

### TensorLogic IR Codec
Serialize and deserialize `tensorlogic::ir::Term` to/from DAG-CBOR IPLD with CID-based content addressing:

- **IR Serialization**: Convert `tensorlogic::ir::Term` to/from IPLD DAG-CBOR
- **CID Addressing**: Content-addressed block storage with deduplication
- **Term Indexing**: Fast lookup via predicate and term index
- **Bidirectional Conversion**: Full round-trip serialization with validation

### Apache Arrow Zero-Copy Tensors
Direct memory access without serialization overhead:

- **ArrowTensor**: Columnar tensor with shape, dtype, and strides metadata
- **ArrowTensorStore**: Managed collection of Arrow tensors
- **IPC Serialization**: Arrow IPC file format reading and writing
- **Typed Accessors**: `as_slice_f32`, `as_slice_f64`, `as_slice_i32`, `as_slice_i64`, `as_bytes`
- **ZeroCopyAccessor trait**: Uniform interface for zero-copy data access

### Safetensors Support
PyTorch-compatible tensor storage with lazy loading:

- **Memory-mapped Loading**: Lazy mmap-based access for large models
- **Chunked Storage**: Automatic model splitting for models exceeding size thresholds
- **Metadata Extraction**: ModelSummary with parameter counts and dtype distribution
- **Arrow Conversion**: `load_as_arrow` for seamless Arrow interop

### Shared Memory
Cross-process mmap buffers for zero-copy IPC:

- **SharedTensorBuffer**: Read/write cross-process memory-mapped buffers
- **SharedTensorBufferReadOnly**: Safe read-only sharing
- **SharedMemoryPool**: Pool management with size limits and tracking
- **Safety Guards**: Checksum validation, magic number verification, version checking

### Abductive and Backward Chaining Reasoning
Full logic programming engine with advanced recursive query support:

- **GoalDecomposition**: Subgoal tracking with rule application and depth
- **CycleDetector**: O(1) cycle detection to prevent infinite loops
- **MemoizedInferenceEngine**: Cache-aware query execution
- **Tabling (SLG resolution)**: TabledInferenceEngine for recursive queries
- **FixpointEngine**: Fixpoint computation for recursive rules
- **StratificationAnalyzer**: Stratification analysis for negation handling

### Remote Knowledge Retrieval
Distributed knowledge base with peer querying:

- **RemoteKnowledgeProvider trait**: Pluggable remote knowledge sources
- **DistributedGoalResolver**: Route subgoals to network peers
- **DistributedProofAssembler**: Assemble proofs from distributed fragments
- **Multi-hop Fact Discovery**: FactDiscoveryRequest with configurable hop limits
- **Incremental Loading**: Streaming/pagination for large knowledge bases

> Note: Actual peer-to-peer communication requires `ipfrs-network` (not yet integrated).

### Proof Storage as IPLD
Content-addressed proof management:

- **ProofFragment**: Proof steps with conclusion, premises, and rule references
- **ProofFragmentStore**: CID-based fragment storage with predicate index
- **ProofAssembler**: Recursive proof tree reconstruction and verification
- **ProofCompressor**: Common subproof elimination and delta encoding

### Automatic Proof Explanation
Natural language interpretability for proofs:

- **ProofExplainer**: Generate natural language explanations from proof trees
- **ExplanationStyle**: Concise, Detailed, Pedagogical, and Formal modes
- **Predicate Naturalization**: Human-readable formatting for common predicates
- **FragmentProofExplainer**: Fragment-based explanation for IPLD proofs
- **ProofExplanationBuilder**: Fluent builder API with configurable depth limits

### Query Optimization
Cost-based planning and materialized views:

- **QueryPlan**: Cost-estimated query plans with join order selection
- **PredicateStats**: Cardinality estimation and selectivity tracking
- **MaterializedViewManager**: Precomputed result views with TTL-based refresh and utility-based eviction
- **QueryCache**: LRU-based query result caching with TTL expiration
- **RemoteFactCache**: Per-predicate remote fact caching with TTL

### Gradient Storage and Compression
Differentiable model storage:

- **SparseGradient**: Sparse gradient encoding with indices and values
- **Top-k / Threshold / Random Sparsification**: Flexible compression strategies
- **QuantizedGradient**: INT4/INT8/INT16 quantization with min/max scaling
- **GradientDelta**: CID-referenced delta format with checksum validation
- **Gradient Aggregation**: Unweighted, weighted, and momentum-based aggregation
- **Gradient Verification**: Outlier detection, finite-value checks, gradient clipping

### Model Version Control
Git-like versioning for neural models:

- **ModelCommit**: CID-based versioning with parent tracking and metadata
- **Branch Management**: Create, list, delete branches; detached HEAD support
- **Fast-forward Merge**: Automatic ancestor-based merge detection
- **ModelDiff**: Layer-wise diff with L2 norm difference and shape change detection

### Provenance Tracking
Merkle DAG lineage for datasets and training runs:

- **DatasetProvenance**: CID-referenced dataset with contributor attribution
- **TrainingProvenance**: Training run with hyperparameters and parent model CIDs
- **ProvenanceGraph**: Recursive lineage tracing with circular dependency detection
- **Attribution**: Name, role, organization, and license metadata (MIT, Apache, GPL, CC, etc.)

### Federated Learning Support
Privacy-preserving distributed training:

- **DP-SGD**: Differential privacy with Gaussian and Laplacian noise injection
- **PrivacyBudget**: Epsilon/delta tracking with budget exhaustion handling
- **SecureAggregation**: Participant management and cryptographic framework
- **ModelSyncProtocol**: Federated round coordination with client state tracking
- **ConvergenceDetector**: Configurable threshold-based convergence detection
- **DeviceCapabilities**: CPU, memory, GPU, and storage detection
- **AdaptiveBatchSizer**: Memory-aware batch size adaptation
- **DeviceProfiler**: Performance measurement and tier classification

### Computation Graphs
IPLD-native computation graph storage and execution:

- **ComputationGraph**: IPLD-serializable graph with CID support
- **TensorOp**: 30+ operations (MatMul, Add, Mul, Einsum, ReLU, GELU, Softmax, LayerNorm, BatchNorm, Dropout, Gather, Scatter, Slice, Pad, and more)
- **Graph Fusion**: MatMul+Add, Add+ReLU, BatchNorm+ReLU, LayerNorm+Dropout fusions
- **Shape Inference**: NumPy-compatible broadcasting with validation for all 30+ ops
- **Parallel Execution**: Rayon-based multi-threaded execution with batch scheduling
- **Streaming Execution**: Chunked pipeline with configurable backpressure
- **GraphOptimizer**: CSE, constant folding, dead node removal, multi-pass convergence

### Model Format Support
Broad model format interop:

- **PyTorch Checkpoints**: StateDict parsing, optimizer state, metadata extraction, Safetensors conversion
- **Model Quantization**: INT4/INT8/INT16 quantization, per-tensor and per-channel modes, symmetric and asymmetric, dynamic quantization, calibration (MinMax, Percentile, Entropy, MSE), INT4 bit packing

> Not yet implemented: ONNX format import/export.

### Visualization
Debugging and interpretability:

- **DOT Export**: Computation graph export for Graphviz
- **Proof Tree Visualization**: Textual and structured proof tree rendering
- **Color-coded Nodes**: Operation-type color coding in DOT output
- **Graph and Proof Statistics**: Node counts, depth, and summary metrics

### FFI Profiling and Allocation Optimization
Performance measurement and allocation management:

- **FfiProfiler**: Call latency measurement and hotspot identification
- **FfiCallStats**: Per-call overhead tracking with global profiler instance
- **BufferPool / TypedBufferPool**: Reusable byte and typed buffer pools
- **StackBuffer / AdaptiveBuffer**: Stack-allocated small buffers with heap fallback
- **ZeroCopyConverter**: Utilities for zero-copy type-casting conversions

### Language Bindings
Cross-language API surface:

- **Python (PyO3)**: Term, Predicate, Rule classes; NumPy/Arrow zero-copy interop; InferenceEngine with backward chaining
- **Node.js (NAPI-RS)**: TypeScript classes with async Promise-based inference; JSON knowledge base serialization
- **WebAssembly**: Browser-side synchronous inference; JSON knowledge base import/export

> GPU execution (CUDA/OpenCL) is not yet implemented.

## Architecture

```
TensorLogic Runtime
         ↓
Zero-Copy FFI (Apache Arrow)
         ↓
ipfrs-tensorlogic
├── ir/            # TensorLogic IR codec
├── inference/     # Distributed reasoning
├── gradient/      # Gradient storage & tracking
└── ffi/           # Foreign function interface
         ↓
ipfrs-core (Blocks & CID)
```

## Design Principles

- **Performance Critical**: No unnecessary copies or allocations
- **Type Safe**: Leverage Rust's type system
- **Composable**: Work with standard IPFRS blocks
- **Explainable**: Full provenance for XAI

## Usage Example

```rust
use ipfrs_tensorlogic::{TensorLogicNode, InferenceEngine};
use tensorlogic::ir::Term;

// Initialize node with TensorLogic support
let node = TensorLogicNode::new(config).await?;

// Store logic term
let term = Term::from_str("knows(alice, bob)")?;
let cid = node.put_term(term).await?;

// Distributed inference
let query = Term::from_str("knows(alice, ?X)")?;
let solutions = node.infer(query).await?;

// Access tensor data (zero-copy)
let weights = node.get_tensor(cid).await?;
let array: ArrayView2<f32> = weights.as_arrow_array()?;
```

## Integration Points

### With TensorLogic
- Share memory space via FFI
- Use TensorLogic types directly
- Support inference callbacks
- Integrate with learning loop

### With IPFRS Core
- Store terms as IPLD blocks
- Content-address by hash
- Version control via CID links
- Network distribution

## Dependencies

- `arrow` (arrow-rs) — columnar memory format and IPC serialization
- `safetensors` — PyTorch-compatible tensor storage
- `memmap2` — memory-mapped file access for shared buffers and lazy loading
- `lru` — LRU cache for query results and lazy evaluation
- `parking_lot` — fast reader-writer locks for thread-safe caches
- `rayon` — data-parallelism for parallel graph execution
- `bytemuck` — zero-copy type casting
- `uuid` — request ID generation for distributed protocols
- `async-trait` — async trait support for remote knowledge providers
- `ipfrs-core` — IPFRS primitives (CID, blocks, IPLD)

## References

- IPFRS v0.2.1 Whitepaper (TensorLogic Architecture)
- IPFRS v0.2.1 Whitepaper (Zero-Copy Tensor Transport)
- TensorLogic Paper: arXiv:2510.12269
