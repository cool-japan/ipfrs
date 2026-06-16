# ipfrs-semantic

**[Stable]** | v0.2.1 | 299 tests passing | 2026-06-16

Semantic routing and vector search for IPFRS.

## Overview

`ipfrs-semantic` extends IPFRS with intelligence-aware data discovery:

- **Semantic Router**: Embedding-based content discovery
- **Vector Search**: HNSW/DiskANN for approximate nearest neighbors
- **Logic Solver**: Backward chaining query resolution
- **Analogical Retrieval**: Find conceptually similar content

## Key Features

### Dual Resolution System
Combine exact (CID) and approximate (embedding) search:

- **Exact Match**: Traditional content-addressed retrieval
- **Semantic Match**: Find similar concepts via embeddings
- **Hybrid Queries**: Blend exact and approximate results
- **Relevance Ranking**: Score results by multiple criteria

### Vector Index
High-performance ANN (Approximate Nearest Neighbor) search:

- **HNSW (In-Memory)**: Hierarchical Navigable Small World graphs with insert/delete/search
- **DiskANN (On-Disk)**: Disk-based index supporting 100M+ vectors via memory-mapped files
- **Product Quantization (PQ)**: 8–32x compression with codebook-based quantization
- **Optimized Product Quantization (OPQ)**: Rotation matrix learning for improved recall
- **Scalar Quantization**: int8/uint8 quantization with 4x compression and <5% accuracy loss
- **Learned Index Structures (RMI)**: Recursive Model Index with Linear, Polynomial, and NeuralNetwork models

### Hybrid Search
Rich filtering layered over vector search:

- **Metadata Filtering**: Boolean filters combined with vector search, pre/post-filter strategies
- **Temporal Filtering**: Time-range queries with recency boosting and time-decay scoring
- **Faceted Search**: Multi-attribute drill-down navigation with facet counting
- **Adaptive Strategy Selection**: Automatic pre/post filter selection to minimise latency

### SIMD Acceleration
Platform-optimised distance computation:

- **ARM NEON**: Vectorised dot products for aarch64 (2–4x speedup target)
- **x86 SSE/AVX/AVX2**: Equivalent acceleration on x86 hardware
- **Runtime Feature Detection**: Selects the best SIMD path at startup
- **Cache-Aligned Storage**: 64-byte aligned `AlignedVector` for SIMD-friendly access

### Semantic DHT Routing
Distributed peer selection via embeddings:

- **Embedding-Based Routing**: Route queries to nearest peers in embedding space
- **Proximity-Aware Peer Selection**: `SemanticRoutingTable` with greedy routing and load balancing
- **Clustering**: k-means cluster-aware peer selection with adaptive load metrics
- **Replication Strategies**: NearestPeers, SameCluster, CrossCluster fault-tolerance modes

### Federated Queries
Multi-index search with privacy:

- **Multi-Index Search**: Concurrent queries across heterogeneous indices with timeout handling
- **Aggregation Strategies**: Simple, RankFusion, ScoreNormalization, BordaCount
- **Privacy-Preserving Search**: Differential privacy noise injection per federated query
- **Extensible via Trait**: `QueryableIndex` trait for custom index backends

### Logic-Aware Routing
Integration with TensorLogic inference:

- **Predicate Embeddings**: Map logic predicates to vectors with compositional generation
- **Backward Chaining**: Goal-driven subgoal decomposition with memoization
- **SPARQL-Like Queries**: Triple pattern matching with AND/OR/NOT operators
- **Provenance Tracking**: Immutable audit trails with generation timestamps

### Multi-Modal Embeddings
Unified embedding space across modalities:

- **Supported Modalities**: Text, Image, Audio, Video, Code
- **Cross-Modal Search**: Query across modalities with modality-specific distance metrics
- **Embedding Projection**: Alignment of heterogeneous embedding spaces

### Privacy and Security
Differential privacy primitives:

- **Laplacian Noise**: Epsilon-DP guarantee for embedding release
- **Gaussian Noise**: Epsilon-delta-DP with configurable sensitivity
- **Budget Tracking**: Per-query epsilon-delta budget management
- **Utility-Privacy Trade-Off Analysis**: Quantitative utility scoring

### Dynamic Updates
Evolving index management:

- **Online Fine-Tuning**: Momentum-based incremental embedding updates
- **Multi-Version Management**: Concurrent version coexistence with migration support
- **Embedding Transformation**: Dimension projection and version migration pipelines

### Production Operations
Operational tooling for deployed systems:

- **Auto-Scaling Advisor**: Workload analysis with horizontal/vertical scaling recommendations
- **Query Analytics**: P50/P90/P99 latency tracking, QPS calculation, query pattern detection
- **Vector Quality Analysis**: Validity checks, anomaly detection, diversity scoring, outlier detection
- **Index Health Diagnostics**: Health scoring (Healthy/Warning/Degraded/Critical), search profiler
- **Automatic Parameter Optimisation**: Goal-based parameter recommendation and adaptive ef_search

### Language Bindings
- **Python (PyO3)**: `SemanticIndex` class, numpy integration, asyncio support
- **Node.js (NAPI-RS)**: TypeScript types, Buffer-based input, Promise API
- **WebAssembly**: Browser-compatible HNSW, Float32Array embeddings, IndexedDB storage

## Architecture

```
ipfrs-semantic
├── router/        # Semantic routing engine
├── index/         # Vector index implementations
│   ├── hnsw/      # In-memory HNSW
│   └── diskann/   # Disk-based ANN
├── embeddings/    # Embedding generation & management
├── logic/         # TensorLogic integration
├── dht/           # Semantic DHT routing
├── federated/     # Federated query execution
├── privacy/       # Differential privacy
├── multimodal/    # Multi-modal embedding support
└── analytics/     # Query analytics and observability
```

## Design Principles

- **Embedding Agnostic**: Support multiple embedding models
- **Scalable**: Handle millions of vectors on edge devices
- **Fast**: Sub-millisecond query latency for cached queries
- **Interpretable**: Explain why results match
- **Privacy-First**: Differential privacy integrated at the query layer
- **Observable**: Built-in diagnostics, profiling, and analytics

## Usage Example

```rust
use ipfrs_semantic::{SemanticRouter, EmbeddingModel};
use ipfrs_core::Cid;

// Initialize router
let router = SemanticRouter::new(config).await?;

// Index content with embeddings
let embedding = model.encode("neural networks")?;
router.index(cid, embedding).await?;

// Semantic search
let query_emb = model.encode("deep learning")?;
let results = router.search(query_emb, k=10).await?;

// Hybrid search (CID + semantic)
let results = router.hybrid_search(
    cid_filter: Some(prefix),
    embedding: query_emb,
    k: 10
).await?;
```

## Performance Characteristics

| Operation | Target Latency | Target Throughput |
|-----------|---------------|-------------------|
| HNSW Query (1M vectors, cached) | < 1ms | 10k qps |
| HNSW Query (1M vectors, uncached) | < 5ms | — |
| DiskANN Query (100M vectors) | < 10ms | 1k qps |
| Index Update | ~100 µs | 10k ops/s |
| Index Build (1M vectors) | < 10 min | — |
| Memory (1M × 768-dim vectors) | < 2 GB | — |
| Recall@10 (k-NN) | > 95% | — |

## Dependencies

- `hnsw_rs` - Hierarchical Navigable Small World
- `nalgebra` - Linear algebra (matrix/vector operations)
- `memmap2` - Memory-mapped file access for DiskANN
- `lru` - LRU cache for embeddings and query results
- `dashmap` - Concurrent hash maps
- `rayon` - Data parallelism for batch operations
- `tokio` - Async runtime
- `serde` - Serialization

## References

- IPFRS v0.2.1 Whitepaper (Reasoning-Ready)
- IPFRS v0.3.0 Whitepaper (Semantic Router)
- HNSW Paper: https://arxiv.org/abs/1603.09320
- DiskANN Paper: https://arxiv.org/abs/1909.06002
