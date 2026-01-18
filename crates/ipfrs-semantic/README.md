# ipfrs-semantic

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

- **HNSW (In-Memory)**: Hierarchical Navigable Small World graphs
- **DiskANN (On-Disk)**: Scalable for billion-scale datasets
- **Quantization**: Reduce memory footprint (PQ, OPQ)
- **GPU Acceleration**: Optional CUDA/ROCm support

### Logic-Aware Routing
Integration with TensorLogic inference:

- **Predicate Resolution**: Find nodes with specific predicates
- **Proof Search**: Locate data needed for backward chaining
- **Fact Discovery**: Query distributed knowledge base
- **Rule Matching**: Find applicable inference rules

## Architecture

```
ipfrs-semantic
├── router/        # Semantic routing engine
├── index/         # Vector index implementations
│   ├── hnsw/      # In-memory HNSW
│   └── diskann/   # Disk-based ANN
├── embeddings/    # Embedding generation & management
└── logic/         # TensorLogic integration
```

## Design Principles

- **Embedding Agnostic**: Support multiple embedding models
- **Scalable**: Handle millions of vectors on edge devices
- **Fast**: Sub-millisecond query latency for cached queries
- **Interpretable**: Explain why results match

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

| Operation | Latency | Throughput |
|-----------|---------|------------|
| HNSW Query (1M vectors) | <1ms | 10k qps |
| DiskANN Query (100M vectors) | <10ms | 1k qps |
| Index Update | ~100μs | 10k ops/s |

## Dependencies

- `hnsw` - Hierarchical Navigable Small World
- `faiss` (optional) - Facebook AI Similarity Search
- `ndarray` - N-dimensional arrays
- `serde` - Serialization

## References

- IPFRS v0.2.0 Whitepaper (Reasoning-Ready)
- IPFRS v0.3.0 Whitepaper (Semantic Router)
- HNSW Paper: https://arxiv.org/abs/1603.09320
- DiskANN Paper: https://arxiv.org/abs/1909.06002
