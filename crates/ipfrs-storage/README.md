# ipfrs-storage

Blockstore implementation for IPFRS - the persistent storage layer.

## Overview

`ipfrs-storage` provides high-performance, Rust-native key-value storage for IPFRS blocks with:

- **Pluggable Backends**: Sled (default), ParityDB, RocksDB
- **Hot/Cold Tiering**: Automatic data migration based on access patterns
- **Differentiable Storage**: Version control for tensor gradients (Git-for-Tensors)
- **Zero-Copy Reads**: Memory-mapped access for large blocks

## Key Features

### Multi-Backend Support
- **Sled**: Embedded, pure Rust, optimized for SSDs
- **ParityDB**: High-performance, designed for blockchain workloads
- **RocksDB**: Battle-tested, C++ backend with Rust bindings

### Differentiable Blockstore (v0.2.0)
- **Version Control System** for model states (Git for Tensors)
- Track gradient updates as IPLD Merkle DAG
- Time-travel to any historical model state
- Commit/checkout operations for reproducible training
- Branch management for collaborative training
- Provenance tracking for XAI (explainable AI)

### Performance Optimization
- Memory-mapped I/O for large tensor blocks
- Bloom filters for fast negative lookups
- LRU caching layer above persistent storage
- Batch write operations for high throughput

## Architecture

```
ipfrs-storage
├── traits/       # BlockStore trait definition
├── backends/     # Backend implementations (Sled, ParityDB)
├── cache/        # In-memory LRU cache layer
└── versioning/   # Gradient tracking & version control
```

## Design Principles

- **Backend Agnostic**: Easy to swap storage engines
- **Performance First**: Optimized for both read and write throughput
- **Memory Efficient**: Suitable for edge devices (2GB RAM)
- **Crash Safe**: ACID guarantees for critical operations

## Usage Example

```rust
use ipfrs_storage::{BlockStore, SledBackend};
use ipfrs_core::{Block, Cid};

// Initialize storage
let store = SledBackend::open("/path/to/datastore")?;

// Put block
let block = Block::new(data);
store.put(&block).await?;

// Get block by CID
let retrieved = store.get(&cid).await?;

// Check existence (fast bloom filter)
if store.has(&cid).await? {
    // ...
}
```

## Dependencies

- `sled` - Default embedded database
- `parity-db` - Optional high-performance backend
- `lru` - LRU cache implementation
- `tokio` - Async runtime

## References

- IPFRS v0.2.0 Whitepaper (Storage Architecture)
- *(planned v0.3.0)*
