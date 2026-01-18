# ipfrs-core

Core primitives and types for IPFRS (Inter-Planet File RUST System).

## Overview

`ipfrs-core` provides the fundamental building blocks for IPFRS, including:

- **CID (Content Identifier)**: Multihash-based content addressing
- **Block**: Raw data blocks with CID verification
- **IPLD Codec**: Serialization/deserialization for TensorLogic types
- **Error Types**: Unified error handling across all IPFRS crates

## Key Components

### CID & Multihash
- Content-addressable identifiers compatible with IPFS
- Support for multiple hash algorithms (SHA256, BLAKE3, etc.)
- Integration with `tensorlogic::ir::Term` for logic-aware addressing

### Block Management
- Immutable data blocks with cryptographic verification
- Zero-copy access patterns for performance
- Memory-safe handling with Rust ownership

### IPLD Integration
- Merkle DAG structures for content-addressable graphs
- Custom codec for TensorLogic IR serialization
- Support for Safetensors and Apache Arrow formats

## Architecture

This crate sits at the foundation of the IPFRS stack:

```
ipfrs-interface (API)
    ↓
ipfrs-core (Types & Primitives)
    ↓
ipfrs-storage | ipfrs-network | ipfrs-semantic
```

## Design Principles

- **Zero-Copy**: Minimize memory allocations and data copying
- **Type Safety**: Leverage Rust's type system for correctness
- **Protocol Compatibility**: Wire-compatible with IPFS where applicable
- **TensorLogic Native**: First-class support for neural-symbolic AI types

## Dependencies

- `multihash` - Content addressing
- `cid` - CID implementation
- `serde` - Serialization framework
- `bytes` - Efficient byte buffer management

## References

- IPFRS v0.3.0 Whitepaper (Unified Strategy)
- IPLD Specification: https://ipld.io/
- Multiformats: https://multiformats.io/
