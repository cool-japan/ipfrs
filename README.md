# IPFRS - Inter-Planet File RUST System

![IPFRS](ipfrs.jpg)

**Version:** 0.2.0 "Network Release"

**Status:** Production Ready — P2P Networking Available

A next-generation distributed file system built in Rust, combining content-addressed storage with semantic search and logic programming capabilities.

---

## 🚀 Quick Start

```bash
# Install (requires Rust 1.70+)
cargo install --path crates/ipfrs-cli

# Initialize repository
ipfrs init

# Add a file
ipfrs add myfile.txt
# Output: CID: bafybeig...

# Retrieve content
ipfrs cat bafybeig...

# Get statistics
ipfrs stats
```

---

## 🎯 What is IPFRS?

IPFRS revolutionizes distributed storage by adding **intelligence** to content-addressed systems. While traditional IPFS is a "static file warehouse," IPFRS transforms it into a "thinking highway."

**Key Innovations:**
- 🧠 **Semantic Search**: Find content by meaning, not just hash
- 🎓 **Logic Programming**: Content-addressed reasoning and inference
- ⚡ **Zero-Copy I/O**: Apache Arrow integration for performance
- 🦀 **Pure Rust**: Memory safety and ARM optimization

---

## 📦 Installation

### From Source (Recommended for 0.2.0)

```bash
git clone https://github.com/cool-japan/ipfrs.git
cd ipfrs
cargo build --release
cargo install --path crates/ipfrs-cli
```

### Requirements
- Rust 1.70 or later
- ~100MB disk space
- Linux, macOS, or Windows

---

## 📖 Usage

### Command-Line Interface

#### Basic File Operations

```bash
# Initialize a repository
ipfrs init
# Creates .ipfrs/ directory

# Add files
ipfrs add document.pdf
ipfrs add image.png

# Output: CID: bafybeig...

# Retrieve by CID
ipfrs get bafybeig... --output recovered.pdf

# View content
ipfrs cat bafybeig... | less

# List all blocks
ipfrs list

# Show statistics
ipfrs stats
# Output:
# Number of blocks: 42
# Total size: 52.43 MB
# Average block size: 1.24 MB
```

#### HTTP Gateway

```bash
# Start HTTP gateway
ipfrs gateway --listen 127.0.0.1:8080

# Access via HTTP
curl http://localhost:8080/ipfs/bafybeig...

# Use REST API
curl -X POST http://localhost:8080/api/v0/add \
  -F file=@myfile.txt
```

### Rust API

```rust
use ipfrs::{Node, NodeConfig};

#[tokio::main]
async fn main() -> ipfrs::Result<()> {
    // Create and start node
    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    // Add content
    let content = b"Hello, IPFRS!";
    let cid = node.add_bytes(content).await?;
    println!("Added: {}", cid);

    // Retrieve content
    if let Some(block) = node.get(&cid).await? {
        println!("Retrieved: {:?}", block.data());
    }

    // Semantic search
    if node.is_semantic_enabled() {
        let embedding = vec![0.1, 0.2, 0.3]; // Your embedding
        node.index_content(&cid, &embedding).await?;

        let results = node.search_similar(&embedding, 10).await?;
        for result in results {
            println!("Found: {} (score: {})", result.cid, result.score);
        }
    }

    Ok(())
}
```

---

## 🌐 HTTP API

IPFRS provides a comprehensive REST API compatible with Kubo (go-ipfs) and extended with semantic/logic features.

### Block Operations

```bash
# Add file
curl -X POST -F file=@document.pdf \
  http://localhost:8080/api/v0/add

# Get block
curl http://localhost:8080/ipfs/bafybeig...

# Block statistics
curl -X POST -d '{"arg":"bafybeig..."}' \
  http://localhost:8080/api/v0/block/stat
```

### DAG Operations

```bash
# Store DAG node
curl -X POST --data-binary @dag.cbor \
  http://localhost:8080/api/v0/dag/put

# Resolve IPLD path
curl -X POST -d '{"arg":"/ipfs/Qm.../path/to/data"}' \
  http://localhost:8080/api/v0/dag/resolve
```

### Semantic Search (NEW!)

```bash
# Index content with embedding
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "cid": "bafybeig...",
    "embedding": [0.1, 0.2, ..., 0.768]
  }' \
  http://localhost:8080/api/v0/semantic/index

# Search similar content
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "query": [0.15, 0.25, ...],
    "k": 10,
    "filter": {"min_score": 0.8}
  }' \
  http://localhost:8080/api/v0/semantic/search

# Get statistics
curl http://localhost:8080/api/v0/semantic/stats
```

### Logic Programming (NEW!)

```bash
# Store logical term
curl -X POST -H "Content-Type: application/json" \
  -d '{"term": {"Variable": "X"}}' \
  http://localhost:8080/api/v0/logic/term

# Store inference rule
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "rule": {
      "head": {"name": "ancestor", "args": [...]},
      "body": [...]
    }
  }' \
  http://localhost:8080/api/v0/logic/rule

# Retrieve term
curl http://localhost:8080/api/v0/logic/term/bafybeig...
```

**Complete API Reference:** [See HTTP API docs](#http-api-reference)

---

## 🏗️ Architecture

IPFRS follows a bi-layer architecture combining intelligence with infrastructure:

### Logical Layer (The Brain)
- **Semantic Router**: HNSW vector search with LRU query caching
- **TensorLogic Store**: Content-addressed logic programming

### Physical Layer (The Body)
- **Block Storage**: Sled embedded database with content addressing
- **Zero-Copy I/O**: Apache Arrow integration (planned)
- **Network Stack**: libp2p with QUIC transport — DHT, Bitswap, TensorSwap

```
┌─────────────────────────────────────┐
│      Application Layer              │
│   (Your Code / HTTP Clients)        │
└──────────────┬──────────────────────┘
               │
┌──────────────┴──────────────────────┐
│         Node API (Rust)             │
│  ┌──────────┐  ┌────────────────┐   │
│  │ Semantic │  │  TensorLogic   │   │
│  │  Router  │  │     Store      │   │
│  │  (HNSW)  │  │   (Logic IR)   │   │
│  └──────────┘  └────────────────┘   │
└──────────────┬──────────────────────┘
               │
┌──────────────┴──────────────────────┐
│       Block Storage (Sled)          │
│   Content-Addressed Blocks (CID)   │
└─────────────────────────────────────┘
```

---

## 📚 Project Structure

```
ipfrs/
├── Cargo.toml                 # Workspace manifest
├── crates/
│   ├── ipfrs-core/            # Core types (Block, CID, Error, IPLD)
│   ├── ipfrs-storage/         # Block storage (Sled), caching
│   ├── ipfrs-semantic/        # Semantic router, HNSW
│   ├── ipfrs-tensorlogic/     # TensorLogic store, logic IR
│   ├── ipfrs-interface/       # HTTP gateway, zero-copy interface
│   ├── ipfrs-network/         # libp2p networking (0.2.0)
│   ├── ipfrs-transport/       # TensorSwap, Bitswap (0.2.0)
│   ├── ipfrs/                 # Main library (unified API)
│   ├── ipfrs-cli/             # Command-line interface
│   ├── ipfrs-wasm/            # WebAssembly bindings
│   ├── ipfrs-nodejs/          # Node.js bindings
│   └── ipfrs-python/          # Python bindings
└── README.md
```

---

## ✨ Key Features

### 1. Content-Addressed Storage ✅ (Implemented)
- Immutable blocks identified by CID (Content Identifier)
- Sled embedded database for persistence
- DAG operations with IPLD support
- Directory tree handling

### 2. Semantic Search ✅ (Implemented)
- HNSW (Hierarchical Navigable Small World) index
- k-NN similarity search with configurable distance metrics
- Query result caching (LRU)
- Hybrid filtered search (by score, prefix, etc.)

### 3. Logic Programming ✅ (Implemented)
- Content-addressed terms, predicates, and rules
- JSON serialization for portability
- Foundation for distributed reasoning
- Compatible with TensorLogic IR

### 4. Comprehensive Observability ✅ (Implemented)
- Storage statistics (block count, total size)
- Semantic index stats (vectors, dimension, cache)
- TensorLogic statistics
- HTTP API monitoring endpoints

### 5. HTTP Gateway ✅ (Implemented)
- 20 REST API endpoints
- Kubo (go-ipfs) compatibility
- HTTP 206 range request support
- JSON responses throughout

### 6. P2P Networking ✅ (Implemented)
- libp2p integration with QUIC transport
- DHT bootstrap and peer discovery
- Bitswap block exchange protocol
- TensorSwap distributed inference protocol

### 7. WASM/Node.js Bindings ✅ (Implemented)
- WebAssembly bindings for browser environments
- Node.js native bindings via ipfrs-nodejs
- Python bindings via ipfrs-python

---

## 🎓 Examples

### Example 1: Basic File Storage

```rust
use ipfrs::{Node, NodeConfig};

#[tokio::main]
async fn main() -> ipfrs::Result<()> {
    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    // Add a file
    let cid = node.add_file("./document.pdf").await?;
    println!("Stored as: {}", cid);

    // Retrieve it
    node.get_to_file(&cid, "./recovered.pdf").await?;
    println!("Retrieved successfully!");

    Ok(())
}
```

### Example 2: Semantic Document Search

```rust
use ipfrs::{Node, NodeConfig};
use ipfrs_semantic::RouterConfig;

#[tokio::main]
async fn main() -> ipfrs::Result<()> {
    let mut config = NodeConfig::default();
    config.semantic_config = Some(RouterConfig {
        dimension: 768,
        max_elements: 100_000,
        ..Default::default()
    });

    let mut node = Node::new(config)?;
    node.start().await?;

    // Add documents with embeddings
    let doc1_cid = node.add_bytes(b"AI research paper").await?;
    let doc1_embedding = get_embedding("AI research paper"); // Your embedding function
    node.index_content(&doc1_cid, &doc1_embedding).await?;

    // Search for similar documents
    let query_embedding = get_embedding("machine learning");
    let results = node.search_similar(&query_embedding, 5).await?;

    for result in results {
        println!("Found: {} (similarity: {:.2})", result.cid, result.score);
    }

    Ok(())
}

fn get_embedding(text: &str) -> Vec<f32> {
    // Use your favorite embedding model (BERT, Sentence Transformers, etc.)
    vec![0.1; 768] // Placeholder
}
```

### Example 3: Logic Programming

```rust
use ipfrs::{Node, NodeConfig};
use ipfrs_tensorlogic::{Term, Predicate, Rule};

#[tokio::main]
async fn main() -> ipfrs::Result<()> {
    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    // Store a term
    let term = Term::Variable("X".to_string());
    let term_cid = node.put_term(&term).await?;
    println!("Term stored: {}", term_cid);

    // Store a predicate: parent(alice, bob)
    let predicate = Predicate {
        name: "parent".to_string(),
        args: vec![
            Term::Constant("alice".to_string()),
            Term::Constant("bob".to_string()),
        ],
    };
    let pred_cid = node.store_predicate(&predicate).await?;
    println!("Predicate stored: {}", pred_cid);

    // Retrieve it
    if let Some(retrieved) = node.get_predicate(&pred_cid).await? {
        println!("Retrieved: {:?}", retrieved);
    }

    Ok(())
}
```

### Example 4: DAG Operations

```rust
use ipfrs::{Node, NodeConfig};
use ipfrs_core::Ipld;
use std::collections::BTreeMap;

#[tokio::main]
async fn main() -> ipfrs::Result<()> {
    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    // Create a DAG structure
    let mut metadata = BTreeMap::new();
    metadata.insert("title".to_string(), Ipld::String("My Document".to_string()));
    metadata.insert("author".to_string(), Ipld::String("Alice".to_string()));

    let dag_node = Ipld::Map(metadata);
    let cid = node.dag_put(dag_node).await?;
    println!("DAG node stored: {}", cid);

    // Retrieve DAG node
    if let Some(node_data) = node.dag_get(&cid).await? {
        println!("Retrieved: {:?}", node_data);
    }

    // Resolve path
    if let Some(resolved_cid) = node.dag_resolve(&cid, "/title").await? {
        println!("Resolved path to: {}", resolved_cid);
    }

    Ok(())
}
```

---

## 🧪 Testing

```bash
# Run all tests (recommended)
cargo nextest run --workspace --all-features

# Run all tests (standard)
cargo test

# Run with logging
RUST_LOG=debug cargo test

# Test specific crate
cargo test -p ipfrs-core

# Integration tests
cargo test --test integration
```

---

## 📊 Performance

### Benchmarks (0.2.0)

| Operation | Time | Throughput |
|-----------|------|------------|
| Block put | ~50µs | 20,000 ops/sec |
| Block get | ~30µs | 33,000 ops/sec |
| DAG put | ~80µs | 12,500 ops/sec |
| Semantic search (k=10) | ~1ms | 1,000 queries/sec |
| HNSW insertion | ~100µs | 10,000 inserts/sec |

*Tested on: AMD Ryzen 9 5900X, NVMe SSD*

### Scalability

- **Storage**: Limited only by disk space
- **HNSW Index**: Scales to millions of vectors
- **Concurrent Operations**: Async I/O with Tokio
- **Memory**: ~50MB base + index data

---

## 🗺️ Roadmap

### ✅ Version 0.1.0 "Foundation" (Released)
- Content-addressed storage with DAG support
- Semantic search (HNSW)
- Logic programming (TensorLogic)
- HTTP API (20 endpoints)
- CLI (13 commands)
- Comprehensive observability

### ✅ Version 0.2.0 "Network Release" (Released 2026-06-15)
- libp2p networking with QUIC transport
- DHT bootstrap and peer discovery
- Bitswap block exchange protocol
- TensorSwap distributed inference protocol
- Traffic shaping and adaptive bandwidth management
- WASM/Node.js bindings (ipfrs-wasm, ipfrs-nodejs, ipfrs-python)
- Temporal pattern matching in TensorLogic
- Abductive reasoning engine
- Probabilistic Program Engine (PPE) with sampling support
- Reinforcement Learning Agent (RLA) types with multiple policies
- OxiARC compression migration (Pure Rust)

### 🚧 Version 0.3.0 "Intelligence" (In Progress)
- Persistent HNSW index
- Advanced distributed reasoning
- Enhanced query features
- Production hardening

### 📅 Version 0.4.0 "Ecosystem"
- GraphQL API
- Enhanced tooling
- Monitoring & metrics
- Extended language binding support

### 📅 Version 1.0.0 "Stable"
- API stability guarantees
- Comprehensive documentation
- Production deployments
- Security audit complete

---

## 🤝 Contributing

IPFRS is part of the COOLJAPAN ecosystem. Contributions are welcome!

### Development Setup

```bash
git clone https://github.com/cool-japan/ipfrs.git
cd ipfrs
cargo build
cargo test
```

### Guidelines

- Follow Rust style guidelines (rustfmt)
- Maintain zero warnings policy
- Add tests for new features
- Update documentation

---

## Sponsorship

IPFRS is developed and maintained by **COOLJAPAN OU (Team Kitasan)**.

If you find IPFRS useful, please consider sponsoring the project to support continued development of the Pure Rust ecosystem.

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red?logo=github)](https://github.com/sponsors/cool-japan)

**[https://github.com/sponsors/cool-japan](https://github.com/sponsors/cool-japan)**

Your sponsorship helps us:
- Maintain and improve the COOLJAPAN ecosystem
- Keep the entire ecosystem (OxiBLAS, OxiFFT, SciRS2, etc.) 100% Pure Rust
- Provide long-term support and security updates

## 📄 License

Apache-2.0

---

## 🙏 Acknowledgments

- **IPFS** - Content-addressed foundation
- **libp2p** - Networking stack
- **Sled** - Embedded database
- **HNSW** - Vector search algorithm
- **TensorLogic** - Reasoning framework

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/cool-japan/ipfrs/issues)
- **Discussions**: [GitHub Discussions](https://github.com/cool-japan/ipfrs/discussions)
- **Documentation**: [docs.rs/ipfrs](https://docs.rs/ipfrs)

---

## 🔖 HTTP API Reference

### Block Operations
- `POST /api/v0/add` - Upload file
- `POST /api/v0/block/get` - Get raw block
- `POST /api/v0/block/put` - Store raw block
- `POST /api/v0/block/stat` - Block statistics
- `POST /api/v0/cat` - Output content
- `GET /ipfs/{cid}` - Retrieve content (HTTP 206 support)

### DAG Operations
- `POST /api/v0/dag/put` - Store DAG node
- `POST /api/v0/dag/get` - Retrieve DAG node
- `POST /api/v0/dag/resolve` - Resolve IPLD path

### Semantic Search
- `POST /api/v0/semantic/index` - Index content
- `POST /api/v0/semantic/search` - Search similar
- `GET /api/v0/semantic/stats` - Index statistics

### Logic Programming
- `POST /api/v0/logic/term` - Store term
- `GET /api/v0/logic/term/{cid}` - Retrieve term
- `POST /api/v0/logic/predicate` - Store predicate
- `POST /api/v0/logic/rule` - Store rule
- `GET /api/v0/logic/stats` - Logic statistics

### Utility
- `GET /health` - Health check
- `POST /api/v0/version` - Version information

---

## 💭 Philosophy

> "IPFRS is not just a storage reinvention. It is an attempt to unify human knowledge (data) and machine intelligence (reasoning) under the same physical law (Protocol)."

By fusing Rust's robust implementation (The Body) with TensorLogic's flexible reasoning (The Brain), IPFRS becomes the core of an autonomous distributed knowledge mesh.

---

**Status**: v0.2.0 - Production Ready (P2P Networking Available)

🎉 **v0.2.0 released 2026-06-15 — P2P networking, distributed inference, and ML agent types now available.**
