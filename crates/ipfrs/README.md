# ipfrs

Main library crate for IPFRS (Inter-Planet File RUST System).

## Overview

`ipfrs` is the unified entry point for the IPFRS ecosystem, providing:

- **Complete Node**: Full IPFRS node implementation
- **Embedded Usage**: Library for embedding in applications
- **Plugin Architecture**: Extensible component system
- **High-Level API**: Simplified interface to all features

## Key Features

### Unified Node
Single crate that brings together all IPFRS components:

- Storage (ipfrs-storage)
- Networking (ipfrs-network)
- Transport (ipfrs-transport)
- Semantic routing (ipfrs-semantic)
- TensorLogic integration (ipfrs-tensorlogic)
- API interfaces (ipfrs-interface)

### High-Level API
Simple, ergonomic interface:

```rust
use ipfrs::Node;

// Start a node
let node = Node::new(config).await?;

// Add content
let cid = node.add_file("path/to/file").await?;

// Retrieve content
let data = node.get(cid).await?;

// Semantic search
let results = node.search_similar("neural networks", 10).await?;

// TensorLogic inference
let solutions = node.infer("knows(alice, ?X)").await?;
```

### Embedded Mode
Use IPFRS as a library in your application:

- No separate daemon process
- Direct API access
- Custom configuration
- Resource control

### Plugin System
Extensible architecture:

- Custom storage backends
- Additional protocols
- Custom content types
- Hook system for events

## Architecture

```
ipfrs (Main Library)
├── Node           # Unified node orchestrator
├── Builder        # Configuration builder
├── Events         # Event system
└── Plugins        # Plugin registry
    ↓
All ipfrs-* crates
```

## Design Principles

- **Batteries Included**: Everything needed for full functionality
- **Modular**: Use only what you need
- **Zero-Config**: Sensible defaults, easy to customize
- **Production Ready**: Robust error handling, observability

## Usage Example

### Basic Usage
```rust
use ipfrs::{Node, Config};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize with defaults
    let node = Node::builder()
        .with_storage("sled")
        .with_network_mode("public")
        .build()
        .await?;

    // Add a file
    let cid = node.add_bytes(b"Hello, IPFRS!").await?;
    println!("Added with CID: {}", cid);

    // Retrieve the file
    let data = node.get_bytes(&cid).await?;
    println!("Retrieved: {}", String::from_utf8(data)?);

    Ok(())
}
```

### Advanced Configuration
```rust
use ipfrs::{Node, Config, StorageBackend, NetworkMode};

let config = Config::builder()
    .storage(StorageBackend::ParityDb)
    .network_mode(NetworkMode::Public)
    .cache_size_mb(2048)
    .max_connections(1000)
    .enable_tensorlogic()
    .enable_semantic_search()
    .build()?;

let node = Node::new(config).await?;
```

### Event Handling
```rust
use ipfrs::{Node, Event};

let mut node = Node::new(config).await?;

// Subscribe to events
let mut events = node.subscribe();

tokio::spawn(async move {
    while let Some(event) = events.recv().await {
        match event {
            Event::BlockAdded(cid) => println!("Added: {}", cid),
            Event::PeerConnected(peer) => println!("Peer: {}", peer),
            Event::InferenceComplete(result) => println!("Result: {:?}", result),
            _ => {}
        }
    }
});
```

### Custom Plugins
```rust
use ipfrs::{Node, Plugin};

struct MyPlugin;

impl Plugin for MyPlugin {
    fn on_block_add(&self, cid: &Cid, block: &Block) {
        // Custom logic on block addition
    }
}

let node = Node::builder()
    .add_plugin(Box::new(MyPlugin))
    .build()
    .await?;
```

## Feature Flags

Control which components to include:

```toml
[dependencies]
ipfrs = { version = "0.2.0", features = ["full"] }

# Or selectively enable features:
ipfrs = {
    version = "0.2.0",
    features = ["storage", "network", "tensorlogic"]
}
```

Available features:
- `full` - All features enabled
- `storage` - Storage layer
- `network` - P2P networking
- `transport` - Data exchange protocols
- `semantic` - Vector search
- `tensorlogic` - TensorLogic integration
- `interface` - HTTP/gRPC APIs
- `cli` - Command-line interface

## Performance Characteristics

| Metric | Kubo (Go) | IPFRS (Rust) |
|--------|-----------|--------------|
| Memory (Idle) | 200 MB | 20 MB |
| Memory (Active) | 800 MB | 150 MB |
| Startup Time | 5s | 0.5s |
| Block Add (1MB) | 50ms | 5ms |
| Block Get (1MB) | 30ms | 3ms |

## Dependencies

- `ipfrs-core` - Core primitives
- `ipfrs-storage` - Storage layer
- `ipfrs-network` - Networking
- `ipfrs-transport` - Data exchange
- `ipfrs-semantic` - Vector search
- `ipfrs-tensorlogic` - TensorLogic
- `ipfrs-interface` - APIs
- `tokio` - Async runtime

## References

- IPFRS v0.2.0 Whitepaper (Network Architecture)
- IPFRS v0.3.0 Whitepaper (Intelligence/Unified Architecture) *(planned)*
