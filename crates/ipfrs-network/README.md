# ipfrs-network

Production-ready P2P networking layer for IPFRS built on rust-libp2p.

## Overview

`ipfrs-network` provides comprehensive networking capabilities for IPFRS with 48 modules covering all aspects of P2P networking:

- **Core Networking**: libp2p-based swarm with QUIC and TCP transports
- **Peer Discovery**: Kademlia DHT, mDNS, Bootstrap nodes, Semantic DHT
- **Connection Management**: Intelligent limits, priority-based pruning, rate limiting
- **NAT Traversal**: AutoNAT, DCUtR hole punching, Circuit Relay v2
- **Pub/Sub**: GossipSub with topic-based messaging and mesh optimization
- **Advanced Routing**: Geographic routing, quality prediction, intelligent peer selection
- **Transport**: Multipath QUIC, connection migration, QUIC monitoring
- **Privacy**: Tor integration with onion routing and hidden services
- **Mobile/IoT**: Battery optimization, adaptive polling, background mode, offline queue
- **Monitoring**: Comprehensive metrics, Prometheus export, health checks, diagnostics
- **Reliability**: Retry logic, circuit breakers, fallback strategies
- **Performance**: Auto-tuning, load testing, traffic analysis, benchmarking
- **Policy**: Fine-grained network control with connection/bandwidth/content policies

## Key Features

### Core Networking
- **Multi-transport**: QUIC (primary) with TCP fallback for maximum compatibility
- **NAT Traversal**: Three-layer approach (AutoNAT + DCUtR + Circuit Relay)
- **Peer Discovery**: Multiple mechanisms (DHT, mDNS, bootstrap, peer exchange)
- **Connection Management**: Intelligent limits, priority-based pruning, bandwidth tracking

### Advanced DHT
- **Content Routing**: Provider record publishing and discovery with automatic refresh
- **Peer Routing**: Find closest peers, routing table management
- **Semantic DHT**: Vector-based content discovery using embeddings and LSH
- **Query Optimization**: Early termination, pipelining, quality scoring, caching
- **Pluggable Providers**: Custom DHT implementations via trait interface

### Pub/Sub & Messaging
- **GossipSub**: Topic-based publish/subscribe with mesh optimization
- **Message Deduplication**: Efficient duplicate detection with TTL
- **Peer Scoring**: Quality-based peer selection for reliable delivery
- **Content Announcements**: Standard topics for network-wide broadcasts

### Mobile & IoT Optimizations
- **Adaptive Polling**: Activity-based interval adjustment for power saving
- **Bandwidth Throttling**: Token bucket rate limiting with burst support
- **Background Mode**: Pause/resume for mobile app lifecycle
- **Offline Queue**: Request queuing with priority-based replay
- **Memory Monitoring**: Component-level tracking with budget enforcement
- **Network Monitoring**: Interface change detection and handling
- **Connection Migration**: Seamless QUIC migration on network switches

### Advanced Routing & Selection
- **Geographic Routing**: Proximity-based peer selection using Haversine distance
- **Quality Prediction**: Historical performance tracking for intelligent routing
- **Peer Selection**: Multi-factor scoring combining geography and quality
- **Multipath QUIC**: Multiple path management with quality-based selection

### Privacy & Security
- **Tor Integration**: Onion routing with circuit management
- **Hidden Services**: Host and connect to .onion addresses
- **Stream Isolation**: Maximum privacy with separate circuits
- **Policy Engine**: Fine-grained control over connections and content

### Monitoring & Observability
- **Comprehensive Metrics**: Connection, DHT, bandwidth, latency tracking
- **Prometheus Export**: Ready-to-use metrics for monitoring systems
- **Health Checks**: Component-level and overall health assessment
- **Diagnostics**: Automated troubleshooting and issue detection
- **Traffic Analysis**: Pattern detection and anomaly identification
- **Time-Series Aggregation**: Historical metrics with statistical analysis

### Performance & Reliability
- **Auto-Tuning**: Automatic configuration based on system resources
- **Load Testing**: Comprehensive stress testing utilities
- **Benchmarking**: Performance measurement and regression tracking
- **Circuit Breakers**: Prevent cascading failures
- **Retry Logic**: Exponential backoff with configurable limits
- **Fallback Strategies**: Alternative peers, relay fallback, degraded mode

### Testing & Simulation
- **Network Simulation**: Test under adverse conditions (latency, packet loss, partitions)
- **Load Testing**: Stress test with connection storms, query floods, bandwidth saturation
- **Chaos Testing**: Verify resilience under extreme conditions
- **IPFS Compatibility**: Full compatibility testing with Kubo

## Quick Start

### Basic Usage

```rust
use ipfrs_network::{NetworkConfig, NetworkNode};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create configuration
    let config = NetworkConfig {
        listen_addrs: vec!["/ip4/0.0.0.0/udp/0/quic-v1".to_string()],
        enable_quic: true,
        enable_mdns: true,
        enable_nat_traversal: true,
        ..Default::default()
    };

    // Create and start network node
    let mut node = NetworkNode::new(config)?;
    node.start().await?;

    // Check network health
    let health = node.get_network_health();
    println!("Network status: {:?}", health.status);

    // Announce content to DHT
    let cid = cid::Cid::default();
    node.provide(&cid).await?;

    // Find providers
    let providers = node.find_providers(&cid).await?;
    println!("Found {} providers", providers.len());

    // Get network statistics
    let stats = node.stats();
    println!("Connected peers: {}", stats.connected_peers);

    Ok(())
}
```

### High-Level Facade

For easy integration of all features, use the `NetworkFacade`:

```rust
use ipfrs_network::NetworkFacadeBuilder;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a mobile-optimized node with advanced features
    let mut facade = NetworkFacadeBuilder::new()
        .with_preset_mobile()        // Mobile-optimized settings
        .with_semantic_dht()          // Vector-based content discovery
        .with_gossipsub()             // Pub/sub messaging
        .with_geo_routing()           // Geographic optimization
        .with_quality_prediction()    // Smart peer selection
        .build()?;

    facade.start().await?;
    println!("Peer ID: {}", facade.peer_id());

    // Semantic search
    let embedding = vec![0.1, 0.2, 0.3];
    let results = facade.semantic_search("text", &embedding, 10).await?;

    // Publish message
    facade.publish("my-topic", b"Hello, network!".to_vec()).await?;

    // Get network statistics
    let stats = facade.get_network_stats()?;
    println!("Connected peers: {}", stats.connected_peers);

    Ok(())
}
```

### Configuration Presets

Use pre-configured presets for common scenarios:

```rust
use ipfrs_network::NetworkPreset;

// Low-memory devices (< 128 MB RAM)
let preset = NetworkPreset::low_memory();

// IoT devices (128-512 MB RAM)
let preset = NetworkPreset::iot();

// Mobile devices
let preset = NetworkPreset::mobile();

// High-performance servers
let preset = NetworkPreset::high_performance();

// Privacy-focused with Tor
let preset = NetworkPreset::privacy();

// Development/testing
let preset = NetworkPreset::development();
```

## Architecture

The crate consists of 48 modules organized into logical groups:

### Core Components
- **node**: Core NetworkNode implementation with libp2p swarm
- **dht**: Kademlia DHT operations and caching
- **peer**: Peer store for tracking known peers
- **connection_manager**: Connection limits and pruning
- **bootstrap**: Bootstrap peer management

### Advanced Features
- **facade**: High-level API for easy integration
- **semantic_dht**: Vector-based semantic routing
- **gossipsub**: Pub/sub messaging
- **multipath_quic**: Multipath transport support
- **tor**: Privacy-preserving networking

### Mobile & IoT
- **adaptive_polling**: Power-saving polling strategies
- **throttle**: Bandwidth throttling
- **background_mode**: Mobile app lifecycle integration
- **offline_queue**: Offline request queuing
- **memory_monitor**: Memory usage tracking

### Routing & Selection
- **geo_routing**: Geographic proximity routing
- **quality_predictor**: Performance-based routing
- **peer_selector**: Multi-factor peer selection
- **dht_provider**: Pluggable DHT backends

### Monitoring & Analysis
- **metrics**: Comprehensive metrics tracking
- **metrics_aggregator**: Time-series aggregation
- **health**: Health monitoring
- **diagnostics**: Troubleshooting utilities
- **traffic_analyzer**: Traffic pattern analysis

### Performance & Testing
- **auto_tuner**: Automatic configuration tuning
- **benchmarking**: Performance benchmarks
- **load_tester**: Load and stress testing
- **network_simulator**: Network condition simulation

### Reliability & Policy
- **fallback**: Fallback strategies
- **rate_limiter**: Connection rate limiting
- **reputation**: Peer reputation system
- **policy**: Network policy engine
- **session**: Session lifecycle management

## Examples

The `examples/` directory contains 38 comprehensive examples:

### Basic Examples
- `basic_node.rs` - Creating and starting a network node
- `dht_operations.rs` - Content announcement and discovery
- `connection_management.rs` - Connection tracking and bandwidth

### Advanced Features
- `network_facade_demo.rs` - Using the high-level facade
- `semantic_search.rs` - Vector-based content discovery
- `pubsub_messaging.rs` - Pub/sub messaging with GossipSub
- `multipath_quic.rs` - Multipath QUIC demonstration
- `tor_privacy.rs` - Privacy-preserving networking

### Mobile & IoT
- `low_memory_node.rs` - Low-memory configuration
- `background_mode.rs` - Mobile lifecycle integration
- `network_monitoring.rs` - Network interface monitoring
- `offline_queue_demo.rs` - Offline request handling

### Performance & Monitoring
- `auto_tuning.rs` - Automatic configuration
- `performance_benchmarking.rs` - Benchmarking utilities
- `load_testing.rs` - Load and stress testing
- `metrics_prometheus.rs` - Metrics export
- `health_monitoring.rs` - Health checking

### Advanced Routing
- `geographic_routing.rs` - Geographic optimization
- `quality_prediction.rs` - Quality-based routing
- `intelligent_peer_selection.rs` - Multi-factor selection

### Testing & Analysis
- `network_simulation.rs` - Network condition simulation
- `traffic_analysis.rs` - Traffic pattern analysis
- `network_diagnostics.rs` - Troubleshooting

## Testing

The crate includes:
- **507+ unit tests** covering all modules
- **35 doc tests** validating documentation examples
- **Integration tests** with IPFS Kubo compatibility
- **Chaos tests** for resilience verification
- **Zero warnings** - strict code quality enforcement

Run tests:
```bash
cargo test
cargo test --lib          # Library tests only
cargo test --examples     # Example compilation
cargo clippy              # Linting
cargo doc --no-deps       # Documentation
```

## Performance

Performance targets (achieved on typical hardware):
- **Connection establishment**: < 100ms (local), < 500ms (remote)
- **DHT lookup**: < 2s (20 hops)
- **Concurrent connections**: 1000+ peers
- **Memory per peer**: < 10KB
- **Zero-copy operations**: Where possible
- **ARM optimization**: Full ARMv7/AArch64 support

## Dependencies

- `libp2p` - P2P networking framework
- `quinn` - QUIC implementation
- `tokio` - Async runtime
- `cid` - Content addressing
- `multihash` - Hash functions
- `prometheus` - Metrics export
- `dashmap` - Concurrent data structures
- `parking_lot` - Fast synchronization

## Documentation

Comprehensive documentation is available:
- **Module docs**: See `cargo doc --open`
- **Architecture guide**: `docs/ARCHITECTURE.md`
- **Peer discovery guide**: `docs/PEER_DISCOVERY.md`
- **NAT traversal guide**: `docs/NAT_TRAVERSAL.md`
- **Configuration reference**: `docs/CONFIGURATION.md`

## License

See LICENSE file in the root of the repository.

## Contributing

This is part of the IPFRS project. See the main repository for contribution guidelines.
