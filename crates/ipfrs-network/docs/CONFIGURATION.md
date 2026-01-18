# IPFRS Network Configuration Reference

Complete reference for all configuration options in IPFRS Network.

## Table of Contents

1. [NetworkConfig](#networkconfig)
2. [KademliaConfig](#kademliaconfig)
3. [ConnectionLimitsConfig](#connectionlimitsconfig)
4. [BootstrapConfig](#bootstrapconfig)
5. [DhtConfig](#dhtconfig)
6. [ProviderCacheConfig](#providercacheconfig)
7. [QueryOptimizerConfig](#queryoptimizerconfig)
8. [SemanticDhtConfig](#semanticdhtconfig)
9. [GossipSubConfig](#gossipsubconfig)
10. [FallbackConfig](#fallbackconfig)
11. [LoggingConfig](#loggingconfig)
12. [Complete Examples](#complete-examples)

---

## NetworkConfig

Main configuration for the network node.

```rust
pub struct NetworkConfig {
    pub listen_addrs: Vec<String>,
    pub bootstrap_peers: Vec<String>,
    pub enable_mdns: bool,
    pub enable_quic: bool,
    pub enable_nat_traversal: bool,
    pub relay_servers: Vec<String>,
    pub kademlia_config: KademliaConfig,
    pub connection_limits: ConnectionLimitsConfig,
    pub bootstrap_config: BootstrapConfig,
    pub peer_id: Option<PeerId>,
    pub keypair: Option<Keypair>,
}
```

### Fields

#### `listen_addrs: Vec<String>`

**Description:** Addresses to listen on for incoming connections.

**Default:** `vec!["/ip4/0.0.0.0/tcp/0".to_string()]`

**Examples:**
```rust
// Listen on all interfaces, random port
listen_addrs: vec!["/ip4/0.0.0.0/tcp/0".to_string()]

// Specific port
listen_addrs: vec!["/ip4/0.0.0.0/tcp/4001".to_string()]

// IPv6
listen_addrs: vec!["/ip6/::/tcp/4001".to_string()]

// QUIC
listen_addrs: vec!["/ip4/0.0.0.0/udp/4001/quic-v1".to_string()]

// Multiple transports
listen_addrs: vec![
    "/ip4/0.0.0.0/tcp/4001".to_string(),
    "/ip4/0.0.0.0/udp/4001/quic-v1".to_string(),
]

// WebSocket (future)
listen_addrs: vec!["/ip4/0.0.0.0/tcp/8080/ws".to_string()]
```

#### `bootstrap_peers: Vec<String>`

**Description:** Initial peers to connect to for network bootstrapping.

**Default:** `vec![]` (empty)

**Format:** Multiaddr with PeerID: `/ip4/1.2.3.4/tcp/4001/p2p/12D3KooW...`

**Examples:**
```rust
// Public IPFS bootstrap nodes
bootstrap_peers: vec![
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN".to_string(),
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmQCU2EcMqAqQPR2i9bChDtGNJchTbq5TbXJJ16u19uLTa".to_string(),
]

// Private network
bootstrap_peers: vec![
    "/ip4/192.168.1.10/tcp/4001/p2p/12D3KooWABC...".to_string(),
    "/ip4/192.168.1.11/tcp/4001/p2p/12D3KooWDEF...".to_string(),
]
```

#### `enable_mdns: bool`

**Description:** Enable mDNS for local peer discovery.

**Default:** `true`

**When to enable:**
- Local development
- Private LANs
- IoT devices on same network

**When to disable:**
- Public internet nodes (no benefit)
- Security-sensitive environments
- Large networks (broadcast storm risk)

```rust
enable_mdns: true  // Enable for local discovery
```

#### `enable_quic: bool`

**Description:** Enable QUIC transport.

**Default:** `true`

**Advantages:**
- Better NAT traversal
- Connection migration (mobile)
- Built-in encryption (TLS 1.3)
- Multiplexing without head-of-line blocking

**Disadvantages:**
- May be blocked by some firewalls
- Slightly higher CPU usage

```rust
enable_quic: true  // Recommended for modern networks
```

#### `enable_nat_traversal: bool`

**Description:** Enable NAT traversal (AutoNAT, DCUtR, Circuit Relay).

**Default:** `true`

**Components enabled:**
- AutoNAT: Detect NAT status
- DCUtR: Hole punching
- Circuit Relay Client: Use relays

```rust
enable_nat_traversal: true  // Essential for home/mobile users
```

#### `relay_servers: Vec<String>`

**Description:** Circuit relay servers for NAT traversal fallback.

**Default:** `vec![]`

**Format:** Multiaddr with PeerID

```rust
relay_servers: vec![
    "/ip4/relay.example.com/tcp/4001/p2p/12D3KooW...".to_string(),
    "/dns4/relay2.example.com/tcp/443/wss/p2p/12D3KooW...".to_string(),
]
```

#### `peer_id: Option<PeerID>`

**Description:** Fixed peer ID (derived from keypair).

**Default:** `None` (generated)

**Usage:**
```rust
// Generated (recommended)
peer_id: None

// Fixed (for servers)
let keypair = identity::Keypair::generate_ed25519();
peer_id: Some(PeerId::from(keypair.public()))
```

#### `keypair: Option<Keypair>`

**Description:** Cryptographic keypair for node identity.

**Default:** `None` (generated)

**Usage:**
```rust
// Generated
keypair: None

// Load from file
let keypair = load_keypair_from_file("identity.key")?;
keypair: Some(keypair)

// Generate specific type
use libp2p::identity;
let keypair = identity::Keypair::generate_ed25519();
keypair: Some(keypair)
```

---

## KademliaConfig

Configuration for the Kademlia DHT.

```rust
pub struct KademliaConfig {
    pub alpha: usize,
    pub replication_factor: usize,
    pub query_timeout: Duration,
    pub k_bucket_size: usize,
}
```

### Fields

#### `alpha: usize`

**Description:** Number of concurrent DHT queries.

**Default:** `3`

**Range:** `1-10`

**Trade-offs:**
- Higher = Faster queries, more bandwidth
- Lower = Slower queries, less bandwidth

```rust
alpha: 3  // Balanced
alpha: 1  // Low bandwidth
alpha: 5  // Fast discovery
```

#### `replication_factor: usize`

**Description:** Number of peers to replicate provider records to.

**Default:** `20`

**Range:** `5-50`

```rust
replication_factor: 20  // Standard
replication_factor: 5   // Minimal replication
replication_factor: 50  // High redundancy
```

#### `query_timeout: Duration`

**Description:** Timeout for DHT queries.

**Default:** `Duration::from_secs(60)`

**Range:** `10s-300s`

```rust
query_timeout: Duration::from_secs(60)   // Default
query_timeout: Duration::from_secs(30)   // Fast fail
query_timeout: Duration::from_secs(120)  // Patient
```

#### `k_bucket_size: usize`

**Description:** Maximum peers per k-bucket in routing table.

**Default:** `20`

**Range:** `4-100`

```rust
k_bucket_size: 20  // Standard Kademlia
k_bucket_size: 10  // Constrained devices
k_bucket_size: 50  // High-connectivity nodes
```

---

## ConnectionLimitsConfig

Configuration for connection limits and management.

```rust
pub struct ConnectionLimitsConfig {
    pub max_connections: usize,
    pub max_inbound: usize,
    pub max_outbound: usize,
    pub max_per_peer: usize,
}
```

### Fields

#### `max_connections: usize`

**Description:** Total maximum connections.

**Default:** `100`

**Considerations:**
- File descriptor limits
- Memory per connection (~10KB)
- CPU for connection management

```rust
max_connections: 100  // Default
max_connections: 20   // Constrained device
max_connections: 500  // High-capacity server
```

#### `max_inbound: usize`

**Description:** Maximum inbound connections.

**Default:** `60`

**Purpose:** Reserve capacity for outbound connections

```rust
max_inbound: 60   // Default (60% of total)
max_inbound: 10   // Client-focused node
max_inbound: 200  // Server node
```

#### `max_outbound: usize`

**Description:** Maximum outbound connections.

**Default:** `50`

```rust
max_outbound: 50  // Default
max_outbound: 30  // Bandwidth constrained
max_outbound: 100 // Aggressive discovery
```

#### `max_per_peer: usize`

**Description:** Maximum connections to single peer.

**Default:** `2`

**Purpose:** Multiple transports/addresses to same peer

```rust
max_per_peer: 2  // TCP + QUIC
max_per_peer: 1  // Single connection only
max_per_peer: 4  // Multiple interfaces
```

---

## BootstrapConfig

Configuration for bootstrap behavior.

```rust
pub struct BootstrapConfig {
    pub initial_delay: Duration,
    pub interval: Duration,
    pub max_retries: usize,
    pub min_peer_threshold: usize,
}
```

### Fields

#### `initial_delay: Duration`

**Description:** Delay before first bootstrap attempt.

**Default:** `Duration::from_millis(100)`

```rust
initial_delay: Duration::from_millis(100)  // Quick start
initial_delay: Duration::from_secs(1)      // Let other services start
```

#### `interval: Duration`

**Description:** Interval between bootstrap attempts.

**Default:** `Duration::from_secs(300)` (5 minutes)

```rust
interval: Duration::from_secs(300)  // Default
interval: Duration::from_secs(60)   // Aggressive
interval: Duration::from_secs(600)  // Conservative
```

#### `max_retries: usize`

**Description:** Maximum retry attempts per peer.

**Default:** `5`

```rust
max_retries: 5   // Default
max_retries: 3   // Fast fail
max_retries: 10  // Persistent
```

#### `min_peer_threshold: usize`

**Description:** Minimum connected peers before bootstrap is satisfied.

**Default:** `1`

```rust
min_peer_threshold: 1  // Any connection
min_peer_threshold: 3  // Multiple peers
min_peer_threshold: 5  // Well-connected
```

---

## DhtConfig

Configuration for DHT manager.

```rust
pub struct DhtConfig {
    pub provider_ttl: Duration,
    pub provider_refresh_interval: Duration,
    pub query_cache_ttl: Duration,
    pub peer_cache_ttl: Duration,
}
```

### Fields

#### `provider_ttl: Duration`

**Description:** Time-to-live for provider records.

**Default:** `Duration::from_secs(86400)` (24 hours)

```rust
provider_ttl: Duration::from_secs(86400)   // 24 hours
provider_ttl: Duration::from_secs(3600)    // 1 hour (dynamic content)
provider_ttl: Duration::from_secs(604800)  // 1 week (static content)
```

#### `provider_refresh_interval: Duration`

**Description:** How often to refresh provider records.

**Default:** `Duration::from_secs(43200)` (12 hours, half of TTL)

```rust
provider_refresh_interval: Duration::from_secs(43200)  // Default
```

#### `query_cache_ttl: Duration`

**Description:** Cache duration for DHT query results.

**Default:** `Duration::from_secs(300)` (5 minutes)

```rust
query_cache_ttl: Duration::from_secs(300)  // Default
query_cache_ttl: Duration::from_secs(60)   // Frequent changes
query_cache_ttl: Duration::from_secs(3600) // Rare changes
```

#### `peer_cache_ttl: Duration`

**Description:** Cache duration for peer information.

**Default:** `Duration::from_secs(3600)` (1 hour)

```rust
peer_cache_ttl: Duration::from_secs(3600)  // Default
```

---

## ProviderCacheConfig

Configuration for provider record caching.

```rust
pub struct ProviderCacheConfig {
    pub max_providers_per_cid: usize,
    pub cache_size: usize,
    pub ttl: Duration,
}
```

### Fields

#### `max_providers_per_cid: usize`

**Description:** Maximum providers to cache per CID.

**Default:** `100`

```rust
max_providers_per_cid: 100  // Default
max_providers_per_cid: 20   // Memory constrained
max_providers_per_cid: 500  // Popular content
```

#### `cache_size: usize`

**Description:** Maximum CIDs in cache.

**Default:** `10000`

```rust
cache_size: 10000  // Default (~1MB memory)
cache_size: 1000   // Constrained
cache_size: 100000 // Large cache (~10MB)
```

#### `ttl: Duration`

**Description:** Provider cache entry TTL.

**Default:** `Duration::from_secs(600)` (10 minutes)

```rust
ttl: Duration::from_secs(600)  // Default
ttl: Duration::from_secs(60)   // Frequently changing
ttl: Duration::from_secs(3600) // Stable network
```

---

## QueryOptimizerConfig

Configuration for DHT query optimization.

```rust
pub struct QueryOptimizerConfig {
    pub enable_early_termination: bool,
    pub quality_threshold: f64,
    pub enable_pipelining: bool,
    pub max_concurrent_queries: usize,
}
```

### Fields

#### `enable_early_termination: bool`

**Description:** Stop query when sufficient quality results found.

**Default:** `true`

```rust
enable_early_termination: true  // Faster queries
enable_early_termination: false // Exhaustive search
```

#### `quality_threshold: f64`

**Description:** Minimum result quality (0.0-1.0) for early termination.

**Default:** `0.8`

```rust
quality_threshold: 0.8  // High quality required
quality_threshold: 0.5  // Accept medium quality
quality_threshold: 0.9  // Very high quality only
```

#### `enable_pipelining: bool`

**Description:** Pipeline multiple queries concurrently.

**Default:** `true`

```rust
enable_pipelining: true  // Better throughput
enable_pipelining: false // Sequential queries
```

#### `max_concurrent_queries: usize`

**Description:** Maximum concurrent pipelined queries.

**Default:** `10`

```rust
max_concurrent_queries: 10  // Default
max_concurrent_queries: 3   // Limited concurrency
max_concurrent_queries: 50  // High concurrency
```

---

## SemanticDhtConfig

Configuration for semantic DHT.

```rust
pub struct SemanticDhtConfig {
    pub lsh_hash_functions: usize,
    pub lsh_hash_tables: usize,
    pub lsh_bucket_width: f32,
    pub max_query_peers: usize,
    pub query_timeout: Duration,
    pub enable_caching: bool,
    pub cache_ttl: Duration,
    pub max_cache_size: usize,
    pub top_k: usize,
}
```

### Fields

#### `lsh_hash_functions: usize`

**Description:** Number of LSH hash functions per table.

**Default:** `8`

**Trade-offs:**
- More functions = Better precision, more computation
- Fewer functions = Faster, less precision

```rust
lsh_hash_functions: 8   // Default
lsh_hash_functions: 4   // Fast, less accurate
lsh_hash_functions: 16  // Slow, more accurate
```

#### `lsh_hash_tables: usize`

**Description:** Number of LSH hash tables.

**Default:** `4`

**Trade-offs:**
- More tables = Better recall, more memory
- Fewer tables = Less memory, may miss results

```rust
lsh_hash_tables: 4  // Default
lsh_hash_tables: 2  // Memory constrained
lsh_hash_tables: 8  // High recall needed
```

#### `lsh_bucket_width: f32`

**Description:** Quantization width for LSH.

**Default:** `4.0`

**Tuning:**
- Larger = Coarser buckets, more collisions
- Smaller = Finer buckets, fewer collisions

```rust
lsh_bucket_width: 4.0  // Default
lsh_bucket_width: 2.0  // Fine-grained
lsh_bucket_width: 8.0  // Coarse
```

#### `max_query_peers: usize`

**Description:** Maximum peers to query for semantic search.

**Default:** `20`

```rust
max_query_peers: 20  // Default
max_query_peers: 5   // Fast, may miss results
max_query_peers: 50  // Thorough search
```

#### `top_k: usize`

**Description:** Number of results to return.

**Default:** `10`

```rust
top_k: 10  // Default
top_k: 5   // Quick results
top_k: 50  // Comprehensive results
```

---

## GossipSubConfig

Configuration for GossipSub pub/sub.

```rust
pub struct GossipSubConfig {
    pub mesh_n_low: usize,
    pub mesh_n: usize,
    pub mesh_n_high: usize,
    pub gossip_n: usize,
    pub heartbeat_interval: Duration,
    pub max_message_size: usize,
    pub enable_scoring: bool,
    pub duplicate_cache_time: Duration,
    pub max_duplicate_cache_size: usize,
    pub enable_validation: bool,
}
```

### Fields

#### `mesh_n_low: usize` (D_low)

**Description:** Minimum peers in mesh before grafting.

**Default:** `4`

```rust
mesh_n_low: 4  // Default
```

#### `mesh_n: usize` (D)

**Description:** Target number of mesh peers.

**Default:** `6`

```rust
mesh_n: 6  // Default (Goldilocks number)
```

#### `mesh_n_high: usize` (D_high)

**Description:** Maximum peers in mesh before pruning.

**Default:** `12`

```rust
mesh_n_high: 12  // Default (2× target)
```

#### `gossip_n: usize` (D_lazy)

**Description:** Number of peers to gossip to.

**Default:** `3`

```rust
gossip_n: 3  // Default
```

#### `heartbeat_interval: Duration`

**Description:** Mesh maintenance interval.

**Default:** `Duration::from_secs(1)`

```rust
heartbeat_interval: Duration::from_secs(1)  // Default
```

#### `max_message_size: usize`

**Description:** Maximum message size in bytes.

**Default:** `1048576` (1 MB)

```rust
max_message_size: 1024 * 1024      // 1 MB
max_message_size: 10 * 1024 * 1024 // 10 MB
```

---

## FallbackConfig

Configuration for fallback strategies.

```rust
pub struct FallbackConfig {
    pub max_retries: usize,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
    pub circuit_breaker_threshold: usize,
    pub circuit_breaker_timeout: Duration,
}
```

### Fields

#### `max_retries: usize`

**Description:** Maximum retry attempts.

**Default:** `3`

```rust
max_retries: 3  // Default
max_retries: 1  // Fast fail
max_retries: 10 // Persistent
```

#### `initial_backoff: Duration`

**Description:** Initial retry delay.

**Default:** `Duration::from_millis(100)`

```rust
initial_backoff: Duration::from_millis(100)  // Default
```

#### `max_backoff: Duration`

**Description:** Maximum retry delay (exponential backoff cap).

**Default:** `Duration::from_secs(60)`

```rust
max_backoff: Duration::from_secs(60)  // Default (1 minute max)
```

#### `circuit_breaker_threshold: usize`

**Description:** Consecutive failures before opening circuit.

**Default:** `5`

```rust
circuit_breaker_threshold: 5  // Default
circuit_breaker_threshold: 3  // Fail fast
circuit_breaker_threshold: 10 // Tolerant
```

#### `circuit_breaker_timeout: Duration`

**Description:** Time before trying half-open state.

**Default:** `Duration::from_secs(60)`

```rust
circuit_breaker_timeout: Duration::from_secs(60)  // Default
```

---

## LoggingConfig

Configuration for structured logging.

```rust
pub struct LoggingConfig {
    pub log_level: LogLevel,
    pub enable_tracing: bool,
    pub log_connections: bool,
    pub log_dht_events: bool,
}
```

### Fields

#### `log_level: LogLevel`

**Description:** Minimum log level to emit.

**Default:** `LogLevel::Info`

```rust
log_level: LogLevel::Trace  // Everything
log_level: LogLevel::Debug  // Debug info
log_level: LogLevel::Info   // Default
log_level: LogLevel::Warn   // Warnings only
log_level: LogLevel::Error  // Errors only
```

---

## Complete Examples

### Minimal Config

```rust
let config = NetworkConfig::default();
```

### Development Config

```rust
let config = NetworkConfig {
    listen_addrs: vec!["/ip4/127.0.0.1/tcp/0".to_string()],
    enable_mdns: true,
    bootstrap_peers: vec![],
    ..Default::default()
};
```

### Production Server Config

```rust
let config = NetworkConfig {
    listen_addrs: vec![
        "/ip4/0.0.0.0/tcp/4001".to_string(),
        "/ip4/0.0.0.0/udp/4001/quic-v1".to_string(),
    ],
    bootstrap_peers: public_bootstrap_nodes(),
    enable_mdns: false,
    enable_nat_traversal: false,  // Public server
    connection_limits: ConnectionLimitsConfig {
        max_connections: 500,
        max_inbound: 400,
        max_outbound: 100,
        max_per_peer: 2,
    },
    kademlia_config: KademliaConfig {
        alpha: 5,
        k_bucket_size: 50,
        ..Default::default()
    },
    ..Default::default()
};
```

### IoT Device Config

```rust
let config = NetworkConfig {
    listen_addrs: vec!["/ip4/0.0.0.0/tcp/0".to_string()],
    bootstrap_peers: vec![gateway_peer()],
    enable_mdns: true,
    enable_quic: false,  // Save memory
    connection_limits: ConnectionLimitsConfig {
        max_connections: 5,
        max_inbound: 2,
        max_outbound: 3,
        max_per_peer: 1,
    },
    kademlia_config: KademliaConfig {
        alpha: 1,
        k_bucket_size: 10,
        ..Default::default()
    },
    ..Default::default()
};
```

### High-Performance Config

```rust
let config = NetworkConfig {
    listen_addrs: vec![
        "/ip4/0.0.0.0/tcp/4001".to_string(),
        "/ip4/0.0.0.0/udp/4001/quic-v1".to_string(),
    ],
    enable_quic: true,
    connection_limits: ConnectionLimitsConfig {
        max_connections: 1000,
        max_inbound: 800,
        max_outbound: 200,
        max_per_peer: 4,
    },
    kademlia_config: KademliaConfig {
        alpha: 10,
        replication_factor: 50,
        k_bucket_size: 100,
        ..Default::default()
    },
    ..Default::default()
};
```
