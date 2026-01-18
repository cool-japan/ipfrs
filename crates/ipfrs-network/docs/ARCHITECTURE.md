# IPFRS Network Architecture

This document describes the architecture and design of the IPFRS network layer.

## Overview

IPFRS Network is a comprehensive peer-to-peer networking layer built on libp2p, providing content-addressed storage with semantic routing capabilities. It combines traditional DHT-based content routing with vector-based semantic discovery.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         NetworkNode                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   libp2p     │  │    DHT       │  │  Semantic    │          │
│  │    Swarm     │  │   Manager    │  │     DHT      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  GossipSub   │  │  Connection  │  │   Bootstrap  │          │
│  │   Manager    │  │   Manager    │  │   Manager    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Transport     │  │   Discovery     │  │   Protocols     │
│                 │  │                 │  │                 │
│  • QUIC         │  │  • Kademlia     │  │  • Identify     │
│  • TCP          │  │  • mDNS         │  │  • Ping         │
│  • WebSocket    │  │  • Bootstrap    │  │  • Bitswap      │
│                 │  │  • AutoNAT      │  │  • Custom       │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Core Components

### 1. NetworkNode

The `NetworkNode` is the main entry point for network operations. It manages:

- **libp2p Swarm**: Core event loop and connection management
- **Protocol Behaviors**: Kademlia, mDNS, Ping, Identify, AutoNAT, DCUtR, Relay
- **Event Processing**: Asynchronous event handling with tokio
- **Lifecycle Management**: Start, stop, and graceful shutdown

**Key Methods:**
- `new(config)` - Create a new network node
- `start()` - Begin listening and processing events
- `connect(peer_id, addrs)` - Connect to a peer
- `provide(cid)` - Announce content to DHT
- `find_providers(cid)` - Discover content providers

### 2. DHT Manager

The `DhtManager` provides advanced Kademlia DHT operations:

**Features:**
- Provider record management with automatic refresh
- Query result caching with TTL
- Routing table health monitoring
- Peer discovery and routing

**Architecture:**
```
DhtManager
├── Provider Records (DashMap<Cid, ProviderInfo>)
├── Query Cache (DashMap<QueryKey, CachedResult>)
├── Peer Cache (DashMap<PeerId, PeerCacheEntry>)
└── Background Tasks (Tokio spawn)
    ├── Provider refresh (every 12h)
    └── Cache cleanup (every 5min)
```

### 3. Semantic DHT

The `SemanticDht` extends traditional DHT with vector-based routing:

**Design:**
```
SemanticDht
├── Namespaces (text, image, audio, custom)
├── LSH Projections (random vectors per namespace)
├── Local Index (Cid → Embedding mapping)
├── Query Cache (TTL-based)
└── Hash→Peer Mapping (for distributed routing)
```

**LSH Process:**
1. Input: High-dimensional embedding (e.g., 768-dim)
2. Project onto random hyperplanes (8 functions × 4 tables)
3. Quantize projections into hash buckets
4. Map hash to DHT key (CID)
5. Query DHT for peers in similar buckets

**Distance Metrics:**
- **Euclidean**: L2 distance, good for spatial data
- **Cosine**: Angular distance, good for text embeddings
- **Manhattan**: L1 distance, robust to outliers
- **Dot Product**: Direct similarity, for normalized vectors

### 4. GossipSub Manager

The `GossipSubManager` implements topic-based pub/sub:

**Mesh Formation:**
```
Topic Mesh
├── Target peers (D = 6)
├── Low watermark (D_low = 4)
├── High watermark (D_high = 12)
└── Gossip peers (D_lazy = 3)
```

**Message Flow:**
1. Publisher sends to mesh peers (D peers)
2. Mesh peers forward to their mesh
3. Non-mesh peers receive gossip (IHAVE messages)
4. Peers with missing messages request (IWANT)
5. Duplicate detection via seen cache

**Peer Scoring:**
```
Score = avg(topic_scores) × (1 - invalid_ratio)

Where:
- topic_scores: Per-topic behavior scores
- invalid_ratio: Invalid messages / Total messages
```

### 5. Connection Manager

Manages connection limits and quality:

**Connection Limits:**
- Max total connections (default: 100)
- Max inbound connections (default: 60)
- Max outbound connections (default: 50)
- Reserved slots for important peers

**Pruning Strategy:**
1. Calculate connection value: `1.0 / (latency_ms + 1.0)`
2. Sort connections by value (lowest first)
3. Prune lowest-value connections
4. Preserve reserved and recent connections

### 6. Bootstrap Manager

Handles network bootstrapping with resilience:

**Features:**
- Exponential backoff retry (100ms → 6.4s)
- Circuit breaker pattern (5 consecutive failures → open)
- Per-peer connection tracking
- Configurable bootstrap peer list

**Bootstrap Flow:**
```
1. Load bootstrap peers from config
2. For each peer:
   a. Check circuit breaker state
   b. If closed: attempt connection
   c. On success: reset backoff
   d. On failure: increase backoff, increment failure count
3. If connected peers < threshold: retry after backoff
```

## Transport Layer

### QUIC Transport

Primary transport for modern networking:

**Advantages:**
- Built-in encryption (TLS 1.3)
- Connection migration (mobile/WiFi switching)
- Multiplexing without head-of-line blocking
- 0-RTT connection resumption
- UDP-based (NAT-friendly)

**Configuration:**
```rust
QuicConfig::new(&keypair)
```

### TCP Transport

Fallback for environments without UDP:

**Stack:**
```
TCP → Noise (encryption) → Yamux (multiplexing)
```

**Configuration:**
```rust
let tcp = TcpTransport::new(config)
    .upgrade(noise_authenticated)
    .multiplex(yamux_config)
```

## NAT Traversal

### Three-Layer Approach

1. **AutoNAT**: Detect NAT type and external address
2. **DCUtR** (Direct Connection Upgrade through Relay): Hole punching
3. **Circuit Relay v2**: Fallback relayed connection

**Flow:**
```
1. AutoNAT probe → Determine if behind NAT
2. If behind NAT:
   a. Try DCUtR hole punching
   b. If DCUtR fails → Use relay
3. If public → Direct connection
```

### NAT Types Handled

- **Full Cone**: Direct connection possible
- **Restricted Cone**: DCUtR usually succeeds
- **Port-Restricted**: DCUtR with coordination
- **Symmetric**: Relay required

## Discovery Mechanisms

### 1. Kademlia DHT

**Routing Table:**
- K-buckets (k=20) organized by XOR distance
- Replacement cache for full buckets
- Periodic refresh to maintain health

**Queries:**
- `FIND_NODE`: Discover peers near a key
- `GET_PROVIDERS`: Find content providers
- `ADD_PROVIDER`: Announce content

**Optimization:**
- Alpha (α=3) concurrent queries
- Query timeout: 60s
- Replication factor: 20

### 2. mDNS

Local network discovery without infrastructure:

**Configuration:**
- Query interval: 5s
- Service name: `_ipfrs._udp.local`
- Auto-connect on discovery

### 3. Bootstrap Nodes

Static peer list for initial network entry:

**Default Bootstrap Peers:**
- Configurable via `NetworkConfig::bootstrap_peers`
- Persistent storage of successful peers
- Automatic retry with backoff

## Protocol Registry

Extensible protocol system for custom protocols:

**Architecture:**
```
ProtocolRegistry
├── Handlers (HashMap<ProtocolId, Box<dyn ProtocolHandler>>)
├── Capabilities (HashMap<ProtocolId, ProtocolCapabilities>)
└── Lifecycle Management (init/shutdown hooks)
```

**Version Negotiation:**
```
Client: Supports [1.0.0, 1.1.0, 1.2.0]
Server: Supports [1.1.0, 1.2.0, 1.3.0]
Result: Negotiate 1.2.0 (highest common)
```

## Fallback Strategies

Multi-layer resilience approach:

### 1. Alternative Peers

When primary peer fails, try alternatives:
- Select from provider list
- Sort by latency/reputation
- Implement max retry limit

### 2. Relay Fallback

If direct connection fails:
- Use Circuit Relay v2
- Automatic relay selection
- Bandwidth accounting

### 3. Degraded Mode

When network is constrained:
- Reduce connection limits
- Disable non-essential protocols
- Increase query timeouts

### 4. Circuit Breaker

Prevent cascading failures:
- **Closed**: Normal operation
- **Open**: Fast-fail after threshold failures
- **Half-Open**: Test recovery after timeout

## Metrics and Monitoring

### Prometheus Metrics

**Connection Metrics:**
- `connections_established_total`
- `connections_failed_total`
- `connections_active`
- `connections_closed_total`

**DHT Metrics:**
- `dht_queries_total`
- `dht_successful_queries`
- `dht_failed_queries`
- `dht_providers_announced`
- `dht_providers_found`

**Bandwidth Metrics:**
- `bandwidth_bytes_sent`
- `bandwidth_bytes_received`

**Health Metrics:**
- `health_score` (0.0-1.0)
- `routing_table_peers`
- `connected_peers`

### Health Checking

**Component Health:**
- **Connections**: Are we connected to peers?
- **DHT**: Is routing table healthy?
- **Bandwidth**: Are we transmitting data?

**Overall Health:**
```
Healthy:     score >= 0.7
Degraded:    0.3 <= score < 0.7
Unhealthy:   score < 0.3
Unknown:     No data yet
```

## Event Processing

**Event Loop:**
```rust
loop {
    tokio::select! {
        event = swarm.next() => {
            // Process libp2p events
            match event {
                SwarmEvent::ConnectionEstablished { .. } => { ... }
                SwarmEvent::Behaviour(event) => { ... }
                ...
            }
        }
        _ = shutdown_rx.changed() => {
            // Graceful shutdown
            break;
        }
    }
}
```

**Event Types:**
- Connection events (established, closed, failed)
- DHT events (bootstrap, providers, routing)
- Protocol events (identify, ping, custom)
- NAT status changes

## Data Flow

### Content Announcement

```
1. Application calls provide(cid)
2. DhtManager computes DHT key
3. Announce to k closest peers (k=20)
4. Store provider record (TTL=24h)
5. Background task refreshes every 12h
```

### Content Discovery

```
1. Application calls find_providers(cid)
2. Check provider cache (TTL=5min)
3. If miss: Query DHT
4. Query k closest peers in parallel
5. Aggregate results (remove duplicates)
6. Cache results
7. Return provider list
```

### Semantic Search

```
1. Application submits SemanticQuery
2. Compute LSH hashes for embedding
3. Convert hashes to DHT keys
4. Query DHT for each hash
5. Collect candidate peers
6. Compute exact distances (optional)
7. Rank by similarity score
8. Return top-k results
```

## Performance Characteristics

### Latency

- **Local Connection**: <100ms
- **Remote Connection**: <500ms
- **DHT Lookup**: <2s (20 hops)
- **Semantic Query**: <3s (LSH + DHT)
- **Provider Refresh**: Background (non-blocking)

### Throughput

- **QUIC**: Up to 1 Gbps (hardware limited)
- **TCP**: Up to 800 Mbps (hardware limited)
- **DHT Queries**: ~100 concurrent
- **GossipSub Topics**: Unlimited
- **Semantic Namespaces**: Unlimited

### Scalability

- **Concurrent Connections**: 1000+ peers
- **Memory per Peer**: <10KB
- **DHT Routing Table**: ~1000 peers (auto-pruned)
- **Provider Cache**: 10,000 entries
- **Query Cache**: 10,000 entries

## Security Considerations

### Transport Security

- **QUIC**: TLS 1.3 (mandatory)
- **TCP**: Noise protocol (XX handshake)
- **Peer Authentication**: Ed25519 signatures

### Content Verification

- **CID**: Self-verifying content addresses
- **Multihash**: Cryptographic content hashing
- **Provider Verification**: Signature-based

### Denial of Service Protection

- **Connection Limits**: Per-peer and global
- **Rate Limiting**: Query and bandwidth limits
- **Peer Scoring**: Identify and prune bad actors
- **Circuit Breaker**: Fast-fail on repeated failures

### Privacy

- **NAT Traversal**: Minimal information disclosure
- **DHT Privacy**: No content stored in DHT (only CIDs)
- **Relay Privacy**: End-to-end encryption maintained

## Future Enhancements

### Planned Features

- **Geographic Routing**: Optimize by physical location
- **QUIC Multipath**: Aggregate multiple network paths
- **Tor Integration**: Privacy-preserving networking
- **Mobile Optimization**: Pause/resume, background mode
- **Custom DHT**: Pluggable routing algorithms

### Research Directions

- **ML-based Routing**: Predict best peers using learning
- **Hybrid Consensus**: Combine DHT with blockchain
- **Quantum-Resistant**: Post-quantum cryptography
- **Edge Computing**: Optimize for IoT and edge devices

## References

- [libp2p Specification](https://github.com/libp2p/specs)
- [Kademlia Paper](https://pdos.csail.mit.edu/~petar/papers/maymounkov-kademlia-lncs.pdf)
- [GossipSub Specification](https://github.com/libp2p/specs/tree/master/pubsub/gossipsub)
- [QUIC Protocol](https://www.rfc-editor.org/rfc/rfc9000.html)
- [LSH for Approximate Nearest Neighbor](https://web.mit.edu/andoni/www/LSH/)
