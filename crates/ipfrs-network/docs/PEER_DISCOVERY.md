# Peer Discovery Guide

This document explains how peer discovery works in IPFRS Network and how to configure it for different scenarios.

## Overview

IPFRS Network uses multiple discovery mechanisms to find and connect to peers:

1. **Bootstrap Nodes** - Static list of known peers
2. **Kademlia DHT** - Distributed peer discovery
3. **mDNS** - Local network discovery
4. **Peer Exchange** - Learn peers from existing connections

## Discovery Mechanisms

### 1. Bootstrap Nodes

Bootstrap nodes are your entry point to the network.

#### Configuration

```rust
use ipfrs_network::{NetworkConfig, NetworkNode};

let config = NetworkConfig {
    bootstrap_peers: vec![
        ("/ip4/104.131.131.82/tcp/4001/p2p/QmaCpDMGvV2BGHeYERUEnRQAwe3N8SzbUtfsmvsqQLuvuJ".to_string()),
        ("/dnsaddr/bootstrap.libp2p.io/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN".to_string()),
    ],
    ..Default::default()
};

let node = NetworkNode::new(config)?;
```

#### Bootstrap Process

1. **Initial Connection**
   - Node starts and loads bootstrap peer list
   - Attempts to connect to each bootstrap peer
   - Uses exponential backoff for failed connections

2. **Routing Table Population**
   - Once connected, queries bootstrap nodes for nearby peers
   - Fills routing table with discovered peers
   - Continues until routing table is healthy

3. **Health Monitoring**
   - Tracks connection success rate
   - Implements circuit breaker for failing peers
   - Automatically retries after cooldown period

#### Best Practices

- **Multiple Bootstrap Peers**: Use at least 3-5 bootstrap nodes
- **Geographic Distribution**: Choose nodes in different regions
- **Reliability**: Select high-uptime nodes
- **Custom Network**: For private networks, use your own bootstrap nodes

#### Example: Custom Bootstrap Network

```rust
let config = NetworkConfig {
    bootstrap_peers: vec![
        "/ip4/192.168.1.10/tcp/4001/p2p/12D3KooW...".to_string(),
        "/ip4/192.168.1.11/tcp/4001/p2p/12D3KooW...".to_string(),
        "/ip4/192.168.1.12/tcp/4001/p2p/12D3KooW...".to_string(),
    ],
    enable_mdns: true,  // Also discover local peers
    ..Default::default()
};
```

### 2. Kademlia DHT

The DHT provides distributed peer discovery without central authority.

#### How It Works

**Routing Table Structure:**
```
XOR Distance from Local PeerID
├── Bucket 0:  Distance 2^0  to 2^1   (closest)
├── Bucket 1:  Distance 2^1  to 2^2
├── Bucket 2:  Distance 2^2  to 2^3
...
└── Bucket 255: Distance 2^255 to 2^256 (farthest)
```

Each bucket stores up to 20 peers (k-bucket size).

#### Discovery Process

1. **Random Walk**
   ```rust
   // Automatically performed by Kademlia
   // Discovers peers by querying random keys
   ```

2. **FIND_NODE Queries**
   ```rust
   // Find peers close to a specific PeerID
   let peers = node.find_node(target_peer_id).await?;
   ```

3. **Provider Queries**
   ```rust
   // Discover content providers (also discovers peers)
   let providers = node.find_providers(&cid).await?;
   ```

#### Configuration

```rust
use ipfrs_network::KademliaConfig;

let kad_config = KademliaConfig {
    // Number of concurrent queries
    alpha: 3,

    // How many nodes to replicate to
    replication_factor: 20,

    // Query timeout
    query_timeout: Duration::from_secs(60),

    // K-bucket size
    k_bucket_size: 20,
};
```

#### Monitoring DHT Health

```rust
// Check routing table
let info = node.get_routing_table_info();
println!("Routing table has {} peers across {} buckets",
    info.total_peers, info.num_buckets);

// Check DHT health
let health = node.get_dht_health();
match health.status {
    DhtHealthStatus::Healthy => println!("DHT is healthy"),
    DhtHealthStatus::Degraded => println!("DHT is degraded"),
    DhtHealthStatus::Unhealthy => println!("DHT is unhealthy"),
    _ => println!("DHT status unknown"),
}
```

### 3. mDNS (Multicast DNS)

Discover peers on the local network without infrastructure.

#### When to Use mDNS

- **Local Development**: Testing on a single machine or LAN
- **IoT Devices**: Devices on the same network segment
- **Private Networks**: LANs without internet access
- **Fast Discovery**: Sub-second peer discovery locally

#### Configuration

```rust
let config = NetworkConfig {
    enable_mdns: true,  // Enable mDNS discovery
    listen_addrs: vec![
        "/ip4/0.0.0.0/tcp/0".to_string(),  // Listen on all interfaces
    ],
    ..Default::default()
};
```

#### How It Works

1. Node broadcasts service announcement: `_ipfrs._udp.local`
2. Other nodes on the network receive the announcement
3. Nodes automatically connect to discovered peers
4. Works on IPv4 and IPv6

#### Limitations

- **Local Only**: Doesn't work across routers (by design)
- **Network Permission**: May require firewall exceptions
- **Broadcast Storm**: Many peers can cause network congestion
- **Security**: No authentication (trust local network)

#### Troubleshooting mDNS

```bash
# Check if mDNS is working (Linux)
avahi-browse -a

# Check firewall (Linux)
sudo ufw status
sudo ufw allow 5353/udp  # mDNS port

# Check if service is announced
dns-sd -B _ipfrs._udp .
```

### 4. Peer Exchange

Learn about peers from existing connections.

#### Identify Protocol

Every connection exchanges peer information:

```
Connection Established
   ↓
Identify Handshake
   ↓
Exchange:
   - PeerID
   - Listen Addresses
   - Protocol Support
   - Agent Version
```

#### Example

```rust
// This happens automatically
// But you can query peer info:
let peer_info = node.get_peer_info(&peer_id);
if let Some(info) = peer_info {
    println!("Peer addresses: {:?}", info.addrs);
    println!("Protocols: {:?}", info.protocols);
}
```

## Discovery Strategies by Scenario

### Scenario 1: Public Network Node

**Goal**: Join the public IPFS/IPFRS network

```rust
let config = NetworkConfig {
    // Use public bootstrap nodes
    bootstrap_peers: vec![
        "/dnsaddr/bootstrap.libp2p.io/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN".to_string(),
        "/dnsaddr/bootstrap.libp2p.io/p2p/QmQCU2EcMqAqQPR2i9bChDtGNJchTbq5TbXJJ16u19uLTa".to_string(),
    ],
    enable_mdns: false,  // Not useful for public network
    enable_nat_traversal: true,  // Important for home networks
    ..Default::default()
};
```

### Scenario 2: Private Network

**Goal**: Isolated network for organization

```rust
let config = NetworkConfig {
    // Private bootstrap nodes
    bootstrap_peers: vec![
        "/ip4/10.0.0.10/tcp/4001/p2p/12D3KooW...".to_string(),
        "/ip4/10.0.0.11/tcp/4001/p2p/12D3KooW...".to_string(),
    ],
    enable_mdns: true,  // Discover local peers
    enable_nat_traversal: false,  // Not needed in private network
    ..Default::default()
};
```

### Scenario 3: Local Development

**Goal**: Test on a single machine or LAN

```rust
let config = NetworkConfig {
    bootstrap_peers: vec![],  // No bootstrap needed
    enable_mdns: true,  // Primary discovery mechanism
    listen_addrs: vec![
        "/ip4/127.0.0.1/tcp/0".to_string(),  // Localhost
    ],
    ..Default::default()
};
```

### Scenario 4: Edge/IoT Device

**Goal**: Resource-constrained device

```rust
let config = NetworkConfig {
    bootstrap_peers: vec![
        "/ip4/gateway.local/tcp/4001/p2p/12D3KooW...".to_string(),
    ],
    enable_mdns: true,  // Discover local gateway
    connection_limits: ConnectionLimitsConfig {
        max_connections: 10,  // Limit for constrained device
        max_inbound: 5,
        max_outbound: 5,
    },
    ..Default::default()
};
```

### Scenario 5: Mobile Application

**Goal**: Handle network switches gracefully

```rust
let config = NetworkConfig {
    bootstrap_peers: vec![
        "/dns4/mobile-relay.example.com/tcp/443/wss/p2p/12D3KooW...".to_string(),
    ],
    enable_quic: true,  // QUIC supports connection migration
    enable_nat_traversal: true,
    ..Default::default()
};

// Handle network changes
// (future feature - pause/resume)
```

## Advanced Discovery

### Custom Discovery

Implement your own discovery mechanism:

```rust
use ipfrs_network::{NetworkNode, PeerId};

async fn custom_discovery(node: &NetworkNode) {
    // Example: Query central server for peer list
    let peers = fetch_peers_from_server().await;

    for peer in peers {
        if let Err(e) = node.connect(peer.id, peer.addrs).await {
            eprintln!("Failed to connect to {}: {}", peer.id, e);
        }
    }
}
```

### Semantic Peer Discovery

Discover peers with similar content:

```rust
use ipfrs_network::{SemanticDht, NamespaceId, SemanticQuery};

// Find peers hosting similar content
let semantic_dht = SemanticDht::new(config);

// Register namespace
let namespace = SemanticNamespace {
    id: NamespaceId::text(),
    dimension: 768,
    distance_metric: DistanceMetric::Cosine,
    lsh_config: LshConfig::default(),
};
semantic_dht.register_namespace(namespace)?;

// Query for similar content
let query = SemanticQuery {
    embedding: my_embedding,
    namespace: NamespaceId::text(),
    top_k: 10,
    metadata_filter: None,
    timeout: Duration::from_secs(5),
};

let results = semantic_dht.query(query)?;
// Results contain peers with semantically similar content
```

## Monitoring and Debugging

### Check Peer Count

```rust
let stats = node.stats();
println!("Connected peers: {}", stats.connected_peers);
println!("Known peers: {}", stats.known_peers);
```

### List Connected Peers

```rust
let peers = node.get_connected_peers();
for peer_id in peers {
    println!("Connected to: {}", peer_id);
}
```

### Check Bootstrap Status

```rust
let bootstrap_stats = node.get_bootstrap_stats();
println!("Bootstrap attempts: {}", bootstrap_stats.connection_attempts);
println!("Successful: {}", bootstrap_stats.successful_connections);
println!("Failed: {}", bootstrap_stats.failed_connections);
```

### Monitor Discovery Events

```rust
// Events are emitted during discovery
match event {
    NetworkEvent::PeerDiscovered { peer_id, addrs } => {
        println!("Discovered peer {} at {:?}", peer_id, addrs);
    }
    NetworkEvent::ConnectionEstablished { peer_id, endpoint } => {
        println!("Connected to {} via {:?}", peer_id, endpoint);
    }
    _ => {}
}
```

## Performance Tuning

### Optimize for Fast Discovery

```rust
let config = NetworkConfig {
    // More bootstrap peers
    bootstrap_peers: many_bootstrap_peers,

    // Aggressive DHT queries
    kademlia_config: KademliaConfig {
        alpha: 5,  // More concurrent queries
        query_timeout: Duration::from_secs(30),  // Shorter timeout
        ..Default::default()
    },
    ..Default::default()
};
```

### Optimize for Low Bandwidth

```rust
let config = NetworkConfig {
    // Fewer bootstrap peers
    bootstrap_peers: vec![single_reliable_peer],

    // Conservative DHT
    kademlia_config: KademliaConfig {
        alpha: 1,  // Sequential queries
        replication_factor: 5,  // Less replication
        ..Default::default()
    },

    // Disable mDNS (broadcast traffic)
    enable_mdns: false,
    ..Default::default()
};
```

## Common Issues

### Issue: No Peers Discovered

**Symptoms:**
- `connected_peers` stays at 0
- Bootstrap connections fail

**Solutions:**
1. Check network connectivity
   ```bash
   ping 8.8.8.8
   ```

2. Verify bootstrap peer addresses
   ```bash
   nc -zv bootstrap.libp2p.io 4001
   ```

3. Check firewall settings
   ```bash
   sudo ufw status
   sudo ufw allow 4001/tcp
   ```

4. Enable mDNS for local discovery
   ```rust
   config.enable_mdns = true;
   ```

### Issue: Slow Peer Discovery

**Symptoms:**
- Takes minutes to find peers
- DHT queries timeout

**Solutions:**
1. Add more bootstrap peers
2. Increase alpha (concurrent queries)
3. Check network latency
4. Verify NAT traversal is working

### Issue: Too Many Connections

**Symptoms:**
- Resource exhaustion
- Connection churn

**Solutions:**
1. Lower connection limits
   ```rust
   connection_limits: ConnectionLimitsConfig {
       max_connections: 50,  // Reduce from default 100
       ..Default::default()
   }
   ```

2. Enable connection pruning
   ```rust
   // Happens automatically based on connection value
   ```

3. Blacklist problematic peers
   ```rust
   // Use ConnectionManager's ban list
   ```

## Security Considerations

### Bootstrap Node Trust

- Bootstrap nodes see your PeerID and IP
- Use reputable bootstrap nodes
- For sensitive applications, use your own bootstrap nodes

### Discovery Privacy

- mDNS reveals presence on local network
- DHT queries reveal content interests
- Consider using Tor for privacy (future feature)

### Peer Validation

```rust
// Validate peer before connecting
async fn validate_peer(peer_id: &PeerId) -> bool {
    // Check against allowlist/denylist
    // Verify peer reputation
    // Check peer protocols
    true
}
```

## Best Practices

1. **Use Multiple Discovery Methods**
   - Combine bootstrap + DHT + mDNS for reliability

2. **Monitor Discovery Health**
   - Track connected peer count
   - Alert if below threshold

3. **Implement Retry Logic**
   - Bootstrap manager handles this automatically
   - Add application-level retries if needed

4. **Cache Successful Peers**
   - Persistence is handled automatically
   - Load cached peers on startup

5. **Test Discovery Locally**
   - Use mDNS for fast local testing
   - Then test with bootstrap nodes

6. **Document Custom Bootstrap**
   - If running your own network, document bootstrap peers clearly

## References

- [Kademlia DHT Paper](https://pdos.csail.mit.edu/~petar/papers/maymounkov-kademlia-lncs.pdf)
- [libp2p Peer Discovery](https://docs.libp2p.io/concepts/discovery-routing/overview/)
- [mDNS RFC 6762](https://tools.ietf.org/html/rfc6762)
- [IPFS Bootstrap](https://docs.ipfs.tech/concepts/nodes/#bootstrap)
