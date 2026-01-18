# NAT Traversal Troubleshooting Guide

This guide helps you diagnose and resolve NAT (Network Address Translation) traversal issues in IPFRS Network.

## Table of Contents

1. [Understanding NAT](#understanding-nat)
2. [NAT Traversal in IPFRS](#nat-traversal-in-ipfrs)
3. [Diagnosing NAT Issues](#diagnosing-nat-issues)
4. [Solutions by NAT Type](#solutions-by-nat-type)
5. [Common Problems](#common-problems)
6. [Configuration](#configuration)
7. [Advanced Troubleshooting](#advanced-troubleshooting)

## Understanding NAT

### What is NAT?

Network Address Translation (NAT) allows multiple devices on a private network to share a single public IP address. While this conserves IPv4 addresses, it makes peer-to-peer connections challenging.

### NAT Types

**1. Full Cone NAT** (Most permissive)
```
Private: 192.168.1.10:1234 → Public: 1.2.3.4:5678
Any external peer can send to 1.2.3.4:5678
✅ Easy to traverse
```

**2. Restricted Cone NAT**
```
Private: 192.168.1.10:1234 → Public: 1.2.3.4:5678
Only peers that received packets from you can reply
✅ Traversable with coordination
```

**3. Port-Restricted Cone NAT**
```
Same as Restricted Cone, but also checks source port
Only specific IP:Port combinations can reply
⚠️ Requires hole punching
```

**4. Symmetric NAT** (Most restrictive)
```
Maps different external ports for each destination
192.168.1.10:1234 → 1.2.3.4:5678 (to peer A)
192.168.1.10:1234 → 1.2.3.4:9012 (to peer B)
❌ Difficult to traverse, usually needs relay
```

## NAT Traversal in IPFRS

IPFRS uses a three-layer approach:

### 1. AutoNAT (Detection)

Determines if you're behind NAT and your external address.

**How it works:**
1. Ask remote peers to dial back to you
2. If they succeed → You're publicly reachable
3. If they fail → You're behind NAT
4. Remote peers tell you your external address

**Enable AutoNAT:**
```rust
let config = NetworkConfig {
    enable_nat_traversal: true,  // Enables AutoNAT
    ..Default::default()
};
```

### 2. DCUtR (Hole Punching)

Direct Connection Upgrade through Relay - attempts direct connection through NAT.

**How it works:**
```
You (NAT) <---relay---> Peer (NAT)
    |                      |
    |   Coordination       |
    |<------------------->|
    |                      |
    |   Simultaneous Open  |
    |<------------------->|
    |  Direct Connection!  |
```

**Requirements:**
- Both peers behind NAT
- At least one relay connection
- Symmetric UDP support
- Firewall allows outbound UDP

### 3. Circuit Relay v2 (Fallback)

If hole punching fails, use a relay server.

**How it works:**
```
You (NAT) ---> Relay Server <--- Peer (NAT)
         All traffic through relay
```

**Limitations:**
- Higher latency
- Bandwidth costs for relay
- Time-limited connections (2 minutes default)
- Data-limited connections (1 MB default)

## Diagnosing NAT Issues

### Step 1: Check NAT Status

```rust
use ipfrs_network::NetworkNode;

let node = NetworkNode::new(config)?;
node.start().await?;

// Wait for AutoNAT to complete (a few seconds)
tokio::time::sleep(Duration::from_secs(5)).await;

// Check reachability
if node.is_publicly_reachable() {
    println!("✅ Publicly reachable - no NAT issues");
} else {
    println!("⚠️  Behind NAT - traversal may be needed");

    // Get external addresses (if known)
    let addrs = node.get_external_addresses();
    if addrs.is_empty() {
        println!("❌ No external address detected");
    } else {
        println!("External addresses: {:?}", addrs);
    }
}
```

### Step 2: Check Connection Events

```rust
// Monitor connection events
match event {
    NetworkEvent::NatStatusChanged { status, external_addrs } => {
        match status {
            NatStatus::Public => {
                println!("✅ Public IP: {:?}", external_addrs);
            }
            NatStatus::Private => {
                println!("⚠️  Behind NAT");
            }
            NatStatus::Unknown => {
                println!("❓ NAT status unknown");
            }
        }
    }
    NetworkEvent::ConnectionEstablished { peer_id, endpoint } => {
        println!("Connected to {} via {:?}", peer_id, endpoint);
    }
    NetworkEvent::ConnectionFailed { peer_id, error } => {
        println!("Failed to connect to {}: {}", peer_id, error);
    }
    _ => {}
}
```

### Step 3: Test with Known Peer

```bash
# Try connecting to a known public peer
# Should see:
# - AutoNAT probe attempts
# - DCUtR synchronization (if both behind NAT)
# - Or relay connection establishment
```

## Solutions by NAT Type

### Full Cone NAT

**Status:** ✅ No special handling needed

**Verification:**
```rust
// Should connect directly
let peer_id = "12D3KooW...".parse()?;
let addrs = vec!["/ip4/1.2.3.4/tcp/4001".parse()?];
node.connect(peer_id, addrs).await?;
// Should establish direct connection
```

**If it fails:**
- Check firewall rules
- Verify peer address is correct
- Ensure port is reachable

### Restricted/Port-Restricted Cone NAT

**Status:** ✅ Works with DCUtR

**Requirements:**
- Both peers must support DCUtR
- Need relay for initial coordination
- Outbound UDP must be allowed

**Configuration:**
```rust
let config = NetworkConfig {
    enable_nat_traversal: true,  // Enables DCUtR
    enable_quic: true,  // QUIC for better NAT traversal
    ..Default::default()
};
```

**Verification:**
```rust
// Watch for DCUtR events
NetworkEvent::DcutrSyncComplete { peer_id } => {
    println!("✅ DCUtR hole punching successful with {}", peer_id);
}
```

### Symmetric NAT

**Status:** ⚠️ Requires relay

**Problem:** Different external ports per destination makes prediction impossible

**Solution:** Use Circuit Relay

```rust
let config = NetworkConfig {
    enable_nat_traversal: true,
    // Specify relay servers
    relay_servers: vec![
        "/ip4/relay.example.com/tcp/4001/p2p/12D3KooW...".to_string(),
    ],
    ..Default::default()
};
```

**Verification:**
```rust
NetworkEvent::RelayConnectionEstablished { peer_id, relay } => {
    println!("Connected to {} via relay {}", peer_id, relay);
}
```

### Double NAT

**Scenario:** Behind two layers of NAT (e.g., router behind router)

**Status:** ⚠️ Very challenging

**Solutions:**
1. **Port Forwarding:** Configure both NAT levels
2. **DMZ:** Place one router in DMZ of other
3. **VPN:** Use VPN to bypass NAT
4. **Relay:** Use Circuit Relay (recommended)

## Common Problems

### Problem 1: "Not publicly reachable"

**Symptoms:**
```
NAT status: Private
External addresses: []
No incoming connections
```

**Diagnosis:**
```rust
// Check if AutoNAT is enabled
let config = node.get_config();
if !config.enable_nat_traversal {
    println!("❌ NAT traversal disabled");
}

// Check if any protocols are working
let protocols = node.get_supported_protocols();
if !protocols.contains("/libp2p/autonat/1.0.0") {
    println!("❌ AutoNAT protocol not available");
}
```

**Solutions:**
1. Enable NAT traversal:
   ```rust
   config.enable_nat_traversal = true;
   ```

2. Add more peers for AutoNAT probing:
   ```rust
   // AutoNAT needs peers to probe from
   // Connect to more bootstrap nodes
   ```

3. Use relay as workaround:
   ```rust
   config.relay_servers = vec![...];
   ```

### Problem 2: "DCUtR synchronization failed"

**Symptoms:**
```
DCUtR sync started...
DCUtR sync failed: Timeout
Falling back to relay
```

**Diagnosis:**
```bash
# Check UDP connectivity
nc -u -z -v peer_ip peer_port

# Check if QUIC is enabled
# QUIC uses UDP and works better with NAT
```

**Solutions:**
1. Ensure QUIC is enabled:
   ```rust
   config.enable_quic = true;
   ```

2. Check firewall allows outbound UDP:
   ```bash
   sudo ufw allow out to any port 4001 proto udp
   ```

3. Increase synchronization timeout:
   ```rust
   // Currently hardcoded, may need protocol update
   ```

4. Use relay if DCUtR continues to fail:
   ```rust
   // Relay is automatic fallback
   ```

### Problem 3: "Relay connection limited"

**Symptoms:**
```
Connected via relay
Connection closed after 2 minutes
Data limit reached (1 MB)
```

**Explanation:**
Circuit Relay v2 has built-in limits:
- **Time limit:** 2 minutes per connection
- **Data limit:** 1 MB per connection
- **Purpose:** Prevent relay abuse

**Solutions:**
1. For short-lived connections: This is fine
   ```rust
   // Fetch small content via relay
   let data = node.get_block(&cid).await?;
   ```

2. For long-lived connections: Need direct connection
   ```rust
   // Keep trying DCUtR
   // Or use different network setup
   ```

3. Run your own relay with custom limits:
   ```rust
   // Configure relay server with higher limits
   // (relay server configuration)
   ```

### Problem 4: "No relay servers available"

**Symptoms:**
```
DCUtR failed
No relay connection possible
Peer unreachable
```

**Diagnosis:**
```rust
// Check if any relay connections exist
let stats = node.stats();
if stats.relay_connections == 0 {
    println!("❌ No relay connections");
}
```

**Solutions:**
1. Add public relay servers:
   ```rust
   config.relay_servers = vec![
       "/ip4/relay.libp2p.io/tcp/4001/p2p/12D3KooW...".to_string(),
   ];
   ```

2. Run your own relay:
   ```bash
   # Start relay server
   ipfrs-relay --port 4001
   ```

3. Connect to relay-capable peers:
   ```rust
   // Connect to bootstrap nodes that support relay
   ```

### Problem 5: "Firewall blocking connections"

**Symptoms:**
```
Connection attempts timeout
No inbound connections
AutoNAT probes fail
```

**Diagnosis:**
```bash
# Check if port is open
sudo netstat -tulpn | grep 4001

# Test with external tool
nmap -p 4001 your_public_ip

# Check firewall rules
sudo ufw status
sudo iptables -L
```

**Solutions:**
1. Allow IPFRS ports in firewall:
   ```bash
   # UFW (Ubuntu/Debian)
   sudo ufw allow 4001/tcp
   sudo ufw allow 4001/udp

   # Firewalld (RHEL/CentOS)
   sudo firewall-cmd --add-port=4001/tcp --permanent
   sudo firewall-cmd --add-port=4001/udp --permanent
   sudo firewall-cmd --reload

   # iptables
   sudo iptables -A INPUT -p tcp --dport 4001 -j ACCEPT
   sudo iptables -A INPUT -p udp --dport 4001 -j ACCEPT
   ```

2. Configure router port forwarding:
   ```
   External Port: 4001 → Internal IP: 192.168.1.10:4001
   ```

3. Use UPnP (if router supports):
   ```rust
   // Not yet implemented, future feature
   config.enable_upnp = true;
   ```

## Configuration

### Minimal Configuration (Public Server)

```rust
let config = NetworkConfig {
    listen_addrs: vec![
        "/ip4/0.0.0.0/tcp/4001".to_string(),
        "/ip4/0.0.0.0/udp/4001/quic-v1".to_string(),
    ],
    enable_nat_traversal: false,  // Not behind NAT
    ..Default::default()
};
```

### Home User Configuration

```rust
let config = NetworkConfig {
    listen_addrs: vec![
        "/ip4/0.0.0.0/tcp/0".to_string(),  // Random port
        "/ip4/0.0.0.0/udp/0/quic-v1".to_string(),
    ],
    enable_nat_traversal: true,  // Likely behind NAT
    enable_quic: true,  // Better NAT traversal
    relay_servers: vec![
        "/dnsaddr/relay.libp2p.io/p2p/12D3KooW...".to_string(),
    ],
    ..Default::default()
};
```

### Corporate Network Configuration

```rust
let config = NetworkConfig {
    listen_addrs: vec![
        "/ip4/0.0.0.0/tcp/0".to_string(),
    ],
    enable_nat_traversal: true,
    enable_quic: false,  // May be blocked by corporate firewall
    relay_servers: vec![
        "/ip4/internal-relay.corp.com/tcp/4001/p2p/12D3KooW...".to_string(),
    ],
    ..Default::default()
};
```

### Mobile Configuration

```rust
let config = NetworkConfig {
    listen_addrs: vec![
        "/ip4/0.0.0.0/udp/0/quic-v1".to_string(),  // QUIC only
    ],
    enable_nat_traversal: true,
    enable_quic: true,  // QUIC supports connection migration
    relay_servers: vec![
        "/dns4/mobile-relay.example.com/tcp/443/wss/p2p/12D3KooW...".to_string(),
    ],
    ..Default::default()
};
```

## Advanced Troubleshooting

### Enable Debug Logging

```rust
// Set log level
std::env::set_var("RUST_LOG", "ipfrs_network=debug,libp2p=debug");
env_logger::init();

// Or use tracing
use tracing_subscriber;
tracing_subscriber::fmt()
    .with_max_level(tracing::Level::DEBUG)
    .init();
```

### Capture Network Traffic

```bash
# Capture all IPFRS traffic
sudo tcpdump -i any -w ipfrs.pcap port 4001

# Analyze with Wireshark
wireshark ipfrs.pcap
```

### Test NAT Type Manually

```rust
// Use STUN to determine NAT type
use stun_client::StunClient;

let client = StunClient::new("stun.l.google.com:19302");
let nat_type = client.get_nat_type().await?;
println!("NAT type: {:?}", nat_type);
```

### Test Relay Manually

```bash
# Connect to relay server
telnet relay.libp2p.io 4001

# Or use libp2p-relay-cli
libp2p-relay-cli connect --relay /ip4/relay.libp2p.io/tcp/4001/p2p/12D3KooW...
```

### Check External Address

```bash
# Query what external IP you're seen as
curl https://api.ipify.org
curl https://ifconfig.me

# Compare with AutoNAT result
```

## Performance Considerations

### Direct vs Relay Latency

| Connection Type | Typical Latency | Bandwidth |
|----------------|----------------|-----------|
| Direct | 10-50ms | Full (1 Gbps+) |
| DCUtR | 10-50ms | Full (1 Gbps+) |
| Relay | +50-200ms | Limited |

### Bandwidth Costs

**Relay bandwidth:**
- Counted twice (to relay + relay to peer)
- Limited by relay operator
- May have monetary cost for relay operator

**Direct bandwidth:**
- Point-to-point, no intermediary
- Limited only by connection speed

## Security Considerations

### Relay Trust

- Relay servers can see traffic metadata (but not content due to encryption)
- Use trusted relays or run your own
- End-to-end encryption preserved through relay

### Firewall Recommendations

```bash
# Allow outbound on all ports (for NAT traversal)
sudo ufw default allow outgoing

# Allow inbound only on IPFRS port
sudo ufw default deny incoming
sudo ufw allow 4001

# Or use connection tracking
sudo ufw allow out to any
sudo ufw allow in from any established
```

### NAT Slipstreaming Attack

- Attack that exploits ALG (Application Layer Gateway)
- Mitigation: Use encrypted transports (QUIC, Noise)
- IPFRS uses encryption by default

## Best Practices

1. **Always enable NAT traversal for end users**
   ```rust
   enable_nat_traversal: true
   ```

2. **Use QUIC when possible**
   - Better NAT traversal
   - Connection migration
   - Built-in encryption

3. **Provide relay servers**
   - At least 2-3 relay servers
   - Geographically distributed
   - High bandwidth and uptime

4. **Monitor NAT status**
   ```rust
   if !node.is_publicly_reachable() {
       log::warn!("Behind NAT, may have connectivity issues");
   }
   ```

5. **Document network requirements**
   - "Requires outbound TCP/UDP on port 4001"
   - "May use relay servers if behind strict NAT"

6. **Test on various networks**
   - Home broadband
   - Corporate network
   - Mobile network
   - Public WiFi

## References

- [NAT Types Explained](https://en.wikipedia.org/wiki/Network_address_translation)
- [libp2p AutoNAT](https://github.com/libp2p/specs/blob/master/autonat/README.md)
- [libp2p DCUtR](https://github.com/libp2p/specs/blob/master/relay/DCUtR.md)
- [Circuit Relay v2](https://github.com/libp2p/specs/blob/master/relay/circuit-v2.md)
- [STUN RFC 5389](https://tools.ietf.org/html/rfc5389)
- [ICE RFC 8445](https://tools.ietf.org/html/rfc8445)
