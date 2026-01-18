# ipfrs-transport Configuration Guide

Complete reference for all configuration parameters in the ipfrs-transport crate.

## Table of Contents

1. [Bitswap Configuration](#bitswap-configuration)
2. [TensorSwap Configuration](#tensorswap-configuration)
3. [Want List Configuration](#want-list-configuration)
4. [Peer Management Configuration](#peer-management-configuration)
5. [QUIC Transport Configuration](#quic-transport-configuration)
6. [Session Management Configuration](#session-management-configuration)
7. [Bandwidth Throttling Configuration](#bandwidth-throttling-configuration)
8. [Multicast Configuration](#multicast-configuration)
9. [NAT Traversal Configuration](#nat-traversal-configuration)
10. [Partition Detection Configuration](#partition-detection-configuration)

---

## Bitswap Configuration

### `BitswapConfig`

Controls the behavior of the Bitswap protocol implementation.

```rust
pub struct BitswapConfig {
    pub max_want_list_size: usize,
    pub max_concurrent_requests: usize,
    pub request_timeout: Duration,
    pub enable_have_messages: bool,
}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_want_list_size` | `usize` | 256 | Maximum number of CIDs in the want list |
| `max_concurrent_requests` | `usize` | 128 | Maximum number of concurrent block requests |
| `request_timeout` | `Duration` | 30s | Timeout for individual block requests |
| `enable_have_messages` | `bool` | true | Enable HAVE/DONT_HAVE notifications |

#### Usage Example

```rust
use ipfrs_transport::BitswapConfig;
use std::time::Duration;

let config = BitswapConfig {
    max_want_list_size: 512,
    max_concurrent_requests: 256,
    request_timeout: Duration::from_secs(60),
    enable_have_messages: true,
};
```

#### Tuning Recommendations

- **Low Latency**: Reduce `request_timeout` to 10s, increase `max_concurrent_requests`
- **High Throughput**: Increase `max_want_list_size` to 1024
- **Constrained Resources**: Reduce `max_concurrent_requests` to 32

---

## TensorSwap Configuration

### `TensorSwapConfig`

Configuration for tensor-specific block exchange optimizations.

```rust
pub struct TensorSwapConfig {
    pub chunk_size: usize,
    pub max_concurrent_streams: usize,
    pub enable_backpressure: bool,
    pub dependency_priority_boost: i32,
    pub critical_priority_boost: i32,
}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunk_size` | `usize` | 1048576 (1 MB) | Size of tensor chunks in bytes |
| `max_concurrent_streams` | `usize` | 64 | Maximum concurrent tensor streams |
| `enable_backpressure` | `bool` | true | Enable flow control backpressure |
| `dependency_priority_boost` | `i32` | 10 | Priority boost for dependency blocks |
| `critical_priority_boost` | `i32` | 100 | Priority boost for critical tensors |

#### Usage Example

```rust
use ipfrs_transport::TensorSwapConfig;

let config = TensorSwapConfig {
    chunk_size: 512 * 1024, // 512 KB chunks
    max_concurrent_streams: 128,
    enable_backpressure: true,
    dependency_priority_boost: 20,
    critical_priority_boost: 200,
};
```

#### Tuning Recommendations

- **Large Tensors**: Increase `chunk_size` to 4 MB for reduced overhead
- **Many Small Tensors**: Decrease `chunk_size` to 256 KB
- **Training Workloads**: Increase `critical_priority_boost` to 500

---

## Want List Configuration

### `WantListConfig`

Configuration for the priority queue-based want list.

```rust
pub struct WantListConfig {
    pub max_wants: usize,
    pub request_timeout: Duration,
    pub enable_dedup: bool,
    pub cleanup_interval: Duration,
}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_wants` | `usize` | 1000 | Maximum number of wanted blocks |
| `request_timeout` | `Duration` | 30s | Timeout before retrying a request |
| `enable_dedup` | `bool` | true | Deduplicate identical CID requests |
| `cleanup_interval` | `Duration` | 10s | Interval for cleaning up expired wants |

#### Priority Levels

```rust
pub enum Priority {
    Low = 0,
    Normal = 10,
    High = 20,
    Urgent = 30,
    Critical = 40,
}
```

#### Usage Example

```rust
use ipfrs_transport::{WantListConfig, WantList, Priority};
use std::time::Duration;

let config = WantListConfig {
    max_wants: 2000,
    request_timeout: Duration::from_secs(60),
    enable_dedup: true,
    cleanup_interval: Duration::from_secs(5),
};

let want_list = WantList::new(config);
```

---

## Peer Management Configuration

### `PeerScoringConfig`

Configuration for peer scoring and selection.

```rust
pub struct PeerScoringConfig {
    pub latency_weight: f64,
    pub bandwidth_weight: f64,
    pub reliability_weight: f64,
    pub debt_weight: f64,
    pub score_decay_factor: f64,
    pub min_score_threshold: f64,
}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `latency_weight` | `f64` | 0.3 | Weight for latency in score (0.0-1.0) |
| `bandwidth_weight` | `f64` | 0.3 | Weight for bandwidth in score (0.0-1.0) |
| `reliability_weight` | `f64` | 0.3 | Weight for reliability in score (0.0-1.0) |
| `debt_weight` | `f64` | 0.1 | Weight for debt ratio in score (0.0-1.0) |
| `score_decay_factor` | `f64` | 0.95 | Score decay per interval (0.0-1.0) |
| `min_score_threshold` | `f64` | 0.1 | Minimum score before blacklisting |

### `RetryConfig`

Configuration for retry logic with exponential backoff.

```rust
pub struct RetryConfig {
    pub max_retries: usize,
    pub base_delay: Duration,
    pub max_delay: Duration,
    pub jitter_percent: u32,
}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_retries` | `usize` | 3 | Maximum number of retry attempts |
| `base_delay` | `Duration` | 100ms | Initial retry delay |
| `max_delay` | `Duration` | 30s | Maximum retry delay |
| `jitter_percent` | `u32` | 20 | Random jitter percentage (0-100) |

### `CircuitBreakerConfig`

Configuration for circuit breaker pattern.

```rust
pub struct CircuitBreakerConfig {
    pub failure_threshold: usize,
    pub timeout: Duration,
    pub half_open_max_requests: usize,
}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `failure_threshold` | `usize` | 5 | Failures before opening circuit |
| `timeout` | `Duration` | 10s | Time before trying half-open |
| `half_open_max_requests` | `usize` | 1 | Test requests in half-open state |

---

## QUIC Transport Configuration

### `QuicConfig`

Configuration for QUIC transport layer.

```rust
pub struct QuicConfig {
    pub max_concurrent_streams: u64,
    pub initial_window: u64,
    pub max_window: u64,
    pub enable_0rtt: bool,
    pub keep_alive_interval: Duration,
    pub idle_timeout: Duration,
}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_concurrent_streams` | `u64` | 256 | Maximum concurrent QUIC streams |
| `initial_window` | `u64` | 10485760 (10 MB) | Initial congestion window |
| `max_window` | `u64` | 104857600 (100 MB) | Maximum congestion window |
| `enable_0rtt` | `bool` | true | Enable 0-RTT connection establishment |
| `keep_alive_interval` | `Duration` | 5s | Keep-alive ping interval |
| `idle_timeout` | `Duration` | 30s | Connection idle timeout |

#### Usage Example

```rust
use ipfrs_transport::QuicConfig;
use std::time::Duration;

let config = QuicConfig {
    max_concurrent_streams: 512,
    initial_window: 20 * 1024 * 1024, // 20 MB
    max_window: 200 * 1024 * 1024,    // 200 MB
    enable_0rtt: true,
    keep_alive_interval: Duration::from_secs(10),
    idle_timeout: Duration::from_secs(60),
};
```

#### Tuning Recommendations

- **High Bandwidth**: Increase `initial_window` and `max_window`
- **Unstable Networks**: Decrease `idle_timeout` for faster detection
- **Many Small Transfers**: Increase `max_concurrent_streams`

---

## Session Management Configuration

### `SessionConfig`

Configuration for managing grouped block requests.

```rust
pub struct SessionConfig {
    pub max_sessions: usize,
    pub session_timeout: Duration,
    pub max_blocks_per_session: usize,
    pub enable_progress_tracking: bool,
}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_sessions` | `usize` | 100 | Maximum concurrent sessions |
| `session_timeout` | `Duration` | 300s (5 min) | Session inactivity timeout |
| `max_blocks_per_session` | `usize` | 1000 | Maximum blocks per session |
| `enable_progress_tracking` | `bool` | true | Track session progress |

---

## Bandwidth Throttling Configuration

### `BandwidthConfig`

Configuration for rate limiting and QoS.

```rust
pub struct BandwidthConfig {
    pub max_bytes_per_sec: u64,
    pub burst_capacity: u64,
    pub enable_qos: bool,
}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_bytes_per_sec` | `u64` | 10485760 (10 MB/s) | Maximum bandwidth in bytes/sec |
| `burst_capacity` | `u64` | 20971520 (20 MB) | Burst token capacity |
| `enable_qos` | `bool` | true | Enable QoS priority handling |

### QoS Priority Levels

```rust
pub enum QosPriority {
    BestEffort = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}
```

#### Usage Example

```rust
use ipfrs_transport::{BandwidthConfig, BandwidthThrottle, QosPriority};

let config = BandwidthConfig {
    max_bytes_per_sec: 50 * 1024 * 1024, // 50 MB/s
    burst_capacity: 100 * 1024 * 1024,   // 100 MB
    enable_qos: true,
};

let throttle = BandwidthThrottle::new(config);
```

---

## Multicast Configuration

### `MulticastConfig`

Configuration for multicast block announcements.

```rust
pub struct MulticastConfig {
    pub max_batch_size: usize,
    pub batch_interval: Duration,
    pub max_queue_size: usize,
    pub subscription_ttl: Duration,
    pub use_gossip: bool,
    pub gossip_fanout: usize,
}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_batch_size` | `usize` | 50 | Max announcements per batch |
| `batch_interval` | `Duration` | 100ms | Batching interval |
| `max_queue_size` | `usize` | 10000 | Max queued announcements |
| `subscription_ttl` | `Duration` | 3600s (1 hour) | Subscription time-to-live |
| `use_gossip` | `bool` | true | Use gossip-style broadcast |
| `gossip_fanout` | `usize` | 6 | Peers to forward to in gossip |

#### Tuning Recommendations

- **High Fan-out**: Disable gossip, increase `max_batch_size`
- **Low Overhead**: Enable gossip, reduce `gossip_fanout` to 3
- **High Throughput**: Increase `batch_interval` to 500ms

---

## NAT Traversal Configuration

### `NatTraversalConfig`

Configuration for NAT hole punching and traversal.

```rust
pub struct NatTraversalConfig {
    pub stun_config: Option<StunConfig>,
    pub turn_config: Option<TurnConfig>,
    pub enable_upnp: bool,
    pub candidate_timeout: Duration,
    pub check_interval: Duration,
}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stun_config` | `Option<StunConfig>` | None | STUN server configuration |
| `turn_config` | `Option<TurnConfig>` | None | TURN relay configuration |
| `enable_upnp` | `bool` | true | Enable UPnP port mapping |
| `candidate_timeout` | `Duration` | 5s | Candidate gathering timeout |
| `check_interval` | `Duration` | 1s | Connectivity check interval |

### `StunConfig`

```rust
pub struct StunConfig {
    pub servers: Vec<String>,
    pub timeout: Duration,
}
```

### `TurnConfig`

```rust
pub struct TurnConfig {
    pub servers: Vec<String>,
    pub username: String,
    pub password: String,
}
```

---

## Partition Detection Configuration

### `PartitionConfig`

Configuration for network partition detection.

```rust
pub struct PartitionConfig {
    pub failure_threshold: usize,
    pub probe_interval: Duration,
    pub recovery_grace_period: Duration,
    pub enable_auto_recovery: bool,
}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `failure_threshold` | `usize` | 5 | Failures before marking partitioned |
| `probe_interval` | `Duration` | 10s | Health probe interval |
| `recovery_grace_period` | `Duration` | 30s | Grace period after recovery |
| `enable_auto_recovery` | `bool` | true | Auto-recover from partitions |

---

## Best Practices

### Low Latency Scenario

```rust
// Optimized for low latency
let bitswap_config = BitswapConfig {
    request_timeout: Duration::from_secs(10),
    max_concurrent_requests: 256,
    ..Default::default()
};

let quic_config = QuicConfig {
    enable_0rtt: true,
    keep_alive_interval: Duration::from_secs(3),
    ..Default::default()
};
```

### High Throughput Scenario

```rust
// Optimized for high throughput
let tensorswap_config = TensorSwapConfig {
    chunk_size: 4 * 1024 * 1024, // 4 MB chunks
    max_concurrent_streams: 128,
    ..Default::default()
};

let quic_config = QuicConfig {
    initial_window: 50 * 1024 * 1024,  // 50 MB
    max_window: 500 * 1024 * 1024,     // 500 MB
    max_concurrent_streams: 512,
    ..Default::default()
};
```

### Constrained Resources Scenario

```rust
// Optimized for low memory/CPU
let bitswap_config = BitswapConfig {
    max_want_list_size: 128,
    max_concurrent_requests: 32,
    ..Default::default()
};

let bandwidth_config = BandwidthConfig {
    max_bytes_per_sec: 1024 * 1024, // 1 MB/s
    burst_capacity: 2 * 1024 * 1024, // 2 MB
    ..Default::default()
};
```

---

## Performance Targets

| Metric | Target | Configuration |
|--------|--------|---------------|
| Block request latency (local) | < 10ms | Default QuicConfig |
| Throughput (single peer) | > 100 MB/s | High throughput QuicConfig |
| Concurrent requests | 1000+ | Increase max_concurrent_requests |
| Memory per peer | < 10 MB | Default configs |

---

## Monitoring and Observability

All configurations expose statistics through their `stats()` methods:

```rust
// Get statistics
let bitswap_stats = bitswap_exchange.stats().await;
let peer_stats = peer_manager.stats().await;
let session_stats = session_manager.stats().await;

println!("Blocks sent: {}", bitswap_stats.blocks_sent);
println!("Active peers: {}", peer_stats.active_peers);
println!("Completed sessions: {}", session_stats.completed_sessions);
```
