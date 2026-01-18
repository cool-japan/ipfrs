# ipfrs-transport Tuning Guide

Comprehensive guide for optimizing ipfrs-transport for different use cases and network conditions.

## Table of Contents

1. [Quick Start Profiles](#quick-start-profiles)
2. [Scenario-Based Tuning](#scenario-based-tuning)
3. [Network Condition Optimization](#network-condition-optimization)
4. [Resource Constraint Tuning](#resource-constraint-tuning)
5. [Performance Monitoring](#performance-monitoring)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start Profiles

### Low Latency Profile

**Use Case**: Real-time inference, interactive applications

```rust
use ipfrs_transport::*;
use std::time::Duration;

fn low_latency_config() -> TransportConfig {
    TransportConfig {
        bitswap: BitswapConfig {
            max_concurrent_requests: 256,
            request_timeout: Duration::from_secs(5),
            enable_have_messages: false, // Skip for lower latency
            ..Default::default()
        },

        quic: QuicConfig {
            enable_0rtt: true,
            keep_alive_interval: Duration::from_secs(2),
            idle_timeout: Duration::from_secs(10),
            max_concurrent_streams: 512,
            ..Default::default()
        },

        want_list: WantListConfig {
            default_timeout: Duration::from_secs(5),
            max_retries: 2,
            ..Default::default()
        },

        peer_scoring: PeerScoringConfig {
            latency_weight: 0.7, // Prioritize low latency
            bandwidth_weight: 0.2,
            reliability_weight: 0.1,
            ..Default::default()
        },
    }
}
```

**Expected Performance**:
- Block request latency: < 5ms (local network)
- Connection establishment: < 50ms (with 0-RTT)
- Throughput: 50-100 MB/s

---

### High Throughput Profile

**Use Case**: Model training, large dataset transfers

```rust
fn high_throughput_config() -> TransportConfig {
    TransportConfig {
        bitswap: BitswapConfig {
            max_want_list_size: 2048,
            max_concurrent_requests: 512,
            request_timeout: Duration::from_secs(30),
            ..Default::default()
        },

        tensorswap: TensorSwapConfig {
            chunk_size: 4 * 1024 * 1024, // 4 MB chunks
            max_concurrent_streams: 256,
            enable_backpressure: true,
            ..Default::default()
        },

        quic: QuicConfig {
            initial_window: 50 * 1024 * 1024,  // 50 MB
            max_window: 500 * 1024 * 1024,     // 500 MB
            max_concurrent_streams: 1024,
            ..Default::default()
        },

        bandwidth: BandwidthConfig {
            global_max_bytes_per_sec: None, // Unlimited
            token_bucket_capacity: 100 * 1024 * 1024, // 100 MB burst
            enable_qos: true,
        },
    }
}
```

**Expected Performance**:
- Throughput: > 500 MB/s (10 Gbps network)
- Concurrent blocks: 1000+
- Memory usage: 50-100 MB

---

### Balanced Profile

**Use Case**: General purpose, mixed workloads

```rust
fn balanced_config() -> TransportConfig {
    TransportConfig {
        bitswap: BitswapConfig {
            max_want_list_size: 512,
            max_concurrent_requests: 128,
            request_timeout: Duration::from_secs(30),
            enable_have_messages: true,
        },

        quic: QuicConfig {
            max_concurrent_streams: 256,
            initial_window: 10 * 1024 * 1024,
            max_window: 100 * 1024 * 1024,
            enable_0rtt: true,
            ..Default::default()
        },

        peer_scoring: PeerScoringConfig {
            latency_weight: 0.3,
            bandwidth_weight: 0.3,
            reliability_weight: 0.3,
            debt_weight: 0.1,
            ..Default::default()
        },
    }
}
```

**Expected Performance**:
- Block request latency: < 10ms (local), < 100ms (WAN)
- Throughput: 100-200 MB/s
- Memory usage: < 10 MB per peer

---

### Edge/IoT Profile

**Use Case**: Resource-constrained devices, unstable networks

```rust
fn edge_config() -> TransportConfig {
    TransportConfig {
        bitswap: BitswapConfig {
            max_want_list_size: 64,
            max_concurrent_requests: 16,
            request_timeout: Duration::from_secs(60),
            ..Default::default()
        },

        quic: QuicConfig {
            max_concurrent_streams: 32,
            initial_window: 1 * 1024 * 1024,  // 1 MB
            max_window: 10 * 1024 * 1024,     // 10 MB
            keep_alive_interval: Duration::from_secs(10),
            idle_timeout: Duration::from_secs(30),
            ..Default::default()
        },

        bandwidth: BandwidthConfig {
            global_max_bytes_per_sec: Some(1024 * 1024), // 1 MB/s
            token_bucket_capacity: 2 * 1024 * 1024,
            enable_qos: true,
        },

        session: SessionConfig {
            max_concurrent_blocks: 32,
            default_priority: Priority::Normal,
            ..Default::default()
        },
    }
}
```

**Expected Performance**:
- Block request latency: < 500ms
- Throughput: 1-5 MB/s
- Memory usage: < 5 MB total

---

## Scenario-Based Tuning

### Distributed Training

**Characteristics**:
- Large gradient exchanges
- Deadline-sensitive transfers
- Periodic synchronization points

**Recommended Settings**:

```rust
let config = TransportConfig {
    tensorswap: TensorSwapConfig {
        chunk_size: 2 * 1024 * 1024,
        max_concurrent_streams: 128,
        dependency_priority_boost: 50,
        critical_priority_boost: 300,
        enable_backpressure: true,
    },

    want_list: WantListConfig {
        default_timeout: Duration::from_secs(10),
        enable_deadline_boosting: true,
        max_retries: 3,
        ..Default::default()
    },

    session: SessionConfig {
        default_priority: Priority::High,
        progress_notifications: true,
        timeout: Duration::from_secs(120),
        ..Default::default()
    },
};
```

**Key Tuning Parameters**:
- `critical_priority_boost`: 300 (ensure deadline adherence)
- `enable_deadline_boosting`: true
- `chunk_size`: 2 MB (balance overhead vs parallelism)

---

### Model Serving / Inference

**Characteristics**:
- Latency-critical
- Smaller model weights
- Frequent cache hits

**Recommended Settings**:

```rust
let config = TransportConfig {
    bitswap: BitswapConfig {
        max_concurrent_requests: 64,
        request_timeout: Duration::from_secs(3),
        enable_have_messages: false, // Reduce message overhead
        ..Default::default()
    },

    quic: QuicConfig {
        enable_0rtt: true,
        keep_alive_interval: Duration::from_secs(1),
        ..Default::default()
    },

    prefetch: PrefetchConfig {
        enable: true,
        strategy: PrefetchStrategy::Pattern, // Learn access patterns
        max_prefetch_depth: 3,
        confidence_threshold: 0.7,
    },
};
```

**Key Tuning Parameters**:
- `enable_0rtt`: true (minimize connection latency)
- `request_timeout`: 3s (fast failure for cache misses)
- Enable prefetching for predictable access patterns

---

### Bulk Data Transfer

**Characteristics**:
- Large datasets
- Bandwidth-limited
- Less latency-sensitive

**Recommended Settings**:

```rust
let config = TransportConfig {
    tensorswap: TensorSwapConfig {
        chunk_size: 8 * 1024 * 1024, // 8 MB chunks
        max_concurrent_streams: 512,
        ..Default::default()
    },

    quic: QuicConfig {
        initial_window: 100 * 1024 * 1024,  // 100 MB
        max_window: 1024 * 1024 * 1024,     // 1 GB
        max_concurrent_streams: 1024,
        ..Default::default()
    },

    peer_scoring: PeerScoringConfig {
        bandwidth_weight: 0.7,  // Prioritize high bandwidth
        latency_weight: 0.2,
        reliability_weight: 0.1,
        ..Default::default()
    },
};
```

**Key Tuning Parameters**:
- `chunk_size`: 8 MB (reduce overhead)
- `initial_window`: 100 MB (aggressive start)
- `max_concurrent_streams`: 1024 (maximize parallelism)

---

### Federated Learning

**Characteristics**:
- Bidirectional gradient exchange
- Multiple participants
- Privacy-sensitive

**Recommended Settings**:

```rust
let config = TransportConfig {
    gradient: GradientConfig {
        aggregation_strategy: AggregationStrategy::FederatedAvg,
        enable_verification: true,
        max_queue_size: 1000,
    },

    tensorswap: TensorSwapConfig {
        chunk_size: 1 * 1024 * 1024,
        max_concurrent_streams: 64,
        critical_priority_boost: 200,
        ..Default::default()
    },

    bandwidth: BandwidthConfig {
        enable_qos: true,
        // Reserve bandwidth for gradient uploads
        global_max_bytes_per_sec: Some(10 * 1024 * 1024),
        ..Default::default()
    },
};
```

**Key Tuning Parameters**:
- `enable_verification`: true (gradient integrity)
- `aggregation_strategy`: FederatedAvg
- Bandwidth limits to prevent network saturation

---

## Network Condition Optimization

### High Latency Networks (100ms+ RTT)

**Symptoms**:
- Slow block retrieval
- Frequent timeouts
- Poor throughput

**Solutions**:

```rust
let config = TransportConfig {
    quic: QuicConfig {
        initial_window: 50 * 1024 * 1024,  // Larger window
        keep_alive_interval: Duration::from_secs(5),
        idle_timeout: Duration::from_secs(60), // Longer timeout
        ..Default::default()
    },

    bitswap: BitswapConfig {
        max_concurrent_requests: 256,  // More parallelism
        request_timeout: Duration::from_secs(60),
        ..Default::default()
    },

    want_list: WantListConfig {
        default_timeout: Duration::from_secs(60),
        max_retries: 5,  // More retries
        ..Default::default()
    },
};
```

**Tuning Tips**:
- Increase `initial_window` to fill bandwidth-delay product
- Increase `max_concurrent_requests` for parallelism
- Extend timeouts to accommodate RTT
- Use pipelining aggressively

---

### Lossy Networks (>1% packet loss)

**Symptoms**:
- Retransmissions
- Variable throughput
- Connection drops

**Solutions**:

```rust
let config = TransportConfig {
    quic: QuicConfig {
        enable_0rtt: false,  // Reduce spurious retransmits
        keep_alive_interval: Duration::from_secs(3),
        ..Default::default()
    },

    want_list: WantListConfig {
        max_retries: 10,
        enable_exponential_backoff: true,
        ..Default::default()
    },

    circuit_breaker: CircuitBreakerConfig {
        failure_threshold: 10,  // Higher threshold
        timeout: Duration::from_secs(30),
        ..Default::default()
    },
};
```

**Tuning Tips**:
- Disable 0-RTT to avoid early data loss
- Increase retry attempts
- Higher circuit breaker threshold
- Consider using FEC (erasure coding)

---

### Bandwidth-Constrained Networks

**Symptoms**:
- Slow transfers
- Queue buildup
- High memory usage

**Solutions**:

```rust
let config = TransportConfig {
    bandwidth: BandwidthConfig {
        global_max_bytes_per_sec: Some(1024 * 1024), // 1 MB/s limit
        token_bucket_capacity: 512 * 1024,  // Small burst
        enable_qos: true,
    },

    tensorswap: TensorSwapConfig {
        chunk_size: 256 * 1024,  // Smaller chunks
        enable_backpressure: true,
        backpressure_config: BackpressureConfig {
            low_watermark: 5,
            high_watermark: 20,
            ..Default::default()
        },
    },

    session: SessionConfig {
        max_concurrent_blocks: 16,  // Limit concurrency
        ..Default::default()
    },
};
```

**Tuning Tips**:
- Enable bandwidth throttling
- Reduce chunk size
- Enable backpressure
- Limit concurrent operations

---

### Unstable/Mobile Networks

**Symptoms**:
- Frequent disconnections
- Variable latency
- IP address changes

**Solutions**:

```rust
let config = TransportConfig {
    quic: QuicConfig {
        enable_connection_migration: true,
        idle_timeout: Duration::from_secs(10),
        keep_alive_interval: Duration::from_secs(2),
        ..Default::default()
    },

    partition: PartitionDetectorConfig {
        failure_threshold: 3,
        check_interval: Duration::from_secs(5),
        recovery_threshold: 2,
    },

    recovery: RecoveryStrategyConfig {
        max_fallback_peers: 5,
        enable_degraded_mode: true,
        degraded_mode_threshold: 1,
    },
};
```

**Tuning Tips**:
- Enable QUIC connection migration
- Lower failure thresholds for faster detection
- Configure fallback peers
- Enable automatic recovery

---

## Resource Constraint Tuning

### Low Memory (< 100 MB)

```rust
let config = TransportConfig {
    bitswap: BitswapConfig {
        max_want_list_size: 64,
        max_concurrent_requests: 16,
        ..Default::default()
    },

    tensorswap: TensorSwapConfig {
        chunk_size: 256 * 1024,  // 256 KB chunks
        max_concurrent_streams: 8,
        enable_backpressure: true,
    },

    session: SessionConfig {
        max_concurrent_blocks: 16,
        ..Default::default()
    },

    peer_manager: PeerManagerConfig {
        max_peers: 10,
        connection_pool_size: 2,
        ..Default::default()
    },
};
```

**Memory Budget**:
- Want list: ~2 MB
- Block buffers: ~4 MB
- Connection overhead: ~2 MB per peer
- Total: ~30 MB

---

### Low CPU

```rust
let config = TransportConfig {
    bitswap: BitswapConfig {
        max_concurrent_requests: 32,
        ..Default::default()
    },

    peer_scoring: PeerScoringConfig {
        scoring_interval: Duration::from_secs(60),  // Less frequent
        ..Default::default()
    },

    multicast: MulticastConfig {
        use_gossip: true,  // Reduce CPU vs full broadcast
        gossip_fanout: 3,
        ..Default::default()
    },

    // Disable expensive features
    gradient: GradientConfig {
        enable_verification: false,  // Skip checksum validation
        ..Default::default()
    },
};
```

**CPU Optimizations**:
- Reduce scoring frequency
- Use gossip instead of broadcast
- Disable optional verifications
- Limit concurrent operations

---

## Performance Monitoring

### Key Metrics to Track

```rust
// Get statistics
let stats = transport.stats().await;

// Throughput metrics
println!("Bytes sent: {}", stats.bytes_sent);
println!("Bytes received: {}", stats.bytes_received);
println!("Throughput: {:.2} MB/s", stats.throughput_mbps());

// Latency metrics
println!("Avg block latency: {:?}", stats.avg_block_latency);
println!("P50 latency: {:?}", stats.p50_latency);
println!("P99 latency: {:?}", stats.p99_latency);

// Connection metrics
println!("Active peers: {}", stats.active_peers);
println!("Blacklisted peers: {}", stats.blacklisted_peers);
println!("Connection failures: {}", stats.connection_failures);

// Session metrics
println!("Active sessions: {}", stats.active_sessions);
println!("Completed sessions: {}", stats.completed_sessions);
println!("Session success rate: {:.2}%", stats.session_success_rate());
```

### Performance Targets

| Metric | Target | Configuration |
|--------|--------|---------------|
| Block latency (local) | < 10ms | Default QUIC + Low latency profile |
| Throughput (single peer) | > 100 MB/s | High throughput profile |
| P99 latency | < 100ms | Optimized QUIC windows |
| Connection success rate | > 95% | Proper retry configuration |
| Memory per peer | < 10 MB | Default settings |

---

## Troubleshooting

### Problem: High Latency

**Diagnosis**:
```rust
let stats = peer_manager.get_peer_stats(&peer_id);
println!("Avg latency: {:?}", stats.avg_latency);
println!("P99 latency: {:?}", stats.p99_latency);
```

**Solutions**:
1. Check network RTT: `ping <peer_ip>`
2. Enable 0-RTT if not already enabled
3. Reduce `keep_alive_interval`
4. Increase `max_concurrent_streams` for parallelism
5. Use `FastestFirst` peer selection strategy

---

### Problem: Low Throughput

**Diagnosis**:
```rust
let stats = transport.stats().await;
let throughput = stats.bytes_received as f64 / stats.elapsed.as_secs_f64();
println!("Throughput: {:.2} MB/s", throughput / 1024.0 / 1024.0);
```

**Solutions**:
1. Increase QUIC window sizes: `initial_window`, `max_window`
2. Increase `max_concurrent_streams`
3. Larger `chunk_size` for reduced overhead
4. Increase `max_concurrent_requests`
5. Check if bandwidth throttling is enabled

---

### Problem: Frequent Timeouts

**Diagnosis**:
```rust
let stats = want_list.stats();
println!("Timeout rate: {:.2}%",
    stats.timeouts as f64 / stats.total_requests as f64 * 100.0);
```

**Solutions**:
1. Increase `request_timeout`
2. Increase `max_retries`
3. Check peer availability and health
4. Verify network connectivity
5. Enable partition detection

---

### Problem: High Memory Usage

**Diagnosis**:
```rust
let stats = transport.memory_stats();
println!("Want list memory: {} MB", stats.want_list_bytes / 1024 / 1024);
println!("Block buffers: {} MB", stats.block_buffer_bytes / 1024 / 1024);
println!("Connection overhead: {} MB", stats.connection_bytes / 1024 / 1024);
```

**Solutions**:
1. Reduce `max_want_list_size`
2. Reduce `max_concurrent_requests`
3. Smaller `chunk_size`
4. Enable backpressure
5. Reduce `connection_pool_size`

---

### Problem: Connection Failures

**Diagnosis**:
```rust
let stats = peer_manager.stats();
println!("Connection failures: {}", stats.connection_failures);
println!("Blacklisted peers: {}", stats.blacklisted_count);
```

**Solutions**:
1. Check firewall/NAT configuration
2. Enable NAT traversal (STUN/TURN)
3. Increase `idle_timeout`
4. Reduce `failure_threshold` in circuit breaker
5. Verify peer addresses are reachable

---

## Advanced Tuning Tips

### 1. Profile Your Workload

```rust
// Enable detailed profiling
let config = TransportConfig {
    enable_profiling: true,
    profiling_interval: Duration::from_secs(10),
    ..Default::default()
};

// Collect and analyze
let profile = transport.get_profile().await;
analyze_bottlenecks(&profile);
```

### 2. A/B Test Configurations

```rust
// Test two configurations side-by-side
let results_a = benchmark_config(config_a, workload.clone()).await;
let results_b = benchmark_config(config_b, workload.clone()).await;

compare_results(&results_a, &results_b);
```

### 3. Adaptive Tuning

```rust
// Automatically adjust based on observed performance
let mut adaptive_config = AdaptiveConfig::new(base_config);

loop {
    let stats = transport.stats().await;
    adaptive_config.adjust(&stats);
    transport.update_config(adaptive_config.current()).await;

    sleep(Duration::from_secs(60)).await;
}
```

### 4. Environment-Specific Tuning

```rust
match deployment_env {
    Environment::Cloud => cloud_optimized_config(),
    Environment::Edge => edge_optimized_config(),
    Environment::DataCenter => datacenter_optimized_config(),
    Environment::Mobile => mobile_optimized_config(),
}
```

---

## Summary Checklist

- [ ] Choose appropriate profile for your use case
- [ ] Measure baseline performance
- [ ] Adjust QUIC windows for your network
- [ ] Configure retry and timeout parameters
- [ ] Enable monitoring and logging
- [ ] Test under realistic load
- [ ] Monitor resource usage (CPU, memory, network)
- [ ] Iterate and refine based on metrics

---

## Getting Help

If performance is still suboptimal after tuning:

1. Collect detailed statistics: `transport.dump_stats()`
2. Enable debug logging: `RUST_LOG=ipfrs_transport=debug`
3. Profile with `cargo flamegraph`
4. Check network conditions with `iperf3`
5. Review QUIC connection logs

For additional assistance, consult the [protocol specification](PROTOCOL.md) and [configuration guide](CONFIGURATION.md).
