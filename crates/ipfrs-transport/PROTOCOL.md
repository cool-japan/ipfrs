# ipfrs-transport Protocol Specification

Version: 1.0
Status: Draft
Last Updated: 2026-01-18

## Table of Contents

1. [Overview](#overview)
2. [Message Formats](#message-formats)
3. [Protocol State Machines](#protocol-state-machines)
4. [Block Exchange Protocol](#block-exchange-protocol)
5. [Tensor Streaming Protocol](#tensor-streaming-protocol)
6. [Peer Selection and Scoring](#peer-selection-and-scoring)
7. [Session Management](#session-management)
8. [Error Handling and Recovery](#error-handling-and-recovery)
9. [Interoperability Requirements](#interoperability-requirements)
10. [Security Considerations](#security-considerations)

---

## Overview

The ipfrs-transport protocol provides efficient block and tensor exchange mechanisms optimized for distributed machine learning workloads. It extends the IPFS Bitswap protocol with tensor-aware optimizations while maintaining backwards compatibility.

### Key Features

- **Bitswap Compatibility**: Wire-compatible with IPFS Bitswap protocol
- **Tensor Streaming**: Chunked transfer of large tensors with priority scheduling
- **Priority-Based Scheduling**: Dynamic priority adjustment based on computation graph dependencies
- **Multiple Transports**: QUIC, TCP, WebSocket support with automatic selection
- **Session Management**: Grouped block requests with progress tracking
- **NAT Traversal**: STUN/TURN integration with ICE-like connectivity establishment

### Design Principles

1. **Efficiency**: Minimize latency for tensor transfers critical to training loops
2. **Fairness**: Balanced resource allocation across peers
3. **Resilience**: Graceful degradation under network partitions
4. **Extensibility**: Modular design for future protocol extensions

---

## Message Formats

All messages use bincode serialization for compact representation. Messages are framed with a 4-byte length prefix (big-endian) when sent over streaming transports.

### Message Envelope

```rust
struct MessageEnvelope {
    version: u8,        // Protocol version (current: 1)
    message_type: u8,   // Message type identifier
    payload_length: u32,// Payload length in bytes
    payload: Vec<u8>,   // Serialized message
}
```

### Message Types

#### 1. WantList Message

Communicates block requests from requester to provider.

```rust
pub struct WantList {
    pub entries: Vec<WantEntry>,
    pub full: bool,  // true = replace entire want list, false = update
}

pub struct WantEntry {
    pub cid: Cid,              // Content identifier
    pub priority: i32,         // Request priority (higher = more urgent)
    pub send_dont_have: bool,  // Request DONT_HAVE if block not available
    pub cancel: bool,          // Cancel previous want for this CID
}
```

**Priority Levels**:
- 0-99: Low priority (background prefetch)
- 100-199: Normal priority
- 200-299: High priority (on critical path)
- 300-399: Urgent priority (blocking computation)
- 400+: Critical priority (past deadline)

#### 2. Block Message

Delivers requested block data.

```rust
pub struct BlockMessage {
    pub cid: Cid,      // Content identifier
    pub data: Vec<u8>, // Block payload
}
```

#### 3. Have Message

Notifies requester that provider has a block.

```rust
pub struct HaveMessage {
    pub cid: Cid,  // Content identifier
}
```

#### 4. DontHave Message

Notifies requester that provider does not have a block.

```rust
pub struct DontHaveMessage {
    pub cid: Cid,  // Content identifier
}
```

#### 5. Cancel Message

Cancels a previous block request.

```rust
pub struct CancelMessage {
    pub cid: Cid,  // Content identifier to cancel
}
```

### Tensor Extension Messages

#### TensorMetadata Message

Describes tensor structure and chunking strategy.

```rust
pub struct TensorMetadata {
    pub tensor_id: String,         // Unique tensor identifier
    pub shape: Vec<usize>,         // Tensor dimensions
    pub dtype: DataType,           // Element data type
    pub chunks: Vec<ChunkInfo>,    // Chunk descriptors
    pub total_size: u64,           // Total size in bytes
    pub priority_hint: Option<i32>,// Suggested priority
    pub dependencies: Vec<String>, // Dependency tensor IDs
    pub deadline: Option<Instant>, // Transfer deadline
}

pub struct ChunkInfo {
    pub cid: Cid,         // Chunk content identifier
    pub offset: u64,      // Byte offset in tensor
    pub size: usize,      // Chunk size in bytes
    pub received: bool,   // Reception status
}
```

#### GradientMessage

Exchanges gradient information for federated learning.

```rust
pub struct GradientMessage {
    pub id: String,              // Gradient identifier
    pub data: Vec<u8>,           // Gradient data
    pub shape: Vec<usize>,       // Gradient shape
    pub dtype: DataType,         // Data type
    pub checksum: u64,           // FNV-1a checksum
    pub metadata: GradientMeta,  // Additional metadata
}

pub struct GradientMeta {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub timestamp: u64,
}
```

---

## Protocol State Machines

### Peer Connection State Machine

```
┌─────────┐
│  INIT   │
└────┬────┘
     │
     ▼
┌──────────┐  timeout   ┌────────────┐
│CONNECTING├────────────►│ FAILED     │
└────┬─────┘            └────────────┘
     │success
     ▼
┌──────────┐  errors    ┌────────────┐
│CONNECTED ├────────────►│DISCONNECTED│
└────┬─────┘            └────────────┘
     │                         │
     │                         │retry
     └─────────────────────────┘
```

**States**:
- `INIT`: Initial state, no connection attempt
- `CONNECTING`: Connection establishment in progress
- `CONNECTED`: Active connection, can exchange messages
- `DISCONNECTED`: Connection lost, may retry
- `FAILED`: Permanent failure, blacklist peer

### Want State Machine

```
┌─────────┐
│ PENDING │
└────┬────┘
     │
     ▼
┌──────────┐  timeout   ┌────────────┐
│REQUESTED ├────────────►│ RETRY      │
└────┬─────┘            └──────┬─────┘
     │                         │
     │received                 │
     ▼                         │
┌──────────┐                   │
│COMPLETED │◄──────────────────┘
└──────────┘
```

**States**:
- `PENDING`: Want added to queue, not yet sent
- `REQUESTED`: Want sent to peer, awaiting response
- `RETRY`: Request timed out, will retry with backoff
- `COMPLETED`: Block received or want cancelled

### Session State Machine

```
┌────────┐
│ ACTIVE │
└───┬────┘
    │
    ├──pause──►┌────────┐
    │          │ PAUSED │
    │          └────┬───┘
    │               │resume
    │◄──────────────┘
    │
    ├──complete─►┌───────────┐
    │            │ COMPLETED │
    │            └───────────┘
    │
    └──cancel───►┌───────────┐
                 │ CANCELLED │
                 └───────────┘
```

**States**:
- `ACTIVE`: Session accepting requests and processing blocks
- `PAUSED`: Temporarily suspended, no new requests sent
- `COMPLETED`: All blocks received successfully
- `CANCELLED`: User-initiated cancellation

### Circuit Breaker State Machine

```
┌────────┐ failures≥threshold ┌──────┐
│ CLOSED ├───────────────────►│ OPEN │
└───┬────┘                    └───┬──┘
    │                             │timeout
    │                             ▼
    │                         ┌──────────┐
    │         success         │HALF-OPEN │
    └◄────────────────────────┤          │
                              └────┬─────┘
                                   │failure
                                   └─────►OPEN
```

**States**:
- `CLOSED`: Normal operation, requests pass through
- `OPEN`: Too many failures, block all requests
- `HALF-OPEN`: Testing with limited requests

---

## Block Exchange Protocol

### Request Flow

```
Requester                           Provider
    │                                   │
    │  1. WantList (CID, priority)     │
    ├──────────────────────────────────►│
    │                                   │
    │  2. Have (CID) [optional]        │
    │◄──────────────────────────────────┤
    │                                   │
    │  3. Block (CID, data)            │
    │◄──────────────────────────────────┤
    │                                   │
    │  4. Cancel (CID) [optional]      │
    ├──────────────────────────────────►│
    │                                   │
```

### Protocol Rules

1. **Want List Aggregation**: Batch multiple wants in single message
2. **Deduplication**: Provider tracks active wants per peer, ignores duplicates
3. **Priority Respect**: Provider serves higher priority requests first
4. **Have Notifications**: Send if `send_dont_have` flag is set
5. **Timeout Handling**: Requester retries after timeout with exponential backoff
6. **Cancellation**: Requester must send Cancel message to free provider resources

### Priority Scheduling Algorithm

```rust
fn effective_priority(want: &WantEntry, current_time: Instant) -> i32 {
    let mut priority = want.priority;

    // Boost priority as deadline approaches
    if let Some(deadline) = want.deadline {
        if current_time > deadline {
            priority += 400; // Critical: past deadline
        } else {
            let remaining = deadline.duration_since(current_time);
            if remaining < Duration::from_secs(10) {
                priority += 300; // Urgent: within 10s of deadline
            } else if remaining < Duration::from_secs(60) {
                priority += 100; // High: within 1min of deadline
            }
        }
    }

    // Boost priority for dependencies
    if want.has_dependents {
        priority += 50;
    }

    priority
}
```

---

## Tensor Streaming Protocol

### Chunking Strategy

Large tensors are split into chunks for efficient transfer:

1. **Chunk Size**: Default 1 MB, configurable based on network characteristics
2. **Chunk Ordering**: Sequential by default, can be parallelized
3. **Priority Assignment**: Earlier chunks get higher priority for progressive loading

### Streaming Flow

```
Client                              Server
  │                                   │
  │  1. TensorMetadata               │
  ├──────────────────────────────────►│
  │                                   │
  │  2. WantList (all chunk CIDs)    │
  ├──────────────────────────────────►│
  │                                   │
  │  3. Block (chunk 0)              │
  │◄──────────────────────────────────┤
  │                                   │
  │  4. Block (chunk 1)              │
  │◄──────────────────────────────────┤
  │                                   │
  │  5. Progress update              │
  ├──────────────────────────────────►│
  │                                   │
  │  ... (remaining chunks)          │
  │                                   │
```

### Backpressure Mechanism

```rust
struct BackpressureController {
    low_watermark: usize,   // Resume sending (default: 10 chunks)
    high_watermark: usize,  // Pause sending (default: 50 chunks)
    current_queue: usize,   // Current queued chunks
}

impl BackpressureController {
    fn should_send(&self) -> bool {
        self.current_queue < self.high_watermark
    }

    fn should_resume(&self) -> bool {
        self.current_queue < self.low_watermark
    }
}
```

### Dependency-Aware Scheduling

For computation graph scenarios (e.g., Einsum operations):

```
Tensor A (high priority, no deps)
   ├─► Tensor B (normal priority, depends on A)
   └─► Tensor C (normal priority, depends on A)
        └─► Tensor D (low priority, depends on C)
```

**Scheduling Order**: A → B, C (parallel) → D

---

## Peer Selection and Scoring

### Peer Score Calculation

```rust
fn calculate_peer_score(peer: &PeerMetrics, config: &PeerScoringConfig) -> f64 {
    let latency_score = 1.0 / (1.0 + peer.avg_latency_ms / 1000.0);
    let bandwidth_score = peer.bandwidth_mbps / 100.0; // Normalize to 100 Mbps
    let reliability_score = peer.success_rate;
    let debt_score = 1.0 - peer.debt_ratio().abs();

    config.latency_weight * latency_score +
    config.bandwidth_weight * bandwidth_score +
    config.reliability_weight * reliability_score +
    config.debt_weight * debt_score
}
```

### Selection Strategies

#### FastestFirst
Select peer with lowest average latency.

#### HighestBandwidth
Select peer with highest measured bandwidth.

#### BestScore
Select peer with highest composite score.

#### RoundRobin
Distribute requests evenly across all peers.

#### LeastLoaded
Select peer with fewest active requests.

### Blacklist Criteria

A peer is blacklisted if:
1. Score drops below `min_score_threshold` (default: 0.1)
2. More than 5 consecutive failures
3. Sends invalid/corrupted data

Blacklist duration: Exponential backoff starting at 60s, max 1 hour.

---

## Session Management

### Session Lifecycle

```
1. Create session with configuration
2. Add blocks to session (can batch)
3. Send block requests
4. Track progress via events
5. Complete when all blocks received
```

### Session Events

```rust
pub enum SessionEvent {
    Started { session_id: SessionId },

    BlockReceived {
        session_id: SessionId,
        cid: Cid,
        size: usize,
    },

    BlockFailed {
        session_id: SessionId,
        cid: Cid,
        error: String,
    },

    Progress {
        session_id: SessionId,
        stats: SessionStats,
    },

    Completed {
        session_id: SessionId,
        stats: SessionStats,
    },

    Cancelled {
        session_id: SessionId,
    },
}
```

### Session Statistics

```rust
pub struct SessionStats {
    pub total_blocks: usize,
    pub blocks_received: usize,
    pub blocks_failed: usize,
    pub bytes_transferred: u64,
    pub started_at: Option<Instant>,
    pub completed_at: Option<Instant>,
    pub avg_block_time: Option<Duration>,
}
```

---

## Error Handling and Recovery

### Error Types

```rust
pub enum TransportError {
    // Connection errors
    ConnectionFailed(String),
    ConnectionTimeout,
    ConnectionLost,

    // Transfer errors
    BlockNotFound(Cid),
    TransferTimeout(Cid),
    InvalidData(Cid),

    // Protocol errors
    ProtocolVersionMismatch,
    InvalidMessage,
    SerializationError,

    // Resource errors
    QuotaExceeded,
    RateLimitExceeded,
}
```

### Retry Strategy

```rust
struct RetryPolicy {
    max_retries: usize,
    base_delay: Duration,
    max_delay: Duration,
    jitter_percent: u32,
}

impl RetryPolicy {
    fn next_backoff(&mut self) -> Duration {
        let delay = self.base_delay * 2u32.pow(self.attempt as u32);
        let delay = delay.min(self.max_delay);

        // Add jitter (±jitter_percent%)
        let jitter = delay * self.jitter_percent / 100;
        let jitter_offset = rand::random::<u64>() % (2 * jitter.as_millis() as u64);
        delay + Duration::from_millis(jitter_offset) - jitter
    }
}
```

### Partition Detection and Recovery

```
1. Monitor peer health via periodic probes
2. After N consecutive failures, mark peer as partitioned
3. Queue requests for partitioned peers
4. Periodically retry partitioned peers
5. On successful probe, mark as recovered
6. Flush queued requests
```

**Configuration**:
- `failure_threshold`: 5 failures
- `probe_interval`: 10s
- `recovery_grace_period`: 30s

---

## Interoperability Requirements

### Bitswap Compatibility

To maintain compatibility with IPFS Bitswap:

1. **Message Format**: Use protobuf-compatible wire format for WantList, Block, Have, DontHave messages
2. **Protocol ID**: Identify as `/ipfs/bitswap/1.2.0` in libp2p multistream negotiation
3. **CID Support**: Support CIDv0 and CIDv1 with all multihash types
4. **Ledger Tracking**: Maintain per-peer byte exchange ledgers

### Tensor Extensions

Tensor-specific features are opt-in:

1. **Capability Negotiation**: Use protocol ID `/ipfs/tensorswap/1.0.0` for tensor features
2. **Fallback**: Degrade to standard Bitswap if peer doesn't support TensorSwap
3. **Metadata Exchange**: TensorMetadata messages only sent to TensorSwap peers

### Multi-Transport Support

Transports tried in order:
1. QUIC (preferred for performance)
2. WebSocket (for browser/gateway compatibility)
3. TCP (fallback for maximum compatibility)

Protocol negotiation via libp2p multistream-select.

---

## Security Considerations

### Authentication

- **Peer Identity**: Use libp2p PeerID derived from public key
- **Message Signing**: Optional signing for gradient messages in federated learning

### Integrity

- **Block Verification**: Validate received blocks match requested CID
- **Gradient Checksums**: FNV-1a checksum for gradient data integrity
- **Metadata Validation**: Verify tensor metadata consistency

### DoS Protection

- **Rate Limiting**: Token bucket per peer (default: 10 MB/s)
- **Want List Size Limits**: Max 1000 wants per peer
- **Session Limits**: Max 100 concurrent sessions per peer
- **Blacklisting**: Auto-blacklist abusive peers

### Privacy

- **Content Privacy**: Blocks transferred as opaque data
- **Metadata Privacy**: CIDs reveal content hashes but not content
- **Traffic Analysis**: Consider QUIC connection migration to prevent tracking

---

## Implementation Notes

### Performance Optimizations

1. **Zero-Copy**: Use `Bytes` for block data to enable zero-copy forwarding
2. **Batching**: Aggregate multiple wants in single message
3. **Pipelining**: Send multiple requests before receiving responses
4. **Connection Pooling**: Reuse QUIC connections across sessions

### Testing Requirements

1. **Unit Tests**: All message serialization/deserialization
2. **Integration Tests**: Multi-peer block exchange
3. **Interop Tests**: Compatibility with go-ipfs/kubo
4. **Performance Tests**: Throughput and latency benchmarks

### Monitoring

Required metrics:
- Blocks sent/received per second
- Average block transfer latency (p50, p99)
- Active peer count
- Blacklisted peer count
- Session completion rate
- Bandwidth utilization

---

## References

1. [IPFS Bitswap Specification](https://github.com/ipfs/specs/blob/master/BITSWAP.md)
2. [libp2p Specifications](https://github.com/libp2p/specs)
3. [QUIC RFC 9000](https://www.rfc-editor.org/rfc/rfc9000)
4. [ICE RFC 8445](https://www.rfc-editor.org/rfc/rfc8445)
5. [Content Addressing Specification](https://github.com/multiformats/cid)

---

## Version History

- **v1.0** (2026-01-18): Initial specification
  - Core Bitswap protocol
  - Tensor streaming extensions
  - Multi-transport support
  - Session management
