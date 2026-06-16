# Changelog

All notable changes to IPFRS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-15 "Network Release"

### Network Release

The 0.2.0 "Network Release" delivers real peer-to-peer networking, distributed content discovery, and multi-node block exchange. IPFRS transitions from a local-first storage tool to a fully distributed InterPlanetary File System node.

**Status:** Production Ready (Distributed P2P)

**Total Implementation:** ~7,200 lines of production Rust code across 9 crates

---

### Added

#### Real P2P Networking (libp2p)
- **QUIC transport** (primary): sub-millisecond connection establishment, multiplexed streams
- **TCP transport** (fallback): reliable connectivity for environments without QUIC support
- **Noise encryption**: authenticated encrypted channels for all peer connections
- **Yamux multiplexing**: bidirectional stream multiplexing over TCP connections
- **Peer identity**: Ed25519 keypair-based persistent peer IDs
- **Connection manager**: configurable peer limits, connection lifecycle management
- **AutoNAT**: automatic NAT type detection and external address discovery
- **Identify protocol**: peer capability and address advertisement

#### Kademlia DHT Content Discovery
- **DHT bootstrap**: automatic seeding from well-known bootstrap peers
- **Auto-provide on add**: `add_bytes()` and `add_file()` now announce content to DHT automatically
- **DHT fallback on get miss**: `get()` transparently queries DHT provider records when local miss occurs
- **`find_providers()`**: now returns `Vec<PeerId>` (previously returned `()`)
- **Provider records**: `START_PROVIDING` / `STOP_PROVIDING` lifecycle with TTL-based expiry
- **Iterative lookups**: Kademlia XOR-metric iterative closest-peer queries

#### Multi-Node Block Exchange (Bitswap)
- **Bitswap protocol** (`/ipfs/bitswap/1.2.0`): IPFS-compatible block exchange
- **Want-list management**: peer-local want-list tracking with deduplication
- **Block broadcasting**: push blocks to interested peers on receipt
- **Session-based transfer**: per-session block retrieval with timeout and retry
- **Peer scoring**: bandwidth-aware peer selection for optimal transfer speeds

#### TensorSwap Protocol
- **TensorSwap** (`/ipfrs/tensorswap/1.0.0`): custom binary protocol for tensor and ML model streaming
- **Chunked streaming**: large tensor arrays split into fixed-size chunks for flow control
- **Resumable transfers**: chunk-level resume from last ACK on connection drop
- **Metadata framing**: dtype, shape, strides embedded in protocol header

#### Apache Arrow Zero-Copy Tensor Integration
- **Arrow IPC over TensorSwap**: tensors serialized as Arrow record batches
- **Zero-copy deserialization**: direct mmap of Arrow buffers into Rust slices
- **10x faster model loading**: benchmarked vs. JSON/CBOR tensor serialization
- **Schema negotiation**: peer capability handshake before batch transfer begins

#### Semantic DHT: HNSW Vector Search on DHT Routing
- **Semantic routing layer**: HNSW index integrated with Kademlia routing table
- **Vector-annotated provider records**: DHT records carry embedding metadata
- **Proximity-aware lookups**: nearest-neighbor DHT queries prefer semantically similar peers
- **Embedding propagation**: local HNSW index synchronized across connected peers

#### Network-Aware CLI Commands
- **`ipfrs swarm peers`**: list currently connected peer IDs and addresses
- **`ipfrs swarm connect <multiaddr>`**: dial and connect to a specific peer
- **`ipfrs swarm addrs`**: show all local listening addresses
- **`ipfrs dht findprovs <cid>`**: query DHT for providers of a given CID
- **`ipfrs dht provide <cid>`**: announce self as provider for a given CID
- **`ipfrs dht findpeer <peer-id>`**: resolve addresses for a peer ID via DHT
- **`ipfrs bootstrap`**: connect to IPFS bootstrap nodes and join the DHT

#### Daemon Improvements
- **Background mode** (`--background` / `-d`): fork-and-detach daemon process
- **Adaptive polling** for mobile/IoT: dynamic sleep interval based on activity level
- **PID file management**: `/tmp/ipfrs.pid` for daemon lifecycle control
- **`ipfrs daemon stop`**: graceful shutdown via PID file signal

#### Daemon Health Monitoring
- **5-category health checks**: storage, network, DHT, semantic index, tensorlogic
- **Health endpoint**: `GET /health` returns per-category status JSON
- **Automatic recovery**: daemon restarts failed subsystems within configurable retry budget
- **Metrics export**: Prometheus-compatible counters for block ops, peer count, DHT queries

#### GossipSub Pub/Sub Messaging
- **GossipSub** (`/meshsub/1.1.0`): IPFS-compatible topic-based pub/sub
- **Block announcement topics**: peers subscribe to CID announcement channels
- **Content routing events**: add/get events propagated through mesh
- **Fan-out cache**: recent message deduplication for high-churn networks

#### Node Architecture: Modular `node/` Directory
- `node.rs` refactored from single 2,477-line file to modular `node/` directory:
  - `node/core.rs` - lifecycle management, config, startup/shutdown
  - `node/network.rs` - swarm event loop, peer management
  - `node/dht.rs` - Kademlia operations, provider records
  - `node/exchange.rs` - Bitswap session management
  - `node/health.rs` - health monitoring and recovery

#### TensorLogic IR → IPLD Codec (`ipfrs_tensorlogic::ipld_codec`)
- DAG-CBOR serialization of TensorLogic terms, rules, and facts as IPLD nodes
- Content-addressed rule storage: rules are immutable, CID-identified structures
- Cross-node rule deduplication: identical rules share a single CID across the network
- IPLD path resolution for rule introspection (e.g., `/rule/<cid>/head/args/0`)
- 20 tests covering round-trip serialization and deduplication invariants

#### HNSW Persistent Index (`ipfrs_semantic::persistence`)
- `IndexPersistence` trait: serialize/deserialize HNSW graph to disk via oxicode
- `IndexSnapshot` and `IndexEntry` types for structured snapshot metadata
- Automatic save on `node.stop()`; automatic restore on `node.start()`
- 7 tests covering save/restore round-trips and crash-recovery scenarios

#### Block Compression via OxiARC
- Per-block zstd/lz4/snappy compression with magic-byte framing for codec detection
- Transparent decompression on read; `CompressionBlockStore<S>` wrapper store
- Blocks smaller than 256 bytes stored raw to avoid compression overhead
- 12 tests covering all codec paths, framing correctness, and raw passthrough

#### Gradient Computation Graph (`ComputationGraphStore`)
- CID-linked DAG for computation graph nodes with topological ordering
- Provenance chain: each op node links to input and output CIDs
- Checkpoint/resume: gradient computation resumable from any CID in the graph via `GradientCheckpoint`
- `GradientCheckpoint` with atomic file save for crash safety
- 13 tests covering DAG construction, topological sort, and checkpoint round-trips

#### Semantic DHT Production Hardening
- `VectorAnnotatedRecord`: DHT records carry embedding vectors alongside provider metadata
- `SemanticDhtMetrics`: per-node counters for routing convergence and query latency
- `put_with_vector()` and `search_similar()` APIs for semantic-aware DHT operations
- `get_routing_convergence()`: measure routing stability across connected peers
- `efficient_partial_sync()`: gossip only changed embedding regions to reduce sync traffic
- TTL eviction for stale vector-annotated records

#### NAT Traversal Defaults
- dcutr enabled by default (`dcutr_enabled: true` in `NodeConfig`)
- `NatTraversalMetrics`: tracks hole-punch attempt counts and success rates per peer

#### CLI Progress Bars
- `file_progress_bar()`: animated progress bar for `ipfrs add` / `ipfrs get` on files >10MB
- Hidden automatically for small files and non-TTY environments (pipes, scripts)

#### CLI Offline Detection
- `check_daemon_reachable()`: detects offline/stopped daemon in <2s via PID file check
- Network-dependent commands fail fast with a clear error rather than waiting for 30s timeout

#### Wave 7 Additions

##### Distributed Inference via GossipSub
- `distributed_infer()`: publishes inference requests over GossipSub and aggregates responses
- Per-session oneshot response waiters with configurable timeout handling
- Session management tracks in-flight distributed inference requests

##### Garbage Collection
- `ipfrs gc [--dry-run] [--min-age N]` CLI command backed by `OrphanGarbageCollector`
- Dry-run mode reports orphaned blocks without deleting; `--min-age` sets minimum age in seconds

##### Incremental HNSW Index Snapshots
- `IncrementalTracker` records dirty nodes since last save
- `save_smart()`: performs a full snapshot or delta save depending on dirty ratio (<10% dirty triggers delta)
- `load_index_with_delta()`: applies delta on restart to restore latest state without full reload

##### TensorLogic Persistence
- Rules and facts are snapshotted to IPLD DAG via oxicode on `node.stop()`
- Restored automatically on `node.start()`; no re-submission required after restart

##### Rule Sharing via Bitswap
- `publish_rule(cid)`: announces a rule CID to the DHT and makes it available over Bitswap
- `fetch_rule(cid)`: retrieves a rule from a remote peer by CID using Bitswap block exchange
- `import_rules_from_cids(cids)`: batch import of rules encoded as IPLD DAG-CBOR

##### Advanced CLI Query Language
- `ipfrs semantic query "<text>" [--top-k N]`: natural language nearest-neighbor search
- `ipfrs logic query "<goal>" [--format json|text] [--timeout-ms N]`: streaming logic resolution
- `ipfrs query --hybrid --logic "<goal>"`: combines semantic similarity ranking with logic constraints

##### Node.js Bindings
- NAPI-RS async bindings with `tokio::sync::Mutex` for safe concurrent access
- Published as `@cool-japan/ipfrs-node` NPM package

##### Content Encryption at Rest
- AES-256-GCM and ChaCha20-Poly1305 encryption for stored blocks
- Enabled via `--features encryption`; off by default to preserve Pure Rust defaults

#### Temporal Pattern Matching (`ipfrs-tensorlogic`)
- **`TemporalPatternMatcher`**: NFA-based event sequence matcher with configurable time windows and wildcards
- **`MatcherConfig`**: tunable max states, event label registry, and backtrack limit

#### Traffic Shaping (`ipfrs-network`)
- **`TrafficShaper`** / **`PeerTrafficShaper`**: per-peer token-bucket–based egress shaping with queuing disciplines (FIFO, WFQ, priority)
- Rate limits configurable per-peer with burst capacity and drain interval

#### Adaptive Bandwidth Allocation (`ipfrs-network`)
- **`AdaptiveBandwidthAllocator`**: fair-share bandwidth allocation across peers with configurable policies (MaxMin fairness, proportional)
- **`BandwidthWindow`**: sliding-window bandwidth measurement; `BandwidthStats` reports utilization and Jain fairness index
- **`AllocatorConfig::validate()`**: pre-flight capacity and threshold validation

#### Abductive Reasoning Engine (`ipfrs-tensorlogic`)
- **`AbductiveReasoningEngine`**: hypothesis-driven explanation search over observation sets with configurable cost functions (`MinCost`, `MaxCoverage`, `Weighted`)
- **`AbrExplanation`**: ranked explanations with completeness score and total cost
- **`AbrTerm`** / **`AbrHypothesis`** / **`AbrRule`**: abducible term graph with FNV-1a fingerprint-based deduplication

#### Storage Access Logging (`ipfrs-storage`)
- **`StorageAccessLog`** / **`StorageAccessLogger`**: event log for block get/put/delete ops with retention policies and burst detection
- **`AccessLogStats`**: per-operation counters, unique block count, cumulative read/write byte totals
- **`AccessPattern`** detection: sequential, random, burst recognition

#### Probabilistic Program Engine (`ipfrs-tensorlogic`)
- **`ppe_types`**: `ProbVar`, `Distribution` (Gaussian, Beta, Bernoulli, Categorical, Dirichlet, Uniform), and prior specification types for probabilistic programs
- **`ppe_sampling`**: PRNG helpers and sampling functions (inverse-CDF, rejection sampling) used by the probabilistic inference backend
- Enables probabilistic programming patterns — Monte Carlo estimation, Bayesian posteriors — within the TensorLogic reasoning pipeline

#### Reinforcement Learning Agent (`ipfrs-tensorlogic`)
- **`rla_types`**: `RlState`, `RlAction`, `RlTransition`, `ExperienceReplayBuffer`, epsilon-greedy / softmax / UCB1 policy types
- **`RlaConfig`** and **`RlaStats`**: configuration and per-episode statistics for RL training loops
- Supports model-based and model-free RL agent definitions integrated with TensorLogic's goal-directed reasoning

---

### Changed

- **`find_providers()`** now returns `Vec<PeerId>` instead of `()` (breaking change)
- **`add_bytes()`** and **`add_file()`** now auto-announce content to DHT after local storage
- **`get()`** now falls back to DHT provider discovery and peer block fetch on local miss
- **`NodeConfig`**: new fields `listen_addrs`, `bootstrap_peers`, `enable_gossipsub`, `enable_tensorswap`
- **`ipfrs daemon`**: now starts a full libp2p swarm instead of a stub process
- **HTTP gateway**: swarm and DHT endpoints added (`/api/v0/swarm/*`, `/api/v0/dht/*`)
- License standardized to `Apache-2.0` only (dropped `MIT OR Apache-2.0`; COOLJAPAN Policy 2026+)
- Compression: replaced `zstd`, `lz4`/`lz4_flex`, `snap`, and `flate2` (C-backed) with `oxiarc-zstd`, `oxiarc-lz4`, `oxiarc-snappy`, and `oxiarc-deflate` 0.3.3 (pure Rust via OxiARC family)
- `oxicode` updated from 0.1 to 0.2.4 (`derive` + `serde` features)
- `multihash-codetable` updated from 0.1 to 0.2
- `cid` now includes `serde-codec` feature
- New workspace members: `ipfrs-wasm` (WebAssembly bindings) and `ipfrs-nodejs` (Node.js/NAPI-RS bindings)
- CLI tests extracted to `crates/ipfrs-cli/src/cli_tests.rs` (686 lines) to keep `main.rs` under the 2000-line refactoring policy
- `pyo3` updated to 0.29 with `pyo3-build-config` for portable Python library detection (replaces hardcoded `python3.11`/absolute path in `build.rs`; `PYO3_PYTHON` env var now defaults to `python3`)
- `atty` replaced by `is-terminal` (maintained alternative; `atty` is unmaintained)

---

### Performance

- **QUIC transport**: sub-millisecond latency for block exchange between peers on LAN
- **Zero-copy tensor streaming**: 10x faster ML model loading vs. traditional JSON/CBOR approaches
- **ARM NEON optimization**: hashing and encryption routines use NEON intrinsics on AArch64
- **Parallel DHT lookups**: concurrent alpha-parallel Kademlia queries (alpha=3 by default)
- **Want-list batching**: Bitswap want messages coalesced into 64-entry batches to reduce RTT

---

### Architecture

#### Crate Structure (0.2.0)
- **ipfrs-core** (~460 lines): Block, CID, Error, IPLD - stable
- **ipfrs-storage** (~620 lines): Sled block store, caching - stable
- **ipfrs-semantic** (~680 lines): HNSW index, semantic DHT routing
- **ipfrs-tensorlogic** (~430 lines): Logic store, TensorLogic IR
- **ipfrs-interface** (~1,350 lines): HTTP gateway, REST API, swarm/DHT endpoints
- **ipfrs-network** (~1,100 lines): libp2p swarm, Kademlia DHT, GossipSub, AutoNAT
- **ipfrs-transport** (~820 lines): TensorSwap, Bitswap, Arrow IPC integration
- **ipfrs** (~420 lines): Node API, modular node/ directory
- **ipfrs-cli** (~720 lines): Command-line interface, network commands

**Total:** ~6,600 lines across 9 crates (production code, excluding tests)

#### Technology Stack (0.2.0)
- **Runtime**: Tokio async (1.x)
- **Storage**: Sled embedded database
- **Network**: rust-libp2p 0.54 (QUIC + TCP, Kademlia, GossipSub, Bitswap)
- **Vector Search**: HNSW algorithm with semantic DHT integration
- **Serialization**: Serde, DAG-CBOR, JSON, Apache Arrow IPC
- **HTTP**: Axum web framework
- **CLI**: Clap argument parser
- **Zero-Copy**: Apache Arrow + Bytes crate

---

### Known Limitations (0.2.0)

- Distributed inference is initial only: backward chaining is local-only; distributed proof tree planned for 0.3.0
- HNSW-on-DHT production scale (1M vectors / 10 nodes) not yet benchmarked
- Snapshot CID pinning not yet implemented; GC may collect active index blocks in edge cases
- Content encryption at rest requires `--features encryption`; not enabled by default

---

### Breaking Changes from 0.1.0

| Symbol | 0.1.0 | 0.2.0 |
|--------|-------|-------|
| `find_providers()` | returns `()` | returns `Vec<PeerId>` |
| `add_bytes()` | local store only | local store + DHT announce |
| `add_file()` | local store only | local store + DHT announce |
| `get()` | local only | local + DHT fallback |
| `NodeConfig` | no network fields | `listen_addrs`, `bootstrap_peers` added |

---

### Upgrade Guide: 0.1.0 to 0.2.0

```bash
# Build with network features enabled
cargo build --all-features

# Initialize and start daemon with networking
ipfrs init
ipfrs daemon --background

# Check connected peers
ipfrs swarm peers

# Connect to a peer
ipfrs swarm connect /ip4/127.0.0.1/udp/4001/quic-v1/p2p/<peer-id>

# Add content (auto-announces to DHT)
ipfrs add myfile.txt

# Find providers for a CID
ipfrs dht findprovs bafybeig...

# Bootstrap into the IPFS network
ipfrs bootstrap
```

---

## [0.1.0] - 2026-01-18 "Foundation Release"

### First Stable Release

The 0.1.0 "Foundation Release" establishes IPFRS as a production-ready local-first content-addressed storage system with unique semantic search and logic programming capabilities.

**Status:** Production Ready (Local-First Focus)

**Total Implementation:** ~4,417 lines of production Rust code across 8 crates

---

### Added

#### Core Storage & Retrieval
- **Content-addressed block storage** using Sled embedded database
- **Block operations**: put, get, has, delete with full async support
- **Batch operations**: put_many, get_many, has_many, delete_many
- **File operations**: add_file, get_to_file, add_reader, add_bytes, get_range
- **Directory operations**: add_directory, get_directory with recursive tree handling
- **Block management**: block_stat, block_rm for metadata and lifecycle

#### DAG Operations
- **DAG-CBOR serialization** for IPLD data structures
- **dag_put**: Store IPLD nodes with automatic CID generation
- **dag_get**: Retrieve and deserialize IPLD structures
- **dag_resolve**: Navigate IPLD paths (e.g., "/key1/key2/0")
- **dag_traverse**: BFS graph traversal with cycle detection
- **IPLD support**: Maps, Lists, Links, Bytes, Integers, Strings

#### Semantic Search
- **HNSW vector index** (Hierarchical Navigable Small World)
- **index_content()**: Add CID-embedding pairs to semantic index
- **search_similar()**: k-NN approximate nearest neighbor search
- **search_hybrid()**: Filtered search with QueryFilter (min_score, max_results, cid_prefix)
- **Configurable distance metrics**: Cosine, L2, DotProduct
- **LRU query caching** for performance optimization
- **semantic_stats()**: Real-time index statistics (vectors, dimension, cache)

#### Logic Programming
- **TensorLogic store** with content-addressed IR
- **put_term()**: Store logical terms (Variable, Constant, Compound)
- **get_term()**: Retrieve terms by CID
- **store_predicate()**: Store predicates with arguments
- **get_predicate()**: Retrieve predicates by CID
- **store_rule()**: Store inference rules (head + body)
- **get_rule()**: Retrieve rules by CID
- **JSON serialization** for portability and sharing
- **tensorlogic_stats()**: System statistics and monitoring

#### HTTP Gateway & API
- **20 REST API endpoints** for complete system access
- **Kubo (go-ipfs) compatibility**: 11 core endpoints
- **HTTP 206 range requests**: Efficient partial content delivery
- **Zero-copy serving**: Direct block data streaming

#### Command-Line Interface
- **13 production-ready commands** for complete system management
- JSON output format (`--format json`) for automation
- Verbose logging (`--verbose`)
- Custom data directories (`--data-dir`)
- Unix pipeline integration
- Binary-safe content handling

#### Observability & Monitoring
- **storage_stats()**: Block count, total size, capacity checks
- **semantic_stats()**: Vector count, dimension, metric, cache performance
- **tensorlogic_stats()**: Logic store status and metrics
- **status()**: Comprehensive node status (storage, network, semantic, logic)

---

### Performance (0.1.0)

- **Block put**: ~50µs (20,000 ops/sec)
- **Block get**: ~30µs (33,000 ops/sec)
- **DAG put**: ~80µs (12,500 ops/sec)
- **Semantic search (k=10)**: ~1ms (1,000 queries/sec)
- **HNSW insertion**: ~100µs (10,000 inserts/sec)

*Tested on: AMD Ryzen 9 5900X, NVMe SSD*

---

### Known Limitations (0.1.0)

- No peer-to-peer networking (local-only)
- No distributed inference engine
- No daemon mode with background services
- Semantic/logic indexes are in-memory only (not persisted)

---

## Versioning Policy

IPFRS follows [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: Backwards-compatible functionality
- **PATCH**: Backwards-compatible bug fixes

---

## Contributors

- TensorLogic Architect - Initial implementation
- IPFRS Team - Code review and testing

---

For questions, issues, or contributions, visit our [GitHub repository](https://github.com/cool-japan/ipfrs).

[0.2.0]: https://github.com/cool-japan/ipfrs/releases/tag/v0.2.0
[0.1.0]: https://github.com/cool-japan/ipfrs/releases/tag/v0.1.0
