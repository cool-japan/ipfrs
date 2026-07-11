# ipfrs-nodejs TODO

> **Status: 0.3.0 "Hardening Release" milestone complete.** Items marked ⏳ below are *post-0.3.0 roadmap* — deferred, not release-blocking — with the deferral reason in `_(post-0.3.0: …)_`. They are **honestly deferred, not claimed as done**. Completed work remains marked ✅ / `- [x]`.

## ✅ Completed (Phase 1: Foundation)

### NAPI-RS Binding Setup
- ✅ Set up napi-rs for Node.js bindings
- ✅ Configure build.rs for native module compilation
- ✅ Create package.json with npm publishing metadata

### Core Node Interface
- ✅ **`Node` class** - Main IPFRS node interface
  - Constructor with `NodeConfig`
  - `start()` / `stop()` lifecycle methods
  - Tokio runtime integration for async operations

### Block Operations
- ✅ **`putBlock(data)`** - Store block data
  - Accept Buffer as input
  - Return CID string
  - Promise-based async API

- ✅ **`getBlock(cid)`** - Retrieve block data
  - Parse CID string
  - Return Buffer or null
  - Promise-based async API

- ✅ **`hasBlock(cid)`** - Check block existence
- ✅ **`deleteBlock(cid)`** - Remove block from storage

### Semantic Search
- ✅ **`indexContent(cid, embedding)`** - Index content with vector
- ✅ **`searchSimilar(query, k)`** - Vector similarity search
- ✅ **`searchFiltered(query, k, filter)`** - Filtered search with `QueryFilter`
- ✅ **`saveSemanticIndex(path)`** - Persist index to disk
- ✅ **`loadSemanticIndex(path)`** - Load index from disk

### TensorLogic Integration
- ✅ **`addFact(predicate)`** - Add fact to knowledge base
- ✅ **`addRule(rule)`** - Add inference rule
- ✅ **`infer(goal)`** - Run backward chaining inference
- ✅ **`prove(goal)`** - Generate proof tree
- ✅ **`kbStats()`** - Get knowledge base statistics
- ✅ **`saveKb(path)`** / **`loadKb(path)`** - Knowledge base persistence

### Type Definitions
- ✅ **`Term`** - Logical term (int, float, string, bool, var)
- ✅ **`Predicate`** - Logical predicate with args
- ✅ **`Rule`** - Inference rule (head + body)
- ✅ **`SearchResult`** - Search result with CID and score
- ✅ **`QueryFilter`** - Filter for semantic search
- ✅ **`KbStats`** - Knowledge base statistics

---

## Phase 2: TypeScript Enhancement (Priority: High)

### Type Definition Files
- ⏳ **Generate comprehensive `.d.ts` files**  _(post-0.3.0: binding-toolchain)_
  - All public APIs with full type signatures
  - JSDoc comments for IntelliSense
  - Generic types where appropriate

- ⏳ **Add branded types for safety**  _(post-0.3.0: binding-toolchain)_
  - `CidString` type for validated CID strings
  - `EmbeddingVector` type for float arrays
  - Type guards for runtime validation

### Error Handling
- ⏳ **Custom error classes**  _(post-0.3.0: binding-toolchain)_
  - `IpfrsError` base class
  - `NetworkError`, `StorageError`, `LogicError` subclasses
  - Error codes for programmatic handling
  - Stack trace preservation

- ⏳ **Typed error returns**  _(post-0.3.0: binding-toolchain)_
  - Result-like types for operations that can fail
  - Discriminated unions for error handling

---

## Phase 3: Streaming & Performance (Priority: High)

### Streaming API
- ⏳ **Implement streaming block upload**  _(post-0.3.0: binding-toolchain)_
  - Accept `ReadableStream` for large files
  - Progress callbacks during upload
  - Chunked transfer support

- ⏳ **Implement streaming block download**  _(post-0.3.0: binding-toolchain)_
  - Return `ReadableStream` for large blocks
  - DAG traversal with streaming
  - Memory-efficient for large files

- ⏳ **Add async iterators**  _(post-0.3.0: binding-toolchain)_
  - `AsyncIterator<Block>` for batch operations
  - `for await...of` support

### Performance Optimization
- ⏳ **Worker thread support**  _(post-0.3.0: binding-toolchain)_
  - Move heavy operations to worker threads
  - Thread pool configuration
  - CPU-bound task offloading

- ⏳ **Buffer pooling**  _(post-0.3.0: binding-toolchain)_
  - Reuse Buffer allocations
  - Reduce GC pressure
  - Configurable pool size

- ⏳ **N-API ThreadSafe functions**  _(post-0.3.0: binding-toolchain)_
  - Proper async callback handling
  - Memory leak prevention
  - Better error propagation

---

## Phase 4: DAG Operations (Priority: Medium)

### IPLD Support
- ⏳ **`dagPut(data, codec)`** - Store IPLD data  _(post-0.3.0: binding-toolchain)_
  - DAG-CBOR encoding
  - DAG-JSON encoding
  - Link preservation

- ⏳ **`dagGet(cid, path?)`** - Get IPLD data with path traversal  _(post-0.3.0: binding-toolchain)_
  - IPLD path resolution
  - Partial DAG fetching

- ⏳ **`dagResolve(cid, path)`** - Resolve IPLD paths  _(post-0.3.0: binding-toolchain)_
  - Cross-block path resolution
  - Link dereferencing

### File System Operations
- ⏳ **`addFile(path)`** - Add file from filesystem  _(post-0.3.0: binding-toolchain)_
  - Chunking support
  - Progress reporting
  - Return UnixFS CID

- ⏳ **`addDirectory(path)`** - Add directory recursively  _(post-0.3.0: binding-toolchain)_
  - Directory listing preservation
  - Symlink handling

- ⏳ **`cat(cid)`** - Output file content  _(post-0.3.0: binding-toolchain)_
  - Streaming output
  - UnixFS support

- ⏳ **`get(cid, outputPath)`** - Export to filesystem  _(post-0.3.0: binding-toolchain)_
  - Directory reconstruction
  - Permission preservation

---

## Phase 5: Advanced Features (Priority: Medium)

### Pinning API
- ⏳ **`pin.add(cid)`** - Pin content  _(post-0.3.0: binding-toolchain)_
- ⏳ **`pin.rm(cid)`** - Unpin content  _(post-0.3.0: binding-toolchain)_
- ⏳ **`pin.ls()`** - List pinned content  _(post-0.3.0: binding-toolchain)_
- ⏳ **Recursive vs direct pinning**  _(post-0.3.0: binding-toolchain)_

### Networking (Future)
- ⏳ **`swarm.peers()`** - List connected peers  _(post-0.3.0: needs-network)_
- ⏳ **`swarm.connect(multiaddr)`** - Connect to peer  _(post-0.3.0: needs-network)_
- ⏳ **`swarm.disconnect(peerId)`** - Disconnect from peer  _(post-0.3.0: needs-network)_
- ⏳ **DHT operations** - findProviders, provide, etc.  _(post-0.3.0: needs-network)_

### Bitswap (Future)
- ⏳ **Block exchange with remote peers**  _(post-0.3.0: needs-network)_
- ⏳ **Wantlist management**  _(post-0.3.0: needs-network)_
- ⏳ **Session-based fetching**  _(post-0.3.0: needs-network)_

---

## Phase 6: Developer Experience (Priority: Medium)

### Documentation
- ⏳ **API reference documentation**  _(post-0.3.0: external-infra)_
  - TypeDoc generation
  - Usage examples for each method
  - Common patterns guide

- ⏳ **Getting started guide**  _(post-0.3.0: external-infra)_
  - Installation instructions
  - Basic usage tutorial
  - Configuration options

### Examples
- ⏳ **Basic block storage example**  _(post-0.3.0: binding-toolchain)_
- ⏳ **Semantic search with embeddings**  _(post-0.3.0: binding-toolchain)_
- ⏳ **Logic programming example**  _(post-0.3.0: binding-toolchain)_
- ⏳ **Express.js integration example**  _(post-0.3.0: binding-toolchain)_
- ⏳ **Next.js/React integration example**  _(post-0.3.0: binding-toolchain)_

### Testing
- ⏳ **Unit tests with Jest/Vitest**  _(post-0.3.0: binding-toolchain)_
  - All public API methods
  - Error conditions
  - Edge cases

- ⏳ **Integration tests**  _(post-0.3.0: binding-toolchain)_
  - Full workflow tests
  - Persistence tests
  - Multi-node scenarios

- ⏳ **Benchmarks**  _(post-0.3.0: binding-toolchain)_
  - Throughput measurements
  - Memory usage profiling
  - Comparison with ipfs-http-client

---

## Phase 7: Publishing & Distribution (Priority: Low)

### npm Package
- ⏳ **Prebuilt binaries**  _(post-0.3.0: external-infra)_
  - Linux x64/arm64
  - macOS x64/arm64 (Apple Silicon)
  - Windows x64

- ⏳ **postinstall fallback compilation**  _(post-0.3.0: external-infra)_
  - Rust toolchain detection
  - Graceful error messages

- ⏳ **Package optimization**  _(post-0.3.0: external-infra)_
  - Minimal package size
  - Proper .npmignore
  - LICENSE and README inclusion

### CI/CD
- ⏳ **GitHub Actions workflow**  _(post-0.3.0: external-infra)_
  - Multi-platform builds
  - Automated npm publishing
  - Version tagging

---

## Future Considerations

### TensorLogic Deep Integration
- ⏳ **Native tensor operations**  _(post-0.3.0: gpu-hardware)_
  - Direct Float32Array/Float64Array support
  - GPU tensor backing (WebGPU)
  - Safetensors format support

- ⏳ **Distributed inference**  _(post-0.3.0: needs-network)_
  - Remote knowledge base queries
  - Proof streaming from network

### WebSocket/gRPC API
- ⏳ **Real-time subscriptions**  _(post-0.3.0: large-subsystem)_
  - Block arrival notifications
  - DHT event streaming
  - Inference result streaming

### ESM/CJS Dual Package
- ⏳ **ES Module support**  _(post-0.3.0: binding-toolchain)_
  - Pure ESM build
  - Named exports
  - Tree-shaking friendly
