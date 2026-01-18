# ipfrs-nodejs TODO

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
- [ ] **Generate comprehensive `.d.ts` files**
  - All public APIs with full type signatures
  - JSDoc comments for IntelliSense
  - Generic types where appropriate

- [ ] **Add branded types for safety**
  - `CidString` type for validated CID strings
  - `EmbeddingVector` type for float arrays
  - Type guards for runtime validation

### Error Handling
- [ ] **Custom error classes**
  - `IpfrsError` base class
  - `NetworkError`, `StorageError`, `LogicError` subclasses
  - Error codes for programmatic handling
  - Stack trace preservation

- [ ] **Typed error returns**
  - Result-like types for operations that can fail
  - Discriminated unions for error handling

---

## Phase 3: Streaming & Performance (Priority: High)

### Streaming API
- [ ] **Implement streaming block upload**
  - Accept `ReadableStream` for large files
  - Progress callbacks during upload
  - Chunked transfer support

- [ ] **Implement streaming block download**
  - Return `ReadableStream` for large blocks
  - DAG traversal with streaming
  - Memory-efficient for large files

- [ ] **Add async iterators**
  - `AsyncIterator<Block>` for batch operations
  - `for await...of` support

### Performance Optimization
- [ ] **Worker thread support**
  - Move heavy operations to worker threads
  - Thread pool configuration
  - CPU-bound task offloading

- [ ] **Buffer pooling**
  - Reuse Buffer allocations
  - Reduce GC pressure
  - Configurable pool size

- [ ] **N-API ThreadSafe functions**
  - Proper async callback handling
  - Memory leak prevention
  - Better error propagation

---

## Phase 4: DAG Operations (Priority: Medium)

### IPLD Support
- [ ] **`dagPut(data, codec)`** - Store IPLD data
  - DAG-CBOR encoding
  - DAG-JSON encoding
  - Link preservation

- [ ] **`dagGet(cid, path?)`** - Get IPLD data with path traversal
  - IPLD path resolution
  - Partial DAG fetching

- [ ] **`dagResolve(cid, path)`** - Resolve IPLD paths
  - Cross-block path resolution
  - Link dereferencing

### File System Operations
- [ ] **`addFile(path)`** - Add file from filesystem
  - Chunking support
  - Progress reporting
  - Return UnixFS CID

- [ ] **`addDirectory(path)`** - Add directory recursively
  - Directory listing preservation
  - Symlink handling

- [ ] **`cat(cid)`** - Output file content
  - Streaming output
  - UnixFS support

- [ ] **`get(cid, outputPath)`** - Export to filesystem
  - Directory reconstruction
  - Permission preservation

---

## Phase 5: Advanced Features (Priority: Medium)

### Pinning API
- [ ] **`pin.add(cid)`** - Pin content
- [ ] **`pin.rm(cid)`** - Unpin content
- [ ] **`pin.ls()`** - List pinned content
- [ ] **Recursive vs direct pinning**

### Networking (Future)
- [ ] **`swarm.peers()`** - List connected peers
- [ ] **`swarm.connect(multiaddr)`** - Connect to peer
- [ ] **`swarm.disconnect(peerId)`** - Disconnect from peer
- [ ] **DHT operations** - findProviders, provide, etc.

### Bitswap (Future)
- [ ] **Block exchange with remote peers**
- [ ] **Wantlist management**
- [ ] **Session-based fetching**

---

## Phase 6: Developer Experience (Priority: Medium)

### Documentation
- [ ] **API reference documentation**
  - TypeDoc generation
  - Usage examples for each method
  - Common patterns guide

- [ ] **Getting started guide**
  - Installation instructions
  - Basic usage tutorial
  - Configuration options

### Examples
- [ ] **Basic block storage example**
- [ ] **Semantic search with embeddings**
- [ ] **Logic programming example**
- [ ] **Express.js integration example**
- [ ] **Next.js/React integration example**

### Testing
- [ ] **Unit tests with Jest/Vitest**
  - All public API methods
  - Error conditions
  - Edge cases

- [ ] **Integration tests**
  - Full workflow tests
  - Persistence tests
  - Multi-node scenarios

- [ ] **Benchmarks**
  - Throughput measurements
  - Memory usage profiling
  - Comparison with ipfs-http-client

---

## Phase 7: Publishing & Distribution (Priority: Low)

### npm Package
- [ ] **Prebuilt binaries**
  - Linux x64/arm64
  - macOS x64/arm64 (Apple Silicon)
  - Windows x64

- [ ] **postinstall fallback compilation**
  - Rust toolchain detection
  - Graceful error messages

- [ ] **Package optimization**
  - Minimal package size
  - Proper .npmignore
  - LICENSE and README inclusion

### CI/CD
- [ ] **GitHub Actions workflow**
  - Multi-platform builds
  - Automated npm publishing
  - Version tagging

---

## Future Considerations

### TensorLogic Deep Integration
- [ ] **Native tensor operations**
  - Direct Float32Array/Float64Array support
  - GPU tensor backing (WebGPU)
  - Safetensors format support

- [ ] **Distributed inference**
  - Remote knowledge base queries
  - Proof streaming from network

### WebSocket/gRPC API
- [ ] **Real-time subscriptions**
  - Block arrival notifications
  - DHT event streaming
  - Inference result streaming

### ESM/CJS Dual Package
- [ ] **ES Module support**
  - Pure ESM build
  - Named exports
  - Tree-shaking friendly
