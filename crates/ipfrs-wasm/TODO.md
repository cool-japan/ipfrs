# ipfrs-wasm TODO

> **Status: 0.3.0 "Hardening Release" milestone complete.** Items marked ⏳ below are *post-0.3.0 roadmap* — deferred, not release-blocking — with the deferral reason in `_(post-0.3.0: …)_`. They are **honestly deferred, not claimed as done**. Completed work remains marked ✅ / `- [x]`.

## ✅ Completed (Phase 1: Foundation)

### wasm-bindgen Setup
- ✅ Set up wasm-bindgen for browser bindings
- ✅ Configure package.json for npm publishing
- ✅ Add console_error_panic_hook for debugging

### Core Node Interface
- ✅ **`Node` class** - Main IPFRS node interface
  - Constructor with optional `NodeConfig`
  - `start()` / `stop()` lifecycle methods (placeholder)
  - Synchronous API for WASM constraints

### Configuration (Builder Pattern)
- ✅ **`NodeConfig` class**
  - `new()` - Create default config
  - `setStoragePath(path)` - Set storage path
  - `setEnableSemantic(enable)` - Toggle semantic search
  - `setEnableTensorlogic(enable)` - Toggle logic engine

### Block Operations (Simplified)
- ✅ **`putBlock(data)`** - Store block data
  - Accept Uint8Array as input
  - Return CID string

- ✅ **`hasBlock(cid)`** - Check block existence (placeholder)

### Semantic Search (Placeholder)
- ✅ **`indexContent(cid, embedding)`** - Index content (placeholder)
- ✅ **`searchSimilar(query, k)`** - Vector search (placeholder)

### TensorLogic Integration
- ✅ **`addFact(fact)`** - Add fact to knowledge base
  - Accept JS object with `{name, args}` structure
  - serde_wasm_bindgen deserialization

- ✅ **`addRule(rule)`** - Add inference rule
  - Accept JS object with `{head, body}` structure

- ✅ **`infer(goal)`** - Run backward chaining inference
  - Return array of substitution strings

- ✅ **`kbStats()`** - Get knowledge base statistics
  - Return `{num_facts, num_rules}` object

### JS Type Definitions
- ✅ **`TermJs`** - Logical term (kind + value)
- ✅ **`PredicateJs`** - Predicate with name and args
- ✅ **`RuleJs`** - Rule with head and body
- ✅ **`SearchResult`** - CID and score
- ✅ **`KbStatsJs`** - Knowledge base statistics

### Testing
- ✅ **wasm-bindgen-test setup**
  - `test_node_creation` - Basic node instantiation
  - `test_kb_stats` - TensorLogic integration test

---

## Phase 2: Async Operations (Priority: Critical)

### wasm-bindgen-futures Integration
- ⏳ **Implement proper async/await support**  _(post-0.3.0: binding-toolchain)_
  - Use `wasm_bindgen_futures::spawn_local`
  - Return `Promise` from async methods
  - Proper error propagation

- ⏳ **Async block operations**  _(post-0.3.0: binding-toolchain)_
  - `async putBlock(data)` - Return Promise<string>
  - `async getBlock(cid)` - Return Promise<Uint8Array | null>
  - `async hasBlock(cid)` - Return Promise<boolean>
  - `async deleteBlock(cid)` - Return Promise<void>

- ⏳ **Async semantic operations**  _(post-0.3.0: binding-toolchain)_
  - `async indexContent(cid, embedding)` - Promise<void>
  - `async searchSimilar(query, k)` - Promise<SearchResult[]>

### Web Worker Support
- ⏳ **Offload heavy operations to worker threads**  _(post-0.3.0: binding-toolchain)_
  - Inference in worker
  - Batch operations in worker
  - SharedArrayBuffer for zero-copy transfer

---

## Phase 3: Storage Backend (Priority: High)

### IndexedDB Storage
- ⏳ **Implement IndexedDB-backed blockstore**  _(post-0.3.0: binding-toolchain)_
  - Use `idb` crate or raw JS interop
  - Persistent storage across sessions
  - Transaction support

- ⏳ **Block storage API**  _(post-0.3.0: binding-toolchain)_
  - Store blocks with CID as key
  - Efficient retrieval by CID
  - Batch put/get operations

### LocalStorage Fallback
- ⏳ **Small data fallback for limited browsers**  _(post-0.3.0: binding-toolchain)_
  - Configuration persistence
  - Knowledge base serialization
  - Size limits handling

### Memory-Only Mode
- ⏳ **In-memory storage for ephemeral use**  _(post-0.3.0: binding-toolchain)_
  - Fast operations without persistence
  - Configurable memory limits
  - LRU eviction policy

---

## Phase 4: Full Block API (Priority: High)

### Block Retrieval
- ⏳ **Implement `getBlock(cid)`**  _(post-0.3.0: binding-toolchain)_
  - Return Uint8Array or null
  - DAG-CBOR/DAG-JSON decoding support
  - Error handling

- ⏳ **Implement `deleteBlock(cid)`**  _(post-0.3.0: binding-toolchain)_
  - Remove from storage
  - Reference counting

### Block Verification
- ⏳ **CID verification on retrieve**  _(post-0.3.0: binding-toolchain)_
  - Hash verification
  - Codec validation
  - Version checking

### Block Statistics
- ⏳ **`blockStat(cid)`** - Get block statistics  _(post-0.3.0: binding-toolchain)_
  - Size, codec, hash algorithm
  - Link count for DAG blocks

---

## Phase 5: Semantic Search (Priority: High)

### Vector Index (Browser-Compatible)
- ⏳ **Implement HNSW in pure Rust/WASM**  _(post-0.3.0: large-subsystem)_
  - Memory-efficient implementation
  - Incremental index building
  - Serialization support

- ⏳ **Full semantic search implementation**  _(post-0.3.0: large-subsystem)_
  - `indexContent(cid, embedding)` - Add to index
  - `searchSimilar(query, k)` - K-nearest neighbors
  - `removeFromIndex(cid)` - Remove from index

### Index Persistence
- ⏳ **Save/Load index to IndexedDB**  _(post-0.3.0: binding-toolchain)_
  - Efficient serialization format
  - Incremental updates

### Embedding Utilities
- ⏳ **Provide embedding helpers (optional)**  _(post-0.3.0: external-dep)_
  - Integration with ONNX Runtime Web
  - Pre-built embedding models
  - Batched embedding generation

---

## Phase 6: TensorLogic Enhancement (Priority: Medium)

### Proof Generation & Verification
- ⏳ **`prove(goal)`** - Generate proof tree  _(post-0.3.0: large-subsystem)_
  - Return structured proof object
  - Support for explanation queries

- ⏳ **`verifyProof(proof)`** - Verify proof  _(post-0.3.0: large-subsystem)_
  - Structural validation
  - Step-by-step verification

### Proof Visualization
- ⏳ **Export proof to visualization format**  _(post-0.3.0: external-dep)_
  - JSON structure for D3.js
  - DOT format for Graphviz
  - Mermaid diagram format

### Knowledge Base Persistence
- ⏳ **`saveKb()`** - Serialize KB to bytes  _(post-0.3.0: binding-toolchain)_
  - Return Uint8Array
  - Efficient binary format

- ⏳ **`loadKb(data)`** - Deserialize KB  _(post-0.3.0: binding-toolchain)_
  - Accept Uint8Array
  - Merge or replace options

---

## Phase 7: TypeScript Support (Priority: Medium)

### Type Definitions
- ⏳ **Generate comprehensive `.d.ts` files**  _(post-0.3.0: binding-toolchain)_
  - All public APIs
  - Proper Promise types
  - Union types for errors

- ⏳ **wasm-bindgen TypeScript support**  _(post-0.3.0: binding-toolchain)_
  - Use `--typescript` flag
  - Custom type mappings
  - Branded types for CIDs

### JSDoc Comments
- ⏳ **Add JSDoc to all public methods**  _(post-0.3.0: binding-toolchain)_
  - Parameter descriptions
  - Return type documentation
  - Usage examples

---

## Phase 8: Size & Performance Optimization (Priority: Medium)

### Bundle Size Reduction
- ⏳ **Enable wasm-opt optimization**  _(post-0.3.0: external-infra)_
  - `-Oz` for size optimization
  - Remove debug info in release

- ⏳ **Feature flags for optional modules**  _(post-0.3.0: external-infra)_
  - `--features tensorlogic` only if needed
  - `--features semantic` only if needed
  - Minimal core bundle

- ⏳ **Code splitting considerations**  _(post-0.3.0: external-infra)_
  - Lazy loading of heavy features
  - Dynamic import support

### Performance
- ⏳ **SIMD support for vector operations**  _(post-0.3.0: gpu-hardware)_
  - Enable WASM SIMD where supported
  - Fallback for unsupported browsers

- ⏳ **Memory optimization**  _(post-0.3.0: binding-toolchain)_
  - Streaming operations
  - Avoid large allocations
  - Efficient string handling

### Benchmarks
- ⏳ **Browser performance benchmarks**  _(post-0.3.0: binding-toolchain)_
  - Block put/get throughput
  - Semantic search latency
  - Inference performance
  - Memory usage profiling

---

## Phase 9: Browser Integration (Priority: Low)

### File API Integration
- ⏳ **`addFile(file)`** - Accept File/Blob objects  _(post-0.3.0: binding-toolchain)_
  - Stream large files
  - Progress reporting

- ⏳ **`addDirectory(dirHandle)`** - File System Access API  _(post-0.3.0: binding-toolchain)_
  - Use FileSystemDirectoryHandle
  - Recursive traversal

### Fetch Integration
- ⏳ **`addFromUrl(url)`** - Fetch and add remote content  _(post-0.3.0: binding-toolchain)_
  - Streaming fetch
  - Progress tracking

### Drag & Drop Support
- ⏳ **Helper for drag & drop file handling**  _(post-0.3.0: binding-toolchain)_
  - DataTransfer processing
  - Multiple file support

---

## Phase 10: Framework Examples (Priority: Low)

### Documentation
- ⏳ **API reference documentation**  _(post-0.3.0: external-infra)_
- ⏳ **Getting started guide**  _(post-0.3.0: external-infra)_
- ⏳ **Browser compatibility matrix**  _(post-0.3.0: external-infra)_

### React Example
- ⏳ **React hooks for IPFRS**  _(post-0.3.0: binding-toolchain)_
  - `useIpfrsNode()` hook
  - `useBlock(cid)` hook
  - `useSearch(query)` hook

### Vue Example
- ⏳ **Vue composables**  _(post-0.3.0: binding-toolchain)_
  - `useIpfrs()` composable
  - Reactive block access

### Vanilla JS Example
- ⏳ **Plain JavaScript usage**  _(post-0.3.0: binding-toolchain)_
  - ES module import
  - Script tag inclusion
  - No build tool required

### Service Worker Example
- ⏳ **Offline-first IPFS gateway**  _(post-0.3.0: binding-toolchain)_
  - Intercept fetch requests
  - Serve from local blocks
  - Cache management

---

## Future Considerations

### WebRTC/libp2p-webrtc
- ⏳ **Browser-to-browser networking**  _(post-0.3.0: large-subsystem)_
  - WebRTC data channels
  - NAT traversal
  - Peer discovery

### WebTransport
- ⏳ **Modern transport protocol**  _(post-0.3.0: large-subsystem)_
  - QUIC over WebTransport
  - Better performance than WebSockets

### OPFS (Origin Private File System)
- ⏳ **Native filesystem storage**  _(post-0.3.0: binding-toolchain)_
  - Better performance than IndexedDB
  - Large file support

### Shared Worker Support
- ⏳ **Single node instance across tabs**  _(post-0.3.0: binding-toolchain)_
  - SharedWorker for multi-tab apps
  - Efficient resource usage

### PWA Integration
- ⏳ **Progressive Web App support**  _(post-0.3.0: binding-toolchain)_
  - Offline capability
  - Background sync
  - Push notifications for updates

### WebGPU Acceleration
- ⏳ **GPU-accelerated vector operations**  _(post-0.3.0: gpu-hardware)_
  - WebGPU compute shaders
  - Batch similarity computation
  - Embedding generation on GPU
