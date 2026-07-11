# ipfrs-python TODO

> **Status: 0.3.0 "Hardening Release" milestone complete.** Items marked ⏳ below are *post-0.3.0 roadmap* — deferred, not release-blocking — with the deferral reason in `_(post-0.3.0: …)_`. They are **honestly deferred, not claimed as done**. Completed work remains marked ✅ / `- [x]`.

## ✅ Completed (Phase 1: Foundation)

### PyO3 Binding Setup
- ✅ Set up PyO3 for Python bindings
- ✅ Configure maturin for wheel building
- ✅ Create pyproject.toml with package metadata

### Core Node Interface
- ✅ **`Node` class** - Main IPFRS node interface
  - Constructor with optional `NodeConfig`
  - `start()` / `stop()` lifecycle methods
  - Tokio runtime integration for blocking operations

### Configuration
- ✅ **`NodeConfig` class**
  - `storage_path` - Path to storage directory
  - `enable_semantic` - Enable semantic search
  - `enable_tensorlogic` - Enable logic engine
  - `default()` static method

### Block Operations
- ✅ **`put_block(data)`** - Store block data
  - Accept bytes as input
  - Return `Cid` object

- ✅ **`get_block(cid)`** - Retrieve block data
  - Return `Block` or None
  - `Block.data()` method for bytes access

- ✅ **`has_block(cid)`** - Check block existence
- ✅ **`delete_block(cid)`** - Remove block from storage

### Block & CID Types
- ✅ **`Block` class**
  - `data()` - Get block bytes
  - `cid()` - Get block CID
  - `size()` - Get block size

- ✅ **`Cid` class**
  - `parse(s)` - Parse CID from string
  - `__str__()` / `__repr__()` - String representations

### Semantic Search
- ✅ **`index_content(cid, embedding)`** - Index content with vector
- ✅ **`search_similar(query, k)`** - Vector similarity search
- ✅ **`search_filtered(query, k, filter)`** - Filtered search with `Filter`
- ✅ **`save_semantic_index(path)`** - Persist index to disk
- ✅ **`load_semantic_index(path)`** - Load index from disk

### TensorLogic Integration
- ✅ **`add_fact(predicate)`** - Add fact to knowledge base
- ✅ **`add_rule(rule)`** - Add inference rule
- ✅ **`infer(goal)`** - Run backward chaining inference
- ✅ **`prove(goal)`** - Generate proof tree
- ✅ **`verify_proof(proof)`** - Verify proof validity
- ✅ **`kb_stats()`** - Get knowledge base statistics (dict)
- ✅ **`save_kb(path)`** / **`load_kb(path)`** - Knowledge base persistence

### Logic Types
- ✅ **`Term` class**
  - `int(value)`, `float(value)`, `string(value)`, `bool(value)` - Constants
  - `var(name)` - Variables

- ✅ **`Predicate` class**
  - Constructor with name and args list

- ✅ **`Rule` class**
  - `fact(head)` - Create a fact
  - `rule(head, body)` - Create a rule with body

- ✅ **`Proof` class** - Proof tree wrapper
- ✅ **`Substitution` class** - Variable bindings with `bindings()` method
- ✅ **`Filter` class** - Search filter with `min_score`, `max_score`, `max_results`

---

## Phase 2: Type Stubs & Developer Experience (Priority: High)

### Type Stubs (.pyi files)
- ⏳ **Generate comprehensive type stubs**  _(post-0.3.0: binding-toolchain)_
  - Full type annotations for all classes
  - Overloaded method signatures
  - Generic types where appropriate

- ⏳ **Update `ipfrs.pyi` in ipfrs-interface**  _(post-0.3.0: binding-toolchain)_
  - Sync with actual Python API
  - Add all new classes and methods
  - Document parameter types and return types

### Docstrings
- ⏳ **Add comprehensive docstrings**  _(post-0.3.0: binding-toolchain)_
  - Google-style docstrings for all public methods
  - Usage examples in docstrings
  - Parameter and return value descriptions

### Context Managers
- ⏳ **Implement `__enter__` / `__exit__`**  _(post-0.3.0: binding-toolchain)_
  - Auto-start on context enter
  - Auto-stop on context exit
  - Exception handling in cleanup

```python
with Node(config) as node:
    cid = node.put_block(data)
```

### Async/Await Support
- ⏳ **Add async versions of methods**  _(post-0.3.0: binding-toolchain)_
  - `async_put_block()`, `async_get_block()`, etc.
  - asyncio integration
  - concurrent.futures fallback

---

## Phase 3: Pythonic API Enhancements (Priority: High)

### Iterator Protocol
- ⏳ **Implement `__iter__` for block traversal**  _(post-0.3.0: binding-toolchain)_
  - Iterate over DAG nodes
  - Lazy loading support

- ⏳ **Add async iterators**  _(post-0.3.0: binding-toolchain)_
  - `async for` support
  - Streaming block retrieval

### Dictionary-like Access
- ⏳ **Implement `__getitem__` / `__setitem__`**  _(post-0.3.0: binding-toolchain)_
  - `node[cid]` for block access
  - `node[cid] = data` for block storage

- ⏳ **Implement `__contains__`**  _(post-0.3.0: binding-toolchain)_
  - `cid in node` for existence check

### Numpy Integration
- ⏳ **Native numpy array support for embeddings**  _(post-0.3.0: binding-toolchain)_
  - Accept `np.ndarray` directly
  - Zero-copy where possible
  - Automatic dtype conversion

- ⏳ **Tensor operations with numpy**  _(post-0.3.0: binding-toolchain)_
  - Return numpy arrays from search results
  - Batch embedding operations

### Pandas Integration
- ⏳ **DataFrame support for bulk operations**  _(post-0.3.0: binding-toolchain)_
  - Add blocks from DataFrame
  - Search results as DataFrame
  - Batch index operations

---

## Phase 4: File Operations (Priority: Medium)

### Path-like Support
- ⏳ **Accept `pathlib.Path` objects**  _(post-0.3.0: binding-toolchain)_
  - Configuration paths
  - Import/export paths
  - Index paths

### File Import/Export
- ⏳ **`add_file(path)`** - Add file from filesystem  _(post-0.3.0: binding-toolchain)_
  - Chunking support
  - Progress callback
  - Return CID

- ⏳ **`add_directory(path)`** - Add directory recursively  _(post-0.3.0: binding-toolchain)_
  - Recursive traversal
  - Pattern filtering (glob)
  - UnixFS directory structure

- ⏳ **`cat(cid)`** - Stream file content  _(post-0.3.0: binding-toolchain)_
  - Return file-like object
  - Lazy chunk loading

- ⏳ **`get(cid, output_path)`** - Export to filesystem  _(post-0.3.0: binding-toolchain)_
  - Directory reconstruction
  - Overwrite handling

### Streaming I/O
- ⏳ **File-like object support**  _(post-0.3.0: binding-toolchain)_
  - Accept `io.BytesIO` for input
  - Return file-like object for output
  - Chunked reading/writing

---

## Phase 5: Advanced TensorLogic (Priority: Medium)

### Enhanced Logic API
- ⏳ **Rule builder pattern**  _(post-0.3.0: binding-toolchain)_
  - Fluent API for complex rules
  - Constraint support

- ⏳ **Query DSL**  _(post-0.3.0: binding-toolchain)_
  - Pythonic query construction
  - Pattern matching syntax

### Proof Serialization
- ⏳ **Export proofs to various formats**  _(post-0.3.0: binding-toolchain)_
  - JSON serialization
  - Graphviz/DOT format
  - IPLD representation

### Distributed Reasoning
- ⏳ **Remote knowledge base queries**  _(post-0.3.0: needs-network)_
  - Federated inference
  - Proof verification from network

---

## Phase 6: Performance & Optimization (Priority: Medium)

### Memory Management
- ⏳ **Buffer protocol support**  _(post-0.3.0: binding-toolchain)_
  - Zero-copy data transfer
  - memoryview compatibility

- ⏳ **GIL release for I/O operations**  _(post-0.3.0: binding-toolchain)_
  - Parallel block operations
  - Background indexing

### Batch Operations
- ⏳ **`put_blocks(data_list)`** - Bulk block storage  _(post-0.3.0: binding-toolchain)_
- ⏳ **`get_blocks(cid_list)`** - Bulk block retrieval  _(post-0.3.0: binding-toolchain)_
- ⏳ **`index_batch(cid_embedding_pairs)`** - Batch indexing  _(post-0.3.0: binding-toolchain)_

### Caching
- ⏳ **LRU cache for frequently accessed blocks**  _(post-0.3.0: binding-toolchain)_
  - Configurable cache size
  - Cache statistics

---

## Phase 7: Documentation & Examples (Priority: Medium)

### Documentation
- ⏳ **Sphinx documentation**  _(post-0.3.0: external-infra)_
  - API reference generation
  - Getting started guide
  - Tutorial sections

- ⏳ **Type annotations documentation**  _(post-0.3.0: external-infra)_
  - mypy compatibility
  - pyright compatibility

### Examples
- ⏳ **Basic block storage example**  _(post-0.3.0: binding-toolchain)_
- ⏳ **Semantic search with sentence-transformers**  _(post-0.3.0: external-dep)_
- ⏳ **Logic programming tutorial**  _(post-0.3.0: binding-toolchain)_
- ⏳ **FastAPI integration example**  _(post-0.3.0: external-dep)_
- ⏳ **Jupyter notebook examples**  _(post-0.3.0: binding-toolchain)_
- ⏳ **ML pipeline integration (scikit-learn, PyTorch)**  _(post-0.3.0: external-dep)_

### Testing
- ⏳ **pytest test suite**  _(post-0.3.0: binding-toolchain)_
  - Unit tests for all public APIs
  - Integration tests
  - Property-based tests (hypothesis)

- ⏳ **Performance benchmarks**  _(post-0.3.0: binding-toolchain)_
  - pytest-benchmark integration
  - Memory profiling
  - Comparison with ipfshttpclient

---

## Phase 8: Publishing & Distribution (Priority: Low)

### PyPI Package
- ⏳ **Prebuilt wheels**  _(post-0.3.0: external-infra)_
  - manylinux2014 x86_64
  - manylinux2014 aarch64
  - macOS x86_64/arm64
  - Windows x86_64

- ⏳ **Source distribution**  _(post-0.3.0: external-infra)_
  - Rust toolchain requirements documented
  - Build from source instructions

### CI/CD
- ⏳ **GitHub Actions workflow**  _(post-0.3.0: external-infra)_
  - Multi-platform wheel building
  - Automated PyPI publishing
  - Test matrix (Python 3.9-3.12)

### Conda Package
- ⏳ **conda-forge recipe**  _(post-0.3.0: external-infra)_
  - Cross-platform support
  - Dependency management

---

## Future Considerations

### Networking Features
- ⏳ **Peer discovery and connection**  _(post-0.3.0: needs-network)_
- ⏳ **DHT operations**  _(post-0.3.0: needs-network)_
- ⏳ **Bitswap integration**  _(post-0.3.0: needs-network)_

### AI/ML Integration
- ⏳ **HuggingFace Transformers integration**  _(post-0.3.0: external-dep)_
  - Automatic embedding generation
  - Model weight storage on IPFRS

- ⏳ **LangChain integration**  _(post-0.3.0: external-dep)_
  - Vector store implementation
  - Document loader

- ⏳ **PyTorch/TensorFlow tensor support**  _(post-0.3.0: external-dep)_
  - Direct tensor storage
  - Safetensors format

### Jupyter Integration
- ⏳ **Rich display representations**  _(post-0.3.0: binding-toolchain)_
  - `_repr_html_()` for blocks
  - Interactive CID explorer
  - Proof tree visualization

### CLI Tool
- ⏳ **Python-based CLI wrapper**  _(post-0.3.0: binding-toolchain)_
  - Click/Typer-based interface
  - Shell completion
