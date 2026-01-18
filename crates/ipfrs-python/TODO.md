# ipfrs-python TODO

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
- [ ] **Generate comprehensive type stubs**
  - Full type annotations for all classes
  - Overloaded method signatures
  - Generic types where appropriate

- [ ] **Update `ipfrs.pyi` in ipfrs-interface**
  - Sync with actual Python API
  - Add all new classes and methods
  - Document parameter types and return types

### Docstrings
- [ ] **Add comprehensive docstrings**
  - Google-style docstrings for all public methods
  - Usage examples in docstrings
  - Parameter and return value descriptions

### Context Managers
- [ ] **Implement `__enter__` / `__exit__`**
  - Auto-start on context enter
  - Auto-stop on context exit
  - Exception handling in cleanup

```python
with Node(config) as node:
    cid = node.put_block(data)
```

### Async/Await Support
- [ ] **Add async versions of methods**
  - `async_put_block()`, `async_get_block()`, etc.
  - asyncio integration
  - concurrent.futures fallback

---

## Phase 3: Pythonic API Enhancements (Priority: High)

### Iterator Protocol
- [ ] **Implement `__iter__` for block traversal**
  - Iterate over DAG nodes
  - Lazy loading support

- [ ] **Add async iterators**
  - `async for` support
  - Streaming block retrieval

### Dictionary-like Access
- [ ] **Implement `__getitem__` / `__setitem__`**
  - `node[cid]` for block access
  - `node[cid] = data` for block storage

- [ ] **Implement `__contains__`**
  - `cid in node` for existence check

### Numpy Integration
- [ ] **Native numpy array support for embeddings**
  - Accept `np.ndarray` directly
  - Zero-copy where possible
  - Automatic dtype conversion

- [ ] **Tensor operations with numpy**
  - Return numpy arrays from search results
  - Batch embedding operations

### Pandas Integration
- [ ] **DataFrame support for bulk operations**
  - Add blocks from DataFrame
  - Search results as DataFrame
  - Batch index operations

---

## Phase 4: File Operations (Priority: Medium)

### Path-like Support
- [ ] **Accept `pathlib.Path` objects**
  - Configuration paths
  - Import/export paths
  - Index paths

### File Import/Export
- [ ] **`add_file(path)`** - Add file from filesystem
  - Chunking support
  - Progress callback
  - Return CID

- [ ] **`add_directory(path)`** - Add directory recursively
  - Recursive traversal
  - Pattern filtering (glob)
  - UnixFS directory structure

- [ ] **`cat(cid)`** - Stream file content
  - Return file-like object
  - Lazy chunk loading

- [ ] **`get(cid, output_path)`** - Export to filesystem
  - Directory reconstruction
  - Overwrite handling

### Streaming I/O
- [ ] **File-like object support**
  - Accept `io.BytesIO` for input
  - Return file-like object for output
  - Chunked reading/writing

---

## Phase 5: Advanced TensorLogic (Priority: Medium)

### Enhanced Logic API
- [ ] **Rule builder pattern**
  - Fluent API for complex rules
  - Constraint support

- [ ] **Query DSL**
  - Pythonic query construction
  - Pattern matching syntax

### Proof Serialization
- [ ] **Export proofs to various formats**
  - JSON serialization
  - Graphviz/DOT format
  - IPLD representation

### Distributed Reasoning
- [ ] **Remote knowledge base queries**
  - Federated inference
  - Proof verification from network

---

## Phase 6: Performance & Optimization (Priority: Medium)

### Memory Management
- [ ] **Buffer protocol support**
  - Zero-copy data transfer
  - memoryview compatibility

- [ ] **GIL release for I/O operations**
  - Parallel block operations
  - Background indexing

### Batch Operations
- [ ] **`put_blocks(data_list)`** - Bulk block storage
- [ ] **`get_blocks(cid_list)`** - Bulk block retrieval
- [ ] **`index_batch(cid_embedding_pairs)`** - Batch indexing

### Caching
- [ ] **LRU cache for frequently accessed blocks**
  - Configurable cache size
  - Cache statistics

---

## Phase 7: Documentation & Examples (Priority: Medium)

### Documentation
- [ ] **Sphinx documentation**
  - API reference generation
  - Getting started guide
  - Tutorial sections

- [ ] **Type annotations documentation**
  - mypy compatibility
  - pyright compatibility

### Examples
- [ ] **Basic block storage example**
- [ ] **Semantic search with sentence-transformers**
- [ ] **Logic programming tutorial**
- [ ] **FastAPI integration example**
- [ ] **Jupyter notebook examples**
- [ ] **ML pipeline integration (scikit-learn, PyTorch)**

### Testing
- [ ] **pytest test suite**
  - Unit tests for all public APIs
  - Integration tests
  - Property-based tests (hypothesis)

- [ ] **Performance benchmarks**
  - pytest-benchmark integration
  - Memory profiling
  - Comparison with ipfshttpclient

---

## Phase 8: Publishing & Distribution (Priority: Low)

### PyPI Package
- [ ] **Prebuilt wheels**
  - manylinux2014 x86_64
  - manylinux2014 aarch64
  - macOS x86_64/arm64
  - Windows x86_64

- [ ] **Source distribution**
  - Rust toolchain requirements documented
  - Build from source instructions

### CI/CD
- [ ] **GitHub Actions workflow**
  - Multi-platform wheel building
  - Automated PyPI publishing
  - Test matrix (Python 3.9-3.12)

### Conda Package
- [ ] **conda-forge recipe**
  - Cross-platform support
  - Dependency management

---

## Future Considerations

### Networking Features
- [ ] **Peer discovery and connection**
- [ ] **DHT operations**
- [ ] **Bitswap integration**

### AI/ML Integration
- [ ] **HuggingFace Transformers integration**
  - Automatic embedding generation
  - Model weight storage on IPFRS

- [ ] **LangChain integration**
  - Vector store implementation
  - Document loader

- [ ] **PyTorch/TensorFlow tensor support**
  - Direct tensor storage
  - Safetensors format

### Jupyter Integration
- [ ] **Rich display representations**
  - `_repr_html_()` for blocks
  - Interactive CID explorer
  - Proof tree visualization

### CLI Tool
- [ ] **Python-based CLI wrapper**
  - Click/Typer-based interface
  - Shell completion
