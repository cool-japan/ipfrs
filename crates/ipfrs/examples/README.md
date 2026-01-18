# IPFRS Examples

This directory contains comprehensive examples demonstrating IPFRS capabilities and production best practices.

## Quick Start

All examples can be run using `cargo run`:

```bash
cargo run --package ipfrs --example <example_name>
```

## Examples Overview

### 1. Basic Storage (`basic_storage.rs`)

**Purpose**: Learn the fundamentals of IPFRS content-addressed storage.

**Demonstrates**:
- Creating and starting an IPFRS node
- Adding bytes to the network
- Retrieving content by CID
- Checking block existence
- Querying block statistics
- Batch operations

**Run**:
```bash
cargo run --package ipfrs --example basic_storage
```

**Key Concepts**:
- Content addressing (CID generation)
- Block storage and retrieval
- Storage statistics tracking

---

### 2. Semantic Search (`semantic_search.rs`)

**Purpose**: Demonstrate vector search and semantic similarity features.

**Demonstrates**:
- Enabling semantic search
- Indexing content with embeddings
- Performing similarity searches
- Using query filters
- Semantic statistics

**Run**:
```bash
cargo run --package ipfrs --example semantic_search
```

**Key Concepts**:
- Vector embeddings (384-dimensional in example)
- HNSW index for approximate nearest neighbor search
- Semantic content discovery
- Query filtering (min_score, max_results)

**Note**: The example uses a simplified hash-based embedding generator for demonstration. In production, use proper embedding models like:
- `sentence-transformers` (Python)
- `rust-bert`
- OpenAI Embeddings API
- Cohere Embeddings API

---

### 3. Logic Programming (`logic_programming.rs`)

**Purpose**: Explore TensorLogic integration for distributed reasoning.

**Demonstrates**:
- Storing logical terms (variables, constants, functions)
- Defining predicates (facts)
- Creating inference rules
- Performing logical inference
- Content-addressed reasoning

**Run**:
```bash
cargo run --package ipfrs --example logic_programming
```

**Key Concepts**:
- Logical terms: `Var`, `Const`, `Fun`, `Ref`
- Predicates and rules (Datalog-style)
- Forward-chaining inference
- Knowledge base persistence

---

### 4. Production Node (`production_node.rs`)

**Purpose**: Comprehensive example of production-ready IPFRS deployment.

**Demonstrates**:
- **Metrics**: Prometheus metrics for observability
- **Health Checks**: Liveness and readiness probes
- **Distributed Tracing**: OpenTelemetry integration with OTLP export
- **Graceful Shutdown**: Signal handling (SIGTERM, SIGINT) with coordinated cleanup
- **Error Recovery**: Retry logic with exponential backoff
- **Circuit Breaker**: Fault tolerance pattern for search operations

**Run**:
```bash
cargo run --package ipfrs --example production_node
```

**Features**:

#### Prometheus Metrics (`:9000/metrics`)
```bash
# In another terminal
curl http://localhost:9000/metrics
```

Available metrics include:
- Block operations (put, get, delete)
- Semantic search (indexing, queries)
- Logic operations (facts, rules, inference)
- Network operations (peers, messages)
- System metrics (uptime, memory)

#### Health Checks
- **Liveness**: Process is alive and responding
- **Readiness**: All components (storage, network, semantic, logic) are ready

#### Distributed Tracing
Configure OTLP endpoint in code:
```rust
let tracing_config = TracingConfig::new("ipfrs-production".to_string())
    .with_otlp_endpoint("http://localhost:4317".to_string());
```

Traces include:
- Block operations with CID tagging
- Semantic search with k and result counts
- Logic inference with predicate details
- Network operations with peer counts

#### Graceful Shutdown
Press `Ctrl+C` to initiate shutdown:
1. Signal received (SIGTERM or SIGINT)
2. Stop accepting new requests
3. Complete in-flight operations
4. Cleanup resources
5. Wait for background tasks (30s timeout)

#### Error Recovery Patterns

**Retry with Exponential Backoff**:
```rust
let retry_policy = RetryPolicy::exponential(3, Duration::from_millis(100));
retry_async(retry_policy, || async {
    // Your operation here
}).await?;
```

**Circuit Breaker**:
```rust
let breaker = CircuitBreaker::new(5, 2, Duration::from_secs(60));
if breaker.is_available() {
    match operation().await {
        Ok(result) => breaker.record_success(),
        Err(e) => breaker.record_failure(),
    }
}
```

**Production Deployment Checklist**:
- [ ] Configure persistent storage path
- [ ] Set up Prometheus scraping
- [ ] Configure OTLP collector (Jaeger, Tempo, etc.)
- [ ] Set appropriate shutdown timeout
- [ ] Configure retry policies for critical operations
- [ ] Set circuit breaker thresholds
- [ ] Enable structured JSON logging
- [ ] Set proper log levels (info, warn, error)

---

### 5. Load Testing (`load_test.rs`)

**Purpose**: Comprehensive load testing tool for performance validation and benchmarking.

**Demonstrates**:
- **Block Operations**: Throughput testing for put/get operations (1000+ blocks)
- **Semantic Search**: Indexing and search performance at scale (500+ vectors)
- **Logic Inference**: KB operations and inference performance (200+ facts)
- **Mixed Workload**: Combined operations simulating real-world usage
- **Persistence**: Save/load performance for indexes and knowledge bases
- **Performance Metrics**: Detailed latency and throughput statistics

**Run**:
```bash
# IMPORTANT: Run in release mode for accurate performance measurements
cargo run --package ipfrs --example load_test --release
```

**Test Scenarios**:

1. **Block Writes** - Measures throughput for storing blocks
   - Creates 1000 blocks with deterministic content
   - Reports ops/sec, avg/min/max latency

2. **Block Reads** - Measures retrieval performance
   - Reads all previously stored blocks
   - Tests cache effectiveness

3. **Semantic Indexing** - Vector indexing at scale
   - Indexes 500 documents with 768-dim embeddings
   - Measures HNSW index insertion performance

4. **Semantic Search** - Query performance
   - Performs 100 k-NN searches (k=10)
   - Tests approximate nearest neighbor performance

5. **Logic Fact Insertion** - KB write throughput
   - Adds 200 facts to knowledge base
   - Measures predicate storage performance

6. **Logic Inference** - Query evaluation performance
   - Performs 50 inference queries
   - Tests forward-chaining performance

7. **Mixed Workload** - Real-world simulation
   - 300 operations mixing blocks, semantic, and logic
   - Simulates concurrent-style rapid operations

8. **Persistence** - Save/load performance
   - Tests semantic index save/load
   - Tests knowledge base save/load

**Sample Output**:
```
=== Block Writes ===
Total operations: 1000
Duration: 2.45s
Throughput: 408.16 ops/sec
Avg latency: 2.45ms
Min latency: 0.52ms
Max latency: 15.32ms

╔═══════════════════════════════════════════════════════════════╗
║                    LOAD TEST SUMMARY                          ║
╠═══════════════════════════════════════════════════════════════╣
║ Block Writes                        408.16 ops/s              ║
║ Block Reads                         1523.45 ops/s             ║
║ Semantic Indexing                   89.32 ops/s               ║
║ Semantic Search                     456.78 ops/s              ║
║ Logic Fact Insertion                892.15 ops/s              ║
║ Logic Inference                     234.56 ops/s              ║
║ Mixed Workload                      125.67 ops/s              ║
║ Persistence Save/Load               12.34 ops/s               ║
╚═══════════════════════════════════════════════════════════════╝
```

**Performance Tuning**:
- Adjust `LoadTestConfig::default()` for different load profiles
- Modify `num_blocks`, `num_vectors`, `num_facts` for stress testing
- Change `vector_dim` to test different embedding sizes
- Run multiple times to measure variance

**Use Cases**:
- **CI/CD Integration**: Automated performance regression detection
- **Capacity Planning**: Determine hardware requirements
- **Optimization Validation**: Verify performance improvements
- **Production Readiness**: Ensure system meets SLAs

---

### 6. Lazy Loading (`lazy_loading.rs`)

**Purpose**: Demonstrate lazy initialization of components for improved startup performance.

**Demonstrates**:
- **Lazy Component Initialization**: Semantic and TensorLogic initialized on first use
- **Startup Optimization**: Faster node startup by deferring component initialization
- **Memory Efficiency**: Reduced memory footprint when features aren't used
- **Warmup Strategy**: Pre-initialize components for predictable latency
- **Initialization Status**: Check which components are loaded

**Run**:
```bash
cargo run --package ipfrs --example lazy_loading
```

**Key Features**:

#### Lazy Initialization
By default, IPFRS now lazily initializes semantic router and TensorLogic store:
- Components are NOT initialized at node startup
- First access to semantic/logic features triggers initialization
- Improves startup time and reduces memory usage

#### Checking Initialization Status
```rust
// Check if components are enabled in config
node.is_semantic_enabled()      // true if configured
node.is_tensorlogic_enabled()   // true if configured

// Check if components have been initialized
node.is_semantic_initialized()     // true if loaded
node.is_tensorlogic_initialized()  // true if loaded
```

#### Warmup for Predictable Latency
For production scenarios where you want consistent latency:
```rust
// Pre-initialize all configured components
node.warmup()?;

// Now all features have predictable first-access time
```

**Benefits**:
- ✅ **Faster Startup**: Only initialize what you use
- ✅ **Lower Memory**: Components not loaded until needed
- ✅ **Flexible Deployment**: Configure features, pay only for what you use
- ✅ **Predictable Latency**: Optional warmup for production environments

**Use Cases**:
- **CLI Tools**: Fast startup for single-purpose commands
- **Microservices**: Only enable needed features per service
- **Development**: Quick node startup during testing
- **Production**: Use warmup for consistent performance

---

## Common Patterns

### Node Initialization

```rust
use ipfrs::{Node, NodeConfig};
use std::path::PathBuf;

let mut config = NodeConfig::default();
config.storage.path = PathBuf::from("/path/to/storage");
config.enable_semantic = true;
config.enable_tensorlogic = true;

let mut node = Node::new(config)?;
node.start().await?;
```

### Adding Content

```rust
// From bytes
let cid = node.add_bytes(b"Hello, IPFRS!").await?;

// From file
let cid = node.add_file("/path/to/file.txt").await?;

// From directory
let cid = node.add_directory("/path/to/dir").await?;
```

### Retrieving Content

```rust
// Get bytes
if let Some(data) = node.get(&cid).await? {
    println!("Data: {}", String::from_utf8_lossy(&data));
}

// Get to file
node.get_to_file(&cid, "/output/file.txt").await?;

// Get directory
node.get_directory(&cid, "/output/dir").await?;
```

### Semantic Indexing

```rust
// Index content
let cid = node.add_bytes(b"Machine learning content").await?;
let embedding: Vec<f32> = /* generate embedding */;
node.index_content(&cid, &embedding).await?;

// Search
let query_embedding: Vec<f32> = /* generate embedding */;
let results = node.search_similar(&query_embedding, 10).await?;

for result in results {
    println!("CID: {}, Score: {}", result.cid, result.score);
}
```

### Logic Programming

```rust
use ipfrs::{Constant, Predicate, Rule, Term};

// Add facts
let fact = Predicate::new(
    "parent".to_string(),
    vec![
        Term::Const(Constant::String("Alice".to_string())),
        Term::Const(Constant::String("Bob".to_string())),
    ],
);
node.add_fact(fact)?;

// Add rules
let rule = Rule::new(
    Predicate::new(
        "ancestor".to_string(),
        vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
    ),
    vec![
        Predicate::new(
            "parent".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
        ),
    ],
);
node.add_rule(rule)?;

// Perform inference
let goal = Predicate::new(
    "ancestor".to_string(),
    vec![
        Term::Var("X".to_string()),
        Term::Const(Constant::String("Bob".to_string())),
    ],
);
let results = node.infer(&goal)?;
```

## Troubleshooting

### Storage Permission Errors

Ensure the storage directory is writable:
```bash
mkdir -p /tmp/ipfrs-storage
chmod 755 /tmp/ipfrs-storage
```

### Prometheus Metrics Port In Use

Change the metrics port in the example:
```rust
let metrics_addr = "127.0.0.1:9001".parse()?; // Changed from 9000
```

### OTLP Collector Connection Refused

Ensure your OpenTelemetry collector is running:
```bash
# Example with Jaeger all-in-one
docker run -d --name jaeger \
  -e COLLECTOR_OTLP_ENABLED=true \
  -p 4317:4317 \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest
```

Then access Jaeger UI at `http://localhost:16686`

### High Memory Usage with Semantic Search

Adjust the HNSW index parameters in `RouterConfig`:
```rust
let mut config = NodeConfig::default();
config.semantic.max_connections = 16;  // Default: 32
config.semantic.cache_size = 100;      // Default: 1000
```

## Next Steps

1. **Explore the API**: Check `crates/ipfrs/src/lib.rs` for all available functions
2. **Read the docs**: Run `cargo doc --open` to view detailed API documentation
3. **Join the community**: Visit our repository for discussions and issues
4. **Contribute**: Found a bug or have a feature idea? We welcome contributions!

## Performance Tips

1. **Batch Operations**: Use batch APIs when adding/retrieving multiple blocks
2. **Connection Pooling**: Reuse `Node` instances across requests
3. **Caching**: Enable semantic cache for repeated queries
4. **Indexing**: Index content asynchronously in background tasks
5. **Monitoring**: Use metrics to identify bottlenecks

## Security Considerations

1. **Storage Isolation**: Use dedicated storage paths per application
2. **Content Validation**: Verify CIDs match expected content
3. **Resource Limits**: Set appropriate timeouts and size limits
4. **Network Security**: Configure firewalls for p2p communication
5. **Access Control**: Implement authentication for HTTP endpoints (metrics, health)

---

For more information, visit the [IPFRS documentation](https://github.com/cool-japan/ipfrs).
