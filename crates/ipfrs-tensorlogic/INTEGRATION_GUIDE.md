# IPFRS TensorLogic Integration Guide

This guide provides comprehensive documentation for integrating and using the IPFRS TensorLogic system.

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Getting Started](#getting-started)
4. [Zero-Copy Tensor Operations](#zero-copy-tensor-operations)
5. [Distributed Reasoning](#distributed-reasoning)
6. [Gradient Management](#gradient-management)
7. [Model Version Control](#model-version-control)
8. [Performance Optimization](#performance-optimization)
9. [Device-Aware Operations](#device-aware-operations)
10. [Memory Profiling](#memory-profiling)
11. [Best Practices](#best-practices)
12. [Troubleshooting](#troubleshooting)

## Overview

IPFRS TensorLogic is a comprehensive system that integrates logic programming with tensor operations, providing:

- **Content-Addressed Storage**: All logical terms and tensor data are stored using IPFS CIDs
- **Zero-Copy Operations**: Efficient tensor access using Apache Arrow and Safetensors
- **Distributed Reasoning**: Query caching, goal decomposition, and proof assembly across nodes
- **Federated Learning**: Gradient compression, aggregation, and differential privacy
- **Version Control**: Git-like versioning for ML models with commit, branch, and merge
- **Performance Tools**: FFI profiling, allocation optimization, and memory tracking

## Core Concepts

### Terms and Predicates

Terms are the basic building blocks:

```rust
use ipfrs_tensorlogic::{Term, Constant};

// Constants
let alice = Term::Const(Constant::String("Alice".to_string()));
let age = Term::Const(Constant::Int(30));
let score = Term::Const(Constant::Float(0.95));

// Variables
let x = Term::Var("X".to_string());
```

Predicates represent relationships:

```rust
use ipfrs_tensorlogic::Predicate;

// person(Alice, 30)
let pred = Predicate::new(
    "person".to_string(),
    vec![alice, age]
);
```

### Knowledge Base

Store facts and rules:

```rust
use ipfrs_tensorlogic::{KnowledgeBase, Rule};

let mut kb = KnowledgeBase::new();

// Add facts
kb.add_fact(Predicate::new("parent".to_string(), vec![alice, bob]));

// Add rules: ancestor(X, Y) :- parent(X, Y)
kb.add_rule(Rule::new(
    Predicate::new("ancestor".to_string(), vec![x.clone(), y.clone()]),
    vec![Predicate::new("parent".to_string(), vec![x, y])]
));
```

### Inference Engine

Query the knowledge base:

```rust
use ipfrs_tensorlogic::InferenceEngine;

let engine = InferenceEngine::new();
let query = Predicate::new("ancestor".to_string(), vec![alice, Term::Var("Y".to_string())]);
let results = engine.query(&query, &kb).unwrap();
```

## Getting Started

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ipfrs-tensorlogic = "0.1"
```

### Basic Example

```rust
use ipfrs_tensorlogic::{
    KnowledgeBase, Predicate, Term, Constant, InferenceEngine, Rule
};

fn main() {
    let mut kb = KnowledgeBase::new();

    // Add facts
    kb.add_fact(Predicate::new(
        "parent".to_string(),
        vec![
            Term::Const(Constant::String("Alice".to_string())),
            Term::Const(Constant::String("Bob".to_string()))
        ]
    ));

    // Add rule
    kb.add_rule(Rule::new(
        Predicate::new(
            "ancestor".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())]
        ),
        vec![Predicate::new(
            "parent".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())]
        )]
    ));

    // Query
    let engine = InferenceEngine::new();
    let query = Predicate::new(
        "ancestor".to_string(),
        vec![
            Term::Const(Constant::String("Alice".to_string())),
            Term::Var("Y".to_string())
        ]
    );

    let results = engine.query(&query, &kb).unwrap();
    println!("Found {} results", results.len());
}
```

## Zero-Copy Tensor Operations

### Apache Arrow Integration

Create tensors with zero-copy access:

```rust
use ipfrs_tensorlogic::{ArrowTensor, ArrowTensorStore};

// Create tensor from f32 data
let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
let tensor = ArrowTensor::from_slice_f32("weights", vec![2, 2], &data);

// Zero-copy access
let slice = tensor.as_slice_f32().unwrap();
assert_eq!(slice[0], 1.0);

// Store multiple tensors
let mut store = ArrowTensorStore::new();
store.insert(tensor);

// Serialize to IPC format
let bytes = store.to_bytes().unwrap();

// Deserialize
let loaded_store = ArrowTensorStore::from_bytes(&bytes).unwrap();
```

### Safetensors Support

Work with Safetensors format:

```rust
use ipfrs_tensorlogic::{SafetensorsWriter, SafetensorsReader};
use bytes::Bytes;

// Write tensors
let mut writer = SafetensorsWriter::new();
writer.add_f32("layer1.weight", vec![128, 64], &vec![0.1; 8192]);
writer.add_f32("layer1.bias", vec![64], &vec![0.01; 64]);

let bytes = writer.serialize().unwrap();

// Read tensors
let reader = SafetensorsReader::from_bytes(Bytes::from(bytes)).unwrap();
let weight_tensor = reader.load_as_arrow("layer1.weight").unwrap();

// Zero-copy access
let weights = weight_tensor.as_slice_f32().unwrap();
```

### Shared Memory

Share tensors across processes:

```rust
use ipfrs_tensorlogic::{SharedTensorBuffer, SharedMemoryPool};

// Create shared buffer
let data: Vec<f32> = vec![1.0; 1000];
let buffer = SharedTensorBuffer::create("my_tensor", vec![10, 100], &data).unwrap();

// In another process, open read-only
let readonly = SharedTensorBuffer::open_readonly("my_tensor").unwrap();
let shared_data = readonly.as_slice_f32().unwrap();
```

## Distributed Reasoning

### Query Caching

Cache query results for performance:

```rust
use ipfrs_tensorlogic::{QueryCache, QueryKey};

let cache = QueryCache::new(1000); // Capacity 1000

// Cache a query result
let key = QueryKey {
    predicate_name: "ancestor".to_string(),
    ground_args: vec![],
};
cache.insert(key.clone(), results);

// Retrieve from cache
if let Some(cached) = cache.get(&key) {
    // Use cached results
}
```

### Remote Fact Caching

Cache facts from remote sources:

```rust
use ipfrs_tensorlogic::{RemoteFactCache, CacheManager};
use std::time::Duration;

// Create cache with TTL
let fact_cache = RemoteFactCache::new(1000, Duration::from_secs(300));

// Cache facts from a predicate
fact_cache.insert("parent".to_string(), facts);

// Retrieve cached facts
if let Some(cached) = fact_cache.get("parent") {
    // Use cached facts
}
```

### Goal Decomposition

Decompose complex queries:

```rust
use ipfrs_tensorlogic::GoalDecomposition;

let decomposition = GoalDecomposition::new(goal.clone(), rule_id.clone());
decomposition.add_subgoal(subgoal1, vec!["X"]);
decomposition.add_subgoal(subgoal2, vec!["Y"]);

// Mark subgoals as solved
decomposition.mark_solved(0);
```

### Proof Assembly

Assemble distributed proofs:

```rust
use ipfrs_tensorlogic::{ProofAssembler, ProofFragmentStore};

let store = ProofFragmentStore::new();
let assembler = ProofAssembler::new(store);

// Assemble proof from fragments
let proof_tree = assembler.assemble(&conclusion_predicate).await.unwrap();

// Verify proof
assert!(assembler.verify_proof(&proof_tree).unwrap());
```

## Gradient Management

### Gradient Compression

Compress gradients for efficient transmission:

```rust
use ipfrs_tensorlogic::GradientCompressor;

let gradient: Vec<f32> = vec![0.1, 0.5, 0.01, 0.8, 0.02];

// Top-k compression (keep largest k values)
let sparse = GradientCompressor::top_k(&gradient, vec![5], 2).unwrap();
println!("Compression ratio: {:.2}x", sparse.compression_ratio());

// Threshold compression (keep values above threshold)
let sparse2 = GradientCompressor::threshold(&gradient, vec![5], 0.1);

// Quantization to int8
let quantized = GradientCompressor::quantize(&gradient, vec![5]);
```

### Gradient Aggregation

Aggregate gradients from multiple sources:

```rust
use ipfrs_tensorlogic::GradientAggregator;

let grad1 = vec![1.0, 2.0, 3.0];
let grad2 = vec![0.5, 1.5, 2.5];
let gradients = vec![grad1, grad2];

// Simple averaging
let avg = GradientAggregator::average(&gradients).unwrap();

// Weighted aggregation
let weights = vec![0.6, 0.4];
let weighted = GradientAggregator::weighted(&gradients, &weights).unwrap();

// With momentum
let momentum = vec![0.1, 0.1, 0.1];
let with_momentum = GradientAggregator::with_momentum(&avg, &momentum, 0.9).unwrap();
```

### Differential Privacy

Add privacy guarantees:

```rust
use ipfrs_tensorlogic::{DifferentialPrivacy, DPMechanism, PrivacyBudget};

// Create DP mechanism
let mut dp = DifferentialPrivacy::new(
    1.0,  // epsilon
    1e-5, // delta
    DPMechanism::Gaussian,
);

// Add noise to gradient
let mut gradient = vec![1.0, 2.0, 3.0];
dp.add_gaussian_noise(&mut gradient).unwrap();

// Check privacy budget
let budget = dp.privacy_budget();
println!("Remaining budget: ε={}, δ={}", budget.epsilon, budget.delta);
```

### Federated Learning

Coordinate federated learning rounds:

```rust
use ipfrs_tensorlogic::{ModelSyncProtocol, ClientInfo, ConvergenceDetector};

let mut protocol = ModelSyncProtocol::new(10, 5); // 10 clients, min 5

// Register clients
protocol.register_client(ClientInfo {
    client_id: "client1".to_string(),
    device_type: DeviceType::Consumer,
});

// Start round
let round_id = protocol.start_round().unwrap();

// Submit gradient
protocol.submit_gradient(&round_id, "client1", gradient).unwrap();

// Finalize when enough clients submitted
if protocol.can_finalize_round(&round_id) {
    let aggregated = protocol.finalize_round(&round_id).unwrap();
}
```

## Model Version Control

### Commits and Checkouts

Version your models:

```rust
use ipfrs_tensorlogic::{ModelRepository, ModelCommit};
use std::collections::HashMap;

let mut repo = ModelRepository::init("my_model").unwrap();

// Create initial commit
let mut layers = HashMap::new();
layers.insert("layer1".to_string(), vec![0.1; 1000]);
let commit1 = repo.commit(layers.clone(), "Initial model").unwrap();

// Make changes
layers.insert("layer2".to_string(), vec![0.2; 500]);
let commit2 = repo.commit(layers, "Add layer2").unwrap();

// Checkout previous version
repo.checkout(&commit1).unwrap();
```

### Branching

Create and manage branches:

```rust
// Create branch
repo.create_branch("experiment", Some(commit1.clone())).unwrap();

// Switch to branch
repo.checkout_branch("experiment").unwrap();

// List branches
let branches = repo.list_branches();
```

### Merging

Merge branches:

```rust
// Fast-forward merge
if repo.can_fast_forward("main", "experiment").unwrap() {
    repo.merge_fast_forward("experiment").unwrap();
}
```

### Diffing

Compare model versions:

```rust
use ipfrs_tensorlogic::ModelDiffer;

let differ = ModelDiffer::new();
let diff = differ.diff(&old_layers, &new_layers);

println!("Added layers: {:?}", diff.added);
println!("Modified layers: {:?}", diff.modified);
println!("Removed layers: {:?}", diff.removed);
```

## Performance Optimization

### Buffer Pooling

Reuse buffers to reduce allocations:

```rust
use ipfrs_tensorlogic::BufferPool;

let pool = BufferPool::new(4096, 16); // 4KB buffers, max 16 pooled

// Acquire buffer
let mut buffer = pool.acquire();
buffer.as_mut().extend_from_slice(&[1, 2, 3, 4]);

// Buffer returned to pool when dropped
```

### Zero-Copy Conversions

Convert between types without copying:

```rust
use ipfrs_tensorlogic::ZeroCopyConverter;

let floats: Vec<f32> = vec![1.0, 2.0, 3.0];

// Zero-copy to bytes
let bytes = ZeroCopyConverter::slice_to_bytes(&floats);

// Zero-copy back
let floats_back: &[f32] = ZeroCopyConverter::bytes_to_slice(bytes);
```

### Query Optimization

Optimize query execution:

```rust
use ipfrs_tensorlogic::QueryOptimizer;

let optimizer = QueryOptimizer::new();

// Collect statistics
optimizer.update_stats("parent", 1000);
optimizer.update_stats("sibling", 500);

// Plan query
let goals = vec![goal1, goal2];
let plan = optimizer.plan_query(&goals, &kb);

// Use optimized order
for node in plan.nodes {
    // Execute in optimized order
}
```

### Materialized Views

Cache common query patterns:

```rust
use ipfrs_tensorlogic::MaterializedViewManager;
use std::time::Duration;

let mut manager = MaterializedViewManager::new();

// Create view with TTL
let view_id = manager.create_view(
    "common_ancestors",
    query.clone(),
    results,
    Duration::from_secs(300)
);

// Query view
if let Some(cached) = manager.get_view(&query) {
    // Use cached results
} else {
    // Execute query and cache
}
```

## Device-Aware Operations

### Device Capabilities

Detect device capabilities:

```rust
use ipfrs_tensorlogic::DeviceCapabilities;

let caps = DeviceCapabilities::detect().unwrap();
println!("Device: {:?}", caps.device_type);
println!("CPU cores: {}", caps.cpu_info.total_cores);
println!("Memory: {} GB", caps.memory.total_bytes / 1024 / 1024 / 1024);
println!("GPU available: {}", caps.gpu_info.is_some());
```

### Adaptive Batch Sizing

Adjust batch sizes based on device:

```rust
use ipfrs_tensorlogic::AdaptiveBatchSizer;
use std::sync::Arc;

let caps = DeviceCapabilities::detect().unwrap();
let sizer = AdaptiveBatchSizer::new(Arc::new(caps))
    .with_min_batch_size(1)
    .with_max_batch_size(256);

// Calculate optimal batch size
let model_size = 500 * 1024 * 1024;  // 500MB
let item_size = 256 * 1024;          // 256KB per item
let batch_size = sizer.calculate(item_size, model_size);

println!("Optimal batch size: {}", batch_size);
```

## Memory Profiling

### Track Memory Usage

Profile memory consumption:

```rust
use ipfrs_tensorlogic::MemoryProfiler;

let profiler = MemoryProfiler::new();

{
    let _guard = profiler.start_tracking("tensor_creation");
    let data: Vec<f32> = vec![0.0; 1000000];
    // ... use data
}

let stats = profiler.get_stats("tensor_creation").unwrap();
println!("Peak memory: {} bytes", stats.peak_bytes);
println!("Avg duration: {:?}", stats.avg_duration);
```

### Generate Reports

Create comprehensive memory reports:

```rust
let report = profiler.generate_report();
report.print();

// Output:
// === Memory Profiling Report ===
// Total operations: 5
// Total bytes: 12345678 (11.77 MB)
// Max peak: 5242880 (5.00 MB)
```

## Best Practices

### 1. Use Zero-Copy When Possible

```rust
// Good: Zero-copy access
let tensor = ArrowTensor::from_slice_f32("data", vec![1000], &data);
let slice = tensor.as_slice_f32().unwrap();

// Avoid: Copying data unnecessarily
let copied: Vec<f32> = tensor.as_slice_f32().unwrap().to_vec();
```

### 2. Cache Frequently Used Queries

```rust
// Use query cache for repeated queries
let cache = QueryCache::new(1000);
if let Some(cached) = cache.get(&key) {
    return cached;
}
let result = engine.query(&query, &kb)?;
cache.insert(key, result.clone());
```

### 3. Use Buffer Pools for Repeated Allocations

```rust
// Good: Reuse buffers
let pool = BufferPool::new(4096, 16);
for _ in 0..1000 {
    let mut buf = pool.acquire();
    // Use buffer
} // Automatically returned

// Avoid: Creating new buffers each time
for _ in 0..1000 {
    let mut buf = Vec::with_capacity(4096);
    // Use buffer
}
```

### 4. Profile Before Optimizing

```rust
// Always profile to find real bottlenecks
let profiler = MemoryProfiler::new();
let _guard = profiler.start_tracking("operation");

// Your code here

let report = profiler.generate_report();
report.print(); // Identify actual hotspots
```

### 5. Use Appropriate Data Types

```rust
// Choose the right dtype for your use case
writer.add_f32("weights", shape, &weights);      // Standard precision
writer.add_f64("high_precision", shape, &data);  // High precision
writer.add_i32("indices", shape, &indices);      // Integer data
```

## Troubleshooting

### Common Issues

#### Out of Memory

```rust
// Symptom: OOM errors during tensor operations
// Solution: Use streaming or chunked processing

let chunked = ChunkedModelStorage::new(1024 * 1024 * 100); // 100MB chunks
chunked.add_model("model", tensors).unwrap();
```

#### Slow Inference

```rust
// Symptom: Queries take too long
// Solution: Use query optimization and caching

let optimizer = QueryOptimizer::new();
let plan = optimizer.plan_query(&goals, &kb);
// Execute optimized plan

let cache = QueryCache::new(1000);
// Cache frequent queries
```

#### Type Mismatches

```rust
// Symptom: Type errors when accessing tensors
// Solution: Check dtype before access

if let Some(slice) = tensor.as_slice_f32() {
    // Use f32 slice
} else if let Some(slice) = tensor.as_slice_f64() {
    // Use f64 slice
}
```

### Performance Tips

1. **Use profiling tools**: Start with `MemoryProfiler` and `FfiProfiler`
2. **Enable LTO**: Add to `Cargo.toml` for release builds:
   ```toml
   [profile.release]
   lto = true
   ```
3. **Use appropriate capacities**: Size caches based on your workload
4. **Batch operations**: Group operations when possible
5. **Monitor memory**: Use shared memory for large tensors across processes

## Additional Resources

- [Examples Directory](./examples/) - Complete working examples
- [API Documentation](https://docs.rs/ipfrs-tensorlogic) - Full API reference
- [Benchmarks](./benches/) - Performance benchmarks
- [Integration Tests](./tests/) - Comprehensive test suite

## License

Licensed under Apache-2.0.
