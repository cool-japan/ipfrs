//! Benchmarks for tensor operations
//!
//! Measures performance of:
//! - Arrow tensor creation and access
//! - Safetensors serialization
//! - Shared memory operations

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ipfrs_tensorlogic::{
    AdaptiveBuffer, ArrowTensor, ArrowTensorStore, BufferPool, ComputationGraph, Constant,
    DistributedExecutor, FfiProfiler, GradientCompressor, GraphNode, GraphOptimizer,
    InferenceEngine, KnowledgeBase, Predicate, QueryCache, QueryKey, QueryOptimizer, Rule,
    SafetensorsWriter, SparseGradient, StackBuffer, TensorOp, Term, TypedBufferPool,
    ZeroCopyConverter,
};
use rand::RngExt;
use std::hint::black_box;

/// Generate random f32 data
fn random_f32_data(size: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    (0..size).map(|_| rng.random::<f32>()).collect()
}

/// Benchmark Arrow tensor creation
fn bench_arrow_tensor_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("arrow_tensor_creation");

    for size in [1024, 4096, 16384, 65536, 262144].iter() {
        let data = random_f32_data(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                ArrowTensor::from_slice_f32(
                    black_box("test_tensor"),
                    black_box(vec![size]),
                    black_box(&data),
                )
            })
        });
    }

    group.finish();
}

/// Benchmark Arrow tensor access (zero-copy)
fn bench_arrow_tensor_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("arrow_tensor_access");

    for size in [1024, 4096, 16384, 65536].iter() {
        let data = random_f32_data(*size);
        let tensor = ArrowTensor::from_slice_f32("test", vec![*size], &data);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let slice = tensor.as_slice_f32().unwrap();
                black_box(slice.iter().sum::<f32>())
            })
        });
    }

    group.finish();
}

/// Benchmark Arrow IPC serialization
fn bench_arrow_ipc_serialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("arrow_ipc_serialize");

    for size in [1024, 4096, 16384].iter() {
        let data = random_f32_data(*size);
        let tensor = ArrowTensor::from_slice_f32("test", vec![*size], &data);

        let mut store = ArrowTensorStore::new();
        store.insert(tensor);

        group.throughput(Throughput::Bytes((*size * 4) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            let mut store_clone = ArrowTensorStore::new();
            let data = random_f32_data(*size);
            store_clone.insert(ArrowTensor::from_slice_f32("test", vec![*size], &data));

            b.iter(|| black_box(store_clone.to_bytes().unwrap()))
        });
    }

    group.finish();
}

/// Benchmark Safetensors serialization
fn bench_safetensors_serialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("safetensors_serialize");

    for size in [1024, 4096, 16384].iter() {
        let data = random_f32_data(*size);

        group.throughput(Throughput::Bytes((*size * 4) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let mut writer = SafetensorsWriter::new();
                writer.add_f32(black_box("test"), black_box(vec![size]), black_box(&data));
                black_box(writer.serialize().unwrap())
            })
        });
    }

    group.finish();
}

/// Benchmark raw byte access vs typed access
fn bench_access_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("access_patterns");

    let size = 16384;
    let data = random_f32_data(size);
    let tensor = ArrowTensor::from_slice_f32("test", vec![size], &data);

    // Zero-copy typed access
    group.bench_function("zero_copy_typed", |b| {
        b.iter(|| {
            let slice = tensor.as_slice_f32().unwrap();
            black_box(slice.iter().sum::<f32>())
        })
    });

    // Raw bytes access
    group.bench_function("raw_bytes", |b| {
        b.iter(|| {
            let bytes = tensor.as_bytes();
            black_box(bytes.len())
        })
    });

    group.finish();
}

/// Benchmark cache operations
fn bench_cache_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_operations");

    let cache = QueryCache::new(1000);

    // Populate cache
    for i in 0..500 {
        let key = QueryKey {
            predicate_name: format!("pred_{}", i % 10),
            ground_args: vec![],
        };
        cache.insert(key, vec![]);
    }

    // Benchmark cache hit
    group.bench_function("cache_hit", |b| {
        let key = QueryKey {
            predicate_name: "pred_5".to_string(),
            ground_args: vec![],
        };
        b.iter(|| black_box(cache.get(black_box(&key))))
    });

    // Benchmark cache miss
    group.bench_function("cache_miss", |b| {
        let key = QueryKey {
            predicate_name: "nonexistent".to_string(),
            ground_args: vec![],
        };
        b.iter(|| black_box(cache.get(black_box(&key))))
    });

    group.finish();
}

/// Benchmark gradient compression
fn bench_gradient_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_compression");

    for size in [1024, 4096, 16384].iter() {
        let data = random_f32_data(*size);

        // Top-k compression
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("top_k_10pct", size), size, |b, &size| {
            let k = (size as f32 * 0.1) as usize;
            b.iter(|| {
                black_box(GradientCompressor::top_k(
                    black_box(&data),
                    black_box(vec![size]),
                    black_box(k),
                ))
            })
        });

        // Threshold compression
        group.bench_with_input(BenchmarkId::new("threshold", size), size, |b, &size| {
            b.iter(|| {
                black_box(GradientCompressor::threshold(
                    black_box(&data),
                    black_box(vec![size]),
                    black_box(0.1),
                ))
            })
        });

        // Quantization
        group.bench_with_input(BenchmarkId::new("quantize", size), size, |b, &size| {
            b.iter(|| {
                black_box(GradientCompressor::quantize(
                    black_box(&data),
                    black_box(vec![size]),
                ))
            })
        });
    }

    group.finish();
}

/// Benchmark sparse gradient operations
fn bench_sparse_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_gradient");

    let size = 10000;
    let sparsity = 0.9; // 90% sparse
    let nnz = ((1.0 - sparsity) * size as f32) as usize;

    let mut rng = rand::rng();
    let indices: Vec<usize> = (0..nnz).map(|_| rng.random_range(0..size)).collect();
    let values = random_f32_data(nnz);

    let sparse = SparseGradient::new(indices, values, vec![size]);

    // Benchmark to_dense conversion
    group.throughput(Throughput::Elements(size as u64));
    group.bench_function("to_dense", |b| b.iter(|| black_box(sparse.to_dense())));

    // Benchmark sparsity_ratio calculation
    group.bench_function("sparsity_ratio", |b| {
        b.iter(|| black_box(sparse.sparsity_ratio()))
    });

    group.finish();
}

/// Benchmark FFI call overhead simulation
fn bench_ffi_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("ffi_overhead");

    let profiler = FfiProfiler::new();

    // Benchmark minimal FFI call
    group.bench_function("minimal_call", |b| {
        b.iter(|| {
            let _guard = profiler.start("minimal");
            black_box(())
        })
    });

    // Benchmark FFI call with small data transfer
    group.bench_function("small_data_transfer", |b| {
        let data = vec![1u8; 64];
        b.iter(|| {
            let _guard = profiler.start("small_transfer");
            black_box(&data);
        })
    });

    // Benchmark FFI call with medium data transfer
    group.bench_function("medium_data_transfer", |b| {
        let data = vec![1u8; 4096];
        b.iter(|| {
            let _guard = profiler.start("medium_transfer");
            black_box(&data);
        })
    });

    // Benchmark profiler overhead itself
    group.bench_function("profiler_overhead", |b| {
        b.iter(|| {
            let _guard = profiler.start("test");
        })
    });

    group.finish();
}

/// Benchmark zero-copy conversions
fn bench_zero_copy_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_copy_conversion");

    for size in [1024, 4096, 16384, 65536].iter() {
        let floats = random_f32_data(*size);

        group.throughput(Throughput::Bytes((*size * 4) as u64));

        // Benchmark zero-copy float to bytes
        group.bench_with_input(BenchmarkId::new("float_to_bytes", size), size, |b, _| {
            b.iter(|| black_box(ZeroCopyConverter::slice_to_bytes(black_box(&floats))))
        });

        // Benchmark copying conversion for comparison
        group.bench_with_input(
            BenchmarkId::new("float_to_bytes_copy", size),
            size,
            |b, _| {
                b.iter(|| {
                    let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
                    black_box(bytes)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark buffer pool operations
fn bench_buffer_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_pool");

    let pool = BufferPool::new(4096, 16);

    // Benchmark buffer acquisition
    group.bench_function("acquire", |b| {
        b.iter(|| {
            let buffer = pool.acquire();
            black_box(buffer)
        })
    });

    // Benchmark buffer acquisition and use
    group.bench_function("acquire_and_use", |b| {
        b.iter(|| {
            let mut buffer = pool.acquire();
            buffer.as_mut().extend_from_slice(&[1, 2, 3, 4]);
            black_box(buffer)
        })
    });

    // Compare with direct allocation
    group.bench_function("direct_allocation", |b| {
        b.iter(|| {
            let mut buffer = Vec::with_capacity(4096);
            buffer.extend_from_slice(&[1, 2, 3, 4]);
            black_box(buffer)
        })
    });

    group.finish();
}

/// Benchmark typed buffer pool
fn bench_typed_buffer_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("typed_buffer_pool");

    let pool = TypedBufferPool::<f32>::new(1024, 16);

    // Benchmark typed buffer acquisition and use
    group.bench_function("acquire_and_fill", |b| {
        b.iter(|| {
            let mut buffer = pool.acquire();
            buffer.extend((0..100).map(|i| i as f32));
            black_box(buffer)
        })
    });

    // Compare with direct allocation
    group.bench_function("direct_allocation", |b| {
        b.iter(|| {
            let mut buffer = Vec::with_capacity(1024);
            buffer.extend((0..100).map(|i| i as f32));
            black_box(buffer)
        })
    });

    group.finish();
}

/// Benchmark stack vs heap allocation
fn bench_stack_vs_heap(c: &mut Criterion) {
    let mut group = c.benchmark_group("stack_vs_heap");

    // Small data - stack should win
    group.bench_function("stack_small", |b| {
        b.iter(|| {
            let mut buf = StackBuffer::<64>::new();
            buf.write(&[1, 2, 3, 4]).unwrap();
            black_box(buf)
        })
    });

    group.bench_function("heap_small", |b| {
        b.iter(|| {
            let mut buf = Vec::with_capacity(64);
            buf.extend_from_slice(&[1, 2, 3, 4]);
            black_box(buf)
        })
    });

    // Adaptive buffer
    group.bench_function("adaptive_small", |b| {
        b.iter(|| {
            let mut buf = AdaptiveBuffer::new(4);
            buf.write(&[1, 2, 3, 4]).unwrap();
            black_box(buf)
        })
    });

    group.finish();
}

/// Benchmark conversion patterns
fn bench_conversion_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("conversion_patterns");

    let size = 10000;
    let floats = random_f32_data(size);

    // Pattern 1: Zero-copy view
    group.bench_function("zero_copy_view", |b| {
        b.iter(|| {
            let bytes = ZeroCopyConverter::slice_to_bytes(&floats);
            black_box(bytes.len())
        })
    });

    // Pattern 2: Copy to new buffer
    group.bench_function("copy_to_buffer", |b| {
        b.iter(|| {
            let mut buffer = Vec::with_capacity(size * 4);
            for &f in &floats {
                buffer.extend_from_slice(&f.to_le_bytes());
            }
            black_box(buffer)
        })
    });

    // Pattern 3: Using buffer pool
    let pool = BufferPool::new(size * 4, 4);
    group.bench_function("pooled_buffer", |b| {
        b.iter(|| {
            let mut buffer = pool.acquire();
            for &f in &floats {
                buffer.as_mut().extend_from_slice(&f.to_le_bytes());
            }
            black_box(buffer)
        })
    });

    // Pattern 4: Adaptive buffer
    group.bench_function("adaptive_buffer", |b| {
        b.iter(|| {
            let mut buffer = AdaptiveBuffer::new(size * 4);
            for &f in &floats {
                buffer.write(&f.to_le_bytes()).unwrap();
            }
            black_box(buffer)
        })
    });

    group.finish();
}

/// Benchmark memory allocation patterns
fn bench_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_patterns");

    // Pattern: Many small allocations
    group.bench_function("many_small_allocs", |b| {
        b.iter(|| {
            let mut vecs = Vec::new();
            for _ in 0..100 {
                vecs.push(vec![1u8; 64]);
            }
            black_box(vecs)
        })
    });

    // Pattern: Pooled small allocations
    group.bench_function("pooled_small_allocs", |b| {
        let pool = BufferPool::new(64, 100);
        b.iter(|| {
            let mut buffers = Vec::new();
            for _ in 0..100 {
                buffers.push(pool.acquire());
            }
            black_box(buffers)
        })
    });

    // Pattern: Single large allocation
    group.bench_function("single_large_alloc", |b| {
        b.iter(|| {
            let vec = vec![1u8; 6400]; // 100 * 64
            black_box(vec)
        })
    });

    group.finish();
}

/// Benchmark graph partitioning
fn bench_graph_partitioning(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_partitioning");

    // Create different sized graphs
    for graph_size in [10, 50, 100, 200].iter() {
        let graph = create_test_graph(*graph_size);
        let workers: Vec<String> = (0..4).map(|i| format!("worker{}", i)).collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(graph_size),
            graph_size,
            |b, &_size| {
                b.iter(|| {
                    let mut executor = DistributedExecutor::new();
                    executor
                        .partition_graph(black_box(&graph), black_box(&workers))
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark graph optimization
fn bench_graph_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_optimization");

    for graph_size in [20, 50, 100].iter() {
        let graph = create_test_graph(*graph_size);

        group.bench_with_input(
            BenchmarkId::from_parameter(graph_size),
            graph_size,
            |b, &_size| {
                b.iter(|| {
                    let mut g = graph.clone();
                    GraphOptimizer::optimize_all(black_box(&mut g)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark topological sort
fn bench_topological_sort(c: &mut Criterion) {
    let mut group = c.benchmark_group("topological_sort");

    for graph_size in [10, 50, 100, 200, 500].iter() {
        let graph = create_test_graph(*graph_size);

        group.throughput(Throughput::Elements(*graph_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(graph_size),
            graph_size,
            |b, &_size| b.iter(|| black_box(&graph).topological_sort().unwrap()),
        );
    }

    group.finish();
}

/// Benchmark communication cost estimation
fn bench_communication_cost(c: &mut Criterion) {
    let mut group = c.benchmark_group("communication_cost");

    for num_workers in [2, 4, 8, 16].iter() {
        let graph = create_test_graph(100);
        let workers: Vec<String> = (0..*num_workers).map(|i| format!("worker{}", i)).collect();

        let mut executor = DistributedExecutor::new();
        executor.partition_graph(&graph, &workers).unwrap();

        group.bench_with_input(
            BenchmarkId::from_parameter(num_workers),
            num_workers,
            |b, &_num| {
                b.iter(|| {
                    let total: usize = workers
                        .iter()
                        .map(|w| black_box(&executor).estimate_communication_cost(w))
                        .sum();
                    black_box(total)
                })
            },
        );
    }

    group.finish();
}

/// Helper function to create a test computation graph
fn create_test_graph(size: usize) -> ComputationGraph {
    let mut graph = ComputationGraph::new();

    // Create input
    let input = GraphNode::new(
        "input".to_string(),
        TensorOp::Input {
            name: "x".to_string(),
        },
    );
    graph.add_node(input).unwrap();
    graph.mark_input("input".to_string());

    let mut prev_id = "input".to_string();

    // Create a chain of operations
    for i in 0..size - 1 {
        let op = match i % 4 {
            0 => TensorOp::ReLU,
            1 => TensorOp::Tanh,
            2 => TensorOp::Sigmoid,
            _ => TensorOp::ReLU,
        };

        let node_id = format!("node_{}", i);
        let node = GraphNode::new(node_id.clone(), op).add_input(prev_id.clone());

        graph.add_node(node).unwrap();
        prev_id = node_id;
    }

    graph.mark_output(prev_id);
    graph
}

/// Benchmark simple fact query (baseline)
fn bench_simple_fact_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference_simple_fact");

    for num_facts in [10, 100, 1000, 10000].iter() {
        let mut kb = KnowledgeBase::new();

        // Add facts: parent(person_i, person_j)
        for i in 0..*num_facts {
            kb.add_fact(Predicate::new(
                "parent".to_string(),
                vec![
                    Term::Const(Constant::String(format!("person_{}", i))),
                    Term::Const(Constant::String(format!("person_{}", i + 1))),
                ],
            ));
        }

        let query = Predicate::new(
            "parent".to_string(),
            vec![
                Term::Const(Constant::String("person_5".to_string())),
                Term::Var("X".to_string()),
            ],
        );

        group.throughput(Throughput::Elements(*num_facts as u64));
        group.bench_with_input(BenchmarkId::from_parameter(num_facts), num_facts, |b, _| {
            b.iter(|| {
                let engine = InferenceEngine::new();
                black_box(engine.query(black_box(&query), black_box(&kb)).unwrap())
            })
        });
    }

    group.finish();
}

/// Benchmark rule-based inference
fn bench_rule_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference_with_rules");

    for num_facts in [10, 50, 100, 500].iter() {
        let mut kb = KnowledgeBase::new();

        // Add facts: parent(person_i, person_j)
        for i in 0..*num_facts {
            kb.add_fact(Predicate::new(
                "parent".to_string(),
                vec![
                    Term::Const(Constant::String(format!("person_{}", i))),
                    Term::Const(Constant::String(format!("person_{}", i + 1))),
                ],
            ));
        }

        // Add rule: ancestor(X, Y) :- parent(X, Y)
        kb.add_rule(Rule::new(
            Predicate::new(
                "ancestor".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            ),
            vec![Predicate::new(
                "parent".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            )],
        ));

        // Add rule: ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)
        kb.add_rule(Rule::new(
            Predicate::new(
                "ancestor".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Z".to_string())],
            ),
            vec![
                Predicate::new(
                    "parent".to_string(),
                    vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
                ),
                Predicate::new(
                    "ancestor".to_string(),
                    vec![Term::Var("Y".to_string()), Term::Var("Z".to_string())],
                ),
            ],
        ));

        let query = Predicate::new(
            "ancestor".to_string(),
            vec![
                Term::Const(Constant::String("person_0".to_string())),
                Term::Var("X".to_string()),
            ],
        );

        group.throughput(Throughput::Elements(*num_facts as u64));
        group.bench_with_input(BenchmarkId::from_parameter(num_facts), num_facts, |b, _| {
            b.iter(|| {
                let engine = InferenceEngine::new();
                black_box(engine.query(black_box(&query), black_box(&kb)).unwrap())
            })
        });
    }

    group.finish();
}

/// Benchmark query optimization
fn bench_query_optimization_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_optimization");

    for num_facts in [100, 500, 1000].iter() {
        let mut kb = KnowledgeBase::new();

        // Add facts with multiple predicates
        for i in 0..*num_facts {
            kb.add_fact(Predicate::new(
                "edge".to_string(),
                vec![
                    Term::Const(Constant::String(format!("node_{}", i))),
                    Term::Const(Constant::String(format!("node_{}", (i + 1) % num_facts))),
                ],
            ));
        }

        let goals = vec![
            Predicate::new(
                "edge".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            ),
            Predicate::new(
                "edge".to_string(),
                vec![
                    Term::Var("Y".to_string()),
                    Term::Const(Constant::String("node_50".to_string())),
                ],
            ),
        ];

        group.throughput(Throughput::Elements(*num_facts as u64));
        group.bench_with_input(BenchmarkId::from_parameter(num_facts), num_facts, |b, _| {
            b.iter(|| {
                let optimizer = QueryOptimizer::new();
                black_box(optimizer.plan_query(black_box(&goals), black_box(&kb)))
            })
        });
    }

    group.finish();
}

/// Benchmark end-to-end inference with caching
fn bench_inference_with_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference_with_cache");

    let mut kb = KnowledgeBase::new();

    // Build a knowledge base
    for i in 0..100 {
        kb.add_fact(Predicate::new(
            "data".to_string(),
            vec![
                Term::Const(Constant::String(format!("key_{}", i))),
                Term::Const(Constant::String(format!("value_{}", i))),
            ],
        ));
    }

    let cache = QueryCache::new(100);
    let engine = InferenceEngine::new();

    group.bench_function("cold_cache", |b| {
        b.iter(|| {
            let query = Predicate::new(
                "data".to_string(),
                vec![
                    Term::Const(Constant::String("key_42".to_string())),
                    Term::Var("V".to_string()),
                ],
            );

            // Clear cache for cold start
            let fresh_cache = QueryCache::new(100);
            let key = QueryKey {
                predicate_name: "data".to_string(),
                ground_args: vec![],
            };

            if let Some(cached) = fresh_cache.get(&key) {
                black_box(cached)
            } else {
                let result = engine.query(&query, &kb).unwrap();
                fresh_cache.insert(key, result.clone());
                black_box(result)
            }
        })
    });

    // Warm up cache
    let query = Predicate::new(
        "data".to_string(),
        vec![
            Term::Const(Constant::String("key_42".to_string())),
            Term::Var("V".to_string()),
        ],
    );
    let result = engine.query(&query, &kb).unwrap();
    let key = QueryKey {
        predicate_name: "data".to_string(),
        ground_args: vec![],
    };
    cache.insert(key.clone(), result);

    group.bench_function("warm_cache", |b| {
        b.iter(|| {
            if let Some(cached) = cache.get(&key) {
                black_box(cached)
            } else {
                let query = Predicate::new(
                    "data".to_string(),
                    vec![
                        Term::Const(Constant::String("key_42".to_string())),
                        Term::Var("V".to_string()),
                    ],
                );
                black_box(engine.query(&query, &kb).unwrap())
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_arrow_tensor_creation,
    bench_arrow_tensor_access,
    bench_arrow_ipc_serialize,
    bench_safetensors_serialize,
    bench_access_patterns,
    bench_cache_operations,
    bench_gradient_compression,
    bench_sparse_gradient,
    bench_ffi_overhead,
    bench_zero_copy_conversion,
    bench_buffer_pool,
    bench_typed_buffer_pool,
    bench_stack_vs_heap,
    bench_conversion_patterns,
    bench_allocation_patterns,
    bench_graph_partitioning,
    bench_graph_optimization,
    bench_topological_sort,
    bench_communication_cost,
    bench_simple_fact_query,
    bench_rule_inference,
    bench_query_optimization_overhead,
    bench_inference_with_cache,
);

criterion_main!(benches);
