//! Memory Profiling, CDC, Pool, Hash Engine benchmarks

use bytes::Bytes;
use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};
use ipfrs_core::{
    global_bytes_pool, global_cid_string_pool, global_hash_registry, Blake2b256Engine,
    Blake2b512Engine, Blake2s256Engine, Blake3Engine, Block, BytesPool, Chunker, ChunkingConfig,
    CidBuilder, CidStringPool, CpuFeatures, HashEngine, HashRegistry, Ipld, Sha256Engine,
    Sha3_256Engine,
};
use multihash_codetable::Code;
use std::collections::BTreeMap;
use std::hint::black_box;

// ============================================================================
// Memory Profiling Benchmarks
// ============================================================================

pub fn bench_zero_copy_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_copy");

    let data = Bytes::from(vec![0u8; 1_000_000]); // 1MB
    let block = Block::new(data.clone()).unwrap();

    // Benchmark: Clone Bytes (zero-copy, just RC increment)
    group.bench_function("bytes_clone", |b| {
        b.iter(|| {
            let _cloned = black_box(block.clone_data());
        });
    });

    // Benchmark: Slice operation (zero-copy)
    group.bench_function("slice_half", |b| {
        b.iter(|| {
            let _slice = black_box(block.slice(0..500_000));
        });
    });

    // Benchmark: as_bytes reference (zero allocation)
    group.bench_function("as_bytes_ref", |b| {
        b.iter(|| {
            let _bytes = black_box(block.as_bytes());
        });
    });

    // Benchmark: Full data() clone
    group.bench_function("data_clone", |b| {
        b.iter(|| {
            let _data = black_box(block.data().clone());
        });
    });

    // Compare: Copy vs reference
    let bytes_data = vec![0u8; 10_000]; // 10KB
    group.bench_function("vec_copy_10kb", |b| {
        b.iter(|| {
            let _copy = black_box(bytes_data.clone());
        });
    });

    let bytes_ref = Bytes::from(bytes_data);
    group.bench_function("bytes_clone_10kb", |b| {
        b.iter(|| {
            let _clone = black_box(bytes_ref.clone());
        });
    });

    group.finish();
}

pub fn bench_block_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_allocation");

    // Different sizes to measure allocation overhead
    let sizes = [64, 1024, 16384, 262144]; // 64B, 1KB, 16KB, 256KB

    for size in sizes {
        let data = vec![0u8; size];

        // Benchmark: Block creation with allocation
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("create_from_vec", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let bytes = Bytes::from(data.clone());
                    let _block = Block::new(black_box(bytes)).unwrap();
                });
            },
        );

        // Benchmark: Block creation from static data (no allocation)
        let static_bytes = Bytes::from(data.clone());
        group.bench_with_input(
            BenchmarkId::new("create_from_bytes", size),
            &static_bytes,
            |b, bytes| {
                b.iter(|| {
                    let _block = Block::new(black_box(bytes.clone())).unwrap();
                });
            },
        );
    }

    group.finish();
}

pub fn bench_memory_sharing(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_sharing");

    let data = Bytes::from(vec![0u8; 100_000]); // 100KB
    let block = Block::new(data).unwrap();

    // Benchmark: Check if blocks share data
    let block_clone = block.clone();
    group.bench_function("shares_data_check", |b| {
        b.iter(|| {
            let _shares = black_box(block.shares_data(&block_clone));
        });
    });

    // Benchmark: Clone block (should be cheap due to Bytes RC)
    group.bench_function("block_clone", |b| {
        b.iter(|| {
            let _cloned = black_box(block.clone());
        });
    });

    // Benchmark: into_parts (move ownership)
    group.bench_function("into_parts", |b| {
        b.iter_batched(
            || block.clone(),
            |b| {
                let (_cid, _data) = black_box(b.into_parts());
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

pub fn bench_chunking_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunking_memory");

    // Test different chunk sizes to measure memory efficiency
    let data = vec![0u8; 1_000_000]; // 1MB
    let chunk_sizes = [32 * 1024, 64 * 1024, 128 * 1024, 256 * 1024];

    for chunk_size in chunk_sizes {
        let config = ChunkingConfig::with_chunk_size(chunk_size).unwrap();
        let chunker = Chunker::with_config(config);

        group.throughput(Throughput::Bytes(data.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("chunk_data", chunk_size),
            &data,
            |b, data| {
                b.iter(|| {
                    let _chunked = chunker.chunk(black_box(data)).unwrap();
                });
            },
        );
    }

    group.finish();
}

pub fn bench_ipld_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("ipld_memory");

    // Create a complex IPLD structure
    let mut map = BTreeMap::new();
    for i in 0..100 {
        map.insert(format!("key_{}", i), Ipld::Integer(i));
    }
    let ipld = Ipld::Map(map);

    // Benchmark: IPLD cloning
    group.bench_function("ipld_clone", |b| {
        b.iter(|| {
            let _cloned = black_box(ipld.clone());
        });
    });

    // Benchmark: Encode to CBOR (measures allocation during encoding)
    group.bench_function("encode_dag_cbor", |b| {
        b.iter(|| {
            let _encoded = black_box(ipld.to_dag_cbor().unwrap());
        });
    });

    // Benchmark: Encode to JSON
    group.bench_function("encode_dag_json", |b| {
        b.iter(|| {
            let _encoded = black_box(ipld.to_dag_json().unwrap());
        });
    });

    group.finish();
}

// ============================================================================
// CDC (Content-Defined Chunking) Benchmarks
// ============================================================================

pub fn bench_cdc_chunking(c: &mut Criterion) {
    let mut group = c.benchmark_group("cdc_chunking");

    let data_sizes = [
        (10 * 1024, "10KB"),
        (100 * 1024, "100KB"),
        (1024 * 1024, "1MB"),
    ];

    for (size, label) in data_sizes {
        // Create test data with some patterns (more realistic than pure random)
        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            data.push(((i / 256) % 256) as u8);
        }

        // Benchmark: Fixed-size chunking
        let fixed_config = ChunkingConfig::with_chunk_size(32 * 1024).unwrap();
        let fixed_chunker = Chunker::with_config(fixed_config);

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("fixed_size", label), &data, |b, data| {
            b.iter(|| fixed_chunker.chunk(black_box(data)));
        });

        // Benchmark: Content-defined chunking
        let cdc_config = ChunkingConfig::content_defined_with_size(32 * 1024).unwrap();
        let cdc_chunker = Chunker::with_config(cdc_config);

        group.bench_with_input(
            BenchmarkId::new("content_defined", label),
            &data,
            |b, data| {
                b.iter(|| cdc_chunker.chunk(black_box(data)));
            },
        );
    }

    group.finish();
}

pub fn bench_cdc_deduplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("cdc_deduplication");

    // Create data with repeated patterns (high deduplication potential)
    let pattern: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
    let mut data = Vec::new();
    for _ in 0..100 {
        data.extend_from_slice(&pattern);
    }

    let cdc_config = ChunkingConfig::content_defined_with_size(4096).unwrap();
    let cdc_chunker = Chunker::with_config(cdc_config);

    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("chunk_with_dedup_tracking", |b| {
        b.iter(|| {
            let result = cdc_chunker.chunk(black_box(&data)).unwrap();
            // Access dedup stats to ensure they're computed
            black_box(result.dedup_stats);
        });
    });

    group.finish();
}

pub fn bench_rabin_fingerprinting(c: &mut Criterion) {
    let mut group = c.benchmark_group("rabin_fingerprinting");

    let sizes = [
        (10 * 1024, "10KB"),
        (100 * 1024, "100KB"),
        (1024 * 1024, "1MB"),
    ];

    for (size, label) in sizes {
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

        let config = ChunkingConfig::content_defined();
        let chunker = Chunker::with_config(config);

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("find_boundaries", label),
            &data,
            |b, data| {
                b.iter(|| chunker.chunk(black_box(data)));
            },
        );
    }

    group.finish();
}

// ============================================================================
// Memory Pooling Benchmarks
// ============================================================================

pub fn bench_bytes_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("bytes_pool");

    let pool = BytesPool::new();
    let sizes = [1024, 4096, 16384, 65536];

    for size in sizes {
        // Benchmark: Get from empty pool (cold miss)
        group.bench_with_input(BenchmarkId::new("get_cold", size), &size, |b, &size| {
            let pool = BytesPool::new(); // Fresh pool for each iteration
            b.iter(|| {
                let _buf = pool.get(black_box(size));
            });
        });

        // Benchmark: Get from warmed pool (hot hit)
        // Warm up the pool
        for _ in 0..10 {
            let buf = pool.get(size);
            pool.put(buf);
        }

        group.bench_with_input(BenchmarkId::new("get_hot", size), &size, |b, &size| {
            b.iter(|| {
                let buf = pool.get(black_box(size));
                pool.put(buf); // Return for next iteration
            });
        });

        // Benchmark: Get and put cycle
        group.bench_with_input(
            BenchmarkId::new("get_put_cycle", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let buf = pool.get(black_box(size));
                    pool.put(buf);
                });
            },
        );
    }

    // Benchmark: Global pool access
    group.bench_function("global_pool_get", |b| {
        b.iter(|| {
            let _buf = global_bytes_pool().get(black_box(4096));
        });
    });

    group.finish();
}

pub fn bench_cid_string_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("cid_string_pool");

    let pool = CidStringPool::new();

    // Generate some CID strings
    let cids: Vec<String> = (0..100)
        .map(|i| {
            let data = format!("test_data_{}", i);
            let cid = CidBuilder::new().build(data.as_bytes()).unwrap();
            cid.to_string()
        })
        .collect();

    // Benchmark: First intern (cold miss)
    group.bench_function("intern_cold", |b| {
        let mut i = 0;
        b.iter(|| {
            let pool = CidStringPool::new(); // Fresh pool
            let cid_str = &cids[i % cids.len()];
            let _arc = pool.intern(black_box(cid_str));
            i += 1;
        });
    });

    // Benchmark: Second intern (hot hit)
    // Warm up the pool
    for cid in &cids[0..50] {
        pool.intern(cid);
    }

    group.bench_function("intern_hot", |b| {
        let mut i = 0;
        b.iter(|| {
            let cid_str = &cids[i % 50]; // Use warmed entries
            let _arc = pool.intern(black_box(cid_str));
            i += 1;
        });
    });

    // Benchmark: Mixed access pattern
    group.bench_function("intern_mixed", |b| {
        let mut i = 0;
        b.iter(|| {
            let cid_str = &cids[i % cids.len()];
            let _arc = pool.intern(black_box(cid_str));
            i += 1;
        });
    });

    // Benchmark: Global pool access
    group.bench_function("global_pool_intern", |b| {
        let mut i = 0;
        b.iter(|| {
            let cid_str = &cids[i % cids.len()];
            let _arc = global_cid_string_pool().intern(black_box(cid_str));
            i += 1;
        });
    });

    group.finish();
}

pub fn bench_pool_vs_direct_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_vs_direct");

    let size = 4096;

    // Benchmark: Direct allocation (no pooling)
    group.bench_function("direct_alloc", |b| {
        b.iter(|| {
            let buf = bytes::BytesMut::with_capacity(black_box(size));
            black_box(buf);
        });
    });

    // Benchmark: Pooled allocation
    let pool = BytesPool::new();
    // Warm up
    for _ in 0..10 {
        let buf = pool.get(size);
        pool.put(buf);
    }

    group.bench_function("pooled_alloc", |b| {
        b.iter(|| {
            let buf = pool.get(black_box(size));
            pool.put(buf);
        });
    });

    // Benchmark: String interning vs cloning
    let test_string = "QmTest123456789abcdef";

    group.bench_function("string_clone", |b| {
        b.iter(|| {
            let _s = black_box(test_string).to_string();
        });
    });

    let pool = CidStringPool::new();
    pool.intern(test_string); // Pre-intern

    group.bench_function("string_intern", |b| {
        b.iter(|| {
            let _arc = pool.intern(black_box(test_string));
        });
    });

    group.finish();
}

// ============================================================================
// Hash Engine Benchmarks
// ============================================================================

pub fn bench_hash_engines(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_engines");

    let sizes = [64, 256, 1024, 4096, 16384, 65536, 262144, 1048576];

    for size in sizes {
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

        // Benchmark SHA256 engine
        let sha256 = Sha256Engine::new();
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("sha256_engine", size), &data, |b, data| {
            b.iter(|| sha256.digest(black_box(data)));
        });

        // Benchmark SHA3-256 engine
        let sha3 = Sha3_256Engine::new();
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("sha3_256_engine", size),
            &data,
            |b, data| {
                b.iter(|| sha3.digest(black_box(data)));
            },
        );

        // Benchmark BLAKE3 engine
        let blake3 = Blake3Engine::new();
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("blake3_engine", size), &data, |b, data| {
            b.iter(|| blake3.digest(black_box(data)));
        });

        // Benchmark BLAKE2b-256 engine
        let blake2b256 = Blake2b256Engine::new();
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("blake2b256_engine", size),
            &data,
            |b, data| {
                b.iter(|| blake2b256.digest(black_box(data)));
            },
        );

        // Benchmark BLAKE2b-512 engine
        let blake2b512 = Blake2b512Engine::new();
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("blake2b512_engine", size),
            &data,
            |b, data| {
                b.iter(|| blake2b512.digest(black_box(data)));
            },
        );

        // Benchmark BLAKE2s-256 engine
        let blake2s = Blake2s256Engine::new();
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("blake2s256_engine", size),
            &data,
            |b, data| {
                b.iter(|| blake2s.digest(black_box(data)));
            },
        );
    }

    group.finish();
}

pub fn bench_hash_registry(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_registry");

    let registry = HashRegistry::new();
    let data = vec![42u8; 4096];

    // Benchmark: SHA256 via registry
    group.bench_function("registry_sha256", |b| {
        b.iter(|| {
            let _hash = registry.digest(Code::Sha2_256, black_box(&data)).unwrap();
        });
    });

    // Benchmark: SHA3-256 via registry
    group.bench_function("registry_sha3_256", |b| {
        b.iter(|| {
            let _hash = registry.digest(Code::Sha3_256, black_box(&data)).unwrap();
        });
    });

    // Benchmark: BLAKE2b-256 via registry
    group.bench_function("registry_blake2b256", |b| {
        b.iter(|| {
            let _hash = registry.digest(Code::Blake2b256, black_box(&data)).unwrap();
        });
    });

    // Benchmark: BLAKE2b-512 via registry
    group.bench_function("registry_blake2b512", |b| {
        b.iter(|| {
            let _hash = registry.digest(Code::Blake2b512, black_box(&data)).unwrap();
        });
    });

    // Benchmark: BLAKE2s-256 via registry
    group.bench_function("registry_blake2s256", |b| {
        b.iter(|| {
            let _hash = registry.digest(Code::Blake2s256, black_box(&data)).unwrap();
        });
    });

    // Benchmark: Global registry access
    group.bench_function("global_registry_sha256", |b| {
        b.iter(|| {
            let _hash = global_hash_registry()
                .digest(Code::Sha2_256, black_box(&data))
                .unwrap();
        });
    });

    group.finish();
}

pub fn bench_cpu_feature_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_features");

    // Benchmark: Runtime feature detection
    group.bench_function("detect_features", |b| {
        b.iter(|| {
            let _features = CpuFeatures::detect();
        });
    });

    group.finish();
}

pub fn bench_simd_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_comparison");

    let data = vec![42u8; 1024 * 1024]; // 1MB

    // Benchmark: SHA256 with SIMD (if available)
    let engine = Sha256Engine::new();
    let simd_enabled = engine.is_simd_enabled();

    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("sha256", if simd_enabled { "simd" } else { "scalar" }),
        &data,
        |b, data| {
            b.iter(|| engine.digest(black_box(data)));
        },
    );

    group.finish();
}

criterion_group!(
    memory_benches,
    bench_zero_copy_operations,
    bench_block_allocation_patterns,
    bench_memory_sharing,
    bench_chunking_memory_usage,
    bench_ipld_memory_efficiency,
);

criterion_group!(
    cdc_benches,
    bench_cdc_chunking,
    bench_cdc_deduplication,
    bench_rabin_fingerprinting,
);

criterion_group!(
    pool_benches,
    bench_bytes_pool,
    bench_cid_string_pool,
    bench_pool_vs_direct_allocation,
);

criterion_group!(
    hash_benches,
    bench_hash_engines,
    bench_hash_registry,
    bench_cpu_feature_detection,
    bench_simd_comparison,
);
