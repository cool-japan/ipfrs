//! Benchmarks for ipfrs-core
//!
//! Run with: cargo bench -p ipfrs-core
//! Results are saved to target/criterion/

use bytes::Bytes;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ipfrs_core::{
    codec, compress, compression_ratio, count_links_by_depth, dag_fanout_by_level, decompress,
    filter_dag, global_bytes_pool, global_cid_string_pool, global_hash_registry, map_dag,
    read_chunked_file, subgraph_size, topological_sort, BatchProcessor, Blake2b256Engine,
    Blake2b512Engine, Blake2s256Engine, Blake3Engine, Block, BlockReader, BytesPool, CarReader,
    CarWriter, Chunker, ChunkingConfig, Cid, CidBuilder, CidExt, CidStringPool, CodecRegistry,
    CompressionAlgorithm, CpuFeatures, DagMetrics, DagNode, HashAlgorithm, HashEngine,
    HashRegistry, Ipld, MemoryBlockFetcher, MultibaseEncoding, Sha256Engine, Sha3_256Engine,
};
use multihash_codetable::Code;
use std::collections::BTreeMap;
use std::io::Read;

// ============================================================================
// CID Benchmarks
// ============================================================================

fn bench_cid_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cid_generation");

    // Different data sizes
    let sizes = [64, 256, 1024, 4096, 16384, 65536, 262144, 1048576];

    for size in sizes {
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("sha256", size), &data, |b, data| {
            b.iter(|| CidBuilder::new().build(black_box(data)));
        });
    }

    group.finish();
}

fn bench_cid_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("cid_parsing");

    // Generate a CID for parsing
    let cid = CidBuilder::new().build(b"benchmark data").unwrap();
    let cid_base32 = cid.to_string_with_base(MultibaseEncoding::Base32Lower);
    let cid_base58 = cid.to_string_with_base(MultibaseEncoding::Base58Btc);
    let cid_base64 = cid.to_string_with_base(MultibaseEncoding::Base64);

    group.bench_function("parse_base32", |b| {
        b.iter(|| {
            let _: Cid = black_box(&cid_base32).parse().unwrap();
        });
    });

    group.bench_function("parse_base58btc", |b| {
        b.iter(|| {
            let _: Cid = black_box(&cid_base58).parse().unwrap();
        });
    });

    group.bench_function("parse_base64", |b| {
        b.iter(|| {
            let _: Cid = black_box(&cid_base64).parse().unwrap();
        });
    });

    group.finish();
}

fn bench_cid_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("cid_encoding");

    let cid = CidBuilder::new().build(b"benchmark data").unwrap();

    group.bench_function("to_base32_lower", |b| {
        b.iter(|| black_box(&cid).to_string_with_base(MultibaseEncoding::Base32Lower));
    });

    group.bench_function("to_base58btc", |b| {
        b.iter(|| black_box(&cid).to_string_with_base(MultibaseEncoding::Base58Btc));
    });

    group.bench_function("to_base64", |b| {
        b.iter(|| black_box(&cid).to_string_with_base(MultibaseEncoding::Base64));
    });

    group.bench_function("to_base64_url", |b| {
        b.iter(|| black_box(&cid).to_string_with_base(MultibaseEncoding::Base64Url));
    });

    group.finish();
}

// ============================================================================
// Block Benchmarks
// ============================================================================

fn bench_block_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_creation");

    let sizes = [64, 256, 1024, 4096, 16384, 65536, 262144, 1048576];

    for size in sizes {
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("new", size), &data, |b, data| {
            b.iter(|| Block::new(Bytes::copy_from_slice(black_box(data))));
        });
    }

    group.finish();
}

fn bench_block_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_verification");

    let sizes = [1024, 16384, 262144, 1048576];

    for size in sizes {
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let block = Block::new(Bytes::from(data)).unwrap();

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("verify", size), &block, |b, block| {
            b.iter(|| black_box(block).verify());
        });
    }

    group.finish();
}

// ============================================================================
// IPLD Benchmarks
// ============================================================================

fn bench_ipld_dag_cbor(c: &mut Criterion) {
    let mut group = c.benchmark_group("ipld_dag_cbor");

    // Simple integer
    let int_value = Ipld::Integer(42);
    group.bench_function("encode_integer", |b| {
        b.iter(|| black_box(&int_value).to_dag_cbor());
    });

    // String
    let string_value = Ipld::String("Hello, IPFS World!".to_string());
    group.bench_function("encode_string", |b| {
        b.iter(|| black_box(&string_value).to_dag_cbor());
    });

    // Bytes (1KB)
    let bytes_value = Ipld::Bytes(vec![0u8; 1024]);
    group.bench_function("encode_bytes_1kb", |b| {
        b.iter(|| black_box(&bytes_value).to_dag_cbor());
    });

    // List of 100 integers
    let list_value = Ipld::List((0..100).map(Ipld::Integer).collect());
    group.bench_function("encode_list_100", |b| {
        b.iter(|| black_box(&list_value).to_dag_cbor());
    });

    // Map with 20 entries
    let map_value = Ipld::Map(
        (0..20)
            .map(|i| (format!("key_{}", i), Ipld::Integer(i)))
            .collect::<BTreeMap<_, _>>(),
    );
    group.bench_function("encode_map_20", |b| {
        b.iter(|| black_box(&map_value).to_dag_cbor());
    });

    // Decode benchmarks
    let encoded_map = map_value.to_dag_cbor().unwrap();
    group.bench_function("decode_map_20", |b| {
        b.iter(|| Ipld::from_dag_cbor(black_box(&encoded_map)));
    });

    group.finish();
}

fn bench_ipld_dag_json(c: &mut Criterion) {
    let mut group = c.benchmark_group("ipld_dag_json");

    // Map with nested structure
    let nested_value = Ipld::Map({
        let mut map = BTreeMap::new();
        map.insert("name".to_string(), Ipld::String("test".to_string()));
        map.insert("count".to_string(), Ipld::Integer(42));
        map.insert("data".to_string(), Ipld::Bytes(vec![1, 2, 3, 4, 5]));
        map.insert(
            "items".to_string(),
            Ipld::List(vec![Ipld::Integer(1), Ipld::Integer(2), Ipld::Integer(3)]),
        );
        map
    });

    group.bench_function("encode_nested", |b| {
        b.iter(|| black_box(&nested_value).to_dag_json());
    });

    let encoded_json = nested_value.to_dag_json().unwrap();
    group.bench_function("decode_nested", |b| {
        b.iter(|| Ipld::from_dag_json(black_box(&encoded_json)));
    });

    group.finish();
}

// ============================================================================
// Chunking Benchmarks
// ============================================================================

fn bench_chunking(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunking");

    let chunk_sizes = [
        (1024, "1KB"),
        (4096, "4KB"),
        (65536, "64KB"),
        (262144, "256KB"),
    ];
    let data_sizes = [
        (1024 * 10, "10KB"),
        (1024 * 100, "100KB"),
        (1024 * 1024, "1MB"),
    ];

    for (chunk_size, chunk_label) in &chunk_sizes {
        let config = ChunkingConfig::with_chunk_size(*chunk_size).unwrap();
        let chunker = Chunker::with_config(config);

        for (data_size, data_label) in &data_sizes {
            if *data_size <= *chunk_size {
                continue; // Skip when data fits in single chunk
            }

            let data: Vec<u8> = (0..*data_size).map(|i| (i % 256) as u8).collect();

            group.throughput(Throughput::Bytes(*data_size as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("chunk_{}", chunk_label), data_label),
                &data,
                |b, data| {
                    b.iter(|| chunker.chunk(black_box(data)));
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// DAG Node Benchmarks
// ============================================================================

fn bench_dag_node(c: &mut Criterion) {
    let mut group = c.benchmark_group("dag_node");

    // Leaf node creation with various data sizes
    let data_sizes = [64, 256, 1024, 4096];
    for size in data_sizes {
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

        group.bench_with_input(BenchmarkId::new("leaf_create", size), &data, |b, data| {
            b.iter(|| DagNode::leaf(black_box(data.clone())));
        });
    }

    // Leaf node serialization
    let leaf_data = vec![0u8; 1024];
    let leaf_node = DagNode::leaf(leaf_data);

    group.bench_function("leaf_to_ipld", |b| {
        b.iter(|| black_box(&leaf_node).to_ipld());
    });

    group.bench_function("leaf_to_dag_cbor", |b| {
        b.iter(|| black_box(&leaf_node).to_dag_cbor());
    });

    group.finish();
}

// ============================================================================
// Streaming Benchmarks
// ============================================================================

fn bench_block_reader(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_reader");

    let sizes = [1024, 16384, 262144, 1048576];

    for size in sizes {
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let block = Block::new(Bytes::from(data)).unwrap();

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("read_all", size), &block, |b, block| {
            b.iter(|| {
                let mut reader = BlockReader::new(black_box(block));
                let mut buf = Vec::with_capacity(size);
                reader.read_to_end(&mut buf).unwrap();
                buf
            });
        });
    }

    group.finish();
}

fn bench_chunked_file_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunked_file_read");
    group.sample_size(20); // Reduce sample size for slower async benchmarks

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let sizes = [(5000, "5KB"), (50000, "50KB")];

    for (size, label) in sizes {
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let config = ChunkingConfig::with_chunk_size(1024).unwrap();
        let chunker = Chunker::with_config(config);
        let chunked = chunker.chunk(&data).unwrap();

        let mut fetcher = MemoryBlockFetcher::new();
        for block in &chunked.blocks {
            fetcher.add_block(block.clone());
        }

        let root_cid = chunked.root_cid;

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_function(BenchmarkId::new("read", label), |b| {
            b.to_async(&rt).iter(|| async {
                read_chunked_file(black_box(&fetcher), black_box(&root_cid)).await
            });
        });
    }

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    cid_benches,
    bench_cid_generation,
    bench_cid_parsing,
    bench_cid_encoding,
);

criterion_group!(
    block_benches,
    bench_block_creation,
    bench_block_verification,
);

criterion_group!(ipld_benches, bench_ipld_dag_cbor, bench_ipld_dag_json,);

criterion_group!(chunking_benches, bench_chunking, bench_dag_node,);

criterion_group!(
    streaming_benches,
    bench_block_reader,
    bench_chunked_file_read,
);

// ============================================================================
// Memory Profiling Benchmarks
// ============================================================================

fn bench_zero_copy_operations(c: &mut Criterion) {
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

fn bench_block_allocation_patterns(c: &mut Criterion) {
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

fn bench_memory_sharing(c: &mut Criterion) {
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

fn bench_chunking_memory_usage(c: &mut Criterion) {
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

fn bench_ipld_memory_efficiency(c: &mut Criterion) {
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

fn bench_cdc_chunking(c: &mut Criterion) {
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

fn bench_cdc_deduplication(c: &mut Criterion) {
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

fn bench_rabin_fingerprinting(c: &mut Criterion) {
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

fn bench_bytes_pool(c: &mut Criterion) {
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

fn bench_cid_string_pool(c: &mut Criterion) {
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

fn bench_pool_vs_direct_allocation(c: &mut Criterion) {
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

fn bench_hash_engines(c: &mut Criterion) {
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

fn bench_hash_registry(c: &mut Criterion) {
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

fn bench_cpu_feature_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_features");

    // Benchmark: Runtime feature detection
    group.bench_function("detect_features", |b| {
        b.iter(|| {
            let _features = CpuFeatures::detect();
        });
    });

    group.finish();
}

fn bench_simd_comparison(c: &mut Criterion) {
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

// ============================================================================
// Batch Processing Benchmarks
// ============================================================================

fn bench_parallel_block_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch/parallel_block_creation");

    let processor = BatchProcessor::new();
    let chunk_counts = [10, 100, 1000];

    for count in chunk_counts {
        // Create test data chunks
        let chunks: Vec<Bytes> = (0..count)
            .map(|i| Bytes::from(format!("chunk data {}", i)))
            .collect();

        let total_bytes: u64 = chunks.iter().map(|c| c.len() as u64).sum();
        group.throughput(Throughput::Bytes(total_bytes));

        group.bench_with_input(BenchmarkId::new("parallel", count), &chunks, |b, chunks| {
            b.iter(|| {
                processor
                    .create_blocks_parallel(black_box(chunks.clone()))
                    .unwrap()
            });
        });

        // Compare with sequential creation
        group.bench_with_input(
            BenchmarkId::new("sequential", count),
            &chunks,
            |b, chunks| {
                b.iter(|| {
                    chunks
                        .iter()
                        .map(|data| Block::new(data.clone()).unwrap())
                        .collect::<Vec<_>>()
                });
            },
        );
    }

    group.finish();
}

fn bench_parallel_cid_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch/parallel_cid_generation");

    let processor = BatchProcessor::new();
    let count = 1000;
    let chunks: Vec<Bytes> = (0..count)
        .map(|i| Bytes::from(format!("data chunk {}", i)))
        .collect();

    let total_bytes: u64 = chunks.iter().map(|c| c.len() as u64).sum();
    group.throughput(Throughput::Bytes(total_bytes));

    group.bench_function("parallel_1000_chunks", |b| {
        b.iter(|| {
            processor
                .generate_cids_parallel(black_box(chunks.clone()))
                .unwrap()
        });
    });

    group.finish();
}

fn bench_parallel_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch/parallel_verification");

    let processor = BatchProcessor::new();
    let count = 1000;

    // Create blocks to verify
    let chunks: Vec<Bytes> = (0..count)
        .map(|i| Bytes::from(format!("verify data {}", i)))
        .collect();
    let blocks = processor.create_blocks_parallel(chunks).unwrap();

    group.bench_function("verify_1000_blocks", |b| {
        b.iter(|| {
            processor
                .verify_blocks_parallel(black_box(&blocks))
                .unwrap()
        });
    });

    group.finish();
}

fn bench_parallel_hashing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch/parallel_hashing");

    let processor = BatchProcessor::new();
    let count = 1000;
    let data_size = 1024; // 1KB per chunk

    let data: Vec<Vec<u8>> = (0..count).map(|_| vec![0x42; data_size]).collect();
    let data_refs: Vec<&[u8]> = data.iter().map(|d| d.as_slice()).collect();

    let total_bytes = (count * data_size) as u64;
    group.throughput(Throughput::Bytes(total_bytes));

    group.bench_function("hash_1000_chunks_1kb", |b| {
        b.iter(|| {
            processor
                .compute_hashes_parallel(black_box(&data_refs))
                .unwrap()
        });
    });

    group.finish();
}

fn bench_batch_operations_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch/scalability");

    let processor = BatchProcessor::new();
    let sizes = [10, 50, 100, 500, 1000, 5000];

    for size in sizes {
        let chunks: Vec<Bytes> = (0..size)
            .map(|i| Bytes::from(format!("scalability test {}", i)))
            .collect();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("blocks", size), &chunks, |b, chunks| {
            b.iter(|| {
                processor
                    .create_blocks_parallel(black_box(chunks.clone()))
                    .unwrap()
            });
        });
    }

    group.finish();
}

fn bench_batch_with_different_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch/hash_algorithms");

    let count = 100;
    let chunks: Vec<Bytes> = (0..count)
        .map(|i| Bytes::from(vec![i as u8; 1024]))
        .collect();

    let total_bytes = (count * 1024) as u64;
    group.throughput(Throughput::Bytes(total_bytes));

    let algorithms = [
        ("sha256", HashAlgorithm::Sha256),
        ("sha3_256", HashAlgorithm::Sha3_256),
    ];

    for (name, algo) in algorithms {
        let processor = BatchProcessor::with_hash_algorithm(algo);
        group.bench_function(name, |b| {
            b.iter(|| {
                processor
                    .create_blocks_parallel(black_box(chunks.clone()))
                    .unwrap()
            });
        });
    }

    group.finish();
}

// ============================================================================
// Codec Registry Benchmarks
// ============================================================================

fn bench_codec_encode_cbor(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec_encode_cbor");

    let registry = CodecRegistry::new();

    // Different IPLD data types
    let test_cases = vec![
        ("null", Ipld::Null),
        ("bool", Ipld::Bool(true)),
        ("integer", Ipld::Integer(42)),
        ("float", Ipld::Float(std::f64::consts::PI)),
        ("string_short", Ipld::String("hello".to_string())),
        ("string_long", Ipld::String("a".repeat(1000))),
        ("bytes_small", Ipld::Bytes(vec![0u8; 64])),
        ("bytes_large", Ipld::Bytes(vec![0u8; 4096])),
    ];

    for (name, ipld) in test_cases {
        group.bench_with_input(BenchmarkId::new("cbor", name), &ipld, |b, ipld| {
            b.iter(|| registry.encode(black_box(codec::DAG_CBOR), black_box(ipld)));
        });
    }

    // Benchmark map encoding
    let mut map = BTreeMap::new();
    for i in 0..100 {
        map.insert(format!("key_{}", i), Ipld::Integer(i as i128));
    }
    let map_ipld = Ipld::Map(map);

    group.bench_with_input(BenchmarkId::new("cbor", "map_100"), &map_ipld, |b, ipld| {
        b.iter(|| registry.encode(black_box(codec::DAG_CBOR), black_box(ipld)));
    });

    group.finish();
}

fn bench_codec_decode_cbor(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec_decode_cbor");

    let registry = CodecRegistry::new();

    // Pre-encode test data
    let test_cases = vec![
        ("null", Ipld::Null),
        ("bool", Ipld::Bool(true)),
        ("integer", Ipld::Integer(42)),
        ("string_short", Ipld::String("hello".to_string())),
        ("string_long", Ipld::String("a".repeat(1000))),
        ("bytes_small", Ipld::Bytes(vec![0u8; 64])),
        ("bytes_large", Ipld::Bytes(vec![0u8; 4096])),
    ];

    for (name, ipld) in test_cases {
        let encoded = registry.encode(codec::DAG_CBOR, &ipld).unwrap();
        group.bench_with_input(BenchmarkId::new("cbor", name), &encoded, |b, encoded| {
            b.iter(|| registry.decode(black_box(codec::DAG_CBOR), black_box(encoded)));
        });
    }

    group.finish();
}

fn bench_codec_encode_json(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec_encode_json");

    let registry = CodecRegistry::new();

    let test_cases = vec![
        ("null", Ipld::Null),
        ("bool", Ipld::Bool(true)),
        ("integer", Ipld::Integer(42)),
        ("string_short", Ipld::String("hello".to_string())),
        ("string_long", Ipld::String("a".repeat(1000))),
        ("bytes_small", Ipld::Bytes(vec![0u8; 64])),
    ];

    for (name, ipld) in test_cases {
        group.bench_with_input(BenchmarkId::new("json", name), &ipld, |b, ipld| {
            b.iter(|| registry.encode(black_box(codec::DAG_JSON), black_box(ipld)));
        });
    }

    group.finish();
}

fn bench_codec_decode_json(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec_decode_json");

    let registry = CodecRegistry::new();

    let test_cases = vec![
        ("null", Ipld::Null),
        ("bool", Ipld::Bool(true)),
        ("integer", Ipld::Integer(42)),
        ("string_short", Ipld::String("hello".to_string())),
        ("string_long", Ipld::String("a".repeat(1000))),
        ("bytes_small", Ipld::Bytes(vec![0u8; 64])),
    ];

    for (name, ipld) in test_cases {
        let encoded = registry.encode(codec::DAG_JSON, &ipld).unwrap();
        group.bench_with_input(BenchmarkId::new("json", name), &encoded, |b, encoded| {
            b.iter(|| registry.decode(black_box(codec::DAG_JSON), black_box(encoded)));
        });
    }

    group.finish();
}

fn bench_codec_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec_roundtrip");

    let registry = CodecRegistry::new();

    // Test CBOR roundtrip
    let ipld = Ipld::String("benchmark data".to_string());
    group.bench_function("cbor_roundtrip", |b| {
        b.iter(|| {
            let encoded = registry.encode(codec::DAG_CBOR, black_box(&ipld)).unwrap();
            registry
                .decode(codec::DAG_CBOR, black_box(&encoded))
                .unwrap()
        });
    });

    // Test JSON roundtrip
    group.bench_function("json_roundtrip", |b| {
        b.iter(|| {
            let encoded = registry.encode(codec::DAG_JSON, black_box(&ipld)).unwrap();
            registry
                .decode(codec::DAG_JSON, black_box(&encoded))
                .unwrap()
        });
    });

    // Test RAW roundtrip
    let bytes_ipld = Ipld::Bytes(vec![0u8; 1024]);
    group.bench_function("raw_roundtrip", |b| {
        b.iter(|| {
            let encoded = registry.encode(codec::RAW, black_box(&bytes_ipld)).unwrap();
            registry.decode(codec::RAW, black_box(&encoded)).unwrap()
        });
    });

    group.finish();
}

fn bench_codec_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec_comparison");

    let registry = CodecRegistry::new();

    // Create a representative data structure
    let mut map = BTreeMap::new();
    map.insert("name".to_string(), Ipld::String("benchmark".to_string()));
    map.insert("count".to_string(), Ipld::Integer(42));
    map.insert("data".to_string(), Ipld::Bytes(vec![0u8; 256]));
    let ipld = Ipld::Map(map);

    // Compare encoding performance
    group.bench_function("cbor_encode", |b| {
        b.iter(|| registry.encode(codec::DAG_CBOR, black_box(&ipld)));
    });

    group.bench_function("json_encode", |b| {
        b.iter(|| registry.encode(codec::DAG_JSON, black_box(&ipld)));
    });

    group.finish();
}

// ============================================================================
// Batch Compression Benchmarks
// ============================================================================

fn bench_batch_compression_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_compression/parallel");

    let processor = BatchProcessor::new();
    let chunk_counts = [10, 50, 100];
    let chunk_size = 4096; // 4KB per chunk

    for count in chunk_counts {
        let chunks: Vec<Bytes> = (0..count)
            .map(|i| Bytes::from(vec![i as u8; chunk_size]))
            .collect();

        let total_bytes = (count * chunk_size) as u64;
        group.throughput(Throughput::Bytes(total_bytes));

        group.bench_with_input(BenchmarkId::new("zstd", count), &chunks, |b, chunks| {
            b.iter(|| {
                processor
                    .compress_data_parallel(
                        black_box(chunks.clone()),
                        CompressionAlgorithm::Zstd,
                        3,
                    )
                    .unwrap()
            });
        });

        group.bench_with_input(BenchmarkId::new("lz4", count), &chunks, |b, chunks| {
            b.iter(|| {
                processor
                    .compress_data_parallel(black_box(chunks.clone()), CompressionAlgorithm::Lz4, 3)
                    .unwrap()
            });
        });
    }

    group.finish();
}

fn bench_batch_decompression_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_compression/decompression");

    let processor = BatchProcessor::new();
    let count = 100;
    let chunk_size = 4096;

    let chunks: Vec<Bytes> = (0..count)
        .map(|i| Bytes::from(vec![i as u8; chunk_size]))
        .collect();

    // Pre-compress data
    let compressed_zstd = processor
        .compress_data_parallel(chunks.clone(), CompressionAlgorithm::Zstd, 3)
        .unwrap();
    let compressed_lz4 = processor
        .compress_data_parallel(chunks.clone(), CompressionAlgorithm::Lz4, 3)
        .unwrap();

    let total_bytes = (count * chunk_size) as u64;
    group.throughput(Throughput::Bytes(total_bytes));

    group.bench_function("zstd", |b| {
        b.iter(|| {
            processor
                .decompress_data_parallel(
                    black_box(compressed_zstd.clone()),
                    CompressionAlgorithm::Zstd,
                )
                .unwrap()
        });
    });

    group.bench_function("lz4", |b| {
        b.iter(|| {
            processor
                .decompress_data_parallel(
                    black_box(compressed_lz4.clone()),
                    CompressionAlgorithm::Lz4,
                )
                .unwrap()
        });
    });

    group.finish();
}

fn bench_batch_compression_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_compression/roundtrip");

    let processor = BatchProcessor::new();
    let count = 50;
    let chunk_size = 8192; // 8KB

    let chunks: Vec<Bytes> = (0..count)
        .map(|i| Bytes::from(vec![i as u8; chunk_size]))
        .collect();

    let total_bytes = (count * chunk_size) as u64;
    group.throughput(Throughput::Bytes(total_bytes));

    group.bench_function("zstd", |b| {
        b.iter(|| {
            let compressed = processor
                .compress_data_parallel(black_box(chunks.clone()), CompressionAlgorithm::Zstd, 3)
                .unwrap();
            processor
                .decompress_data_parallel(compressed, CompressionAlgorithm::Zstd)
                .unwrap()
        });
    });

    group.bench_function("lz4", |b| {
        b.iter(|| {
            let compressed = processor
                .compress_data_parallel(black_box(chunks.clone()), CompressionAlgorithm::Lz4, 3)
                .unwrap();
            processor
                .decompress_data_parallel(compressed, CompressionAlgorithm::Lz4)
                .unwrap()
        });
    });

    group.finish();
}

fn bench_batch_compression_ratio_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_compression/ratio_analysis");

    let processor = BatchProcessor::new();
    let count = 100;

    // Test different data patterns
    let patterns = [
        ("repetitive", vec![0u8; 4096]),
        (
            "sequential",
            (0..4096).map(|i| (i % 256) as u8).collect::<Vec<_>>(),
        ),
    ];

    for (name, pattern) in patterns {
        let chunks: Vec<Bytes> = (0..count).map(|_| Bytes::from(pattern.clone())).collect();

        let total_bytes = (count * pattern.len()) as u64;
        group.throughput(Throughput::Bytes(total_bytes));

        group.bench_with_input(BenchmarkId::new("analyze", name), &chunks, |b, chunks| {
            b.iter(|| {
                processor
                    .analyze_compression_ratios_parallel(
                        black_box(chunks),
                        CompressionAlgorithm::Zstd,
                        5,
                    )
                    .unwrap()
            });
        });
    }

    group.finish();
}

fn bench_batch_compression_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_compression/scalability");

    let processor = BatchProcessor::new();
    let sizes = [10, 50, 100, 200];
    let chunk_size = 2048; // 2KB

    for size in sizes {
        let chunks: Vec<Bytes> = (0..size)
            .map(|i| Bytes::from(vec![i as u8; chunk_size]))
            .collect();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("compress", size), &chunks, |b, chunks| {
            b.iter(|| {
                processor
                    .compress_data_parallel(
                        black_box(chunks.clone()),
                        CompressionAlgorithm::Zstd,
                        3,
                    )
                    .unwrap()
            });
        });
    }

    group.finish();
}

criterion_group!(
    batch_benches,
    bench_parallel_block_creation,
    bench_parallel_cid_generation,
    bench_parallel_verification,
    bench_parallel_hashing,
    bench_batch_operations_scalability,
    bench_batch_with_different_algorithms,
);

criterion_group!(
    batch_compression_benches,
    bench_batch_compression_parallel,
    bench_batch_decompression_parallel,
    bench_batch_compression_roundtrip,
    bench_batch_compression_ratio_analysis,
    bench_batch_compression_scalability,
);

criterion_group!(
    codec_benches,
    bench_codec_encode_cbor,
    bench_codec_decode_cbor,
    bench_codec_encode_json,
    bench_codec_decode_json,
    bench_codec_roundtrip,
    bench_codec_comparison,
);

// ============================================================================
// CAR Format Benchmarks
// ============================================================================

fn bench_car_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("car_write");

    // Different numbers of blocks
    let block_counts = [1, 10, 100, 500];

    for count in block_counts {
        // Create blocks
        let blocks: Vec<Block> = (0..count)
            .map(|i| {
                let data = vec![i as u8; 4096]; // 4KB blocks
                Block::new(Bytes::from(data)).unwrap()
            })
            .collect();

        let total_size = count * 4096;
        group.throughput(Throughput::Bytes(total_size as u64));

        group.bench_with_input(BenchmarkId::new("blocks", count), &blocks, |b, blocks| {
            b.iter(|| {
                let mut car_data = Vec::new();
                let mut writer = CarWriter::new(&mut car_data, vec![*blocks[0].cid()]).unwrap();

                for block in blocks {
                    writer.write_block(black_box(block)).unwrap();
                }
                writer.finish().unwrap();
                car_data
            });
        });
    }

    group.finish();
}

fn bench_car_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("car_read");

    // Different numbers of blocks
    let block_counts = [1, 10, 100, 500];

    for count in block_counts {
        // Create blocks and write to CAR
        let blocks: Vec<Block> = (0..count)
            .map(|i| {
                let data = vec![i as u8; 4096]; // 4KB blocks
                Block::new(Bytes::from(data)).unwrap()
            })
            .collect();

        let mut car_data = Vec::new();
        let mut writer = CarWriter::new(&mut car_data, vec![*blocks[0].cid()]).unwrap();
        for block in &blocks {
            writer.write_block(block).unwrap();
        }
        writer.finish().unwrap();

        let total_size = count * 4096;
        group.throughput(Throughput::Bytes(total_size as u64));

        group.bench_with_input(
            BenchmarkId::new("blocks", count),
            &car_data,
            |b, car_data| {
                b.iter(|| {
                    let mut reader = CarReader::new(&car_data[..]).unwrap();
                    reader.read_all_blocks().unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_car_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("car_roundtrip");

    // Test with different block sizes
    let sizes = [256, 1024, 4096, 16384, 65536];

    for size in sizes {
        let data = vec![0x42u8; size];
        let block = Block::new(Bytes::from(data)).unwrap();

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("size", size), &block, |b, block| {
            b.iter(|| {
                // Write
                let mut car_data = Vec::new();
                let mut writer = CarWriter::new(&mut car_data, vec![*block.cid()]).unwrap();
                writer.write_block(black_box(block)).unwrap();
                writer.finish().unwrap();

                // Read
                let mut reader = CarReader::new(&car_data[..]).unwrap();
                reader.read_block().unwrap()
            });
        });
    }

    group.finish();
}

fn bench_car_large_file(c: &mut Criterion) {
    let mut group = c.benchmark_group("car_large_file");
    group.sample_size(10); // Fewer samples for large files

    // Simulate large file with multiple blocks
    let block_size = 262144; // 256KB per block
    let block_count = 40; // Total: 10MB

    let blocks: Vec<Block> = (0..block_count)
        .map(|i| {
            let data = vec![i as u8; block_size];
            Block::new(Bytes::from(data)).unwrap()
        })
        .collect();

    let total_size = block_size * block_count;
    group.throughput(Throughput::Bytes(total_size as u64));

    group.bench_function("10mb_write", |b| {
        b.iter(|| {
            let mut car_data = Vec::new();
            let mut writer = CarWriter::new(&mut car_data, vec![*blocks[0].cid()]).unwrap();

            for block in &blocks {
                writer.write_block(black_box(block)).unwrap();
            }
            writer.finish().unwrap();
            car_data
        });
    });

    // Pre-create CAR data for read benchmark
    let mut car_data = Vec::new();
    let mut writer = CarWriter::new(&mut car_data, vec![*blocks[0].cid()]).unwrap();
    for block in &blocks {
        writer.write_block(block).unwrap();
    }
    writer.finish().unwrap();

    group.bench_function("10mb_read", |b| {
        b.iter(|| {
            let mut reader = CarReader::new(&car_data[..]).unwrap();
            reader.read_all_blocks().unwrap()
        });
    });

    group.finish();
}

fn bench_car_sequential_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("car_sequential_read");

    let block_count = 100;
    let blocks: Vec<Block> = (0..block_count)
        .map(|i| {
            let data = vec![i as u8; 4096];
            Block::new(Bytes::from(data)).unwrap()
        })
        .collect();

    let mut car_data = Vec::new();
    let mut writer = CarWriter::new(&mut car_data, vec![*blocks[0].cid()]).unwrap();
    for block in &blocks {
        writer.write_block(block).unwrap();
    }
    writer.finish().unwrap();

    let total_size = block_count * 4096;
    group.throughput(Throughput::Bytes(total_size as u64));

    group.bench_function("sequential", |b| {
        b.iter(|| {
            let mut reader = CarReader::new(&car_data[..]).unwrap();
            let mut count = 0;
            while reader.read_block().unwrap().is_some() {
                count += 1;
            }
            count
        });
    });

    group.finish();
}

// ============================================================================
// Compression Benchmarks
// ============================================================================

fn bench_compression_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_algorithms");

    // Different data sizes for compression benchmarks
    let sizes = [1024, 16384, 262144, 1048576]; // 1KB, 16KB, 256KB, 1MB

    for size in sizes {
        // Use semi-compressible data (mix of patterns and random)
        let data: Vec<u8> = (0..size)
            .map(|i| if i % 4 == 0 { (i % 256) as u8 } else { i as u8 })
            .collect();
        let bytes_data = Bytes::from(data);

        group.throughput(Throughput::Bytes(size as u64));

        // Benchmark Zstd compression
        group.bench_with_input(
            BenchmarkId::new("zstd_compress", size),
            &bytes_data,
            |b, data| {
                b.iter(|| compress(black_box(data), CompressionAlgorithm::Zstd, 3));
            },
        );

        // Benchmark Lz4 compression
        group.bench_with_input(
            BenchmarkId::new("lz4_compress", size),
            &bytes_data,
            |b, data| {
                b.iter(|| compress(black_box(data), CompressionAlgorithm::Lz4, 3));
            },
        );

        // Benchmark None (passthrough)
        group.bench_with_input(
            BenchmarkId::new("none_compress", size),
            &bytes_data,
            |b, data| {
                b.iter(|| compress(black_box(data), CompressionAlgorithm::None, 3));
            },
        );
    }

    group.finish();
}

fn bench_decompression_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompression_algorithms");

    let sizes = [1024, 16384, 262144, 1048576];

    for size in sizes {
        let data: Vec<u8> = (0..size)
            .map(|i| if i % 4 == 0 { (i % 256) as u8 } else { i as u8 })
            .collect();
        let bytes_data = Bytes::from(data);

        // Pre-compress the data
        let zstd_compressed = compress(&bytes_data, CompressionAlgorithm::Zstd, 3).unwrap();
        let lz4_compressed = compress(&bytes_data, CompressionAlgorithm::Lz4, 3).unwrap();

        group.throughput(Throughput::Bytes(size as u64));

        // Benchmark Zstd decompression
        group.bench_with_input(
            BenchmarkId::new("zstd_decompress", size),
            &zstd_compressed,
            |b, data| {
                b.iter(|| decompress(black_box(data), CompressionAlgorithm::Zstd));
            },
        );

        // Benchmark Lz4 decompression
        group.bench_with_input(
            BenchmarkId::new("lz4_decompress", size),
            &lz4_compressed,
            |b, data| {
                b.iter(|| decompress(black_box(data), CompressionAlgorithm::Lz4));
            },
        );
    }

    group.finish();
}

fn bench_compression_levels(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_levels");

    // Use 256KB of semi-compressible data
    let data: Vec<u8> = (0..262144)
        .map(|i| if i % 4 == 0 { (i % 256) as u8 } else { i as u8 })
        .collect();
    let bytes_data = Bytes::from(data);

    group.throughput(Throughput::Bytes(262144));

    for level in [0, 3, 6, 9] {
        group.bench_with_input(BenchmarkId::new("zstd", level), &bytes_data, |b, data| {
            b.iter(|| compress(black_box(data), CompressionAlgorithm::Zstd, level));
        });
    }

    group.finish();
}

fn bench_compression_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_roundtrip");

    let sizes = [1024, 16384, 262144];

    for size in sizes {
        let data: Vec<u8> = (0..size)
            .map(|i| if i % 4 == 0 { (i % 256) as u8 } else { i as u8 })
            .collect();
        let bytes_data = Bytes::from(data);

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("zstd_roundtrip", size),
            &bytes_data,
            |b, data| {
                b.iter(|| {
                    let compressed =
                        compress(black_box(data), CompressionAlgorithm::Zstd, 3).unwrap();
                    decompress(&compressed, CompressionAlgorithm::Zstd).unwrap()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("lz4_roundtrip", size),
            &bytes_data,
            |b, data| {
                b.iter(|| {
                    let compressed =
                        compress(black_box(data), CompressionAlgorithm::Lz4, 3).unwrap();
                    decompress(&compressed, CompressionAlgorithm::Lz4).unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_compression_ratio_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_ratio");

    let sizes = [1024, 16384, 262144];

    for size in sizes {
        let data: Vec<u8> = (0..size)
            .map(|i| if i % 4 == 0 { (i % 256) as u8 } else { i as u8 })
            .collect();
        let bytes_data = Bytes::from(data);

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("zstd", size), &bytes_data, |b, data| {
            b.iter(|| compression_ratio(black_box(data), CompressionAlgorithm::Zstd, 3));
        });
    }

    group.finish();
}

criterion_group!(
    compression_benches,
    bench_compression_algorithms,
    bench_decompression_algorithms,
    bench_compression_levels,
    bench_compression_roundtrip,
    bench_compression_ratio_calculation,
);

criterion_group!(
    car_benches,
    bench_car_write,
    bench_car_read,
    bench_car_roundtrip,
    bench_car_large_file,
    bench_car_sequential_read,
);

// CAR Compression Benchmarks
fn bench_car_compression_write(c: &mut Criterion) {
    use bytes::Bytes;
    use ipfrs_core::car::CarWriterBuilder;
    use ipfrs_core::compression::CompressionAlgorithm;
    use ipfrs_core::Block;

    let blocks: Vec<Block> = (0..100)
        .map(|_| Block::new(Bytes::from(vec![0x42u8; 1024])).unwrap())
        .collect();

    let mut group = c.benchmark_group("car_compression_write");

    // Benchmark Zstd compression
    group.bench_function("zstd_level_3", |b| {
        b.iter(|| {
            let mut output = Vec::new();
            let mut writer = CarWriterBuilder::new(vec![*blocks[0].cid()])
                .with_compression(CompressionAlgorithm::Zstd, 3)
                .build(&mut output)
                .unwrap();
            for block in &blocks {
                writer.write_block(black_box(block)).unwrap();
            }
            writer.finish().unwrap();
            black_box(output);
        });
    });

    // Benchmark LZ4 compression
    group.bench_function("lz4_level_1", |b| {
        b.iter(|| {
            let mut output = Vec::new();
            let mut writer = CarWriterBuilder::new(vec![*blocks[0].cid()])
                .with_compression(CompressionAlgorithm::Lz4, 1)
                .build(&mut output)
                .unwrap();
            for block in &blocks {
                writer.write_block(black_box(block)).unwrap();
            }
            writer.finish().unwrap();
            black_box(output);
        });
    });

    // Benchmark uncompressed (baseline)
    group.bench_function("uncompressed", |b| {
        b.iter(|| {
            let mut output = Vec::new();
            let mut writer = CarWriterBuilder::new(vec![*blocks[0].cid()])
                .build(&mut output)
                .unwrap();
            for block in &blocks {
                writer.write_block(black_box(block)).unwrap();
            }
            writer.finish().unwrap();
            black_box(output);
        });
    });

    group.finish();
}

fn bench_car_compression_read(c: &mut Criterion) {
    use bytes::Bytes;
    use ipfrs_core::car::{CarReader, CarWriterBuilder};
    use ipfrs_core::compression::CompressionAlgorithm;
    use ipfrs_core::Block;

    let blocks: Vec<Block> = (0..100)
        .map(|_| Block::new(Bytes::from(vec![0x42u8; 1024])).unwrap())
        .collect();

    // Create compressed CAR data
    let mut zstd_data = Vec::new();
    let mut writer = CarWriterBuilder::new(vec![*blocks[0].cid()])
        .with_compression(CompressionAlgorithm::Zstd, 3)
        .build(&mut zstd_data)
        .unwrap();
    for block in &blocks {
        writer.write_block(block).unwrap();
    }
    writer.finish().unwrap();

    let mut lz4_data = Vec::new();
    let mut writer = CarWriterBuilder::new(vec![*blocks[0].cid()])
        .with_compression(CompressionAlgorithm::Lz4, 1)
        .build(&mut lz4_data)
        .unwrap();
    for block in &blocks {
        writer.write_block(block).unwrap();
    }
    writer.finish().unwrap();

    let mut group = c.benchmark_group("car_compression_read");

    group.bench_function("zstd_decompression", |b| {
        b.iter(|| {
            let mut reader = CarReader::new(black_box(&zstd_data[..])).unwrap();
            while let Some(block) = reader.read_block().unwrap() {
                black_box(block);
            }
        });
    });

    group.bench_function("lz4_decompression", |b| {
        b.iter(|| {
            let mut reader = CarReader::new(black_box(&lz4_data[..])).unwrap();
            while let Some(block) = reader.read_block().unwrap() {
                black_box(block);
            }
        });
    });

    group.finish();
}

fn bench_car_compression_roundtrip(c: &mut Criterion) {
    use bytes::Bytes;
    use ipfrs_core::car::{CarReader, CarWriterBuilder};
    use ipfrs_core::compression::CompressionAlgorithm;
    use ipfrs_core::Block;

    let blocks: Vec<Block> = (0..50)
        .map(|_| Block::new(Bytes::from(vec![0x55u8; 2048])).unwrap())
        .collect();

    let mut group = c.benchmark_group("car_compression_roundtrip");

    group.bench_function("zstd_write_read", |b| {
        b.iter(|| {
            // Write
            let mut output = Vec::new();
            let mut writer = CarWriterBuilder::new(vec![*blocks[0].cid()])
                .with_compression(CompressionAlgorithm::Zstd, 3)
                .build(&mut output)
                .unwrap();
            for block in &blocks {
                writer.write_block(block).unwrap();
            }
            writer.finish().unwrap();

            // Read
            let mut reader = CarReader::new(&output[..]).unwrap();
            while let Some(block) = reader.read_block().unwrap() {
                black_box(block);
            }
        });
    });

    group.bench_function("lz4_write_read", |b| {
        b.iter(|| {
            // Write
            let mut output = Vec::new();
            let mut writer = CarWriterBuilder::new(vec![*blocks[0].cid()])
                .with_compression(CompressionAlgorithm::Lz4, 1)
                .build(&mut output)
                .unwrap();
            for block in &blocks {
                writer.write_block(block).unwrap();
            }
            writer.finish().unwrap();

            // Read
            let mut reader = CarReader::new(&output[..]).unwrap();
            while let Some(block) = reader.read_block().unwrap() {
                black_box(block);
            }
        });
    });

    group.finish();
}

fn bench_car_compression_ratios(c: &mut Criterion) {
    use bytes::Bytes;
    use ipfrs_core::car::CarWriterBuilder;
    use ipfrs_core::compression::CompressionAlgorithm;
    use ipfrs_core::Block;

    let repetitive_blocks: Vec<Block> = (0..100)
        .map(|_| Block::new(Bytes::from(vec![0x00u8; 1024])).unwrap())
        .collect();

    let random_blocks: Vec<Block> = (0..100)
        .map(|i| Block::new(Bytes::from(vec![i as u8; 1024])).unwrap())
        .collect();

    let mut group = c.benchmark_group("car_compression_ratios");

    group.bench_function("repetitive_data_zstd", |b| {
        b.iter(|| {
            let mut output = Vec::new();
            let mut writer = CarWriterBuilder::new(vec![*repetitive_blocks[0].cid()])
                .with_compression(CompressionAlgorithm::Zstd, 6)
                .build(&mut output)
                .unwrap();
            for block in &repetitive_blocks {
                writer.write_block(block).unwrap();
            }
            let stats = writer.stats().clone();
            writer.finish().unwrap();
            black_box((output.len(), stats));
        });
    });

    group.bench_function("random_data_zstd", |b| {
        b.iter(|| {
            let mut output = Vec::new();
            let mut writer = CarWriterBuilder::new(vec![*random_blocks[0].cid()])
                .with_compression(CompressionAlgorithm::Zstd, 6)
                .build(&mut output)
                .unwrap();
            for block in &random_blocks {
                writer.write_block(block).unwrap();
            }
            let stats = writer.stats().clone();
            writer.finish().unwrap();
            black_box((output.len(), stats));
        });
    });

    group.finish();
}

criterion_group!(
    car_compression_benches,
    bench_car_compression_write,
    bench_car_compression_read,
    bench_car_compression_roundtrip,
    bench_car_compression_ratios,
);

// ============================================================================
// DAG Algorithm Benchmarks
// ============================================================================

fn bench_dag_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("dag_metrics");

    // Create IPLD structures of varying complexity
    let sizes = [10, 50, 100, 200];

    for size in sizes {
        // Create a nested IPLD structure
        let items: Vec<Ipld> = (0..size).map(|i| Ipld::Integer(i as i128)).collect();
        let ipld = Ipld::List(items);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &ipld, |b, ipld| {
            b.iter(|| DagMetrics::from_ipld(black_box(ipld)));
        });
    }

    group.finish();
}

fn bench_topological_sort(c: &mut Criterion) {
    let mut group = c.benchmark_group("topological_sort");

    let sizes = [10, 50, 100, 200];

    for size in sizes {
        // Create CIDs
        let cids: Vec<Cid> = (0..size)
            .map(|i| {
                let data = format!("data{}", i);
                CidBuilder::new().build(data.as_bytes()).unwrap()
            })
            .collect();

        // Create IPLD with links (with some duplicates)
        let mut ipld_links = Vec::new();
        for cid in &cids {
            ipld_links.push(Ipld::link(*cid));
            if cids.len() > 10 {
                ipld_links.push(Ipld::link(cids[0])); // Add duplicate
            }
        }
        let ipld = Ipld::List(ipld_links);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &ipld, |b, ipld| {
            b.iter(|| topological_sort(black_box(ipld)));
        });
    }

    group.finish();
}

fn bench_subgraph_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("subgraph_size");

    let sizes = [10, 50, 100, 200, 500];

    for size in sizes {
        // Create nested structure
        let items: Vec<Ipld> = (0..size).map(|i| Ipld::Integer(i as i128)).collect();
        let ipld = Ipld::List(items);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &ipld, |b, ipld| {
            b.iter(|| subgraph_size(black_box(ipld)));
        });
    }

    group.finish();
}

fn bench_count_links_by_depth(c: &mut Criterion) {
    let mut group = c.benchmark_group("count_links_by_depth");

    let sizes = [10, 50, 100, 200];

    for size in sizes {
        // Create CIDs at various depths
        let cids: Vec<Cid> = (0..size)
            .map(|i| {
                let data = format!("data{}", i);
                CidBuilder::new().build(data.as_bytes()).unwrap()
            })
            .collect();

        // Create nested structure with links
        let mut inner = BTreeMap::new();
        for (i, cid) in cids.iter().enumerate().take(size / 2) {
            inner.insert(format!("link{}", i), Ipld::link(*cid));
        }

        let mut outer = BTreeMap::new();
        for (i, cid) in cids.iter().enumerate().skip(size / 2) {
            outer.insert(format!("link{}", i), Ipld::link(*cid));
        }
        outer.insert("nested".to_string(), Ipld::Map(inner));

        let ipld = Ipld::Map(outer);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &ipld, |b, ipld| {
            b.iter(|| count_links_by_depth(black_box(ipld)));
        });
    }

    group.finish();
}

fn bench_dag_fanout_by_level(c: &mut Criterion) {
    let mut group = c.benchmark_group("dag_fanout_by_level");

    let sizes = [10, 50, 100, 200];

    for size in sizes {
        // Create nested structure
        let items: Vec<Ipld> = (0..size).map(|i| Ipld::Integer(i as i128)).collect();
        let ipld = Ipld::List(items);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &ipld, |b, ipld| {
            b.iter(|| dag_fanout_by_level(black_box(ipld)));
        });
    }

    group.finish();
}

fn bench_filter_dag(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_dag");

    let sizes = [10, 50, 100, 200];

    for size in sizes {
        // Create mixed IPLD structure
        let items: Vec<Ipld> = (0..size)
            .map(|i| {
                if i % 2 == 0 {
                    Ipld::Integer(i as i128)
                } else {
                    Ipld::String(format!("str{}", i))
                }
            })
            .collect();
        let ipld = Ipld::List(items);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &ipld, |b, ipld| {
            b.iter(|| {
                filter_dag(black_box(ipld), &|node| {
                    matches!(node, Ipld::Integer(_) | Ipld::List(_))
                })
            });
        });
    }

    group.finish();
}

fn bench_map_dag(c: &mut Criterion) {
    let mut group = c.benchmark_group("map_dag");

    let sizes = [10, 50, 100, 200];

    for size in sizes {
        // Create nested structure
        let items: Vec<Ipld> = (0..size).map(|i| Ipld::Integer(i as i128)).collect();
        let ipld = Ipld::List(items);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &ipld, |b, ipld| {
            b.iter(|| {
                map_dag(black_box(ipld), &|node| match node {
                    Ipld::Integer(n) => Ipld::Integer(n * 2),
                    other => other.clone(),
                })
            });
        });
    }

    group.finish();
}

criterion_group!(
    dag_benches,
    bench_dag_metrics,
    bench_topological_sort,
    bench_subgraph_size,
    bench_count_links_by_depth,
    bench_dag_fanout_by_level,
    bench_filter_dag,
    bench_map_dag,
);

criterion_main!(
    cid_benches,
    block_benches,
    ipld_benches,
    chunking_benches,
    streaming_benches,
    memory_benches,
    cdc_benches,
    pool_benches,
    hash_benches,
    batch_benches,
    batch_compression_benches,
    codec_benches,
    compression_benches,
    car_benches,
    car_compression_benches,
    dag_benches,
);
