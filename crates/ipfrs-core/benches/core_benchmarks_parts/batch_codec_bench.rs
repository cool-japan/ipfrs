//! Batch Processing and Codec Registry benchmarks

use bytes::Bytes;
use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};
use ipfrs_core::{
    codec, BatchProcessor, Block, CodecRegistry, CompressionAlgorithm, HashAlgorithm, Ipld,
};
use std::collections::BTreeMap;
use std::hint::black_box;

// ============================================================================
// Batch Processing Benchmarks
// ============================================================================

pub fn bench_parallel_block_creation(c: &mut Criterion) {
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

pub fn bench_parallel_cid_generation(c: &mut Criterion) {
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

pub fn bench_parallel_verification(c: &mut Criterion) {
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

pub fn bench_parallel_hashing(c: &mut Criterion) {
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

pub fn bench_batch_operations_scalability(c: &mut Criterion) {
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

pub fn bench_batch_with_different_algorithms(c: &mut Criterion) {
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

pub fn bench_codec_encode_cbor(c: &mut Criterion) {
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

pub fn bench_codec_decode_cbor(c: &mut Criterion) {
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

pub fn bench_codec_encode_json(c: &mut Criterion) {
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

pub fn bench_codec_decode_json(c: &mut Criterion) {
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

pub fn bench_codec_roundtrip(c: &mut Criterion) {
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

pub fn bench_codec_comparison(c: &mut Criterion) {
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

pub fn bench_batch_compression_parallel(c: &mut Criterion) {
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

pub fn bench_batch_decompression_parallel(c: &mut Criterion) {
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

pub fn bench_batch_compression_roundtrip(c: &mut Criterion) {
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

pub fn bench_batch_compression_ratio_analysis(c: &mut Criterion) {
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

pub fn bench_batch_compression_scalability(c: &mut Criterion) {
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
