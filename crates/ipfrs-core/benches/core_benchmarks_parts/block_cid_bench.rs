//! CID, Block, IPLD, Chunking, Streaming, DAG Node benchmarks

use bytes::Bytes;
use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};
use ipfrs_core::{
    read_chunked_file, Block, BlockReader, Chunker, ChunkingConfig, Cid, CidBuilder, CidExt,
    DagNode, Ipld, MemoryBlockFetcher, MultibaseEncoding,
};
use std::collections::BTreeMap;
use std::hint::black_box;
use std::io::Read;

// ============================================================================
// CID Benchmarks
// ============================================================================

pub fn bench_cid_generation(c: &mut Criterion) {
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

pub fn bench_cid_parsing(c: &mut Criterion) {
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

pub fn bench_cid_encoding(c: &mut Criterion) {
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

pub fn bench_block_creation(c: &mut Criterion) {
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

pub fn bench_block_verification(c: &mut Criterion) {
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

pub fn bench_ipld_dag_cbor(c: &mut Criterion) {
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

pub fn bench_ipld_dag_json(c: &mut Criterion) {
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

pub fn bench_chunking(c: &mut Criterion) {
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

pub fn bench_dag_node(c: &mut Criterion) {
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

pub fn bench_block_reader(c: &mut Criterion) {
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

pub fn bench_chunked_file_read(c: &mut Criterion) {
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
