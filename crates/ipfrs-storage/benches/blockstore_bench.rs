use bytes::Bytes;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ipfrs_core::Block;
use ipfrs_storage::{
    traits::BlockStore as BlockStoreTrait, BlockStoreConfig, ParityDbBlockStore, ParityDbConfig,
    SledBlockStore,
};
use ipfrs_storage::{ChunkingConfig, DedupBlockStore};
#[cfg(feature = "compression")]
use ipfrs_storage::{CompressionAlgorithm, CompressionBlockStore, CompressionConfig};
use std::hint::black_box;
use tokio::runtime::Runtime;

fn create_test_blocks(count: usize, size: usize) -> Vec<Block> {
    (0..count)
        .map(|i| {
            let mut data = vec![0u8; size];
            // Make each block unique
            let i_bytes = i.to_le_bytes();
            data[..i_bytes.len()].copy_from_slice(&i_bytes);
            Block::new(Bytes::from(data)).expect("bench: create test block")
        })
        .collect()
}

fn create_dedup_blocks(count: usize, size: usize, duplicate_ratio: f32) -> Vec<Block> {
    // Create blocks with controlled duplication
    // duplicate_ratio: 0.0 = all unique, 1.0 = all identical
    let unique_count = ((count as f32) * (1.0 - duplicate_ratio)).max(1.0) as usize;

    let unique_blocks: Vec<Block> = (0..unique_count)
        .map(|i| {
            // Create varied data for better chunking
            let data: Vec<u8> = (0..size).map(|j| ((i * 1000 + j) % 256) as u8).collect();
            Block::new(Bytes::from(data)).expect("bench: create dedup block")
        })
        .collect();

    // Create full block list by repeating unique blocks
    (0..count)
        .map(|i| unique_blocks[i % unique_count].clone())
        .collect()
}

#[cfg(feature = "compression")]
fn create_compressible_blocks(count: usize, size: usize) -> Vec<Block> {
    // Create highly compressible data (repeated patterns)
    (0..count)
        .map(|i| {
            let mut data = vec![42u8; size];
            // Add small unique identifier to make each block unique
            let i_bytes = i.to_le_bytes();
            data[..i_bytes.len()].copy_from_slice(&i_bytes);
            Block::new(Bytes::from(data)).expect("bench: create compressible block")
        })
        .collect()
}

#[cfg(feature = "compression")]
fn create_incompressible_blocks(count: usize, size: usize) -> Vec<Block> {
    // Create random data (incompressible)
    use rand::RngExt;
    let mut rng = rand::rng();

    (0..count)
        .map(|_| {
            let data: Vec<u8> = (0..size).map(|_| rng.random_range(0..=255)).collect();
            Block::new(Bytes::from(data)).expect("bench: create incompressible block")
        })
        .collect()
}

fn bench_sled_single_put(c: &mut Criterion) {
    let rt = Runtime::new().expect("bench: create tokio runtime");
    let mut group = c.benchmark_group("sled_single_put");

    for size in [1024, 10 * 1024, 100 * 1024, 1024 * 1024] {
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let config = BlockStoreConfig {
                path: std::env::temp_dir().join("ipfrs-bench-sled"),
                cache_size: 100 * 1024 * 1024,
            };
            let _ = std::fs::remove_dir_all(&config.path);

            let store = SledBlockStore::new(config).expect("bench: open sled store");
            let blocks = create_test_blocks(100, size);
            let mut idx = 0;

            b.iter(|| {
                rt.block_on(async {
                    store
                        .put(&blocks[idx % blocks.len()])
                        .await
                        .expect("bench: put block");
                });
                idx += 1;
            });
        });
    }

    group.finish();
}

fn bench_sled_single_get(c: &mut Criterion) {
    let rt = Runtime::new().expect("bench: create tokio runtime");
    let mut group = c.benchmark_group("sled_single_get");

    for size in [1024, 10 * 1024, 100 * 1024, 1024 * 1024] {
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let config = BlockStoreConfig {
                path: std::env::temp_dir().join("ipfrs-bench-sled-get"),
                cache_size: 100 * 1024 * 1024,
            };
            let _ = std::fs::remove_dir_all(&config.path);

            let store = SledBlockStore::new(config).expect("bench: open sled store");
            let blocks = create_test_blocks(100, size);

            // Pre-populate
            rt.block_on(async {
                store
                    .put_many(&blocks)
                    .await
                    .expect("bench: put many blocks");
            });

            let mut idx = 0;
            b.iter(|| {
                let cid = blocks[idx % blocks.len()].cid();
                rt.block_on(async {
                    let _ = black_box(store.get(cid).await.expect("bench: get block"));
                });
                idx += 1;
            });
        });
    }

    group.finish();
}

fn bench_sled_batch_put(c: &mut Criterion) {
    let rt = Runtime::new().expect("bench: create tokio runtime");
    let mut group = c.benchmark_group("sled_batch_put");

    for batch_size in [10, 100, 1000] {
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &batch_size| {
                let config = BlockStoreConfig {
                    path: std::env::temp_dir().join("ipfrs-bench-sled-batch"),
                    cache_size: 100 * 1024 * 1024,
                };
                let _ = std::fs::remove_dir_all(&config.path);

                let store = SledBlockStore::new(config).expect("bench: open sled store");
                let blocks = create_test_blocks(batch_size, 1024);

                b.iter(|| {
                    rt.block_on(async {
                        store
                            .put_many(&blocks)
                            .await
                            .expect("bench: put many blocks");
                    });
                });
            },
        );
    }

    group.finish();
}

fn bench_paritydb_single_put(c: &mut Criterion) {
    let rt = Runtime::new().expect("bench: create tokio runtime");
    let mut group = c.benchmark_group("paritydb_single_put");

    for size in [1024, 10 * 1024, 100 * 1024, 1024 * 1024] {
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let config =
                ParityDbConfig::fast_write(std::env::temp_dir().join("ipfrs-bench-paritydb"));
            let _ = std::fs::remove_dir_all(&config.path);

            let store = ParityDbBlockStore::new(config).expect("bench: open paritydb store");
            let blocks = create_test_blocks(100, size);
            let mut idx = 0;

            b.iter(|| {
                rt.block_on(async {
                    store
                        .put(&blocks[idx % blocks.len()])
                        .await
                        .expect("bench: put block");
                });
                idx += 1;
            });
        });
    }

    group.finish();
}

fn bench_paritydb_single_get(c: &mut Criterion) {
    let rt = Runtime::new().expect("bench: create tokio runtime");
    let mut group = c.benchmark_group("paritydb_single_get");

    for size in [1024, 10 * 1024, 100 * 1024, 1024 * 1024] {
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let config =
                ParityDbConfig::balanced(std::env::temp_dir().join("ipfrs-bench-paritydb-get"));
            let _ = std::fs::remove_dir_all(&config.path);

            let store = ParityDbBlockStore::new(config).expect("bench: open paritydb store");
            let blocks = create_test_blocks(100, size);

            // Pre-populate
            rt.block_on(async {
                store
                    .put_many(&blocks)
                    .await
                    .expect("bench: put many blocks");
            });

            let mut idx = 0;
            b.iter(|| {
                let cid = blocks[idx % blocks.len()].cid();
                rt.block_on(async {
                    let _ = black_box(store.get(cid).await.expect("bench: get block"));
                });
                idx += 1;
            });
        });
    }

    group.finish();
}

fn bench_paritydb_batch_put(c: &mut Criterion) {
    let rt = Runtime::new().expect("bench: create tokio runtime");
    let mut group = c.benchmark_group("paritydb_batch_put");

    for batch_size in [10, 100, 1000] {
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &batch_size| {
                let config = ParityDbConfig::fast_write(
                    std::env::temp_dir().join("ipfrs-bench-paritydb-batch"),
                );
                let _ = std::fs::remove_dir_all(&config.path);

                let store = ParityDbBlockStore::new(config).expect("bench: open paritydb store");
                let blocks = create_test_blocks(batch_size, 1024);

                b.iter(|| {
                    rt.block_on(async {
                        store
                            .put_many(&blocks)
                            .await
                            .expect("bench: put many blocks");
                    });
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "compression")]
fn bench_compression_algorithms(c: &mut Criterion) {
    let rt = Runtime::new().expect("bench: create tokio runtime");
    let mut group = c.benchmark_group("compression_algorithms");

    let algorithms = [
        ("zstd", CompressionAlgorithm::Zstd),
        ("lz4", CompressionAlgorithm::Lz4),
        ("snappy", CompressionAlgorithm::Snappy),
    ];

    for (name, algorithm) in algorithms {
        for size in [10 * 1024, 100 * 1024, 1024 * 1024] {
            group.throughput(Throughput::Bytes(size as u64));
            group.bench_with_input(
                BenchmarkId::new(name, size),
                &(algorithm, size),
                |b, &(algo, size)| {
                    let config = BlockStoreConfig {
                        path: std::env::temp_dir()
                            .join(format!("ipfrs-bench-compression-{}", name)),
                        cache_size: 100 * 1024 * 1024,
                    };
                    let _ = std::fs::remove_dir_all(&config.path);

                    let store = SledBlockStore::new(config).expect("bench: open sled store");
                    let compression_config = CompressionConfig::new(algo);
                    let compressed_store = CompressionBlockStore::new(store, compression_config);

                    let blocks = create_compressible_blocks(10, size);
                    let mut idx = 0;

                    b.iter(|| {
                        rt.block_on(async {
                            compressed_store
                                .put(&blocks[idx % blocks.len()])
                                .await
                                .expect("bench: put compressed block");
                        });
                        idx += 1;
                    });
                },
            );
        }
    }

    group.finish();
}

#[cfg(feature = "compression")]
fn bench_compression_vs_uncompressed(c: &mut Criterion) {
    let rt = Runtime::new().expect("bench: create tokio runtime");
    let mut group = c.benchmark_group("compression_vs_uncompressed");

    let size = 100 * 1024; // 100KB blocks
    group.throughput(Throughput::Bytes(size as u64));

    // Uncompressed baseline
    group.bench_function("uncompressed", |b| {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-bench-uncomp"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);

        let store = SledBlockStore::new(config).expect("bench: open sled store");
        let blocks = create_compressible_blocks(10, size);
        let mut idx = 0;

        b.iter(|| {
            rt.block_on(async {
                store
                    .put(&blocks[idx % blocks.len()])
                    .await
                    .expect("bench: put block");
            });
            idx += 1;
        });
    });

    // Zstd compressed
    group.bench_function("zstd_compressed", |b| {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-bench-comp-zstd"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);

        let store = SledBlockStore::new(config).expect("bench: open sled store");
        let compression_config = CompressionConfig::new(CompressionAlgorithm::Zstd);
        let compressed_store = CompressionBlockStore::new(store, compression_config);

        let blocks = create_compressible_blocks(10, size);
        let mut idx = 0;

        b.iter(|| {
            rt.block_on(async {
                compressed_store
                    .put(&blocks[idx % blocks.len()])
                    .await
                    .expect("bench: put compressed block");
            });
            idx += 1;
        });
    });

    group.finish();
}

#[cfg(feature = "compression")]
fn bench_compression_compressible_vs_random(c: &mut Criterion) {
    let rt = Runtime::new().expect("bench: create tokio runtime");
    let mut group = c.benchmark_group("compression_data_types");

    let size = 100 * 1024; // 100KB blocks
    group.throughput(Throughput::Bytes(size as u64));

    // Compressible data
    group.bench_function("compressible", |b| {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-bench-comp-compressible"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);

        let store = SledBlockStore::new(config).expect("bench: open sled store");
        let compression_config = CompressionConfig::new(CompressionAlgorithm::Zstd);
        let compressed_store = CompressionBlockStore::new(store, compression_config);

        let blocks = create_compressible_blocks(10, size);
        let mut idx = 0;

        b.iter(|| {
            rt.block_on(async {
                compressed_store
                    .put(&blocks[idx % blocks.len()])
                    .await
                    .expect("bench: put compressed block");
            });
            idx += 1;
        });
    });

    // Incompressible (random) data
    group.bench_function("incompressible", |b| {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-bench-comp-random"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);

        let store = SledBlockStore::new(config).expect("bench: open sled store");
        let compression_config =
            CompressionConfig::new(CompressionAlgorithm::Zstd).with_max_ratio(0.9);
        let compressed_store = CompressionBlockStore::new(store, compression_config);

        let blocks = create_incompressible_blocks(10, size);
        let mut idx = 0;

        b.iter(|| {
            rt.block_on(async {
                compressed_store
                    .put(&blocks[idx % blocks.len()])
                    .await
                    .expect("bench: put compressed block");
            });
            idx += 1;
        });
    });

    group.finish();
}

fn bench_dedup_unique_blocks(c: &mut Criterion) {
    let rt = Runtime::new().expect("bench: create tokio runtime");
    let mut group = c.benchmark_group("dedup_unique_blocks");

    let size = 256 * 1024; // 256KB blocks
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_function("dedup_put", |b| {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-bench-dedup-unique"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);

        let store = SledBlockStore::new(config).expect("bench: open sled store");
        let chunk_config = ChunkingConfig::default();
        let dedup_store = DedupBlockStore::new(store, chunk_config);

        let blocks = create_dedup_blocks(100, size, 0.0); // All unique
        let mut idx = 0;

        b.iter(|| {
            rt.block_on(async {
                dedup_store
                    .put(&blocks[idx % blocks.len()])
                    .await
                    .expect("bench: put block");
            });
            idx += 1;
        });
    });

    group.finish();
}

fn bench_dedup_duplicate_blocks(c: &mut Criterion) {
    let rt = Runtime::new().expect("bench: create tokio runtime");
    let mut group = c.benchmark_group("dedup_duplicate_blocks");

    let size = 256 * 1024; // 256KB blocks
    group.throughput(Throughput::Bytes(size as u64));

    // Benchmark with 50% duplication
    group.bench_function("50pct_duplication", |b| {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-bench-dedup-50pct"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);

        let store = SledBlockStore::new(config).expect("bench: open sled store");
        let chunk_config = ChunkingConfig::default();
        let dedup_store = DedupBlockStore::new(store, chunk_config);

        let blocks = create_dedup_blocks(100, size, 0.5); // 50% duplicates
        let mut idx = 0;

        b.iter(|| {
            rt.block_on(async {
                dedup_store
                    .put(&blocks[idx % blocks.len()])
                    .await
                    .expect("bench: put block");
            });
            idx += 1;
        });
    });

    // Benchmark with 90% duplication
    group.bench_function("90pct_duplication", |b| {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-bench-dedup-90pct"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);

        let store = SledBlockStore::new(config).expect("bench: open sled store");
        let chunk_config = ChunkingConfig::default();
        let dedup_store = DedupBlockStore::new(store, chunk_config);

        let blocks = create_dedup_blocks(100, size, 0.9); // 90% duplicates
        let mut idx = 0;

        b.iter(|| {
            rt.block_on(async {
                dedup_store
                    .put(&blocks[idx % blocks.len()])
                    .await
                    .expect("bench: put block");
            });
            idx += 1;
        });
    });

    group.finish();
}

fn bench_dedup_chunk_sizes(c: &mut Criterion) {
    let rt = Runtime::new().expect("bench: create tokio runtime");
    let mut group = c.benchmark_group("dedup_chunk_sizes");

    let block_size = 4 * 1024 * 1024; // 4MB blocks
    group.throughput(Throughput::Bytes(block_size as u64));

    // Small chunks (256KB target)
    group.bench_function("small_chunks", |b| {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-bench-dedup-small"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);

        let store = SledBlockStore::new(config).expect("bench: open sled store");
        let chunk_config = ChunkingConfig::small();
        let dedup_store = DedupBlockStore::new(store, chunk_config);

        let blocks = create_dedup_blocks(20, block_size, 0.3);
        let mut idx = 0;

        b.iter(|| {
            rt.block_on(async {
                dedup_store
                    .put(&blocks[idx % blocks.len()])
                    .await
                    .expect("bench: put block");
            });
            idx += 1;
        });
    });

    // Large chunks (4MB target)
    group.bench_function("large_chunks", |b| {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-bench-dedup-large"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);

        let store = SledBlockStore::new(config).expect("bench: open sled store");
        let chunk_config = ChunkingConfig::large();
        let dedup_store = DedupBlockStore::new(store, chunk_config);

        let blocks = create_dedup_blocks(20, block_size, 0.3);
        let mut idx = 0;

        b.iter(|| {
            rt.block_on(async {
                dedup_store
                    .put(&blocks[idx % blocks.len()])
                    .await
                    .expect("bench: put block");
            });
            idx += 1;
        });
    });

    group.finish();
}

criterion_group!(
    dedup_benches,
    bench_dedup_unique_blocks,
    bench_dedup_duplicate_blocks,
    bench_dedup_chunk_sizes,
);

#[cfg(feature = "compression")]
criterion_group!(
    compression_benches,
    bench_compression_algorithms,
    bench_compression_vs_uncompressed,
    bench_compression_compressible_vs_random,
);

criterion_group!(
    benches,
    bench_sled_single_put,
    bench_sled_single_get,
    bench_sled_batch_put,
    bench_paritydb_single_put,
    bench_paritydb_single_get,
    bench_paritydb_batch_put,
);

#[cfg(feature = "compression")]
criterion_main!(benches, compression_benches, dedup_benches);

#[cfg(not(feature = "compression"))]
criterion_main!(benches, dedup_benches);
