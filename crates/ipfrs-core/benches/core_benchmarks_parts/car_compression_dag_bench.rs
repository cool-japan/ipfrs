//! CAR Format, Compression, CAR Compression, and DAG Algorithm benchmarks

use bytes::Bytes;
use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};
use ipfrs_core::{
    compress, compression_ratio, count_links_by_depth, dag_fanout_by_level, decompress, filter_dag,
    map_dag, subgraph_size, topological_sort, Block, CarReader, CarWriter, CarWriterBuilder, Cid,
    CidBuilder, CompressionAlgorithm, DagMetrics, Ipld,
};
use std::collections::BTreeMap;
use std::hint::black_box;

// ============================================================================
// CAR Format Benchmarks
// ============================================================================

pub fn bench_car_write(c: &mut Criterion) {
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

pub fn bench_car_read(c: &mut Criterion) {
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

pub fn bench_car_roundtrip(c: &mut Criterion) {
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

pub fn bench_car_large_file(c: &mut Criterion) {
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

pub fn bench_car_sequential_read(c: &mut Criterion) {
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

pub fn bench_compression_algorithms(c: &mut Criterion) {
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

pub fn bench_decompression_algorithms(c: &mut Criterion) {
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

pub fn bench_compression_levels(c: &mut Criterion) {
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

pub fn bench_compression_roundtrip(c: &mut Criterion) {
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

pub fn bench_compression_ratio_calculation(c: &mut Criterion) {
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

// ============================================================================
// CAR Compression Benchmarks
// ============================================================================

pub fn bench_car_compression_write(c: &mut Criterion) {
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

pub fn bench_car_compression_read(c: &mut Criterion) {
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

pub fn bench_car_compression_roundtrip(c: &mut Criterion) {
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

pub fn bench_car_compression_ratios(c: &mut Criterion) {
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

// ============================================================================
// DAG Algorithm Benchmarks
// ============================================================================

pub fn bench_dag_metrics(c: &mut Criterion) {
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

pub fn bench_topological_sort(c: &mut Criterion) {
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

pub fn bench_subgraph_size(c: &mut Criterion) {
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

pub fn bench_count_links_by_depth(c: &mut Criterion) {
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

pub fn bench_dag_fanout_by_level(c: &mut Criterion) {
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

pub fn bench_filter_dag(c: &mut Criterion) {
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

pub fn bench_map_dag(c: &mut Criterion) {
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

// ============================================================================
// Criterion Groups
// ============================================================================

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

criterion_group!(
    car_compression_benches,
    bench_car_compression_write,
    bench_car_compression_read,
    bench_car_compression_roundtrip,
    bench_car_compression_ratios,
);

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
