//! Benchmarks for block storage operations

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ipfrs::{Node, NodeConfig};
use std::hint::black_box;
use tokio::runtime::Runtime;

/// Benchmark block put operations
fn bench_block_put(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("block_put");

    // Test different block sizes
    for size in [1024, 10 * 1024, 100 * 1024, 1024 * 1024].iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.to_async(&rt).iter(|| async {
                let mut node = Node::new(NodeConfig::default()).unwrap();
                node.start().await.unwrap();

                let data = vec![0u8; size];
                let cid = black_box(node.add_bytes(data.clone()).await.unwrap());

                node.stop().await.unwrap();
                cid
            });
        });
    }
    group.finish();
}

/// Benchmark block get operations
fn bench_block_get(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("block_get");

    for size in [1024, 10 * 1024, 100 * 1024, 1024 * 1024].iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            // Setup: Add block first
            let cid = rt.block_on(async {
                let mut node = Node::new(NodeConfig::default()).unwrap();
                node.start().await.unwrap();
                let data = vec![0u8; size];
                let cid = node.add_bytes(data.clone()).await.unwrap();
                node.stop().await.unwrap();
                cid
            });

            b.to_async(&rt).iter(|| async {
                let mut node = Node::new(NodeConfig::default()).unwrap();
                node.start().await.unwrap();

                let block = black_box(node.get(&cid).await.unwrap());

                node.stop().await.unwrap();
                block
            });
        });
    }
    group.finish();
}

/// Benchmark block stat operations (metadata only)
fn bench_block_stat(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("block_stat");

    // Setup: Add some blocks
    let cids = rt.block_on(async {
        let mut node = Node::new(NodeConfig::default()).unwrap();
        node.start().await.unwrap();

        let mut cids = Vec::new();
        for size in [1024, 10 * 1024, 100 * 1024] {
            let data = vec![0u8; size];
            cids.push(node.add_bytes(data).await.unwrap());
        }

        node.stop().await.unwrap();
        cids
    });

    group.bench_function("block_stat", |b| {
        b.to_async(&rt).iter(|| async {
            let mut node = Node::new(NodeConfig::default()).unwrap();
            node.start().await.unwrap();

            for cid in &cids {
                let _stat = black_box(node.block_stat(cid).await.unwrap());
            }

            node.stop().await.unwrap();
        });
    });
    group.finish();
}

/// Benchmark batch block operations
fn bench_batch_put(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("batch_put");

    for count in [10, 50, 100].iter() {
        let total_bytes = count * 1024; // Each block is 1KB
        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), count, |b, &count| {
            b.to_async(&rt).iter(|| async {
                let mut node = Node::new(NodeConfig::default()).unwrap();
                node.start().await.unwrap();

                let mut cids = Vec::new();
                for _ in 0..count {
                    let data = vec![0u8; 1024];
                    cids.push(node.add_bytes(data.clone()).await.unwrap());
                }

                node.stop().await.unwrap();
                black_box(cids)
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_block_put,
    bench_block_get,
    bench_block_stat,
    bench_batch_put
);
criterion_main!(benches);
