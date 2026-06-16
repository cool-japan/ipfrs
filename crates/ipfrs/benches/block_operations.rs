use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ipfrs::{Node, NodeConfig};
use ipfrs_core::Block;
use std::hint::black_box;
use std::path::PathBuf;
use tokio::runtime::Runtime;

fn create_test_node() -> Node {
    let path = "/tmp/ipfrs-bench-blocks";
    let _ = std::fs::remove_dir_all(path);

    let mut config = NodeConfig::default();
    config.storage.path = PathBuf::from(path);
    config.enable_semantic = false;
    config.enable_tensorlogic = false;

    Node::new(config).expect("Failed to create node")
}

fn bench_put_block(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut node = create_test_node();
    rt.block_on(node.start()).unwrap();

    c.bench_function("block_put_1kb", |b| {
        b.iter(|| {
            let data = vec![0u8; 1024];
            let block = Block::new(data.into()).unwrap();
            rt.block_on(node.put_block(black_box(&block))).unwrap();
        });
    });

    rt.block_on(node.stop()).unwrap();
}

fn bench_get_block(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut node = create_test_node();
    rt.block_on(node.start()).unwrap();

    // Pre-populate with blocks
    let data = vec![0u8; 1024];
    let block = Block::new(data.into()).unwrap();
    let cid = *block.cid();
    rt.block_on(node.put_block(&block)).unwrap();

    c.bench_function("block_get_1kb", |b| {
        b.iter(|| {
            rt.block_on(node.get_block(black_box(&cid))).unwrap();
        });
    });

    rt.block_on(node.stop()).unwrap();
}

fn bench_has_block(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut node = create_test_node();
    rt.block_on(node.start()).unwrap();

    // Pre-populate with blocks
    let data = vec![0u8; 1024];
    let block = Block::new(data.into()).unwrap();
    let cid = *block.cid();
    rt.block_on(node.put_block(&block)).unwrap();

    c.bench_function("block_has", |b| {
        b.iter(|| {
            rt.block_on(node.has_block(black_box(&cid))).unwrap();
        });
    });

    rt.block_on(node.stop()).unwrap();
}

fn bench_batch_put_blocks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("block_batch_put");

    for size in &[10, 100, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let mut node = create_test_node();
                rt.block_on(node.start()).unwrap();

                for i in 0..size {
                    let data = format!("Block {}", i).into_bytes();
                    let block = Block::new(data.into()).unwrap();
                    rt.block_on(node.put_block(black_box(&block))).unwrap();
                }

                rt.block_on(node.stop()).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_storage_stats(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut node = create_test_node();
    rt.block_on(node.start()).unwrap();

    // Pre-populate with blocks
    for i in 0..100 {
        let data = format!("Block {}", i).into_bytes();
        let block = Block::new(data.into()).unwrap();
        rt.block_on(node.put_block(&block)).unwrap();
    }

    c.bench_function("storage_stats", |b| {
        b.iter(|| {
            node.storage_stats().unwrap();
        });
    });

    rt.block_on(node.stop()).unwrap();
}

criterion_group!(
    benches,
    bench_put_block,
    bench_get_block,
    bench_has_block,
    bench_batch_put_blocks,
    bench_storage_stats
);
criterion_main!(benches);
