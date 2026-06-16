use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ipfrs::{Node, NodeConfig, QueryFilter};
use ipfrs_core::Block;
use std::hint::black_box;
use std::path::PathBuf;
use tokio::runtime::Runtime;

fn create_test_node() -> Node {
    let path = "/tmp/ipfrs-bench-semantic";
    let _ = std::fs::remove_dir_all(path);

    let mut config = NodeConfig::default();
    config.storage.path = PathBuf::from(path);
    config.enable_semantic = true;
    config.enable_tensorlogic = false;

    Node::new(config).expect("Failed to create node")
}

fn bench_index_content(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut node = create_test_node();
    rt.block_on(node.start()).unwrap();

    let embedding_dim = 768;

    c.bench_function("semantic_index_768d", |b| {
        let mut i = 0;
        b.iter(|| {
            let data = format!("Document {}", i).into_bytes();
            let block = Block::new(data.into()).unwrap();
            let cid = *block.cid();
            rt.block_on(node.put_block(&block)).unwrap();

            let embedding: Vec<f32> = (0..embedding_dim).map(|j| (i + j) as f32 / 100.0).collect();
            rt.block_on(node.index_content(black_box(&cid), black_box(&embedding)))
                .unwrap();
            i += 1;
        });
    });

    rt.block_on(node.stop()).unwrap();
}

fn bench_search_similar(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut node = create_test_node();
    rt.block_on(node.start()).unwrap();

    let embedding_dim = 768;

    // Pre-populate with vectors
    for i in 0..1000 {
        let data = format!("Document {}", i).into_bytes();
        let block = Block::new(data.into()).unwrap();
        let cid = *block.cid();
        rt.block_on(node.put_block(&block)).unwrap();

        let embedding: Vec<f32> = (0..embedding_dim).map(|j| (i + j) as f32 / 100.0).collect();
        rt.block_on(node.index_content(&cid, &embedding)).unwrap();
    }

    let query: Vec<f32> = (0..embedding_dim).map(|i| i as f32 / 100.0).collect();

    let mut group = c.benchmark_group("semantic_search");

    for k in &[10, 50, 100] {
        group.bench_with_input(BenchmarkId::from_parameter(k), k, |b, &k| {
            b.iter(|| {
                rt.block_on(node.search_similar(black_box(&query), black_box(k)))
                    .unwrap();
            });
        });
    }

    group.finish();

    rt.block_on(node.stop()).unwrap();
}

fn bench_search_filtered(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut node = create_test_node();
    rt.block_on(node.start()).unwrap();

    let embedding_dim = 768;

    // Pre-populate with vectors
    for i in 0..1000 {
        let data = format!("Document {}", i).into_bytes();
        let block = Block::new(data.into()).unwrap();
        let cid = *block.cid();
        rt.block_on(node.put_block(&block)).unwrap();

        let embedding: Vec<f32> = (0..embedding_dim).map(|j| (i + j) as f32 / 100.0).collect();
        rt.block_on(node.index_content(&cid, &embedding)).unwrap();
    }

    let query: Vec<f32> = (0..embedding_dim).map(|i| i as f32 / 100.0).collect();
    let filter = QueryFilter {
        min_score: Some(0.5),
        max_score: None,
        max_results: Some(10),
        cid_prefix: None,
    };

    c.bench_function("semantic_search_filtered", |b| {
        b.iter(|| {
            rt.block_on(node.search_hybrid(
                black_box(&query),
                black_box(10),
                black_box(filter.clone()),
            ))
            .unwrap();
        });
    });

    rt.block_on(node.stop()).unwrap();
}

fn bench_semantic_stats(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut node = create_test_node();
    rt.block_on(node.start()).unwrap();

    let embedding_dim = 768;

    // Pre-populate with vectors
    for i in 0..1000 {
        let data = format!("Document {}", i).into_bytes();
        let block = Block::new(data.into()).unwrap();
        let cid = *block.cid();
        rt.block_on(node.put_block(&block)).unwrap();

        let embedding: Vec<f32> = (0..embedding_dim).map(|j| (i + j) as f32 / 100.0).collect();
        rt.block_on(node.index_content(&cid, &embedding)).unwrap();
    }

    c.bench_function("semantic_stats", |b| {
        b.iter(|| {
            node.semantic_stats().unwrap();
        });
    });

    rt.block_on(node.stop()).unwrap();
}

criterion_group!(
    benches,
    bench_index_content,
    bench_search_similar,
    bench_search_filtered,
    bench_semantic_stats
);
criterion_main!(benches);
