//! Benchmarks for semantic search operations

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ipfrs::{Node, NodeConfig, QueryFilter};
use ipfrs_semantic::{DistanceMetric, RouterConfig};
use std::hint::black_box;
use tokio::runtime::Runtime;

/// Generate a random embedding vector
fn random_embedding(dim: usize, seed: u64) -> Vec<f32> {
    // Simple deterministic pseudo-random generation for benchmarking
    (0..dim)
        .map(|i| {
            let x = ((seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(i as u64))
                >> 32) as f32;
            x / u32::MAX as f32
        })
        .collect()
}

/// Benchmark semantic indexing operations
fn bench_semantic_index(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("semantic_index");

    let dim = 768; // Standard embedding dimension
    for count in [10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(count), count, |b, &count| {
            b.to_async(&rt).iter(|| async {
                let config = NodeConfig::default().with_semantic(RouterConfig {
                    dimension: dim,
                    metric: DistanceMetric::Cosine,
                    max_connections: 16,
                    ef_construction: 200,
                    ef_search: 50,
                    cache_size: 1000,
                    ..RouterConfig::default()
                });

                let mut node = Node::new(config).unwrap();
                node.start().await.unwrap();

                // Index multiple blocks
                for i in 0..count {
                    let data = format!("block_{}", i).into_bytes();
                    let cid = node.add_bytes(data).await.unwrap();
                    let embedding = random_embedding(dim, i as u64);
                    node.index_content(&cid, &embedding).await.unwrap();
                }

                node.stop().await.unwrap();
                black_box(count)
            });
        });
    }
    group.finish();
}

/// Benchmark semantic search queries
fn bench_semantic_search(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("semantic_search");

    let dim = 768;
    let index_size = 100;

    // Setup: Create indexed node
    let setup_node = || async {
        let config = NodeConfig::default().with_semantic(RouterConfig {
            dimension: dim,
            metric: DistanceMetric::Cosine,
            max_connections: 16,
            ef_construction: 200,
            ef_search: 50,
            cache_size: 1000,
            ..RouterConfig::default()
        });

        let mut node = Node::new(config).unwrap();
        node.start().await.unwrap();

        // Index blocks
        for i in 0..index_size {
            let data = format!("block_{}", i).into_bytes();
            let cid = node.add_bytes(data).await.unwrap();
            let embedding = random_embedding(dim, i as u64);
            node.index_content(&cid, &embedding).await.unwrap();
        }

        node
    };

    // Benchmark different k values
    for k in [1, 5, 10, 20].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(k), k, |b, &k| {
            let mut node = rt.block_on(setup_node());

            b.to_async(&rt).iter(|| async {
                let query = random_embedding(dim, 999);
                let results = black_box(node.search_similar(&query, k).await.unwrap());
                results
            });

            rt.block_on(async {
                node.stop().await.unwrap();
            });
        });
    }
    group.finish();
}

/// Benchmark filtered semantic search
fn bench_filtered_search(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("filtered_search");

    let dim = 768;
    let index_size = 100;

    // Setup: Create indexed node
    let config = NodeConfig::default().with_semantic(RouterConfig {
        dimension: dim,
        metric: DistanceMetric::Cosine,
        max_connections: 16,
        ef_construction: 200,
        ef_search: 50,
        cache_size: 1000,
        ..RouterConfig::default()
    });

    let mut node = rt.block_on(async {
        let mut node = Node::new(config).unwrap();
        node.start().await.unwrap();

        for i in 0..index_size {
            let data = format!("block_{}", i).into_bytes();
            let cid = node.add_bytes(data).await.unwrap();
            let embedding = random_embedding(dim, i as u64);
            node.index_content(&cid, &embedding).await.unwrap();
        }

        node
    });

    group.bench_function("filtered_search", |b| {
        b.to_async(&rt).iter(|| async {
            let query = random_embedding(dim, 999);
            let filter = QueryFilter {
                min_score: Some(0.5),
                max_score: None,
                max_results: Some(10),
                cid_prefix: None,
            };

            let results = black_box(node.search_hybrid(&query, 20, filter).await.unwrap());
            results
        });
    });

    rt.block_on(async {
        node.stop().await.unwrap();
    });

    group.finish();
}

/// Benchmark semantic statistics
fn bench_semantic_stats(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("semantic_stats");

    let dim = 768;

    // Setup with different index sizes
    for size in [10, 50, 100].iter() {
        let config = NodeConfig::default().with_semantic(RouterConfig {
            dimension: dim,
            metric: DistanceMetric::Cosine,
            max_connections: 16,
            ef_construction: 200,
            ef_search: 50,
            cache_size: 1000,
            ..RouterConfig::default()
        });

        let mut node = rt.block_on(async {
            let mut node = Node::new(config).unwrap();
            node.start().await.unwrap();

            for i in 0..*size {
                let data = format!("block_{}", i).into_bytes();
                let cid = node.add_bytes(data).await.unwrap();
                let embedding = random_embedding(dim, i as u64);
                node.index_content(&cid, &embedding).await.unwrap();
            }

            node
        });

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _size| {
            b.to_async(&rt)
                .iter(|| async { black_box(node.semantic_stats().unwrap()) });
        });

        rt.block_on(async {
            node.stop().await.unwrap();
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_semantic_index,
    bench_semantic_search,
    bench_filtered_search,
    bench_semantic_stats
);
criterion_main!(benches);
