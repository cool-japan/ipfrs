//! Batch query benchmarks
//!
//! Benchmarks for batch query performance comparing single vs batch queries

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ipfrs_core::Cid;
use ipfrs_semantic::router::{RouterConfig, SemanticRouter};
use multihash_codetable::{Code, MultihashDigest};
use rand::RngExt;
use std::hint::black_box;

fn generate_test_data(dimension: usize, count: usize) -> (SemanticRouter, Vec<Vec<f32>>) {
    let router = SemanticRouter::new(RouterConfig {
        dimension,
        ..Default::default()
    })
    .expect("Failed to create router");

    let mut rng = rand::rng();

    // Index embeddings
    for i in 0..count {
        let data = format!("batch_bench_{}", i);
        let hash = Code::Sha2_256.digest(data.as_bytes());
        let cid = Cid::new_v1(0x55, hash);

        let embedding: Vec<f32> = (0..dimension)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();

        router.add(&cid, &embedding).expect("Failed to add");
    }

    // Generate query embeddings
    let queries: Vec<Vec<f32>> = (0..100)
        .map(|_| {
            (0..dimension)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect()
        })
        .collect();

    (router, queries)
}

fn bench_single_vs_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_single_vs_batch");

    let dimension = 128;
    let index_size = 1000;
    let (router, queries) = generate_test_data(dimension, index_size);

    // Benchmark single queries executed sequentially
    group.throughput(Throughput::Elements(10));
    group.bench_function("single_queries_sequential", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        b.iter(|| {
            rt.block_on(async {
                for query in queries.iter().take(10) {
                    let _ = black_box(router.query(query, 5).await.unwrap());
                }
            })
        });
    });

    // Benchmark batch queries (parallelized)
    group.throughput(Throughput::Elements(10));
    group.bench_function("batch_query_parallel", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        b.iter(|| {
            rt.block_on(async {
                let _ = black_box(router.query_batch(&queries[..10], 5).await.unwrap());
            })
        });
    });

    group.finish();
}

fn bench_batch_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_size_scaling");

    let dimension = 128;
    let index_size = 1000;
    let (router, queries) = generate_test_data(dimension, index_size);

    for batch_size in [1, 10, 50, 100].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &size| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                b.iter(|| {
                    rt.block_on(async {
                        let _ = black_box(router.query_batch(&queries[..size], 5).await.unwrap());
                    })
                });
            },
        );
    }

    group.finish();
}

fn bench_batch_with_filter(c: &mut Criterion) {
    use ipfrs_semantic::router::QueryFilter;

    let mut group = c.benchmark_group("batch_with_filter");

    let dimension = 128;
    let index_size = 1000;
    let (router, queries) = generate_test_data(dimension, index_size);

    let filter = QueryFilter {
        min_score: Some(0.5),
        max_results: Some(10),
        ..Default::default()
    };

    group.throughput(Throughput::Elements(10));
    group.bench_function("batch_no_filter", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        b.iter(|| {
            rt.block_on(async {
                let _ = black_box(router.query_batch(&queries[..10], 5).await.unwrap());
            })
        });
    });

    group.bench_function("batch_with_filter", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        b.iter(|| {
            rt.block_on(async {
                let _ = black_box(
                    router
                        .query_batch_with_filter(&queries[..10], 5, filter.clone())
                        .await
                        .unwrap(),
                );
            })
        });
    });

    group.finish();
}

fn bench_batch_different_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_different_k");

    let dimension = 128;
    let index_size = 1000;
    let (router, queries) = generate_test_data(dimension, index_size);

    for k in [1, 5, 10, 50].iter() {
        group.throughput(Throughput::Elements(10));
        group.bench_with_input(BenchmarkId::from_parameter(k), k, |b, &k_val| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            b.iter(|| {
                rt.block_on(async {
                    let _ = black_box(router.query_batch(&queries[..10], k_val).await.unwrap());
                })
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_vs_batch,
    bench_batch_sizes,
    bench_batch_with_filter,
    bench_batch_different_k
);
criterion_main!(benches);
