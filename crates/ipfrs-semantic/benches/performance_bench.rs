//! Performance benchmarks for semantic search
//!
//! Run with: cargo bench --bench performance_bench

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ipfrs_semantic::hnsw::{DistanceMetric, VectorIndex};
use rand::{Rng, RngExt};
use std::hint::black_box;
use std::time::Duration;

fn generate_random_vector(dim: usize, rng: &mut impl Rng) -> Vec<f32> {
    (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect()
}

fn generate_random_cid(idx: usize) -> ipfrs_core::Cid {
    use multihash_codetable::{Code, MultihashDigest};
    let data = format!("QmTest{:08x}", idx);
    let hash = Code::Sha2_256.digest(data.as_bytes());
    ipfrs_core::Cid::new_v1(0x55, hash) // 0x55 is raw codec
}

fn bench_query_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_latency");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(100);

    let dim = 768;
    let mut rng = rand::rng();

    // Test with different dataset sizes
    for size in [1_000, 10_000, 100_000].iter() {
        let mut index = VectorIndex::new(dim, DistanceMetric::L2, 16, 200).unwrap();

        // Build index
        for i in 0..*size {
            let vec = generate_random_vector(dim, &mut rng);
            let cid = generate_random_cid(i);
            index.insert(&cid, &vec).unwrap();
        }

        let query = generate_random_vector(dim, &mut rng);

        group.bench_with_input(BenchmarkId::new("knn_search", size), size, |bench, _| {
            bench.iter(|| {
                index
                    .search(black_box(&query), black_box(10), black_box(50))
                    .unwrap();
            });
        });
    }

    group.finish();
}

fn bench_index_build_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_build");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10);

    let dim = 768;
    let mut rng = rand::rng();

    for size in [1_000, 10_000].iter() {
        let vectors: Vec<Vec<f32>> = (0..*size)
            .map(|_| generate_random_vector(dim, &mut rng))
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let mut index = VectorIndex::new(dim, DistanceMetric::L2, 16, 200).unwrap();
                for (i, vec) in vectors.iter().enumerate() {
                    let cid = generate_random_cid(i);
                    index.insert(&cid, vec).unwrap();
                }
                black_box(index);
            });
        });
    }

    group.finish();
}

fn bench_concurrent_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_queries");
    group.measurement_time(Duration::from_secs(10));

    let dim = 768;
    let mut rng = rand::rng();
    let size = 10_000;

    let mut index = VectorIndex::new(dim, DistanceMetric::L2, 16, 200).unwrap();

    // Build index
    for i in 0..size {
        let vec = generate_random_vector(dim, &mut rng);
        let cid = generate_random_cid(i);
        index.insert(&cid, &vec).unwrap();
    }

    let queries: Vec<Vec<f32>> = (0..100)
        .map(|_| generate_random_vector(dim, &mut rng))
        .collect();

    // Share index across threads using Arc
    let index = std::sync::Arc::new(index);

    group.bench_function("10_threads_100_queries", |bench| {
        bench.iter(|| {
            let handles: Vec<_> = (0..10)
                .map(|thread_id| {
                    let index_clone = index.clone();
                    let queries_clone = queries.clone();
                    std::thread::spawn(move || {
                        for i in 0..10 {
                            let idx = (thread_id * 10 + i) % queries_clone.len();
                            let query = &queries_clone[idx];
                            black_box(index_clone.search(query, 10, 50).unwrap());
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }
        });
    });

    group.finish();
}

fn bench_cache_effectiveness(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_effectiveness");

    let dim = 768;
    let mut rng = rand::rng();
    let size = 10_000;

    let mut index = VectorIndex::new(dim, DistanceMetric::L2, 16, 200).unwrap();

    // Build index
    for i in 0..size {
        let vec = generate_random_vector(dim, &mut rng);
        let cid = generate_random_cid(i);
        index.insert(&cid, &vec).unwrap();
    }

    // Create a small set of queries that will be repeated
    let hot_queries: Vec<Vec<f32>> = (0..10)
        .map(|_| generate_random_vector(dim, &mut rng))
        .collect();

    group.bench_function("cold_cache", |bench| {
        bench.iter(|| {
            // Different query each time (cold cache)
            let query = generate_random_vector(dim, &mut rng);
            black_box(index.search(&query, 10, 50).unwrap());
        });
    });

    group.bench_function("hot_cache", |bench| {
        let mut idx = 0;
        bench.iter(|| {
            // Repeat queries (hot cache benefit from CPU cache)
            let query = &hot_queries[idx % hot_queries.len()];
            idx += 1;
            black_box(index.search(query, 10, 50).unwrap());
        });
    });

    group.finish();
}

fn bench_different_k_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("k_values");

    let dim = 768;
    let mut rng = rand::rng();
    let size = 10_000;

    let mut index = VectorIndex::new(dim, DistanceMetric::L2, 16, 200).unwrap();

    // Build index
    for i in 0..size {
        let vec = generate_random_vector(dim, &mut rng);
        let cid = generate_random_cid(i);
        index.insert(&cid, &vec).unwrap();
    }

    let query = generate_random_vector(dim, &mut rng);

    for k in [1, 10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(k), k, |bench, &k_val| {
            bench.iter(|| {
                index
                    .search(black_box(&query), black_box(k_val), black_box(50))
                    .unwrap();
            });
        });
    }

    group.finish();
}

fn bench_distance_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_metrics");

    let dim = 768;
    let mut rng = rand::rng();
    let size = 10_000;

    for metric in [
        DistanceMetric::L2,
        DistanceMetric::Cosine,
        DistanceMetric::DotProduct,
    ]
    .iter()
    {
        let mut index = VectorIndex::new(dim, *metric, 16, 200).unwrap();

        // Build index
        for i in 0..size {
            let vec = generate_random_vector(dim, &mut rng);
            let cid = generate_random_cid(i);
            index.insert(&cid, &vec).unwrap();
        }

        let query = generate_random_vector(dim, &mut rng);

        group.bench_with_input(
            BenchmarkId::new("search", format!("{:?}", metric)),
            metric,
            |bench, _| {
                bench.iter(|| {
                    index
                        .search(black_box(&query), black_box(10), black_box(50))
                        .unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_query_latency,
    bench_index_build_time,
    bench_concurrent_queries,
    bench_cache_effectiveness,
    bench_different_k_values,
    bench_distance_metrics
);
criterion_main!(benches);
