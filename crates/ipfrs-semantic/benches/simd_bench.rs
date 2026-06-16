//! Benchmarks for SIMD distance computation
//!
//! Run with: cargo bench --bench simd_bench

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ipfrs_semantic::simd;
use std::hint::black_box;

fn bench_l2_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_distance");

    for size in [64, 128, 256, 384, 512, 768, 1024, 1536, 2048].iter() {
        let a: Vec<f32> = (0..*size).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..*size).map(|i| (i as f32 + 1.0) * 0.1).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                simd::l2_distance(black_box(&a), black_box(&b));
            });
        });
    }

    group.finish();
}

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");

    for size in [64, 128, 256, 384, 512, 768, 1024, 1536, 2048].iter() {
        let a: Vec<f32> = (0..*size).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..*size).map(|i| (i as f32 + 1.0) * 0.1).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                simd::dot_product(black_box(&a), black_box(&b));
            });
        });
    }

    group.finish();
}

fn bench_cosine_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_distance");

    for size in [64, 128, 256, 384, 512, 768, 1024, 1536, 2048].iter() {
        let a: Vec<f32> = (0..*size).map(|i| (i as f32 * 0.1) + 1.0).collect();
        let b: Vec<f32> = (0..*size).map(|i| ((i as f32 + 1.0) * 0.1) + 1.0).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                simd::cosine_distance(black_box(&a), black_box(&b));
            });
        });
    }

    group.finish();
}

fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");

    let query = vec![1.0; 768];
    let database: Vec<Vec<f32>> = (0..1000)
        .map(|i| (0..768).map(|j| ((i + j) as f32) * 0.001).collect())
        .collect();

    group.bench_function("l2_1000x768", |bench| {
        bench.iter(|| {
            for vec in &database {
                black_box(simd::l2_distance(&query, vec));
            }
        });
    });

    group.bench_function("dot_1000x768", |bench| {
        bench.iter(|| {
            for vec in &database {
                black_box(simd::dot_product(&query, vec));
            }
        });
    });

    group.bench_function("cosine_1000x768", |bench| {
        bench.iter(|| {
            for vec in &database {
                black_box(simd::cosine_distance(&query, vec));
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_l2_distance,
    bench_dot_product,
    bench_cosine_distance,
    bench_batch_operations
);
criterion_main!(benches);
