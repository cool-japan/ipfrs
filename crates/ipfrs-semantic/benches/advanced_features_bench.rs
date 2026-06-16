use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ipfrs_core::Cid;
use ipfrs_semantic::{
    analyze_optimization, analyze_quality, compute_batch_stats, detect_anomaly, diagnose_index,
    find_outliers, HealthMonitor, OptimizationGoal, QueryOptimizer, SearchProfiler, VectorIndex,
};
use multihash_codetable::{Code, MultihashDigest};
use std::hint::black_box;
use std::time::Duration;

/// Benchmark vector quality analysis for different vector sizes
fn bench_vector_quality(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_quality");

    for dim in [128, 384, 768, 1024].iter() {
        let vector: Vec<f32> = (0..*dim).map(|i| (i as f32) * 0.01).collect();

        group.bench_with_input(BenchmarkId::new("analyze_quality", dim), dim, |b, _| {
            b.iter(|| {
                let quality = analyze_quality(black_box(&vector));
                black_box(quality);
            });
        });

        group.bench_with_input(BenchmarkId::new("detect_anomaly", dim), dim, |b, _| {
            b.iter(|| {
                let report = detect_anomaly(black_box(&vector), 0.5, 0.3, 1.0, 0.1, 0.1, 0.2);
                black_box(report);
            });
        });
    }

    group.finish();
}

/// Benchmark batch quality analysis
fn bench_batch_quality(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_quality");

    for num_vectors in [10, 100, 1000].iter() {
        let vectors: Vec<Vec<f32>> = (0..*num_vectors)
            .map(|i| {
                (0..768)
                    .map(|j| ((i + j) as f32) * 0.001)
                    .collect::<Vec<f32>>()
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("compute_batch_stats", num_vectors),
            num_vectors,
            |b, _| {
                b.iter(|| {
                    let stats = compute_batch_stats(black_box(&vectors));
                    black_box(stats);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("find_outliers", num_vectors),
            num_vectors,
            |b, _| {
                b.iter(|| {
                    let outliers = find_outliers(black_box(&vectors), 2.0);
                    black_box(outliers);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark index diagnostics
fn bench_diagnostics(c: &mut Criterion) {
    let mut group = c.benchmark_group("diagnostics");

    for size in [100, 1000, 10000].iter() {
        let mut index = VectorIndex::with_defaults(768).unwrap();

        // Populate index
        for i in 0..*size {
            let vector: Vec<f32> = (0..768).map(|j| ((i + j) as f32) * 0.001).collect();
            let data = format!("test-vector-{}", i);
            let hash = Code::Sha2_256.digest(data.as_bytes());
            let cid = Cid::new_v1(0x55, hash);
            index.insert(&cid, &vector).unwrap();
        }

        group.bench_with_input(BenchmarkId::new("diagnose_index", size), size, |b, _| {
            b.iter(|| {
                let report = diagnose_index(black_box(&index));
                black_box(report);
            });
        });
    }

    group.finish();
}

/// Benchmark health monitoring
fn bench_health_monitor(c: &mut Criterion) {
    let mut group = c.benchmark_group("health_monitor");

    let mut index = VectorIndex::with_defaults(768).unwrap();
    for i in 0..1000 {
        let vector: Vec<f32> = (0..768).map(|j| ((i + j) as f32) * 0.001).collect();
        let data = format!("test-vector-{}", i);
        let hash = Code::Sha2_256.digest(data.as_bytes());
        let cid = Cid::new_v1(0x55, hash);
        index.insert(&cid, &vector).unwrap();
    }

    group.bench_function("health_monitor_check", |b| {
        let mut monitor = HealthMonitor::new(Duration::from_secs(60));
        b.iter(|| {
            let report = monitor.check(black_box(&index));
            black_box(report);
        });
    });

    group.bench_function("search_profiler_record", |b| {
        let mut profiler = SearchProfiler::new();
        b.iter(|| {
            profiler.record_query(Duration::from_millis(5));
        });
    });

    group.bench_function("search_profiler_stats", |b| {
        let mut profiler = SearchProfiler::new();
        for _ in 0..100 {
            profiler.record_query(Duration::from_millis(5));
        }
        b.iter(|| {
            let stats = profiler.stats();
            black_box(stats);
        });
    });

    group.finish();
}

/// Benchmark optimization analysis
fn bench_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization");

    for size in [1000, 10_000, 100_000].iter() {
        group.bench_with_input(
            BenchmarkId::new("analyze_optimization", size),
            size,
            |b, &s| {
                b.iter(|| {
                    let result = analyze_optimization(
                        black_box(s),
                        black_box(768),
                        black_box(16),
                        black_box(200),
                        black_box(OptimizationGoal::Balanced),
                    );
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark query optimizer
fn bench_query_optimizer(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_optimizer");

    group.bench_function("query_optimizer_record", |b| {
        let mut optimizer = QueryOptimizer::new(50, Duration::from_millis(10));
        b.iter(|| {
            optimizer.record_query(Duration::from_millis(15));
        });
    });

    group.bench_function("query_optimizer_get_ef", |b| {
        let mut optimizer = QueryOptimizer::new(50, Duration::from_millis(10));
        for _ in 0..100 {
            optimizer.record_query(Duration::from_millis(15));
        }
        b.iter(|| {
            let ef = optimizer.get_ef_search();
            black_box(ef);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_vector_quality,
    bench_batch_quality,
    bench_diagnostics,
    bench_health_monitor,
    bench_optimization,
    bench_query_optimizer
);
criterion_main!(benches);
