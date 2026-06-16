//! HNSW comprehensive benchmark suite
//!
//! Run with: cargo bench --bench hnsw_bench -p ipfrs-semantic

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ipfrs_semantic::hnsw::{DistanceMetric, VectorIndex};
use ipfrs_semantic::{RouterConfig, SemanticRouter};
use std::hint::black_box;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Simple LCG-based random vector generator — no external rand crate needed.
fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut v = Vec::with_capacity(dim);
    let mut x = seed;
    for _ in 0..dim {
        x = x
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        v.push(((x >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0);
    }
    v
}

fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
    v.iter_mut().for_each(|x| *x /= norm);
}

fn make_cid(idx: usize) -> ipfrs_core::Cid {
    use multihash_codetable::{Code, MultihashDigest};
    let data = format!("hnsw-bench-{:010}", idx);
    let hash = Code::Sha2_256.digest(data.as_bytes());
    ipfrs_core::Cid::new_v1(0x55, hash)
}

// ---------------------------------------------------------------------------
// bench_hnsw_add — measure insertion throughput at various scales
// ---------------------------------------------------------------------------

fn bench_hnsw_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_add");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(10);

    const DIM: usize = 128;

    for n in [100_u64, 1_000, 10_000] {
        // Pre-generate vectors so generation cost is not part of the benchmark.
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let mut v = random_vector(DIM, i.wrapping_mul(31) ^ 0xDEAD_BEEF);
                normalize(&mut v);
                v
            })
            .collect();

        group.throughput(Throughput::Elements(n));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                let mut index = VectorIndex::new(DIM, DistanceMetric::Cosine, 16, 200)
                    .expect("VectorIndex::new");
                for (i, vec) in vectors.iter().enumerate().take(n as usize) {
                    let cid = make_cid(i);
                    index.insert(&cid, vec).expect("insert");
                }
                black_box(index.len());
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// bench_hnsw_search — single-query latency on a pre-built 10 000-vector index
// ---------------------------------------------------------------------------

fn bench_hnsw_search(c: &mut Criterion) {
    const DIM: usize = 128;
    const N: usize = 10_000;
    const K: usize = 10;
    const EF: usize = 50;

    // Build the index once outside the benchmark loop.
    let mut index =
        VectorIndex::new(DIM, DistanceMetric::Cosine, 16, 200).expect("VectorIndex::new");
    for i in 0..N {
        let mut v = random_vector(DIM, i as u64 ^ 0x1234_5678);
        normalize(&mut v);
        index.insert(&make_cid(i), &v).expect("insert");
    }

    let mut query = random_vector(DIM, 0xABCD_EF01);
    normalize(&mut query);

    let mut group = c.benchmark_group("hnsw_search");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(200);
    group.throughput(Throughput::Elements(1));

    group.bench_function("single_query_k10_ef50", |b| {
        b.iter(|| {
            index
                .search(black_box(&query), black_box(K), black_box(EF))
                .expect("search")
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// bench_hnsw_search_k — vary k from 1..=100 on the same pre-built index
// ---------------------------------------------------------------------------

fn bench_hnsw_search_k(c: &mut Criterion) {
    const DIM: usize = 128;
    const N: usize = 10_000;
    const EF: usize = 50;

    let mut index =
        VectorIndex::new(DIM, DistanceMetric::Cosine, 16, 200).expect("VectorIndex::new");
    for i in 0..N {
        let mut v = random_vector(DIM, (i as u64).wrapping_mul(1_000_003) ^ 0xFADE_CAFE);
        normalize(&mut v);
        index.insert(&make_cid(i), &v).expect("insert");
    }

    let mut query = random_vector(DIM, 0x0102_0304_0506_0708);
    normalize(&mut query);

    let mut group = c.benchmark_group("hnsw_search_k");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(200);
    group.throughput(Throughput::Elements(1));

    for k in [1_usize, 5, 10, 50, 100] {
        group.bench_with_input(BenchmarkId::new("k", k), &k, |b, &k| {
            b.iter(|| {
                index
                    .search(black_box(&query), black_box(k), black_box(EF))
                    .expect("search")
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// bench_router_add_batch — compare batch add vs. sequential add on SemanticRouter
// ---------------------------------------------------------------------------

fn bench_router_add_batch(c: &mut Criterion) {
    const DIM: usize = 128;
    const N: usize = 1_000;

    // Pre-build (cid, vector) pairs.
    let items: Vec<(ipfrs_core::Cid, Vec<f32>)> = (0..N)
        .map(|i| {
            let mut v = random_vector(DIM, (i as u64).wrapping_mul(7_919) ^ 0xBEEF_CAFE);
            normalize(&mut v);
            (make_cid(i), v)
        })
        .collect();

    let config = RouterConfig {
        dimension: DIM,
        ..RouterConfig::low_latency(DIM)
    };

    let mut group = c.benchmark_group("router_add");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(10);
    group.throughput(Throughput::Elements(N as u64));

    // Batch add
    group.bench_function("batch_1000", |b| {
        b.iter(|| {
            let router = SemanticRouter::new(config.clone()).expect("SemanticRouter::new");
            router.add_batch(black_box(&items)).expect("add_batch");
            black_box(&router);
        });
    });

    // Sequential add (one by one)
    group.bench_function("sequential_1000", |b| {
        b.iter(|| {
            let router = SemanticRouter::new(config.clone()).expect("SemanticRouter::new");
            for (cid, vec) in &items {
                router.add(black_box(cid), black_box(vec)).expect("add");
            }
            black_box(&router);
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// criterion wiring
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_hnsw_add,
    bench_hnsw_search,
    bench_hnsw_search_k,
    bench_router_add_batch
);
criterion_main!(benches);
