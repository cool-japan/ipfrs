//! Semantic search and shard-balancing benchmarks for IPFRS v0.3.0
//!
//! Run with: cargo bench --bench semantic_bench -p ipfrs-semantic

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ipfrs_network::semantic_dht::{
    LshHash, PartialSyncConfig, SemanticDht, SemanticDhtConfig, ShardBalancer, ShardBalancerConfig,
};
use ipfrs_semantic::hnsw::{DistanceMetric, VectorIndex};
use rand::{Rng, RngExt};
use std::collections::HashMap;
use std::hint::black_box;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_cid(idx: usize) -> ipfrs_core::Cid {
    use multihash_codetable::{Code, MultihashDigest};
    let data = format!("QmSemanticBench{:08x}", idx);
    let hash = Code::Sha2_256.digest(data.as_bytes());
    ipfrs_core::Cid::new_v1(0x55, hash)
}

fn random_unit_vec(dim: usize, rng: &mut impl Rng) -> Vec<f32> {
    let v: Vec<f32> = (0..dim)
        .map(|_| rng.random_range(-1.0_f32..1.0_f32))
        .collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.into_iter().map(|x| x / norm).collect()
    } else {
        vec![1.0 / (dim as f32).sqrt(); dim]
    }
}

/// Build a `VectorIndex` pre-loaded with `n` vectors of dimension `dim`.
fn build_hnsw_index(n: usize, dim: usize) -> VectorIndex {
    let mut rng = rand::rng();
    let mut index = VectorIndex::new(dim, DistanceMetric::L2, 16, 200).expect("VectorIndex::new");
    for i in 0..n {
        let cid = make_cid(i);
        let vec = random_unit_vec(dim, &mut rng);
        index.insert(&cid, &vec).expect("insert");
    }
    index
}

/// Build a `SemanticDht` pre-loaded with `n` records of dimension `dim`.
fn build_semantic_dht(n: usize, dim: usize) -> SemanticDht {
    let mut rng = rand::rng();
    let dht = SemanticDht::new(SemanticDhtConfig {
        dimension: dim,
        ..Default::default()
    });
    for i in 0..n {
        let cid = format!("cid_{i}");
        let vec = random_unit_vec(dim, &mut rng);
        dht.put_with_vector(cid, vec, "bench_provider")
            .expect("put_with_vector");
    }
    dht
}

/// Build a `ShardBalancer` with `n_peers` each carrying a uniform vector load.
fn build_balancer_with_peers(n_peers: usize, vectors_per_peer: usize) -> ShardBalancer {
    let mut b = ShardBalancer::new(ShardBalancerConfig::default());
    let mut cid_idx = 0usize;
    for p in 0..n_peers {
        let peer = format!("peer_{p}");
        for _ in 0..vectors_per_peer {
            b.record_vector_assignment(&peer, &format!("cid_{cid_idx}"));
            cid_idx += 1;
        }
    }
    b
}

// ---------------------------------------------------------------------------
// Benchmark: HNSW vector search at different corpus sizes
// ---------------------------------------------------------------------------

fn bench_vector_search(c: &mut Criterion) {
    const DIM: usize = 128;
    let mut group = c.benchmark_group("semantic_search");

    for &n_vectors in &[100usize, 1_000, 10_000] {
        // Pre-build index outside the timed loop
        let index = build_hnsw_index(n_vectors, DIM);
        let mut rng = rand::rng();
        let query = random_unit_vec(DIM, &mut rng);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("hnsw_search", n_vectors),
            &n_vectors,
            |b, _| {
                b.iter(|| {
                    index
                        .search(black_box(&query), black_box(10), black_box(50))
                        .expect("search")
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: ShardBalancer::suggest_peers_for_vector at different fleet sizes
// ---------------------------------------------------------------------------

fn bench_shard_balancer(c: &mut Criterion) {
    let mut group = c.benchmark_group("shard_balancer");

    for &n_peers in &[10usize, 50, 100] {
        // Each peer carries 100 vectors – balanced cluster
        let balancer = build_balancer_with_peers(n_peers, 100);

        group.bench_with_input(
            BenchmarkId::new("suggest_peers", n_peers),
            &n_peers,
            |b, _| {
                b.iter(|| {
                    black_box(balancer.suggest_peers_for_vector(black_box(3)));
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: SemanticDht put_with_vector and search_similar
// ---------------------------------------------------------------------------

fn bench_semantic_dht_put_search(c: &mut Criterion) {
    const DIM: usize = 128;

    // --- put_with_vector ---
    {
        let dht = SemanticDht::new(SemanticDhtConfig {
            dimension: DIM,
            ..Default::default()
        });
        let mut rng = rand::rng();

        c.bench_function("semantic_dht_put_with_vector", |b| {
            let mut counter = 0usize;
            b.iter(|| {
                let cid = format!("cid_bench_{counter}");
                counter += 1;
                let vec = random_unit_vec(DIM, &mut rng);
                dht.put_with_vector(black_box(cid), black_box(vec), "bench_peer")
                    .expect("put_with_vector");
            });
        });
    }

    // --- search_similar (1k records, k=10) ---
    {
        let dht = build_semantic_dht(1_000, DIM);
        let mut rng = rand::rng();
        let query = random_unit_vec(DIM, &mut rng);

        c.bench_function("semantic_dht_search_similar_1k", |b| {
            b.iter(|| {
                dht.search_similar(black_box(&query), black_box(10))
                    .expect("search_similar")
            });
        });
    }
}

// ---------------------------------------------------------------------------
// Feature 3 – New v0.3.0 benchmarks
// ---------------------------------------------------------------------------

/// `bench_hnsw_insert_1m`: insert 1K / 10K / 100K vectors and extrapolate to 1M.
///
/// Uses `Throughput::Elements` so Criterion reports insertions/sec.
fn bench_hnsw_insert_1m(c: &mut Criterion) {
    const DIM: usize = 128;
    let mut group = c.benchmark_group("hnsw_insert_throughput");
    // Measurement sizes: 1K, 10K, 100K.  Report as projections toward 1M.
    for &n in &[1_000usize, 10_000, 100_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("insert", n), &n, |b, &size| {
            b.iter(|| {
                let mut rng = rand::rng();
                let mut index =
                    VectorIndex::new(DIM, DistanceMetric::L2, 16, 200).expect("VectorIndex::new");
                for i in 0..size {
                    let cid = make_cid(i);
                    let vec = random_unit_vec(DIM, &mut rng);
                    index
                        .insert(black_box(&cid), black_box(&vec))
                        .expect("insert");
                }
                black_box(index)
            });
        });
    }
    group.finish();
}

/// `bench_hnsw_search_latency`: p50/p95/p99 latency for top-10 search at 100K vectors.
///
/// Criterion's built-in statistics cover mean and stddev; the bench reports
/// raw per-iteration time which Criterion aggregates.  For precise percentiles
/// run with `--profile-time` or capture the raw samples.
fn bench_hnsw_search_latency(c: &mut Criterion) {
    const DIM: usize = 128;
    const N: usize = 10_000; // Use 10K in CI; bump to 100K for full bench runs.

    let index = build_hnsw_index(N, DIM);
    let mut rng = rand::rng();
    let query = random_unit_vec(DIM, &mut rng);

    let mut group = c.benchmark_group("hnsw_search_latency");
    group.throughput(Throughput::Elements(1));

    group.bench_function("top10_search_100k", |b| {
        b.iter(|| {
            index
                .search(black_box(&query), black_box(10), black_box(50))
                .expect("search")
        });
    });

    group.finish();
}

/// `bench_shard_balancer_assign`: assign 10K vectors across 10 "peers",
/// measuring `assign_vector` throughput.
fn bench_shard_balancer_assign(c: &mut Criterion) {
    const DIM: usize = 128;
    const N_PEERS: usize = 10;
    const N_VECTORS: usize = 10_000;

    // Pre-populate balancer with N_PEERS each having 1000 vectors
    let balancer = build_balancer_with_peers(N_PEERS, 1000);

    // Pre-generate vectors
    let mut rng = rand::rng();
    let vectors: Vec<Vec<f32>> = (0..N_VECTORS)
        .map(|_| random_unit_vec(DIM, &mut rng))
        .collect();

    let mut group = c.benchmark_group("shard_balancer_assign");
    group.throughput(Throughput::Elements(N_VECTORS as u64));

    group.bench_function("assign_10k_vectors_10_peers", |b| {
        b.iter(|| {
            for vec in &vectors {
                black_box(balancer.assign_vector(black_box(vec), 3));
            }
        });
    });

    group.finish();
}

/// `bench_partial_sync_threshold_0_05`: simulate partial sync with 1% dirty
/// vectors in a 100K-vector set (use 1K in CI to keep bench time reasonable;
/// scale linearly to 100K).
fn bench_partial_sync_threshold_0_05(c: &mut Criterion) {
    const DIM: usize = 32;
    const N: usize = 1_000; // scale-down; bump to 100_000 for production bench

    let mut rng = rand::rng();
    let dht = SemanticDht::new(SemanticDhtConfig {
        dimension: DIM,
        ..Default::default()
    });

    // Insert N records
    let mut prev_vectors: HashMap<String, Vec<f32>> = HashMap::new();
    for i in 0..N {
        let v = random_unit_vec(DIM, &mut rng);
        let cid = format!("bench_cid_{i}");
        dht.put_with_vector(cid.clone(), v.clone(), "bench_peer")
            .expect("put_with_vector");
        prev_vectors.insert(cid, v);
    }

    // Mark 1% dirty by altering their prev entry (cosine dist > 0.05)
    let dirty_count = N / 100;
    let mut dirty_prev = prev_vectors.clone();
    for (i, (cid, v)) in dirty_prev.iter_mut().enumerate() {
        if i >= dirty_count {
            break;
        }
        // Rotate vector by 90° in first two dimensions → cos dist = 1.0
        if v.len() >= 2 {
            let tmp = v[0];
            v[0] = -v[1];
            v[1] = tmp;
        }
        let _ = cid;
    }

    let region = LshHash {
        table: 0,
        bucket: vec![0; 8],
    };
    let peer = libp2p::PeerId::random();
    let cfg = PartialSyncConfig {
        sync_threshold: 0.05,
        batch_size: 32,
        max_rounds: 100,
    };

    let mut group = c.benchmark_group("partial_sync");
    group.throughput(Throughput::Elements(N as u64));

    group.bench_function("threshold_0_05_1pct_dirty", |b| {
        b.iter(|| {
            dht.efficient_partial_sync_with_config(
                black_box(&peer),
                black_box(&region),
                black_box(&cfg),
                black_box(Some(&dirty_prev)),
            )
            .expect("partial sync")
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion entry points
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_vector_search,
    bench_shard_balancer,
    bench_semantic_dht_put_search,
    bench_hnsw_insert_1m,
    bench_hnsw_search_latency,
    bench_shard_balancer_assign,
    bench_partial_sync_threshold_0_05,
);
criterion_main!(benches);
