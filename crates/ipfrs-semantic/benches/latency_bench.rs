//! Latency distribution benchmarks for semantic search
//!
//! This benchmark measures query latency percentiles (P50, P90, P99)
//! Run with: cargo bench --bench latency_bench

use criterion::{criterion_group, criterion_main, Criterion};
use ipfrs_semantic::hnsw::{DistanceMetric, VectorIndex};
use rand::{Rng, RngExt};
use std::hint::black_box;
use std::time::Instant;

fn generate_random_vector(dim: usize, rng: &mut impl Rng) -> Vec<f32> {
    (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect()
}

fn generate_random_cid(idx: usize) -> ipfrs_core::Cid {
    use multihash_codetable::{Code, MultihashDigest};
    let data = format!("QmTest{:08x}", idx);
    let hash = Code::Sha2_256.digest(data.as_bytes());
    ipfrs_core::Cid::new_v1(0x55, hash)
}

fn measure_latency_percentiles(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_percentiles");

    let dim = 768;
    let mut rng = rand::rng();

    for size in [1_000, 10_000, 100_000].iter() {
        println!("\nBuilding index with {} vectors...", size);
        let mut index = VectorIndex::new(dim, DistanceMetric::L2, 16, 200).unwrap();

        // Build index
        for i in 0..*size {
            let vec = generate_random_vector(dim, &mut rng);
            let cid = generate_random_cid(i);
            index.insert(&cid, &vec).unwrap();
        }

        // Generate test queries
        let num_queries = 1000;
        let queries: Vec<Vec<f32>> = (0..num_queries)
            .map(|_| generate_random_vector(dim, &mut rng))
            .collect();

        // Measure latencies for all queries
        let mut latencies = Vec::with_capacity(num_queries);
        for query in &queries {
            let start = Instant::now();
            let _ = index.search(query, 10, 50).unwrap();
            let duration = start.elapsed();
            latencies.push(duration.as_micros() as f64);
        }

        // Sort latencies to compute percentiles
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p50_idx = (num_queries as f64 * 0.50) as usize;
        let p90_idx = (num_queries as f64 * 0.90) as usize;
        let p99_idx = (num_queries as f64 * 0.99) as usize;

        let p50 = latencies[p50_idx];
        let p90 = latencies[p90_idx];
        let p99 = latencies[p99_idx];
        let mean = latencies.iter().sum::<f64>() / num_queries as f64;

        println!("\nLatency distribution for {} vectors:", size);
        println!("  Mean: {:.2} µs", mean);
        println!("  P50:  {:.2} µs", p50);
        println!("  P90:  {:.2} µs", p90);
        println!("  P99:  {:.2} µs", p99);

        // Just use criterion for comparative benchmarking
        group.bench_function(format!("size_{}", size), |bench| {
            let mut query_idx = 0;
            bench.iter(|| {
                let query = &queries[query_idx % queries.len()];
                query_idx += 1;
                black_box(index.search(query, 10, 50).unwrap());
            });
        });
    }

    group.finish();
}

fn measure_latency_breakdown(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_breakdown");

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

    // Benchmark different ef_search values
    for ef_search in [10, 50, 100, 200].iter() {
        group.bench_function(format!("ef_search_{}", ef_search), |bench| {
            bench.iter(|| {
                black_box(index.search(&query, 10, *ef_search).unwrap());
            });
        });
    }

    group.finish();
}

fn measure_insert_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_latency");

    let dim = 768;
    let mut rng = rand::rng();

    for size in [100, 1_000, 10_000].iter() {
        let mut index = VectorIndex::new(dim, DistanceMetric::L2, 16, 200).unwrap();

        // Pre-populate index
        for i in 0..*size {
            let vec = generate_random_vector(dim, &mut rng);
            let cid = generate_random_cid(i);
            index.insert(&cid, &vec).unwrap();
        }

        // Measure insert latency at different index sizes
        group.bench_function(format!("insert_at_size_{}", size), |bench| {
            let mut i = *size;
            bench.iter(|| {
                let vec = generate_random_vector(dim, &mut rng);
                let cid = generate_random_cid(i);
                i += 1;
                index.insert(&cid, &vec).unwrap();
                black_box(());
            });
        });
    }

    group.finish();
}

fn measure_memory_footprint(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_footprint");

    let dim = 768;
    let mut rng = rand::rng();

    for size in [1_000, 10_000].iter() {
        let memory_before = get_process_memory();

        let mut index = VectorIndex::new(dim, DistanceMetric::L2, 16, 200).unwrap();

        for i in 0..*size {
            let vec = generate_random_vector(dim, &mut rng);
            let cid = generate_random_cid(i);
            index.insert(&cid, &vec).unwrap();
        }

        let memory_after = get_process_memory();
        let memory_used = memory_after - memory_before;

        // Memory per vector in bytes
        let memory_per_vector = memory_used as f64 / *size as f64;

        println!("\nMemory usage for {} vectors:", size);
        println!("  Total: {:.2} MB", memory_used as f64 / 1_048_576.0);
        println!("  Per vector: {:.2} KB", memory_per_vector / 1024.0);

        // Keep index alive for the benchmark
        group.bench_function(format!("memory_size_{}", size), |bench| {
            bench.iter(|| {
                black_box(&index);
            });
        });
    }

    group.finish();
}

#[cfg(target_os = "linux")]
fn get_process_memory() -> usize {
    use std::fs;

    // Read /proc/self/status to get VmRSS (Resident Set Size)
    if let Ok(status) = fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(kb) = parts[1].parse::<usize>() {
                        return kb * 1024; // Convert to bytes
                    }
                }
            }
        }
    }
    0
}

#[cfg(not(target_os = "linux"))]
fn get_process_memory() -> usize {
    // Fallback for non-Linux systems
    0
}

criterion_group!(
    benches,
    measure_latency_percentiles,
    measure_latency_breakdown,
    measure_insert_latency,
    measure_memory_footprint
);
criterion_main!(benches);
