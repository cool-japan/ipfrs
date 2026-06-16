//! Scale tests for HNSW — skipped in normal CI, run explicitly with:
//!   cargo test -p ipfrs-semantic -- --ignored

use ipfrs_semantic::hnsw::{DistanceMetric, VectorIndex};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn lcg_vector(dim: usize, seed: u64) -> Vec<f32> {
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
    let data = format!("scale-test-{:012}", idx);
    let hash = Code::Sha2_256.digest(data.as_bytes());
    ipfrs_core::Cid::new_v1(0x55, hash)
}

// ---------------------------------------------------------------------------
// 1 M vector insertion + query latency test
// ---------------------------------------------------------------------------

/// Inserts 1 000 000 random 128-d vectors in batches of 10 000,
/// measures total insert time, then runs 100 queries and checks
/// p99 query latency.
///
/// Skipped in normal CI — run with:
///   cargo test -p ipfrs-semantic test_scale_1m_vectors -- --ignored --nocapture
#[test]
#[ignore]
fn test_scale_1m_vectors() {
    const DIM: usize = 128;
    const TOTAL: usize = 1_000_000;
    const BATCH: usize = 10_000;
    const K: usize = 10;
    const EF: usize = 50;
    const MAX_SECONDS: u64 = 300;
    const QUERY_COUNT: usize = 100;
    const P99_LIMIT_MS: u128 = 100;

    let mut index =
        VectorIndex::new(DIM, DistanceMetric::Cosine, 16, 200).expect("VectorIndex::new failed");

    let insert_start = Instant::now();

    for batch_id in 0..(TOTAL / BATCH) {
        for i in 0..BATCH {
            let global_idx = batch_id * BATCH + i;
            let mut v = lcg_vector(DIM, global_idx as u64 ^ 0xCAFE_BABE);
            normalize(&mut v);
            let cid = make_cid(global_idx);
            index.insert(&cid, &v).unwrap_or_else(|e| {
                panic!("insert failed at index {global_idx}: {e}");
            });
        }
        let elapsed = insert_start.elapsed().as_secs();
        eprintln!(
            "[scale_test] batch {} / {} done — {:.0}s elapsed, {:.0} vec/s",
            batch_id + 1,
            TOTAL / BATCH,
            elapsed,
            ((batch_id + 1) * BATCH) as f64 / elapsed.max(1) as f64
        );
    }

    let total_insert_secs = insert_start.elapsed().as_secs();
    let vecs_per_sec = TOTAL as f64 / total_insert_secs.max(1) as f64;
    eprintln!(
        "[scale_test] insert complete: {}s, {:.0} vec/s",
        total_insert_secs, vecs_per_sec
    );

    assert!(
        total_insert_secs <= MAX_SECONDS,
        "Insert of {TOTAL} vectors took {total_insert_secs}s, limit is {MAX_SECONDS}s"
    );

    // ---- query latency ----
    let mut latencies_ms: Vec<u128> = Vec::with_capacity(QUERY_COUNT);

    for q in 0..QUERY_COUNT {
        let mut query = lcg_vector(DIM, q as u64 ^ 0x1234_ABCD);
        normalize(&mut query);

        let t0 = Instant::now();
        let results = index
            .search(&query, K, EF)
            .unwrap_or_else(|e| panic!("search failed: {e}"));
        let elapsed_ms = t0.elapsed().as_millis();

        latencies_ms.push(elapsed_ms);
        assert!(
            !results.is_empty(),
            "query {q} returned no results — index may be empty"
        );
    }

    latencies_ms.sort_unstable();
    let p99_idx = ((QUERY_COUNT as f64 * 0.99) as usize).min(QUERY_COUNT - 1);
    let p99_ms = latencies_ms[p99_idx];
    let mean_ms: f64 = latencies_ms.iter().map(|&x| x as f64).sum::<f64>() / QUERY_COUNT as f64;

    eprintln!(
        "[scale_test] query latency: mean={:.2}ms, p99={}ms (limit={}ms)",
        mean_ms, p99_ms, P99_LIMIT_MS
    );

    assert!(
        p99_ms <= P99_LIMIT_MS,
        "p99 query latency {p99_ms}ms exceeds {P99_LIMIT_MS}ms limit"
    );
}
