//! Distributed / shard performance tests for HNSW routing

use ipfrs_semantic::cache::HotEmbeddingCache;
use ipfrs_semantic::shard_balancer::{ShardBalancer, ShardConfig};
use std::sync::Arc;

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

// ---------------------------------------------------------------------------
// test_sharded_query_performance
//
// Simulates 10 shards:
//  - Routes 1 000 vector IDs through ShardBalancer.
//  - Verifies that each assignment is within [0, num_shards).
//  - Records load via increment_shard_load and verifies distribution
//    is reasonably balanced (no shard exceeds 3× the average).
// ---------------------------------------------------------------------------

#[test]
fn test_sharded_query_performance() {
    const NUM_SHARDS: usize = 10;
    const NUM_VECTORS: usize = 1_000;

    let config = ShardConfig {
        num_shards: NUM_SHARDS,
        replication_factor: 3,
        max_vectors_per_shard: 200_000,
    };
    let balancer = Arc::new(ShardBalancer::new(config));

    // Route all vectors and record assignment counts manually.
    let mut assignment_count = [0usize; NUM_SHARDS];

    for i in 0..NUM_VECTORS {
        let shard_id = balancer.assign_vector(i as u64);

        // Verify the assignment is within bounds.
        assert!(
            shard_id < NUM_SHARDS,
            "shard_id {shard_id} out of range [0, {NUM_SHARDS})"
        );

        assignment_count[shard_id] += 1;
        balancer.increment_shard_load(shard_id);
    }

    // Verify every vector was assigned to exactly one shard.
    let total_assigned: usize = assignment_count.iter().sum();
    assert_eq!(
        total_assigned, NUM_VECTORS,
        "expected {NUM_VECTORS} total assignments, got {total_assigned}"
    );

    // Confirm live load counters match what we incremented.
    let snapshot = balancer.shard_loads_snapshot();
    assert_eq!(snapshot.len(), NUM_SHARDS, "load snapshot length mismatch");
    for (shard, &load) in snapshot.iter().enumerate() {
        assert_eq!(
            load, assignment_count[shard],
            "shard {shard}: load counter {load} != assignment count {}",
            assignment_count[shard]
        );
    }

    // Check balance: no shard should hold more than 3× the average.
    let avg = NUM_VECTORS / NUM_SHARDS;
    let threshold = avg * 3 + 1; // +1 to handle rounding when avg is 0
    for (shard, &count) in assignment_count.iter().enumerate() {
        assert!(
            count <= threshold,
            "shard {shard} has {count} vectors, exceeds 3× average ({avg})"
        );
    }

    // Confirm least-loaded shard lookup works and returns a valid index.
    let least = balancer.least_loaded_shard();
    assert!(
        least < NUM_SHARDS,
        "least_loaded_shard() returned {least}, out of range"
    );

    // Confirm rebalance_needed returns a bool without panic.
    let _needs_rebalance = balancer.rebalance_needed();

    // Spot-check: verify a selection of vectors with known LCG vectors routes
    // deterministically (same call twice gives the same shard).
    for i in [0usize, 7, 42, 99, 512, 999] {
        let s1 = balancer.assign_vector(i as u64);
        let s2 = balancer.assign_vector(i as u64);
        assert_eq!(
            s1, s2,
            "assign_vector({i}) is not deterministic: {s1} != {s2}"
        );
    }

    // Verify vector content never changes after LCG (sanity check on helper).
    let v = lcg_vector(4, 42);
    assert_eq!(v.len(), 4);
    for &x in &v {
        assert!(x.is_finite(), "LCG vector contains non-finite value");
    }
}

// ---------------------------------------------------------------------------
// test_lookup_cache_hit_rate
//
// Fills HotEmbeddingCache with 1 000 entries, then performs repeated lookups
// for all keys and verifies the overall hit rate exceeds 99%.
// ---------------------------------------------------------------------------

#[test]
fn test_lookup_cache_hit_rate() {
    const CAPACITY: usize = 2_000; // Capacity large enough to hold all entries
    const NUM_ENTRIES: usize = 1_000;
    const REPEAT_LOOKUPS: usize = 5; // Repeated lookup passes
    const MIN_HIT_RATE: f64 = 0.99;
    const DIM: usize = 32;

    let cache = HotEmbeddingCache::new(CAPACITY);

    // Insert 1 000 distinct entries.
    let keys: Vec<String> = (0..NUM_ENTRIES)
        .map(|i| format!("vec-key-{:06}", i))
        .collect();

    for (idx, key) in keys.iter().enumerate() {
        let v = lcg_vector(DIM, idx as u64 ^ 0xABBA_1234);
        cache.insert(key.clone(), v);
    }

    // Verify all entries were actually cached (no eviction since capacity > entries).
    assert_eq!(
        cache.len(),
        NUM_ENTRIES,
        "cache should hold all {NUM_ENTRIES} entries before lookups"
    );

    // Perform repeated lookups and count hits.
    let mut hits = 0usize;
    let total_lookups = NUM_ENTRIES * REPEAT_LOOKUPS;

    for _ in 0..REPEAT_LOOKUPS {
        for key in &keys {
            if cache.get(key).is_some() {
                hits += 1;
            }
        }
    }

    let hit_rate = hits as f64 / total_lookups as f64;

    let stats = cache.stats();
    eprintln!(
        "[distributed_perf] cache hit_rate={:.4} hits={} misses={} (stats.hit_rate={:.4})",
        hit_rate, stats.hits, stats.misses, stats.hit_rate
    );

    assert!(
        hit_rate >= MIN_HIT_RATE,
        "hit rate {hit_rate:.4} is below minimum {MIN_HIT_RATE:.4}"
    );

    assert!(
        stats.hit_rate >= MIN_HIT_RATE,
        "stats.hit_rate {:.4} is below minimum {MIN_HIT_RATE:.4}",
        stats.hit_rate
    );

    // Confirm capacity is reported correctly.
    assert_eq!(
        stats.capacity, CAPACITY,
        "cache capacity mismatch: expected {CAPACITY}, got {}",
        stats.capacity
    );

    // Confirm a cache miss for an unknown key.
    let miss = cache.get("non-existent-key-xyz-999");
    assert!(
        miss.is_none(),
        "expected cache miss for unknown key, got Some"
    );
}
