//! Property-based tests for ipfrs-core — Memory Pooling, Hash Engines, DAG-JOSE
//!
//! These tests use proptest to validate system invariants across
//! a wide range of randomly generated inputs.

use ipfrs_core::{
    Blake2b256Engine, Blake2b512Engine, Blake2s256Engine, Blake3Engine, BytesPool, CidBuilder,
    CidStringPool, HashEngine, Ipld, JoseBuilder, Sha256Engine,
};
use proptest::prelude::*;

// Reduce proptest cases for faster test execution
const PROPTEST_CASES: u32 = 32;

/// Generate arbitrary byte vectors for blocks (1 byte to 8KB)
fn arb_block_data() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 1..=8192)
}

// ============================================================================
// Memory Pooling Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]
    /// Property: BytesPool get and put maintains capacity
    #[test]
    fn prop_bytes_pool_capacity(size in 1024usize..=65536) {
        let pool = BytesPool::new();

        let buf = pool.get(size);
        prop_assert!(buf.capacity() >= size);

        pool.put(buf);

        // Get another buffer of similar size - should reuse
        let buf2 = pool.get(size);
        prop_assert!(buf2.capacity() >= size);
    }

    /// Property: BytesPool hit rate improves with reuse
    #[test]
    fn prop_bytes_pool_hit_rate(ops in 10usize..100) {
        let pool = BytesPool::new();
        let size = 4096;

        // Warm up
        for _ in 0..5 {
            let buf = pool.get(size);
            pool.put(buf);
        }

        let stats_before = pool.stats();

        // Perform more operations
        for _ in 0..ops {
            let buf = pool.get(size);
            pool.put(buf);
        }

        let stats_after = pool.stats();

        // Total operations should have increased
        prop_assert!(stats_after.hits + stats_after.misses > stats_before.hits + stats_before.misses);

        // After warmup, hit rate should be > 0
        prop_assert!(stats_after.hit_rate() > 0.0);
    }

    /// Property: CidStringPool deduplicates identical strings
    #[test]
    fn prop_cid_string_pool_deduplicates(
        data_items in prop::collection::vec(
            prop::collection::vec(any::<u8>(), 10..100),
            5..20
        )
    ) {
        let pool = CidStringPool::new();

        // Generate CIDs from data
        let cids: Vec<String> = data_items
            .iter()
            .map(|data| CidBuilder::new().build(data).unwrap().to_string())
            .collect();

        // Intern all CIDs
        let arcs: Vec<_> = cids.iter().map(|s| pool.intern(s)).collect();

        // Intern them again - should get same Arcs
        for (i, cid_str) in cids.iter().enumerate() {
            let arc = pool.intern(cid_str);
            prop_assert!(std::sync::Arc::ptr_eq(&arc, &arcs[i]));
        }

        // Pool size should equal number of unique CIDs
        let unique_set: std::collections::HashSet<_> = cids.iter().collect();
        prop_assert_eq!(pool.len(), unique_set.len());
    }

    /// Property: CidStringPool stats are consistent
    #[test]
    fn prop_cid_string_pool_stats(
        strings in prop::collection::vec("[a-zA-Z0-9]{10,20}", 10..50)
    ) {
        let pool = CidStringPool::new();

        // Intern each string twice
        for s in &strings {
            pool.intern(s); // First time (miss)
            pool.intern(s); // Second time (hit)
        }

        let stats = pool.stats();

        // We should have exactly strings.len() misses (one per unique string)
        // and at least strings.len() hits (one per duplicate intern)
        prop_assert!(stats.misses > 0);
        prop_assert!(stats.hits >= stats.misses);
        prop_assert_eq!(stats.hits + stats.misses, strings.len() as u64 * 2);
    }

    /// Property: Pool clear resets state
    #[test]
    fn prop_pool_clear_resets(size in 1024usize..=8192) {
        let bytes_pool = BytesPool::new();
        let cid_pool = CidStringPool::new();

        // Use the pools
        for _ in 0..10 {
            let buf = bytes_pool.get(size);
            bytes_pool.put(buf);
        }

        cid_pool.intern("test1");
        cid_pool.intern("test2");

        // Clear the pools
        bytes_pool.clear();
        cid_pool.clear();

        // CID pool should be empty
        prop_assert_eq!(cid_pool.len(), 0);
        prop_assert!(cid_pool.is_empty());

        // Next access should be a miss
        let stats_before = bytes_pool.stats();
        let _buf = bytes_pool.get(size);
        let stats_after = bytes_pool.stats();

        prop_assert_eq!(stats_after.misses, stats_before.misses + 1);
    }
}

// ============================================================================
// BLAKE3 Hash Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]
    /// Property: BLAKE3 is deterministic - same input produces same hash
    #[test]
    fn prop_blake3_deterministic(data in arb_block_data()) {
        let engine = Blake3Engine::new();
        let hash1 = engine.digest(&data);
        let hash2 = engine.digest(&data);
        prop_assert_eq!(hash1, hash2);
    }

    /// Property: BLAKE3 produces 32-byte hashes
    #[test]
    fn prop_blake3_hash_length(data in arb_block_data()) {
        let engine = Blake3Engine::new();
        let hash = engine.digest(&data);
        prop_assert_eq!(hash.len(), 32);
    }

    /// Property: BLAKE3 different inputs produce different hashes
    #[test]
    fn prop_blake3_collision_resistance(
        data1 in arb_block_data(),
        data2 in arb_block_data()
    ) {
        if data1 != data2 {
            let engine = Blake3Engine::new();
            let hash1 = engine.digest(&data1);
            let hash2 = engine.digest(&data2);
            prop_assert_ne!(hash1, hash2);
        }
    }

    /// Property: BLAKE3 vs SHA256 produce different hashes for same input
    #[test]
    fn prop_blake3_differs_from_sha256(data in arb_block_data()) {
        let blake3 = Blake3Engine::new();
        let sha256 = Sha256Engine::new();

        let blake3_hash = blake3.digest(&data);
        let sha256_hash = sha256.digest(&data);

        // Both produce 32-byte hashes
        prop_assert_eq!(blake3_hash.len(), 32);
        prop_assert_eq!(sha256_hash.len(), 32);

        // But hashes should differ (different algorithms)
        prop_assert_ne!(blake3_hash, sha256_hash);
    }

    /// Property: BLAKE3 empty input is deterministic
    #[test]
    fn prop_blake3_empty_deterministic(_data in any::<u8>()) {
        let engine = Blake3Engine::new();
        let hash1 = engine.digest(&[]);
        let hash2 = engine.digest(&[]);
        prop_assert_eq!(hash1.len(), 32);
        prop_assert_eq!(hash1, hash2);
    }

    /// Property: BLAKE3 always reports SIMD as enabled
    #[test]
    fn prop_blake3_simd_enabled(_data in any::<u8>()) {
        let engine = Blake3Engine::new();
        prop_assert!(engine.is_simd_enabled());
    }

    /// Property: BLAKE2b-256 is deterministic
    #[test]
    fn prop_blake2b256_deterministic(data in arb_block_data()) {
        let engine = Blake2b256Engine::new();
        let hash1 = engine.digest(&data);
        let hash2 = engine.digest(&data);
        prop_assert_eq!(hash1, hash2);
    }

    /// Property: BLAKE2b-256 produces 32-byte hashes
    #[test]
    fn prop_blake2b256_hash_length(data in arb_block_data()) {
        let engine = Blake2b256Engine::new();
        let hash = engine.digest(&data);
        prop_assert_eq!(hash.len(), 32);
    }

    /// Property: BLAKE2b-512 is deterministic
    #[test]
    fn prop_blake2b512_deterministic(data in arb_block_data()) {
        let engine = Blake2b512Engine::new();
        let hash1 = engine.digest(&data);
        let hash2 = engine.digest(&data);
        prop_assert_eq!(hash1, hash2);
    }

    /// Property: BLAKE2b-512 produces 64-byte hashes
    #[test]
    fn prop_blake2b512_hash_length(data in arb_block_data()) {
        let engine = Blake2b512Engine::new();
        let hash = engine.digest(&data);
        prop_assert_eq!(hash.len(), 64);
    }

    /// Property: BLAKE2s-256 is deterministic
    #[test]
    fn prop_blake2s256_deterministic(data in arb_block_data()) {
        let engine = Blake2s256Engine::new();
        let hash1 = engine.digest(&data);
        let hash2 = engine.digest(&data);
        prop_assert_eq!(hash1, hash2);
    }

    /// Property: BLAKE2s-256 produces 32-byte hashes
    #[test]
    fn prop_blake2s256_hash_length(data in arb_block_data()) {
        let engine = Blake2s256Engine::new();
        let hash = engine.digest(&data);
        prop_assert_eq!(hash.len(), 32);
    }

    /// Property: BLAKE2b-256 different inputs produce different hashes
    #[test]
    fn prop_blake2b256_collision_resistance(
        data1 in arb_block_data(),
        data2 in arb_block_data()
    ) {
        if data1 != data2 {
            let engine = Blake2b256Engine::new();
            let hash1 = engine.digest(&data1);
            let hash2 = engine.digest(&data2);
            prop_assert_ne!(hash1, hash2);
        }
    }

    /// Property: BLAKE2b vs BLAKE2s produce different hashes for same input
    #[test]
    fn prop_blake2b_differs_from_blake2s(data in arb_block_data()) {
        let blake2b = Blake2b256Engine::new();
        let blake2s = Blake2s256Engine::new();

        let blake2b_hash = blake2b.digest(&data);
        let blake2s_hash = blake2s.digest(&data);

        // Both produce 32-byte hashes
        prop_assert_eq!(blake2b_hash.len(), 32);
        prop_assert_eq!(blake2s_hash.len(), 32);

        // But hashes should differ (different algorithms)
        prop_assert_ne!(blake2b_hash, blake2s_hash);
    }

    /// Property: BLAKE2b empty input is deterministic
    #[test]
    fn prop_blake2b_empty_deterministic(_data in any::<u8>()) {
        let engine = Blake2b256Engine::new();
        let hash1 = engine.digest(&[]);
        let hash2 = engine.digest(&[]);
        prop_assert_eq!(hash1.len(), 32);
        prop_assert_eq!(hash1, hash2);
    }

    /// Property: BLAKE2 engines always report SIMD as enabled
    #[test]
    fn prop_blake2_simd_enabled(_data in any::<u8>()) {
        let blake2b256 = Blake2b256Engine::new();
        let blake2b512 = Blake2b512Engine::new();
        let blake2s = Blake2s256Engine::new();

        prop_assert!(blake2b256.is_simd_enabled());
        prop_assert!(blake2b512.is_simd_enabled());
        prop_assert!(blake2s.is_simd_enabled());
    }
}

// ============================================================================
// DAG-JOSE Property Tests
// ============================================================================

/// Generate arbitrary IPLD data for JOSE testing
fn arb_ipld_simple() -> impl Strategy<Value = Ipld> {
    prop_oneof![
        Just(Ipld::Null),
        any::<bool>().prop_map(Ipld::Bool),
        any::<i64>().prop_map(|i| Ipld::Integer(i as i128)),
        any::<f64>()
            .prop_filter("Valid float", |f| f.is_finite())
            .prop_map(Ipld::Float),
        "[a-zA-Z0-9 ]{1,100}".prop_map(Ipld::String),
        prop::collection::vec(any::<u8>(), 0..100).prop_map(Ipld::Bytes),
    ]
}

/// Generate a valid HMAC secret (32+ bytes)
fn arb_hmac_secret() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 32..=64)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]
    /// Property: JOSE signing and verification roundtrip
    #[test]
    fn prop_jose_sign_verify_roundtrip(
        ipld in arb_ipld_simple(),
        secret in arb_hmac_secret()
    ) {
        let jose = JoseBuilder::new()
            .with_payload(ipld.clone())
            .sign_hs256(&secret)
            .unwrap();

        // Should verify with correct secret
        prop_assert!(jose.verify_hs256(&secret).unwrap());

        // Should fail with different secret
        let mut wrong_secret = secret.clone();
        wrong_secret[0] = wrong_secret[0].wrapping_add(1);
        prop_assert!(!jose.verify_hs256(&wrong_secret).unwrap());

        // Payload should match
        prop_assert_eq!(jose.payload, ipld);
    }

    /// Property: JOSE signatures are deterministic
    #[test]
    fn prop_jose_deterministic(
        ipld in arb_ipld_simple(),
        secret in arb_hmac_secret()
    ) {
        let jose1 = JoseBuilder::new()
            .with_payload(ipld.clone())
            .sign_hs256(&secret)
            .unwrap();

        let jose2 = JoseBuilder::new()
            .with_payload(ipld.clone())
            .sign_hs256(&secret)
            .unwrap();

        // Same payload + secret should produce same signature
        prop_assert_eq!(jose1.signature, jose2.signature);
        prop_assert_eq!(jose1.algorithm, jose2.algorithm);
    }

    /// Property: JOSE different payloads produce different signatures
    #[test]
    fn prop_jose_different_payloads_different_sigs(
        ipld1 in arb_ipld_simple(),
        ipld2 in arb_ipld_simple(),
        secret in arb_hmac_secret()
    ) {
        if ipld1 != ipld2 {
            let jose1 = JoseBuilder::new()
                .with_payload(ipld1)
                .sign_hs256(&secret)
                .unwrap();

            let jose2 = JoseBuilder::new()
                .with_payload(ipld2)
                .sign_hs256(&secret)
                .unwrap();

            // Different payloads should produce different signatures
            prop_assert_ne!(jose1.signature, jose2.signature);
        }
    }

    /// Property: JOSE DAG-JOSE encoding roundtrip
    #[test]
    fn prop_jose_dag_jose_roundtrip(
        ipld in arb_ipld_simple(),
        secret in arb_hmac_secret()
    ) {
        let jose = JoseBuilder::new()
            .with_payload(ipld.clone())
            .sign_hs256(&secret)
            .unwrap();

        // Encode to DAG-JOSE
        let dag_jose = jose.to_dag_jose().unwrap();

        // Should be valid JSON
        let parsed: serde_json::Value = serde_json::from_slice(&dag_jose).unwrap();
        prop_assert!(parsed.get("payload").is_some());
        prop_assert!(parsed.get("signatures").is_some());

        // Decode back
        let decoded = ipfrs_core::JoseSignature::from_dag_jose(&dag_jose).unwrap();

        // Should still verify
        prop_assert!(decoded.verify_hs256(&secret).unwrap());

        // Payload should match (with special handling for floats due to JSON precision)
        match (&decoded.payload, &ipld) {
            (Ipld::Float(f1), Ipld::Float(f2)) => {
                // Allow small difference due to JSON serialization precision
                let diff = (f1 - f2).abs();
                let rel_diff = diff / f2.abs().max(1e-10);
                prop_assert!(rel_diff < 1e-10 || diff < 1e-10,
                    "Float values differ: {} vs {}, diff={}, rel_diff={}", f1, f2, diff, rel_diff);
            }
            _ => prop_assert_eq!(decoded.payload, ipld),
        }
    }

    /// Property: JOSE algorithm field is always set correctly
    #[test]
    fn prop_jose_algorithm_correct(
        ipld in arb_ipld_simple(),
        secret in arb_hmac_secret()
    ) {
        let jose = JoseBuilder::new()
            .with_payload(ipld)
            .sign_hs256(&secret)
            .unwrap();

        prop_assert_eq!(jose.algorithm, "HS256");
    }

    /// Property: JOSE signature is non-empty
    #[test]
    fn prop_jose_signature_nonempty(
        ipld in arb_ipld_simple(),
        secret in arb_hmac_secret()
    ) {
        let jose = JoseBuilder::new()
            .with_payload(ipld)
            .sign_hs256(&secret)
            .unwrap();

        prop_assert!(!jose.signature.is_empty());
        // JWT signatures are typically base64-encoded
        prop_assert!(jose.signature.len() > 50);
    }
}
