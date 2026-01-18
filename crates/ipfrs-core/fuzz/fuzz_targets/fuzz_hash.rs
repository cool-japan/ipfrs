#![no_main]

use libfuzzer_sys::fuzz_target;
use ipfrs_core::hash::{
    HashEngine, Sha256Engine, Sha3_256Engine, Blake3Engine,
    Blake2b256Engine, Blake2b512Engine, Blake2s256Engine,
};
use multihash_codetable::Code;

fuzz_target!(|data: &[u8]| {
    // Test all hash engines with arbitrary data
    let engines: Vec<Box<dyn HashEngine>> = vec![
        Box::new(Sha256Engine::new()),
        Box::new(Sha3_256Engine),
        Box::new(Blake3Engine),
        Box::new(Blake2b256Engine),
        Box::new(Blake2b512Engine),
        Box::new(Blake2s256Engine),
    ];

    for engine in engines {
        // Should never panic
        let hash = engine.digest(data);

        // Verify hash length is non-zero
        assert!(!hash.is_empty());

        // Verify determinism - same input produces same output
        let hash2 = engine.digest(data);
        assert_eq!(hash, hash2);
    }

    // Test hash registry with multihash codes
    let registry = ipfrs_core::global_hash_registry();
    // Test a few common hash codes (only ones that exist in Code enum)
    let codes = [
        Code::Sha2_256,
        Code::Sha3_256,
        Code::Blake2b256,
        Code::Blake2b512,
        Code::Blake2s256,
    ];
    for code in codes {
        if let Ok(hash) = registry.digest(code, data) {
            // Hash should be deterministic
            let hash2 = registry.digest(code, data).unwrap();
            assert_eq!(hash, hash2);
        }
    }
});
