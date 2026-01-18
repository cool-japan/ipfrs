#![no_main]

use libfuzzer_sys::fuzz_target;
use ipfrs_core::{Config, ConfigBuilder, ChunkingStrategy, HashAlgorithm};

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    // Extract fuzzing parameters from input data
    let chunk_size = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let max_threads = data[4] as usize;
    let enable_parallel = data[5] & 1 == 1;
    let enable_pooling = data[6] & 1 == 1;
    let enable_validation = data[7] & 1 == 1;

    // Test config builder with fuzzy inputs
    let mut builder = ConfigBuilder::new();

    // Set chunk size (clamped to valid range)
    let chunk_size = chunk_size.max(1024).min(16 * 1024 * 1024) as usize;
    builder = builder.chunk_size(chunk_size);

    // Set chunking strategy
    let strategy = if data.len() > 8 && data[8] & 1 == 1 {
        ChunkingStrategy::ContentDefined
    } else {
        ChunkingStrategy::FixedSize
    };
    builder = builder.chunking_strategy(strategy);

    // Set hash algorithm
    let hash_algo = if data.get(9).unwrap_or(&0) & 1 == 0 {
        HashAlgorithm::Sha256
    } else {
        HashAlgorithm::Sha3_256
    };
    builder = builder.hash_algorithm(hash_algo);

    // Set other parameters
    builder = builder
        .num_threads(max_threads.max(1).min(64))
        .enable_parallel_chunking(enable_parallel)
        .enable_pooling(enable_pooling)
        .verify_blocks(enable_validation);

    // Build config - should never panic
    if let Ok(config) = builder.build() {
        // Config should be valid
        assert!(config.chunk_size >= 1024);
        assert!(config.num_threads.is_some() && config.num_threads.unwrap() >= 1);
    }

    // Test preset configs - should never panic
    let _ = Config::high_performance();
    let _ = Config::storage_optimized();
    let _ = Config::embedded();
    let _ = Config::testing();
});
