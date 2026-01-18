//! Fuzz test for data chunking
//!
//! Tests robustness of chunking operations against arbitrary data

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Test different chunk sizes
    for chunk_size in [256, 512, 1024, 4096, 16384] {
        // Chunk the data
        let chunks: Vec<&[u8]> = data.chunks(chunk_size).collect();

        // Reassemble and verify
        let reassembled: Vec<u8> = chunks.into_iter().flat_map(|c| c.iter().copied()).collect();
        assert_eq!(&reassembled[..], data);
    }

    // Test that chunking preserves data integrity
    if data.len() > 0 {
        let chunk_size = (data.len() / 3).max(1);
        let chunks: Vec<Vec<u8>> = data
            .chunks(chunk_size)
            .map(|c| c.to_vec())
            .collect();

        // Reassemble
        let reassembled: Vec<u8> = chunks.into_iter().flatten().collect();
        assert_eq!(&reassembled[..], data);
    }
});
