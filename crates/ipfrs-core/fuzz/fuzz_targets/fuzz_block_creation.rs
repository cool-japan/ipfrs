//! Fuzz test for Block creation
//!
//! Tests robustness of Block::new() against arbitrary data inputs

#![no_main]

use libfuzzer_sys::fuzz_target;
use ipfrs_core::Block;
use bytes::Bytes;

fuzz_target!(|data: &[u8]| {
    // Attempt to create a block from fuzzed data
    let bytes = Bytes::copy_from_slice(data);
    let _ = Block::new(bytes.clone());

    // If creation succeeds, verify invariants
    if let Ok(block) = Block::new(bytes.clone()) {
        // Block should contain the same data
        assert_eq!(block.data(), data);

        // CID should be deterministic - creating another block with same data
        // should produce the same CID
        if let Ok(block2) = Block::new(bytes.clone()) {
            assert_eq!(block.cid(), block2.cid());
        }

        // Block size should match data length
        assert_eq!(block.size(), data.len() as u64);
    }
});
