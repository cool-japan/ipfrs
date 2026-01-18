#![no_main]

use libfuzzer_sys::fuzz_target;
use ipfrs_core::utils::*;
use bytes::Bytes;

fuzz_target!(|data: &[u8]| {
    // Test CID utilities
    if let Ok(s) = std::str::from_utf8(data) {
        // Should never panic
        let _ = parse_cid_string(s);
        let _ = validate_cid_string(s);
    }

    // Test block utilities
    if data.len() > 0 && data.len() < 100_000 {
        let bytes = Bytes::copy_from_slice(data);

        // Should never panic
        if let Ok(block1) = quick_block(&bytes) {
            if let Ok(block2) = quick_block(&bytes) {
                assert!(blocks_equal(&block1, &block2));
            }
            let _ = verify_block(&block1);
            let _ = inspect_block(&block1);
        }

        // Test CID generation
        let _ = sha256_cid(data);
        let _ = sha3_cid(data);
    }

    // Test size formatting
    if data.len() >= 8 {
        let size = u64::from_le_bytes([
            data[0], data[1], data[2], data[3],
            data[4], data[5], data[6], data[7],
        ]);
        // Should never panic
        let _ = format_size(size);
    }

    // Test chunking estimation
    if data.len() >= 4 {
        let size = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as u64;

        // Should never panic
        let _ = estimate_chunks(size);
        let _ = needs_chunking(size);
    }

    // Test IPLD utilities with small data
    if data.len() > 0 && data.len() < 1000 {
        let list = ipld_list(vec![
            ipfrs_core::Ipld::Bytes(data.to_vec()),
        ]);

        // Test encoding roundtrip
        if let Ok(cbor) = ipld_to_cbor(&list) {
            let _ = ipld_from_cbor(&cbor);
        }

        if let Ok(json) = ipld_to_json(&list) {
            let _ = ipld_from_json(&json);
        }
    }

    // Test block validation
    if data.len() >= 2 {
        let count = (data[0] % 10) as usize;
        let mut blocks = Vec::new();

        for _ in 0..count {
            if let Ok(block) = quick_block(&Bytes::from_static(b"test")) {
                blocks.push(block);
            }
        }

        // Should never panic
        let _ = validate_blocks(&blocks);
        let _ = find_invalid_blocks(&blocks);
        let _ = count_unique_blocks(&blocks);
        let _ = deduplication_ratio(&blocks);
        let _ = total_blocks_size(&blocks);
    }
});
