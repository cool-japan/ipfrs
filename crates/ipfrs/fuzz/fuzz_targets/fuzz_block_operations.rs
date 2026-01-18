#![no_main]

use libfuzzer_sys::fuzz_target;
use ipfrs::Block;
use bytes::Bytes;

fuzz_target!(|data: &[u8]| {
    // Don't panic on empty data
    if data.is_empty() {
        return;
    }

    // Try to create a block from arbitrary data
    if let Ok(block) = Block::new(Bytes::from(data.to_vec())) {
        // Verify CID can be obtained
        let _cid = block.cid();

        // Verify data can be retrieved
        let retrieved_data = block.data();
        assert_eq!(retrieved_data, data);

        // Verify size matches
        assert_eq!(block.data().len(), data.len());
    }
});
