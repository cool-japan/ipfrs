#![no_main]

use libfuzzer_sys::fuzz_target;
use ipfrs_core::{Block, CarReader, CarWriter};
use bytes::Bytes;

fuzz_target!(|data: &[u8]| {
    // Skip empty input
    if data.is_empty() {
        return;
    }

    // Fuzz CAR reading
    if let Ok(mut reader) = CarReader::new(data) {
        // Try to read all blocks
        let _ = reader.read_all_blocks();

        // Try sequential reading
        let _ = CarReader::new(data).and_then(|mut r| {
            while r.read_block()?.is_some() {}
            Ok(())
        });
    }

    // Fuzz CAR writing with arbitrary data split into blocks
    let chunk_size = (data.len() / 10).max(1);
    let blocks: Vec<Block> = data
        .chunks(chunk_size)
        .filter_map(|chunk| Block::new(Bytes::copy_from_slice(chunk)).ok())
        .collect();

    if !blocks.is_empty() {
        // Write with first block's CID as root
        let mut car_data = Vec::new();
        if let Ok(mut writer) = CarWriter::new(&mut car_data, vec![*blocks[0].cid()]) {
            for block in &blocks {
                let _ = writer.write_block(block);
            }
            let _ = writer.finish();

            // Try reading back what we wrote
            if !car_data.is_empty() {
                let _ = CarReader::new(&car_data[..]).and_then(|mut r| {
                    r.read_all_blocks()
                });
            }
        }

        // Write with no roots
        let mut car_data = Vec::new();
        if let Ok(mut writer) = CarWriter::new(&mut car_data, vec![]) {
            for block in &blocks {
                let _ = writer.write_block(block);
            }
            let _ = writer.finish();
        }
    }

    // Fuzz with random root CIDs
    if data.len() >= 40 {
        use ipfrs_core::CidBuilder;

        let roots: Vec<_> = data
            .chunks(data.len() / 3)
            .filter_map(|chunk| CidBuilder::new().build(chunk).ok())
            .take(5)
            .collect();

        if !roots.is_empty() && !blocks.is_empty() {
            let mut car_data = Vec::new();
            if let Ok(mut writer) = CarWriter::new(&mut car_data, roots) {
                for block in &blocks {
                    let _ = writer.write_block(block);
                }
                let _ = writer.finish();
            }
        }
    }
});
