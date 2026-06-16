//! Property-based tests for ipfrs-core — Compression, Batch Compression, CAR Compression, Advanced DAG
//!
//! These tests use proptest to validate system invariants across
//! a wide range of randomly generated inputs.

use ipfrs_core::{
    collect_all_links, compress, compression_ratio, count_links_by_depth, dag_fanout_by_level,
    decompress, filter_dag, map_dag, subgraph_size, topological_sort, BatchProcessor, Block,
    CarReader, CarWriterBuilder, Cid, CidBuilder, CompressionAlgorithm, DagMetrics, Ipld,
};
use proptest::prelude::*;

// Reduce proptest cases for faster test execution
const PROPTEST_CASES: u32 = 32;

/// Generate arbitrary data for compression tests (1 byte to 10KB)
fn arb_compression_data() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 1..=10240)
}

/// Generate arbitrary compression level (0-9)
fn arb_compression_level() -> impl Strategy<Value = u8> {
    0u8..=9
}

/// Generate a random compression algorithm for testing
fn arb_compression_algorithm() -> impl Strategy<Value = CompressionAlgorithm> {
    prop_oneof![
        Just(CompressionAlgorithm::None),
        Just(CompressionAlgorithm::Zstd),
        Just(CompressionAlgorithm::Lz4),
    ]
}

/// Generate arbitrary compression data chunks for batch operations
fn arb_batch_compression_chunks() -> impl Strategy<Value = Vec<Vec<u8>>> {
    prop::collection::vec(arb_compression_data(), 1..=20)
}

/// Generate arbitrary blocks for CAR compression tests
/// Reduced from 1-10 blocks to 1-3 blocks for faster testing
fn arb_car_blocks() -> impl Strategy<Value = Vec<Vec<u8>>> {
    prop::collection::vec(prop::collection::vec(any::<u8>(), 1..=8192), 1..=3)
}

// ============================================================================
// Compression Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

    /// Property: Compression roundtrip preserves data
    #[test]
    fn prop_compression_roundtrip(
        data in arb_compression_data(),
        algorithm in prop::sample::select(vec![
            CompressionAlgorithm::None,
            CompressionAlgorithm::Zstd,
            CompressionAlgorithm::Lz4,
        ]),
        level in arb_compression_level()
    ) {
        let original = bytes::Bytes::from(data);
        let compressed = compress(&original, algorithm, level).unwrap();
        let decompressed = decompress(&compressed, algorithm).unwrap();
        prop_assert_eq!(original, decompressed);
    }

    /// Property: None algorithm produces identical output
    #[test]
    fn prop_compression_none_identity(
        data in arb_compression_data(),
        level in arb_compression_level()
    ) {
        let original = bytes::Bytes::from(data);
        let compressed = compress(&original, CompressionAlgorithm::None, level).unwrap();
        prop_assert_eq!(original, compressed);
    }

    /// Property: Compression is deterministic
    #[test]
    fn prop_compression_deterministic(
        data in arb_compression_data(),
        algorithm in prop::sample::select(vec![
            CompressionAlgorithm::None,
            CompressionAlgorithm::Zstd,
            CompressionAlgorithm::Lz4,
        ]),
        level in arb_compression_level()
    ) {
        let original = bytes::Bytes::from(data);
        let compressed1 = compress(&original, algorithm, level).unwrap();
        let compressed2 = compress(&original, algorithm, level).unwrap();
        prop_assert_eq!(compressed1, compressed2);
    }

    /// Property: Higher compression levels shouldn't produce much larger output
    #[test]
    fn prop_compression_level_difference_reasonable(
        data in arb_compression_data()
    ) {
        // Use highly compressible data (repetitive) with sufficient size
        let repetitive_data: Vec<u8> = data.iter().cycle().take(5000).copied().collect();
        let original = bytes::Bytes::from(repetitive_data);

        let compressed_low = compress(&original, CompressionAlgorithm::Zstd, 1).unwrap();
        let compressed_high = compress(&original, CompressionAlgorithm::Zstd, 9).unwrap();

        // Both should be able to compress the data
        // Higher level shouldn't be more than 10% larger than lower level
        // (some variation is okay due to different strategies)
        let ratio = compressed_high.len() as f64 / compressed_low.len() as f64;
        prop_assert!(ratio <= 1.1, "High level compression produced significantly worse results: {}", ratio);
    }

    /// Property: Compression ratio is between 0 and infinity
    #[test]
    fn prop_compression_ratio_bounds(
        data in arb_compression_data(),
        algorithm in prop::sample::select(vec![
            CompressionAlgorithm::None,
            CompressionAlgorithm::Zstd,
            CompressionAlgorithm::Lz4,
        ]),
        level in arb_compression_level()
    ) {
        let original = bytes::Bytes::from(data);
        let ratio = compression_ratio(&original, algorithm, level).unwrap();

        if algorithm == CompressionAlgorithm::None {
            prop_assert_eq!(ratio, 1.0);
        } else {
            // Ratio should be positive (may be >1 for incompressible data due to overhead)
            prop_assert!(ratio > 0.0);
        }
    }

    /// Property: Invalid compression level returns error
    #[test]
    fn prop_compression_invalid_level(
        data in arb_compression_data(),
        level in 10u8..=255
    ) {
        let original = bytes::Bytes::from(data);
        let result = compress(&original, CompressionAlgorithm::Zstd, level);
        prop_assert!(result.is_err());
    }

    /// Property: All compression algorithms support all valid levels
    #[test]
    fn prop_compression_all_levels_supported(
        data in arb_compression_data(),
        level in arb_compression_level()
    ) {
        let original = bytes::Bytes::from(data);

        for algorithm in CompressionAlgorithm::all() {
            let result = compress(&original, *algorithm, level);
            prop_assert!(result.is_ok(), "Algorithm {:?} failed at level {}", algorithm, level);
        }
    }

    /// Property: Highly repetitive data compresses well (with sufficient size)
    #[test]
    fn prop_compression_repetitive_data(
        byte in any::<u8>(),
        len in 1000usize..=5000  // Increased minimum size
    ) {
        let data = bytes::Bytes::from(vec![byte; len]);
        let compressed = compress(&data, CompressionAlgorithm::Zstd, 5).unwrap();

        // Repetitive data with sufficient size should compress to much less than 10% of original size
        prop_assert!(compressed.len() < data.len() / 10);
    }
}

// ============================================================================
// Batch Compression Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]
    /// Property: Batch compression roundtrip preserves all data
    #[test]
    fn prop_batch_compression_roundtrip(
        chunks in arb_batch_compression_chunks(),
        level in arb_compression_level()
    ) {
        let processor = BatchProcessor::new();
        let original: Vec<bytes::Bytes> = chunks.iter()
            .map(|c| bytes::Bytes::from(c.clone()))
            .collect();

        for algorithm in CompressionAlgorithm::all() {
            let compressed = processor.compress_data_parallel(
                original.clone(),
                *algorithm,
                level
            ).unwrap();

            let decompressed = processor.decompress_data_parallel(
                compressed,
                *algorithm
            ).unwrap();

            prop_assert_eq!(original.len(), decompressed.len());
            for (i, decomp) in decompressed.iter().enumerate() {
                prop_assert_eq!(&original[i], decomp);
            }
        }
    }

    /// Property: Batch compression is deterministic
    #[test]
    fn prop_batch_compression_deterministic(
        chunks in arb_batch_compression_chunks(),
        level in arb_compression_level()
    ) {
        let processor = BatchProcessor::new();
        let data: Vec<bytes::Bytes> = chunks.iter()
            .map(|c| bytes::Bytes::from(c.clone()))
            .collect();

        let compressed1 = processor.compress_data_parallel(
            data.clone(),
            CompressionAlgorithm::Zstd,
            level
        ).unwrap();

        let compressed2 = processor.compress_data_parallel(
            data,
            CompressionAlgorithm::Zstd,
            level
        ).unwrap();

        prop_assert_eq!(compressed1.len(), compressed2.len());
        for (i, comp1) in compressed1.iter().enumerate() {
            prop_assert_eq!(comp1, &compressed2[i]);
        }
    }

    /// Property: Batch compression with None algorithm preserves data unchanged
    #[test]
    fn prop_batch_compression_none_preserves(
        chunks in arb_batch_compression_chunks()
    ) {
        let processor = BatchProcessor::new();
        let original: Vec<bytes::Bytes> = chunks.iter()
            .map(|c| bytes::Bytes::from(c.clone()))
            .collect();

        let compressed = processor.compress_data_parallel(
            original.clone(),
            CompressionAlgorithm::None,
            0
        ).unwrap();

        prop_assert_eq!(original.len(), compressed.len());
        for (i, comp) in compressed.iter().enumerate() {
            prop_assert_eq!(&original[i], comp);
        }
    }

    /// Property: Batch compression ratios are non-negative and reasonable
    #[test]
    fn prop_batch_compression_ratio_bounds(
        chunks in arb_batch_compression_chunks(),
        level in arb_compression_level()
    ) {
        let processor = BatchProcessor::new();
        let data: Vec<bytes::Bytes> = chunks.iter()
            .map(|c| bytes::Bytes::from(c.clone()))
            .collect();

        let ratios = processor.analyze_compression_ratios_parallel(
            &data,
            CompressionAlgorithm::Zstd,
            level
        ).unwrap();

        prop_assert_eq!(ratios.len(), data.len());
        for ratio in ratios {
            // Ratio should be non-negative and finite
            // Note: ratio can be > 1.0 for small/incompressible data
            prop_assert!(ratio >= 0.0 && ratio.is_finite());
        }
    }

    /// Property: Empty batch returns empty results
    #[test]
    fn prop_batch_compression_empty(
        level in arb_compression_level()
    ) {
        let processor = BatchProcessor::new();
        let empty: Vec<bytes::Bytes> = vec![];

        let compressed = processor.compress_data_parallel(
            empty.clone(),
            CompressionAlgorithm::Lz4,
            level
        ).unwrap();

        prop_assert_eq!(compressed.len(), 0);

        let ratios = processor.analyze_compression_ratios_parallel(
            &empty,
            CompressionAlgorithm::Zstd,
            level
        ).unwrap();

        prop_assert_eq!(ratios.len(), 0);
    }

    /// Property: Batch compression preserves chunk count
    #[test]
    fn prop_batch_compression_preserves_count(
        chunks in arb_batch_compression_chunks(),
        level in arb_compression_level()
    ) {
        let processor = BatchProcessor::new();
        let data: Vec<bytes::Bytes> = chunks.iter()
            .map(|c| bytes::Bytes::from(c.clone()))
            .collect();

        let original_count = data.len();

        for algorithm in CompressionAlgorithm::all() {
            let compressed = processor.compress_data_parallel(
                data.clone(),
                *algorithm,
                level
            ).unwrap();

            prop_assert_eq!(compressed.len(), original_count);
        }
    }

    /// Property: Repetitive batch data compresses well
    #[test]
    fn prop_batch_compression_repetitive_efficient(
        byte in any::<u8>(),
        chunk_count in 1usize..=10,
        chunk_size in 1000usize..=2000
    ) {
        let processor = BatchProcessor::new();
        let data: Vec<bytes::Bytes> = (0..chunk_count)
            .map(|_| bytes::Bytes::from(vec![byte; chunk_size]))
            .collect();

        let ratios = processor.analyze_compression_ratios_parallel(
            &data,
            CompressionAlgorithm::Zstd,
            6
        ).unwrap();

        // Repetitive data should have good compression ratio (< 0.1)
        for ratio in ratios {
            prop_assert!(ratio < 0.1, "Expected ratio < 0.1, got {}", ratio);
        }
    }

    /// Property: Batch decompression is inverse of compression
    #[test]
    fn prop_batch_decompression_inverse(
        chunks in arb_batch_compression_chunks(),
        level in arb_compression_level()
    ) {
        let processor = BatchProcessor::new();
        let original: Vec<bytes::Bytes> = chunks.iter()
            .map(|c| bytes::Bytes::from(c.clone()))
            .collect();

        let compressed = processor.compress_data_parallel(
            original.clone(),
            CompressionAlgorithm::Lz4,
            level
        ).unwrap();

        let decompressed = processor.decompress_data_parallel(
            compressed,
            CompressionAlgorithm::Lz4
        ).unwrap();

        prop_assert_eq!(decompressed, original);
    }

    /// Property: Different compression algorithms produce different results
    #[test]
    fn prop_batch_compression_algorithms_differ(
        chunks in arb_batch_compression_chunks(),
        level in arb_compression_level()
    ) {
        // Skip if chunks are empty or too small
        if chunks.is_empty() || chunks.iter().any(|c| c.len() < 100) {
            return Ok(());
        }

        let processor = BatchProcessor::new();
        let data: Vec<bytes::Bytes> = chunks.iter()
            .map(|c| bytes::Bytes::from(c.clone()))
            .collect();

        let compressed_zstd = processor.compress_data_parallel(
            data.clone(),
            CompressionAlgorithm::Zstd,
            level
        ).unwrap();

        let compressed_lz4 = processor.compress_data_parallel(
            data.clone(),
            CompressionAlgorithm::Lz4,
            level
        ).unwrap();

        // Different algorithms should produce different compressed data
        // (at least for some chunks)
        let mut found_difference = false;
        for (i, zstd) in compressed_zstd.iter().enumerate() {
            if zstd != &compressed_lz4[i] {
                found_difference = true;
                break;
            }
        }

        prop_assert!(found_difference);
    }

    /// Property: Batch compression analysis doesn't modify data
    #[test]
    fn prop_batch_compression_analysis_no_modify(
        chunks in arb_batch_compression_chunks(),
        level in arb_compression_level()
    ) {
        let processor = BatchProcessor::new();
        let data: Vec<bytes::Bytes> = chunks.iter()
            .map(|c| bytes::Bytes::from(c.clone()))
            .collect();

        let data_clone = data.clone();

        let _ratios = processor.analyze_compression_ratios_parallel(
            &data,
            CompressionAlgorithm::Zstd,
            level
        ).unwrap();

        // Data should remain unchanged after analysis
        prop_assert_eq!(data, data_clone);
    }
}

// ============================================================================
// CAR Compression Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

    /// Property: CAR compression roundtrip preserves all blocks
    /// Updated to test ONE random algorithm per case instead of iterating through all
    #[test]
    fn prop_car_compression_roundtrip(
        block_data in arb_car_blocks(),
        level in arb_compression_level(),
        algorithm in arb_compression_algorithm()
    ) {
        use bytes::Bytes;

        // Create blocks
        let blocks: Vec<Block> = block_data.iter()
            .map(|data| Block::new(Bytes::from(data.clone())).unwrap())
            .collect();

        if blocks.is_empty() {
            return Ok(());
        }

        // Test with ONE algorithm per test case instead of iterating through all
        // Write compressed CAR
        let mut car_data = Vec::new();
        let mut writer = CarWriterBuilder::new(vec![*blocks[0].cid()])
            .with_compression(algorithm, level as i32)
            .build(&mut car_data)
            .unwrap();

        for block in &blocks {
            writer.write_block(block).unwrap();
        }
        writer.finish().unwrap();

        // Read and verify
        let mut reader = CarReader::new(&car_data[..]).unwrap();
        for (i, expected_block) in blocks.iter().enumerate() {
            let read_block = reader.read_block().unwrap()
                .unwrap_or_else(|| panic!("Expected block {} but got None", i));

            prop_assert_eq!(read_block.cid(), expected_block.cid(),
                "CID mismatch at block {}", i);
            prop_assert_eq!(read_block.data(), expected_block.data(),
                "Data mismatch at block {}", i);
        }

        // Ensure no extra blocks
        prop_assert!(reader.read_block().unwrap().is_none(),
            "Expected end of file but found more blocks");
    }

    /// Property: CAR compression is deterministic
    #[test]
    fn prop_car_compression_deterministic(
        block_data in arb_car_blocks(),
        level in arb_compression_level()
    ) {
        use bytes::Bytes;

        let blocks: Vec<Block> = block_data.iter()
            .map(|data| Block::new(Bytes::from(data.clone())).unwrap())
            .collect();

        if blocks.is_empty() {
            return Ok(());
        }

        // Write twice with same settings
        let mut car_data1 = Vec::new();
        let mut writer1 = CarWriterBuilder::new(vec![*blocks[0].cid()])
            .with_compression(CompressionAlgorithm::Zstd, level as i32)
            .build(&mut car_data1)
            .unwrap();

        for block in &blocks {
            writer1.write_block(block).unwrap();
        }
        writer1.finish().unwrap();

        let mut car_data2 = Vec::new();
        let mut writer2 = CarWriterBuilder::new(vec![*blocks[0].cid()])
            .with_compression(CompressionAlgorithm::Zstd, level as i32)
            .build(&mut car_data2)
            .unwrap();

        for block in &blocks {
            writer2.write_block(block).unwrap();
        }
        writer2.finish().unwrap();

        // Both should produce identical output
        prop_assert_eq!(car_data1, car_data2,
            "Compression should be deterministic");
    }

    /// Property: CAR compression with None algorithm preserves exact data
    #[test]
    fn prop_car_compression_none_preserves(
        block_data in arb_car_blocks()
    ) {
        use bytes::Bytes;

        let blocks: Vec<Block> = block_data.iter()
            .map(|data| Block::new(Bytes::from(data.clone())).unwrap())
            .collect();

        if blocks.is_empty() {
            return Ok(());
        }

        // Write with None compression
        let mut car_data = Vec::new();
        let mut writer = CarWriterBuilder::new(vec![*blocks[0].cid()])
            .with_compression(CompressionAlgorithm::None, 0)
            .build(&mut car_data)
            .unwrap();

        for block in &blocks {
            writer.write_block(block).unwrap();
        }

        let stats = writer.stats();
        prop_assert_eq!(stats.uncompressed_bytes, stats.compressed_bytes,
            "None algorithm should not change byte count");
        prop_assert_eq!(stats.blocks_compressed, 0,
            "None algorithm should not count as compression");

        writer.finish().unwrap();

        // Read and verify
        let mut reader = CarReader::new(&car_data[..]).unwrap();
        for expected_block in &blocks {
            let read_block = reader.read_block().unwrap().unwrap();
            prop_assert_eq!(read_block.data(), expected_block.data());
        }
    }

    /// Property: CAR compression statistics are accurate
    #[test]
    fn prop_car_compression_stats_accurate(
        block_data in arb_car_blocks(),
        level in arb_compression_level()
    ) {
        use bytes::Bytes;

        let blocks: Vec<Block> = block_data.iter()
            .map(|data| Block::new(Bytes::from(data.clone())).unwrap())
            .collect();

        if blocks.is_empty() {
            return Ok(());
        }

        let total_uncompressed: usize = blocks.iter()
            .map(|b| b.data().len())
            .sum();

        let mut car_data = Vec::new();
        let mut writer = CarWriterBuilder::new(vec![*blocks[0].cid()])
            .with_compression(CompressionAlgorithm::Zstd, level as i32)
            .build(&mut car_data)
            .unwrap();

        for block in &blocks {
            writer.write_block(block).unwrap();
        }

        let stats = writer.stats();
        prop_assert_eq!(stats.blocks_processed, blocks.len(),
            "Block count should match");
        prop_assert_eq!(stats.uncompressed_bytes, total_uncompressed,
            "Uncompressed bytes should match");
        prop_assert!(stats.compression_ratio() >= 0.0 && stats.compression_ratio() <= 10.0,
            "Compression ratio should be reasonable");
        prop_assert!(stats.bytes_saved() <= stats.uncompressed_bytes,
            "Bytes saved cannot exceed uncompressed size");

        writer.finish().unwrap();
    }

    /// Property: CAR backward compatibility - uncompressed files still work
    #[test]
    fn prop_car_backward_compat(
        block_data in arb_car_blocks()
    ) {
        use bytes::Bytes;
        use ipfrs_core::CarWriter;

        let blocks: Vec<Block> = block_data.iter()
            .map(|data| Block::new(Bytes::from(data.clone())).unwrap())
            .collect();

        if blocks.is_empty() {
            return Ok(());
        }

        // Write without compression (legacy format)
        let mut car_data = Vec::new();
        let mut writer = CarWriter::new(&mut car_data, vec![*blocks[0].cid()]).unwrap();

        for block in &blocks {
            writer.write_block(block).unwrap();
        }
        writer.finish().unwrap();

        // Read should work fine
        let mut reader = CarReader::new(&car_data[..]).unwrap();
        for expected_block in &blocks {
            let read_block = reader.read_block().unwrap().unwrap();
            prop_assert_eq!(read_block.cid(), expected_block.cid());
            prop_assert_eq!(read_block.data(), expected_block.data());
        }
    }

    /// Property: CAR compression preserves block count
    #[test]
    fn prop_car_compression_preserves_count(
        block_data in arb_car_blocks(),
        level in arb_compression_level()
    ) {
        use bytes::Bytes;

        let blocks: Vec<Block> = block_data.iter()
            .map(|data| Block::new(Bytes::from(data.clone())).unwrap())
            .collect();

        if blocks.is_empty() {
            return Ok(());
        }

        let mut car_data = Vec::new();
        let mut writer = CarWriterBuilder::new(vec![*blocks[0].cid()])
            .with_compression(CompressionAlgorithm::Lz4, level as i32)
            .build(&mut car_data)
            .unwrap();

        for block in &blocks {
            writer.write_block(block).unwrap();
        }
        writer.finish().unwrap();

        // Count blocks in output
        let mut reader = CarReader::new(&car_data[..]).unwrap();
        let mut count = 0;
        while reader.read_block().unwrap().is_some() {
            count += 1;
        }

        prop_assert_eq!(count, blocks.len(),
            "Output should have same number of blocks as input");
    }

    /// Property: CAR compression on repetitive data is efficient
    #[test]
    fn prop_car_compression_repetitive_efficient(
        byte in any::<u8>(),
        size in 1000usize..=10000,
        level in 3u8..=9
    ) {
        use bytes::Bytes;

        // Create highly repetitive block
        let data = vec![byte; size];
        let block = Block::new(Bytes::from(data)).unwrap();

        let mut car_data = Vec::new();
        let mut writer = CarWriterBuilder::new(vec![*block.cid()])
            .with_compression(CompressionAlgorithm::Zstd, level as i32)
            .build(&mut car_data)
            .unwrap();

        writer.write_block(&block).unwrap();
        let stats = writer.stats().clone();
        writer.finish().unwrap();

        // Repetitive data should compress to much less than 10% of original
        let compression_ratio = stats.compression_ratio();
        prop_assert!(compression_ratio < 0.1,
            "Repetitive data should compress well, got ratio {}", compression_ratio);

        // Verify decompression still works
        let mut reader = CarReader::new(&car_data[..]).unwrap();
        let read_block = reader.read_block().unwrap().unwrap();
        prop_assert_eq!(read_block.data().len(), size,
            "Decompressed size should match original");
    }

    /// Property: CAR compression algorithms differ in output
    #[test]
    fn prop_car_compression_algorithms_differ(
        block_data in arb_car_blocks(),
        level in arb_compression_level()
    ) {
        use bytes::Bytes;

        let blocks: Vec<Block> = block_data.iter()
            .map(|data| Block::new(Bytes::from(data.clone())).unwrap())
            .collect();

        if blocks.is_empty() || blocks.iter().all(|b| b.data().len() < 100) {
            return Ok(()); // Skip if data is too small to show algorithm differences
        }

        // Compress with Zstd
        let mut zstd_data = Vec::new();
        let mut zstd_writer = CarWriterBuilder::new(vec![*blocks[0].cid()])
            .with_compression(CompressionAlgorithm::Zstd, level as i32)
            .build(&mut zstd_data)
            .unwrap();

        for block in &blocks {
            zstd_writer.write_block(block).unwrap();
        }
        zstd_writer.finish().unwrap();

        // Compress with LZ4
        let mut lz4_data = Vec::new();
        let mut lz4_writer = CarWriterBuilder::new(vec![*blocks[0].cid()])
            .with_compression(CompressionAlgorithm::Lz4, level as i32)
            .build(&mut lz4_data)
            .unwrap();

        for block in &blocks {
            lz4_writer.write_block(block).unwrap();
        }
        lz4_writer.finish().unwrap();

        // Different algorithms should produce different output
        // (may be same for very small data, so we use "usually different")
        if zstd_data.len() > 200 && lz4_data.len() > 200 {
            prop_assert_ne!(zstd_data, lz4_data,
                "Different algorithms should produce different compressed output");
        }
    }

    /// Property: CAR read_all_blocks matches sequential reads
    #[test]
    fn prop_car_read_all_matches_sequential(
        block_data in arb_car_blocks(),
        level in arb_compression_level()
    ) {
        use bytes::Bytes;

        let blocks: Vec<Block> = block_data.iter()
            .map(|data| Block::new(Bytes::from(data.clone())).unwrap())
            .collect();

        if blocks.is_empty() {
            return Ok(());
        }

        let mut car_data = Vec::new();
        let mut writer = CarWriterBuilder::new(vec![*blocks[0].cid()])
            .with_compression(CompressionAlgorithm::Zstd, level as i32)
            .build(&mut car_data)
            .unwrap();

        for block in &blocks {
            writer.write_block(block).unwrap();
        }
        writer.finish().unwrap();

        // Read sequentially
        let mut reader1 = CarReader::new(&car_data[..]).unwrap();
        let mut sequential_blocks = Vec::new();
        while let Some(block) = reader1.read_block().unwrap() {
            sequential_blocks.push(block);
        }

        // Read all at once
        let mut reader2 = CarReader::new(&car_data[..]).unwrap();
        let all_blocks = reader2.read_all_blocks().unwrap();

        // Should match
        prop_assert_eq!(sequential_blocks.len(), all_blocks.len(),
            "Sequential and batch reads should return same count");

        for (i, (seq, batch)) in sequential_blocks.iter().zip(all_blocks.iter()).enumerate() {
            prop_assert_eq!(seq.cid(), batch.cid(),
                "Block {} CID mismatch between sequential and batch", i);
            prop_assert_eq!(seq.data(), batch.data(),
                "Block {} data mismatch between sequential and batch", i);
        }
    }
}

// ============================================================================
// Advanced DAG Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]
    /// Property: Subgraph size is always at least 1 (for the root)
    #[test]
    fn prop_dag_subgraph_size_at_least_one(
        data in prop::collection::vec(any::<u8>(), 1..=8192)
    ) {
        let ipld = Ipld::Integer(data.len() as i128);
        let size = subgraph_size(&ipld);
        prop_assert!(size >= 1, "Subgraph size should be at least 1");
    }

    /// Property: Topological sort contains all unique links
    #[test]
    fn prop_dag_topological_sort_deduplicates(
        cid_count in 1usize..=10usize
    ) {

        // Generate unique CIDs
        let cids: Vec<Cid> = (0..cid_count)
            .map(|i| {
                let data = format!("data{}", i);
                CidBuilder::new().build(data.as_bytes()).unwrap()
            })
            .collect();

        // Create IPLD with duplicate links
        let mut ipld_links = Vec::new();
        for cid in &cids {
            ipld_links.push(Ipld::link(*cid));
            ipld_links.push(Ipld::link(*cid)); // Add duplicate
        }
        let ipld = Ipld::List(ipld_links);

        let sorted = topological_sort(&ipld);

        // Should deduplicate - sorted length should equal unique CID count
        prop_assert_eq!(sorted.len(), cid_count,
            "Topological sort should deduplicate CIDs");

        // All CIDs should be present
        for cid in &cids {
            prop_assert!(sorted.contains(cid),
                "All CIDs should be in sorted result");
        }
    }

    /// Property: Filtering with always-true predicate preserves size
    #[test]
    fn prop_dag_filter_always_true(
        size in 1usize..=10usize
    ) {
        let items: Vec<Ipld> = (0..size).map(|i| Ipld::Integer(i as i128)).collect();
        let ipld = Ipld::List(items);

        let filtered = filter_dag(&ipld, &|_| true);
        prop_assert!(filtered.is_some(), "Always-true filter should preserve structure");

        if let Some(result) = filtered {
            prop_assert_eq!(subgraph_size(&result), subgraph_size(&ipld),
                "Size should be preserved with always-true filter");
        }
    }

    /// Property: Filtering with always-false predicate returns None
    #[test]
    fn prop_dag_filter_always_false(
        value in any::<i128>()
    ) {
        let ipld = Ipld::Integer(value);
        let filtered = filter_dag(&ipld, &|_| false);
        prop_assert!(filtered.is_none(),
            "Always-false filter should return None");
    }

    /// Property: Map with identity function preserves structure
    #[test]
    fn prop_dag_map_identity(
        size in 1usize..=10usize
    ) {
        let items: Vec<Ipld> = (0..size).map(|i| Ipld::Integer(i as i128)).collect();
        let ipld = Ipld::List(items);

        let mapped = map_dag(&ipld, &|node| node.clone());
        prop_assert_eq!(subgraph_size(&mapped), subgraph_size(&ipld),
            "Identity map should preserve size");
    }

    /// Property: DAG metrics values are sensible
    #[test]
    fn prop_dag_metrics_sensible(
        size in 1usize..=20usize
    ) {
        let items: Vec<Ipld> = (0..size).map(|i| Ipld::Integer(i as i128)).collect();
        let ipld = Ipld::List(items);

        let metrics = DagMetrics::from_ipld(&ipld);

        prop_assert!(metrics.avg_branching_factor >= 0.0,
            "Average branching factor should be non-negative");
        prop_assert!(metrics.max_branching_factor < 100,
            "Max branching factor should be reasonable");
        prop_assert!(metrics.width <= metrics.total_nodes,
            "Width should not exceed total nodes");
        prop_assert!(metrics.width > 0,
            "Width should be at least 1");
        prop_assert_eq!(metrics.total_nodes, subgraph_size(&ipld),
            "Total nodes should match subgraph size");
    }

    /// Property: Count links by depth returns valid counts
    #[test]
    fn prop_dag_count_links_valid(
        cid_count in 0usize..=10usize
    ) {

        // Generate CIDs
        let cids: Vec<Cid> = (0..cid_count)
            .map(|i| {
                let data = format!("data{}", i);
                CidBuilder::new().build(data.as_bytes()).unwrap()
            })
            .collect();

        // Create IPLD with these CIDs
        let ipld_links: Vec<Ipld> = cids.iter().map(|cid| Ipld::link(*cid)).collect();
        let ipld = Ipld::List(ipld_links);

        let counts = count_links_by_depth(&ipld);

        // Total count should match number of CIDs
        let total: usize = counts.iter().sum();
        prop_assert_eq!(total, cid_count,
            "Total link count should match number of CIDs");

        // All counts should be non-negative (always true for usize)
        for &count in &counts {
            prop_assert!(count < 100, "Each count should be reasonable");
        }
    }

    /// Property: DAG fanout by level returns valid values
    #[test]
    fn prop_dag_fanout_valid(
        size in 1usize..=20usize
    ) {
        let items: Vec<Ipld> = (0..size).map(|i| Ipld::Integer(i as i128)).collect();
        let ipld = Ipld::List(items);

        let fanout = dag_fanout_by_level(&ipld);

        // All fanout values should be reasonable
        for &f in &fanout {
            prop_assert!(f < 1000, "Fanout should be reasonable");
        }
    }

    /// Property: Subgraph size equals 1 + sum of children for lists
    #[test]
    fn prop_dag_subgraph_size_additive(
        item_count in 0usize..=10usize
    ) {
        // Create a list of integers
        let items: Vec<Ipld> = (0..item_count)
            .map(|i| Ipld::Integer(i as i128))
            .collect();

        let ipld = Ipld::List(items.clone());

        let total_size = subgraph_size(&ipld);
        let children_size: usize = items.iter().map(subgraph_size).sum();

        prop_assert_eq!(total_size, 1 + children_size,
            "List size should equal 1 + sum of children");
    }

    /// Property: Map dag preserves number of CID links
    #[test]
    fn prop_dag_map_preserves_links(
        cid_count in 1usize..=5usize
    ) {
        // Generate CIDs
        let cids: Vec<Cid> = (0..cid_count)
            .map(|i| {
                let data = format!("data{}", i);
                CidBuilder::new().build(data.as_bytes()).unwrap()
            })
            .collect();

        // Create IPLD with these CIDs
        let ipld_links: Vec<Ipld> = cids.iter().map(|cid| Ipld::link(*cid)).collect();
        let ipld = Ipld::List(ipld_links);

        // Transform integers only (not links)
        let mapped = map_dag(&ipld, &|node| {
            match node {
                Ipld::Integer(n) => Ipld::Integer(n * 2),
                other => other.clone(),
            }
        });

        // Links should be preserved
        let original_links = collect_all_links(&ipld);
        let mapped_links = collect_all_links(&mapped);

        prop_assert_eq!(original_links.len(), mapped_links.len(),
            "Number of links should be preserved");

        for (orig, mapped) in original_links.iter().zip(mapped_links.iter()) {
            prop_assert_eq!(orig, mapped, "Links should be identical");
        }
    }
}
