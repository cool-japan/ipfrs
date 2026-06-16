//! Property-based tests for ipfrs-core — Batch Processing, Codec Registry, CAR Format
//!
//! These tests use proptest to validate system invariants across
//! a wide range of randomly generated inputs.

use ipfrs_core::{
    BatchProcessor, Block, CarReader, CarWriter, Cid, CidBuilder, CodecRegistry, HashAlgorithm,
    Ipld, Sha256Engine,
};
use proptest::prelude::*;

// Reduce proptest cases for faster test execution
const PROPTEST_CASES: u32 = 32;

/// Generate arbitrary IPLD data for JOSE/codec testing (simple values)
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

/// Generate arbitrary byte vectors for blocks (1 byte to 8KB)
fn arb_block_data() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 1..=8192)
}

/// Generate a list of blocks for CAR testing
fn arb_blocks() -> impl Strategy<Value = Vec<Vec<u8>>> {
    prop::collection::vec(prop::collection::vec(any::<u8>(), 1..=4096), 1..=20)
}

/// Generate a list of CIDs for roots
fn arb_root_cids() -> impl Strategy<Value = Vec<Cid>> {
    prop::collection::vec(
        prop::collection::vec(any::<u8>(), 1..=256)
            .prop_map(|data| CidBuilder::new().build(&data).unwrap()),
        0..=5,
    )
}

// ============================================================================
// Batch Processing Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]
    /// Property: Parallel block creation produces same results as sequential
    #[test]
    fn prop_batch_parallel_equals_sequential(
        chunks in prop::collection::vec(
            prop::collection::vec(any::<u8>(), 1..=1000),
            1..=100
        )
    ) {
        use bytes::Bytes;

        let processor = BatchProcessor::new();

        // Convert to Bytes
        let bytes_chunks: Vec<Bytes> = chunks.iter()
            .map(|v| Bytes::from(v.clone()))
            .collect();

        // Parallel creation
        let parallel_blocks = processor.create_blocks_parallel(bytes_chunks.clone()).unwrap();

        // Sequential creation
        let sequential_blocks: Vec<_> = bytes_chunks.iter()
            .map(|data| Block::new(data.clone()).unwrap())
            .collect();

        // Compare CIDs
        prop_assert_eq!(parallel_blocks.len(), sequential_blocks.len());
        for (par, seq) in parallel_blocks.iter().zip(sequential_blocks.iter()) {
            prop_assert_eq!(par.cid(), seq.cid());
            prop_assert_eq!(par.data(), seq.data());
        }
    }

    /// Property: Parallel CID generation is deterministic
    #[test]
    fn prop_batch_cid_generation_deterministic(
        chunks in prop::collection::vec(
            prop::collection::vec(any::<u8>(), 10..=500),
            1..=50
        )
    ) {
        use bytes::Bytes;

        let processor = BatchProcessor::new();
        let bytes_chunks: Vec<Bytes> = chunks.iter()
            .map(|v| Bytes::from(v.clone()))
            .collect();

        let result1 = processor.generate_cids_parallel(bytes_chunks.clone()).unwrap();
        let result2 = processor.generate_cids_parallel(bytes_chunks).unwrap();

        prop_assert_eq!(result1.len(), result2.len());
        for ((data1, cid1), (data2, cid2)) in result1.iter().zip(result2.iter()) {
            prop_assert_eq!(data1, data2);
            prop_assert_eq!(cid1, cid2);
        }
    }

    /// Property: All blocks created in parallel are valid
    #[test]
    fn prop_batch_all_blocks_valid(
        chunks in prop::collection::vec(
            prop::collection::vec(any::<u8>(), 1..=1000),
            1..=50
        )
    ) {
        use bytes::Bytes;

        let processor = BatchProcessor::new();
        let bytes_chunks: Vec<Bytes> = chunks.iter()
            .map(|v| Bytes::from(v.clone()))
            .collect();

        let blocks = processor.create_blocks_parallel(bytes_chunks).unwrap();

        // All blocks should verify successfully
        prop_assert!(processor.verify_blocks_parallel(&blocks).is_ok());

        // Each individual block should also be valid
        for block in &blocks {
            prop_assert!(block.verify().unwrap());
        }
    }

    /// Property: Parallel hash computation matches sequential
    #[test]
    fn prop_batch_hashing_matches_sequential(
        data_chunks in prop::collection::vec(
            prop::collection::vec(any::<u8>(), 10..=500),
            1..=50
        )
    ) {
        let processor = BatchProcessor::new();
        let engine = Sha256Engine::new();

        use ipfrs_core::HashEngine;
        let data_refs: Vec<&[u8]> = data_chunks.iter()
            .map(|v| v.as_slice())
            .collect();

        let parallel_hashes = processor.compute_hashes_parallel(&data_refs).unwrap();

        // Sequential hashing
        let sequential_hashes: Vec<Vec<u8>> = data_chunks.iter()
            .map(|data| engine.digest(data))
            .collect();

        prop_assert_eq!(parallel_hashes, sequential_hashes);
    }

    /// Property: Different hash algorithms produce different CIDs
    #[test]
    fn prop_batch_different_algorithms_different_cids(
        data in prop::collection::vec(any::<u8>(), 100..=500)
    ) {
        use bytes::Bytes;

        let data_bytes = Bytes::from(data);
        let chunks = vec![data_bytes.clone()];

        let processor_sha256 = BatchProcessor::with_hash_algorithm(HashAlgorithm::Sha256);
        let processor_sha3 = BatchProcessor::with_hash_algorithm(HashAlgorithm::Sha3_256);

        let blocks_sha256 = processor_sha256.create_blocks_parallel(chunks.clone()).unwrap();
        let blocks_sha3 = processor_sha3.create_blocks_parallel(chunks).unwrap();

        prop_assert_ne!(blocks_sha256[0].cid(), blocks_sha3[0].cid());
    }

    /// Property: Total bytes calculation is accurate
    #[test]
    fn prop_batch_total_bytes_accurate(
        chunks in prop::collection::vec(
            prop::collection::vec(any::<u8>(), 1..=1000),
            1..=50
        )
    ) {
        use bytes::Bytes;

        let processor = BatchProcessor::new();
        let bytes_chunks: Vec<Bytes> = chunks.iter()
            .map(|v| Bytes::from(v.clone()))
            .collect();

        let expected_total: usize = bytes_chunks.iter()
            .map(|b| b.len())
            .sum();

        let blocks = processor.create_blocks_parallel(bytes_chunks).unwrap();
        let actual_total = processor.total_bytes_parallel(&blocks);

        prop_assert_eq!(expected_total, actual_total);
    }

    /// Property: Unique CID collection works correctly
    #[test]
    fn prop_batch_unique_cids_correct(
        data in prop::collection::vec(any::<u8>(), 100..=200)
    ) {
        use bytes::Bytes;
        use std::collections::HashSet;

        let processor = BatchProcessor::new();

        // Create some duplicate chunks
        let chunks = vec![
            Bytes::from(data.clone()),
            Bytes::from(data.clone()), // duplicate
            Bytes::from(vec![1, 2, 3]),
            Bytes::from(vec![1, 2, 3]), // duplicate
        ];

        let blocks = processor.create_blocks_parallel(chunks).unwrap();
        let unique_cids = processor.unique_cids_parallel(&blocks);

        // Should have exactly 2 unique CIDs
        prop_assert_eq!(unique_cids.len(), 2);

        // Verify uniqueness using a HashSet
        let unique_set: HashSet<_> = unique_cids.iter()
            .map(|cid| cid.to_string())
            .collect();
        prop_assert_eq!(unique_set.len(), unique_cids.len());
    }

    /// Property: Empty batch handling (always succeeds)
    #[test]
    fn prop_batch_empty_input_ok(_dummy in 0..1u8) {
        use bytes::Bytes;

        let processor = BatchProcessor::new();
        let empty: Vec<Bytes> = vec![];

        let blocks = processor.create_blocks_parallel(empty.clone()).unwrap();
        prop_assert_eq!(blocks.len(), 0);

        let cids = processor.generate_cids_parallel(empty).unwrap();
        prop_assert_eq!(cids.len(), 0);

        // Empty verification should succeed
        prop_assert!(processor.verify_blocks_parallel(&[]).is_ok());
    }
}

// ============================================================================
// Codec Registry Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]
    /// Property: Codec encode/decode roundtrip preserves data (DAG-CBOR)
    #[test]
    fn prop_codec_cbor_roundtrip(ipld in arb_ipld_simple()) {
        use ipfrs_core::codec;

        let registry = CodecRegistry::new();
        let encoded = registry.encode(codec::DAG_CBOR, &ipld).unwrap();
        let decoded = registry.decode(codec::DAG_CBOR, &encoded).unwrap();
        prop_assert_eq!(ipld, decoded);
    }

    /// Property: Codec encode/decode roundtrip preserves data (DAG-JSON)
    /// Note: Skips Float due to JSON serialization precision limits
    #[test]
    fn prop_codec_json_roundtrip(ipld in arb_ipld_simple()) {
        use ipfrs_core::codec;

        // Skip floats due to JSON precision issues
        if matches!(ipld, Ipld::Float(_)) {
            return Ok(());
        }

        let registry = CodecRegistry::new();
        let encoded = registry.encode(codec::DAG_JSON, &ipld).unwrap();
        let decoded = registry.decode(codec::DAG_JSON, &encoded).unwrap();
        prop_assert_eq!(ipld, decoded);
    }

    /// Property: RAW codec roundtrip for bytes
    #[test]
    fn prop_codec_raw_roundtrip(bytes in prop::collection::vec(any::<u8>(), 0..1000)) {
        use ipfrs_core::codec;

        let registry = CodecRegistry::new();
        let ipld = Ipld::Bytes(bytes.clone());
        let encoded = registry.encode(codec::RAW, &ipld).unwrap();
        let decoded = registry.decode(codec::RAW, &encoded).unwrap();

        match decoded {
            Ipld::Bytes(decoded_bytes) => prop_assert_eq!(bytes, decoded_bytes),
            _ => prop_assert!(false, "Expected Ipld::Bytes"),
        }
    }

    /// Property: All registered codecs can be retrieved
    #[test]
    fn prop_codec_registry_list_complete(_dummy in 0..1u8) {
        use ipfrs_core::codec;

        let registry = CodecRegistry::new();
        let codecs = registry.list_codecs();

        // Default codecs should all be present
        prop_assert!(codecs.contains(&codec::RAW));
        prop_assert!(codecs.contains(&codec::DAG_CBOR));
        prop_assert!(codecs.contains(&codec::DAG_JSON));
        prop_assert_eq!(codecs.len(), 3);
    }

    /// Property: Codec has_codec is consistent with get
    #[test]
    fn prop_codec_has_get_consistent(code in 0x50u64..0x100) {
        let registry = CodecRegistry::new();
        let has = registry.has_codec(code);
        let get = registry.get(code);

        prop_assert_eq!(has, get.is_some());
    }

    /// Property: Codec names are non-empty for registered codecs
    #[test]
    fn prop_codec_names_nonempty(_dummy in 0..1u8) {
        use ipfrs_core::codec;

        let registry = CodecRegistry::new();

        let name_raw = registry.get_name(codec::RAW).unwrap();
        let name_cbor = registry.get_name(codec::DAG_CBOR).unwrap();
        let name_json = registry.get_name(codec::DAG_JSON).unwrap();

        prop_assert!(!name_raw.is_empty());
        prop_assert!(!name_cbor.is_empty());
        prop_assert!(!name_json.is_empty());
    }

    /// Property: Encoding same data twice produces same result
    #[test]
    fn prop_codec_encoding_deterministic(ipld in arb_ipld_simple()) {
        use ipfrs_core::codec;

        let registry = CodecRegistry::new();
        let encoded1 = registry.encode(codec::DAG_CBOR, &ipld).unwrap();
        let encoded2 = registry.encode(codec::DAG_CBOR, &ipld).unwrap();
        prop_assert_eq!(encoded1, encoded2);
    }

    /// Property: Different codecs produce different encodings (for most data)
    #[test]
    fn prop_codec_different_codecs_different_encoding(
        s in "[a-zA-Z0-9]{10,20}"
    ) {
        use ipfrs_core::codec;

        let registry = CodecRegistry::new();
        let ipld = Ipld::String(s);

        let cbor = registry.encode(codec::DAG_CBOR, &ipld).unwrap();
        let json = registry.encode(codec::DAG_JSON, &ipld).unwrap();

        // CBOR and JSON encodings should differ
        prop_assert_ne!(cbor, json);
    }
}

// ============================================================================
// CAR Format Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]
    /// Property: CAR write/read roundtrip preserves all blocks
    #[test]
    fn prop_car_roundtrip_preserves_blocks(
        block_data in arb_blocks()
    ) {
        // Create blocks
        let blocks: Vec<Block> = block_data
            .iter()
            .map(|data| Block::new(data.clone().into()).unwrap())
            .collect();

        if blocks.is_empty() {
            return Ok(());
        }

        // Write to CAR
        let mut car_data = Vec::new();
        let roots = vec![*blocks[0].cid()];
        let mut writer = CarWriter::new(&mut car_data, roots).unwrap();

        for block in &blocks {
            writer.write_block(block).unwrap();
        }
        writer.finish().unwrap();

        // Read from CAR
        let mut reader = CarReader::new(&car_data[..]).unwrap();
        let read_blocks = reader.read_all_blocks().unwrap();

        // Verify all blocks match
        prop_assert_eq!(read_blocks.len(), blocks.len());

        for (original, read) in blocks.iter().zip(read_blocks.iter()) {
            prop_assert_eq!(original.cid(), read.cid());
            prop_assert_eq!(original.data(), read.data());
        }
    }

    /// Property: CAR roots are preserved in roundtrip
    #[test]
    fn prop_car_roots_preserved(
        roots in arb_root_cids(),
        block_data in arb_block_data()
    ) {
        let block = Block::new(block_data.into()).unwrap();

        // Write CAR with multiple roots
        let mut car_data = Vec::new();
        let mut writer = CarWriter::new(&mut car_data, roots.clone()).unwrap();
        writer.write_block(&block).unwrap();
        writer.finish().unwrap();

        // Read and verify roots
        let reader = CarReader::new(&car_data[..]).unwrap();
        let read_roots = reader.roots();

        prop_assert_eq!(read_roots.len(), roots.len());
        for (original, read) in roots.iter().zip(read_roots.iter()) {
            prop_assert_eq!(original, read);
        }
    }

    /// Property: CAR encoding is deterministic
    #[test]
    fn prop_car_encoding_deterministic(
        block_data in arb_blocks()
    ) {
        if block_data.is_empty() {
            return Ok(());
        }

        let blocks: Vec<Block> = block_data
            .iter()
            .map(|data| Block::new(data.clone().into()).unwrap())
            .collect();

        let roots = vec![*blocks[0].cid()];

        // Encode twice
        let mut car_data1 = Vec::new();
        let mut writer1 = CarWriter::new(&mut car_data1, roots.clone()).unwrap();
        for block in &blocks {
            writer1.write_block(block).unwrap();
        }
        writer1.finish().unwrap();

        let mut car_data2 = Vec::new();
        let mut writer2 = CarWriter::new(&mut car_data2, roots).unwrap();
        for block in &blocks {
            writer2.write_block(block).unwrap();
        }
        writer2.finish().unwrap();

        // Encodings should be identical
        prop_assert_eq!(car_data1, car_data2);
    }

    /// Property: Empty roots list is valid
    #[test]
    fn prop_car_empty_roots_valid(
        block_data in arb_block_data()
    ) {
        let block = Block::new(block_data.into()).unwrap();

        let mut car_data = Vec::new();
        let mut writer = CarWriter::new(&mut car_data, vec![]).unwrap();
        writer.write_block(&block).unwrap();
        writer.finish().unwrap();

        let reader = CarReader::new(&car_data[..]).unwrap();
        prop_assert_eq!(reader.roots().len(), 0);
    }

    /// Property: Block order is preserved in CAR format
    #[test]
    fn prop_car_preserves_block_order(
        block_data in arb_blocks()
    ) {
        if block_data.is_empty() {
            return Ok(());
        }

        let blocks: Vec<Block> = block_data
            .iter()
            .map(|data| Block::new(data.clone().into()).unwrap())
            .collect();

        let mut car_data = Vec::new();
        let mut writer = CarWriter::new(&mut car_data, vec![*blocks[0].cid()]).unwrap();

        for block in &blocks {
            writer.write_block(block).unwrap();
        }
        writer.finish().unwrap();

        let mut reader = CarReader::new(&car_data[..]).unwrap();
        let read_blocks = reader.read_all_blocks().unwrap();

        // Verify order is preserved
        for (i, (original, read)) in blocks.iter().zip(read_blocks.iter()).enumerate() {
            prop_assert_eq!(original.cid(), read.cid(), "Mismatch at index {}", i);
        }
    }

    /// Property: CAR can handle large blocks
    #[test]
    fn prop_car_handles_large_blocks(
        size in 100_000usize..=500_000
    ) {
        let large_data = vec![0x42u8; size];
        let block = Block::new(large_data.clone().into()).unwrap();

        let mut car_data = Vec::new();
        let mut writer = CarWriter::new(&mut car_data, vec![*block.cid()]).unwrap();
        writer.write_block(&block).unwrap();
        writer.finish().unwrap();

        let mut reader = CarReader::new(&car_data[..]).unwrap();
        let read_block = reader.read_block().unwrap().unwrap();

        prop_assert_eq!(read_block.cid(), block.cid());
        prop_assert_eq!(read_block.data().len(), size);
    }

    /// Property: CAR reader detects end of stream correctly
    #[test]
    fn prop_car_reader_eof(
        block_data in arb_blocks()
    ) {
        if block_data.is_empty() {
            return Ok(());
        }

        let blocks: Vec<Block> = block_data
            .iter()
            .map(|data| Block::new(data.clone().into()).unwrap())
            .collect();

        let mut car_data = Vec::new();
        let mut writer = CarWriter::new(&mut car_data, vec![*blocks[0].cid()]).unwrap();

        for block in &blocks {
            writer.write_block(block).unwrap();
        }
        writer.finish().unwrap();

        let mut reader = CarReader::new(&car_data[..]).unwrap();

        // Read all blocks
        for _ in 0..blocks.len() {
            prop_assert!(reader.read_block().unwrap().is_some());
        }

        // Next read should return None (EOF)
        prop_assert!(reader.read_block().unwrap().is_none());
    }

    /// Property: CAR format size is reasonable (not excessive overhead)
    #[test]
    fn prop_car_size_reasonable(
        block_data in arb_blocks()
    ) {
        if block_data.is_empty() {
            return Ok(());
        }

        let blocks: Vec<Block> = block_data
            .iter()
            .map(|data| Block::new(data.clone().into()).unwrap())
            .collect();

        let total_data_size: usize = blocks.iter().map(|b| b.data().len()).sum();

        let mut car_data = Vec::new();
        let mut writer = CarWriter::new(&mut car_data, vec![*blocks[0].cid()]).unwrap();

        for block in &blocks {
            writer.write_block(block).unwrap();
        }
        writer.finish().unwrap();

        // CAR overhead should be less than 2x the data size (very conservative)
        // Actual overhead is header + varints + CIDs, much smaller than this
        prop_assert!(car_data.len() < total_data_size * 2 + 1000);
    }
}
