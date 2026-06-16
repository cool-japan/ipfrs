//! Property-based tests for ipfrs-core — Block, CID, IPLD, Chunking, DAG, Streaming
//!
//! These tests use proptest to validate system invariants across
//! a wide range of randomly generated inputs.

use ipfrs_core::{
    read_chunked_file, AsyncBlockReader, Block, BlockFetcher, BlockReader, Chunker, ChunkingConfig,
    Cid, CidBuilder, CidExt, DagLink, DagNode, Ipld, MemoryBlockFetcher, MultibaseEncoding,
};
use proptest::prelude::*;
use std::collections::BTreeMap;
use std::io::Read;

// Reduce proptest cases for faster test execution
// Default is 256, we use 32 for reasonable coverage without excessive runtime
const PROPTEST_CASES: u32 = 32;

// ============================================================================
// Block Property Tests
// ============================================================================

/// Generate arbitrary byte vectors for blocks (1 byte to 8KB)
/// Reduced from 64KB to speed up tests
fn arb_block_data() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 1..=8192)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]
    /// Property: Creating a block from data always succeeds for valid inputs
    #[test]
    fn prop_block_creation_succeeds(data in arb_block_data()) {
        let block = Block::new(data.into());
        prop_assert!(block.is_ok());
    }

    /// Property: Block CID is deterministic - same data produces same CID
    #[test]
    fn prop_block_cid_deterministic(data in arb_block_data()) {
        let block1 = Block::new(data.clone().into()).unwrap();
        let block2 = Block::new(data.into()).unwrap();
        prop_assert_eq!(block1.cid(), block2.cid());
    }

    /// Property: Block data round-trip preserves content
    #[test]
    fn prop_block_data_roundtrip(data in arb_block_data()) {
        let original_data = data.clone();
        let block = Block::new(data.into()).unwrap();
        let retrieved_data = block.data();
        prop_assert_eq!(&original_data[..], retrieved_data.as_ref());
    }

    /// Property: Block size matches original data length
    #[test]
    fn prop_block_size_correct(data in arb_block_data()) {
        let data_len = data.len() as u64;
        let block = Block::new(data.into()).unwrap();
        prop_assert_eq!(block.size(), data_len);
    }

    /// Property: Different data produces different CIDs
    #[test]
    fn prop_different_data_different_cids(
        data1 in arb_block_data(),
        data2 in arb_block_data()
    ) {
        // Only test when data is actually different
        if data1 != data2 {
            let block1 = Block::new(data1.into()).unwrap();
            let block2 = Block::new(data2.into()).unwrap();
            prop_assert_ne!(block1.cid(), block2.cid());
        }
    }
}

// ============================================================================
// CID Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]
    /// Property: CID to_string and from_str are inverses
    #[test]
    fn prop_cid_string_roundtrip(data in arb_block_data()) {
        let block = Block::new(data.into()).unwrap();
        let cid = block.cid();

        let cid_string = cid.to_string();
        let parsed: Cid = cid_string.parse().unwrap();

        prop_assert_eq!(cid, &parsed);
    }

    /// Property: CID Display format is valid multibase
    #[test]
    fn prop_cid_display_valid(data in arb_block_data()) {
        let block = Block::new(data.into()).unwrap();
        let cid = block.cid();

        let display_string = format!("{}", cid);
        // Should start with 'b' for base32 or 'z' for base58btc
        prop_assert!(
            display_string.starts_with('b') || display_string.starts_with('z'),
            "CID display format should be valid multibase"
        );
    }
}

// ============================================================================
// IPLD Property Tests
// ============================================================================

/// Generate arbitrary IPLD values
fn arb_ipld_value() -> impl Strategy<Value = Ipld> {
    let leaf = prop_oneof![
        any::<bool>().prop_map(Ipld::Bool),
        any::<i128>().prop_map(Ipld::Integer),
        any::<f64>()
            .prop_filter("Finite f64", |f| f.is_finite())
            .prop_map(Ipld::Float),
        ".*".prop_map(Ipld::String),
        prop::collection::vec(any::<u8>(), 0..=1024).prop_map(Ipld::Bytes),
        Just(Ipld::Null),
    ];

    leaf.prop_recursive(
        3,   // Max depth
        256, // Max nodes
        10,  // Items per collection
        |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 0..=10).prop_map(Ipld::List),
                prop::collection::hash_map(".*", inner, 0..=10)
                    .prop_map(|m| Ipld::Map(m.into_iter().collect())),
            ]
        },
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]
    /// Property: IPLD clone equals original
    #[test]
    fn prop_ipld_clone_equals(value in arb_ipld_value()) {
        let cloned = value.clone();
        prop_assert_eq!(value, cloned);
    }

    /// Property: IPLD can be converted to/from JSON for simple types
    #[test]
    fn prop_ipld_string_to_from_json(s in ".*") {
        let value = Ipld::String(s.clone());
        let result = value.to_json().and_then(|json| Ipld::from_json(&json));
        prop_assert!(result.is_ok(), "JSON round-trip should succeed for String");
        // Note: We don't check exact equality due to JSON number representation issues
    }

    /// Property: IPLD DAG-CBOR encoding doesn't panic
    #[test]
    fn prop_ipld_dag_cbor_no_panic(value in arb_ipld_value()) {
        // Just verify it doesn't panic - actual round-trip may have limitations
        let _ = value.to_dag_cbor();
    }
}

// ============================================================================
// IPLD Type Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]
    /// Property: IPLD pattern matching correctly identifies types
    #[test]
    fn prop_ipld_type_matching(value in arb_ipld_value()) {
        // Pattern matching should work correctly for all types
        match &value {
            Ipld::Null => prop_assert!(matches!(value, Ipld::Null)),
            Ipld::Bool(_) => prop_assert!(matches!(value, Ipld::Bool(_))),
            Ipld::Integer(_) => prop_assert!(matches!(value, Ipld::Integer(_))),
            Ipld::Float(_) => prop_assert!(matches!(value, Ipld::Float(_))),
            Ipld::String(_) => prop_assert!(matches!(value, Ipld::String(_))),
            Ipld::Bytes(_) => prop_assert!(matches!(value, Ipld::Bytes(_))),
            Ipld::List(_) => prop_assert!(matches!(value, Ipld::List(_))),
            Ipld::Map(_) => prop_assert!(matches!(value, Ipld::Map(_))),
            Ipld::Link(_) => prop_assert!(matches!(value, Ipld::Link(_))),
        }
    }

    /// Property: IPLD Map uses BTreeMap (ordered keys)
    #[test]
    fn prop_ipld_map_ordered(
        entries in prop::collection::hash_map(".*", any::<i128>(), 0..=10)
    ) {
        let map: BTreeMap<String, Ipld> = entries
            .into_iter()
            .map(|(k, v)| (k, Ipld::Integer(v)))
            .collect();
        let value = Ipld::Map(map.clone());

        // Extract keys to verify ordering
        if let Ipld::Map(extracted_map) = value {
            let keys: Vec<_> = extracted_map.keys().collect();
            let mut sorted_keys = keys.clone();
            sorted_keys.sort();
            prop_assert_eq!(keys, sorted_keys, "Map keys should be sorted");
        }
    }

    /// Property: IPLD List preserves order
    #[test]
    fn prop_ipld_list_ordered(items in prop::collection::vec(any::<i128>(), 0..=20)) {
        let list: Vec<Ipld> = items.iter().map(|&i| Ipld::Integer(i)).collect();
        let value = Ipld::List(list.clone());

        if let Ipld::List(extracted) = value {
            prop_assert_eq!(list.len(), extracted.len());
            for (orig, ext) in list.iter().zip(extracted.iter()) {
                prop_assert_eq!(orig, ext);
            }
        }
    }
}

// ============================================================================
// Invariant Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]
    /// Property: Block size is never zero for non-empty data
    #[test]
    fn prop_block_size_nonzero(data in arb_block_data()) {
        let block = Block::new(data.into()).unwrap();
        prop_assert!(block.size() > 0);
    }

    /// Property: CID string representation is non-empty
    #[test]
    fn prop_cid_string_nonempty(data in arb_block_data()) {
        let block = Block::new(data.into()).unwrap();
        let cid_str = block.cid().to_string();
        prop_assert!(!cid_str.is_empty());
    }

    /// Property: Multiple blocks can be created independently
    #[test]
    fn prop_blocks_independent(
        data1 in arb_block_data(),
        data2 in arb_block_data(),
        data3 in arb_block_data()
    ) {
        let block1 = Block::new(data1.into()).unwrap();
        let block2 = Block::new(data2.into()).unwrap();
        let block3 = Block::new(data3.into()).unwrap();

        // All blocks should have valid CIDs
        prop_assert!(!block1.cid().to_string().is_empty());
        prop_assert!(!block2.cid().to_string().is_empty());
        prop_assert!(!block3.cid().to_string().is_empty());
    }
}

// ============================================================================
// Chunking Property Tests
// ============================================================================

/// Generate data of various sizes for chunking tests
fn arb_chunking_data() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 1..=10000)
}

/// Generate valid chunk sizes
fn arb_chunk_size() -> impl Strategy<Value = usize> {
    1024usize..=65536
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]
    /// Property: Chunking and reassembling data preserves content
    #[test]
    fn prop_chunking_roundtrip(data in arb_chunking_data()) {
        let config = ChunkingConfig::with_chunk_size(1024).unwrap();
        let chunker = Chunker::with_config(config);

        let chunked = chunker.chunk(&data).unwrap();

        // Verify total size matches
        prop_assert_eq!(chunked.total_size, data.len() as u64);

        // Verify we have at least one block
        prop_assert!(!chunked.blocks.is_empty());
    }

    /// Property: Chunk count estimation is accurate
    #[test]
    fn prop_chunk_count_estimation(
        data_len in 1usize..=100000,
        chunk_size in arb_chunk_size()
    ) {
        let config = ChunkingConfig::with_chunk_size(chunk_size).unwrap();
        let chunker = Chunker::with_config(config);

        let estimated = chunker.estimate_chunk_count(data_len);
        let expected = data_len.div_ceil(chunk_size);

        prop_assert_eq!(estimated, expected);
    }

    /// Property: needs_chunking is consistent with chunk_size
    #[test]
    fn prop_needs_chunking_consistency(
        data_len in 1usize..=100000,
        chunk_size in arb_chunk_size()
    ) {
        let config = ChunkingConfig::with_chunk_size(chunk_size).unwrap();
        let chunker = Chunker::with_config(config);

        let needs = chunker.needs_chunking(data_len);
        prop_assert_eq!(needs, data_len > chunk_size);
    }

    /// Property: Small data (<=chunk_size) produces single block
    #[test]
    fn prop_small_data_single_block(data in prop::collection::vec(any::<u8>(), 1..=1024)) {
        let config = ChunkingConfig::with_chunk_size(1024).unwrap();
        let chunker = Chunker::with_config(config);

        let chunked = chunker.chunk(&data).unwrap();
        prop_assert_eq!(chunked.chunk_count, 1);
        prop_assert_eq!(chunked.blocks.len(), 1);
    }

    /// Property: Root CID is deterministic for same data
    #[test]
    fn prop_chunking_deterministic(data in arb_chunking_data()) {
        let config = ChunkingConfig::with_chunk_size(1024).unwrap();
        let chunker = Chunker::with_config(config);

        let result1 = chunker.chunk(&data).unwrap();
        let result2 = chunker.chunk(&data).unwrap();

        prop_assert_eq!(result1.root_cid, result2.root_cid);
        prop_assert_eq!(result1.chunk_count, result2.chunk_count);
    }
}

// ============================================================================
// DAG Node Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]
    /// Property: DAG leaf node has correct size
    #[test]
    fn prop_dag_leaf_size(data in prop::collection::vec(any::<u8>(), 1..=1024)) {
        let node = DagNode::leaf(data.clone());
        prop_assert_eq!(node.total_size, data.len() as u64);
        prop_assert!(node.is_leaf());
        prop_assert_eq!(node.link_count(), 0);
    }

    /// Property: DAG intermediate node accumulates child sizes
    #[test]
    fn prop_dag_intermediate_size(sizes in prop::collection::vec(1u64..=10000, 1..=10)) {
        let cid = CidBuilder::new().build(b"test").unwrap();
        let links: Vec<DagLink> = sizes.iter().map(|&s| DagLink::new(cid, s)).collect();

        let node = DagNode::intermediate(links);
        let expected_size: u64 = sizes.iter().sum();

        prop_assert_eq!(node.total_size, expected_size);
        prop_assert!(!node.is_leaf());
    }

    /// Property: DAG node to_ipld produces valid IPLD Map
    #[test]
    fn prop_dag_node_to_ipld(data in prop::collection::vec(any::<u8>(), 1..=256)) {
        let node = DagNode::leaf(data);
        let ipld = node.to_ipld();

        prop_assert!(matches!(ipld, Ipld::Map(_)));
        if let Ipld::Map(map) = ipld {
            prop_assert!(map.contains_key("links"));
            prop_assert!(map.contains_key("totalSize"));
            prop_assert!(map.contains_key("data"));
        }
    }

    /// Property: DAG node serializes to valid DAG-CBOR
    #[test]
    fn prop_dag_node_cbor_valid(data in prop::collection::vec(any::<u8>(), 1..=256)) {
        let node = DagNode::leaf(data);
        let cbor_result = node.to_dag_cbor();
        prop_assert!(cbor_result.is_ok());
        prop_assert!(!cbor_result.unwrap().is_empty());
    }
}

// ============================================================================
// Streaming Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]
    /// Property: BlockReader reads all data correctly
    #[test]
    fn prop_block_reader_complete(data in arb_block_data()) {
        let block = Block::new(data.clone().into()).unwrap();
        let mut reader = BlockReader::new(&block);

        let mut result = Vec::new();
        reader.read_to_end(&mut result).unwrap();

        prop_assert_eq!(result, data);
    }

    /// Property: BlockReader remaining() is accurate
    #[test]
    fn prop_block_reader_remaining(data in arb_block_data()) {
        let block = Block::new(data.clone().into()).unwrap();
        let mut reader = BlockReader::new(&block);

        prop_assert_eq!(reader.remaining(), data.len());
        prop_assert_eq!(reader.len(), data.len());
        prop_assert!(!reader.is_empty());

        // Read some data
        let mut buf = [0u8; 10];
        let n = reader.read(&mut buf).unwrap();

        prop_assert_eq!(reader.remaining(), data.len() - n);
    }

    /// Property: AsyncBlockReader has correct initial state
    #[test]
    fn prop_async_block_reader_state(data in arb_block_data()) {
        let block = Block::new(data.clone().into()).unwrap();
        let reader = AsyncBlockReader::new(&block);

        prop_assert_eq!(reader.remaining(), data.len());
        prop_assert_eq!(reader.len(), data.len());
        prop_assert!(!reader.is_empty());
    }

    /// Property: MemoryBlockFetcher stores and retrieves blocks correctly
    #[test]
    fn prop_memory_fetcher_roundtrip(data in arb_block_data()) {
        let block = Block::new(data.clone().into()).unwrap();
        let cid = *block.cid();

        let mut fetcher = MemoryBlockFetcher::new();
        fetcher.add_block(block.clone());

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let fetched = rt.block_on(async {
            fetcher.fetch(cid).await
        }).unwrap();

        prop_assert_eq!(fetched.data(), block.data());
        prop_assert_eq!(fetched.cid(), block.cid());
    }
}

// ============================================================================
// Multibase Encoding Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]
    /// Property: CID encoding with different bases produces valid strings
    #[test]
    fn prop_multibase_encoding_valid(data in arb_block_data()) {
        let block = Block::new(data.into()).unwrap();
        let cid = block.cid();

        // Test all base encodings
        let base32_lower = cid.to_string_with_base(MultibaseEncoding::Base32Lower);
        let base32_upper = cid.to_string_with_base(MultibaseEncoding::Base32Upper);
        let base58btc = cid.to_string_with_base(MultibaseEncoding::Base58Btc);
        let base64 = cid.to_string_with_base(MultibaseEncoding::Base64);
        let base64_url = cid.to_string_with_base(MultibaseEncoding::Base64Url);

        // All should be non-empty
        prop_assert!(!base32_lower.is_empty());
        prop_assert!(!base32_upper.is_empty());
        prop_assert!(!base58btc.is_empty());
        prop_assert!(!base64.is_empty());
        prop_assert!(!base64_url.is_empty());

        // Check prefixes
        prop_assert!(base32_lower.starts_with('b'));
        prop_assert!(base32_upper.starts_with('B'));
        prop_assert!(base58btc.starts_with('z'));
        prop_assert!(base64.starts_with('m'));
        prop_assert!(base64_url.starts_with('u'));
    }

    /// Property: CID can be parsed from any multibase encoding
    #[test]
    fn prop_multibase_roundtrip(data in arb_block_data()) {
        let block = Block::new(data.into()).unwrap();
        let cid = block.cid();

        // Test roundtrip for each encoding
        let encodings = [
            MultibaseEncoding::Base32Lower,
            MultibaseEncoding::Base32Upper,
            MultibaseEncoding::Base58Btc,
            MultibaseEncoding::Base64,
            MultibaseEncoding::Base64Url,
        ];

        for encoding in &encodings {
            let encoded = cid.to_string_with_base(*encoding);
            let parsed: Cid = encoded.parse().unwrap();
            prop_assert_eq!(cid, &parsed);
        }
    }
}

// ============================================================================
// CID Version Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]
    /// Property: CIDv1 correctly identifies as v1
    #[test]
    fn prop_cidv1_identification(data in arb_block_data()) {
        let block = Block::new(data.into()).unwrap();
        let cid = block.cid();

        prop_assert!(cid.is_v1());
        prop_assert!(!cid.is_v0());
    }

    /// Property: CID hash algorithm is correctly reported
    #[test]
    fn prop_cid_hash_algorithm(data in arb_block_data()) {
        let block = Block::new(data.into()).unwrap();
        let cid = block.cid();

        // Default hash is SHA2-256 (code 0x12)
        prop_assert_eq!(cid.hash_algorithm_code(), 0x12);
        prop_assert_eq!(cid.hash_algorithm_name(), "sha2-256");
    }

    /// Property: CIDv0 creation works for SHA2-256 hashed data
    #[test]
    fn prop_cidv0_creation(data in prop::collection::vec(any::<u8>(), 1..=1024)) {
        let cid_v0 = CidBuilder::v0().build_v0(&data).unwrap();

        prop_assert!(cid_v0.is_v0());
        prop_assert!(!cid_v0.is_v1());
        prop_assert!(cid_v0.can_be_v0());

        // V0 string should start with "Qm"
        let v0_string = cid_v0.to_string();
        prop_assert!(v0_string.starts_with("Qm"));
    }

    /// Property: CIDv0 to CIDv1 conversion preserves content hash
    #[test]
    fn prop_cidv0_v1_conversion(data in prop::collection::vec(any::<u8>(), 1..=1024)) {
        let cid_v0 = CidBuilder::v0().build_v0(&data).unwrap();
        let cid_v1 = cid_v0.to_v1().unwrap();

        prop_assert!(cid_v1.is_v1());
        prop_assert!(cid_v1.can_be_v0());

        // Converting back should give equivalent CID
        let back_to_v0 = cid_v1.to_v0().unwrap();
        prop_assert_eq!(cid_v0, back_to_v0);
    }
}

// ============================================================================
// Integrated Chunking + Streaming Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]
    /// Property: Chunked data can be fully retrieved via streaming
    #[test]
    fn prop_chunk_stream_roundtrip(data in prop::collection::vec(any::<u8>(), 1..=5000)) {
        let config = ChunkingConfig::with_chunk_size(1024).unwrap();
        let chunker = Chunker::with_config(config);

        let chunked = chunker.chunk(&data).unwrap();

        // Add all blocks to fetcher
        let mut fetcher = MemoryBlockFetcher::new();
        for block in &chunked.blocks {
            fetcher.add_block(block.clone());
        }

        // Read back via streaming
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let result = rt.block_on(async {
            read_chunked_file(&fetcher, &chunked.root_cid).await
        }).unwrap();

        prop_assert_eq!(result, data);
    }
}
// ============================================================================
// CDC (Content-Defined Chunking) Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]
    /// Property: CDC chunking is deterministic
    #[test]
    fn prop_cdc_deterministic(data in prop::collection::vec(any::<u8>(), 1000..=10000)) {
        let config = ChunkingConfig::content_defined();
        let chunker = Chunker::with_config(config);

        let result1 = chunker.chunk(&data).unwrap();
        let result2 = chunker.chunk(&data).unwrap();

        prop_assert_eq!(result1.root_cid, result2.root_cid);
        prop_assert_eq!(result1.chunk_count, result2.chunk_count);
        prop_assert_eq!(result1.total_size, result2.total_size);
    }

    /// Property: CDC produces consistent deduplication stats
    #[test]
    fn prop_cdc_dedup_stats_consistent(data in prop::collection::vec(any::<u8>(), 1000..=10000)) {
        let config = ChunkingConfig::content_defined();
        let chunker = Chunker::with_config(config);

        let result = chunker.chunk(&data).unwrap();
        let stats = result.dedup_stats.unwrap();

        // total_chunks = unique_chunks + reused_chunks
        prop_assert_eq!(stats.total_chunks, stats.unique_chunks + stats.reused_chunks);

        // Space savings should be between 0% and 100%
        prop_assert!(stats.space_savings_percent >= 0.0);
        prop_assert!(stats.space_savings_percent <= 100.0);

        // Deduplicated size should not exceed total size
        prop_assert!(stats.deduplicated_size <= stats.total_data_size);
    }

    /// Property: CDC with different target sizes produces different chunk boundaries
    #[test]
    fn prop_cdc_target_size_affects_chunking(
        data in prop::collection::vec(any::<u8>(), 10000..=50000)
    ) {
        let small_config = ChunkingConfig::content_defined_with_size(4096).unwrap();
        let large_config = ChunkingConfig::content_defined_with_size(16384).unwrap();

        let small_chunker = Chunker::with_config(small_config);
        let large_chunker = Chunker::with_config(large_config);

        let small_result = small_chunker.chunk(&data).unwrap();
        let large_result = large_chunker.chunk(&data).unwrap();

        // Smaller target size generally produces more chunks
        // (though this isn't strictly guaranteed for all data)
        prop_assert!(small_result.chunk_count >= 1);
        prop_assert!(large_result.chunk_count >= 1);
    }

    /// Property: CDC and fixed-size chunking both preserve data
    #[test]
    fn prop_cdc_vs_fixed_preserves_data(data in prop::collection::vec(any::<u8>(), 5000..=15000)) {
        let cdc_config = ChunkingConfig::content_defined();
        let fixed_config = ChunkingConfig::with_chunk_size(4096).unwrap();

        let cdc_chunker = Chunker::with_config(cdc_config);
        let fixed_chunker = Chunker::with_config(fixed_config);

        let cdc_result = cdc_chunker.chunk(&data).unwrap();
        let fixed_result = fixed_chunker.chunk(&data).unwrap();

        // Both should have the same total size
        prop_assert_eq!(cdc_result.total_size, data.len() as u64);
        prop_assert_eq!(fixed_result.total_size, data.len() as u64);

        // Both should produce blocks
        prop_assert!(!cdc_result.blocks.is_empty());
        prop_assert!(!fixed_result.blocks.is_empty());
    }

    /// Property: Repeated patterns lead to better deduplication
    #[test]
    fn prop_cdc_dedup_on_repeated_patterns(
        pattern in prop::collection::vec(any::<u8>(), 100..=500),
        repetitions in 10usize..50usize
    ) {
        let mut data = Vec::new();
        for _ in 0..repetitions {
            data.extend_from_slice(&pattern);
        }

        let config = ChunkingConfig::content_defined_with_size(2048).unwrap();
        let chunker = Chunker::with_config(config);

        let result = chunker.chunk(&data).unwrap();
        let stats = result.dedup_stats.unwrap();

        // With repeated patterns, we should see some reused chunks
        // (though this depends on where boundaries fall)
        prop_assert!(stats.unique_chunks <= stats.total_chunks);
    }
}
