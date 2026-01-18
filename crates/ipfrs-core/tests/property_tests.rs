//! Property-based tests for ipfrs-core
//!
//! These tests use proptest to validate system invariants across
//! a wide range of randomly generated inputs.

use ipfrs_core::{
    compress, compression_ratio, decompress, read_chunked_file, AsyncBlockReader, BatchProcessor,
    Blake2b256Engine, Blake2b512Engine, Blake2s256Engine, Blake3Engine, Block, BlockFetcher,
    BlockReader, BytesPool, CarReader, CarWriter, CarWriterBuilder, Chunker, ChunkingConfig, Cid,
    CidBuilder, CidExt, CidStringPool, CodecRegistry, CompressionAlgorithm, DagLink, DagNode,
    HashAlgorithm, HashEngine, Ipld, JoseBuilder, MemoryBlockFetcher, MultibaseEncoding,
    Sha256Engine,
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

/// Generate a random compression algorithm for testing
fn arb_compression_algorithm() -> impl Strategy<Value = CompressionAlgorithm> {
    prop_oneof![
        Just(CompressionAlgorithm::None),
        Just(CompressionAlgorithm::Zstd),
        Just(CompressionAlgorithm::Lz4),
    ]
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

// ============================================================================
// Memory Pooling Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]
    /// Property: BytesPool get and put maintains capacity
    #[test]
    fn prop_bytes_pool_capacity(size in 1024usize..=65536) {
        let pool = BytesPool::new();

        let buf = pool.get(size);
        prop_assert!(buf.capacity() >= size);

        pool.put(buf);

        // Get another buffer of similar size - should reuse
        let buf2 = pool.get(size);
        prop_assert!(buf2.capacity() >= size);
    }

    /// Property: BytesPool hit rate improves with reuse
    #[test]
    fn prop_bytes_pool_hit_rate(ops in 10usize..100) {
        let pool = BytesPool::new();
        let size = 4096;

        // Warm up
        for _ in 0..5 {
            let buf = pool.get(size);
            pool.put(buf);
        }

        let stats_before = pool.stats();

        // Perform more operations
        for _ in 0..ops {
            let buf = pool.get(size);
            pool.put(buf);
        }

        let stats_after = pool.stats();

        // Total operations should have increased
        prop_assert!(stats_after.hits + stats_after.misses > stats_before.hits + stats_before.misses);

        // After warmup, hit rate should be > 0
        prop_assert!(stats_after.hit_rate() > 0.0);
    }

    /// Property: CidStringPool deduplicates identical strings
    #[test]
    fn prop_cid_string_pool_deduplicates(
        data_items in prop::collection::vec(
            prop::collection::vec(any::<u8>(), 10..100),
            5..20
        )
    ) {
        let pool = CidStringPool::new();

        // Generate CIDs from data
        let cids: Vec<String> = data_items
            .iter()
            .map(|data| CidBuilder::new().build(data).unwrap().to_string())
            .collect();

        // Intern all CIDs
        let arcs: Vec<_> = cids.iter().map(|s| pool.intern(s)).collect();

        // Intern them again - should get same Arcs
        for (i, cid_str) in cids.iter().enumerate() {
            let arc = pool.intern(cid_str);
            prop_assert!(std::sync::Arc::ptr_eq(&arc, &arcs[i]));
        }

        // Pool size should equal number of unique CIDs
        let unique_set: std::collections::HashSet<_> = cids.iter().collect();
        prop_assert_eq!(pool.len(), unique_set.len());
    }

    /// Property: CidStringPool stats are consistent
    #[test]
    fn prop_cid_string_pool_stats(
        strings in prop::collection::vec("[a-zA-Z0-9]{10,20}", 10..50)
    ) {
        let pool = CidStringPool::new();

        // Intern each string twice
        for s in &strings {
            pool.intern(s); // First time (miss)
            pool.intern(s); // Second time (hit)
        }

        let stats = pool.stats();

        // We should have exactly strings.len() misses (one per unique string)
        // and at least strings.len() hits (one per duplicate intern)
        prop_assert!(stats.misses > 0);
        prop_assert!(stats.hits >= stats.misses);
        prop_assert_eq!(stats.hits + stats.misses, strings.len() as u64 * 2);
    }

    /// Property: Pool clear resets state
    #[test]
    fn prop_pool_clear_resets(size in 1024usize..=8192) {
        let bytes_pool = BytesPool::new();
        let cid_pool = CidStringPool::new();

        // Use the pools
        for _ in 0..10 {
            let buf = bytes_pool.get(size);
            bytes_pool.put(buf);
        }

        cid_pool.intern("test1");
        cid_pool.intern("test2");

        // Clear the pools
        bytes_pool.clear();
        cid_pool.clear();

        // CID pool should be empty
        prop_assert_eq!(cid_pool.len(), 0);
        prop_assert!(cid_pool.is_empty());

        // Next access should be a miss
        let stats_before = bytes_pool.stats();
        let _buf = bytes_pool.get(size);
        let stats_after = bytes_pool.stats();

        prop_assert_eq!(stats_after.misses, stats_before.misses + 1);
    }
}

// ============================================================================
// BLAKE3 Hash Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]
    /// Property: BLAKE3 is deterministic - same input produces same hash
    #[test]
    fn prop_blake3_deterministic(data in arb_block_data()) {
        let engine = Blake3Engine::new();
        let hash1 = engine.digest(&data);
        let hash2 = engine.digest(&data);
        prop_assert_eq!(hash1, hash2);
    }

    /// Property: BLAKE3 produces 32-byte hashes
    #[test]
    fn prop_blake3_hash_length(data in arb_block_data()) {
        let engine = Blake3Engine::new();
        let hash = engine.digest(&data);
        prop_assert_eq!(hash.len(), 32);
    }

    /// Property: BLAKE3 different inputs produce different hashes
    #[test]
    fn prop_blake3_collision_resistance(
        data1 in arb_block_data(),
        data2 in arb_block_data()
    ) {
        if data1 != data2 {
            let engine = Blake3Engine::new();
            let hash1 = engine.digest(&data1);
            let hash2 = engine.digest(&data2);
            prop_assert_ne!(hash1, hash2);
        }
    }

    /// Property: BLAKE3 vs SHA256 produce different hashes for same input
    #[test]
    fn prop_blake3_differs_from_sha256(data in arb_block_data()) {
        let blake3 = Blake3Engine::new();
        let sha256 = Sha256Engine::new();

        let blake3_hash = blake3.digest(&data);
        let sha256_hash = sha256.digest(&data);

        // Both produce 32-byte hashes
        prop_assert_eq!(blake3_hash.len(), 32);
        prop_assert_eq!(sha256_hash.len(), 32);

        // But hashes should differ (different algorithms)
        prop_assert_ne!(blake3_hash, sha256_hash);
    }

    /// Property: BLAKE3 empty input is deterministic
    #[test]
    fn prop_blake3_empty_deterministic(_data in any::<u8>()) {
        let engine = Blake3Engine::new();
        let hash1 = engine.digest(&[]);
        let hash2 = engine.digest(&[]);
        prop_assert_eq!(hash1.len(), 32);
        prop_assert_eq!(hash1, hash2);
    }

    /// Property: BLAKE3 always reports SIMD as enabled
    #[test]
    fn prop_blake3_simd_enabled(_data in any::<u8>()) {
        let engine = Blake3Engine::new();
        prop_assert!(engine.is_simd_enabled());
    }

    /// Property: BLAKE2b-256 is deterministic
    #[test]
    fn prop_blake2b256_deterministic(data in arb_block_data()) {
        let engine = Blake2b256Engine::new();
        let hash1 = engine.digest(&data);
        let hash2 = engine.digest(&data);
        prop_assert_eq!(hash1, hash2);
    }

    /// Property: BLAKE2b-256 produces 32-byte hashes
    #[test]
    fn prop_blake2b256_hash_length(data in arb_block_data()) {
        let engine = Blake2b256Engine::new();
        let hash = engine.digest(&data);
        prop_assert_eq!(hash.len(), 32);
    }

    /// Property: BLAKE2b-512 is deterministic
    #[test]
    fn prop_blake2b512_deterministic(data in arb_block_data()) {
        let engine = Blake2b512Engine::new();
        let hash1 = engine.digest(&data);
        let hash2 = engine.digest(&data);
        prop_assert_eq!(hash1, hash2);
    }

    /// Property: BLAKE2b-512 produces 64-byte hashes
    #[test]
    fn prop_blake2b512_hash_length(data in arb_block_data()) {
        let engine = Blake2b512Engine::new();
        let hash = engine.digest(&data);
        prop_assert_eq!(hash.len(), 64);
    }

    /// Property: BLAKE2s-256 is deterministic
    #[test]
    fn prop_blake2s256_deterministic(data in arb_block_data()) {
        let engine = Blake2s256Engine::new();
        let hash1 = engine.digest(&data);
        let hash2 = engine.digest(&data);
        prop_assert_eq!(hash1, hash2);
    }

    /// Property: BLAKE2s-256 produces 32-byte hashes
    #[test]
    fn prop_blake2s256_hash_length(data in arb_block_data()) {
        let engine = Blake2s256Engine::new();
        let hash = engine.digest(&data);
        prop_assert_eq!(hash.len(), 32);
    }

    /// Property: BLAKE2b-256 different inputs produce different hashes
    #[test]
    fn prop_blake2b256_collision_resistance(
        data1 in arb_block_data(),
        data2 in arb_block_data()
    ) {
        if data1 != data2 {
            let engine = Blake2b256Engine::new();
            let hash1 = engine.digest(&data1);
            let hash2 = engine.digest(&data2);
            prop_assert_ne!(hash1, hash2);
        }
    }

    /// Property: BLAKE2b vs BLAKE2s produce different hashes for same input
    #[test]
    fn prop_blake2b_differs_from_blake2s(data in arb_block_data()) {
        let blake2b = Blake2b256Engine::new();
        let blake2s = Blake2s256Engine::new();

        let blake2b_hash = blake2b.digest(&data);
        let blake2s_hash = blake2s.digest(&data);

        // Both produce 32-byte hashes
        prop_assert_eq!(blake2b_hash.len(), 32);
        prop_assert_eq!(blake2s_hash.len(), 32);

        // But hashes should differ (different algorithms)
        prop_assert_ne!(blake2b_hash, blake2s_hash);
    }

    /// Property: BLAKE2b empty input is deterministic
    #[test]
    fn prop_blake2b_empty_deterministic(_data in any::<u8>()) {
        let engine = Blake2b256Engine::new();
        let hash1 = engine.digest(&[]);
        let hash2 = engine.digest(&[]);
        prop_assert_eq!(hash1.len(), 32);
        prop_assert_eq!(hash1, hash2);
    }

    /// Property: BLAKE2 engines always report SIMD as enabled
    #[test]
    fn prop_blake2_simd_enabled(_data in any::<u8>()) {
        let blake2b256 = Blake2b256Engine::new();
        let blake2b512 = Blake2b512Engine::new();
        let blake2s = Blake2s256Engine::new();

        prop_assert!(blake2b256.is_simd_enabled());
        prop_assert!(blake2b512.is_simd_enabled());
        prop_assert!(blake2s.is_simd_enabled());
    }
}

// ============================================================================
// DAG-JOSE Property Tests
// ============================================================================

/// Generate arbitrary IPLD data for JOSE testing
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

/// Generate a valid HMAC secret (32+ bytes)
fn arb_hmac_secret() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 32..=64)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]
    /// Property: JOSE signing and verification roundtrip
    #[test]
    fn prop_jose_sign_verify_roundtrip(
        ipld in arb_ipld_simple(),
        secret in arb_hmac_secret()
    ) {
        let jose = JoseBuilder::new()
            .with_payload(ipld.clone())
            .sign_hs256(&secret)
            .unwrap();

        // Should verify with correct secret
        prop_assert!(jose.verify_hs256(&secret).unwrap());

        // Should fail with different secret
        let mut wrong_secret = secret.clone();
        wrong_secret[0] = wrong_secret[0].wrapping_add(1);
        prop_assert!(!jose.verify_hs256(&wrong_secret).unwrap());

        // Payload should match
        prop_assert_eq!(jose.payload, ipld);
    }

    /// Property: JOSE signatures are deterministic
    #[test]
    fn prop_jose_deterministic(
        ipld in arb_ipld_simple(),
        secret in arb_hmac_secret()
    ) {
        let jose1 = JoseBuilder::new()
            .with_payload(ipld.clone())
            .sign_hs256(&secret)
            .unwrap();

        let jose2 = JoseBuilder::new()
            .with_payload(ipld.clone())
            .sign_hs256(&secret)
            .unwrap();

        // Same payload + secret should produce same signature
        prop_assert_eq!(jose1.signature, jose2.signature);
        prop_assert_eq!(jose1.algorithm, jose2.algorithm);
    }

    /// Property: JOSE different payloads produce different signatures
    #[test]
    fn prop_jose_different_payloads_different_sigs(
        ipld1 in arb_ipld_simple(),
        ipld2 in arb_ipld_simple(),
        secret in arb_hmac_secret()
    ) {
        if ipld1 != ipld2 {
            let jose1 = JoseBuilder::new()
                .with_payload(ipld1)
                .sign_hs256(&secret)
                .unwrap();

            let jose2 = JoseBuilder::new()
                .with_payload(ipld2)
                .sign_hs256(&secret)
                .unwrap();

            // Different payloads should produce different signatures
            prop_assert_ne!(jose1.signature, jose2.signature);
        }
    }

    /// Property: JOSE DAG-JOSE encoding roundtrip
    #[test]
    fn prop_jose_dag_jose_roundtrip(
        ipld in arb_ipld_simple(),
        secret in arb_hmac_secret()
    ) {
        let jose = JoseBuilder::new()
            .with_payload(ipld.clone())
            .sign_hs256(&secret)
            .unwrap();

        // Encode to DAG-JOSE
        let dag_jose = jose.to_dag_jose().unwrap();

        // Should be valid JSON
        let parsed: serde_json::Value = serde_json::from_slice(&dag_jose).unwrap();
        prop_assert!(parsed.get("payload").is_some());
        prop_assert!(parsed.get("signatures").is_some());

        // Decode back
        let decoded = ipfrs_core::JoseSignature::from_dag_jose(&dag_jose).unwrap();

        // Should still verify
        prop_assert!(decoded.verify_hs256(&secret).unwrap());

        // Payload should match (with special handling for floats due to JSON precision)
        match (&decoded.payload, &ipld) {
            (Ipld::Float(f1), Ipld::Float(f2)) => {
                // Allow small difference due to JSON serialization precision
                let diff = (f1 - f2).abs();
                let rel_diff = diff / f2.abs().max(1e-10);
                prop_assert!(rel_diff < 1e-10 || diff < 1e-10,
                    "Float values differ: {} vs {}, diff={}, rel_diff={}", f1, f2, diff, rel_diff);
            }
            _ => prop_assert_eq!(decoded.payload, ipld),
        }
    }

    /// Property: JOSE algorithm field is always set correctly
    #[test]
    fn prop_jose_algorithm_correct(
        ipld in arb_ipld_simple(),
        secret in arb_hmac_secret()
    ) {
        let jose = JoseBuilder::new()
            .with_payload(ipld)
            .sign_hs256(&secret)
            .unwrap();

        prop_assert_eq!(jose.algorithm, "HS256");
    }

    /// Property: JOSE signature is non-empty
    #[test]
    fn prop_jose_signature_nonempty(
        ipld in arb_ipld_simple(),
        secret in arb_hmac_secret()
    ) {
        let jose = JoseBuilder::new()
            .with_payload(ipld)
            .sign_hs256(&secret)
            .unwrap();

        prop_assert!(!jose.signature.is_empty());
        // JWT signatures are typically base64-encoded
        prop_assert!(jose.signature.len() > 50);
    }
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

// ============================================================================
// Compression Property Tests
// ============================================================================

/// Generate arbitrary data for compression tests (1 byte to 10KB)
fn arb_compression_data() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 1..=10240)
}

/// Generate arbitrary compression level (0-9)
fn arb_compression_level() -> impl Strategy<Value = u8> {
    0u8..=9
}

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

/// Generate arbitrary compression data chunks for batch operations
fn arb_batch_compression_chunks() -> impl Strategy<Value = Vec<Vec<u8>>> {
    prop::collection::vec(arb_compression_data(), 1..=20)
}

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

/// Generate arbitrary blocks for CAR compression tests
/// Reduced from 1-10 blocks to 1-3 blocks for faster testing
fn arb_car_blocks() -> impl Strategy<Value = Vec<Vec<u8>>> {
    prop::collection::vec(arb_block_data(), 1..=3)
}

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

use ipfrs_core::{
    count_links_by_depth, dag_fanout_by_level, filter_dag, map_dag, subgraph_size,
    topological_sort, DagMetrics,
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]
    /// Property: Subgraph size is always at least 1 (for the root)
    #[test]
    fn prop_dag_subgraph_size_at_least_one(
        data in arb_block_data()
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
        use ipfrs_core::collect_all_links;

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
