//! # IPFRS Core
//!
//! Core types and traits for the IPFRS (InterPlanetary File Replication System).
//!
//! This crate provides fundamental building blocks for content-addressed storage:
//!
//! - **[`Block`]** - Content-addressed data blocks with CID verification
//! - **[`Cid`]** - Content Identifiers for unique data addressing
//! - **[`Ipld`]** - InterPlanetary Linked Data for structured content
//! - **Chunking** - Split large files into Merkle DAG structures
//! - **Streaming** - Async readers for DAG traversal
//!
//! ## Quick Start
//!
//! ```rust
//! use ipfrs_core::{Block, CidBuilder};
//! use bytes::Bytes;
//!
//! // Create a block from data
//! let block = Block::new(Bytes::from_static(b"Hello, IPFS!")).unwrap();
//! println!("CID: {}", block.cid());
//!
//! // Generate a CID directly
//! let cid = CidBuilder::new().build(b"some data").unwrap();
//! println!("Generated CID: {}", cid);
//! ```
//!
//! ## Chunking Large Files
//!
//! ```rust
//! use ipfrs_core::{Chunker, ChunkingConfig};
//!
//! let data = vec![0u8; 1_000_000]; // 1MB of data
//! let chunker = Chunker::new();
//! let chunked = chunker.chunk(&data).unwrap();
//!
//! println!("Root CID: {}", chunked.root_cid);
//! println!("Chunks: {}", chunked.chunk_count);
//! ```
//!
//! ## IPLD Encoding
//!
//! ```rust
//! use ipfrs_core::Ipld;
//! use std::collections::BTreeMap;
//!
//! // Create structured data
//! let mut map = BTreeMap::new();
//! map.insert("name".to_string(), Ipld::String("example".to_string()));
//! map.insert("version".to_string(), Ipld::Integer(1));
//! let ipld = Ipld::Map(map);
//!
//! // Encode to DAG-CBOR
//! let cbor = ipld.to_dag_cbor().unwrap();
//!
//! // Decode back
//! let decoded = Ipld::from_dag_cbor(&cbor).unwrap();
//! ```
//!
//! ## Features
//!
//! - **SHA2-256, SHA2-512, SHA3-256, SHA3-512, BLAKE2b, BLAKE2s, and BLAKE3** hash algorithms with SIMD acceleration
//! - **CIDv0 and CIDv1** support with conversion
//! - **Multibase encoding** (Base32, Base58btc, Base64)
//! - **DAG-CBOR, DAG-JSON, and DAG-JOSE** codecs
//! - **Pluggable codec registry** for custom encoding/decoding
//! - **DAG traversal and analysis** utilities for Merkle DAGs
//! - **CAR (Content Addressable aRchive)** format support for data portability
//! - **Compression support** with Zstd and LZ4 algorithms for storage efficiency
//! - **Streaming compression** for efficient compression/decompression of large files
//! - **Async streaming** for large files
//! - **LRU block cache** for fast repeated access to frequently used blocks
//! - **Apache Arrow integration** for zero-copy tensor access
//! - **Parallel batch processing** with Rayon for high performance
//! - **Parallel chunking** for multi-core large file processing
//! - **Content-defined chunking** with deduplication
//! - **Production metrics** and observability with percentile tracking

pub mod arrow;
pub mod batch;
pub mod block;
pub mod block_cache;
pub mod car;
pub mod chunking;
pub mod cid;
pub mod codec_registry;
pub mod compression;
pub mod config;
pub mod dag;
pub mod error;
pub mod hash;
pub mod integration;
pub mod ipld;
pub mod jose;
pub mod manifest;
pub mod merkle_batch;
pub mod metrics;
pub mod parallel_chunking;
pub mod pool;
pub mod safetensors;
pub mod streaming;
pub mod streaming_compression;
pub mod tensor;
pub mod types;
pub mod utils;
pub mod wasm_compat;

pub use self::arrow::{
    arrow_dtype_to_tensor, arrow_to_tensor_block, tensor_dtype_to_arrow, TensorBlockArrowExt,
};
pub use self::batch::{BatchProcessor, BatchStats};
pub use self::block::{Block, BlockBuilder, BlockMetadata, MAX_BLOCK_SIZE, MIN_BLOCK_SIZE};
pub use self::block_cache::{BlockCache, CacheStats};
pub use self::car::{CarCompressionStats, CarHeader, CarReader, CarWriter, CarWriterBuilder};
pub use self::chunking::{
    ChunkedFile, Chunker, ChunkingConfig, ChunkingConfigBuilder, ChunkingStrategy, DagBuilder,
    DagLink, DagNode, DeduplicationStats,
};
pub use self::cid::{
    codec, parse_cid, parse_cid_with_base, Cid, CidBuilder, CidExt, HashAlgorithm,
    MultibaseEncoding,
};
pub use self::codec_registry::{
    global_codec_registry, Codec, CodecRegistry, DagCborCodec, DagJsonCodec, RawCodec,
};
pub use self::compression::{compress, compression_ratio, decompress, CompressionAlgorithm};
pub use self::config::{global_config, set_global_config, Config, ConfigBuilder};
pub use self::dag::{
    collect_all_links, collect_unique_links, count_links_by_depth, dag_fanout_by_level,
    extract_links, filter_dag, find_paths_to_cid, is_dag, map_dag, subgraph_size, topological_sort,
    traverse_bfs, traverse_dfs, DagMetrics, DagStats,
};
pub use self::error::{Error, Result};
pub use self::hash::{
    global_hash_registry, Blake2b256Engine, Blake2b512Engine, Blake2s256Engine, Blake3Engine,
    CpuFeatures, HashEngine, HashRegistry, Sha256Engine, Sha3_256Engine, Sha3_512Engine,
    Sha512Engine,
};
pub use self::integration::{
    DeduplicationStats as TensorDeduplicationStats, TensorBatchProcessor, TensorDeduplicator,
    TensorStore,
};
pub use self::ipld::Ipld;
pub use self::jose::{JoseBuilder, JoseSignature};
pub use self::manifest::{ContentManifest, ManifestDiff, ManifestEntry, MerkleTree};
pub use self::metrics::{global_metrics, Metrics, MetricsSnapshot, PercentileStats, Timer};
pub use self::parallel_chunking::{
    ParallelChunker, ParallelChunkingConfig, ParallelChunkingResult, ParallelDeduplicator,
};
pub use self::pool::{
    freeze_bytes, global_bytes_pool, global_cid_string_pool, BytesPool, CidStringPool, PoolStats,
};
pub use self::safetensors::{SafetensorInfo, SafetensorsFile};
pub use self::streaming::{
    read_chunked_file, AsyncBlockReader, BlockFetcher, BlockReader, DagChunkStream,
    MemoryBlockFetcher,
};
pub use self::streaming_compression::{CompressingStream, DecompressingStream, StreamingStats};
pub use self::tensor::{TensorBlock, TensorDtype, TensorMetadata, TensorShape};
pub use self::types::{BlockSize, PeerId, Priority};
pub use self::wasm_compat::{PlatformTime, TargetCapabilities, IS_WASM32};
