//! TensorSwap protocol for efficient tensor streaming
//!
//! TensorSwap extends Bitswap with optimizations for tensor data:
//! - Priority-based scheduling for computation graphs
//! - Progressive streaming for early inference
//! - Safetensors format support
//! - Computation-aware block ordering
//! - Chunked tensor transfer with backpressure
//! - Deadline-based priority elevation
//!
//! # Example
//!
//! ```
//! use ipfrs_transport::TensorMetadata;
//! use ipfrs_core::Cid;
//! use multihash::Multihash;
//! use std::time::{Duration, Instant};
//!
//! // Create a CID for the tensor
//! let hash = Multihash::wrap(0x12, &[1, 2, 3, 4]).expect("test: wrap hash bytes");
//! let cid = Cid::new_v1(0x55, hash);
//!
//! // Create tensor metadata with shape and type information
//! let metadata = TensorMetadata::new(cid)
//!     .with_shape(vec![768, 768])
//!     .with_dtype("f32".to_string())
//!     .critical()
//!     .with_deadline(Instant::now() + Duration::from_secs(5));
//!
//! println!("Tensor shape: {:?}", metadata.shape);
//! println!("Data type: {:?}", metadata.dtype);
//! println!("Is critical: {}", metadata.is_critical);
//! ```

pub mod core;
pub mod einsum;
pub mod gradient;
pub mod streaming;

// Re-export public API
pub use core::{TensorSwap, TensorSwapConfig, TensorSwapStats};
pub use einsum::{EinsumExpression, EinsumGraph};
pub use gradient::{GradientChunk, GradientStreamError, GradientStreamSession};
pub use streaming::{
    BackpressureConfig, BackpressureController, ChunkInfo, SafetensorEntry, SafetensorsHeader,
    StreamProgress, StreamRequest, StreamRequestQueue, TensorMetadata, TensorStream,
};

#[cfg(test)]
mod tests {
    use super::*;
    use ipfrs_storage::{BlockStoreConfig, SledBlockStore};
    use std::sync::Arc;
    use std::time::Instant;

    fn test_cid() -> ipfrs_core::Cid {
        "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse()
            .expect("parse test CID")
    }

    fn test_cid2() -> ipfrs_core::Cid {
        "bafybeiczsscdsbs7ffqz55asqdf3smv6klcw3gofszvwlyarci47bgf354"
            .parse()
            .expect("parse test CID2")
    }

    #[tokio::test]
    async fn test_tensorswap_creation() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-tensorswap-create"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("store"));
        let tensorswap = TensorSwap::with_defaults(store);
        assert!(tensorswap.is_ok());
    }

    #[tokio::test]
    async fn test_tensor_metadata() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-tensorswap-meta"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("store"));
        let tensorswap = TensorSwap::with_defaults(store).expect("tensorswap");

        let cid = test_cid();
        let metadata = TensorMetadata::new(cid)
            .with_shape(vec![256, 256, 3])
            .with_dtype("f32");

        tensorswap.register_tensor(metadata);

        let stats = tensorswap.stats();
        assert_eq!(stats.num_tensors_registered, 1);
    }

    #[tokio::test]
    async fn test_dependency_scheduling() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-tensorswap-dep"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("store"));
        let tensorswap = TensorSwap::with_defaults(store).expect("tensorswap");

        let cid1 = test_cid();
        let cid2 = test_cid2();

        // Register tensor with dependency
        let metadata = TensorMetadata::new(cid1)
            .with_shape(vec![256, 256])
            .with_dtype("f32")
            .with_dependencies(vec![cid2]);

        tensorswap.register_tensor(metadata);
        tensorswap.want_tensor(cid1).expect("want_tensor");

        // Both tensor and dependency should be wanted
        assert!(tensorswap.bitswap().is_wanted(&cid1));
        assert!(tensorswap.bitswap().is_wanted(&cid2));
    }

    #[test]
    fn test_tensor_metadata_builder() {
        let cid = test_cid();
        let metadata = TensorMetadata::new(cid)
            .with_shape(vec![1024, 768])
            .with_dtype("f32")
            .with_size(1024 * 768 * 4)
            .with_layer_name("encoder.layer.0.attention.query")
            .with_priority_hint(100)
            .critical();

        assert!(metadata.is_critical);
        assert_eq!(metadata.priority_hint, Some(100));
        assert_eq!(
            metadata.layer_name.as_deref(),
            Some("encoder.layer.0.attention.query")
        );
        assert_eq!(metadata.estimated_size(), Some(1024 * 768 * 4));
    }

    #[test]
    fn test_tensor_size_estimation() {
        let cid = test_cid();

        // f32 tensor
        let f32_meta = TensorMetadata::new(cid)
            .with_shape(vec![100, 100])
            .with_dtype("f32");
        assert_eq!(f32_meta.estimated_size(), Some(100 * 100 * 4));

        // f16 tensor
        let f16_meta = TensorMetadata::new(cid)
            .with_shape(vec![100, 100])
            .with_dtype("f16");
        assert_eq!(f16_meta.estimated_size(), Some(100 * 100 * 2));

        // i8 tensor
        let i8_meta = TensorMetadata::new(cid)
            .with_shape(vec![100, 100])
            .with_dtype("i8");
        assert_eq!(i8_meta.estimated_size(), Some(100 * 100));
    }

    #[test]
    fn test_backpressure_controller() {
        let config = BackpressureConfig {
            max_pending: 10,
            high_watermark: 8,
            low_watermark: 2,
            max_buffer_bytes: 1024,
        };

        let mut bp = BackpressureController::new(config);
        assert!(bp.should_accept());
        assert!(!bp.is_paused());

        // Send until high watermark
        for _ in 0..8 {
            bp.on_send(100);
        }
        assert!(bp.is_paused());
        assert!(!bp.should_accept());

        // Ack until low watermark
        for _ in 0..6 {
            bp.on_ack(100);
        }
        assert!(!bp.is_paused());
        assert!(bp.should_accept());
    }

    #[test]
    fn test_stream_request_queue() {
        let mut queue = StreamRequestQueue::new(10);
        let cid1 = test_cid();
        let cid2 = test_cid2();

        // Add low priority
        queue.push(StreamRequest {
            cid: cid1,
            priority: 10,
            deadline: None,
            queued_at: Instant::now(),
        });

        // Add high priority
        queue.push(StreamRequest {
            cid: cid2,
            priority: 100,
            deadline: None,
            queued_at: Instant::now(),
        });

        // High priority should come first
        let first = queue.pop().expect("first");
        assert_eq!(first.cid, cid2);
        assert_eq!(first.priority, 100);

        let second = queue.pop().expect("second");
        assert_eq!(second.cid, cid1);
    }

    #[test]
    fn test_tensor_stream() {
        let cid = test_cid();
        let chunk_cids = vec![test_cid(), test_cid2()];

        let metadata = TensorMetadata::new(cid)
            .with_chunks(chunk_cids.clone())
            .with_size(2 * 1024 * 1024);

        let stream = TensorStream::new(metadata);

        assert!(!stream.is_complete());
        assert_eq!(stream.progress(), 0.0);
        assert_eq!(stream.missing_chunks().len(), 2);
    }

    #[test]
    fn test_safetensors_header_parse() {
        // Minimal safetensors header format
        let header_json =
            r#"{"weight":{"dtype":"F32","shape":[768,768],"data_offsets":[0,2359296]}}"#;
        let header_bytes = header_json.as_bytes();
        let header_size = header_bytes.len() as u64;

        let mut data = Vec::new();
        data.extend_from_slice(&header_size.to_le_bytes());
        data.extend_from_slice(header_bytes);

        let header = SafetensorsHeader::parse(&data).expect("parse header");
        assert_eq!(header.tensors.len(), 1);

        let weight = header.get_tensor("weight").expect("weight");
        assert_eq!(weight.dtype, "F32");
        assert_eq!(weight.shape, vec![768, 768]);
        assert_eq!(weight.data_length, 2359296);
    }

    #[tokio::test]
    async fn test_tensor_streaming() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-tensorswap-stream"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("store"));
        let tensorswap = TensorSwap::with_defaults(store).expect("tensorswap");

        let cid = test_cid();
        let metadata = TensorMetadata::new(cid)
            .with_shape(vec![1024, 1024])
            .with_dtype("f32")
            .with_size(4 * 1024 * 1024);

        tensorswap.register_tensor(metadata);

        // Start streaming
        tensorswap.start_stream(cid).expect("start_stream");

        let stats = tensorswap.stats();
        assert_eq!(stats.active_streams, 1);
        assert!(!stats.backpressure_paused);
    }

    #[test]
    fn test_einsum_expression_parse() {
        // Matrix multiplication
        let expr = EinsumExpression::parse("ij,jk->ik").expect("parse");
        assert_eq!(expr.num_inputs(), 2);
        assert_eq!(expr.inputs[0], "ij");
        assert_eq!(expr.inputs[1], "jk");
        assert_eq!(expr.output, "ik");
        assert!(!expr.is_transpose());
        assert!(!expr.is_reduction());

        let shared = expr.shared_indices();
        assert_eq!(shared.len(), 1);
        assert!(shared.contains(&'j'));

        // Reduction
        let expr2 = EinsumExpression::parse("ij->i").expect("parse2");
        assert!(expr2.is_reduction());
        assert_eq!(expr2.num_inputs(), 1);

        // Transpose
        let expr3 = EinsumExpression::parse("ij->ji").expect("parse3");
        assert!(expr3.is_transpose());
    }

    #[test]
    fn test_einsum_graph() {
        let mut graph = EinsumGraph::new();

        let cid_a = test_cid();
        let cid_b = test_cid2();
        let cid_c: ipfrs_core::Cid = "bafybeibxm2nsadl3fnxv2sxcxmxaco2jl53wpeorjdziber7rnz5gvv5h4"
            .parse()
            .expect("cid_c");

        // Register tensors
        graph.register_tensor("A", cid_a);
        graph.register_tensor("B", cid_b);
        graph.register_tensor("C", cid_c);

        // Add expressions: A and B are inputs, C = A @ B
        let expr = EinsumExpression::parse("ij,jk->ik").expect("parse");
        let mut expr_with_names = expr.clone();
        expr_with_names.inputs = vec!["A".to_string(), "B".to_string()];
        expr_with_names.output = "C".to_string();

        graph.add_expression(expr_with_names);

        // Check dependencies
        let deps = graph.get_dependencies("C").expect("deps");
        assert_eq!(deps.len(), 2);
        assert!(deps.contains(&cid_a));
        assert!(deps.contains(&cid_b));

        // Check priorities (A and B are leaves, should have higher priority than C)
        let priority_a = graph.compute_priority("A");
        let priority_c = graph.compute_priority("C");
        assert!(priority_a > priority_c);

        // Check topological order
        let order = graph.topological_order().expect("order");
        assert_eq!(order.len(), 3);

        // A and B should come before C
        let pos_c = order
            .iter()
            .position(|(name, _)| name == "C")
            .expect("pos_c");
        let pos_a = order
            .iter()
            .position(|(name, _)| name == "A")
            .expect("pos_a");
        let pos_b = order
            .iter()
            .position(|(name, _)| name == "B")
            .expect("pos_b");
        assert!(pos_a < pos_c);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn test_einsum_metadata_generation() {
        let mut graph = EinsumGraph::new();

        let cid_a = test_cid();
        let cid_b = test_cid2();
        let cid_c: ipfrs_core::Cid = "bafybeibxm2nsadl3fnxv2sxcxmxaco2jl53wpeorjdziber7rnz5gvv5h4"
            .parse()
            .expect("cid_c");

        graph.register_tensor("A", cid_a);
        graph.register_tensor("B", cid_b);
        graph.register_tensor("C", cid_c);

        let expr = EinsumExpression {
            expression: "ij,jk->ik".to_string(),
            inputs: vec!["A".to_string(), "B".to_string()],
            output: "C".to_string(),
        };
        graph.add_expression(expr);

        // Generate metadata for C
        let metadata = graph.generate_metadata("C").expect("metadata");
        assert_eq!(metadata.cid, cid_c);
        assert_eq!(metadata.dependencies.len(), 2);
        assert!(metadata.priority_hint.is_some());
    }

    #[tokio::test]
    async fn test_backpressure_integration() {
        let ts_config = TensorSwapConfig {
            max_concurrent_streams: 2,
            ..Default::default()
        };

        let store_config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-tensorswap-bp"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&store_config.path);
        let store = Arc::new(SledBlockStore::new(store_config).expect("store"));
        let tensorswap = TensorSwap::new(store, ts_config).expect("tensorswap");

        let cid1 = test_cid();
        let cid2 = test_cid2();

        // Start two streams (at limit)
        tensorswap.start_stream(cid1).expect("stream1");
        tensorswap.start_stream(cid2).expect("stream2");

        // Third stream should fail (at limit)
        let cid3: ipfrs_core::Cid = "bafybeibxm2nsadl3fnxv2sxcxmxaco2jl53wpeorjdziber7rnz5gvv5h4"
            .parse()
            .expect("cid3");
        let result = tensorswap.start_stream(cid3);
        assert!(result.is_err());

        let stats = tensorswap.stats();
        assert_eq!(stats.active_streams, 2);
    }

    // ── GradientStreamSession tests ───────────────────────────────────────

    /// Encode 1 M f32 values and verify chunk count = ceil(1_000_000 / 65_536),
    /// then decode and verify element-wise equality.
    #[test]
    fn test_gradient_chunk_encode_decode() {
        let n = 1_000_000usize;
        let gradient: Vec<f32> = (0u32..n as u32).map(|i| i as f32 * 1e-6).collect();

        let session = GradientStreamSession::with_defaults("sess-encode-decode");
        let chunks = session.encode_gradient(&gradient).expect("encode_gradient");

        let expected_chunks = n.div_ceil(65_536);
        assert_eq!(
            chunks.len(),
            expected_chunks,
            "chunk count should be ceil(1_000_000 / 65_536)"
        );

        // Verify all chunks carry consistent metadata.
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.chunk_index, i as u32);
            assert_eq!(chunk.total_chunks, expected_chunks as u32);
            assert_eq!(chunk.session_id, "sess-encode-decode");
        }

        let decoded = session.decode_chunks(chunks).expect("decode_chunks");
        assert_eq!(decoded.len(), n, "decoded length must equal original");
        for (i, (&orig, &dec)) in gradient.iter().zip(decoded.iter()).enumerate() {
            assert!(
                (orig - dec).abs() < 1e-7,
                "value mismatch at index {i}: orig={orig}, decoded={dec}"
            );
        }
    }

    /// Corrupt one byte in a chunk and verify checksum mismatch is detected.
    #[test]
    fn test_gradient_chunk_checksum() {
        let gradient: Vec<f32> = (0u32..100).map(|i| i as f32).collect();
        let session = GradientStreamSession::with_defaults("sess-checksum");
        let mut chunks = session.encode_gradient(&gradient).expect("encode");

        // Flip a byte in the middle of the Arrow IPC payload.
        let mid = chunks[0].arrow_ipc_bytes.len() / 2;
        chunks[0].arrow_ipc_bytes[mid] ^= 0xFF;

        let result = session.decode_chunks(chunks);
        match result {
            Err(GradientStreamError::ChecksumMismatch { chunk_index, .. }) => {
                assert_eq!(chunk_index, 0);
            }
            other => panic!("expected ChecksumMismatch, got: {other:?}"),
        }
    }

    /// A gradient smaller than one chunk should produce exactly one chunk.
    #[test]
    fn test_gradient_session_small() {
        let gradient: Vec<f32> = vec![1.0, 2.0, 3.0];
        let session = GradientStreamSession::with_defaults("sess-small");
        let chunks = session.encode_gradient(&gradient).expect("encode");

        assert_eq!(chunks.len(), 1, "small gradient must fit in a single chunk");
        assert_eq!(chunks[0].total_chunks, 1);
        assert_eq!(chunks[0].chunk_index, 0);

        let decoded = session.decode_chunks(chunks).expect("decode");
        assert_eq!(decoded, gradient);
    }

    /// Single chunk Arrow IPC round-trip at the `GradientChunk` level.
    #[test]
    fn test_gradient_chunk_roundtrip_arrow() {
        use ipfrs_tensorlogic::gradient::arrow_ipc::{
            load_gradient_from_arrow, store_gradient_as_arrow,
        };

        let original: Vec<f32> = (0u32..256).map(|i| i as f32 * 0.5).collect();
        let ipc_bytes = store_gradient_as_arrow(&original).expect("store");
        let checksum = GradientChunk::compute_checksum(&ipc_bytes);

        let chunk = GradientChunk {
            session_id: "sess-roundtrip".to_string(),
            chunk_index: 0,
            total_chunks: 1,
            arrow_ipc_bytes: ipc_bytes.clone(),
            checksum,
        };

        assert!(
            chunk.verify_checksum(),
            "checksum must verify on fresh chunk"
        );

        let decoded = load_gradient_from_arrow(&ipc_bytes).expect("load");
        assert_eq!(decoded.len(), original.len());
        for (i, (&o, &d)) in original.iter().zip(decoded.iter()).enumerate() {
            assert!(
                (o - d).abs() < 1e-6,
                "value mismatch at index {i}: o={o}, d={d}"
            );
        }
    }

    /// Stream gradient through an in-memory duplex and verify round-trip.
    #[tokio::test]
    async fn test_gradient_stream_to_receive() {
        let gradient: Vec<f32> = (0u32..512).map(|i| i as f32 * 0.01).collect();

        let session = GradientStreamSession::new("sess-stream", 128);

        // Use a tokio duplex stream as a loopback transport.
        let (mut server_half, mut client_half) = tokio::io::duplex(1024 * 1024);

        let gradient_clone = gradient.clone();
        let sender = tokio::spawn(async move {
            let s = GradientStreamSession::new("sess-stream", 128);
            s.stream_to(&gradient_clone, &mut server_half)
                .await
                .expect("stream_to")
        });

        let n_chunks = sender.await.expect("sender task");
        assert_eq!(n_chunks, gradient.len().div_ceil(128));

        let decoded = session
            .receive_from(&mut client_half)
            .await
            .expect("receive_from");

        assert_eq!(decoded.len(), gradient.len());
        for (i, (&o, &d)) in gradient.iter().zip(decoded.iter()).enumerate() {
            assert!(
                (o - d).abs() < 1e-6,
                "value mismatch at index {i}: o={o}, d={d}"
            );
        }
    }
}
